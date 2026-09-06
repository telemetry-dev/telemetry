import {
  type Attributes,
  type Histogram,
  type Span,
  TraceFlags,
  type Tracer,
} from "@opentelemetry/api";
import { ExportResultCode } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { MeterProvider, PeriodicExportingMetricReader } from "@opentelemetry/sdk-metrics";
import {
  BasicTracerProvider,
  type ReadableSpan,
  type Sampler,
} from "@opentelemetry/sdk-trace-base";

import { DORMANT_INTERVAL_MS, DURATION_BUCKETS, TOKEN_BUCKETS } from "./metrics.ts";
import { sessionSampler } from "./session.ts";
import {
  createMetricExporter,
  createTraceExporter,
  otlpHeaders,
  type Transport,
} from "./transport.ts";

/** Each flush sends sampled spans and releases that generation's metrics resources. */
export interface GenerationEmitter {
  tracer: Tracer;
  recordDuration(seconds: number, attributes: Attributes): void;
  recordTokens(tokenType: "input" | "output", count: number, attributes: Attributes): void;
  flush(spans: Span[]): Promise<void>;
}

/** Replaces span export or metric collection without changes to sampling and flush behavior. */
export interface GenerationEmitterOverrides {
  sendSpans?: (spans: ReadableSpan[]) => Promise<void>;
  recordDuration?: (seconds: number, attributes: Attributes) => void;
  recordTokens?: (tokenType: "input" | "output", count: number, attributes: Attributes) => void;
}

interface GenerationEmitterConfig {
  readonly sdkName: string;
  readonly apiKey: string | undefined;
  readonly baseUrl: string;
  readonly environment: string;
  readonly serviceName: string;
  readonly sampler?: Sampler;
  readonly fetchImpl: typeof fetch;
  readonly waitUntil?: (p: Promise<unknown>) => void;
  readonly onError?: (error: Error | string) => void;
}

const SCOPE_VERSION = "0.0.0";
const cache = new Map<string, EmitterCore>();

/** Shares tracers by SDK identity but keeps each emitter's transport and metrics separate. */
export function createGenerationEmitter(
  config: GenerationEmitterConfig,
  overrides?: GenerationEmitterOverrides,
): GenerationEmitter {
  const core = coreFor(config);
  const transport: Transport = { fetchImpl: config.fetchImpl, onError: config.onError };
  const onError = config.onError;
  const sendSpans =
    overrides?.sendSpans ?? ((spans: ReadableSpan[]) => core.sendSpans(spans, transport));

  // Each generation gets a new metrics pipeline bound to this emitter's transport.
  let metrics: MetricsPipeline | undefined;
  const ensureMetrics = (): MetricsPipeline => (metrics ??= core.buildMetrics(transport));

  const recordDuration =
    overrides?.recordDuration ??
    ((seconds: number, attributes: Attributes) =>
      ensureMetrics().durationHistogram.record(seconds, attributes));
  const recordTokens =
    overrides?.recordTokens ??
    ((tokenType: "input" | "output", count: number, attributes: Attributes) =>
      ensureMetrics().tokenHistogram.record(count, {
        ...attributes,
        "gen_ai.token.type": tokenType,
      }));

  const flush = (spans: Span[]): Promise<void> => {
    // Detach before the first await so overlapping flushes cannot share metrics.
    const pipeline = metrics;
    metrics = undefined;
    const p = (async () => {
      try {
        const sampled = spans.filter(isReadableSpan);
        if (sampled.length) await sendSpans(sampled);
      } catch (e) {
        onError?.(e instanceof Error ? e : String(e));
      }
      try {
        if (pipeline) await pipeline.shutdown();
      } catch (e) {
        onError?.(e instanceof Error ? e : String(e));
      }
    })();
    // Await export unless the serverless runtime receives the promise through waitUntil.
    if (config.waitUntil) {
      config.waitUntil(p);
      return Promise.resolve();
    }
    return p;
  };

  return { tracer: core.tracer, recordDuration, recordTokens, flush };
}

interface MetricsPipeline {
  durationHistogram: Histogram;
  tokenHistogram: Histogram;
  shutdown: () => Promise<void>;
}

interface EmitterCore {
  tracer: Tracer;
  sendSpans: (spans: ReadableSpan[], transport: Transport) => Promise<void>;
  buildMetrics: (transport: Transport) => MetricsPipeline;
}

const isReadableSpan = (span: Span): span is Span & ReadableSpan =>
  "resource" in span &&
  "instrumentationScope" in span &&
  "events" in span &&
  (span.spanContext().traceFlags & TraceFlags.SAMPLED) !== 0;

function coreFor(config: GenerationEmitterConfig): EmitterCore {
  if (config.sampler) return buildCore(config);
  const key = `${config.sdkName}|${config.apiKey}|${config.baseUrl}|${config.environment}|${config.serviceName}`;
  const cached = cache.get(key);
  if (cached) return cached;
  const core = buildCore(config);
  cache.set(key, core);
  return core;
}

function buildCore(config: GenerationEmitterConfig): EmitterCore {
  const { apiKey, baseUrl, environment, serviceName, sdkName } = config;
  const resource = resourceFromAttributes({
    "service.name": serviceName,
    "deployment.environment.name": environment,
  });
  const headers = otlpHeaders(apiKey ?? "", sdkName);

  const provider = new BasicTracerProvider({ resource, sampler: sessionSampler(config.sampler) });
  const tracer = provider.getTracer(sdkName, SCOPE_VERSION);

  const sendSpans = (spans: ReadableSpan[], transport: Transport): Promise<void> =>
    new Promise((resolve, reject) => {
      createTraceExporter(
        { url: `${baseUrl}/v1/traces`, headers },
        { fetchImpl: transport.fetchImpl },
      ).export(spans, (result) => {
        if (result.code === ExportResultCode.SUCCESS) resolve();
        else reject(result.error ?? new Error("telemetry.dev trace export failed"));
      });
    });

  const buildMetrics = (transport: Transport): MetricsPipeline => {
    // The timer stays dormant. shutdown() exports the metrics and clears the timer.
    const reader = new PeriodicExportingMetricReader({
      exporter: createMetricExporter({ url: `${baseUrl}/v1/metrics`, headers }, transport),
      exportIntervalMillis: DORMANT_INTERVAL_MS,
    });
    const meterProvider = new MeterProvider({ resource, readers: [reader] });
    const meter = meterProvider.getMeter(sdkName, SCOPE_VERSION);

    const durationHistogram = meter.createHistogram("gen_ai.client.operation.duration", {
      unit: "s",
      advice: { explicitBucketBoundaries: DURATION_BUCKETS },
    });
    const tokenHistogram = meter.createHistogram("gen_ai.client.token.usage", {
      unit: "{token}",
      advice: { explicitBucketBoundaries: TOKEN_BUCKETS },
    });

    return { durationHistogram, tokenHistogram, shutdown: () => meterProvider.shutdown() };
  };

  return { tracer, sendSpans, buildMetrics };
}
