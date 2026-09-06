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
import { BasicTracerProvider, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  createMetricExporter,
  createTraceExporter,
  DORMANT_INTERVAL_MS,
  DURATION_BUCKETS,
  otlpHeaders,
  sessionSampler,
  TOKEN_BUCKETS,
} from "@telemetry-dev/otel";

import type { ResolvedConfig, TelemetryError } from "./config.ts";

const SCOPE_NAME = "@telemetry-dev/ai-sdk";
const SCOPE_VERSION = "0.0.0";

// The sampler can return non-recording API spans.
export interface Emitter {
  tracer: Tracer;
  recordDuration(seconds: number, attributes: Attributes): void;
  recordTokens(tokenType: "input" | "output", count: number, attributes: Attributes): void;
  flush(spans: Span[]): Promise<void>;
}

export interface EmitterOverrides {
  sendSpans?: (spans: ReadableSpan[]) => Promise<void>;
  recordDuration?: (seconds: number, attributes: Attributes) => void;
  recordTokens?: (tokenType: "input" | "output", count: number, attributes: Attributes) => void;
}

type Transport = { fetchImpl: typeof fetch; onError?: (error: TelemetryError) => void };

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

const cache = new Map<string, EmitterCore>();

const buildCore = (config: ResolvedConfig): EmitterCore => {
  const { apiKey, baseUrl, environment, serviceName } = config;
  const resource = resourceFromAttributes({
    "service.name": serviceName,
    "deployment.environment.name": environment,
  });
  const headers = otlpHeaders(apiKey ?? "", "@telemetry-dev/ai-sdk");

  const provider = new BasicTracerProvider({ resource, sampler: sessionSampler(config.sampler) });
  const tracer = provider.getTracer(SCOPE_NAME, SCOPE_VERSION);

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

  // Each emitter builds its OWN metrics pipeline bound to its OWN transport: the exporter closes
  // over this call's fetch/onError, so concurrent same-identity emitters never flush metrics
  // through one another's context. The pipeline is torn down on flush (which clears the reader's
  // interval timer) and lazily rebuilt, so a reused globally-registered emitter gets a fresh meter
  // per generation without leaking timers.
  const buildMetrics = (transport: Transport): MetricsPipeline => {
    // A long interval keeps the periodic timer dormant; the single flush is driven by shutdown().
    const reader = new PeriodicExportingMetricReader({
      exporter: createMetricExporter({ url: `${baseUrl}/v1/metrics`, headers }, transport),
      exportIntervalMillis: DORMANT_INTERVAL_MS,
    });
    const meterProvider = new MeterProvider({ resource, readers: [reader] });
    const meter = meterProvider.getMeter(SCOPE_NAME, SCOPE_VERSION);

    const durationHistogram = meter.createHistogram("gen_ai.client.operation.duration", {
      unit: "s",
      advice: { explicitBucketBoundaries: DURATION_BUCKETS },
    });
    const tokenHistogram = meter.createHistogram("gen_ai.client.token.usage", {
      unit: "{token}",
      advice: { explicitBucketBoundaries: TOKEN_BUCKETS },
    });

    // shutdown() flushes the collected deltas through the exporter AND clears the interval timer.
    return { durationHistogram, tokenHistogram, shutdown: () => meterProvider.shutdown() };
  };

  return { tracer, sendSpans, buildMetrics };
};

const coreFor = (config: ResolvedConfig): EmitterCore => {
  if (config.sampler) return buildCore(config);
  const key = `${config.apiKey}|${config.baseUrl}|${config.environment}|${config.serviceName}`;
  const cached = cache.get(key);
  if (cached) return cached;
  const core = buildCore(config);
  cache.set(key, core);
  return core;
};

export const createEmitter = (config: ResolvedConfig, overrides?: EmitterOverrides): Emitter => {
  const core = coreFor(config);
  const transport: Transport = { fetchImpl: config.fetchImpl, onError: config.onError };
  const onError = config.onError;

  // Spans are request-scoped: each emitter sends through its OWN transport, so concurrent
  // same-identity emitters never cross-talk.
  const sendSpans =
    overrides?.sendSpans ?? ((spans: ReadableSpan[]) => core.sendSpans(spans, transport));

  // Metrics are also per-emitter and bound to this call's transport. The pipeline is built lazily on
  // first record and detached on flush, so a reused emitter rebuilds a fresh one each generation.
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

  // Default: await the flush so serverless runtimes don't tear down before spans/metrics leave.
  // When a waitUntil extender is supplied, hand off the combined promise and return immediately.
  const flush = (spans: Span[]): Promise<void> => {
    // Detach this generation's metrics pipeline synchronously so the next generation builds its own.
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
    if (config.waitUntil) {
      config.waitUntil(p);
      return Promise.resolve();
    }
    return p;
  };

  return { tracer: core.tracer, recordDuration, recordTokens, flush };
};
