import type { Context } from "@opentelemetry/api";
import { ExportResultCode } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import type {
  ReadableSpan,
  Span,
  SpanExporter,
  SpanProcessor,
} from "@opentelemetry/sdk-trace-base";

import {
  type BatchOptions,
  DEFAULT_BASE_URL,
  DEFAULT_BATCH,
  type ExportMode,
  resolveEnv,
} from "./config.ts";
import { diag } from "./debug.ts";
import {
  BATCHED_METRIC_INTERVAL_MS,
  createMetricsPipeline,
  DORMANT_INTERVAL_MS,
  type MetricsPipeline,
} from "./metrics.ts";
import { StampingSpanProcessor } from "./processor.ts";
import {
  createMetricExporter,
  createTraceExporter,
  otlpHeaders,
  type Transport,
} from "./transport.ts";

export interface TelemetrySpanProcessorOptions {
  /** telemetry.dev ingest key. Falls back to `TELEMETRY_DEV_API_KEY`; absent ⇒ no-op processor. */
  apiKey?: string;
  /** Falls back to `TELEMETRY_DEV_BASE_URL`, then `https://ingest.telemetry.dev`. */
  baseUrl?: string;
  /** "batched" (default) or "immediate". */
  exportMode?: ExportMode;
  batch?: BatchOptions;
  /** Default for BYO providers: export EVERY span (the user deliberately attached the processor). */
  spanFilter?: (span: ReadableSpan) => boolean;
  /** Record GenAI duration/token histograms for exported chat/agent/embedding/tool spans. Default true. */
  metrics?: boolean;
  /** service.name on the metrics resource. Falls back to `OTEL_SERVICE_NAME`. */
  serviceName?: string;
  /** deployment.environment.name on the metrics resource. Falls back to `TELEMETRY_DEV_ENVIRONMENT`. */
  environment?: string;
  fetch?: typeof fetch;
  onError?: (error: Error) => void;
  /** Advanced/test seam: replaces the OTLP fetch exporter. */
  spanExporter?: SpanExporter;
}

/**
 * BYO-OpenTelemetry surface: add this processor to your own TracerProvider (NodeSDK,
 * registerOTel, …) to ship its spans to telemetry.dev. Also stamps propagateAttributes
 * correlation attributes onto every span it sees.
 */
export class TelemetrySpanProcessor implements SpanProcessor {
  private readonly inner?: StampingSpanProcessor;
  private readonly metrics?: MetricsPipeline;

  constructor(options: TelemetrySpanProcessorOptions = {}) {
    const env = resolveEnv();
    const apiKey = options.apiKey ?? env.TELEMETRY_DEV_API_KEY;

    const baseUrl = (options.baseUrl ?? env.TELEMETRY_DEV_BASE_URL ?? DEFAULT_BASE_URL).replace(
      /\/+$/,
      "",
    );

    const transport: Transport = {
      fetchImpl: options.fetch ?? globalThis.fetch,
      onError: options.onError,
      exportTimeoutMillis: options.batch?.exportTimeoutMillis,
    };

    const exporter =
      options.spanExporter ??
      (apiKey
        ? createTraceExporter(
            { url: `${baseUrl}/v1/traces`, headers: otlpHeaders(apiKey) },
            transport,
          )
        : undefined);

    if (!exporter) {
      diag.debug(
        "no api key (apiKey option or TELEMETRY_DEV_API_KEY); TelemetrySpanProcessor is a no-op",
      );

      return;
    }

    const exportMode = options.exportMode ?? "batched";

    if (options.metrics !== false && apiKey) {
      const resource = resourceFromAttributes({
        "service.name": options.serviceName ?? env.OTEL_SERVICE_NAME ?? "unknown_service",
        "deployment.environment.name":
          options.environment ?? env.TELEMETRY_DEV_ENVIRONMENT ?? "production",
      });

      this.metrics = createMetricsPipeline({
        resource,
        exporter: createMetricExporter(
          { url: `${baseUrl}/v1/metrics`, headers: otlpHeaders(apiKey) },
          transport,
        ),
        exportIntervalMillis:
          exportMode === "batched" ? BATCHED_METRIC_INTERVAL_MS : DORMANT_INTERVAL_MS,
      });
    }

    const metrics = this.metrics;
    this.inner = new StampingSpanProcessor({
      exporter,
      exportMode,
      batch: { ...DEFAULT_BATCH, ...options.batch },
      spanFilter: options.spanFilter,
      recordMetrics: metrics ? (span) => metrics.record(span) : undefined,
      onError: options.onError,
    });
  }

  onStart(span: Span, parentContext: Context): void {
    this.inner?.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.inner?.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return Promise.all([this.inner?.forceFlush(), this.metrics?.forceFlush()]).then(
      () => undefined,
    );
  }

  shutdown(): Promise<void> {
    return Promise.all([this.inner?.shutdown(), this.metrics?.shutdown()]).then(() => undefined);
  }
}

/** Raw OTLP/protobuf fetch exporter for users wiring their own BatchSpanProcessor. */
export function createTelemetrySpanExporter(
  options: {
    apiKey?: string;
    baseUrl?: string;
    fetch?: typeof fetch;
    onError?: (error: Error) => void;
    exportTimeoutMillis?: number;
  } = {},
): SpanExporter {
  const env = resolveEnv();
  const apiKey = options.apiKey ?? env.TELEMETRY_DEV_API_KEY;

  const baseUrl = (options.baseUrl ?? env.TELEMETRY_DEV_BASE_URL ?? DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );

  if (!apiKey) {
    diag.debug("no api key (apiKey option or TELEMETRY_DEV_API_KEY); exporter is a no-op");

    return {
      export: (_spans, resultCallback) => resultCallback({ code: ExportResultCode.SUCCESS }),
      forceFlush: () => Promise.resolve(),
      shutdown: () => Promise.resolve(),
    };
  }

  return createTraceExporter(
    { url: `${baseUrl}/v1/traces`, headers: otlpHeaders(apiKey) },
    {
      fetchImpl: options.fetch ?? globalThis.fetch,
      onError: options.onError,
      exportTimeoutMillis: options.exportTimeoutMillis,
    },
  );
}
