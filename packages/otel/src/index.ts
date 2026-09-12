export { jsonAttr, omitUndefined, SCOPE_NAME, SCOPE_VERSION } from "./attrs.ts";

export {
  type BatchOptions,
  DEFAULT_BASE_URL,
  DEFAULT_BATCH,
  type ExportMode,
  type LogLevel,
  resolveEnv,
  type SdkLogLevel,
} from "./config.ts";

export {
  activeContext,
  als,
  AlsContextManager,
  buildPropagatedAttributes,
  propagateAttributes,
  type PropagatedAttributes,
  PROPAGATED_KEY,
  propagatedFromContext,
  withContext,
} from "./context.ts";

export { diag, reportError, setLogLevel } from "./debug.ts";

export {
  createGenerationEmitter,
  type GenerationEmitter,
  type GenerationEmitterOverrides,
} from "./generation.ts";

export {
  BATCHED_METRIC_INTERVAL_MS,
  createMetricsPipeline,
  DORMANT_INTERVAL_MS,
  DURATION_BUCKETS,
  OUTPUT_CHUNK_HISTOGRAM,
  type OutputChunkHistogram,
  type MetricsPipeline,
  TOKEN_BUCKETS,
} from "./metrics.ts";

export {
  createTelemetrySpanExporter,
  TelemetrySpanProcessor,
  type TelemetrySpanProcessorOptions,
} from "./otel.ts";

export { StampingSpanProcessor, type StampingProcessorOptions } from "./processor.ts";

export {
  sessionIdOf,
  type SessionRootOf,
  sessionRootTracerProvider,
  sessionSampler,
  sessionSpanContext,
  sha256,
  withSessionParent,
} from "./session.ts";

export {
  createLogExporter,
  createMetricExporter,
  createTraceExporter,
  maybeGzip,
  otlpHeaders,
  type OtlpTarget,
  postOtlp,
  type Transport,
} from "./transport.ts";
