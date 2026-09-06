export type {
  AgentFields,
  GenerationFields,
  SpanFields,
  SpanType,
  TokenUsage,
  ToolFields,
} from "./attrs.ts";
export { type ClientOverrides, flush, init, shutdown, type TelemetryClient } from "./client.ts";
export type {
  BatchOptions,
  ExportMode,
  LogLevel,
  MaskFn,
  SdkLogLevel,
  TelemetryOptions,
} from "./config.ts";
export {
  propagateAttributes,
  type PropagatedAttributes,
  SCOPE_NAME,
  TelemetrySpanProcessor,
  type TelemetrySpanProcessorOptions,
} from "@telemetry-dev/otel";
export { log, type LogOptions } from "./logs.ts";
export { observe, type ObserveOptions } from "./observe.ts";
export {
  activeContext,
  extractW3cContext,
  getTraceparent,
  injectW3cContext,
  type ParentRef,
  type SpanHandle,
  startActiveSpan,
  startSpan,
  type StartSpanOptions,
  updateActiveSpan,
  withContext,
} from "./span.ts";
