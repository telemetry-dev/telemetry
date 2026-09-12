export type LogLevel = "debug" | "info" | "warn" | "error";

export type SdkLogLevel = LogLevel | "silent";

export type ExportMode = "batched" | "immediate";

export interface BatchOptions {
  /** Spans per export batch. LLM spans are large; keep batches small. Default 64. */
  maxExportBatchSize?: number;
  /** Delay between batch exports in milliseconds. Default 1000. */
  scheduledDelayMillis?: number;
  /** Maximum spans buffered before drops. Default 2048. */
  maxQueueSize?: number;
  /** Per-export timeout in milliseconds. Default 30000. */
  exportTimeoutMillis?: number;
}

export const DEFAULT_BASE_URL = "https://ingest.telemetry.dev";

export const DEFAULT_BATCH: Required<BatchOptions> = {
  maxExportBatchSize: 64,
  scheduledDelayMillis: 1000,
  maxQueueSize: 2048,
  exportTimeoutMillis: 30000,
};

export function resolveEnv(): Record<string, string | undefined> {
  if (globalThis.process !== undefined && process.env) return process.env;

  return {};
}
