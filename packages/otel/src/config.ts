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
  /**
   * Maximum size of the buffered spans before drops, in bytes of their string attribute values,
   * each UTF-16 unit counted as one byte. The core SDK bounds its buffered log records the same
   * way. A record is still buffered when its queue is empty. Drops are reported to onError.
   * Default 67108864 (64 MiB).
   */
  maxQueueBytes?: number;
  /** Timeout of each export request in milliseconds. Default 30000. */
  exportTimeoutMillis?: number;
}

export const DEFAULT_BASE_URL = "https://api.telemetry.dev";

export const DEFAULT_BATCH: Required<BatchOptions> = {
  maxExportBatchSize: 64,
  scheduledDelayMillis: 1000,
  maxQueueSize: 2048,
  maxQueueBytes: 64 * 1024 * 1024,
  exportTimeoutMillis: 30000,
};

/** The batch options with DEFAULT_BATCH for each option that is absent or undefined. */
export function resolveBatch(batch: BatchOptions = {}): Required<BatchOptions> {
  const resolved = { ...DEFAULT_BATCH };

  for (const key of Object.keys(resolved) as (keyof BatchOptions)[]) {
    resolved[key] = batch[key] ?? resolved[key];
  }

  return resolved;
}

export function resolveEnv(): Record<string, string | undefined> {
  if (globalThis.process !== undefined && process.env) return process.env;

  return {};
}

const MAX_TIMER_DELAY_MS = 2_147_483_647;

/**
 * The timeout a batch processor gives one export call when the exporter applies
 * exportTimeoutMillis to each request: a batch can split into one request per record.
 */
export function batchExportTimeoutMillis(exportTimeoutMillis: number, maxExportBatchSize: number) {
  return Math.min(exportTimeoutMillis * (maxExportBatchSize + 1), MAX_TIMER_DELAY_MS);
}
