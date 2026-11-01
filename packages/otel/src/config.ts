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
  /**
   * Maximum bytes, across the process, that provider integrations keep for the output of
   * streams that have not ended, counted as UTF-8 bytes of JSON. Output past it is not kept,
   * and its span carries the truncation flag. While no other stream holds bytes in it, a
   * stream may grow to 16 MiB even past this bound. The bound is process-wide, so the span
   * processor started last sets it. Default 67108864 (64 MiB).
   */
  maxStreamBytes?: number;
  /** Timeout of each export request in milliseconds. Default 30000. */
  exportTimeoutMillis?: number;
}

export const DEFAULT_BASE_URL = "https://api.telemetry.dev";

export const DEFAULT_BATCH: Required<BatchOptions> = {
  maxExportBatchSize: 64,
  scheduledDelayMillis: 1000,
  maxQueueSize: 2048,
  maxQueueBytes: 64 * 1024 * 1024,
  maxStreamBytes: 64 * 1024 * 1024,
  exportTimeoutMillis: 30000,
};

/**
 * The batch options with DEFAULT_BATCH for each option that is absent or undefined. Throws a
 * RangeError when a byte bound is not a non-negative safe integer or Infinity.
 */
export function resolveBatch(batch: BatchOptions = {}): Required<BatchOptions> {
  const resolved = { ...DEFAULT_BATCH };

  for (const key of Object.keys(resolved) as (keyof BatchOptions)[]) {
    resolved[key] = batch[key] ?? resolved[key];
  }

  for (const key of ["maxQueueBytes", "maxStreamBytes"] as const) {
    const bound = resolved[key];

    if (bound !== Infinity && (!Number.isSafeInteger(bound) || bound < 0)) {
      throw new RangeError(`batch.${key} must be a non-negative safe integer or Infinity`);
    }
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
