import type { ExportResult } from "@opentelemetry/core";

import { DEFAULT_BATCH } from "./config.ts";
import { reportError } from "./debug.ts";

/**
 * The largest OTLP request the exporters send, in bytes. The ingest rejects a body over 16 MiB
 * after gzip decoding; keep this equal to DECOMPRESSED_BODY_LIMIT in the ingest. A record that
 * alone is larger has its longest values cut at export, so provider integrations keep at most
 * this much of one stream.
 */
export const MAX_RECORD_BYTES = 16 * 1024 * 1024;

/**
 * Bounds the bytes that provider integrations keep, across the process, for streams that have
 * not ended, since each stream may keep up to MAX_RECORD_BYTES. A stream holds its bytes through
 * a StreamCapture. While no other stream holds bytes, a stream may grow to MAX_RECORD_BYTES even
 * past maxBytes, so that one large stream is not cut by the shared bound.
 */
export class StreamCaptureBudget {
  private held = 0;

  constructor(public maxBytes: number) {}

  /** The total a stream that holds `own` bytes may hold now. */
  limit(own: number): number {
    const others = this.held - own;
    const room = others > 0 ? this.maxBytes - others : MAX_RECORD_BYTES;

    return Math.max(own, Math.min(MAX_RECORD_BYTES, room));
  }

  move(from: number, to: number): void {
    this.held = Math.max(this.held + to - from, 0);
  }
}

/** The budget of this process, sized by `batch.maxStreamBytes` when a span processor starts. */
export const streamCaptureBudget = new StreamCaptureBudget(DEFAULT_BATCH.maxStreamBytes);

interface HeldBytes {
  bytes: number;
}

const releaseDropped =
  typeof FinalizationRegistry === "function"
    ? new FinalizationRegistry<HeldBytes>((held) => streamCaptureBudget.move(held.bytes, 0))
    : undefined;

/**
 * The output bytes one stream keeps, held in streamCaptureBudget. A provider integration compares
 * a new total against `limit` before it keeps more, sets `bytes` to what it keeps, and calls
 * release() when the stream ends. A stream that is dropped without ending releases its bytes
 * when it is garbage collected.
 */
export class StreamCapture {
  private readonly held: HeldBytes = { bytes: 0 };
  private releasedLimit: number | undefined;

  constructor(initialBytes = 0) {
    releaseDropped?.register(this, this.held, this);
    this.bytes = initialBytes;
  }

  get limit(): number {
    return this.releasedLimit ?? streamCaptureBudget.limit(this.held.bytes);
  }

  get bytes(): number {
    return this.held.bytes;
  }

  set bytes(bytes: number) {
    if (this.releasedLimit === undefined) streamCaptureBudget.move(this.held.bytes, bytes);
    this.held.bytes = bytes;
  }

  /**
   * Returns the bytes to the budget. After it, `limit` stays at the bytes held at release, so a
   * stream that is still read after it ended keeps no more outside the bound.
   */
  release(): void {
    if (this.releasedLimit !== undefined) return;
    this.releasedLimit = Math.min(this.limit, this.held.bytes);
    streamCaptureBudget.move(this.held.bytes, 0);
    releaseDropped?.unregister(this);
  }
}

interface QueuedRecord {
  readonly attributes: Readonly<Record<string, unknown>>;
  readonly events?: readonly { readonly attributes?: Readonly<Record<string, unknown>> }[];
  readonly body?: unknown;
}

const valueSize = (value: unknown): number => {
  if (typeof value === "string") return value.length;

  if (Array.isArray(value)) {
    let size = 0;

    for (const item of value) size += valueSize(item);

    return size;
  }

  return 0;
};

const attributesSize = (attributes: Readonly<Record<string, unknown>> | undefined): number => {
  let size = 0;

  for (const key in attributes) size += valueSize(attributes[key]);

  return size;
};

/** The UTF-16 length of the record's string attribute values, event attributes, and body. */
const recordSize = (record: QueuedRecord): number => {
  let size = attributesSize(record.attributes) + valueSize(record.body);

  for (const event of record.events ?? []) size += attributesSize(event.attributes);

  return size;
};

/**
 * Bounds the records that wait in a batch processor's queue by count and by the size of their
 * text, since a count alone does not bound memory once values are sent whole. The processor
 * calls admit() before it queues a record and release() with each batch it hands to the exporter.
 * The count limit equals the processor's own, so the processor never drops a record the budget
 * counted. A record over the size limit is still admitted into an empty queue, so that one large
 * record is not dropped forever. The first drop of a series is reported to onError.
 */
export class ExportQueueBudget {
  private count = 0;
  private size = 0;
  private reported = false;

  constructor(
    private readonly maxCount: number,
    private readonly maxSize: number,
    private readonly label: string,
    private readonly onError?: (error: Error) => void,
  ) {}

  admit(record: QueuedRecord): boolean {
    const size = recordSize(record);

    if (this.count >= this.maxCount || (this.count > 0 && this.size + size > this.maxSize)) {
      if (!this.reported) {
        this.reported = true;
        reportError(
          this.onError,
          new Error(
            `telemetry.dev: the ${this.label} export queue is full (${this.maxCount} records or ${this.maxSize} bytes); dropping ${this.label} records until it drains`,
          ),
        );
      }

      return false;
    }

    this.reported = false;
    this.count += 1;
    this.size += size;

    return true;
  }

  release(records: readonly QueuedRecord[]): void {
    for (const record of records) {
      this.count = Math.max(this.count - 1, 0);
      this.size = Math.max(this.size - recordSize(record), 0);
    }
  }

  /** Wraps an exporter so that each batch it receives leaves the budget. */
  releasing<T extends QueuedRecord>(exporter: {
    export(items: T[], done: (result: ExportResult) => void): void;
    shutdown(): Promise<void>;
    forceFlush?(): Promise<void>;
  }) {
    return {
      export: (items: T[], done: (result: ExportResult) => void) => {
        this.release(items);
        exporter.export(items, done);
      },
      shutdown: () => exporter.shutdown(),
      forceFlush: () => exporter.forceFlush?.() ?? Promise.resolve(),
    };
  }
}
