import type { ExportResult } from "@opentelemetry/core";

import { reportError } from "./debug.ts";

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
