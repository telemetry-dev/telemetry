import { type Attributes, type Context, TraceFlags } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  type ReadableSpan,
  SimpleSpanProcessor,
  type Span,
  type SpanExporter,
  type SpanProcessor,
} from "@opentelemetry/sdk-trace-base";

import {
  type BatchOptions,
  batchExportTimeoutMillis,
  DEFAULT_BATCH,
  type ExportMode,
  resolveBatch,
} from "./config.ts";
import { activeContext, PROPAGATED_KEY, propagatedFromContext } from "./context.ts";
import { reportError } from "./debug.ts";
import { ExportQueueBudget, streamCaptureBudget } from "./queue.ts";

export interface StampingProcessorOptions {
  exporter: SpanExporter;
  exportMode: ExportMode;
  /** The byte bounds default as in resolveBatch; an invalid one is reported and defaulted. */
  batch: Required<Omit<BatchOptions, "maxQueueBytes" | "maxStreamBytes">> &
    Pick<BatchOptions, "maxQueueBytes" | "maxStreamBytes">;
  spanFilter?: (span: ReadableSpan) => boolean;
  recordMetrics?: (span: ReadableSpan) => void;
  onError?: (error: Error) => void;
  /** The exporter applies `batch.exportTimeoutMillis` to each request, as the telemetry.dev exporter does. */
  exporterTimesEachRequest?: boolean;
}

/**
 * The vendor span processor: stamps propagated correlation attributes onto every span at start,
 * then filters, records auto-metrics, and delegates to a Batch/SimpleSpanProcessor at end.
 */
export class StampingSpanProcessor implements SpanProcessor {
  private readonly inner: SpanProcessor;
  private readonly budget?: ExportQueueBudget;

  constructor(private readonly options: StampingProcessorOptions) {
    const batch = {
      ...options.batch,
      maxQueueBytes: DEFAULT_BATCH.maxQueueBytes,
      maxStreamBytes: DEFAULT_BATCH.maxStreamBytes,
    };

    // Each bound on its own, so that an invalid one does not reset the other.
    for (const key of ["maxQueueBytes", "maxStreamBytes"] as const) {
      try {
        batch[key] = resolveBatch({ [key]: options.batch[key] })[key];
      } catch (error) {
        reportError(options.onError, error);
      }
    }
    streamCaptureBudget.maxBytes = batch.maxStreamBytes;

    if (options.exportMode === "immediate") {
      this.inner = new SimpleSpanProcessor(options.exporter);
    } else {
      this.budget = new ExportQueueBudget(
        batch.maxQueueSize,
        batch.maxQueueBytes,
        "span",
        options.onError,
      );
      this.inner = new BatchSpanProcessor(this.budget.releasing(options.exporter), {
        ...batch,
        exportTimeoutMillis: options.exporterTimesEachRequest
          ? batchExportTimeoutMillis(batch.exportTimeoutMillis, batch.maxExportBatchSize)
          : batch.exportTimeoutMillis,
      });
    }
  }

  onStart(span: Span, parentContext: Context): void {
    try {
      const propagated =
        (parentContext.getValue(PROPAGATED_KEY) as Attributes | undefined) ??
        propagatedFromContext(activeContext());

      if (propagated) span.setAttributes(propagated);
    } catch (error) {
      reportError(this.options.onError, error);
    }

    this.inner.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    try {
      if (this.options.spanFilter && !this.options.spanFilter(span)) return;
    } catch (error) {
      // A throwing filter must not drop spans; fall through and export.
      reportError(this.options.onError, error);
    }

    try {
      this.options.recordMetrics?.(span);
    } catch (error) {
      reportError(this.options.onError, error);
    }

    // The batch processor drops unsampled spans itself, so the budget does not count them.
    const sampled = (span.spanContext().traceFlags & TraceFlags.SAMPLED) !== 0;

    if (this.budget && sampled && !this.budget.admit(span)) return;

    this.inner.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown();
  }
}
