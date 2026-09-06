import type { Attributes, Context } from "@opentelemetry/api";
import {
  BatchSpanProcessor,
  type ReadableSpan,
  SimpleSpanProcessor,
  type Span,
  type SpanExporter,
  type SpanProcessor,
} from "@opentelemetry/sdk-trace-base";

import type { BatchOptions, ExportMode } from "./config.ts";
import { activeContext, PROPAGATED_KEY, propagatedFromContext } from "./context.ts";
import { reportError } from "./debug.ts";

export interface StampingProcessorOptions {
  exporter: SpanExporter;
  exportMode: ExportMode;
  batch: Required<BatchOptions>;
  spanFilter?: (span: ReadableSpan) => boolean;
  recordMetrics?: (span: ReadableSpan) => void;
  onError?: (error: Error) => void;
}

/**
 * The vendor span processor: stamps propagated correlation attributes onto every span at start,
 * then filters, records auto-metrics, and delegates to a Batch/SimpleSpanProcessor at end.
 */
export class StampingSpanProcessor implements SpanProcessor {
  private readonly inner: SpanProcessor;

  constructor(private readonly options: StampingProcessorOptions) {
    this.inner =
      options.exportMode === "immediate"
        ? new SimpleSpanProcessor(options.exporter)
        : new BatchSpanProcessor(options.exporter, options.batch);
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
    this.inner.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return this.inner.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.inner.shutdown();
  }
}
