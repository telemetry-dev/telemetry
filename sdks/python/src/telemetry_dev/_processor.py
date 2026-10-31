from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)

from ._config import report_error
from ._context import propagated_attributes
from ._export import DEFAULT_MAX_QUEUE_BYTES, ExportQueueBudget
from ._metrics import MetricsRecorder

ExportMode = Literal["batched", "immediate"]


class StampingSpanProcessor(SpanProcessor):
    """Stamps propagated attributes, filters spans, records metrics, then exports."""

    def __init__(
        self,
        span_exporter: SpanExporter,
        *,
        export_mode: ExportMode = "batched",
        max_export_batch_size: int = 64,
        schedule_delay_millis: float = 1000,
        max_queue_size: int = 2048,
        max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES,
        export_timeout_millis: float = 30000,
        span_filter: Callable[[ReadableSpan], bool] | None = None,
        metrics_recorder: MetricsRecorder | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._span_filter = span_filter
        self._metrics_recorder = metrics_recorder
        self._on_error = on_error
        self._budget: ExportQueueBudget | None = None
        if export_mode == "immediate":
            self._inner: SpanProcessor = SimpleSpanProcessor(span_exporter)
        else:
            self._budget = ExportQueueBudget(max_queue_size, max_queue_bytes, "span", on_error)
            self._inner = BatchSpanProcessor(
                self._budget.releasing(span_exporter),
                max_queue_size=max_queue_size,
                schedule_delay_millis=schedule_delay_millis,
                max_export_batch_size=max_export_batch_size,
                export_timeout_millis=export_timeout_millis,
            )

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            attrs = propagated_attributes(parent_context)
            if attrs:
                span.set_attributes(attrs)
        except BaseException as exc:
            report_error(self._on_error, "failed to stamp propagated attributes", exc)
        self._inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        if self._span_filter is not None:
            try:
                if not self._span_filter(span):
                    return
            except BaseException as exc:
                # Fail open: a broken filter must not drop telemetry.
                report_error(self._on_error, "span_filter raised; exporting span anyway", exc)
        # Metrics are recorded only for exported spans (after the filter), matching the TS SDK.
        if self._metrics_recorder is not None:
            self._metrics_recorder.record_span(span)
        # The batch processor drops unsampled spans itself, so the budget does not count them.
        sampled = span.context is not None and span.context.trace_flags.sampled
        if self._budget is not None and sampled and not self._budget.admit(span):
            return
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
