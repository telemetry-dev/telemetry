from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics.export import MetricExportResult, MetricsData
from opentelemetry.sdk.trace import ReadableSpan

from ._config import report_error
from ._semconv import (
    ATTR_ERROR_TYPE,
    DURATION_BUCKETS,
    DURATION_METRIC_OPERATIONS,
    METRIC_ATTR_KEYS,
    TOKEN_BUCKETS,
    TOKEN_METRIC_OPERATIONS,
    USAGE_ATTRS,
)

_INPUT_TOKENS_ATTR = USAGE_ATTRS["input_tokens"]
_OUTPUT_TOKENS_ATTR = USAGE_ATTRS["output_tokens"]


def _has_data_points(metrics_data: MetricsData) -> bool:
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if len(metric.data.data_points) > 0:
                    return True
    return False


class GuardedOTLPMetricExporter(OTLPMetricExporter):
    """OTLP metric exporter that skips POSTs for collections with zero data points
    (mirrors the TS emitter's empty-datapoint guard) and surfaces failures to on_error."""

    def __init__(
        self,
        *args: Any,
        on_error: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        self._td_on_error = on_error

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        if not _has_data_points(metrics_data):
            return MetricExportResult.SUCCESS
        try:
            result = super().export(  # pyright: ignore[reportUnknownMemberType]
                metrics_data, timeout_millis=timeout_millis, **kwargs
            )
        except BaseException as exc:
            report_error(self._td_on_error, "metric export failed", exc)
            return MetricExportResult.FAILURE
        if result is MetricExportResult.FAILURE:
            report_error(
                self._td_on_error,
                "metric export failed",
                RuntimeError("OTLP metric export failed (check the API key and ingest URL)"),
            )
        return result


class MetricsRecorder:
    """Records the auto GenAI histograms from ended spans (called by the span processor)."""

    def __init__(
        self,
        meter: Meter,
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._on_error = on_error
        self._duration = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI client operations",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )
        self._tokens = meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used by GenAI clients",
            explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
        )

    def record_span(self, span: ReadableSpan) -> None:
        try:
            attributes = span.attributes or {}
            operation = attributes.get("gen_ai.operation.name")
            if not isinstance(operation, str) or operation not in DURATION_METRIC_OPERATIONS:
                return
            metric_attrs = {key: attributes[key] for key in METRIC_ATTR_KEYS if key in attributes}
            if span.end_time is not None and span.start_time is not None:
                duration_s = max(span.end_time - span.start_time, 0) / 1e9
                error_type = attributes.get(ATTR_ERROR_TYPE)
                duration_attrs = (
                    {**metric_attrs, ATTR_ERROR_TYPE: error_type}
                    if isinstance(error_type, str)
                    else metric_attrs
                )
                self._duration.record(duration_s, duration_attrs)
            if operation not in TOKEN_METRIC_OPERATIONS:
                return
            input_tokens = attributes.get(_INPUT_TOKENS_ATTR)
            if isinstance(input_tokens, int):
                self._tokens.record(input_tokens, {**metric_attrs, "gen_ai.token.type": "input"})
            output_tokens = attributes.get(_OUTPUT_TOKENS_ATTR)
            if isinstance(output_tokens, int):
                self._tokens.record(output_tokens, {**metric_attrs, "gen_ai.token.type": "output"})
        except BaseException as exc:
            report_error(self._on_error, "failed to record auto-metrics for span", exc)
