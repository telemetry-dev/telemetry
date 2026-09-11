from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Histogram,
    HistogramDataPoint,
    Metric,
    MetricExportResult,
    MetricsData,
    ResourceMetrics,
    ScopeMetrics,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan, set_span_in_context

from ._config import report_error
from ._semconv import (
    ATTR_ERROR_TYPE,
    ATTR_TIME_TO_FIRST_CHUNK,
    DURATION_BUCKETS,
    DURATION_METRIC_OPERATIONS,
    METRIC_ATTR_KEYS,
    SCOPE_NAME,
    TOKEN_BUCKETS,
    TOKEN_METRIC_OPERATIONS,
    USAGE_ATTRS,
)

_INPUT_TOKENS_ATTR = USAGE_ATTRS["input_tokens"]
_OUTPUT_TOKENS_ATTR = USAGE_ATTRS["output_tokens"]
_MAX_CHUNK_ATTRIBUTE_SETS = 2000
_CHUNK_METRIC_NAME = "gen_ai.client.operation.time_per_output_chunk"


@dataclass
class _ChunkAggregate:
    count: int
    sum: float
    min: float
    max: float
    buckets: list[int]


class OutputChunkAggregation:
    def __init__(self, max_attribute_sets: int = _MAX_CHUNK_ATTRIBUTE_SETS) -> None:
        self._max_attribute_sets = max_attribute_sets
        self._values: dict[tuple[tuple[str, Any], ...], _ChunkAggregate] = {}
        self._lock = threading.Lock()
        self._collection_start = time.time_ns()

    def add(
        self,
        attributes: dict[str, Any],
        aggregate: tuple[int, float, float, float, tuple[int, ...]],
    ) -> None:
        key = tuple(sorted(attributes.items()))
        with self._lock:
            if key not in self._values and len(self._values) >= self._max_attribute_sets - 1:
                key = (("otel.metric.overflow", True),)
            count, total, minimum, maximum, buckets = aggregate
            existing = self._values.get(key)
            if existing is None:
                self._values[key] = _ChunkAggregate(count, total, minimum, maximum, list(buckets))
                return
            existing.count += count
            existing.sum += total
            existing.min = min(existing.min, minimum)
            existing.max = max(existing.max, maximum)
            for index, value in enumerate(buckets):
                existing.buckets[index] += value

    def drain_points(self) -> tuple[HistogramDataPoint, ...]:
        with self._lock:
            values, self._values = self._values, {}
            now = time.time_ns()
            start, self._collection_start = self._collection_start, now
        return tuple(
            HistogramDataPoint(
                attributes=dict(key),
                start_time_unix_nano=start,
                time_unix_nano=now,
                count=value.count,
                sum=value.sum,
                bucket_counts=tuple(value.buckets),
                explicit_bounds=tuple(DURATION_BUCKETS),
                min=value.min,
                max=value.max,
                exemplars=(),
            )
            for key, value in values.items()
        )


def _with_output_chunks(
    metrics_data: MetricsData, aggregation: OutputChunkAggregation
) -> MetricsData:
    resource_metrics = list(metrics_data.resource_metrics)
    target = next(
        (
            (resource_index, scope_index)
            for resource_index, resource in enumerate(resource_metrics)
            for scope_index, scope in enumerate(resource.scope_metrics)
            if scope.scope.name == SCOPE_NAME
        ),
        None,
    )
    if target is None:
        return metrics_data
    points = aggregation.drain_points()
    if not points:
        return metrics_data
    metric = Metric(
        name=_CHUNK_METRIC_NAME,
        description="Time between consecutive non-empty GenAI output chunks",
        unit="s",
        data=Histogram(points, AggregationTemporality.DELTA),
    )
    resource_index, scope_index = target
    resource = resource_metrics[resource_index]
    scopes = list(resource.scope_metrics)
    scope = scopes[scope_index]
    scopes[scope_index] = ScopeMetrics(scope.scope, [*scope.metrics, metric], scope.schema_url)
    resource_metrics[resource_index] = ResourceMetrics(
        resource.resource, scopes, resource.schema_url
    )
    return MetricsData(resource_metrics)


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
        self.output_chunks: OutputChunkAggregation | None = None

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        if self.output_chunks is not None:
            metrics_data = _with_output_chunks(metrics_data, self.output_chunks)
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
        output_chunks: OutputChunkAggregation | None = None,
    ) -> None:
        self._on_error = on_error
        self._output_chunks = output_chunks
        self._custom_reader_limitation_reported = False
        self._custom_reader_limitation_lock = threading.Lock()
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
        self._first_chunk = meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
        )

    def record_span(self, span: ReadableSpan) -> None:
        try:
            attributes = span.attributes or {}
            operation = attributes.get("gen_ai.operation.name")
            if not isinstance(operation, str) or operation not in DURATION_METRIC_OPERATIONS:
                return
            metric_attrs = {key: attributes[key] for key in METRIC_ATTR_KEYS if key in attributes}
            if operation == "chat":
                from ._spans import output_chunk_aggregate

                chunk_aggregate = output_chunk_aggregate(span)
                if chunk_aggregate is not None:
                    if self._output_chunks is not None:
                        self._output_chunks.add(metric_attrs, chunk_aggregate)
                    else:
                        with self._custom_reader_limitation_lock:
                            should_report = not self._custom_reader_limitation_reported
                            self._custom_reader_limitation_reported = True
                        if should_report:
                            report_error(
                                self._on_error,
                                "output chunk interval metric is unavailable with a custom "
                                "metric_reader",
                                RuntimeError(
                                    "gen_ai.client.operation.time_per_output_chunk requires the "
                                    "built-in OTLP metric exporter; ordinary auto-metrics remain "
                                    "enabled"
                                ),
                            )
            context = set_span_in_context(NonRecordingSpan(span.context or INVALID_SPAN_CONTEXT))
            if span.end_time is not None and span.start_time is not None:
                duration_s = max(span.end_time - span.start_time, 0) / 1e9
                error_type = attributes.get(ATTR_ERROR_TYPE)
                duration_attrs = (
                    {**metric_attrs, ATTR_ERROR_TYPE: error_type}
                    if isinstance(error_type, str)
                    else metric_attrs
                )
                self._duration.record(duration_s, duration_attrs, context=context)
            first_chunk_seconds = attributes.get(ATTR_TIME_TO_FIRST_CHUNK)
            if (
                operation == "chat"
                and isinstance(first_chunk_seconds, (int, float))
                and not isinstance(first_chunk_seconds, bool)
                and math.isfinite(first_chunk_seconds)
                and first_chunk_seconds >= 0
            ):
                self._first_chunk.record(first_chunk_seconds, metric_attrs, context=context)
            if operation not in TOKEN_METRIC_OPERATIONS:
                return
            input_tokens = attributes.get(_INPUT_TOKENS_ATTR)
            if isinstance(input_tokens, int):
                self._tokens.record(
                    input_tokens, {**metric_attrs, "gen_ai.token.type": "input"}, context=context
                )
            output_tokens = attributes.get(_OUTPUT_TOKENS_ATTR)
            if isinstance(output_tokens, int):
                self._tokens.record(
                    output_tokens, {**metric_attrs, "gen_ai.token.type": "output"}, context=context
                )
        except BaseException as exc:
            report_error(self._on_error, "failed to record auto-metrics for span", exc)
