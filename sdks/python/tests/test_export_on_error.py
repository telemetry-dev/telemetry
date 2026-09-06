from __future__ import annotations

# pyright: reportPrivateUsage=false
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.metrics.export import (
    Gauge,
    Metric,
    MetricExportResult,
    MetricsData,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.util.types import AttributeValue

from telemetry_dev._client import _ReportingLogExporter, _ReportingSpanExporter
from telemetry_dev._metrics import GuardedOTLPMetricExporter

_EMPTY_ATTRIBUTES: Mapping[str, AttributeValue] = {}


class FakeSpanExporter(SpanExporter):
    def __init__(self, outcome: SpanExportResult | BaseException) -> None:
        self._outcome = outcome
        self.export_calls = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.export_calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class FakeLogExporter(LogRecordExporter):
    def __init__(self, outcome: LogRecordExportResult | BaseException) -> None:
        self._outcome = outcome
        self.export_calls = 0

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        self.export_calls += 1
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    def shutdown(self) -> None:
        pass


def _metrics_data_with_data_points(count: int) -> MetricsData:
    return MetricsData(
        resource_metrics=[
            ResourceMetrics(
                resource=Resource(_EMPTY_ATTRIBUTES),
                scope_metrics=[
                    ScopeMetrics(
                        scope=InstrumentationScope("test"),
                        metrics=[
                            Metric(
                                name="requests",
                                description=None,
                                unit="1",
                                data=Gauge(
                                    data_points=[
                                        NumberDataPoint(
                                            attributes=_EMPTY_ATTRIBUTES,
                                            start_time_unix_nano=1,
                                            time_unix_nano=2,
                                            value=index,
                                        )
                                        for index in range(count)
                                    ]
                                ),
                            )
                        ],
                        schema_url="",
                    )
                ],
                schema_url="",
            )
        ]
    )


def test_span_export_failure_result_reports_error_and_passes_result_through() -> None:
    errors: list[BaseException] = []
    inner = FakeSpanExporter(SpanExportResult.FAILURE)

    result = _ReportingSpanExporter(inner, errors.append).export(())

    assert result is SpanExportResult.FAILURE
    assert inner.export_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "span export failed" in str(errors[0])


def test_span_export_exception_reports_original_error_and_returns_failure() -> None:
    errors: list[BaseException] = []
    boom = ConnectionError("span transport down")
    inner = FakeSpanExporter(boom)

    result = _ReportingSpanExporter(inner, errors.append).export(())

    assert result is SpanExportResult.FAILURE
    assert inner.export_calls == 1
    assert errors == [boom]


def test_span_export_success_does_not_report_error() -> None:
    errors: list[BaseException] = []
    inner = FakeSpanExporter(SpanExportResult.SUCCESS)

    result = _ReportingSpanExporter(inner, errors.append).export(())

    assert result is SpanExportResult.SUCCESS
    assert inner.export_calls == 1
    assert errors == []


def test_log_export_failure_result_reports_error_and_passes_result_through() -> None:
    errors: list[BaseException] = []
    inner = FakeLogExporter(LogRecordExportResult.FAILURE)

    result = _ReportingLogExporter(inner, errors.append).export(())

    assert result is LogRecordExportResult.FAILURE
    assert inner.export_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "log export failed" in str(errors[0])


def test_log_export_exception_reports_original_error_and_returns_failure() -> None:
    errors: list[BaseException] = []
    boom = ConnectionError("log transport down")
    inner = FakeLogExporter(boom)

    result = _ReportingLogExporter(inner, errors.append).export(())

    assert result is LogRecordExportResult.FAILURE
    assert inner.export_calls == 1
    assert errors == [boom]


def test_log_export_success_does_not_report_error() -> None:
    errors: list[BaseException] = []
    inner = FakeLogExporter(LogRecordExportResult.SUCCESS)

    result = _ReportingLogExporter(inner, errors.append).export(())

    assert result is LogRecordExportResult.SUCCESS
    assert inner.export_calls == 1
    assert errors == []


def test_metric_export_failure_result_reports_error_and_passes_result_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[BaseException] = []
    calls: list[MetricsData] = []

    def export_failure(
        self: OTLPMetricExporter,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        calls.append(metrics_data)
        return MetricExportResult.FAILURE

    monkeypatch.setattr(OTLPMetricExporter, "export", export_failure)
    exporter = GuardedOTLPMetricExporter("http://example.test/v1/metrics", on_error=errors.append)
    metrics_data = _metrics_data_with_data_points(1)

    result = exporter.export(metrics_data)

    assert result is MetricExportResult.FAILURE
    assert calls == [metrics_data]
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "metric export failed" in str(errors[0])


def test_metric_export_exception_reports_original_error_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[BaseException] = []
    calls: list[MetricsData] = []
    boom = TimeoutError("metric transport down")

    def export_raises(
        self: OTLPMetricExporter,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        calls.append(metrics_data)
        raise boom

    monkeypatch.setattr(OTLPMetricExporter, "export", export_raises)
    exporter = GuardedOTLPMetricExporter("http://example.test/v1/metrics", on_error=errors.append)
    metrics_data = _metrics_data_with_data_points(1)

    result = exporter.export(metrics_data)

    assert result is MetricExportResult.FAILURE
    assert calls == [metrics_data]
    assert errors == [boom]


def test_metric_export_success_does_not_report_error(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[BaseException] = []
    calls: list[MetricsData] = []

    def export_success(
        self: OTLPMetricExporter,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        calls.append(metrics_data)
        return MetricExportResult.SUCCESS

    monkeypatch.setattr(OTLPMetricExporter, "export", export_success)
    exporter = GuardedOTLPMetricExporter("http://example.test/v1/metrics", on_error=errors.append)
    metrics_data = _metrics_data_with_data_points(1)

    result = exporter.export(metrics_data)

    assert result is MetricExportResult.SUCCESS
    assert calls == [metrics_data]
    assert errors == []


def test_metric_export_skips_empty_collections_without_reporting_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[BaseException] = []
    calls: list[MetricsData] = []

    def export_unexpected(
        self: OTLPMetricExporter,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        calls.append(metrics_data)
        raise AssertionError("empty metrics should not be exported")

    monkeypatch.setattr(OTLPMetricExporter, "export", export_unexpected)
    exporter = GuardedOTLPMetricExporter("http://example.test/v1/metrics", on_error=errors.append)

    result = exporter.export(_metrics_data_with_data_points(0))

    assert result is MetricExportResult.SUCCESS
    assert calls == []
    assert errors == []
