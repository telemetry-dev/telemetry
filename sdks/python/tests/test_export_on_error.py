from __future__ import annotations

# pyright: reportPrivateUsage=false
import threading
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
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.util.types import AttributeValue

import telemetry_dev
from telemetry_dev._export import ExportFailures, ReportingLogExporter, ReportingSpanExporter
from telemetry_dev._metrics import GuardedOTLPMetricExporter
from telemetry_dev.otel import TelemetrySpanProcessor
from tests.otlp_capture import otlp_capture_server

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

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

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

    result = ReportingSpanExporter(inner, ExportFailures(errors.append)).export(())

    assert result is SpanExportResult.FAILURE
    assert inner.export_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "span export failed" in str(errors[0])


def test_span_export_exception_reports_original_error_and_returns_failure() -> None:
    errors: list[BaseException] = []
    boom = ConnectionError("span transport down")
    inner = FakeSpanExporter(boom)

    result = ReportingSpanExporter(inner, ExportFailures(errors.append)).export(())

    assert result is SpanExportResult.FAILURE
    assert inner.export_calls == 1
    assert errors == [boom]


def test_span_export_success_does_not_report_error() -> None:
    errors: list[BaseException] = []
    inner = FakeSpanExporter(SpanExportResult.SUCCESS)

    result = ReportingSpanExporter(inner, ExportFailures(errors.append)).export(())

    assert result is SpanExportResult.SUCCESS
    assert inner.export_calls == 1
    assert errors == []


def test_log_export_failure_result_reports_error_and_passes_result_through() -> None:
    errors: list[BaseException] = []
    inner = FakeLogExporter(LogRecordExportResult.FAILURE)

    result = ReportingLogExporter(inner, ExportFailures(errors.append)).export(())

    assert result is LogRecordExportResult.FAILURE
    assert inner.export_calls == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "log export failed" in str(errors[0])


def test_log_export_exception_reports_original_error_and_returns_failure() -> None:
    errors: list[BaseException] = []
    boom = ConnectionError("log transport down")
    inner = FakeLogExporter(boom)

    result = ReportingLogExporter(inner, ExportFailures(errors.append)).export(())

    assert result is LogRecordExportResult.FAILURE
    assert inner.export_calls == 1
    assert errors == [boom]


def test_log_export_success_does_not_report_error() -> None:
    errors: list[BaseException] = []
    inner = FakeLogExporter(LogRecordExportResult.SUCCESS)

    result = ReportingLogExporter(inner, ExportFailures(errors.append)).export(())

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
    exporter = GuardedOTLPMetricExporter(
        "http://example.test/v1/metrics", failures=ExportFailures(errors.append)
    )
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
    exporter = GuardedOTLPMetricExporter(
        "http://example.test/v1/metrics", failures=ExportFailures(errors.append)
    )
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
    exporter = GuardedOTLPMetricExporter(
        "http://example.test/v1/metrics", failures=ExportFailures(errors.append)
    )
    metrics_data = _metrics_data_with_data_points(1)

    result = exporter.export(metrics_data)

    assert result is MetricExportResult.SUCCESS
    assert calls == [metrics_data]
    assert errors == []


def test_metric_export_skips_empty_collections_without_reporting_or_ending_a_failure_series(
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
    exporter = GuardedOTLPMetricExporter(
        "http://example.test/v1/metrics", failures=ExportFailures(errors.append)
    )
    empty = _metrics_data_with_data_points(0)
    failing = _metrics_data_with_data_points(1)

    assert exporter.export(empty) is MetricExportResult.SUCCESS
    assert errors == []
    assert exporter.export(failing) is MetricExportResult.FAILURE
    # The empty collection sends no request, so the series of failures continues.
    assert exporter.export(empty) is MetricExportResult.SUCCESS
    assert exporter.export(failing) is MetricExportResult.FAILURE

    assert calls == [failing, failing]
    assert len(errors) == 1


def test_an_ingest_outage_is_one_report_until_an_export_reaches_the_ingest_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors: list[BaseException] = []

    with otlp_capture_server() as ingest:
        ingest.status = 503
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            export_mode="immediate",
            log_level="error",
            disable_atexit=True,
            # Shorter than the first retry delay, so the exporter does not retry a 503.
            timeout=0.5,
            on_error=errors.append,
        )

        # One outage stops traces, logs, and metrics.
        telemetry_dev.start_span("chat", type="generation").end()
        telemetry_dev.log("first outage")
        telemetry_dev.flush()
        assert sorted(request.path for request in ingest.requests) == [
            "/v1/logs",
            "/v1/metrics",
            "/v1/traces",
        ]
        # A flush with no new metrics sends no metric request, and the series continues.
        telemetry_dev.flush()
        assert len(ingest.requests_for("/v1/metrics")) == 1
        telemetry_dev.log("still in the first outage")
        assert len(errors) == 1

        ingest.status = 200
        telemetry_dev.log("the ingest is back")
        ingest.status = 503
        telemetry_dev.log("second outage")
        telemetry_dev.log("still in the second outage")
        telemetry_dev.shutdown()

    assert len(errors) == 2
    assert "span export failed" in str(errors[0])
    assert "log export failed" in str(errors[1])
    logged = [record for record in caplog.records if record.name == "telemetry_dev"]
    assert len(logged) == 2


def test_a_failure_during_a_slow_report_on_another_thread_is_quiet_and_does_not_wait() -> None:
    errors: list[BaseException] = []
    first_report_started = threading.Event()
    second_failure_done = threading.Event()

    def slow_on_error(error: BaseException) -> None:
        errors.append(error)
        if len(errors) == 1:
            first_report_started.set()
            second_failure_done.wait(timeout=5)

    failures = ExportFailures(slow_on_error)
    spans = ReportingSpanExporter(FakeSpanExporter(SpanExportResult.FAILURE), failures)
    logs = ReportingLogExporter(FakeLogExporter(LogRecordExportResult.FAILURE), failures)

    span_thread = threading.Thread(target=spans.export, args=((),))
    span_thread.start()
    assert first_report_started.wait(timeout=5)
    assert logs.export(()) is LogRecordExportResult.FAILURE
    # The second export did not wait for the first on_error call.
    assert span_thread.is_alive()
    second_failure_done.set()
    span_thread.join(timeout=5)

    assert len(errors) == 1


def test_an_outage_that_stops_byo_traces_and_metrics_is_one_report() -> None:
    errors: list[BaseException] = []

    with otlp_capture_server() as ingest:
        # The exporters do not retry a 400, thus each export stops after one request.
        ingest.status = 400
        processor = TelemetrySpanProcessor(
            api_key="td_live_test",
            base_url=ingest.url,
            export_mode="immediate",
            on_error=errors.append,
        )
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(processor)
        span = provider.get_tracer("user-app").start_span("chat")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.end()
        # The shutdown sends the metrics.
        provider.shutdown()

    assert sorted(request.path for request in ingest.requests) == ["/v1/metrics", "/v1/traces"]
    assert len(errors) == 1
    assert "span export failed" in str(errors[0])
