from __future__ import annotations

# pyright: reportPrivateUsage=false
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    LogRecordExporter,
    LogRecordExportResult,
)
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
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.util._once import Once
from opentelemetry.util.types import AttributeValue

import telemetry_dev
from telemetry_dev._export import (
    ExportFailures,
    ReportingLogExporter,
    ReportingSpanExporter,
    _span_with_attributes,
    after_export,
    exporting,
)
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


_MAX_BODY_BYTES = 16 * 1024 * 1024


def test_batches_over_the_ingest_limit_are_split_into_requests_that_fit_and_time_out_alone() -> (
    None
):
    errors: list[BaseException] = []
    with otlp_capture_server() as ingest:
        # Two requests take longer than one timeout, so a timeout over the whole export fails this.
        ingest.post_delay = 0.6
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            log_level="error",
            disable_atexit=True,
            timeout=1.0,
            on_error=errors.append,
        )
        # Two 9 MB spans: together they exceed the 16 MiB limit, each one fits.
        for name in ("first", "second"):
            telemetry_dev.start_span(name, attributes={"payload": "p" * 9_000_000}).end()
        telemetry_dev.flush()
        telemetry_dev.shutdown()

        bodies = [request.decompressed() for request in ingest.requests_for("/v1/traces")]
        assert errors == []
        assert len(bodies) == 2
        for body in bodies:
            assert len(body) <= _MAX_BODY_BYTES
            assert body.count(b"p" * 9_000_000) == 1


@pytest.mark.parametrize("kind", ["span", "log"])
def test_a_record_over_the_ingest_limit_is_sent_with_its_longest_attribute_cut(
    kind: str,
) -> None:
    errors: list[BaseException] = []
    attributes = {"first": "a" * 10_000_000, "other": "z" * 9_000_000, "keep": "kept"}
    with otlp_capture_server() as ingest:
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            export_mode="immediate",
            log_level="error",
            disable_atexit=True,
            on_error=errors.append,
        )
        if kind == "span":
            telemetry_dev.start_span("oversized", attributes=attributes).end()
        else:
            telemetry_dev.log("oversized", attributes=attributes)
        telemetry_dev.shutdown()

        requests = ingest.requests_for("/v1/traces" if kind == "span" else "/v1/logs")
        assert len(requests) == 1
        body = requests[0].decompressed()
        # Only the excess is cut: the request ends up just under the limit, not far below it.
        assert _MAX_BODY_BYTES - 64 < len(body) <= _MAX_BODY_BYTES
        assert b"z" * 9_000_000 in body
        assert b"a...[truncated]" in body
        assert b"kept" in body
        # The flag's key followed by its value, an AnyValue with bool_value true.
        assert b"telemetry.dev.capture.truncated\x12\x02\x10\x01" in body
        assert any("longest values were truncated" in str(error) for error in errors)


def test_a_log_record_whose_message_is_over_the_ingest_limit_is_sent_with_its_message_cut() -> None:
    errors: list[BaseException] = []
    with otlp_capture_server() as ingest:
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            export_mode="immediate",
            log_level="error",
            disable_atexit=True,
            on_error=errors.append,
        )
        telemetry_dev.log("b" * 17_000_000, attributes={"keep": "kept"})
        telemetry_dev.shutdown()

        requests = ingest.requests_for("/v1/logs")
        assert len(requests) == 1
        body = requests[0].decompressed()
        assert _MAX_BODY_BYTES - 64 < len(body) <= _MAX_BODY_BYTES
        assert b"b...[truncated]" in body
        assert b"kept" in body
        assert b"telemetry.dev.capture.truncated\x12\x02\x10\x01" in body
        assert any("longest values were truncated" in str(error) for error in errors)


def test_a_span_that_fits_once_its_multi_byte_text_is_cut_is_sent_not_dropped() -> None:
    finished = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(finished))
    provider.get_tracer("test").start_span("oversized").end()
    span = finished.get_finished_spans()[0]

    def with_payload(length: int) -> ReadableSpan:
        return _span_with_attributes(span, {"payload": ["p" * length], "emoji": "\U0001f916" * 100})

    # The array cannot be cut. Cutting the emoji to the marker removes 386 bytes but only 186
    # UTF-16 units, so the count of units alone says the span still does not fit.
    base = encode_spans([with_payload(16_000_000)]).ByteSize()
    record = with_payload(16_000_000 + _MAX_BODY_BYTES - base + 300)
    sent: list[bytes] = []

    class Capture(FakeSpanExporter):
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            sent.append(encode_spans(spans).SerializeToString())
            return SpanExportResult.SUCCESS

    exporter = ReportingSpanExporter(Capture(SpanExportResult.SUCCESS), ExportFailures(None))
    assert exporter.export([record]) is SpanExportResult.SUCCESS
    assert len(sent) == 1
    assert len(sent[0]) <= _MAX_BODY_BYTES
    assert b"telemetry.dev.capture.truncated\x12\x02\x10\x01" in sent[0]
    assert "\U0001f916".encode() not in sent[0]
    provider.shutdown()


def test_a_span_over_the_ingest_limit_without_a_string_to_cut_is_dropped() -> None:
    errors: list[BaseException] = []
    with otlp_capture_server() as ingest:
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            export_mode="immediate",
            log_level="error",
            disable_atexit=True,
            on_error=errors.append,
        )
        telemetry_dev.start_span("oversized", attributes={"payload": ["p" * 17_000_000]}).end()
        telemetry_dev.shutdown()

        assert ingest.requests_for("/v1/traces") == []
        assert any("exceeds max export size, dropped" in str(error) for error in errors)


def test_spans_past_max_queue_bytes_are_dropped_and_reported_once_until_the_queue_drains() -> None:
    errors: list[BaseException] = []
    exporter = InMemorySpanExporter()
    processor = TelemetrySpanProcessor(
        exporter,
        # Nothing exports before force_flush(), so every span waits in the queue.
        schedule_delay_millis=1_000_000,
        max_queue_bytes=2_500,
        metrics=False,
        on_error=errors.append,
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")

    def end_spans(count: int) -> None:
        for _ in range(count):
            tracer.start_span("tool", attributes={"input": "i" * 1_000}).end()

    # Two spans of 1,000 characters fit in 2,500; the third and later are dropped.
    end_spans(4)
    assert ["span export queue is full" in str(error) for error in errors] == [True]
    processor.force_flush()
    assert len(exporter.get_finished_spans()) == 2

    # The flush freed the queue, so new spans are admitted and a new drop is reported again.
    end_spans(3)
    processor.force_flush()
    assert len(exporter.get_finished_spans()) == 4
    assert len(errors) == 2

    # A span larger than max_queue_bytes still enters an empty queue.
    tracer.start_span("tool", attributes={"input": "i" * 5_000}).end()
    processor.force_flush()
    assert len(exporter.get_finished_spans()) == 5
    assert len(errors) == 2
    provider.shutdown()


def test_batched_logs_past_the_queue_bound_are_dropped_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("telemetry_dev._client.DEFAULT_MAX_QUEUE_BYTES", 2_500)
    errors: list[BaseException] = []
    logs = InMemoryLogRecordExporter()
    telemetry_dev.init(
        api_key="td_live_test",
        log_level="error",
        disable_atexit=True,
        span_exporter=InMemorySpanExporter(),
        log_exporter=logs,
        on_error=errors.append,
    )

    for _ in range(4):
        telemetry_dev.log("l" * 1_000)
    telemetry_dev.flush()

    assert len(logs.get_finished_logs()) == 2
    assert ["log export queue is full" in str(error) for error in errors] == [True]
    telemetry_dev.shutdown()


def _wait_until(condition: Callable[[], bool], seconds: float = 10) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("kind", ["span", "log", "metric"])
def test_an_on_error_that_flushes_during_an_export_does_not_hang_and_flushes_after_it(
    kind: str,
) -> None:
    reports: list[BaseException] = []

    def on_error(error: BaseException) -> None:
        # The SDK calls this inside the export, while the batch processor or the metric reader
        # holds its export lock. The span gives the metric reader new data to export, as other
        # threads can at any time. Its trace export fails too, so that a trace export that reaches
        # the ingest does not end the series of failed exports before the deferred flush.
        reports.append(error)
        ingest.path_status["/v1/traces"] = 400
        telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        telemetry_dev.flush(timeout_s=2)

    with otlp_capture_server() as ingest:
        if kind == "metric":
            ingest.path_status["/v1/metrics"] = 400
        else:
            ingest.status = 400
        telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            log_level="silent",
            disable_atexit=True,
            timeout=1.0,
            on_error=on_error,
        )
        if kind == "log":
            telemetry_dev.log("failed export")
        # A chat span records metrics, which only a flush sends before the 60 s interval.
        telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        if kind != "metric":
            # The batch processor exports on its own, and the flush that on_error asks for runs
            # once that failed export ends.
            assert _wait_until(lambda: len(ingest.requests_for("/v1/metrics")) == 1)
        flushed = threading.Event()

        def flush() -> None:
            telemetry_dev.flush(timeout_s=3)
            flushed.set()

        threading.Thread(target=flush, daemon=True).start()
        assert flushed.wait(10)
        if kind == "metric":
            # The flush that on_error asked for ran: it sends the metrics of the span that
            # on_error ended, which only a flush sends before the 60 s interval.
            assert _wait_until(lambda: len(ingest.requests_for("/v1/metrics")) == 2)
        assert len(reports) == 1
        telemetry_dev.shutdown()


@pytest.mark.parametrize("kind", ["span", "log", "metric"])
def test_an_on_error_that_shuts_down_during_an_export_ends_the_client_without_hanging(
    kind: str,
) -> None:
    reports: list[BaseException] = []

    def on_error(error: BaseException) -> None:
        reports.append(error)
        if kind == "metric":
            # New metric data, which only the shutdown sends. Its trace export fails too, so
            # that it continues the series of failed exports.
            ingest.path_status["/v1/traces"] = 400
            telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        telemetry_dev.shutdown(timeout_s=2)

    with otlp_capture_server() as ingest:
        # The exporters do not retry a 400. The other signals reach the ingest.
        ingest.path_status[
            {"span": "/v1/traces", "log": "/v1/logs", "metric": "/v1/metrics"}[kind]
        ] = 400
        client = telemetry_dev.init(
            api_key="td_live_test",
            base_url=ingest.url,
            log_level="silent",
            disable_atexit=True,
            timeout=1.0,
            on_error=on_error,
        )
        telemetry_dev.log("queued log")
        # A chat span records metrics, which only a flush or the shutdown sends before the 60 s
        # interval.
        telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        if kind == "metric":
            # The flush makes the metric export. It runs on another thread, so that a flush that
            # waits for itself does not stop the test.
            threading.Thread(
                target=telemetry_dev.flush, kwargs={"timeout_s": 3}, daemon=True
            ).start()
        assert _wait_until(lambda: len(reports) == 1)
        assert telemetry_dev.get_client() is None
        # The deferred shutdown runs once the export ends. It sends the queued logs and the
        # metrics, and in the metric row also the metrics of the span that on_error ended.
        metric_requests = 2 if kind == "metric" else 1
        assert _wait_until(lambda: len(ingest.requests_for("/v1/metrics")) == metric_requests)
        assert len(ingest.requests_for("/v1/logs")) == 1
        processor = client.processor
        assert processor is not None
        # A processor that is shut down does not flush.
        assert _wait_until(lambda: processor.force_flush(100) is False)
        assert len(reports) == 1


_EXIT_AFTER_ON_ERROR_SHUTS_DOWN = """
import sys
import threading

import telemetry_dev

reported = threading.Event()


def on_error(error):
    telemetry_dev.shutdown(timeout_s=5)
    reported.set()


telemetry_dev.init(
    api_key="td_test", base_url=sys.argv[1], log_level="silent", timeout=2.0, on_error=on_error
)
telemetry_dev.log("queued log")
# More spans than one batch of 64, so that the batch processor exports at once.
for _ in range(100):
    telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
assert reported.wait(10)
# The main thread returns while the deferred shutdown runs.
"""


def test_a_process_that_exits_after_on_error_shuts_down_sends_the_queued_telemetry_first() -> None:
    with otlp_capture_server() as ingest:
        ingest.path_status["/v1/traces"] = 400
        # Each request takes this long, so the deferred shutdown still runs when the main thread
        # of the process returns.
        ingest.post_delay = 0.3
        result = subprocess.run(
            [sys.executable, "-c", _EXIT_AFTER_ON_ERROR_SHUTS_DOWN, ingest.url],
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        paths = {request.path for request in ingest.requests}
    assert paths == {"/v1/traces", "/v1/logs", "/v1/metrics"}


def test_a_shutdown_from_on_error_runs_at_exit_when_its_thread_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports: list[BaseException] = []
    at_exit: list[Callable[[], object]] = []
    start = threading.Thread.start

    def start_or_fail(self: threading.Thread) -> None:
        if self.name == "telemetry-dev-after-export":
            raise RuntimeError("can't start new thread")
        start(self)

    def on_error(error: BaseException) -> None:
        reports.append(error)
        telemetry_dev.shutdown(timeout_s=2)

    with otlp_capture_server() as ingest:
        ingest.path_status["/v1/traces"] = 400
        telemetry_dev.init(
            api_key="td_test",
            base_url=ingest.url,
            log_level="silent",
            disable_atexit=True,
            timeout=1.0,
            on_error=on_error,
        )
        monkeypatch.setattr(threading.Thread, "start", start_or_fail)
        monkeypatch.setattr(atexit, "register", at_exit.append)
        telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        assert _wait_until(lambda: len(at_exit) == 1)
        assert telemetry_dev.get_client() is None
        # Only the shutdown sends the metrics, and it waits for the exit.
        assert ingest.requests_for("/v1/metrics") == []
        at_exit[0]()
        assert len(ingest.requests_for("/v1/metrics")) == 1
        assert len(reports) == 1


@pytest.mark.parametrize(("shutdown", "daemon"), [(True, False), (False, True)])
def test_only_a_deferred_shutdown_holds_the_process_open(shutdown: bool, daemon: bool) -> None:
    # A flush can wait for a whole queue in an outage, so the process exit must not wait for it;
    # the shutdown at exit sends what is left.
    ran_on: list[bool] = []
    done = threading.Event()

    def export() -> None:
        with exporting():
            after_export(
                object(),
                lambda: (ran_on.append(threading.current_thread().daemon), done.set()),
                shutdown=shutdown,
            )

    # Export threads of OpenTelemetry are daemons, and a new thread inherits that.
    exporter = threading.Thread(target=export, daemon=True)
    exporter.start()
    exporter.join(10)

    assert done.wait(10)
    assert ran_on == [daemon]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_process_runs_the_work_that_its_parent_has_pending() -> None:
    key = object()
    release = threading.Event()
    with exporting():
        assert after_export(key, lambda: release.wait(10))
    ran = threading.Event()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            # The child has the key of the parent, but not its thread.
            with exporting():
                after_export(key, ran.set)
            code = 0 if ran.wait(5) else 2
        finally:
            # The child must not return into pytest.
            os._exit(code)
    release.set()
    # A fork hook that deadlocks the child fails the test instead of hanging the suite.
    deadline = time.monotonic() + 10
    while (waited := os.waitpid(pid, os.WNOHANG)) == (0, 0):
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("The forked process did not end in 10 s")
        time.sleep(0.01)
    assert os.waitstatus_to_exitcode(waited[1]) == 0


@pytest.mark.parametrize(
    ("owner", "call"),
    [("byo", "flush"), ("byo", "shutdown"), ("global", "flush"), ("global", "shutdown")],
)
def test_an_on_error_that_flushes_or_shuts_down_a_tracer_provider_during_an_export_does_not_hang(
    owner: str, call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports: list[BaseException] = []
    raised: list[BaseException] = []
    byo_provider = TracerProvider(shutdown_on_exit=False)

    def end_chat_span() -> None:
        if owner == "global":
            telemetry_dev.start_span("chat", type="generation", model="m", provider="p").end()
        else:
            span = byo_provider.get_tracer("user-app").start_span("chat")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.end()

    def on_error(error: BaseException) -> None:
        reports.append(error)
        # The span gives the deferred call new data to send. Its trace export fails too, so that
        # it continues the series of failed exports.
        ingest.path_status["/v1/traces"] = 400
        end_chat_span()
        try:
            provider = trace.get_tracer_provider() if owner == "global" else byo_provider
            assert isinstance(provider, TracerProvider)
            if call == "flush":
                provider.force_flush(2_000)
            else:
                provider.shutdown()
        except BaseException as exc:
            raised.append(exc)

    with otlp_capture_server() as ingest:
        if owner == "byo":
            # The meter provider of the processor waits for the metric reader, which holds its
            # lock during a metric export, so only the guard of the processor itself can defer
            # the call.
            ingest.path_status["/v1/metrics"] = 400
        else:
            ingest.status = 400
        stamping = None
        if owner == "global":
            monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
            monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once())
            stamping = telemetry_dev.init(
                api_key="td_test",
                base_url=ingest.url,
                register_global=True,
                log_level="silent",
                disable_atexit=True,
                timeout=1.0,
                on_error=on_error,
            ).processor
        else:
            byo_provider.add_span_processor(
                TelemetrySpanProcessor(
                    api_key="td_test",
                    base_url=ingest.url,
                    metrics=True,
                    on_error=on_error,
                )
            )
        provider = trace.get_tracer_provider() if owner == "global" else byo_provider
        assert isinstance(provider, TracerProvider)
        end_chat_span()
        flushed = threading.Event()

        def flush() -> None:
            # In the byo rows, this flush makes the metric export.
            provider.force_flush(3_000)
            flushed.set()

        threading.Thread(target=flush, daemon=True).start()
        assert flushed.wait(10)
        assert len(reports) == 1
        assert raised == []
        if call == "shutdown":
            # The deferred shutdown ran once the export ended.
            if stamping is not None:
                assert _wait_until(lambda: stamping.force_flush(100) is False)
            else:
                # It sends the metrics of the span that on_error ended.
                assert _wait_until(lambda: len(ingest.requests_for("/v1/metrics")) == 2)
        assert len(reports) == 1
        provider.shutdown()
