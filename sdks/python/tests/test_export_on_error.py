from __future__ import annotations

# pyright: reportPrivateUsage=false
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
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
from opentelemetry.util.types import AttributeValue

import telemetry_dev
from telemetry_dev._export import (
    ExportFailures,
    ReportingLogExporter,
    ReportingSpanExporter,
    _span_with_attributes,
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


@pytest.mark.parametrize("signal", ["span", "log"])
def test_a_record_over_the_ingest_limit_is_sent_with_its_longest_attribute_cut(
    signal: str,
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
        if signal == "span":
            telemetry_dev.start_span("oversized", attributes=attributes).end()
        else:
            telemetry_dev.log("oversized", attributes=attributes)
        telemetry_dev.shutdown()

        requests = ingest.requests_for("/v1/traces" if signal == "span" else "/v1/logs")
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
