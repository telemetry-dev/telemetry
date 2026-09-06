"""BYO OpenTelemetry processor tests."""

# Generated protobuf classes ship without type stubs; relax unknown-type rules here only.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import KeyValue
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry_dev
from telemetry_dev.otel import TelemetrySpanProcessor
from tests.otlp_capture import CapturedRequest, otlp_capture_server


def attr_map(attributes: Iterable[KeyValue]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in attributes:
        value = attr.value
        kind = value.WhichOneof("value")
        out[attr.key] = getattr(value, kind) if kind else None
    return out


def span_attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def provider_with(processor: TelemetrySpanProcessor) -> TracerProvider:
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(processor)
    return provider


def end_span(
    provider: TracerProvider,
    name: str,
    attributes: dict[str, str | int] | None = None,
) -> None:
    span = provider.get_tracer("otel-test").start_span(name)
    for key, value in (attributes or {}).items():
        span.set_attribute(key, value)
    span.end()


def end_genai_span(provider: TracerProvider) -> None:
    end_span(
        provider,
        "gen",
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 7,
        },
    )


def decoded_metrics(request: CapturedRequest) -> tuple[dict[str, Any], set[str]]:
    decoded = ExportMetricsServiceRequest()
    decoded.ParseFromString(request.decompressed())
    resource_attrs: dict[str, Any] = {}
    metric_names: set[str] = set()
    for resource_metrics in decoded.resource_metrics:
        resource_attrs.update(attr_map(resource_metrics.resource.attributes))
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                metric_names.add(metric.name)
    return resource_attrs, metric_names


def test_root_processor_matches_otel_and_accepts_positional_exporter() -> None:
    from telemetry_dev import TelemetrySpanProcessor as RootTelemetrySpanProcessor
    from telemetry_dev.otel import TelemetrySpanProcessor as OtelTelemetrySpanProcessor

    assert RootTelemetrySpanProcessor is OtelTelemetrySpanProcessor
    exporter = InMemorySpanExporter()
    processor = RootTelemetrySpanProcessor(exporter, export_mode="immediate")
    provider = provider_with(processor)
    try:
        end_span(provider, "positional-exporter")
        assert [span.name for span in exporter.get_finished_spans()] == ["positional-exporter"]
    finally:
        provider.shutdown()


def test_create_span_exporter_without_key_is_noop() -> None:
    from telemetry_dev.otel import create_telemetry_span_exporter

    with otlp_capture_server() as server:
        exporter = create_telemetry_span_exporter(api_key=None, base_url=server.url)
        processor = TelemetrySpanProcessor(
            span_exporter=exporter,
            export_mode="immediate",
        )
        provider = provider_with(processor)
        try:
            end_span(provider, "noop-exporter")
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        assert server.requests == []


def test_create_span_exporter_posts_traces_with_bearer_gzip() -> None:
    from telemetry_dev.otel import create_telemetry_span_exporter

    with otlp_capture_server() as server:
        exporter = create_telemetry_span_exporter(
            api_key="td_live_x",
            base_url=server.url,
        )
        processor = TelemetrySpanProcessor(
            span_exporter=exporter,
            export_mode="immediate",
        )
        provider = provider_with(processor)
        try:
            end_span(provider, "factory-exporter")
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        (request,) = server.requests_for("/v1/traces")
        assert request.headers["authorization"] == "Bearer td_live_x"
        assert request.headers["x-telemetry-dev-sdk"] == "telemetry-dev"
        assert request.headers["content-encoding"] == "gzip"


def test_no_op_without_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="telemetry_dev")
    processor = TelemetrySpanProcessor()
    provider = provider_with(processor)
    try:
        end_span(provider, "noop")
        assert processor.force_flush() is True
    finally:
        provider.shutdown()
    assert "no-op" in caplog.text


def test_no_posts_keyless_even_with_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_capture_server() as server:
        monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", server.url)
        processor = TelemetrySpanProcessor()
        provider = provider_with(processor)
        try:
            end_span(provider, "noop")
            assert processor.force_flush() is True
        finally:
            provider.shutdown()
        assert server.requests == []


def test_env_var_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_capture_server() as server:
        monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
        monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", f"{server.url}///")
        processor = TelemetrySpanProcessor(export_mode="immediate", metrics=False)
        provider = provider_with(processor)
        try:
            end_span(provider, "env")
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        (request,) = server.requests_for("/v1/traces")
        assert request.headers["authorization"] == "Bearer td_live_env"
        assert request.headers["content-encoding"] == "gzip"


def test_kwargs_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_capture_server() as server:
        monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
        monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", f"{server.url}/env")
        processor = TelemetrySpanProcessor(
            api_key="td_live_kwarg",
            base_url=server.url,
            export_mode="immediate",
            metrics=False,
        )
        provider = provider_with(processor)
        try:
            end_span(provider, "kwarg")
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        (request,) = server.requests_for("/v1/traces")
        assert request.headers["authorization"] == "Bearer td_live_kwarg"


def test_span_exporter_seam_activates_without_key_metrics_stay_off() -> None:
    exporter = InMemorySpanExporter()
    with otlp_capture_server() as server:
        processor = TelemetrySpanProcessor(
            base_url=server.url,
            export_mode="immediate",
            metrics=True,
            span_exporter=exporter,
        )
        provider = provider_with(processor)
        try:
            end_genai_span(provider)
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

    assert [span.name for span in exporter.get_finished_spans()] == ["gen"]
    assert server.requests == []


def test_span_filter_narrows() -> None:
    exporter = InMemorySpanExporter()

    def keep(span: ReadableSpan) -> bool:
        return span.name.startswith("keep")

    processor = TelemetrySpanProcessor(
        export_mode="immediate",
        span_exporter=exporter,
        span_filter=keep,
    )
    provider = provider_with(processor)
    try:
        end_span(provider, "keep-me")
        end_span(provider, "drop-me")
        assert [span.name for span in exporter.get_finished_spans()] == ["keep-me"]
    finally:
        provider.shutdown()


def test_raising_span_filter_is_fail_open() -> None:
    exporter = InMemorySpanExporter()
    errors: list[BaseException] = []

    def boom(_span: ReadableSpan) -> bool:
        raise RuntimeError("boom")

    processor = TelemetrySpanProcessor(
        export_mode="immediate",
        span_exporter=exporter,
        span_filter=boom,
        on_error=errors.append,
    )
    provider = provider_with(processor)
    try:
        end_span(provider, "exported")
        assert [span.name for span in exporter.get_finished_spans()] == ["exported"]
        assert errors and isinstance(errors[0], RuntimeError)
    finally:
        provider.shutdown()


def test_span_exporter_shutdown_is_fail_open() -> None:
    shutdown_error = RuntimeError("span shutdown failed")

    class ShutdownExporter(InMemorySpanExporter):
        def shutdown(self) -> None:
            raise shutdown_error

    errors: list[BaseException] = []
    processor = TelemetrySpanProcessor(
        export_mode="immediate",
        span_exporter=ShutdownExporter(),
        on_error=errors.append,
    )

    processor.shutdown()

    assert errors == [shutdown_error]


def test_metric_provider_lifecycle_is_fail_open() -> None:
    force_flush_error = RuntimeError("metric force_flush failed")
    shutdown_error = RuntimeError("metric shutdown failed")

    class FailingMeterProvider:
        def force_flush(self, *, timeout_millis: int) -> bool:
            raise force_flush_error

        def shutdown(self, *, timeout_millis: int) -> None:
            raise shutdown_error

    errors: list[BaseException] = []
    processor = TelemetrySpanProcessor(
        export_mode="immediate",
        span_exporter=InMemorySpanExporter(),
        on_error=errors.append,
    )
    processor._meter_provider = FailingMeterProvider()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    assert processor.force_flush() is False
    processor.shutdown()

    assert errors == [force_flush_error, shutdown_error]


def test_propagate_attributes_are_stamped() -> None:
    exporter = InMemorySpanExporter()
    processor = TelemetrySpanProcessor(export_mode="immediate", span_exporter=exporter)
    provider = provider_with(processor)
    try:
        with telemetry_dev.propagate_attributes(user_id="u_byo", session_id="conv_byo"):
            end_span(provider, "propagated")
        (span,) = exporter.get_finished_spans()
        attributes = span_attrs(span)
        assert attributes["user.id"] == "u_byo"
        assert attributes["gen_ai.conversation.id"] == "conv_byo"
    finally:
        provider.shutdown()


def test_immediate_vs_batched_export() -> None:
    immediate_exporter = InMemorySpanExporter()
    immediate_processor = TelemetrySpanProcessor(
        export_mode="immediate",
        span_exporter=immediate_exporter,
    )
    immediate_provider = provider_with(immediate_processor)
    try:
        end_span(immediate_provider, "immediate")
        assert [span.name for span in immediate_exporter.get_finished_spans()] == ["immediate"]
    finally:
        immediate_provider.shutdown()

    batched_exporter = InMemorySpanExporter()
    batched_processor = TelemetrySpanProcessor(span_exporter=batched_exporter)
    batched_provider = provider_with(batched_processor)
    try:
        end_span(batched_provider, "batched")
        assert batched_exporter.get_finished_spans() == ()
        assert batched_processor.force_flush() is True
        assert [span.name for span in batched_exporter.get_finished_spans()] == ["batched"]
    finally:
        batched_provider.shutdown()


def test_metrics_default_wire_export() -> None:
    with otlp_capture_server() as server:
        processor = TelemetrySpanProcessor(
            api_key="td_live_x",
            base_url=server.url,
            export_mode="immediate",
        )
        provider = provider_with(processor)
        try:
            end_genai_span(provider)
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        assert server.requests_for("/v1/traces")
        (request,) = server.requests_for("/v1/metrics")
        assert request.headers["authorization"] == "Bearer td_live_x"
        assert request.headers["content-encoding"] == "gzip"
        resource_attrs, metric_names = decoded_metrics(request)
        assert resource_attrs["service.name"] == "unknown_service"
        assert resource_attrs["deployment.environment.name"] == "production"
        assert metric_names >= {
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        }


def test_metrics_resource_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_capture_server() as server:
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-svc")
        monkeypatch.setenv("TELEMETRY_DEV_ENVIRONMENT", "staging")
        processor = TelemetrySpanProcessor(
            api_key="td_live_x",
            base_url=server.url,
            export_mode="immediate",
        )
        provider = provider_with(processor)
        try:
            end_genai_span(provider)
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        (request,) = server.requests_for("/v1/metrics")
        resource_attrs, _metric_names = decoded_metrics(request)
        assert resource_attrs["service.name"] == "env-svc"
        assert resource_attrs["deployment.environment.name"] == "staging"


def test_metrics_resource_kwargs_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with otlp_capture_server() as server:
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-svc")
        monkeypatch.setenv("TELEMETRY_DEV_ENVIRONMENT", "staging")
        processor = TelemetrySpanProcessor(
            api_key="td_live_x",
            base_url=server.url,
            export_mode="immediate",
            service_name="kwarg-svc",
            environment="prodlike",
        )
        provider = provider_with(processor)
        try:
            end_genai_span(provider)
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        (request,) = server.requests_for("/v1/metrics")
        resource_attrs, _metric_names = decoded_metrics(request)
        assert resource_attrs["service.name"] == "kwarg-svc"
        assert resource_attrs["deployment.environment.name"] == "prodlike"


def test_metrics_off_skips_metrics_post() -> None:
    with otlp_capture_server() as server:
        processor = TelemetrySpanProcessor(
            api_key="td_live_x",
            base_url=server.url,
            export_mode="immediate",
            metrics=False,
        )
        provider = provider_with(processor)
        try:
            end_genai_span(provider)
            assert processor.force_flush() is True
        finally:
            provider.shutdown()

        assert server.requests_for("/v1/traces")
        assert server.requests_for("/v1/metrics") == []
