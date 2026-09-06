"""Wire-level tests: real OTLP HTTP exporters against a local capture server, asserting the
exact request shape the telemetry.dev ingest expects (Bearer auth, protobuf, gzip)."""

# Generated protobuf classes ship without type stubs; relax unknown-type rules here only.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import Any

import pytest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import KeyValue

import telemetry_dev
from tests.otlp_capture import CaptureServer, otlp_capture_server


@pytest.fixture
def server() -> Generator[CaptureServer]:
    with otlp_capture_server() as capture:
        telemetry_dev.init(
            api_key="td_live_wire",
            base_url=capture.url,
            service_name="wire-svc",
            environment="wiretest",
            export_mode="immediate",
            log_level="silent",
            disable_atexit=True,
        )
        yield capture
        telemetry_dev.shutdown()


def attr_map(attributes: Iterable[KeyValue]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in attributes:
        value = attr.value
        kind = value.WhichOneof("value")
        out[attr.key] = getattr(value, kind) if kind else None
    return out


def test_trace_export_wire_format(server: CaptureServer) -> None:
    telemetry_dev.start_span(
        "gen", type="generation", model="gpt-4o", provider="openai", input="hi"
    ).end(output="hello", usage={"input_tokens": 11, "output_tokens": 7})
    telemetry_dev.flush()

    (request,) = server.requests_for("/v1/traces")
    assert request.headers["authorization"] == "Bearer td_live_wire"
    assert request.headers["content-type"] == "application/x-protobuf"
    assert request.headers["content-encoding"] == "gzip"

    decoded = ExportTraceServiceRequest()
    decoded.ParseFromString(request.decompressed())
    (resource_spans,) = decoded.resource_spans
    resource_attrs = attr_map(resource_spans.resource.attributes)
    assert resource_attrs["service.name"] == "wire-svc"
    assert resource_attrs["deployment.environment.name"] == "wiretest"

    (scope_spans,) = resource_spans.scope_spans
    assert scope_spans.scope.name == "telemetry_dev"
    assert scope_spans.scope.version == telemetry_dev.__version__

    (span,) = scope_spans.spans
    assert span.name == "gen"
    span_attrs = attr_map(span.attributes)
    assert span_attrs["gen_ai.operation.name"] == "chat"
    assert span_attrs["gen_ai.request.model"] == "gpt-4o"
    assert span_attrs["gen_ai.provider.name"] == "openai"
    assert span_attrs["gen_ai.input.messages"] == "hi"
    assert span_attrs["gen_ai.output.messages"] == "hello"
    assert span_attrs["gen_ai.usage.input_tokens"] == 11
    assert span_attrs["gen_ai.usage.output_tokens"] == 7


def test_log_export_wire_format(server: CaptureServer) -> None:
    with telemetry_dev.start_span("parent") as handle:
        telemetry_dev.log("wired", level="warn", event_name="wire.test")
    telemetry_dev.flush()

    (request,) = server.requests_for("/v1/logs")
    assert request.headers["authorization"] == "Bearer td_live_wire"
    assert request.headers["content-type"] == "application/x-protobuf"
    assert request.headers["content-encoding"] == "gzip"

    decoded = ExportLogsServiceRequest()
    decoded.ParseFromString(request.decompressed())
    (resource_logs,) = decoded.resource_logs
    resource_attrs = attr_map(resource_logs.resource.attributes)
    assert resource_attrs["deployment.environment.name"] == "wiretest"
    (scope_logs,) = resource_logs.scope_logs
    assert scope_logs.scope.name == "telemetry_dev"
    (record,) = scope_logs.log_records
    assert record.body.string_value == "wired"
    assert record.severity_number == 13
    assert record.event_name == "wire.test"
    span_context = handle.span.get_span_context()
    assert record.trace_id == span_context.trace_id.to_bytes(16, "big")
    assert record.span_id == span_context.span_id.to_bytes(8, "big")


def test_no_posts_when_disabled() -> None:
    with otlp_capture_server() as capture:
        telemetry_dev.init(
            base_url=capture.url,
            log_level="silent",
            disable_atexit=True,
        )
        telemetry_dev.start_span("nope", type="generation", input="x").end()
        telemetry_dev.log("nope")
        telemetry_dev.flush()
        telemetry_dev.shutdown()
        assert capture.requests == []
