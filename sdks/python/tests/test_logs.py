from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry.sdk._logs import ReadableLogRecord

import telemetry_dev
from telemetry_dev._serialize import TRUNCATION_MARKER
from tests.conftest import MakeClient


def only_log(env: SimpleNamespace) -> ReadableLogRecord:
    records = env.log_exporter.get_finished_logs()
    assert len(records) == 1
    return records[0]


@pytest.mark.parametrize(
    ("level", "severity_number", "severity_text"),
    [
        ("debug", 5, "DEBUG"),
        ("info", 9, "INFO"),
        ("warn", 13, "WARN"),
        ("warning", 13, "WARN"),  # alias, normalized on the wire
        ("error", 17, "ERROR"),
    ],
)
def test_severity_mapping(
    memory: SimpleNamespace, level: str, severity_number: int, severity_text: str
) -> None:
    telemetry_dev.log("msg", level=level)  # type: ignore[arg-type]
    record = only_log(memory).log_record
    assert record.severity_number is not None
    assert record.severity_number.value == severity_number
    assert record.severity_text == severity_text


def test_default_level_is_info(memory: SimpleNamespace) -> None:
    telemetry_dev.log("msg")
    record = only_log(memory).log_record
    assert record.severity_number is not None
    assert record.severity_number.value == 9


def test_body_is_message(memory: SimpleNamespace) -> None:
    telemetry_dev.log("hello world")
    assert only_log(memory).log_record.body == "hello world"


def test_trace_correlation_inside_span(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("parent") as handle:
        telemetry_dev.log("inside")
    record = only_log(memory).log_record
    span_context = handle.span.get_span_context()
    assert record.trace_id == span_context.trace_id
    assert record.span_id == span_context.span_id


def test_no_trace_correlation_outside_span(memory: SimpleNamespace) -> None:
    telemetry_dev.log("outside")
    record = only_log(memory).log_record
    assert not record.trace_id
    assert not record.span_id


def test_event_name(memory: SimpleNamespace) -> None:
    telemetry_dev.log("msg", event_name="cache.miss")
    assert only_log(memory).log_record.event_name == "cache.miss"


def test_custom_attributes(memory: SimpleNamespace) -> None:
    telemetry_dev.log("msg", attributes={"request.id": "r-1", "retry": 2})
    log_attrs = dict(only_log(memory).log_record.attributes or {})
    assert log_attrs["request.id"] == "r-1"
    assert log_attrs["retry"] == 2


def test_bad_attribute_does_not_drop_log_record(memory: SimpleNamespace) -> None:
    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError("broken repr")

    telemetry_dev.log(
        "msg",
        attributes={"bad": BrokenRepr(), "ok": "v"},  # type: ignore[arg-type]
    )
    log_attrs = dict(only_log(memory).log_record.attributes or {})
    assert log_attrs["ok"] == "v"
    assert "bad" not in log_attrs


def test_propagated_attributes_on_records(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(
        user_id="u1", session_id="s1", metadata={"plan": "pro"}
    ):
        telemetry_dev.log("msg")
    log_attrs = dict(only_log(memory).log_record.attributes or {})
    assert log_attrs["user.id"] == "u1"
    assert log_attrs["gen_ai.conversation.id"] == "s1"
    assert log_attrs["td.metadata.plan"] == "pro"


def test_propagated_attributes_use_client_attribute_cap(make: MakeClient) -> None:
    env = make(max_attribute_length=30)
    with telemetry_dev.propagate_attributes(metadata={"blob": "z" * 100}):
        telemetry_dev.log("msg")
    log_attrs = dict(only_log(env).log_record.attributes or {})
    value = str(log_attrs["td.metadata.blob"])
    assert value == "z" * (30 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER


def test_log_records_use_sdk_scope(memory: SimpleNamespace) -> None:
    telemetry_dev.log("msg")
    scope = only_log(memory).instrumentation_scope
    assert scope is not None and scope.name == "telemetry_dev"


def test_log_is_noop_without_init() -> None:
    telemetry_dev.log("nowhere")


def test_no_span_event_for_log_inside_span(memory: SimpleNamespace) -> None:
    # ALL log() calls are OTLP log records; none become span events.
    with telemetry_dev.start_span("parent"):
        telemetry_dev.log("inside")
    (span,) = memory.span_exporter.get_finished_spans()
    assert list(span.events) == []
    assert len(memory.log_exporter.get_finished_logs()) == 1
