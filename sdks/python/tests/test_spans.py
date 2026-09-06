from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import telemetry_dev
from tests.conftest import span_context


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


@pytest.mark.parametrize(
    ("span_type", "operation"),
    [
        ("span", "function"),
        ("generation", "chat"),
        ("tool", "execute_tool"),
        ("agent", "invoke_agent"),
        ("embedding", "embeddings"),
    ],
)
def test_type_to_operation_mapping(memory: SimpleNamespace, span_type: str, operation: str) -> None:
    telemetry_dev.start_span("s", type=span_type).end()  # type: ignore[arg-type]
    assert attrs(only_span(memory))["gen_ai.operation.name"] == operation


def test_plain_span_always_function_even_with_propagated_session(
    memory: SimpleNamespace,
) -> None:
    with telemetry_dev.propagate_attributes(session_id="sess"):
        telemetry_dev.start_span("plain").end()
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "function"
    assert a["gen_ai.conversation.id"] == "sess"


def test_plain_span_io_uses_messages_keys(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("plain", input={"q": 1}).end(output={"a": 2})
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == {"q": 1}
    assert json.loads(str(a["gen_ai.output.messages"])) == {"a": 2}


def test_tool_span_io_and_name(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span(
        "lookup",
        type="tool",
        input={"city": "Berlin"},
        tool_call_id="call_1",
        tool_description="Weather lookup",
    ).end(output={"temp": 21})
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert a["gen_ai.tool.name"] == "lookup"
    assert a["gen_ai.tool.call.id"] == "call_1"
    assert a["gen_ai.tool.description"] == "Weather lookup"
    assert json.loads(str(a["gen_ai.tool.call.arguments"])) == {"city": "Berlin"}
    assert json.loads(str(a["gen_ai.tool.call.result"])) == {"temp": 21}


def test_tool_name_field_overrides_span_name(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("span-name", type="tool", tool_name="actual_tool").end()
    assert attrs(only_span(memory))["gen_ai.tool.name"] == "actual_tool"


def test_agent_span_defaults_agent_name(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("planner", type="agent", agent_id="agent-7").end()
    a = attrs(only_span(memory))
    assert a["gen_ai.agent.name"] == "planner"
    assert a["gen_ai.agent.id"] == "agent-7"


def test_full_field_mapping(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span(
        "gen",
        type="generation",
        input=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
        provider="openai",
        system_instructions="be nice",
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        max_tokens=256,
        stop_sequences=["END"],
        seed=7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        metadata={"plan": "pro", "userId": "dropped"},
        attributes={"custom.key": "v"},
    ).end(
        output="hello",
        response_model="gpt-4o-2024-08-06",
        response_id="resp_1",
        output_type="text",
        finish_reason="stop",
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1,
            "reasoning_output_tokens": 3,
        },
        cost_usd=0.0123,
        time_to_first_chunk_ms=250,
    )
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.system_instructions"] == "be nice"
    assert a["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert a["gen_ai.response.id"] == "resp_1"
    assert a["gen_ai.output.type"] == "text"
    assert list(a["gen_ai.response.finish_reasons"]) == ["stop"]  # type: ignore[call-overload]
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.usage.total_tokens"] == 18
    assert a["gen_ai.usage.cache_read.input_tokens"] == 2
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 1
    assert a["gen_ai.usage.reasoning.output_tokens"] == 3
    assert a["gen_ai.usage.cost"] == 0.0123
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["gen_ai.request.top_k"] == 40
    assert a["gen_ai.request.max_tokens"] == 256
    assert list(a["gen_ai.request.stop_sequences"]) == ["END"]  # type: ignore[call-overload]
    assert a["gen_ai.request.seed"] == 7
    assert a["gen_ai.request.frequency_penalty"] == 0.1
    assert a["gen_ai.request.presence_penalty"] == 0.2
    # Emitted in seconds; the ingest converts non-"ms" duration keys back to ms.
    assert a["gen_ai.response.time_to_first_chunk"] == 0.25
    assert a["td.metadata.plan"] == "pro"
    assert "td.metadata.userId" not in a
    assert a["custom.key"] == "v"
    assert json.loads(str(a["gen_ai.input.messages"])) == [{"role": "user", "content": "hi"}]
    assert a["gen_ai.output.messages"] == "hello"


def test_usage_unknown_keys_dropped(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span(
        "gen", type="generation", usage={"input_tokens": 1, "bogus_tokens": 9}
    ).end()
    a = attrs(only_span(memory))
    assert a["gen_ai.usage.input_tokens"] == 1
    assert not any("bogus" in key for key in a)


def test_bad_raw_attribute_is_skipped_without_dropping_span(memory: SimpleNamespace) -> None:
    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError("broken repr")

    telemetry_dev.start_span(
        "safe",
        attributes={"bad": BrokenRepr(), "ok": "v"},  # type: ignore[arg-type]
    ).end()
    a = attrs(only_span(memory))
    assert a["ok"] == "v"
    assert "bad" not in a


def test_parent_traceparent_join(memory: SimpleNamespace) -> None:
    parent = telemetry_dev.start_span("parent")
    traceparent = parent.traceparent()
    assert traceparent is not None and traceparent.startswith("00-")
    child = telemetry_dev.start_span("child", parent=traceparent)
    child.end()
    parent.end()
    child_span, parent_span = memory.span_exporter.get_finished_spans()
    assert child_span.name == "child"
    assert span_context(child_span).trace_id == span_context(parent_span).trace_id
    assert child_span.parent is not None
    assert child_span.parent.span_id == span_context(parent_span).span_id


def test_parent_span_context_join(memory: SimpleNamespace) -> None:
    parent = telemetry_dev.start_span("parent")
    child = telemetry_dev.start_span("child", parent=parent.span.get_span_context())
    child.end()
    parent.end()
    child_span, parent_span = memory.span_exporter.get_finished_spans()
    assert span_context(child_span).trace_id == span_context(parent_span).trace_id


def test_with_activates_context_detached_does_not(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("active") as active:
        assert trace.get_current_span() is active.span
        detached = telemetry_dev.start_span("detached")
        assert trace.get_current_span() is active.span
        detached.end()
    assert not trace.get_current_span().get_span_context().is_valid
    detached_span, active_span = memory.span_exporter.get_finished_spans()
    # The detached span still parents to the active context.
    assert detached_span.parent is not None
    assert detached_span.parent.span_id == span_context(active_span).span_id


def test_exception_in_with_block_records_error_and_reraises(memory: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="bad"):
        with telemetry_dev.start_span("failing"):
            raise ValueError("bad")
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "ValueError"
    events = list(span.events)
    assert len(events) == 1
    event = events[0]
    assert event.name == "exception"
    event_attrs = dict(event.attributes or {})
    assert event_attrs["exception.type"] == "ValueError"
    assert event_attrs["exception.message"] == "bad"
    assert "ValueError: bad" in str(event_attrs["exception.stacktrace"])
    assert event_attrs["log.severity_number"] == 17


def test_end_with_explicit_error(memory: SimpleNamespace) -> None:
    handle = telemetry_dev.start_span("manual")
    handle.end(error=RuntimeError("kaput"))
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"


def test_start_and_end_time(memory: SimpleNamespace) -> None:
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
    telemetry_dev.start_span("timed", start_time=start).end(end_time=end)
    span = only_span(memory)
    assert span.start_time == int(start.timestamp() * 1e9)
    assert span.end_time == int(end.timestamp() * 1e9)
    assert span.end_time is not None and span.start_time is not None
    assert span.end_time - span.start_time == 5_000_000_000


def test_update_name(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("old").end(name="new")
    assert only_span(memory).name == "new"


def test_update_current_span_inside_with(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("gen", type="generation"):
        telemetry_dev.update_current_span(model="gpt-4o", usage={"input_tokens": 5}, output="done")
    a = attrs(only_span(memory))
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.output.messages"] == "done"


def test_update_current_span_uses_span_operation_for_io_keys(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("tool", type="tool"):
        telemetry_dev.update_current_span(output={"ok": True})
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.tool.call.result"])) == {"ok": True}
    assert "gen_ai.output.messages" not in a


def test_update_current_span_noop_without_active_span(memory: SimpleNamespace) -> None:
    telemetry_dev.update_current_span(output="nowhere")
    assert memory.span_exporter.get_finished_spans() == ()


def test_double_end_is_safe(memory: SimpleNamespace) -> None:
    handle = telemetry_dev.start_span("once")
    handle.end()
    handle.end()
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_end_inside_with_block_not_ended_twice(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("early") as handle:
        handle.end(output="done")
    assert len(memory.span_exporter.get_finished_spans()) == 1
