from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from opentelemetry.sdk.trace import ReadableSpan

import telemetry_dev
from telemetry_dev import MaskContext
from telemetry_dev._serialize import TRUNCATION_MARKER, truncate
from tests.conftest import MakeClient


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_mask_runs_before_stringify_and_receives_key(make: MakeClient) -> None:
    seen: list[tuple[Any, MaskContext]] = []

    def mask(value: Any, ctx: MaskContext) -> Any:
        seen.append((value, ctx))
        if isinstance(value, dict):
            masked = dict(cast("dict[str, Any]", value))
            masked["password"] = "[redacted]"
            return masked
        return value

    env = make(mask=mask)
    telemetry_dev.start_span("login", input={"password": "hunter2"}).end(output="ok")
    a = attrs(only_span(env))
    assert json.loads(str(a["gen_ai.input.messages"])) == {"password": "[redacted]"}
    assert "hunter2" not in str(a)
    keys = [ctx.key for _value, ctx in seen]
    assert keys == ["gen_ai.input.messages", "gen_ai.output.messages"]
    assert isinstance(seen[0][0], dict)  # pre-stringify value


def test_mask_error_fails_open(make: MakeClient) -> None:
    errors: list[BaseException] = []

    def mask(_value: Any, _ctx: MaskContext) -> Any:
        raise RuntimeError("mask broke")

    env = make(mask=mask, on_error=errors.append)
    telemetry_dev.start_span("s", input="secret").end()
    a = attrs(only_span(env))
    # A throwing mask drops the content attribute (never leaks it), matching the TS SDK.
    assert "gen_ai.input.messages" not in a
    assert errors and isinstance(errors[0], RuntimeError)


def test_mask_not_applied_to_correlation_identifiers(make: MakeClient) -> None:
    def mask(_value: Any, _ctx: MaskContext) -> Any:
        return "MASKED"

    env = make(mask=mask)
    with telemetry_dev.propagate_attributes(user_id="real-user", session_id="real-session"):
        telemetry_dev.start_span("s").end()
    a = attrs(only_span(env))
    assert a["user.id"] == "real-user"
    assert a["gen_ai.conversation.id"] == "real-session"


def test_mask_applies_to_log_message(make: MakeClient) -> None:
    def mask(value: Any, ctx: MaskContext) -> Any:
        assert ctx.key == "log.message"
        return "***"

    env = make(mask=mask)
    telemetry_dev.log("secret message")
    (record,) = env.log_exporter.get_finished_logs()
    assert record.log_record.body == "***"


def test_truncation_cap_and_marker(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("big", input="x" * 70_000).end()
    value = str(attrs(only_span(memory))["gen_ai.input.messages"])
    assert value.endswith(TRUNCATION_MARKER)
    assert len(value) == 65_536
    assert TRUNCATION_MARKER == "...[truncated]"


def test_truncation_custom_cap(make: MakeClient) -> None:
    env = make(max_attribute_length=100)
    telemetry_dev.start_span("big", input="y" * 200).end()
    value = str(attrs(only_span(env))["gen_ai.input.messages"])
    assert len(value) == 100
    assert value == "y" * (100 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER


def test_truncation_counts_utf16_units(memory: SimpleNamespace) -> None:
    # 40_000 robots are 80_000 UTF-16 units (JS String.length), so both SDKs truncate
    # at the same point even though Python sees only 40_000 code points.
    telemetry_dev.start_span("emoji", input="\U0001f916" * 40_000).end()
    value = str(attrs(only_span(memory))["gen_ai.input.messages"])
    assert value == "\U0001f916" * 32_761 + TRUNCATION_MARKER
    assert len(value.encode("utf-16-le")) // 2 == 65_536


def test_truncation_never_splits_a_surrogate_pair() -> None:
    # keep = 6 UTF-16 units lands mid-emoji; the partial pair is dropped, not emitted.
    assert truncate("a" + "\U0001f916" * 40, 20) == "a" + "\U0001f916" * 2 + TRUNCATION_MARKER


def test_unserializable_value_drops_content(make: MakeClient) -> None:
    errors: list[BaseException] = []
    env = make(on_error=errors.append)
    payload: dict[str, Any] = {}
    payload["loop"] = payload
    telemetry_dev.start_span("circular", input=payload).end()
    # Matches the TS SDK: an unserializable value drops the attribute, never a repr leak.
    assert "gen_ai.input.messages" not in attrs(only_span(env))
    assert errors and isinstance(errors[0], ValueError)


def test_init_capture_flags_off(make: MakeClient) -> None:
    env = make(capture_input=False, capture_output=False)
    telemetry_dev.start_span("gen", type="generation", input="in").end(output="out")
    a = attrs(only_span(env))
    assert "gen_ai.input.messages" not in a
    assert "gen_ai.output.messages" not in a
    assert a["gen_ai.operation.name"] == "chat"


def test_per_call_capture_overrides_init_default(make: MakeClient) -> None:
    env = make(capture_input=False)
    telemetry_dev.start_span("s", input="visible", capture_input=True).end()
    assert attrs(only_span(env))["gen_ai.input.messages"] == "visible"


def test_per_call_capture_off_overrides_init_on(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("s", input="hidden", capture_input=False).end(output="hidden-too")
    a = attrs(only_span(memory))
    assert "gen_ai.input.messages" not in a
    assert a["gen_ai.output.messages"] == "hidden-too"


def test_capture_input_gates_system_instructions(make: MakeClient) -> None:
    env = make(capture_input=False)
    telemetry_dev.start_span("gen", type="generation", system_instructions="secret").end()
    assert "gen_ai.system_instructions" not in attrs(only_span(env))
