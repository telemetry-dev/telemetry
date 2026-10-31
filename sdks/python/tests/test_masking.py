from __future__ import annotations

import enum
import json
import logging
import traceback
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

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


def _rewrite(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("secret", "[masked]")
    if isinstance(value, list):
        return [_rewrite(item) for item in cast("list[Any]", value)]
    if isinstance(value, dict):
        return {key: _rewrite(item) for key, item in cast("dict[str, Any]", value).items()}
    return value


def test_serialize_content_masks_serializes_and_truncates(make: MakeClient) -> None:
    keys: list[str] = []

    def mask(value: Any, ctx: MaskContext) -> Any:
        keys.append(ctx.key)
        return _rewrite(value)

    make(mask=mask, max_attribute_length=40)
    span = telemetry_dev.start_span("integration", capture_input=False, capture_output=False)

    assert (
        span.serialize_content("tool failed: secret path", "exception.message")
        == "tool failed: [masked] path"
    )
    grounding = span.serialize_content({"chunks": [{"text": "secret doc"}]}, "grounding")
    assert grounding is not None
    assert json.loads(grounding) == {"chunks": [{"text": "[masked] doc"}]}
    assert span.serialize_content("secret " + "x" * 60, "long") == (
        "[masked] " + "x" * 17 + "...[truncated]"
    )
    assert keys == ["exception.message", "grounding", "long"]
    span.end()


def test_serialize_content_withholds_content_the_mask_cannot_handle(make: MakeClient) -> None:
    assert telemetry_dev.start_span("before-init").serialize_content("secret", "key") is None

    errors: list[BaseException] = []

    def mask(value: Any, ctx: MaskContext) -> Any:
        if ctx.key == "broken":
            raise RuntimeError("mask broke")
        return value

    make(mask=mask, on_error=errors.append, max_attribute_length=40)
    span = telemetry_dev.start_span("integration")
    circular: dict[str, Any] = {"text": "secret"}
    circular["self"] = circular

    assert span.serialize_content("secret", "broken") is None
    assert span.serialize_content(circular, "circular") is None
    assert span.serialize_content("\udcff" + "x" * 60, "lone-surrogate") is None
    assert [type(error) for error in errors] == [RuntimeError, ValueError, ValueError]
    assert all(error.__context__ is None and error.__cause__ is None for error in errors)
    span.end()


def test_content_is_whole_by_default_and_marked_at_a_custom_cap(make: MakeClient) -> None:
    uncapped = make()
    telemetry_dev.start_span("big", input="x" * 70_000).end()
    assert attrs(only_span(uncapped))["gen_ai.input.messages"] == "x" * 70_000

    capped = make(max_attribute_length=100)
    telemetry_dev.start_span("small-cap", input="y" * 500).end()
    value = str(attrs(only_span(capped))["gen_ai.input.messages"])
    assert value == "y" * (100 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER
    assert TRUNCATION_MARKER == "...[truncated]"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "null"),
        (
            {"score": float("inf"), "items": [float("-inf"), "NaN"]},
            '{"score":null,"items":[null,"NaN"]}',
        ),
        (
            {"text": 'é東🤖\n\x01"\\\u2028\u2029', "ok": True, "missing": None},
            '{"text":"é東🤖\\n\\u0001\\"\\\\\u2028\u2029","ok":true,"missing":null}',
        ),
        ({"values": [1] * 30_000}, '{"values":[' + ",".join(["1"] * 30_000) + "]}"),
        (
            [1.0, -0.0, 1e-7, 1e-6, 1e20, 1e21, 9007199254740993],
            "[1,0,1e-7,0.000001,100000000000000000000,1e+21,9007199254740992]",
        ),
        (
            {"z": 0, "10": 10, "2": 2, "01": 1, "4294967295": 5, "a": {"3": 3, "0": 0}},
            '{"2":2,"10":10,"z":0,"01":1,"4294967295":5,"a":{"0":0,"3":3}}',
        ),
        (["\ud800", "\udc00", "\ud83e\udd16"], '["\\ud800","\\udc00","🤖"]'),
        ({"\ud83e\udd16": 1, "after": 2, "🤖": 3}, '{"🤖":3,"after":2}'),
        ({float("inf"): 1, float("nan"): 2}, '{"Infinity":1,"NaN":2}'),
    ],
    ids=[
        "nan",
        "nested-infinities",
        "unicode-escapes",
        "compact-cap",
        "numbers",
        "keys",
        "surrogates",
        "equivalent-surrogate-keys",
        "nonfinite-keys",
    ],
)
def test_captured_json_matches_shared_contract(make: MakeClient, value: Any, expected: str) -> None:
    env = make()
    telemetry_dev.start_span("json", input=value).end()
    assert attrs(only_span(env))["gen_ai.input.messages"] == expected


class _LabelledInt(int):
    def __str__(self) -> str:
        return "label"


class _Priority(enum.IntEnum):
    HIGH = 3


def test_captured_json_ignores_int_subclass_str(make: MakeClient) -> None:
    # Python 3.10 IntEnum.__str__ returns "Priority.HIGH"; any int subclass may override __str__.
    env = make()
    value = {"count": _LabelledInt(7), "items": [_Priority.HIGH], _LabelledInt(2): "key"}
    telemetry_dev.start_span("json", input=value).end()
    captured = attrs(only_span(env))["gen_ai.input.messages"]
    assert captured == '{"2":"key","count":7,"items":[3]}'
    assert json.loads(str(captured)) == {"2": "key", "count": 7, "items": [3]}


def test_masked_nonfinite_values_and_shared_references(make: MakeClient) -> None:
    shared = [float("nan"), 2]
    value = {"first": shared, "second": shared}

    def mask(_value: Any, _context: MaskContext) -> Any:
        return value

    env = make(mask=mask)
    telemetry_dev.start_span("masked", input="secret").end()
    assert attrs(only_span(env))["gen_ai.input.messages"] == (
        '{"first":[null,2],"second":[null,2]}'
    )
    assert value["first"] is shared and value["second"] is shared
    assert shared[0] != shared[0]


def test_truncation_custom_cap(make: MakeClient) -> None:
    env = make(max_attribute_length=100)
    telemetry_dev.start_span("big", input="y" * 200).end()
    value = str(attrs(only_span(env))["gen_ai.input.messages"])
    assert len(value) == 100
    assert value == "y" * (100 - len(TRUNCATION_MARKER)) + TRUNCATION_MARKER


def test_truncation_counts_utf16_units(make: MakeClient) -> None:
    # 40_000 robots are 80_000 UTF-16 units (JS String.length), so both SDKs truncate
    # at the same point even though Python sees only 40_000 code points.
    memory = make(max_attribute_length=65_536)
    telemetry_dev.start_span("emoji", input="\U0001f916" * 40_000).end()
    value = str(attrs(only_span(memory))["gen_ai.input.messages"])
    assert value == "\U0001f916" * 32_761 + TRUNCATION_MARKER
    assert len(value.encode("utf-16-le")) // 2 == 65_536


def test_truncation_never_splits_a_surrogate_pair() -> None:
    # keep = 6 UTF-16 units lands mid-emoji; the partial pair is dropped, not emitted.
    assert truncate("a" + "\U0001f916" * 40, 20) == "a" + "\U0001f916" * 2 + TRUNCATION_MARKER
    assert truncate("\U0001f916" * 40 + "a", 20, keep_end=True) == (
        TRUNCATION_MARKER + "\U0001f916" * 2 + "a"
    )
    marker_len = len(TRUNCATION_MARKER)
    assert truncate("a" * 30, marker_len, keep_end=True) == TRUNCATION_MARKER
    assert truncate("x" * 20 + "\U0001f916", marker_len + 1, keep_end=True) == TRUNCATION_MARKER


def test_truncated_stacktrace_keeps_the_innermost_frame(make: MakeClient) -> None:
    env = make(max_attribute_length=300)

    def fail(depth: int) -> None:
        if depth == 0:
            raise ValueError("innermost failure")
        fail(depth - 1)

    with pytest.raises(ValueError), telemetry_dev.start_span("deep"):
        fail(30)
    event_attrs = dict(only_span(env).events[0].attributes or {})
    stack = str(event_attrs["exception.stacktrace"])
    assert len(stack) == 300
    assert stack.startswith(TRUNCATION_MARKER)
    assert stack.endswith("ValueError: innermost failure\n")
    assert 'raise ValueError("innermost failure")' in stack
    assert event_attrs["exception.message"] == "innermost failure"


@pytest.mark.parametrize(
    ("max_len", "expected"),
    [
        (0, ""),
        (len(TRUNCATION_MARKER) - 1, TRUNCATION_MARKER[:-1]),
        (len(TRUNCATION_MARKER), TRUNCATION_MARKER),
    ],
)
def test_truncation_marker_stays_within_small_limits(max_len: int, expected: str) -> None:
    assert truncate("x" * (len(TRUNCATION_MARKER) + 1), max_len) == expected


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


@pytest.mark.parametrize(
    "capture_input,capture_output", [(False, True), (True, False), (False, False)]
)
async def test_exception_capture_flags_and_overrides(
    make: MakeClient, capture_input: bool, capture_output: bool
) -> None:
    env = make(capture_input=capture_input, capture_output=capture_output)
    error = ValueError("PRIVATE_EXCEPTION")
    telemetry_dev.start_span("ended").end(error=error)
    with telemetry_dev.start_span("active"):
        telemetry_dev.update_current_span(error=error)

    @telemetry_dev.observe()
    async def failed() -> None:
        raise error

    with pytest.raises(ValueError) as raised:
        await failed()
    assert raised.value is error
    telemetry_dev.start_span("override", capture_input=True, capture_output=True).end(error=error)
    exported = env.span_exporter.get_finished_spans()
    assert len(exported) == 4
    for span in exported:
        assert span.status.status_code == StatusCode.ERROR
        assert attrs(span)["error.type"] == "ValueError"
        event_attrs = dict(span.events[0].attributes or {})
        assert event_attrs["exception.type"] == "ValueError"
        if span.name == "override":
            assert event_attrs["exception.message"] == "PRIVATE_EXCEPTION"
        else:
            assert "exception.message" not in event_attrs
            assert "exception.stacktrace" not in event_attrs
            assert "PRIVATE_EXCEPTION" not in str(event_attrs)


@pytest.mark.parametrize("mode", ["redact", "throw", "truncate"])
def test_exception_masking(make: MakeClient, mode: str) -> None:
    def mask(value: Any, ctx: MaskContext) -> Any:
        if mode == "throw":
            raise RuntimeError("mask failed")
        if mode == "truncate":
            return "x" * 100
        if ctx.key == "exception.message":
            return "safe message"
        if ctx.key == "exception.stacktrace":
            return "safe stack"
        return value

    env = make(mask=mask, max_attribute_length=50)
    error = ValueError("PRIVATE_EXCEPTION")
    with pytest.raises(ValueError) as raised, telemetry_dev.start_span("failure"):
        raise error
    assert raised.value is error
    telemetry_dev.start_span("private", capture_output=False).end(
        error=ValueError("PRIVATE_EXCEPTION")
    )
    span, private_span = env.span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "ValueError"
    event_attrs = dict(span.events[0].attributes or {})
    assert event_attrs.get("exception.message") == (
        "safe message"
        if mode == "redact"
        else "x" * 36 + TRUNCATION_MARKER
        if mode == "truncate"
        else None
    )
    assert event_attrs.get("exception.stacktrace") == (
        "safe stack"
        if mode == "redact"
        else TRUNCATION_MARKER + "x" * 36
        if mode == "truncate"
        else None
    )
    private_attrs = dict(private_span.events[0].attributes or {})
    assert "exception.message" not in private_attrs
    assert "PRIVATE_EXCEPTION" not in str([event_attrs, private_attrs])


@pytest.mark.parametrize("callback_mode", ["return", "raise", "reraise"])
@pytest.mark.parametrize("failure_mode", ["mask", "serialize", "metadata", "truncate"])
def test_mask_failure_diagnostics_do_not_expose_content(
    make: MakeClient, caplog: pytest.LogCaptureFixture, callback_mode: str, failure_mode: str
) -> None:
    errors: list[str] = []
    chains: list[tuple[Any, Any, Any]] = []
    original = ValueError("PRIVATE_APPLICATION")

    class Unserializable:
        def __repr__(self) -> str:
            raise ValueError("PRIVATE_SERIALIZATION") from original

    def mask(_value: Any, _ctx: MaskContext) -> Any:
        if failure_mode == "serialize":
            return Unserializable()
        if failure_mode == "truncate":
            return "\ud800" * 40
        raise RuntimeError("PRIVATE_MASK") from original

    def on_error(error: BaseException) -> None:
        errors.append("".join(traceback.format_exception(error)))
        chains.append((error.__context__, error.__cause__, error.__traceback__))
        if callback_mode == "reraise":
            raise error
        if callback_mode == "raise":
            raise RuntimeError("PRIVATE_CALLBACK") from original

    env = make(mask=mask, on_error=on_error, log_level="debug", max_attribute_length=50)
    with caplog.at_level(logging.DEBUG, logger="telemetry_dev"):
        with pytest.raises(ValueError) as raised, telemetry_dev.start_span("failure"):
            try:
                raise original
            except ValueError:
                if failure_mode == "metadata":
                    telemetry_dev.update_current_span(metadata={"private": Unserializable()})
                raise
    assert raised.value is original
    assert errors
    assert all(chain == (None, None, None) for chain in chains)
    assert "content dropped" in caplog.text
    assert "PRIVATE_" not in caplog.text
    assert "PRIVATE_" not in "".join(errors)
    span = only_span(env)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "ValueError"
    assert len(span.events) == 1
    event_attrs = dict(span.events[0].attributes or {})
    assert event_attrs["exception.type"] == "ValueError"
    assert "exception.message" not in event_attrs
    assert "exception.stacktrace" not in event_attrs
    assert "PRIVATE_" not in str(event_attrs)
