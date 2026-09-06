from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from opentelemetry.trace import SpanContext, TraceFlags, TraceState

import telemetry_dev
from telemetry_dev._config import SessionMode
from tests.conftest import MakeClient, span_context


def attrs(value: Any) -> dict[str, object]:
    return dict(value.attributes or {})


def test_process_mode_groups_roots_and_logs(make: MakeClient) -> None:
    env = make(api_key="td_live_k", session_mode="process")
    telemetry_dev.start_span("a").end()
    telemetry_dev.start_span("b").end()
    telemetry_dev.log("outside")
    a, b = env.span_exporter.get_finished_spans()
    session_id = attrs(a)["gen_ai.conversation.id"]
    assert isinstance(session_id, str) and len(session_id) == 36
    assert span_context(a).trace_id == span_context(b).trace_id
    assert attrs(b)["gen_ai.conversation.id"] == session_id
    (record,) = env.log_exporter.get_finished_logs()
    assert attrs(record.log_record)["gen_ai.conversation.id"] == session_id


def test_explicit_parent_session_wins_for_children_and_logs(make: MakeClient) -> None:
    env = make(api_key="td_live_k", session_mode="process")
    with telemetry_dev.start_span(
        "explicit-root", attributes={"gen_ai.conversation.id": "explicit"}
    ):
        telemetry_dev.start_span("explicit-child").end()
        telemetry_dev.log("explicit-log")
    for span in env.span_exporter.get_finished_spans():
        assert attrs(span)["gen_ai.conversation.id"] == "explicit"
    (record,) = env.log_exporter.get_finished_logs()
    assert attrs(record.log_record)["gen_ai.conversation.id"] == "explicit"


def test_concurrently_interleaved_explicit_sessions_do_not_mix(make: MakeClient) -> None:
    env = make(api_key="td_live_k", session_mode="process")

    async def emit(session_id: str) -> None:
        with telemetry_dev.propagate_attributes(session_id=session_id):
            await asyncio.sleep(0)
            with telemetry_dev.start_span(f"{session_id}-root"):
                await asyncio.sleep(0)
                telemetry_dev.start_span(f"{session_id}-child").end()

    async def main() -> None:
        await asyncio.gather(emit("one"), emit("two"))

    asyncio.run(main())
    for span in env.span_exporter.get_finished_spans():
        assert attrs(span)["gen_ai.conversation.id"] == span.name.split("-")[0]


def test_process_mode_honors_remote_parent(make: MakeClient) -> None:
    env = make(api_key="td_live_k", session_mode="process")
    parent = SpanContext(
        trace_id=int("0123456789abcdef0123456789abcdef", 16),
        span_id=int("0123456789abcdef", 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )
    telemetry_dev.start_span("joined", parent=parent).end()
    (span,) = env.span_exporter.get_finished_spans()
    assert span_context(span).trace_id == parent.trace_id
    assert span.parent is not None and span.parent.span_id == parent.span_id


def test_reinit_generates_a_new_process_session(make: MakeClient) -> None:
    first = make(api_key="td_live_k", session_mode="process")
    first_id = first.client._process_session_id  # pyright: ignore[reportPrivateUsage]
    telemetry_dev.shutdown()
    second = make(api_key="td_live_k", session_mode="process")
    assert second.client._process_session_id != first_id  # pyright: ignore[reportPrivateUsage]


def test_invalid_mode_fails_open_via_on_error() -> None:
    errors: list[BaseException] = []
    client = telemetry_dev.init(
        api_key="td_live_k",
        session_mode=cast(SessionMode, "invalid"),
        on_error=errors.append,
        log_level="silent",
        disable_atexit=True,
    )
    assert client.enabled is False
    assert isinstance(errors[0], ValueError)


def test_disabled_process_mode_is_a_noop(make: MakeClient) -> None:
    env: SimpleNamespace = make(enabled=False, session_mode="process")
    assert env.client._process_session_id is None  # pyright: ignore[reportPrivateUsage]
