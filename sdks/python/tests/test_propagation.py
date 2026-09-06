from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from opentelemetry import baggage, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)
from opentelemetry.trace import Link, SpanContext, SpanKind, TraceFlags, TraceState
from opentelemetry.util.types import Attributes

import telemetry_dev
from telemetry_dev._context import session_span_context, with_session_parent
from telemetry_dev._serialize import MaskContext
from tests.conftest import MakeClient, span_context


def spans_by_name(env: SimpleNamespace) -> dict[str, ReadableSpan]:
    return {span.name: span for span in env.span_exporter.get_finished_spans()}


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_every_nested_span_is_stamped(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(
        user_id="u1", session_id="s1", metadata={"plan": "pro"}
    ):
        with telemetry_dev.start_span("root"):
            with telemetry_dev.start_span("child"):
                telemetry_dev.start_span("grandchild").end()
    by_name = spans_by_name(memory)
    assert set(by_name) == {"root", "child", "grandchild"}
    for name, span in by_name.items():
        a = attrs(span)
        assert a["user.id"] == "u1", name
        assert a["gen_ai.conversation.id"] == "s1", name
        assert a["td.metadata.plan"] == "pro", name


def test_not_stamped_outside_context(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(user_id="u1"):
        pass
    telemetry_dev.start_span("outside").end()
    assert "user.id" not in attrs(spans_by_name(memory)["outside"])


def test_nested_merge_inner_wins(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(user_id="outer", metadata={"a": "1", "b": "1"}):
        with telemetry_dev.propagate_attributes(session_id="s", metadata={"b": "2"}):
            telemetry_dev.start_span("inner").end()
        telemetry_dev.start_span("outer").end()
    by_name = spans_by_name(memory)
    inner = attrs(by_name["inner"])
    assert inner["user.id"] == "outer"
    assert inner["gen_ai.conversation.id"] == "s"
    assert inner["td.metadata.a"] == "1"
    assert inner["td.metadata.b"] == "2"
    outer = attrs(by_name["outer"])
    assert outer["td.metadata.b"] == "1"
    assert "gen_ai.conversation.id" not in outer


def test_reserved_metadata_keys_dropped(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(
        metadata={"userId": "x", "sessionId": "y", "user_id": "z", "session_id": "w", "ok": "v"}
    ):
        telemetry_dev.start_span("span").end()
    a = attrs(spans_by_name(memory)["span"])
    assert a["td.metadata.ok"] == "v"
    for reserved in ("userId", "sessionId", "user_id", "session_id"):
        assert f"td.metadata.{reserved}" not in a


def test_propagation_across_threads(memory: SimpleNamespace) -> None:
    def work() -> None:
        telemetry_dev.start_span("threaded").end()

    with telemetry_dev.propagate_attributes(user_id="u-thread"):
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(work,))
        thread.start()
        thread.join()
    assert attrs(spans_by_name(memory)["threaded"])["user.id"] == "u-thread"


def test_propagation_across_asyncio_tasks(memory: SimpleNamespace) -> None:
    async def child() -> None:
        telemetry_dev.start_span("task-span").end()

    async def main() -> None:
        with telemetry_dev.propagate_attributes(user_id="u-async"):
            await asyncio.create_task(child())

    asyncio.run(main())
    assert attrs(spans_by_name(memory)["task-span"])["user.id"] == "u-async"


def test_explicit_parent_keeps_propagated_attributes(memory: SimpleNamespace) -> None:
    foreign_traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    with telemetry_dev.propagate_attributes(user_id="kept"):
        telemetry_dev.start_span("joined", parent=foreign_traceparent).end()
    span = spans_by_name(memory)["joined"]
    assert attrs(span)["user.id"] == "kept"
    assert span_context(span).trace_id == int("0123456789abcdef0123456789abcdef", 16)


def test_get_traceparent_round_trip(memory: SimpleNamespace) -> None:
    assert telemetry_dev.get_traceparent() is None
    with telemetry_dev.start_span("root") as root:
        current = telemetry_dev.get_traceparent()
        assert current == root.traceparent()
        assert current is not None
    telemetry_dev.start_span("joined", parent=current).end()
    by_name = spans_by_name(memory)
    assert span_context(by_name["joined"]).trace_id == span_context(by_name["root"]).trace_id
    assert telemetry_dev.get_traceparent() is None


def test_logs_get_propagated_attributes(memory: SimpleNamespace) -> None:
    with telemetry_dev.propagate_attributes(user_id="u-log", metadata={"plan": "pro"}):
        telemetry_dev.log("hello")
    (record,) = memory.log_exporter.get_finished_logs()
    log_attrs = dict(record.log_record.attributes or {})
    assert log_attrs["user.id"] == "u-log"
    assert log_attrs["td.metadata.plan"] == "pro"


def test_invalid_traceparent_falls_back_to_active_context(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("root") as root:
        telemetry_dev.start_span("child", parent="not-a-traceparent").end()
        root_context = root.span.get_span_context()
    by_name = spans_by_name(memory)
    child = by_name["child"]
    assert span_context(child).trace_id == root_context.trace_id
    parent = child.parent
    assert parent is not None
    assert parent.span_id == root_context.span_id


def test_propagate_attributes_never_raises_on_bad_metadata(memory: SimpleNamespace) -> None:
    circular: dict[str, object] = {}
    circular["self"] = circular

    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError("broken repr")

    with telemetry_dev.propagate_attributes(
        user_id="safe", metadata={"circular": circular, "broken": BrokenRepr(), "ok": "v"}
    ):
        telemetry_dev.start_span("survives").end()
    span = spans_by_name(memory)["survives"]
    assert attrs(span)["user.id"] == "safe"
    assert attrs(span)["td.metadata.ok"] == "v"
    assert "td.metadata.circular" not in attrs(span)


def test_propagate_attributes_before_init_does_not_raise() -> None:
    telemetry_dev.shutdown()
    with telemetry_dev.propagate_attributes(user_id="early", metadata={"k": object()}):
        pass


def test_session_span_context_is_pinned_to_the_ts_sdk() -> None:
    # python -c "import hashlib;print(hashlib.sha256(b'td_live_k\x00s1').hexdigest()[:48])"
    ctx = session_span_context("td_live_k", "s1")
    assert f"{ctx.trace_id:032x}{ctx.span_id:016x}" == (
        "9e21d32713f05502cf537dba02d3d6fee1dbbf7220b0fab7"
    )
    assert ctx.is_remote
    assert session_span_context("td_live_k", "s2").trace_id != ctx.trace_id
    assert session_span_context("other", "s1").trace_id != ctx.trace_id


def test_session_span_context_replaces_lone_surrogates_like_ts() -> None:
    ctx = session_span_context("td_live_k", "surrogate-\ud800")
    assert f"{ctx.trace_id:032x}{ctx.span_id:016x}" == (
        "9df9d5baf81b578cec335645f2d63a8e345189a4d765e6d5"
    )
    # A valid pair must combine to one code point, not become two U+FFFD.
    ctx = session_span_context("td_live_k", "pair-\ud83d\ude00")
    assert f"{ctx.trace_id:032x}{ctx.span_id:016x}" == (
        "76b57d1fe53960b397a8f94a09fdb9a57ac606d0c477796b"
    )


def test_root_spans_of_one_session_share_the_session_trace(make: MakeClient) -> None:
    env = make(api_key="td_live_k")
    session = session_span_context("td_live_k", "s1")
    with telemetry_dev.propagate_attributes(session_id="s1"):
        telemetry_dev.start_span("a").end()
        telemetry_dev.start_span("b").end()
    telemetry_dev.start_span("c", attributes={"gen_ai.conversation.id": "s1"}).end()
    spans = env.span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["a", "b", "c"]
    for span in spans:
        assert span_context(span).trace_id == session.trace_id
        assert span.parent is not None
        assert span.parent.span_id == session.span_id


def test_nested_spans_keep_their_real_parent_inside_a_session(make: MakeClient) -> None:
    env = make(api_key="td_live_k")
    with telemetry_dev.propagate_attributes(session_id="s1"):
        with telemetry_dev.start_span("root") as root:
            telemetry_dev.start_span("child").end()
            root_context = root.span.get_span_context()
    by_name = spans_by_name(env)
    child = by_name["child"]
    assert child.parent is not None
    assert child.parent.span_id == root_context.span_id
    root_span = by_name["root"]
    assert root_span.parent is not None
    assert root_span.parent.span_id == session_span_context("td_live_k", "s1").span_id


def test_empty_session_inputs_do_not_add_a_session_parent(make: MakeClient) -> None:
    env = make(api_key="")
    with telemetry_dev.propagate_attributes(session_id="s1"):
        telemetry_dev.start_span("no-key").end()
    (no_key,) = env.span_exporter.get_finished_spans()
    assert no_key.parent is None


def test_explicit_empty_session_overrides_the_propagated_session(make: MakeClient) -> None:
    env = make(api_key="td_live_k")
    with telemetry_dev.propagate_attributes(session_id="from-context"):
        telemetry_dev.start_span("empty-explicit", attributes={"gen_ai.conversation.id": ""}).end()
    (empty_explicit,) = env.span_exporter.get_finished_spans()
    assert empty_explicit.parent is None


def test_spans_without_a_session_keep_separate_traces(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("a").end()
    telemetry_dev.start_span("b").end()
    a, b = memory.span_exporter.get_finished_spans()
    assert span_context(a).trace_id != span_context(b).trace_id
    assert a.parent is None


@pytest.mark.parametrize("root", [ALWAYS_OFF, TraceIdRatioBased(0.5)])
def test_session_roots_obey_the_root_sampler(make: MakeClient, root: Sampler) -> None:
    sampler = ParentBased(root)
    env = make(api_key="td_live_k", sampler=sampler)
    expected: list[int] = []
    for i in range(20):
        session_id = f"session-{i}"
        trace_id = session_span_context("td_live_k", session_id).trace_id
        sampled = sampler.should_sample(
            Context(), trace_id, "turn", SpanKind.INTERNAL
        ).decision.is_sampled()
        with telemetry_dev.propagate_attributes(session_id=session_id):
            with telemetry_dev.start_span("turn") as turn:
                assert turn.span.is_recording() is sampled
                assert turn.span.get_span_context().trace_id == trace_id
                with telemetry_dev.start_span("child") as child:
                    assert child.span.is_recording() is sampled
                    assert child.span.get_span_context().trace_id == trace_id
        if sampled:
            expected.extend([trace_id, trace_id])
    assert [span_context(s).trace_id for s in env.span_exporter.get_finished_spans()] == expected


@pytest.mark.parametrize("sampled", [False, True])
def test_session_spans_keep_real_parent_sampling(make: MakeClient, sampled: bool) -> None:
    env = make(api_key="td_live_k", sampler=ParentBased(ALWAYS_ON))
    parent = SpanContext(
        trace_id=int("12345678901234567890123456789012", 16),
        span_id=int("1234567890123456", 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT),
        trace_state=TraceState([("vendor", "kept")]),
    )
    with telemetry_dev.propagate_attributes(session_id="s1"):
        with telemetry_dev.start_span("turn", parent=parent) as turn:
            assert trace.get_current_span() is turn.span
            assert turn.span.is_recording() is sampled
            with telemetry_dev.start_span("child") as child:
                assert child.span.is_recording() is sampled
                assert child.span.get_span_context().trace_id == parent.trace_id
                assert child.span.get_span_context().trace_state == parent.trace_state
    spans = env.span_exporter.get_finished_spans()
    assert [s.name for s in spans] == (["child", "turn"] if sampled else [])
    if sampled:
        assert spans[0].parent is not None
        assert spans[0].parent.span_id == turn.span.get_span_context().span_id


@pytest.mark.parametrize(
    ("session_id", "capture_input"), [(None, True), ("s1", True), ("s1", False)]
)
def test_root_sampling_sees_safe_initial_attributes(
    make: MakeClient, session_id: str | None, capture_input: bool
) -> None:
    class AttributeSampler(Sampler):
        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Attributes = None,
            links: Sequence[Link] | None = None,
            trace_state: TraceState | None = None,
        ) -> SamplingResult:
            a = attributes or {}
            accepted = (
                a.get("gen_ai.operation.name") == "chat"
                and a.get("gen_ai.request.model") == "sampled-model"
                and a.get("gen_ai.provider.name") == "explicit-provider"
                and a.get("user.id") == "inherited-user"
                and a.get("td.metadata.plan") == "explicit-plan"
                and a.get("gen_ai.conversation.id") == session_id
                and a.get("gen_ai.input.messages") == ("masked-1" if capture_input else None)
                and "gen_ai.output.messages" not in a
            )
            return SamplingResult(
                Decision.RECORD_AND_SAMPLE if accepted else Decision.DROP, attributes
            )

        def get_description(self) -> str:
            return "AttributeSampler"

    masked = 0

    def mask(_value: object, _context: MaskContext) -> str:
        nonlocal masked
        masked += 1
        return f"masked-{masked}"

    env = make(
        api_key="td_live_k",
        sampler=ParentBased(AttributeSampler()),
        capture_input=False,
        capture_output=False,
        mask=mask,
    )
    with telemetry_dev.propagate_attributes(
        user_id="inherited-user", session_id=session_id, metadata={"plan": "inherited-plan"}
    ):
        telemetry_dev.start_span(
            "accepted",
            type="generation",
            model="sampled-model",
            provider="field-provider",
            metadata={"plan": "field-plan"},
            attributes={
                "gen_ai.provider.name": "explicit-provider",
                "td.metadata.plan": "explicit-plan",
            },
            input="private input",
            output="private output",
            capture_input=capture_input,
        ).end()
        telemetry_dev.start_span(
            "rejected", type="generation", model="other-model", capture_input=capture_input
        ).end()
    exported = env.span_exporter.get_finished_spans()
    assert [span.name for span in exported] == ["accepted"]
    a = attrs(exported[0])
    assert a["gen_ai.request.model"] == "sampled-model"
    assert a["gen_ai.provider.name"] == "explicit-provider"
    assert a["user.id"] == "inherited-user"
    assert a["td.metadata.plan"] == "explicit-plan"
    assert a.get("gen_ai.input.messages") == ("masked-1" if capture_input else None)
    assert "gen_ai.output.messages" not in a


def test_session_sampling_keeps_baggage_changed_after_attaching_the_session_parent(
    make: MakeClient,
) -> None:
    class TenantSampler(Sampler):
        def should_sample(
            self,
            parent_context: Context | None,
            trace_id: int,
            name: str,
            kind: SpanKind | None = None,
            attributes: Attributes = None,
            links: Sequence[Link] | None = None,
            trace_state: TraceState | None = None,
        ) -> SamplingResult:
            accepted = baggage.get_baggage("tenant", parent_context) == "allowed"
            return SamplingResult(
                Decision.RECORD_AND_SAMPLE if accepted else Decision.DROP, attributes
            )

        def get_description(self) -> str:
            return "TenantSampler"

    env = make(api_key="td_live_k", sampler=ParentBased(TenantSampler()))
    before = baggage.set_baggage("tenant", "denied", Context())
    parent = with_session_parent(before, "s1", "td_live_k")
    allowed = baggage.set_baggage("tenant", "allowed", parent)
    telemetry_dev.start_span("allowed", parent=allowed).end()
    telemetry_dev.start_span("denied", parent=parent).end()
    telemetry_dev.start_span("removed", parent=baggage.remove_baggage("tenant", allowed)).end()
    assert [span.name for span in env.span_exporter.get_finished_spans()] == ["allowed"]
