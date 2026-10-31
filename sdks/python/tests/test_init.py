from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.util._once import Once

import telemetry_dev
from tests.conftest import MakeClient


def test_enabled_false_kill_switch_beats_exporter_seams(make: MakeClient) -> None:
    env = make(enabled=False, session_mode="process")
    assert env.client.enabled is False
    telemetry_dev.start_span("x").end()
    telemetry_dev.log("x")
    assert env.span_exporter.get_finished_spans() == ()
    assert env.log_exporter.get_finished_logs() == ()


@pytest.mark.parametrize("max_attribute_length", [-1, 1.5, float("nan"), float("inf"), True])
def test_invalid_max_attribute_length_fails_open_through_on_error(
    max_attribute_length: Any,
) -> None:
    errors: list[BaseException] = []

    client = telemetry_dev.init(
        api_key="[REDACTED:api-key]",
        max_attribute_length=max_attribute_length,
        on_error=errors.append,
        log_level="silent",
        disable_atexit=True,
    )

    assert client.enabled is False
    assert isinstance(errors[0], ValueError)


def test_large_max_attribute_length_remains_enabled(make: MakeClient) -> None:
    env = make(max_attribute_length=2**53)

    assert env.client.enabled is True
    assert env.client.max_attribute_length == 2**53


def test_zero_max_attribute_length_drops_content_and_leaves_raw_attributes_whole(
    make: MakeClient,
) -> None:
    env = make(max_attribute_length=0)

    telemetry_dev.start_span(
        "zero-limit", type="generation", model="m" * 70_000, output="secret"
    ).end()

    attributes = dict(env.span_exporter.get_finished_spans()[0].attributes or {})
    assert attributes.get("gen_ai.output.messages", "") == ""
    assert attributes["gen_ai.request.model"] == "m" * 70_000


@pytest.mark.parametrize(
    ("key", "value", "max_attribute_length", "expected"),
    [
        ("gen_ai.response.finish_reasons", ["stop", "length"], 0, ("stop", "length")),
        ("gen_ai.response.finish_reasons", ["stop", "length"], 4, ("stop", "leng")),
        ("gen_ai.operation.name", "generate_content", 0, "generate_content"),
        ("gen_ai.operation.name", "generate_content", 5, "gener"),
        ("payload", "x" * 100_000, 200_000, "x" * 100_000),
        ("payload", "x" * 100_000, 0, "x" * 100_000),
        ("payload", "x" * 100_000, None, "x" * 100_000),
    ],
)
def test_raw_attributes_follow_the_attribute_backstop_not_the_content_cap(
    make: MakeClient, key: str, value: Any, max_attribute_length: int | None, expected: Any
) -> None:
    env = make(max_attribute_length=max_attribute_length)

    telemetry_dev.start_span("chat", type="generation", attributes={key: value}).end()

    attributes = dict(env.span_exporter.get_finished_spans()[0].attributes or {})
    assert attributes[key] == expected


def test_double_init_replaces_previous_client(make: MakeClient) -> None:
    first = make()
    second = make()
    assert telemetry_dev.get_client() is second.client
    assert first.client._shutdown is True  # pyright: ignore[reportPrivateUsage]
    telemetry_dev.start_span("x").end()
    assert env_spans(first) == 0
    assert env_spans(second) == 1


def env_spans(env: SimpleNamespace) -> int:
    return len(env.span_exporter.get_finished_spans())


def test_shutdown_resets_singleton(make: MakeClient) -> None:
    env = make()
    telemetry_dev.shutdown()
    assert telemetry_dev.get_client() is None
    telemetry_dev.start_span("after").end()
    assert env.span_exporter.get_finished_spans() == ()


def _reset_global_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once())


def test_isolated_by_default(make: MakeClient) -> None:
    env = make()
    provider = env.client._tracer_provider  # pyright: ignore[reportPrivateUsage]
    assert trace.get_tracer_provider() is not provider
    trace.get_tracer("other").start_span("foreign").end()
    assert env.span_exporter.get_finished_spans() == ()


def test_register_global_default_filter_is_scope_name_only(
    make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_global_tracer_provider(monkeypatch)
    env = make(register_global=True)
    telemetry_dev.start_span("ours").end()
    trace.get_tracer("some.other.scope").start_span("foreign").end()
    names = [span.name for span in env.span_exporter.get_finished_spans()]
    assert names == ["ours"]


def test_register_global_span_filter_escape_hatch(
    make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_global_tracer_provider(monkeypatch)

    def allow_all(_span: ReadableSpan) -> bool:
        return True

    env = make(register_global=True, span_filter=allow_all)
    telemetry_dev.start_span("ours").end()
    trace.get_tracer("some.other.scope").start_span("foreign").end()
    names = sorted(span.name for span in env.span_exporter.get_finished_spans())
    assert names == ["foreign", "ours"]


def test_isolated_mode_is_unfiltered(make: MakeClient) -> None:
    env = make()
    provider = env.client._tracer_provider  # pyright: ignore[reportPrivateUsage]
    assert isinstance(provider, TracerProvider)
    provider.get_tracer("third.party").start_span("foreign").end()
    assert [span.name for span in env.span_exporter.get_finished_spans()] == ["foreign"]


def test_init_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[BaseException] = []

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("telemetry_dev._client.StampingSpanProcessor", boom)
    client = telemetry_dev.init(
        api_key="td_live_x",
        log_level="silent",
        disable_atexit=True,
        on_error=errors.append,
    )
    assert client.enabled is False
    assert errors and isinstance(errors[0], RuntimeError)
    telemetry_dev.start_span("still fine").end()
