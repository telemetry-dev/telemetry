from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.util._once import Once

import telemetry_dev
from tests.conftest import MakeClient


def test_no_api_key_is_silent_noop() -> None:
    client = telemetry_dev.init(log_level="silent", disable_atexit=True)
    assert client.enabled is False
    handle = telemetry_dev.start_span("x", type="generation", input="hi")
    assert handle.traceparent() is None
    handle.update(output="y")
    handle.end()
    telemetry_dev.log("nope")
    telemetry_dev.update_current_span(output="y")
    telemetry_dev.flush()


def test_enabled_false_kill_switch_beats_exporter_seams(make: MakeClient) -> None:
    env = make(enabled=False)
    assert env.client.enabled is False
    telemetry_dev.start_span("x").end()
    assert env.span_exporter.get_finished_spans() == ()


def test_exporter_seam_enables_without_api_key(make: MakeClient) -> None:
    env = make()
    assert env.client.enabled is True
    telemetry_dev.start_span("x").end()
    assert len(env.span_exporter.get_finished_spans()) == 1


def test_api_key_from_env_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_test")
    client = telemetry_dev.init(log_level="silent", disable_atexit=True)
    assert client.enabled is True


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


def test_register_global_sets_global_provider(
    make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_global_tracer_provider(monkeypatch)
    env = make(register_global=True)
    assert trace.get_tracer_provider() is env.client._tracer_provider  # pyright: ignore[reportPrivateUsage]


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
