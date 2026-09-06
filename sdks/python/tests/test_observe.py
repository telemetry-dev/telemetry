from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from telemetry_dev import observe
from tests.conftest import span_context


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_sync_function(memory: SimpleNamespace) -> None:
    @observe
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, b=3) == 5
    span = only_span(memory)
    assert span.name.endswith("add")
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "function"
    assert json.loads(str(a["gen_ai.input.messages"])) == {"a": 2, "b": 3}
    assert a["gen_ai.output.messages"] == "5"


async def test_async_function(memory: SimpleNamespace) -> None:
    @observe(name="fetch", type="tool")
    async def fetch(url: str) -> dict[str, str]:
        return {"url": url}

    assert await fetch("https://x.test") == {"url": "https://x.test"}
    span = only_span(memory)
    assert span.name == "fetch"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert json.loads(str(a["gen_ai.tool.call.arguments"])) == {"url": "https://x.test"}
    assert json.loads(str(a["gen_ai.tool.call.result"])) == {"url": "https://x.test"}


def test_exception_marks_error_and_reraises(memory: SimpleNamespace) -> None:
    @observe
    def explode() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        explode()
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"
    assert [event.name for event in span.events] == ["exception"]


async def test_async_exception_marks_error_and_reraises(memory: SimpleNamespace) -> None:
    @observe
    async def explode() -> None:
        raise RuntimeError("async-nope")

    with pytest.raises(RuntimeError, match="async-nope"):
        await explode()
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"


def test_nesting_parents_inner_to_outer(memory: SimpleNamespace) -> None:
    @observe
    def inner() -> str:
        return "in"

    @observe
    def outer() -> str:
        return inner()

    outer()
    inner_span, outer_span = memory.span_exporter.get_finished_spans()
    assert inner_span.name.endswith("inner")
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == span_context(outer_span).span_id
    assert span_context(inner_span).trace_id == span_context(outer_span).trace_id


def test_self_dropped_from_input(memory: SimpleNamespace) -> None:
    class Service:
        @observe
        def run(self, task: str) -> str:
            return task

    Service().run("go")
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == {"task": "go"}


def test_capture_overrides(memory: SimpleNamespace) -> None:
    @observe(capture_input=False, capture_output=False)
    def secret(password: str) -> str:
        return password

    secret("hunter2")
    a = attrs(only_span(memory))
    assert "gen_ai.input.messages" not in a
    assert "gen_ai.output.messages" not in a
    assert a["gen_ai.operation.name"] == "function"


def test_static_attributes_option(memory: SimpleNamespace) -> None:
    @observe(attributes={"custom.flag": True})
    def noop() -> None:
        return None

    noop()
    assert attrs(only_span(memory))["custom.flag"] is True


def test_sync_generator(memory: SimpleNamespace) -> None:
    @observe
    def counter(n: int) -> Iterator[int]:
        yield from range(n)

    assert list(counter(3)) == [0, 1, 2]
    span = only_span(memory)
    a = attrs(span)
    assert json.loads(str(a["gen_ai.input.messages"])) == {"n": 3}
    assert "gen_ai.output.messages" not in a


def test_sync_generator_error(memory: SimpleNamespace) -> None:
    @observe
    def broken() -> Iterator[int]:
        yield 1
        raise ValueError("mid-stream")

    gen = broken()
    assert next(gen) == 1
    with pytest.raises(ValueError, match="mid-stream"):
        next(gen)
    assert only_span(memory).status.status_code == StatusCode.ERROR


async def test_async_generator(memory: SimpleNamespace) -> None:
    @observe
    async def stream(n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    assert [i async for i in stream(2)] == [0, 1]
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == {"n": 2}


def test_uninitialized_calls_through() -> None:
    @observe
    def plain(x: int) -> int:
        return x * 2

    assert plain(4) == 8


def test_wraps_preserves_metadata(memory: SimpleNamespace) -> None:
    @observe
    def documented() -> None:
        """Docs."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docs."
