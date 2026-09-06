from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, overload

from ._client import get_client
from ._semconv import SpanType
from ._serialize import AttributeValue
from ._spans import start_span

F = TypeVar("F", bound=Callable[..., Any])


def _bound_input(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.pop("self", None)
        arguments.pop("cls", None)
        return arguments
    except BaseException:
        return {"args": list(args), "kwargs": kwargs}


@overload
def observe(func: F) -> F: ...
@overload
def observe(
    func: None = None,
    *,
    name: str | None = None,
    type: SpanType = "span",
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Callable[[F], F]: ...


def observe(
    func: F | None = None,
    *,
    name: str | None = None,
    type: SpanType = "span",
    capture_input: bool | None = None,
    capture_output: bool | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> F | Callable[[F], F]:
    """Wrap a function in a span: arguments become `input` (param-name dict, self/cls dropped),
    the return value becomes `output`, exceptions are captured and re-raised. Supports sync,
    async, sync-generator, and async-generator functions; activates the span context for
    sync/async functions."""

    def decorate(fn: F) -> F:
        span_name = name or getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "observe")

        def open_span(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            return start_span(
                span_name,
                type=type,
                input=_bound_input(fn, args, kwargs),
                attributes=attributes,
                capture_input=capture_input,
                capture_output=capture_output,
            )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                client = get_client()
                if client is None or not client.enabled:
                    return await fn(*args, **kwargs)
                with open_span(args, kwargs) as handle:
                    result = await fn(*args, **kwargs)
                    handle.update(output=result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                client = get_client()
                if client is None or not client.enabled:
                    async for item in fn(*args, **kwargs):
                        yield item
                    return
                handle = open_span(args, kwargs)
                try:
                    async for item in fn(*args, **kwargs):
                        yield item
                except GeneratorExit:
                    handle.end()
                    raise
                except BaseException as exc:
                    handle.end(error=exc)
                    raise
                else:
                    handle.end()

            return async_gen_wrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                client = get_client()
                if client is None or not client.enabled:
                    yield from fn(*args, **kwargs)
                    return
                handle = open_span(args, kwargs)
                try:
                    yield from fn(*args, **kwargs)
                except GeneratorExit:
                    handle.end()
                    raise
                except BaseException as exc:
                    handle.end(error=exc)
                    raise
                else:
                    handle.end()

            return gen_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_client()
            if client is None or not client.enabled:
                return fn(*args, **kwargs)
            with open_span(args, kwargs) as handle:
                result = fn(*args, **kwargs)
                handle.update(output=result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorate(func)
    return decorate
