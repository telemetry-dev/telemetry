from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

import telemetry_dev
from openrouter.chat import Chat
from openrouter.embeddings import Embeddings
from openrouter.responses import Responses

__version__ = "0.1.1"

RequestMapper = Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]

_PROVIDER = "openrouter"
_WRAPPED_ATTR = "_telemetry_dev_openrouter_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_openrouter_original"
_CAPTURE_TRUNCATED_ATTRIBUTE = "telemetry.dev.capture.truncated"
_ORIGINALS: list[tuple[type[Any], str, Any]] = []
_installed = False
_install_lock = threading.Lock()
_T = TypeVar("_T")


def _absent(value: Any) -> bool:
    if value is None:
        return True
    value_type = type(cast(object, value))
    return value_type.__module__ == "openrouter.types.basemodel" and value_type.__name__ == "Unset"


def _field(value: Any, name: str) -> Any:
    if _absent(value):
        return None
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name)
    return getattr(value, name, None)


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[Any], value))
    return []


def _native(value: Any) -> Any:
    if _absent(value):
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _native(item) for key, item in mapping.items() if item is not None}
    sequence = _sequence_items(value)
    if sequence:
        return [_native(item) for item in sequence]
    return value


def _snake_case(name: str) -> str:
    return "".join(
        f"_{character.lower()}" if character.isupper() else character for character in name
    )


def _response_native(value: Any, seen: dict[int, Any] | None = None) -> Any:
    if _absent(value):
        return None
    if seen is None:
        seen = {}
    if isinstance(value, str | bytes | bytearray | bool | int | float):
        return value
    value_id = id(value)
    if value_id in seen:
        return seen[value_id]
    raw = _field(value, "raw")
    if raw is not None and (
        _field(value, "is_unknown") is True
        or _field(value, "isUnknown") is True
        or _field(value, "type") == "UNKNOWN"
    ):
        return _response_native(raw, seen)
    if hasattr(value, "model_dump"):
        return _response_native(value.model_dump(mode="json", exclude_none=True), seen)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        seen[value_id] = result
        for key, item in cast(Mapping[Any, Any], value).items():
            result[_snake_case(str(key))] = _response_native(item, seen)
        return result
    sequence = _sequence_items(value)
    if sequence:
        result_list: list[Any] = []
        seen[value_id] = result_list
        result_list.extend(_response_native(item, seen) for item in sequence)
        return result_list
    return value


def _response_event(event: Any) -> Any:
    raw = _field(event, "raw")
    unknown = (
        _field(event, "is_unknown") is True
        or _field(event, "isUnknown") is True
        or _field(event, "type") == "UNKNOWN"
    )
    return raw if raw is not None and unknown else event


def _event_field(event: Any, snake_name: str, camel_name: str) -> Any:
    value = _field(event, snake_name)
    return value if value is not None else _field(event, camel_name)


def _number(value: Any) -> int | float | None:
    if _absent(value) or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _error_code(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return str(value)


def _usage(fields: dict[str, int | float | None]) -> dict[str, int | float] | None:
    usage = {key: value for key, value in fields.items() if value is not None}
    return usage or None


def _stop_sequences(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    strings = [item for item in _sequence_items(value) if isinstance(item, str)]
    return strings or None


def _clean_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _cost_usd(raw_usage: Any) -> float | None:
    cost = _number(_field(raw_usage, "cost"))
    if cost is not None:
        return cost
    return _number(_field(_field(raw_usage, "cost_details"), "upstream_inference_cost"))


def _chat_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    max_completion_tokens = _number(params.get("max_completion_tokens"))
    return (
        f"chat {model or 'unknown'}",
        {
            "type": "generation",
            "model": model,
            "input": _native(params.get("messages")),
            "temperature": _number(params.get("temperature")),
            "top_p": _number(params.get("top_p")),
            "top_k": _number(params.get("top_k")),
            "max_tokens": max_completion_tokens
            if max_completion_tokens is not None
            else _number(params.get("max_tokens")),
            "stop_sequences": _stop_sequences(params.get("stop")),
            "seed": _number(params.get("seed")),
            "frequency_penalty": _number(params.get("frequency_penalty")),
            "presence_penalty": _number(params.get("presence_penalty")),
        },
    )


def _chat_usage(raw: Any) -> dict[str, int | float] | None:
    prompt_details = _field(raw, "prompt_tokens_details")
    completion_details = _field(raw, "completion_tokens_details")
    return _usage(
        {
            "input_tokens": _number(_field(raw, "prompt_tokens")),
            "output_tokens": _number(_field(raw, "completion_tokens")),
            "total_tokens": _number(_field(raw, "total_tokens")),
            "cache_read_input_tokens": _number(_field(prompt_details, "cached_tokens")),
            "cache_creation_input_tokens": _number(_field(prompt_details, "cache_write_tokens")),
            "reasoning_output_tokens": _number(_field(completion_details, "reasoning_tokens")),
        }
    )


def _chat_output_message(message: Any) -> dict[str, Any]:
    if _absent(message):
        return {}
    native = _native(message)
    if not isinstance(native, dict):
        return cast(dict[str, Any], native)
    if _field(message, "content") is None:
        native["content"] = None
    return cast(dict[str, Any], native)


def _chat_response(response: Any) -> dict[str, Any]:
    choices = list(_field(response, "choices") or [])
    finish_reasons = [
        reason
        for choice in choices
        if (reason := _string(_field(choice, "finish_reason"))) is not None
    ]
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "output": [_chat_output_message(_field(choice, "message")) for choice in choices],
        "usage": _chat_usage(_field(response, "usage")),
        "cost_usd": _cost_usd(_field(response, "usage")),
        "attributes": (
            {"gen_ai.response.finish_reasons": finish_reasons} if len(finish_reasons) > 1 else None
        ),
    }


def _responses_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    return (
        f"chat {model or 'unknown'}",
        {
            "type": "generation",
            "model": model,
            "input": _native(params.get("input")),
            "system_instructions": _native(params.get("instructions")),
            "temperature": _number(params.get("temperature")),
            "top_p": _number(params.get("top_p")),
            "top_k": _number(params.get("top_k")),
            "max_tokens": _number(params.get("max_output_tokens")),
            "frequency_penalty": _number(params.get("frequency_penalty")),
            "presence_penalty": _number(params.get("presence_penalty")),
        },
    )


def _responses_usage(raw: Any) -> dict[str, int | float] | None:
    input_details = _field(raw, "input_tokens_details")
    output_details = _field(raw, "output_tokens_details")
    return _usage(
        {
            "input_tokens": _number(_field(raw, "input_tokens")),
            "output_tokens": _number(_field(raw, "output_tokens")),
            "total_tokens": _number(_field(raw, "total_tokens")),
            "cache_read_input_tokens": _number(_field(input_details, "cached_tokens")),
            "cache_creation_input_tokens": _number(_field(input_details, "cache_write_tokens")),
            "reasoning_output_tokens": _number(_field(output_details, "reasoning_tokens")),
        }
    )


def _response_failed_error(response: Any) -> RuntimeError:
    error = _field(response, "error")
    if error is None:
        return RuntimeError("response.failed")
    code = _error_code(_field(error, "code"))
    message = _string(_field(error, "message"))
    if code and message:
        return RuntimeError(f"response.failed: {code}: {message}")
    if code:
        return RuntimeError(f"response.failed: {code}")
    if message:
        return RuntimeError(f"response.failed: {message}")
    return RuntimeError("response.failed")


def _response_stream_error(event: Any) -> RuntimeError:
    code = _error_code(_field(event, "code"))
    message = _string(_field(event, "message"))
    if code and message:
        return RuntimeError(f"response.error: {code}: {message}")
    if code:
        return RuntimeError(f"response.error: {code}")
    if message:
        return RuntimeError(f"response.error: {message}")
    return RuntimeError("response.error")


def _chunk_error(chunk: Any) -> RuntimeError | None:
    error = _field(chunk, "error")
    if error is None:
        return None
    code = _error_code(_field(error, "code"))
    message = _string(_field(error, "message"))
    if code and message:
        return RuntimeError(f"stream error {code}: {message}")
    if message:
        return RuntimeError(f"stream error: {message}")
    return RuntimeError(f"stream error {code}" if code else "stream error")


def _responses_response(
    response: Any, *, include_error: bool = True, include_output: bool = True
) -> dict[str, Any]:
    status = _string(_field(response, "status"))
    incomplete_details = _field(response, "incomplete_details")
    fields: dict[str, Any] = {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "usage": _responses_usage(_field(response, "usage")),
        "cost_usd": _cost_usd(_field(response, "usage")),
        "finish_reason": "stop"
        if status == "completed"
        else _string(_field(incomplete_details, "reason")) or status,
    }
    if include_output:
        fields["output"] = _response_native(_field(response, "output"))
    if include_error and status == "failed":
        fields["error"] = _response_failed_error(response)
    return fields


def _embeddings_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    return (
        f"embeddings {model or 'unknown'}",
        {"type": "embedding", "model": model, "input": _native(params.get("input"))},
    )


def _embeddings_response(response: Any) -> dict[str, Any]:
    if isinstance(response, str) or _absent(response):
        return {}
    raw_usage = _field(response, "usage")
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "usage": _usage(
            {
                "input_tokens": _number(_field(raw_usage, "prompt_tokens")),
                "total_tokens": _number(_field(raw_usage, "total_tokens")),
            }
        ),
        "cost_usd": _cost_usd(raw_usage),
    }


def _end_once(handle: telemetry_dev.SpanHandle) -> Callable[..., None]:
    ended = False

    def end(**fields: Any) -> None:
        nonlocal ended
        if ended:
            return
        ended = True
        handle.end(**_clean_fields(fields))

    return end


class _ChatChoice:
    def __init__(self) -> None:
        self.role: str | None = None
        self.content = ""
        self.reasoning = ""
        self.reasoning_details: list[Any] = []
        self.refusal = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None


def _choice_state(
    states: dict[int, _ChatChoice], index: int, budget: telemetry_dev.CaptureBudget
) -> _ChatChoice | None:
    if index not in states:
        if len(states) >= budget.max_items:
            budget.truncated = True
            return None
        states[index] = _ChatChoice()
    return states[index]


def _merge_tool_call(state: _ChatChoice, delta: Any) -> None:
    index = _field(delta, "index")
    tool_index = index if isinstance(index, int) else len(state.tool_calls)
    current = dict(state.tool_calls.get(tool_index, {}))
    tool_id = _field(delta, "id")
    tool_type = _field(delta, "type")
    if tool_id is not None:
        current["id"] = tool_id
    if tool_type is not None:
        current["type"] = tool_type
    incoming_function = _field(delta, "function")
    if incoming_function is not None:
        current_function = dict(cast(Mapping[str, Any], current.get("function", {})))
        name = _field(incoming_function, "name")
        arguments = _field(incoming_function, "arguments")
        if name is not None:
            current_function["name"] = name
        if isinstance(arguments, str):
            current_function["arguments"] = f"{current_function.get('arguments', '')}{arguments}"
        current["function"] = current_function
    state.tool_calls[tool_index] = current


def _record_chat_chunk(
    chunk: Any, states: dict[int, _ChatChoice], budget: telemetry_dev.CaptureBudget
) -> dict[str, Any]:
    for choice in _sequence_items(_field(chunk, "choices")):
        index = _field(choice, "index")
        state = _choice_state(states, index if isinstance(index, int) else 0, budget)
        if state is None:
            continue
        delta = _field(choice, "delta")
        role = _field(delta, "role")
        content = _field(delta, "content")
        reasoning = _field(delta, "reasoning")
        refusal = _field(delta, "refusal")
        if isinstance(role, str):
            state.role = role
        if isinstance(content, str) and budget.accept(content):
            state.content += content
        if isinstance(reasoning, str) and budget.accept(reasoning):
            state.reasoning += reasoning
        for detail in _sequence_items(_field(delta, "reasoning_details")):
            native_detail = _native(detail)
            if native_detail is not None and budget.accept(native_detail):
                state.reasoning_details.append(native_detail)
        if isinstance(refusal, str) and budget.accept(refusal):
            state.refusal += refusal
        for tool_call in _sequence_items(_field(delta, "tool_calls")):
            if budget.accept(tool_call):
                _merge_tool_call(state, tool_call)
        finish_reason = _field(choice, "finish_reason")
        if isinstance(finish_reason, str):
            state.finish_reason = finish_reason
    return {
        "response_id": _string(_field(chunk, "id")),
        "response_model": _string(_field(chunk, "model")),
        "usage": _chat_usage(_field(chunk, "usage")),
        "cost_usd": _cost_usd(_field(chunk, "usage")),
    }


def _chat_output(states: Mapping[int, _ChatChoice]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _, state in sorted(states.items()):
        message: dict[str, Any] = {"role": state.role or "assistant"}
        if state.content:
            message["content"] = state.content
        elif state.tool_calls:
            message["content"] = None
        if state.reasoning:
            message["reasoning"] = state.reasoning
        if state.reasoning_details:
            message["reasoning_details"] = state.reasoning_details
        if state.refusal:
            message["refusal"] = state.refusal
        if state.tool_calls:
            message["tool_calls"] = [call for _, call in sorted(state.tool_calls.items())]
        output.append(message)
    return output


def _chat_partial(
    states: Mapping[int, _ChatChoice],
    usage: dict[str, int | float] | None,
    cost_usd: float | None,
    budget: telemetry_dev.CaptureBudget,
) -> dict[str, Any]:
    finish_reasons = [
        state.finish_reason
        for _, state in sorted(states.items())
        if state.finish_reason is not None
    ]
    return _mark_capture_truncated(
        {
            "output": _chat_output(states) if states else None,
            "usage": usage,
            "cost_usd": cost_usd,
            "finish_reason": finish_reasons[0] if finish_reasons else None,
            "attributes": (
                {"gen_ai.response.finish_reasons": finish_reasons}
                if len(finish_reasons) > 1
                else None
            ),
        },
        budget,
    )


def _mark_capture_truncated(
    fields: dict[str, Any], budget: telemetry_dev.CaptureBudget
) -> dict[str, Any]:
    if budget.truncated:
        attributes = dict(fields.get("attributes") or {})
        attributes[_CAPTURE_TRUNCATED_ATTRIBUTE] = True
        fields["attributes"] = attributes
    elif _CAPTURE_TRUNCATED_ATTRIBUTE in (fields.get("attributes") or {}):
        attributes = dict(fields["attributes"])
        del attributes[_CAPTURE_TRUNCATED_ATTRIBUTE]
        fields["attributes"] = attributes or None
    return fields


def _hook_response_close(inner: Any, finish: Callable[[], None]) -> None:
    response = getattr(inner, "response", None)
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):

        def _close_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return close(*args, **kwargs)
            finally:
                finish()

        response.close = _close_hook
    aclose = getattr(response, "aclose", None)
    if callable(aclose):
        aclose_fn = cast(Callable[..., Any], aclose)

        async def _aclose_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return await aclose_fn(*args, **kwargs)
            finally:
                finish()

        response.aclose = _aclose_hook


async def _close_async_stream(inner: Any) -> None:
    close = getattr(inner, "close", None)
    if not callable(close):
        return
    result = close()
    if not hasattr(result, "__await__"):
        return
    task = asyncio.ensure_future(cast(Awaitable[Any], result))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    if cancellation is not None:
        task.result()
        raise cancellation
    task.result()


class _InstrumentedStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._handle = handle
        self._started_at = started_at
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
        self._cost_usd: float | None = None
        self._saw_first = False
        self._consume: Iterator[Any] | None = None
        self._in_next = False
        self._budget = telemetry_dev.CaptureBudget.from_client()
        _hook_response_close(inner, self._on_response_close)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = next(self._inner)
                except StopIteration:
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                if not self._record(chunk):
                    self._end(**self._partial(), error=_chunk_error(chunk))
                yield chunk
        finally:
            self.close()

    def __iter__(self) -> Iterator[Any]:
        if self._consume is None:
            self._consume = self._iterate()
        return self._consume

    def __next__(self) -> Any:
        return next(self.__iter__())

    def __enter__(self) -> _InstrumentedStream:
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc is not None:
            self._end(**self._partial(), error=exc)
        self.close()

    def _partial(self) -> dict[str, Any]:
        return _chat_partial(self._states, self._usage, self._cost_usd, self._budget)

    def _finish(self) -> None:
        self._end(**self._partial())

    def _on_response_close(self) -> None:
        if not self._in_next:
            self._finish()

    def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, chunk: Any) -> bool:
        update = _record_chat_chunk(chunk, self._states, self._budget)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
                response_id=update.get("response_id"),
                response_model=update.get("response_model"),
            )
        if update.get("usage") is not None:
            self._usage = update["usage"]
        if update.get("cost_usd") is not None:
            self._cost_usd = update["cost_usd"]
        return _chunk_error(chunk) is None


class _InstrumentedAsyncStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._handle = handle
        self._started_at = started_at
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
        self._cost_usd: float | None = None
        self._saw_first = False
        self._consume: AsyncIterator[Any] | None = None
        self._in_next = False
        self._budget = telemetry_dev.CaptureBudget.from_client()
        _hook_response_close(inner, self._on_response_close)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = await self._inner.__anext__()
                except StopAsyncIteration:
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                if not self._record(chunk):
                    self._end(**self._partial(), error=_chunk_error(chunk))
                yield chunk
        finally:
            await self.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        if self._consume is None:
            self._consume = self._aiterate()
        return self._consume

    async def __anext__(self) -> Any:
        return await self.__aiter__().__anext__()

    async def __aenter__(self) -> _InstrumentedAsyncStream:
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc is not None:
            self._end(**self._partial(), error=exc)
        await self.close()

    def _partial(self) -> dict[str, Any]:
        return _chat_partial(self._states, self._usage, self._cost_usd, self._budget)

    def _finish(self) -> None:
        self._end(**self._partial())

    def _on_response_close(self) -> None:
        if not self._in_next:
            self._finish()

    async def close(self) -> None:
        try:
            await _close_async_stream(self._inner)
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, chunk: Any) -> bool:
        update = _record_chat_chunk(chunk, self._states, self._budget)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
                response_id=update.get("response_id"),
                response_model=update.get("response_model"),
            )
        if update.get("usage") is not None:
            self._usage = update["usage"]
        if update.get("cost_usd") is not None:
            self._cost_usd = update["cost_usd"]
        return _chunk_error(chunk) is None


class _ResponsesStreamState:
    def __init__(self) -> None:
        self.partial: dict[str, Any] = {}
        self.retained_output: Any | None = None
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._items: dict[int, dict[str, Any]] = {}
        self._content: dict[tuple[int, int], dict[str, Any]] = {}
        self._summary: dict[tuple[int, int], dict[str, Any]] = {}
        self._synthetic_events: list[dict[str, Any]] = []
        self._provider_events_truncated = False

    def _output_index(self, event: Any) -> int:
        return self._event_index(event, "output_index", "outputIndex")

    @staticmethod
    def _event_index(event: Any, snake_name: str, camel_name: str) -> int:
        value = _event_field(event, snake_name, camel_name)
        return value if type(value) is int and value >= 0 else 0

    def _output(self) -> list[dict[str, Any]]:
        return [
            *[value for _, value in sorted(self._items.items())],
            *self._synthetic_events,
        ]

    def _output_with_item(self, output_index: int, item: dict[str, Any]) -> list[dict[str, Any]]:
        items = {**self._items, output_index: item}
        return [*[value for _, value in sorted(items.items())], *self._synthetic_events]

    @staticmethod
    def _can_accept(budget: telemetry_dev.CaptureBudget, value: Any) -> bool:
        candidate = telemetry_dev.CaptureBudget(budget.max_bytes, budget.max_items)
        candidate.bytes_used = budget.bytes_used
        candidate.items_used = budget.items_used
        candidate.truncated = budget.truncated
        return candidate.accept(value)

    def _replace_output(self, output: Any) -> bool:
        replacement = telemetry_dev.CaptureBudget.from_client()
        if not replacement.accept(output):
            self._budget.truncated = True
            return False
        replacement.truncated = self._provider_events_truncated
        self._budget = replacement
        return True

    def _sync_output(self) -> None:
        self.retained_output = self._output()
        self.partial["output"] = self.retained_output

    def _hydrate_item(self, output_index: int, item: dict[str, Any]) -> dict[str, Any]:
        self._items[output_index] = item
        self._content = {key: part for key, part in self._content.items() if key[0] != output_index}
        self._summary = {key: part for key, part in self._summary.items() if key[0] != output_index}
        for index, part in enumerate(_sequence_items(item.get("content"))):
            if isinstance(part, dict):
                self._content[(output_index, index)] = cast(dict[str, Any], part)
        for index, part in enumerate(_sequence_items(item.get("summary"))):
            if isinstance(part, dict):
                self._summary[(output_index, index)] = cast(dict[str, Any], part)
        return item

    def _item(self, event: Any, event_type: str) -> dict[str, Any]:
        output_index = self._output_index(event)
        existing = self._items.get(output_index)
        if existing is not None:
            return existing
        reasoning = event_type.startswith("response.reasoning_")
        item: dict[str, Any] = {
            "id": _string(_event_field(event, "item_id", "itemId")) or f"output_{output_index}",
            "type": "reasoning" if reasoning else "message",
            "status": "in_progress",
            **({"summary": []} if reasoning else {"role": "assistant"}),
            "content": [],
        }
        self._items[output_index] = item
        return item

    def _sync_item(self, output_index: int, item: dict[str, Any]) -> None:
        content = [
            value
            for (item_index, _), value in sorted(self._content.items())
            if item_index == output_index
        ]
        summary = [
            value
            for (item_index, _), value in sorted(self._summary.items())
            if item_index == output_index
        ]
        if content:
            item["content"] = content
        if summary:
            item["summary"] = summary

    def _item_candidate(
        self,
        item: dict[str, Any],
        *,
        output_index: int,
        content_index: int | None = None,
        content: dict[str, Any] | None = None,
        summary_index: int | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = dict(item)
        if content_index is not None and content is not None:
            values = {
                index: value
                for (item_index, index), value in self._content.items()
                if item_index == output_index
            }
            values[content_index] = content
            candidate["content"] = [value for _, value in sorted(values.items())]
        if summary_index is not None and summary is not None:
            values = {
                index: value
                for (item_index, index), value in self._summary.items()
                if item_index == output_index
            }
            values[summary_index] = summary
            candidate["summary"] = [value for _, value in sorted(values.items())]
        return candidate

    def _record_event(self, event: Any) -> bool:
        event_type = _string(_field(event, "type"))
        if event_type is None:
            return False
        output_index = self._output_index(event)
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            raw_item = _field(event, "item")
            replacement = event_type.endswith(".done")
            preflight = telemetry_dev.CaptureBudget.from_client() if replacement else self._budget
            if not self._can_accept(preflight, raw_item):
                self._budget.truncated = True
                return False
            item = _response_native(raw_item)
            if not isinstance(item, dict):
                return False
            item = cast(dict[str, Any], item)
            if replacement:
                if not self._replace_output(self._output_with_item(output_index, item)):
                    return False
            elif not self._budget.accept(item):
                return False
            self._hydrate_item(output_index, item)
            return True
        if event_type in {"response.content_part.added", "response.content_part.done"}:
            event_part = _field(event, "part")
            replacement = event_type.endswith(".done")
            preflight = telemetry_dev.CaptureBudget.from_client() if replacement else self._budget
            if not self._can_accept(preflight, event_part):
                self._budget.truncated = True
                return False
            raw_part = _response_native(event_part)
            if not isinstance(raw_part, dict):
                return False
            part = cast(dict[str, Any], raw_part)
            existing = self._items.get(output_index)
            item = self._item(event, event_type)
            content_index = self._event_index(event, "content_index", "contentIndex")
            if replacement:
                candidate = self._item_candidate(
                    item,
                    output_index=output_index,
                    content_index=content_index,
                    content=part,
                )
                if not self._replace_output(self._output_with_item(output_index, candidate)):
                    if existing is None:
                        self._items.pop(output_index, None)
                    return False
            elif not self._budget.accept(part):
                return False
            self._content[(output_index, content_index)] = part
            self._sync_item(output_index, item)
            return True
        if event_type == "response.output_text.annotation.added":
            content_index = self._event_index(event, "content_index", "contentIndex")
            content_key = (output_index, content_index)
            part = self._content.get(content_key)
            annotations = _sequence_items(part.get("annotations")) if part is not None else []
            raw_annotation_index = _event_field(event, "annotation_index", "annotationIndex")
            if raw_annotation_index is not None and (
                isinstance(raw_annotation_index, bool)
                or not isinstance(raw_annotation_index, int)
                or raw_annotation_index < 0
                or raw_annotation_index > len(annotations)
            ):
                self._budget.truncated = True
                return False
            raw_annotation = _field(event, "annotation")
            if not self._can_accept(self._budget, raw_annotation):
                self._budget.truncated = True
                return False
            annotation = _response_native(raw_annotation)
            if annotation is None or not self._budget.accept(annotation):
                return False
            item = self._item(event, event_type)
            if part is None:
                part = cast(dict[str, Any], {"type": "output_text", "text": "", "annotations": []})
                self._content[content_key] = part
            if isinstance(raw_annotation_index, int):
                if raw_annotation_index == len(annotations):
                    annotations.append(annotation)
                else:
                    annotations[raw_annotation_index] = annotation
            else:
                annotations.append(annotation)
            part["annotations"] = annotations
            self._sync_item(output_index, item)
            return True
        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
        }:
            function_call = event_type.startswith("response.function_call_arguments")
            existing = self._items.get(output_index)
            item = existing
            if item is None:
                item = {
                    "id": _string(_event_field(event, "item_id", "itemId"))
                    or f"output_{output_index}",
                    "type": "function_call" if function_call else "custom_tool_call",
                    "status": "in_progress",
                    "arguments" if function_call else "input": "",
                }
                self._items[output_index] = item
            field = "arguments" if function_call else "input"
            done = event_type.endswith(".done")
            value = _string(_field(event, field) if done else _field(event, "delta"))
            if value is None:
                return False
            if done:
                candidate = {
                    **item,
                    field: value,
                    **(
                        {"name": _field(event, "name")} if _field(event, "name") is not None else {}
                    ),
                    "status": "completed",
                }
                if not self._replace_output(self._output_with_item(output_index, candidate)):
                    if existing is None:
                        self._items.pop(output_index, None)
                    return False
                self._items[output_index] = candidate
            else:
                if not self._budget.accept(value):
                    return False
                item[field] = f"{item.get(field, '')}{value}"
                name = _field(event, "name")
                if name is not None:
                    item["name"] = name
            return True
        if event_type in {
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            event_part = _field(event, "part")
            replacement = event_type.endswith(".done")
            preflight = telemetry_dev.CaptureBudget.from_client() if replacement else self._budget
            if not self._can_accept(preflight, event_part):
                self._budget.truncated = True
                return False
            raw_part = _response_native(event_part)
            if not isinstance(raw_part, dict):
                return False
            part = cast(dict[str, Any], raw_part)
            existing = self._items.get(output_index)
            item = self._item(event, event_type)
            summary_index = self._event_index(event, "summary_index", "summaryIndex")
            if replacement:
                candidate = self._item_candidate(
                    item,
                    output_index=output_index,
                    summary_index=summary_index,
                    summary=part,
                )
                if not self._replace_output(self._output_with_item(output_index, candidate)):
                    if existing is None:
                        self._items.pop(output_index, None)
                    return False
            elif not self._budget.accept(part):
                return False
            self._summary[(output_index, summary_index)] = part
            self._sync_item(output_index, item)
            return True
        if event_type in {
            "response.image_generation_call.partial_image",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.apply_patch_call_operation_diff.delta",
            "response.apply_patch_call_operation_diff.done",
            "response.fusion_call.in_progress",
            "response.fusion_call.completed",
            "response.fusion_call.analysis.in_progress",
            "response.fusion_call.analysis.completed",
            "response.fusion_call.panel.added",
            "response.fusion_call.panel.delta",
            "response.fusion_call.panel.reasoning.delta",
            "response.fusion_call.panel.completed",
            "response.fusion_call.panel.failed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.debug",
        }:
            if event_type == "response.debug":
                debug = _field(event, "debug")
                timings = _field(debug, "timings")
                sequence_number = _event_field(event, "sequence_number", "sequenceNumber")
                payload = {
                    "type": event_type,
                    **({"sequence_number": sequence_number} if sequence_number is not None else {}),
                    "debug": {} if timings is None else {"timings": _response_native(timings)},
                }
            else:
                payload = _response_native(event)
            synthetic = {
                "type": "telemetry.dev.response_stream_event",
                "event_type": event_type,
                "payload": payload,
            }
            if not self._budget.accept(synthetic):
                self._budget.truncated = True
                self._provider_events_truncated = True
                return False
            self._synthetic_events.append(synthetic)
            return True
        if event_type not in {
            "response.output_text.delta",
            "response.output_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.refusal.delta",
            "response.refusal.done",
        }:
            return False
        done = event_type.endswith(".done")
        refusal = event_type.startswith("response.refusal.")
        value = _string(
            _field(event, "refusal" if refusal else "text") if done else _field(event, "delta")
        )
        if value is None or (not done and not self._budget.accept(value)):
            return False
        summary_text = event_type.startswith("response.reasoning_summary_text.")
        existing = self._items.get(output_index)
        item = self._item(event, event_type)
        if summary_text:
            summary_index = self._event_index(event, "summary_index", "summaryIndex")
            summary_key = (output_index, summary_index)
            summary = self._summary.get(summary_key)
            if summary is None:
                summary = {"type": "summary_text", "text": ""}
                if not done:
                    self._summary[summary_key] = summary
                    self._sync_item(output_index, item)
            if done:
                part = {**summary, "text": value}
                candidate = self._item_candidate(
                    item,
                    output_index=output_index,
                    summary_index=summary_index,
                    summary=part,
                )
                if not self._replace_output(self._output_with_item(output_index, candidate)):
                    if existing is None:
                        self._items.pop(output_index, None)
                    return False
                self._summary[summary_key] = part
                self._sync_item(output_index, item)
            else:
                summary["text"] = f"{summary.get('text', '')}{value}"
            return True
        content_index = self._event_index(event, "content_index", "contentIndex")
        content_key = (output_index, content_index)
        part = self._content.get(content_key)
        if part is None:
            if refusal:
                new_part: dict[str, Any] = {"type": "refusal", "refusal": ""}
            else:
                new_part = {
                    "type": "reasoning_text"
                    if event_type.startswith("response.reasoning_text.")
                    else "output_text",
                    "text": "",
                    **(
                        {"annotations": []}
                        if event_type.startswith("response.output_text.")
                        else {}
                    ),
                }
            part = new_part
            if not done:
                self._content[content_key] = part
                self._sync_item(output_index, item)
        if done:
            replacement = {**part, "refusal" if refusal else "text": value}
            candidate = self._item_candidate(
                item,
                output_index=output_index,
                content_index=content_index,
                content=replacement,
            )
            if not self._replace_output(self._output_with_item(output_index, candidate)):
                if existing is None:
                    self._items.pop(output_index, None)
                return False
            self._content[content_key] = replacement
            self._sync_item(output_index, item)
        elif refusal:
            part["refusal"] = f"{part.get('refusal', '')}{value}"
        else:
            part["text"] = f"{part.get('text', '')}{value}"
        return True

    def record(self, event: Any) -> None:
        event = _response_event(event)
        response = _field(event, "response")
        if response is not None:
            fields = _responses_response(response, include_error=False, include_output=False)
            response_budget = telemetry_dev.CaptureBudget.from_client()
            raw_output = _field(response, "output")
            if raw_output is not None:
                if not self._can_accept(response_budget, raw_output):
                    self._budget.truncated = True
                else:
                    output = _response_native(raw_output)
                    response_items = cast(list[Any], output) if isinstance(output, list) else None
                    if response_items is None:
                        self._budget.truncated = True
                    else:
                        response_artifact = [*response_items, *self._synthetic_events]
                        if response_budget.accept(response_artifact):
                            self._items.clear()
                            self._content.clear()
                            self._summary.clear()
                            for index, item in enumerate(response_items):
                                if isinstance(item, dict):
                                    self._hydrate_item(index, cast(dict[str, Any], item))
                            self.retained_output = response_artifact
                            response_budget.truncated = self._provider_events_truncated
                            self._budget = response_budget
                        else:
                            self._budget.truncated = True
            self.partial = fields
            if self.retained_output is not None:
                self.partial["output"] = self.retained_output
        elif self._record_event(event):
            self._sync_output()
        _mark_capture_truncated(self.partial, self._budget)


class _InstrumentedResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._saw_first = False
        self._state = _ResponsesStreamState()
        self._consume: Iterator[Any] | None = None
        self._in_next = False
        _hook_response_close(inner, self._on_response_close)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    event = next(self._inner)
                except StopIteration:
                    break
                except BaseException as exc:
                    self._end(**self._state.partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event)
                yield event
        finally:
            self.close()

    def __iter__(self) -> Iterator[Any]:
        if self._consume is None:
            self._consume = self._iterate()
        return self._consume

    def __next__(self) -> Any:
        return next(self.__iter__())

    def __enter__(self) -> _InstrumentedResponsesStream:
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc is not None:
            self._end(**self._state.partial, error=exc)
        self.close()

    def _finish(self) -> None:
        self._end(**self._state.partial)

    def _on_response_close(self) -> None:
        if not self._in_next:
            self._finish()

    def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, event: Any) -> None:
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        self._state.record(event)
        event = _response_event(event)
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._state.partial)
        elif event_type == "response.failed":
            self._end(
                **self._state.partial, error=_response_failed_error(_field(event, "response"))
            )
        elif event_type == "response.incomplete":
            self._end(**self._state.partial)
        elif event_type == "error":
            self._end(**self._state.partial, error=_response_stream_error(event))


class _InstrumentedAsyncResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._saw_first = False
        self._state = _ResponsesStreamState()
        self._consume: AsyncIterator[Any] | None = None
        self._in_next = False
        _hook_response_close(inner, self._on_response_close)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    event = await self._inner.__anext__()
                except StopAsyncIteration:
                    break
                except BaseException as exc:
                    self._end(**self._state.partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event)
                yield event
        finally:
            await self.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        if self._consume is None:
            self._consume = self._aiterate()
        return self._consume

    async def __anext__(self) -> Any:
        return await self.__aiter__().__anext__()

    async def __aenter__(self) -> _InstrumentedAsyncResponsesStream:
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc is not None:
            self._end(**self._state.partial, error=exc)
        await self.close()

    def _finish(self) -> None:
        self._end(**self._state.partial)

    def _on_response_close(self) -> None:
        if not self._in_next:
            self._finish()

    async def close(self) -> None:
        try:
            await _close_async_stream(self._inner)
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, event: Any) -> None:
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        self._state.record(event)
        event = _response_event(event)
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._state.partial)
        elif event_type == "response.failed":
            self._end(
                **self._state.partial, error=_response_failed_error(_field(event, "response"))
            )
        elif event_type == "response.incomplete":
            self._end(**self._state.partial)
        elif event_type == "error":
            self._end(**self._state.partial, error=_response_stream_error(event))


def _start_span(
    params: Mapping[str, Any], mapper: RequestMapper
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float]:
    name, fields = mapper(params)
    handle = telemetry_dev.start_span(name, provider=_PROVIDER, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter()


def _wrap_sync(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        streaming = kwargs.get("stream") is True
        handle, end, started_at = _start_span(kwargs, request_mapper)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming and operation == "chat":
            return _InstrumentedStream(result, handle, started_at)
        if streaming and operation == "responses":
            return _InstrumentedResponsesStream(result, handle, started_at)
        end(**response_mapper(result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        streaming = kwargs.get("stream") is True
        handle, end, started_at = _start_span(kwargs, request_mapper)
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming and operation == "chat":
            return _InstrumentedAsyncStream(result, handle, started_at)
        if streaming and operation == "responses":
            return _InstrumentedAsyncResponsesStream(result, handle, started_at)
        end(**response_mapper(result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _patch_instance(
    resource: object,
    method: str,
    wrapper_factory: Callable[
        [Callable[..., Any], str, RequestMapper, ResponseMapper], Callable[..., Any]
    ],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
) -> None:
    current = getattr(resource, method)
    if getattr(current, _WRAPPED_ATTR, False):
        if method in vars(resource):
            return
        original = getattr(current, _ORIGINAL_ATTR, None)
        if original is None:
            return
        current = original.__get__(resource, type(resource))
    setattr(resource, method, wrapper_factory(current, operation, request_mapper, response_mapper))


def _patch_class(
    cls: type[Any],
    method: str,
    wrapper_factory: Callable[
        [Callable[..., Any], str, RequestMapper, ResponseMapper], Callable[..., Any]
    ],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
) -> None:
    original = getattr(cls, method)
    if getattr(original, _WRAPPED_ATTR, False):
        return
    _ORIGINALS.append((cls, method, original))
    setattr(cls, method, wrapper_factory(original, operation, request_mapper, response_mapper))


def wrap_open_router(client: _T) -> _T:
    if getattr(client, _WRAPPED_ATTR, False):
        return client
    _patch_instance(
        cast(Any, client).chat,
        "send",
        _wrap_sync,
        "chat",
        _chat_request,
        _chat_response,
    )
    _patch_instance(
        cast(Any, client).chat,
        "send_async",
        _wrap_async,
        "chat",
        _chat_request,
        _chat_response,
    )
    _patch_instance(
        cast(Any, client).responses,
        "send",
        _wrap_sync,
        "responses",
        _responses_request,
        _responses_response,
    )
    _patch_instance(
        cast(Any, client).responses,
        "send_async",
        _wrap_async,
        "responses",
        _responses_request,
        _responses_response,
    )
    _patch_instance(
        cast(Any, client).embeddings,
        "generate",
        _wrap_sync,
        "embeddings",
        _embeddings_request,
        _embeddings_response,
    )
    _patch_instance(
        cast(Any, client).embeddings,
        "generate_async",
        _wrap_async,
        "embeddings",
        _embeddings_request,
        _embeddings_response,
    )
    setattr(client, _WRAPPED_ATTR, True)
    return client


def instrument_openrouter() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        _patch_class(Chat, "send", _wrap_sync, "chat", _chat_request, _chat_response)
        _patch_class(Chat, "send_async", _wrap_async, "chat", _chat_request, _chat_response)
        _patch_class(
            Responses, "send", _wrap_sync, "responses", _responses_request, _responses_response
        )
        _patch_class(
            Responses,
            "send_async",
            _wrap_async,
            "responses",
            _responses_request,
            _responses_response,
        )
        _patch_class(
            Embeddings,
            "generate",
            _wrap_sync,
            "embeddings",
            _embeddings_request,
            _embeddings_response,
        )
        _patch_class(
            Embeddings,
            "generate_async",
            _wrap_async,
            "embeddings",
            _embeddings_request,
            _embeddings_response,
        )
        _installed = True


def uninstrument_openrouter() -> None:
    global _installed
    with _install_lock:
        while _ORIGINALS:
            cls, method, original = _ORIGINALS.pop()
            setattr(cls, method, original)
        _installed = False


__all__ = [
    "__version__",
    "instrument_openrouter",
    "uninstrument_openrouter",
    "wrap_open_router",
]
