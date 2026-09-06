from __future__ import annotations

import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

import anthropic
import telemetry_dev
from anthropic.resources.messages import AsyncMessages, Messages

__version__ = "0.1.0"

ProviderResolver = Callable[[object | None], str]
RequestMapper = Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]

_WRAPPED_ATTR = "_telemetry_dev_anthropic_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_anthropic_original"
_ORIGINALS: list[tuple[type[Any], str, Any]] = []
_installed = False
_install_lock = threading.Lock()
_T = TypeVar("_T")


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return mapping.get(name)
    return getattr(value, name, None)


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[Any], value))
    return []


def _request_iterable(value: Any) -> Any:
    if isinstance(value, str | bytes | bytearray):
        return value
    if callable(getattr(value, "model_dump", None)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _request_iterable(item) for key, item in mapping.items()}
    if isinstance(value, Iterable):
        return [_request_iterable(item) for item in cast(Iterable[Any], value)]
    return value


def _normalize_request_params(params: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for key in ("messages", "system", "tools"):
        if key in normalized:
            normalized[key] = _request_iterable(normalized[key])
    return normalized


def _raw_response_requested(params: Mapping[str, Any]) -> bool:
    extra_headers = params.get("extra_headers")
    if not isinstance(extra_headers, Mapping):
        return False
    headers = cast(Mapping[str, Any], extra_headers)
    return headers.get("X-Stainless-Raw-Response") in {"true", "raw", "stream"}


def _native(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _native(item) for key, item in mapping.items() if item is not None}
    sequence = _sequence_items(value)
    if sequence:
        return [_native(item) for item in sequence]
    return value


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _usage(fields: dict[str, int | float | None]) -> dict[str, int | float] | None:
    usage = {key: value for key, value in fields.items() if value is not None}
    return usage or None


def _merge_usage(
    current: dict[str, int | float] | None, incoming: dict[str, int | float] | None
) -> dict[str, int | float] | None:
    if incoming is None:
        return current
    merged = dict(current or {})
    merged.update(incoming)
    return merged or None


def _stop_sequences(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    strings = [item for item in _sequence_items(value) if isinstance(item, str)]
    return strings or None


def _messages_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    input_value: Any
    if "tools" in params or "tool_choice" in params:
        input_value = {"messages": _native(params.get("messages"))}
        tools = _native(params.get("tools"))
        tool_choice = _native(params.get("tool_choice"))
        if tools is not None:
            input_value["tools"] = tools
        if tool_choice is not None:
            input_value["tool_choice"] = tool_choice
    else:
        input_value = _native(params.get("messages"))
    return (
        f"chat {model or 'unknown'}",
        {
            "type": "generation",
            "model": model,
            "input": input_value,
            "system_instructions": _native(params.get("system")),
            "temperature": _number(params.get("temperature")),
            "top_p": _number(params.get("top_p")),
            "top_k": _number(params.get("top_k")),
            "max_tokens": _number(params.get("max_tokens")),
            "stop_sequences": _stop_sequences(params.get("stop_sequences")),
        },
    )


def _messages_usage(raw: Any) -> dict[str, int | float] | None:
    output_details = _field(raw, "output_tokens_details")
    return _usage(
        {
            "input_tokens": _number(_field(raw, "input_tokens")),
            "output_tokens": _number(_field(raw, "output_tokens")),
            "cache_read_input_tokens": _number(_field(raw, "cache_read_input_tokens")),
            "cache_creation_input_tokens": _number(_field(raw, "cache_creation_input_tokens")),
            "reasoning_output_tokens": _number(_field(output_details, "thinking_tokens")),
        }
    )


def _messages_response(response: Any) -> dict[str, Any]:
    content = _field(response, "content")
    role = _string(_field(response, "role")) or "assistant"
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "finish_reason": _string(_field(response, "stop_reason")),
        "output": [{"role": role, "content": _native(content)}] if content is not None else None,
        "usage": _messages_usage(_field(response, "usage")),
    }


def _is_anthropic_class(client: object | None, name: str) -> bool:
    cls = getattr(anthropic, name, None)
    return isinstance(cls, type) and isinstance(client, cls)


def _provider_for_client(client: object | None) -> str:
    if _is_anthropic_class(client, "AnthropicBedrock") or _is_anthropic_class(
        client, "AsyncAnthropicBedrock"
    ):
        return "aws.bedrock"
    if _is_anthropic_class(client, "AnthropicVertex") or _is_anthropic_class(
        client, "AsyncAnthropicVertex"
    ):
        return "gcp.vertex_ai"
    return "anthropic"


def _provider_for_resource(resource: object | None) -> str:
    return _provider_for_client(getattr(resource, "_client", None))


def _clean_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _end_once(handle: telemetry_dev.SpanHandle) -> Callable[..., None]:
    ended = False

    def end(**fields: Any) -> None:
        nonlocal ended
        if ended:
            return
        ended = True
        handle.end(**_clean_fields(fields))

    return end


class _StreamState:
    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.tool_json: dict[int, str] = {}
        self.usage: dict[str, int | float] | None = None
        self.finish_reason: str | None = None
        self.budget = telemetry_dev.CaptureBudget.from_client()


def _parse_tool_input(raw: str) -> Any:
    if raw == "":
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _finalize_block(index: int, block: Mapping[str, Any], state: _StreamState) -> dict[str, Any]:
    raw_input = state.tool_json.get(index)
    if raw_input is None:
        return dict(block)
    finalized = dict(block)
    if raw_input == "" and "input" in finalized:
        return finalized
    parsed_input = _parse_tool_input(raw_input)
    existing_input = finalized.get("input")
    if isinstance(existing_input, Mapping) and isinstance(parsed_input, Mapping):
        finalized["input"] = {**existing_input, **parsed_input}
    else:
        finalized["input"] = parsed_input
    return finalized


def _stream_output(state: _StreamState) -> list[dict[str, Any]] | None:
    if not state.blocks:
        return None
    return [
        {
            "role": "assistant",
            "content": [
                _finalize_block(index, block, state)
                for index, block in sorted(state.blocks.items())
            ],
        }
    ]


def _stream_partial(state: _StreamState) -> dict[str, Any]:
    return {
        "output": _stream_output(state),
        "usage": state.usage,
        "finish_reason": state.finish_reason,
    }


def _append_string(target: dict[str, Any], key: str, value: Any) -> None:
    text = _string(value)
    if text is None:
        return
    target[key] = f"{_string(target.get(key)) or ''}{text}"


def _append_item(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    existing = target.get(key)
    items: list[Any] = (
        list(cast(Sequence[Any], existing))
        if isinstance(existing, Sequence) and not isinstance(existing, str)
        else []
    )
    items.append(_native(value))
    target[key] = items


def _record_content_block_start(event: Any, state: _StreamState) -> None:
    index = _field(event, "index")
    block_index = index if isinstance(index, int) else len(state.blocks)
    content_block = _native(_field(event, "content_block"))
    if not state.budget.accept(content_block):
        return
    block_type = _string(_field(content_block, "type"))
    if block_type == "text":
        state.blocks[block_index] = {
            "type": "text",
            "text": _string(_field(content_block, "text")) or "",
        }
        return
    if block_type in {"tool_use", "server_tool_use"}:
        state.blocks[block_index] = (
            content_block if isinstance(content_block, dict) else {"type": block_type}
        )
        state.tool_json[block_index] = ""
        return
    if block_type == "thinking":
        state.blocks[block_index] = {
            "type": "thinking",
            "thinking": _string(_field(content_block, "thinking")) or "",
        }
        return
    state.blocks[block_index] = (
        content_block if isinstance(content_block, dict) else {"type": block_type}
    )


def _block_for_delta(index: int, delta_type: str | None, state: _StreamState) -> dict[str, Any]:
    if index in state.blocks:
        return state.blocks[index]
    if delta_type == "input_json_delta":
        state.blocks[index] = {"type": "tool_use"}
        state.tool_json[index] = ""
    elif delta_type == "thinking_delta":
        state.blocks[index] = {"type": "thinking", "thinking": ""}
    else:
        state.blocks[index] = {"type": "text", "text": ""}
    return state.blocks[index]


def _record_content_block_delta(event: Any, state: _StreamState) -> None:
    index = _field(event, "index")
    block_index = index if isinstance(index, int) else 0
    delta = _native(_field(event, "delta"))
    if not state.budget.accept(delta):
        return
    delta_type = _string(_field(delta, "type"))
    block = _block_for_delta(block_index, delta_type, state)
    if delta_type == "input_json_delta":
        state.tool_json[block_index] = (
            f"{state.tool_json.get(block_index, '')}{_string(_field(delta, 'partial_json')) or ''}"
        )
    elif delta_type == "text_delta":
        _append_string(block, "text", _field(delta, "text"))
    elif delta_type == "citations_delta":
        _append_item(block, "citations", _field(delta, "citation"))
    elif delta_type == "thinking_delta":
        _append_string(block, "thinking", _field(delta, "thinking"))
    elif delta_type == "signature_delta" and _field(delta, "signature") is not None:
        block["signature"] = _field(delta, "signature")


def _record_stream_event(event: Any, state: _StreamState) -> dict[str, Any]:
    event_type = _string(_field(event, "type"))
    update: dict[str, Any] = {}
    if event_type == "message_start":
        message = _field(event, "message")
        update["response_id"] = _string(_field(message, "id"))
        update["response_model"] = _string(_field(message, "model"))
        state.usage = _merge_usage(state.usage, _messages_usage(_field(message, "usage")))
    elif event_type == "content_block_start":
        _record_content_block_start(event, state)
    elif event_type == "content_block_delta":
        _record_content_block_delta(event, state)
    elif event_type == "message_delta":
        delta = _field(event, "delta")
        state.finish_reason = _string(_field(delta, "stop_reason")) or state.finish_reason
        state.usage = _merge_usage(state.usage, _messages_usage(_field(event, "usage")))
    return update


class _InstrumentedStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._state = _StreamState()
        self._saw_first = False
        self._consume: Iterator[Any] | None = None

    def _update_first(self, update: Mapping[str, Any]) -> None:
        if self._saw_first:
            return
        self._saw_first = True
        self._handle.update(
            time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
            response_id=update.get("response_id"),
            response_model=update.get("response_model"),
        )

    def finish(self, error: BaseException | None = None) -> None:
        fields = _stream_partial(self._state)
        if error is not None:
            fields["error"] = error
        self._end(**fields)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                try:
                    event = next(self._inner)
                except StopIteration:
                    break
                except BaseException as exc:
                    self.finish(exc)
                    raise
                try:
                    update = _record_stream_event(event, self._state)
                except BaseException as exc:
                    client = telemetry_dev.get_client()
                    if client is not None:
                        client.report("failed to map Anthropic stream event", exc)
                    update = {}
                self._update_first(update)
                yield event
        finally:
            self.close()

    def __iter__(self) -> Iterator[Any]:
        return self._iterate()

    def __next__(self) -> Any:
        if self._consume is None:
            self._consume = self._iterate()
        return next(self._consume)

    def __enter__(self) -> _InstrumentedStream:
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc is not None:
            self.finish(exc)
        self.close()

    def close(self) -> None:
        self.finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InstrumentedAsyncStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._state = _StreamState()
        self._saw_first = False
        self._consume: AsyncIterator[Any] | None = None

    def _update_first(self, update: Mapping[str, Any]) -> None:
        if self._saw_first:
            return
        self._saw_first = True
        self._handle.update(
            time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
            response_id=update.get("response_id"),
            response_model=update.get("response_model"),
        )

    def finish(self, error: BaseException | None = None) -> None:
        fields = _stream_partial(self._state)
        if error is not None:
            fields["error"] = error
        self._end(**fields)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                try:
                    event = await self._inner.__anext__()
                except StopAsyncIteration:
                    break
                except BaseException as exc:
                    self.finish(exc)
                    raise
                try:
                    update = _record_stream_event(event, self._state)
                except BaseException as exc:
                    client = telemetry_dev.get_client()
                    if client is not None:
                        client.report("failed to map Anthropic stream event", exc)
                    update = {}
                self._update_first(update)
                yield event
        finally:
            await self.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiterate()

    async def __anext__(self) -> Any:
        if self._consume is None:
            self._consume = self._aiterate()
        return await self._consume.__anext__()

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
            self.finish(exc)
        await self.close()

    async def close(self) -> None:
        self.finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InstrumentedMessageStreamManager:
    def __init__(self, inner: Any, name: str, fields: Mapping[str, Any], provider: str) -> None:
        self._inner = inner
        self._name = name
        self._fields = fields
        self._provider = provider
        self._proxy: _InstrumentedStream | None = None

    def __enter__(self) -> Any:
        handle = telemetry_dev.start_span(
            self._name, provider=self._provider, **_clean_fields(self._fields)
        )
        end = _end_once(handle)
        started_at = time.perf_counter()
        try:
            message_stream = self._inner.__enter__()
        except BaseException as exc:
            end(error=exc)
            raise
        raw_stream = getattr(message_stream, "_raw_stream", None)
        proxy = _InstrumentedStream(raw_stream, handle, started_at)
        message_stream._raw_stream = proxy
        self._proxy = proxy
        return message_stream

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> Any:
        if exc is not None and self._proxy is not None:
            self._proxy.finish(exc)
        result = self._inner.__exit__(exc_type, exc, tb)
        if exc is None and self._proxy is not None:
            self._proxy.finish()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InstrumentedAsyncMessageStreamManager:
    def __init__(self, inner: Any, name: str, fields: Mapping[str, Any], provider: str) -> None:
        self._inner = inner
        self._name = name
        self._fields = fields
        self._provider = provider
        self._proxy: _InstrumentedAsyncStream | None = None

    async def __aenter__(self) -> Any:
        handle = telemetry_dev.start_span(
            self._name, provider=self._provider, **_clean_fields(self._fields)
        )
        end = _end_once(handle)
        started_at = time.perf_counter()
        try:
            message_stream = await self._inner.__aenter__()
        except BaseException as exc:
            end(error=exc)
            raise
        raw_stream = getattr(message_stream, "_raw_stream", None)
        proxy = _InstrumentedAsyncStream(raw_stream, handle, started_at)
        message_stream._raw_stream = proxy
        self._proxy = proxy
        return message_stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Any:
        if exc is not None and self._proxy is not None:
            self._proxy.finish(exc)
        result = await self._inner.__aexit__(exc_type, exc, tb)
        if exc is None and self._proxy is not None:
            self._proxy.finish()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _start_span(
    params: Mapping[str, Any], mapper: RequestMapper, provider: str
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float]:
    name, fields = mapper(params)
    handle = telemetry_dev.start_span(name, provider=provider, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter()


def _wrap_sync(
    original: Callable[..., Any],
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs = _normalize_request_params(kwargs)
        if _raw_response_requested(call_kwargs):
            return original(*args, **call_kwargs)
        streaming = call_kwargs.get("stream") is True
        handle, end, started_at = _start_span(call_kwargs, request_mapper, provider(resource))
        try:
            result = original(*args, **call_kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming:
            return _InstrumentedStream(result, handle, started_at)
        end(**response_mapper(result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async(
    original: Callable[..., Any],
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs = _normalize_request_params(kwargs)
        if _raw_response_requested(call_kwargs):
            return await original(*args, **call_kwargs)
        streaming = call_kwargs.get("stream") is True
        handle, end, started_at = _start_span(call_kwargs, request_mapper, provider(resource))
        try:
            result = await original(*args, **call_kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming:
            return _InstrumentedAsyncStream(result, handle, started_at)
        end(**response_mapper(result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_stream_manager_sync(
    original: Callable[..., Any], request_mapper: RequestMapper, provider: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs = _normalize_request_params(kwargs)
        name, fields = request_mapper(call_kwargs)
        manager = original(*args, **call_kwargs)
        return _InstrumentedMessageStreamManager(manager, name, fields, provider(resource))

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_stream_manager_async(
    original: Callable[..., Any], request_mapper: RequestMapper, provider: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs = _normalize_request_params(kwargs)
        name, fields = request_mapper(call_kwargs)
        manager = original(*args, **call_kwargs)
        return _InstrumentedAsyncMessageStreamManager(manager, name, fields, provider(resource))

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _own_method(resource: object, name: str) -> bool:
    namespace = getattr(resource, "__dict__", {})
    return isinstance(namespace, Mapping) and name in namespace


def _original_method(current: Any, resource: object) -> Any:
    function = getattr(current, "__func__", current)
    original = getattr(function, _ORIGINAL_ATTR, None)
    if original is None:
        return current
    bind = getattr(original, "__get__", None)
    return bind(resource, type(resource)) if bind is not None else original


def _patch_instance_create(resource: object, async_resource: bool, provider_name: str) -> None:
    resource_any: Any = resource
    current = resource_any.create
    if getattr(current, _WRAPPED_ATTR, False) and _own_method(resource, "create"):
        return
    original = _original_method(current, resource)
    factory = _wrap_async if async_resource else _wrap_sync
    wrapped = factory(original, _messages_request, _messages_response, lambda _: provider_name)
    resource_any.create = wrapped


def _patch_instance_stream(resource: object, async_resource: bool, provider_name: str) -> None:
    resource_any: Any = resource
    current = resource_any.stream
    if getattr(current, _WRAPPED_ATTR, False) and _own_method(resource, "stream"):
        return
    original = _original_method(current, resource)
    factory = _wrap_stream_manager_async if async_resource else _wrap_stream_manager_sync
    wrapped = factory(original, _messages_request, lambda _: provider_name)
    resource_any.stream = wrapped


def _patch_class_create(cls: type[Any], async_resource: bool) -> None:
    original = cls.create
    if getattr(original, _WRAPPED_ATTR, False):
        return
    _ORIGINALS.append((cls, "create", original))
    factory = _wrap_async if async_resource else _wrap_sync
    cls.create = factory(original, _messages_request, _messages_response, _provider_for_resource)


def _patch_class_stream(cls: type[Any], async_resource: bool) -> None:
    original = cls.stream
    if getattr(original, _WRAPPED_ATTR, False):
        return
    _ORIGINALS.append((cls, "stream", original))
    factory = _wrap_stream_manager_async if async_resource else _wrap_stream_manager_sync
    cls.stream = factory(original, _messages_request, _provider_for_resource)


def wrap_anthropic(client: _T) -> _T:
    if getattr(client, _WRAPPED_ATTR, False):
        return client
    client_any: Any = client
    messages = client_any.messages
    async_resource = isinstance(messages, AsyncMessages)
    provider_name = _provider_for_client(cast(object, client))
    _patch_instance_create(messages, async_resource, provider_name)
    _patch_instance_stream(messages, async_resource, provider_name)
    setattr(client, _WRAPPED_ATTR, True)
    return client


def instrument_anthropic() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        _patch_class_create(Messages, async_resource=False)
        _patch_class_stream(Messages, async_resource=False)
        _patch_class_create(AsyncMessages, async_resource=True)
        _patch_class_stream(AsyncMessages, async_resource=True)
        _installed = True


def uninstrument_anthropic() -> None:
    global _installed
    with _install_lock:
        while _ORIGINALS:
            cls, method, original = _ORIGINALS.pop()
            setattr(cls, method, original)
        _installed = False


__all__ = ["__version__", "instrument_anthropic", "uninstrument_anthropic", "wrap_anthropic"]
