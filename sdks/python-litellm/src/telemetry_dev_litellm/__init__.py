from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import wraps
from inspect import Parameter
from typing import Any, TypeVar, cast

import litellm
import telemetry_dev
from opentelemetry import trace

__version__ = "0.1.0"

RequestMapper = Callable[[tuple[Any, ...], Mapping[str, Any]], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]
_T = TypeVar("_T")

_WRAPPED_ATTR = "_telemetry_dev_litellm_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_litellm_original"
_ROUTER_WRAPPED_ATTR = "_telemetry_dev_litellm_router_wrapped"
_ORIGINALS: list[tuple[object, str, Any]] = []
_DROPIN_WRAPPERS: dict[str, tuple[Any, Callable[..., Any]]] = {}
_PENDING_CLOSE_TASKS: set[asyncio.Future[Any]] = set()
_installed = False
_install_lock = threading.Lock()

_PROVIDER_NAMES: dict[str, str] = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "azure_text": "azure.ai.openai",
    "azure_ai": "azure.ai.inference",
    "anthropic": "anthropic",
    "anthropic_text": "anthropic",
    "bedrock": "aws.bedrock",
    "vertex_ai": "gcp.vertex_ai",
    "vertex_ai_beta": "gcp.vertex_ai",
    "gemini": "gcp.gemini",
    "mistral": "mistral_ai",
    "groq": "groq",
    "deepseek": "deepseek",
    "xai": "x_ai",
    "cohere": "cohere",
    "cohere_chat": "cohere",
    "perplexity": "perplexity",
    "watsonx": "ibm.watsonx.ai",
    "watsonx_text": "ibm.watsonx.ai",
}


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return mapping.get(name)
    return getattr(value, name, None)


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[Any], value))
    return []


def _native(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
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


def _usage(fields: Mapping[str, int | float | None]) -> dict[str, int | float] | None:
    usage = {key: value for key, value in fields.items() if value is not None}
    return usage or None


def _stop_sequences(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    strings = [item for item in _sequence_items(value) if isinstance(item, str)]
    return strings or None


def _clean_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _provider_name(provider: Any) -> str | None:
    raw = _string(provider)
    if raw is None:
        return None
    return _PROVIDER_NAMES.get(raw, raw)


def _arg(args: tuple[Any, ...], kwargs: Mapping[str, Any], name: str, index: int) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def _stream_enabled(args: tuple[Any, ...], kwargs: Mapping[str, Any], index: int | None) -> bool:
    if kwargs.get("stream") is True:
        return True
    return index is not None and len(args) > index and args[index] is True


def _bound_kwargs(
    original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return kwargs
    mapped: dict[str, Any] = dict(kwargs)
    for name, value in bound.arguments.items():
        parameter = signature.parameters.get(name)
        if parameter is None:
            continue
        if parameter.kind is Parameter.VAR_KEYWORD and isinstance(value, Mapping):
            mapped.update(cast(Mapping[str, Any], value))
        elif parameter.kind is not Parameter.VAR_POSITIONAL:
            mapped[name] = value
    return mapped


def _stream_index(original: Callable[..., Any]) -> int | None:
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return None
    positional_index = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name == "stream":
                return positional_index
            positional_index += 1
    return None


def _resolve_provider(model_value: Any, kwargs: Mapping[str, Any]) -> tuple[str | None, str | None]:
    model = _string(model_value)
    if model is None:
        return None, None
    model_for_provider = _string(kwargs.get("deployment_id")) or model
    custom_llm_provider = kwargs.get("custom_llm_provider")
    if kwargs.get("azure") is True or kwargs.get("deployment_id") is not None:
        custom_llm_provider = "azure"
    try:
        resolved_model, provider, _dynamic_api_key, _api_base = litellm.get_llm_provider(
            model=model_for_provider,
            custom_llm_provider=custom_llm_provider,
            api_base=kwargs.get("api_base") or kwargs.get("base_url"),
        )
    except Exception:
        return model_for_provider, None
    return _string(resolved_model) or model_for_provider, _provider_name(provider)


def _metadata(value: Any) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def _request_metadata(kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _metadata(kwargs.get("litellm_metadata")) or _metadata(kwargs.get("metadata"))


def _output_type(response_format: Any) -> str | None:
    if response_format is None:
        return None
    if isinstance(response_format, type):
        return "json"
    kind = _string(_field(response_format, "type"))
    if kind in ("json_object", "json_schema"):
        return "json"
    return kind


def _completion_request(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    raw_model = _arg(args, kwargs, "model", 0)
    model, provider = _resolve_provider(raw_model, kwargs)
    return (
        f"chat {model or _string(raw_model) or 'unknown'}",
        {
            "type": "generation",
            "model": model or _string(raw_model),
            "provider": provider,
            "input": _arg(args, kwargs, "messages", 1),
            "temperature": _number(kwargs.get("temperature")),
            "top_p": _number(kwargs.get("top_p")),
            "top_k": _number(kwargs.get("top_k")),
            "max_tokens": _number(kwargs.get("max_completion_tokens"))
            or _number(kwargs.get("max_tokens")),
            "stop_sequences": _stop_sequences(kwargs.get("stop")),
            "seed": _number(kwargs.get("seed")),
            "frequency_penalty": _number(kwargs.get("frequency_penalty")),
            "presence_penalty": _number(kwargs.get("presence_penalty")),
            "output_type": _output_type(kwargs.get("response_format")),
            "metadata": _request_metadata(kwargs),
        },
    )


def _embedding_request(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    raw_model = _arg(args, kwargs, "model", 0)
    model, provider = _resolve_provider(raw_model, kwargs)
    return (
        f"embeddings {model or _string(raw_model) or 'unknown'}",
        {
            "type": "embedding",
            "model": model or _string(raw_model),
            "provider": provider,
            "input": _arg(args, kwargs, "input", 1),
            "metadata": _request_metadata(kwargs),
        },
    )


def _usage_from(raw: Any) -> dict[str, int | float] | None:
    prompt_details = _field(raw, "prompt_tokens_details")
    completion_details = _field(raw, "completion_tokens_details")
    cache_creation_tokens = _number(_field(prompt_details, "cache_creation_tokens"))
    if cache_creation_tokens is None:
        cache_creation_tokens = _number(_field(prompt_details, "cache_write_tokens"))
    return _usage(
        {
            "input_tokens": _number(_field(raw, "prompt_tokens")),
            "output_tokens": _number(_field(raw, "completion_tokens")),
            "total_tokens": _number(_field(raw, "total_tokens")),
            "cache_read_input_tokens": _number(_field(prompt_details, "cached_tokens")),
            "cache_creation_input_tokens": cache_creation_tokens,
            "reasoning_output_tokens": _number(_field(completion_details, "reasoning_tokens")),
        }
    )


def _hidden_params(response: Any) -> Any:
    return _field(response, "_hidden_params")


def _cost_from(response: Any) -> float | None:
    hidden_cost = _number(_field(_hidden_params(response), "response_cost"))
    if hidden_cost is not None and hidden_cost >= 0:
        return float(hidden_cost)
    try:
        cost = _number(litellm.completion_cost(completion_response=response))
    except Exception:
        return None
    if cost is None or cost < 0:
        return None
    return float(cost)


def _chat_output_message(message: Any) -> dict[str, Any]:
    if message is None:
        return {}
    native = _native(message)
    if not isinstance(native, dict):
        return cast(dict[str, Any], native)
    if _field(message, "content") is None:
        native["content"] = None
    return cast(dict[str, Any], native)


def _completion_response(response: Any) -> dict[str, Any]:
    choices = list(_field(response, "choices") or [])
    finish_reasons = [
        reason
        for choice in choices
        if (reason := _string(_field(choice, "finish_reason"))) is not None
    ]
    fields: dict[str, Any] = {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "output": [_chat_output_message(_field(choice, "message")) for choice in choices],
        "usage": _usage_from(_field(response, "usage")),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
        "attributes": (
            {"gen_ai.response.finish_reasons": finish_reasons} if len(finish_reasons) > 1 else None
        ),
    }
    return fields


def _embedding_response(response: Any) -> dict[str, Any]:
    raw_usage = _field(response, "usage")
    return {
        "response_model": _string(_field(response, "model")),
        "usage": _usage(
            {
                "input_tokens": _number(_field(raw_usage, "prompt_tokens")),
                "total_tokens": _number(_field(raw_usage, "total_tokens")),
            }
        ),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
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


def _safe_response_fields(mapper: ResponseMapper, response: Any) -> dict[str, Any]:
    try:
        return mapper(response)
    except Exception:
        return {}


def _safe_start_span(
    op: str, mapper: RequestMapper, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float, str | None]:
    fallback_type = "embedding" if op == "embeddings" else "generation"
    try:
        name, fields = mapper(args, kwargs)
    except Exception:
        name = f"{op} unknown"
        fields = {"type": fallback_type}
    request_provider = _provider_name(fields.get("provider"))
    handle = telemetry_dev.start_span(name, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter(), request_provider


def _provider_from_error(error: BaseException) -> str | None:
    return _provider_name(getattr(error, "llm_provider", None))


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await cast(Awaitable[Any], value)
    return value


def _run_sync_awaitable(value: Any) -> None:
    if not inspect.isawaitable(value):
        return
    awaitable = value
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(cast(Any, awaitable))
    else:
        task = asyncio.ensure_future(awaitable, loop=loop)
        _PENDING_CLOSE_TASKS.add(task)
        task.add_done_callback(_PENDING_CLOSE_TASKS.discard)


class _InstrumentedStream:
    def __init__(
        self,
        inner: Any,
        handle: telemetry_dev.SpanHandle,
        messages: Any,
        started_at: float,
        request_provider: str | None,
    ) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._messages = messages
        self._started_at = started_at
        self._request_provider = request_provider
        self._chunks: list[Any] = []
        self._saw_first = False
        self._finished = False
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._response_id: str | None = None
        self._response_model: str | None = None
        self._usage: dict[str, int | float] | None = None
        self._finish_reasons: dict[int, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._inner)
        except StopIteration:
            self.close()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            raise
        self._record(chunk)
        return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._inner.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            raise
        self._record(chunk)
        return chunk

    def __enter__(self) -> _InstrumentedStream:
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        self.close()
        exit_method = getattr(self._inner, "__exit__", None)
        if callable(exit_method):
            return bool(exit_method(exc_type, exc, tb))
        return False

    async def __aenter__(self) -> _InstrumentedStream:
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            await _maybe_await(enter())
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        await self.aclose()
        exit_method = getattr(self._inner, "__aexit__", None)
        if callable(exit_method):
            return bool(await _maybe_await(exit_method(exc_type, exc, tb)))
        return False

    def _record(self, chunk: Any) -> None:
        response_id = _string(_field(chunk, "id"))
        if self._response_id is None and response_id is not None:
            self._response_id = response_id
        response_model = _string(_field(chunk, "model"))
        if self._response_model is None and response_model is not None:
            self._response_model = response_model
        usage = _usage_from(_field(chunk, "usage"))
        if usage is not None:
            self._usage = usage
        for fallback_index, choice in enumerate(_sequence_items(_field(chunk, "choices"))):
            finish_reason = _string(_field(choice, "finish_reason"))
            if finish_reason is not None:
                choice_index = _number(_field(choice, "index"))
                self._finish_reasons[
                    int(choice_index) if choice_index is not None else fallback_index
                ] = finish_reason
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
                response_id=response_id,
                response_model=response_model,
            )
        if self._budget.accept(chunk):
            self._chunks.append(chunk)

    def _finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        fields: dict[str, Any] = {}
        rebuilt: Any = None
        try:
            stream_chunk_builder = cast(Callable[..., Any], cast(Any, litellm).stream_chunk_builder)
            rebuilt = stream_chunk_builder(self._chunks, messages=self._messages)
        except Exception:
            rebuilt = None
        if rebuilt is not None:
            fields = _safe_response_fields(_completion_response, rebuilt)
        if fields.get("usage") is None and self._usage is not None:
            fields["usage"] = self._usage
        if fields.get("finish_reason") is None and self._finish_reasons:
            finish_reasons = [self._finish_reasons[index] for index in sorted(self._finish_reasons)]
            fields["finish_reason"] = finish_reasons[0]
            if len(self._finish_reasons) > 1:
                attributes = fields.get("attributes")
                if not isinstance(attributes, dict):
                    attributes = {}
                    fields["attributes"] = attributes
                attributes["gen_ai.response.finish_reasons"] = finish_reasons
        if fields.get("response_id") is None and self._response_id is not None:
            fields["response_id"] = self._response_id
        if fields.get("response_model") is None and self._response_model is not None:
            fields["response_model"] = self._response_model
        if self._request_provider is not None:
            fields.pop("provider", None)
        elif fields.get("provider") is None:
            fields["provider"] = _provider_name(getattr(self._inner, "custom_llm_provider", None))
        if error is not None:
            fields["error"] = error
            if self._request_provider is None and fields.get("provider") is None:
                fields["provider"] = _provider_from_error(error)
        self._end(**fields)

    def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if callable(close):
            _run_sync_awaitable(close())
            return
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            _run_sync_awaitable(aclose())

    async def aclose(self) -> None:
        self._finish()
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await _maybe_await(aclose())
            return
        close = getattr(self._inner, "close", None)
        if callable(close):
            await _maybe_await(close())


def _wrap_sync(
    original: Callable[..., Any],
    op: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    stream_index: int | None = None,
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mapped_kwargs = _bound_kwargs(original, args, kwargs)
        handle, end, started_at, request_provider = _safe_start_span(
            op, request_mapper, args, mapped_kwargs
        )
        try:
            with trace.use_span(
                handle.span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                result = original(*args, **kwargs)
        except BaseException as exc:
            end(
                error=exc,
                provider=None if request_provider is not None else _provider_from_error(exc),
            )
            raise
        if op == "chat" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedStream(
                result,
                handle,
                messages=_arg(args, mapped_kwargs, "messages", 1),
                started_at=started_at,
                request_provider=request_provider,
            )
        try:
            fields = _safe_response_fields(response_mapper, result)
        except BaseException as exc:
            end(
                error=exc,
                provider=None if request_provider is not None else _provider_from_error(exc),
            )
            raise
        end(**fields)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async(
    original: Callable[..., Any],
    op: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    stream_index: int | None = None,
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        mapped_kwargs = _bound_kwargs(original, args, kwargs)
        handle, end, started_at, request_provider = _safe_start_span(
            op, request_mapper, args, mapped_kwargs
        )
        try:
            with trace.use_span(
                handle.span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                result = await original(*args, **kwargs)
        except BaseException as exc:
            end(
                error=exc,
                provider=None if request_provider is not None else _provider_from_error(exc),
            )
            raise
        if op == "chat" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedStream(
                result,
                handle,
                messages=_arg(args, mapped_kwargs, "messages", 1),
                started_at=started_at,
                request_provider=request_provider,
            )
        try:
            fields = _safe_response_fields(response_mapper, result)
        except BaseException as exc:
            end(
                error=exc,
                provider=None if request_provider is not None else _provider_from_error(exc),
            )
            raise
        end(**fields)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrapper_for(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    if name in {"completion", "acompletion"}:
        mapper = _completion_request
        response_mapper = _completion_response
        op = "chat"
    else:
        mapper = _embedding_request
        response_mapper = _embedding_response
        op = "embeddings"
    stream_index = _stream_index(original)
    if name in {"acompletion", "aembedding"}:
        return _wrap_async(original, op, mapper, response_mapper, stream_index=stream_index)
    return _wrap_sync(original, op, mapper, response_mapper, stream_index=stream_index)


def _dropin(name: str, *args: Any, **kwargs: Any) -> Any:
    current = getattr(litellm, name)
    if getattr(current, _WRAPPED_ATTR, False):
        return current(*args, **kwargs)
    cached = _DROPIN_WRAPPERS.get(name)
    if cached is None or cached[0] is not current:
        wrapped = _wrapper_for(name, current)
        _DROPIN_WRAPPERS[name] = (current, wrapped)
    else:
        wrapped = cached[1]
    return wrapped(*args, **kwargs)


def completion(*args: Any, **kwargs: Any) -> Any:
    return _dropin("completion", *args, **kwargs)


async def acompletion(*args: Any, **kwargs: Any) -> Any:
    return await _dropin("acompletion", *args, **kwargs)


def embedding(*args: Any, **kwargs: Any) -> Any:
    return _dropin("embedding", *args, **kwargs)


async def aembedding(*args: Any, **kwargs: Any) -> Any:
    return await _dropin("aembedding", *args, **kwargs)


def _patch_litellm_function(name: str) -> None:
    current = getattr(litellm, name)
    if getattr(current, _WRAPPED_ATTR, False):
        return
    _ORIGINALS.append((litellm, name, current))
    setattr(litellm, name, _wrapper_for(name, current))


def instrument_litellm() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        for name in ("completion", "acompletion", "embedding", "aembedding"):
            _patch_litellm_function(name)
        _installed = True


def uninstrument_litellm() -> None:
    global _installed
    with _install_lock:
        while _ORIGINALS:
            target, name, original = _ORIGINALS.pop()
            current = getattr(target, name)
            if getattr(current, _ORIGINAL_ATTR, None) is original:
                setattr(target, name, original)
        _installed = False


def _patch_router_method(router: object, name: str) -> None:
    current = getattr(router, name)
    if getattr(current, _WRAPPED_ATTR, False):
        return
    wrapped = _wrapper_for(name, current)
    setattr(router, name, wrapped)


def wrap_router(router: _T) -> _T:
    if getattr(router, _ROUTER_WRAPPED_ATTR, False):
        return router
    for name in ("completion", "acompletion", "embedding", "aembedding"):
        if hasattr(router, name):
            _patch_router_method(cast(object, router), name)
    setattr(router, _ROUTER_WRAPPED_ATTR, True)
    return router


__all__ = [
    "__version__",
    "acompletion",
    "aembedding",
    "completion",
    "embedding",
    "instrument_litellm",
    "uninstrument_litellm",
    "wrap_router",
]
