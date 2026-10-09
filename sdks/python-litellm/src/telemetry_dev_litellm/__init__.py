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

__version__ = "0.1.4"

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


class _CaptureLimit(Exception):
    pass


def _bounded_utf8_size(value: str, limit: int) -> int:
    size = 0
    for character in value:
        code = ord(character)
        if code <= 0x7F:
            size += 1
        elif code <= 0x7FF:
            size += 2
        elif 0xD800 <= code <= 0xDFFF:
            return max(limit + 1, 1)
        elif code <= 0xFFFF:
            size += 3
        else:
            size += 4
        if size > limit:
            return size
    return size


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


def _sequence_length(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(cast(Sequence[Any], value))
    return None


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


def _bounded_responses_native(
    value: Any,
    parent_type: str | None = None,
    budget: telemetry_dev.CaptureBudget | None = None,
) -> tuple[Any | None, telemetry_dev.CaptureBudget]:
    active_budget = budget or telemetry_dev.CaptureBudget.from_client()
    ancestors: set[int] = set()

    def reserve(byte_count: int, item_count: int = 1) -> None:
        if (
            active_budget.bytes_used + byte_count > active_budget.max_bytes
            or active_budget.items_used + item_count > active_budget.max_items
        ):
            active_budget.truncated = True
            raise _CaptureLimit
        active_budget.bytes_used += byte_count
        active_budget.items_used += item_count

    def convert(item: Any, item_type: str | None, depth: int) -> Any:
        if depth > 32:
            active_budget.truncated = True
            raise _CaptureLimit
        if isinstance(item, bytes | bytearray | memoryview):
            return None
        if item is None or isinstance(item, bool | int | float):
            reserve(16)
            return item
        if isinstance(item, str):
            reserve(
                16
                + _bounded_utf8_size(item, active_budget.max_bytes - active_budget.bytes_used - 16)
            )
            return item
        item_id = id(item)
        if item_id in ancestors:
            reserve(16)
            return None
        ancestors.add(item_id)
        try:
            if isinstance(item, Mapping):
                source = cast(Mapping[Any, Any], item)
            else:
                attributes = getattr(item, "__dict__", None)
                source = (
                    cast(Mapping[Any, Any], attributes) if isinstance(attributes, Mapping) else None
                )
            if source is not None:
                reserve(16)
                current_type = _string(source.get("type")) or item_type
                result: dict[str, Any] = {}
                for raw_key, child in source.items():
                    if child is None:
                        continue
                    key = str(raw_key)
                    binary = (
                        key in {"b64_json", "file_data", "partial_image_b64"}
                        or (
                            key == "image_url"
                            and isinstance(child, str)
                            and child[:5].lower() == "data:"
                        )
                        or (
                            key in {"data", "audio"}
                            and current_type in {"audio", "input_audio", "output_audio"}
                        )
                        or (key == "result" and current_type == "image_generation_call")
                    )
                    if binary:
                        continue
                    reserve(
                        16
                        + _bounded_utf8_size(
                            key,
                            active_budget.max_bytes - active_budget.bytes_used - 16,
                        )
                    )
                    result[key] = convert(child, current_type, depth + 1)
                return result
            if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
                sequence = cast(Sequence[Any], item)
                reserve(16)
                return [convert(child, item_type, depth + 1) for child in sequence]
            reserve(16)
            return None
        finally:
            ancestors.remove(item_id)

    try:
        return convert(value, parent_type, 0), active_budget
    except _CaptureLimit:
        return None, active_budget


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


def _reports_tokens(usage: Mapping[str, int | float] | None) -> bool:
    return usage is not None and any(value > 0 for value in usage.values())


def _is_zero_usage(usage: Mapping[str, int | float] | None) -> bool:
    return (
        usage is not None
        and usage.get("input_tokens") == 0
        and usage.get("output_tokens") == 0
        and not _reports_tokens(usage)
    )


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


def _rerank_request(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_model = _arg(args, kwargs, "model", 0)
    model, provider = _resolve_provider(raw_model, kwargs)
    input_value = _clean_fields(
        {
            "query": _arg(args, kwargs, "query", 1),
            "documents": _arg(args, kwargs, "documents", 2),
            "top_n": kwargs.get("top_n"),
            "rank_fields": kwargs.get("rank_fields"),
            "return_documents": kwargs.get("return_documents"),
            "max_chunks_per_doc": kwargs.get("max_chunks_per_doc"),
            "max_tokens_per_doc": kwargs.get("max_tokens_per_doc"),
        }
    )
    client = telemetry_dev.get_client()
    captured_input: Any = None
    input_budget: telemetry_dev.CaptureBudget | None = None
    if client is not None and client.capture_input:
        captured_input, input_budget = _bounded_responses_native(input_value)
    attributes: dict[str, Any] = {"gen_ai.operation.name": "rerank"}
    if input_budget is not None and input_budget.truncated:
        attributes["telemetry.dev.capture.truncated"] = True
    return (
        f"rerank {model or _string(raw_model) or 'unknown'}",
        {
            "type": "span",
            "model": model or _string(raw_model),
            "provider": provider,
            "input": captured_input,
            "metadata": _request_metadata(kwargs),
            "attributes": attributes,
        },
    )


def _responses_request(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    raw_model = _arg(args, kwargs, "model", 1)
    model, provider = _resolve_provider(raw_model, kwargs)
    client = telemetry_dev.get_client()
    capture_input = client is not None and client.capture_input
    input_value: Any = None
    input_budget: telemetry_dev.CaptureBudget | None = None
    if capture_input:
        input_value, input_budget = _bounded_responses_native(_arg(args, kwargs, "input", 0))
    attributes: dict[str, Any] = {"gen_ai.operation.name": "chat"}
    if input_budget is not None and input_budget.truncated:
        attributes["telemetry.dev.capture.truncated"] = True
    return (
        f"chat {model or _string(raw_model) or 'unknown'}",
        {
            "type": "generation",
            "model": model or _string(raw_model),
            "provider": provider,
            "input": input_value,
            "system_instructions": kwargs.get("instructions"),
            "temperature": _number(kwargs.get("temperature")),
            "top_p": _number(kwargs.get("top_p")),
            "max_tokens": _number(kwargs.get("max_output_tokens")),
            "output_type": _output_type(kwargs.get("text_format"))
            or _output_type(kwargs.get("text")),
            "metadata": _request_metadata(kwargs),
            "attributes": attributes,
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


def _responses_usage(raw: Any) -> dict[str, int | float] | None:
    input_details = _field(raw, "input_tokens_details")
    output_details = _field(raw, "output_tokens_details")
    fields: dict[str, int | float | None] = {
        "input_tokens": _number(_field(raw, "input_tokens")),
        "output_tokens": _number(_field(raw, "output_tokens")),
        "total_tokens": _number(_field(raw, "total_tokens")),
        "cache_read_input_tokens": _number(_field(input_details, "cached_tokens")),
        "reasoning_output_tokens": _number(_field(output_details, "reasoning_tokens")),
    }
    for modality in ("text", "image", "audio"):
        fields[f"{modality}_input_tokens"] = _number(_field(input_details, f"{modality}_tokens"))
        fields[f"{modality}_output_tokens"] = _number(_field(output_details, f"{modality}_tokens"))
    return _usage(fields)


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


def _rerank_response(response: Any) -> dict[str, Any]:
    meta = _field(response, "meta")
    billed_units = _field(meta, "billed_units")
    tokens = _field(meta, "tokens")
    raw_results = _field(response, "results")
    results, capture_budget = _bounded_responses_native(raw_results)
    input_tokens = _number(_field(tokens, "input_tokens"))
    output_tokens = _number(_field(tokens, "output_tokens"))
    total_tokens = _number(_field(billed_units, "total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "response_id": _string(_field(response, "id")),
        "output": results,
        "usage": _usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        ),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
        "attributes": {"telemetry.dev.capture.truncated": True}
        if capture_budget.truncated
        else None,
        "metadata": _clean_fields(
            {
                "result_count": _sequence_length(raw_results),
                "search_units": _number(_field(billed_units, "search_units")),
            }
        ),
    }


def _response_error(response: Any) -> RuntimeError | None:
    if _string(_field(response, "status")) != "failed" and _field(response, "error") is None:
        return None
    raw_error = _field(response, "error")
    message = _string(_field(raw_error, "message")) or "LiteLLM Responses API request failed"
    return RuntimeError(message)


def _responses_response(response: Any) -> dict[str, Any]:
    status = _string(_field(response, "status"))
    output, output_budget = _bounded_responses_native(_field(response, "output"))
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "output": output,
        "usage": _responses_usage(_field(response, "usage")),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
        "finish_reason": status,
        "attributes": {
            **({"gen_ai.response.status": status} if status is not None else {}),
            **({"telemetry.dev.capture.truncated": True} if output_budget.truncated else {}),
        }
        or None,
        "error": _response_error(response),
    }


def _responses_response_without_output(response: Any) -> dict[str, Any]:
    status = _string(_field(response, "status"))
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "usage": _responses_usage(_field(response, "usage")),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
        "finish_reason": status,
        "attributes": {"gen_ai.response.status": status} if status is not None else None,
        "error": _response_error(response),
    }


def _rerank_response_without_output(response: Any) -> dict[str, Any]:
    meta = _field(response, "meta")
    billed_units = _field(meta, "billed_units")
    tokens = _field(meta, "tokens")
    raw_results = _field(response, "results")
    input_tokens = _number(_field(tokens, "input_tokens"))
    output_tokens = _number(_field(tokens, "output_tokens"))
    total_tokens = _number(_field(billed_units, "total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "response_id": _string(_field(response, "id")),
        "usage": _usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        ),
        "cost_usd": _cost_from(response),
        "provider": _provider_name(_field(_hidden_params(response), "custom_llm_provider")),
        "metadata": _clean_fields(
            {
                "result_count": _sequence_length(raw_results),
                "search_units": _number(_field(billed_units, "search_units")),
            }
        ),
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
        if op == "rerank":
            fields = {
                "type": "span",
                "attributes": {"gen_ai.operation.name": "rerank"},
            }
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
        self._saw_output = False
        self._finish_reasons: dict[int, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._inner)
            received_at = time.perf_counter()
        except StopIteration:
            self.close()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            try:
                self.close()
            except BaseException:
                pass
            raise
        self._record(chunk, received_at)
        return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._inner.__anext__()
            received_at = time.perf_counter()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            try:
                await self.aclose()
            except BaseException:
                pass
            raise
        self._record(chunk, received_at)
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

    def _record(self, chunk: Any, received_at: float) -> None:
        response_id = _string(_field(chunk, "id"))
        if self._response_id is None and response_id is not None:
            self._response_id = response_id
        response_model = _string(_field(chunk, "model"))
        if self._response_model is None and response_model is not None:
            self._response_model = response_model
        usage = _usage_from(_field(chunk, "usage"))
        if usage is not None and (self._usage is None or _reports_tokens(usage)):
            self._usage = usage
        has_output = False
        for fallback_index, choice in enumerate(_sequence_items(_field(chunk, "choices"))):
            delta = _field(choice, "delta")
            has_output = (
                has_output
                or any(
                    isinstance(value, str) and bool(value)
                    for value in (
                        _field(delta, "content"),
                        _field(delta, "reasoning_content"),
                        _field(delta, "refusal"),
                        _field(_field(delta, "audio"), "data"),
                    )
                )
                or (
                    isinstance(_field(_field(delta, "function_call"), "arguments"), str)
                    and bool(_field(_field(delta, "function_call"), "arguments"))
                )
                or any(
                    isinstance(arguments, str) and bool(arguments)
                    for arguments in (
                        _field(_field(tool, "function"), "arguments")
                        for tool in _sequence_items(_field(delta, "tool_calls"))
                    )
                )
            )
            finish_reason = _string(_field(choice, "finish_reason"))
            if finish_reason is not None:
                choice_index = _number(_field(choice, "index"))
                self._finish_reasons[
                    int(choice_index) if choice_index is not None else fallback_index
                ] = finish_reason
        if has_output:
            self._saw_output = True
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
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
        rebuilt_usage = fields.get("usage")
        if self._usage is not None and (
            rebuilt_usage is None
            or (not _reports_tokens(rebuilt_usage) and _reports_tokens(self._usage))
        ):
            fields["usage"] = self._usage
            if rebuilt_usage is not None:
                fields.pop("cost_usd", None)
        # LiteLLM 1.104+ rebuilds 0/0 usage when the provider reported none; zero tokens
        # for a response with output means unknown, not free.
        if self._saw_output and _is_zero_usage(fields.get("usage")):
            fields.pop("usage", None)
            fields.pop("cost_usd", None)
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


class _InstrumentedResponsesStream:
    def __init__(
        self,
        inner: Any,
        handle: telemetry_dev.SpanHandle,
        started_at: float,
        request_provider: str | None,
    ) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._request_provider = request_provider
        client = telemetry_dev.get_client()
        self._capture_output = client is not None and client.capture_output
        self._saw_first = False
        self._finished = False
        self._partial_content: dict[object, dict[object, dict[str, Any]]] = {}
        self._partial_calls: dict[object, dict[str, Any]] = {}
        self._completed_items: dict[object, Any] = {}
        self._item_indexes: dict[object, int] = {}
        self._truncated_items: set[object] = set()
        self._truncated_item_keys: dict[object, set[object]] = {}
        self._retained_item_count = 0
        self._truncated_key_count = 0
        self._truncation_overflow = False
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._charges: dict[object, tuple[int, int]] = {}
        self._annotation_charge_keys: dict[object, dict[int, object]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._inner)
            received_at = time.perf_counter()
        except StopIteration:
            self.close()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            try:
                self.close()
            except BaseException:
                pass
            raise
        self._record(event, received_at)
        return event

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._inner.__anext__()
            received_at = time.perf_counter()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException as exc:
            self._finish(error=exc)
            try:
                await self.aclose()
            except BaseException:
                pass
            raise
        self._record(event, received_at)
        return event

    def __enter__(self) -> _InstrumentedResponsesStream:
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        exit_method = getattr(self._inner, "__exit__", None)
        return bool(exit_method(exc_type, exc, tb)) if callable(exit_method) else False

    async def __aenter__(self) -> _InstrumentedResponsesStream:
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            await _maybe_await(enter())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.aclose()
        exit_method = getattr(self._inner, "__aexit__", None)
        if callable(exit_method):
            return bool(await _maybe_await(exit_method(exc_type, exc, tb)))
        return False

    def _item_key(self, event: Any) -> object:
        index = _number(_field(event, "output_index"))
        if index is not None:
            return ("index", int(index))
        item_id = _string(_field(event, "item_id")) or _string(_field(_field(event, "item"), "id"))
        return ("id", item_id) if item_id is not None else ("index", 0)

    def _remember_item_index(self, event: Any, key: object) -> None:
        index = _number(_field(event, "output_index"))
        if index is not None:
            self._item_indexes[key] = int(index)

    def _content_key(self, event: Any) -> object:
        index = _number(_field(event, "content_index"))
        if index is not None:
            return ("index", int(index))
        return ("type", _string(_field(event, "type")) or "content")

    def _summary_key(self, event: Any) -> object:
        index = _number(_field(event, "summary_index"))
        return ("summary", int(index) if index is not None else 0)

    def _partial_output(self) -> list[Any]:
        keys = set(self._completed_items) | set(self._partial_calls)
        keys.update(self._partial_content)
        ordered = sorted(keys, key=lambda key: (self._item_indexes.get(key, 0), repr(key)))
        output: list[Any] = []
        for key in ordered:
            completed = self._completed_items.get(key)
            if completed is not None:
                content = self._partial_content.get(key)
                if isinstance(completed, dict) and content:
                    merged = dict(cast(dict[str, Any], completed))
                    summaries: list[dict[str, Any]] = []
                    remaining: list[dict[str, Any]] = []
                    for content_key, value in sorted(
                        content.items(), key=lambda pair: repr(pair[0])
                    ):
                        materialized = self._materialize_partial(value)
                        target = (
                            summaries
                            if isinstance(content_key, tuple) and content_key[0] == "summary"
                            else remaining
                        )
                        target.append(materialized)
                    if summaries:
                        merged["summary"] = summaries
                    if merged.get("type") == "code_interpreter_call" and len(remaining) == 1:
                        merged.update(remaining[0])
                    elif remaining:
                        merged["content"] = remaining
                    output.append(merged)
                else:
                    output.append(completed)
                continue
            call = self._partial_calls.get(key)
            if call is not None:
                output.append(self._materialize_partial(call))
            content = self._partial_content.get(key, {}).items()
            output.extend(
                self._materialize_partial(value)
                for _, value in sorted(content, key=lambda pair: repr(pair[0]))
            )
        return output

    def _materialize_partial(self, value: dict[str, Any]) -> dict[str, Any]:
        if "_field" not in value:
            return value
        field = cast(str, value["_field"])
        fragments = cast(list[str], value["_fragments"])
        return {key: item for key, item in value.items() if key not in {"_field", "_fragments"}} | {
            field: "".join(fragments)
        }

    def _set_completed_item(self, key: object, value: Any) -> None:
        if key not in self._completed_items:
            self._retained_item_count += 1
        self._completed_items[key] = value

    def _pop_completed_item(self, key: object) -> Any:
        if key not in self._completed_items:
            return None
        self._retained_item_count -= 1
        return self._completed_items.pop(key)

    def _set_partial_call(self, key: object, value: dict[str, Any]) -> None:
        if key not in self._partial_calls:
            self._retained_item_count += 1
        self._partial_calls[key] = value

    def _pop_partial_call(self, key: object) -> dict[str, Any] | None:
        if key not in self._partial_calls:
            return None
        self._retained_item_count -= 1
        return self._partial_calls.pop(key)

    def _set_partial_content(
        self, item_key: object, content_key: object, value: dict[str, Any]
    ) -> None:
        content = self._partial_content.setdefault(item_key, {})
        if content_key not in content:
            self._retained_item_count += 1
        content[content_key] = value

    def _pop_partial_content(self, item_key: object) -> dict[object, dict[str, Any]]:
        content = self._partial_content.pop(item_key, {})
        self._retained_item_count -= len(content)
        return content

    def _replace_charge(self, keys: list[object], value: Any, replacement_key: object) -> bool:
        removed_bytes = sum(self._charges.get(key, (0, 0))[0] for key in keys)
        removed_items = sum(self._charges.get(key, (0, 0))[1] for key in keys)
        budget = telemetry_dev.CaptureBudget.from_client()
        budget.bytes_used = max(0, self._budget.bytes_used - removed_bytes)
        budget.items_used = max(0, self._budget.items_used - removed_items)
        before = (budget.bytes_used, budget.items_used)
        if not budget.accept(value):
            return False
        for key in keys:
            self._charges.pop(key, None)
        self._charges[replacement_key] = (
            budget.bytes_used - before[0],
            budget.items_used - before[1],
        )
        self._budget = budget
        self._budget.truncated = bool(self._truncated_items) or self._truncation_overflow
        return True

    def _charge_keys(self, key: object) -> list[object]:
        return [key, *self._annotation_charge_keys.get(key, {}).values()]

    def _clear_annotation_charges(self, key: object) -> None:
        self._annotation_charge_keys.pop(key, None)

    def _replace_raw_charge(
        self,
        keys: list[object],
        value: Any,
        replacement_key: object,
    ) -> tuple[bool, Any | None]:
        removed_bytes = sum(self._charges.get(key, (0, 0))[0] for key in keys)
        removed_items = sum(self._charges.get(key, (0, 0))[1] for key in keys)
        budget = telemetry_dev.CaptureBudget.from_client()
        budget.bytes_used = max(0, self._budget.bytes_used - removed_bytes)
        budget.items_used = max(0, self._budget.items_used - removed_items)
        before = (budget.bytes_used, budget.items_used)
        converted, budget = _bounded_responses_native(value, budget=budget)
        if converted is None or budget.truncated:
            return False, None
        for key in keys:
            self._charges.pop(key, None)
        self._charges[replacement_key] = (
            budget.bytes_used - before[0],
            budget.items_used - before[1],
        )
        self._budget = budget
        self._budget.truncated = bool(self._truncated_items) or self._truncation_overflow
        return True, converted

    def _append_delta(self, storage_key: object, target: dict[str, Any], delta: str) -> bool:
        remaining_bytes = self._budget.max_bytes - self._budget.bytes_used
        delta_bytes = _bounded_utf8_size(delta, remaining_bytes)
        if delta_bytes > remaining_bytes:
            self._budget.truncated = True
            return False
        if storage_key not in self._charges:
            candidate = self._materialize_partial({**target, "_fragments": [delta]})
            return self._replace_charge([], candidate, storage_key)
        if self._budget.items_used >= self._budget.max_items:
            self._budget.truncated = True
            return False
        self._budget.bytes_used += delta_bytes
        self._budget.items_used += 1
        charged_bytes, charged_items = self._charges[storage_key]
        self._charges[storage_key] = (charged_bytes + delta_bytes, charged_items + 1)
        return True

    @staticmethod
    def _truncation_owner(key: object) -> object:
        tuple_key: tuple[object, ...] = (
            cast(tuple[object, ...], key) if isinstance(key, tuple) else ()
        )
        if tuple_key and isinstance(tuple_key[0], tuple):
            return cast(object, tuple_key[0])
        return cast(object, key)

    def _delta_rejected_before(self, key: object) -> bool:
        owner = self._truncation_owner(key)
        if owner not in self._truncated_items:
            return False
        return key in self._truncated_item_keys.get(owner, set()) or self._truncation_overflow

    def _mark_truncated(self, key: object) -> None:
        self._budget.truncated = True
        owner = self._truncation_owner(key)
        if owner in self._truncated_items:
            keys = self._truncated_item_keys[owner]
            if key in keys:
                return
            if self._truncated_key_count < self._budget.max_items:
                keys.add(key)
                self._truncated_key_count += 1
            else:
                self._truncation_overflow = True
            return
        if self._retained_item_count < self._budget.max_items:
            self._truncated_items.add(owner)
            self._retained_item_count += 1
            if self._truncated_key_count < self._budget.max_items:
                self._truncated_item_keys[owner] = {key}
                self._truncated_key_count += 1
            else:
                self._truncated_item_keys[owner] = set()
                self._truncation_overflow = True
        else:
            self._truncation_overflow = True

    def _resolve_truncated(self, item_key: object) -> None:
        if item_key in self._truncated_items:
            self._truncated_items.remove(item_key)
            self._retained_item_count -= 1
        self._truncated_key_count -= len(self._truncated_item_keys.pop(item_key, ()))

    def _resolve_content_truncated(self, item_key: object, content_key: object) -> None:
        keys = self._truncated_item_keys.get(item_key)
        if keys is None:
            return
        if content_key in keys:
            keys.remove(content_key)
            self._truncated_key_count -= 1
        if not keys and not self._truncation_overflow:
            self._resolve_truncated(item_key)

    def _record(self, event: Any, received_at: float) -> None:
        event_type = _string(_field(event, "type"))
        response = _field(event, "response")
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(received_at - self._started_at) * 1000,
                response_id=_string(_field(response, "id")),
                response_model=_string(_field(response, "model")),
            )
        if event_type in {
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_summary_part.added",
        }:
            if not self._capture_output:
                return
            item_key = self._item_key(event)
            payload = _field(event, "item") or _field(event, "part")
            if payload is not None:
                if event_type == "response.output_item.added":
                    accepted, converted = self._replace_raw_charge([], payload, item_key)
                    if accepted:
                        self._set_completed_item(item_key, converted)
                        self._remember_item_index(event, item_key)
                    else:
                        self._mark_truncated(item_key)
                else:
                    content_id = (
                        self._summary_key(event)
                        if event_type == "response.reasoning_summary_part.added"
                        else self._content_key(event)
                    )
                    storage_key = (item_key, content_id)
                    accepted, converted = self._replace_raw_charge([], payload, storage_key)
                    if accepted:
                        self._set_partial_content(
                            item_key, content_id, cast(dict[str, Any], converted)
                        )
                        self._remember_item_index(event, item_key)
                    else:
                        self._mark_truncated(storage_key)
        if event_type is not None and event_type.endswith(".delta"):
            delta = _string(_field(event, "delta"))
            if delta:
                if not self._capture_output:
                    self._handle.record_output_chunk(received_at * 1000)
                    return
                item_key = self._item_key(event)
                target: dict[str, Any] | None = None
                storage_key: object = item_key
                if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                    content_key = (item_key, self._content_key(event))
                    storage_key = content_key
                    target = self._partial_content.get(item_key, {}).get(self._content_key(event))
                    field = "text" if event_type == "response.output_text.delta" else "refusal"
                    if target is None or "_field" not in target:
                        target = {
                            **(target or {}),
                            "type": "output_text" if field == "text" else "refusal",
                            "_field": field,
                            "_fragments": [],
                        }
                elif event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                    "response.audio.transcript.delta",
                    "response.code_interpreter_call_code.delta",
                }:
                    content_id = (
                        self._summary_key(event)
                        if event_type == "response.reasoning_summary_text.delta"
                        else self._content_key(event)
                    )
                    storage_key = (item_key, content_id)
                    target = self._partial_content.get(item_key, {}).get(content_id)
                    kind, field = {
                        "response.reasoning_summary_text.delta": ("summary_text", "text"),
                        "response.reasoning_text.delta": ("reasoning_text", "text"),
                        "response.audio.transcript.delta": ("audio", "transcript"),
                        "response.code_interpreter_call_code.delta": (
                            "code_interpreter_call",
                            "code",
                        ),
                    }[event_type]
                    if target is None or "_field" not in target:
                        target = {
                            **(target or {}),
                            "type": kind,
                            "_field": field,
                            "_fragments": [],
                        }
                elif event_type in {
                    "response.function_call_arguments.delta",
                    "response.mcp_call_arguments.delta",
                    "response.custom_tool_call_input.delta",
                }:
                    target = self._partial_calls.get(item_key)
                    seeded = self._completed_items.get(item_key)
                    if target is None:
                        event_kind = {
                            "response.function_call_arguments.delta": (
                                "function_call",
                                "arguments",
                            ),
                            "response.mcp_call_arguments.delta": ("mcp_call", "arguments"),
                            "response.custom_tool_call_input.delta": ("custom_tool_call", "input"),
                        }[event_type]
                        target = {
                            **(seeded if isinstance(seeded, dict) else {}),
                            "type": event_kind[0],
                            "_field": event_kind[1],
                            "_fragments": [],
                        }
                        item_id = _string(_field(event, "item_id"))
                        if item_id is not None:
                            target["id"] = item_id
                        for field in ("name", "call_id", "server_label"):
                            value = _string(_field(event, field))
                            if value is not None:
                                target[field] = value
                if target is not None:
                    fragments = cast(list[str], target["_fragments"])
                    accepted = not self._delta_rejected_before(storage_key) and self._append_delta(
                        storage_key, target, delta
                    )
                    if accepted:
                        fragments.append(delta)
                        self._remember_item_index(event, item_key)
                        if event_type in {
                            "response.function_call_arguments.delta",
                            "response.mcp_call_arguments.delta",
                            "response.custom_tool_call_input.delta",
                        }:
                            self._pop_completed_item(item_key)
                            self._set_partial_call(item_key, target)
                        else:
                            _, content_id = cast(tuple[object, object], storage_key)
                            self._set_partial_content(item_key, content_id, target)
                    else:
                        self._mark_truncated(storage_key)
                self._handle.record_output_chunk(received_at * 1000)
        if event_type == "response.output_text.annotation.added" and self._capture_output:
            item_key = self._item_key(event)
            content_id = self._content_key(event)
            storage_key = (item_key, content_id)
            target = self._partial_content.get(item_key, {}).get(content_id)
            annotation, annotation_budget = _bounded_responses_native(_field(event, "annotation"))
            if (
                isinstance(target, dict)
                and annotation is not None
                and not annotation_budget.truncated
            ):
                annotations = list(_sequence_items(target.get("annotations")))
                annotation_index = _field(event, "annotation_index")
                if annotation_index is None:
                    position = len(annotations)
                elif (
                    isinstance(annotation_index, bool)
                    or not isinstance(annotation_index, int)
                    or annotation_index < 0
                    or annotation_index > len(annotations)
                ):
                    self._mark_truncated(storage_key)
                    position = None
                else:
                    position = annotation_index
                if position is not None:
                    charge_keys = self._annotation_charge_keys.setdefault(storage_key, {})
                    previous_charge_key = charge_keys.get(position)
                    annotation_charge_key = (storage_key, "annotation", position)
                    if position < len(annotations):
                        annotations[position] = annotation
                    else:
                        annotations.append(annotation)
                    if self._replace_charge(
                        [previous_charge_key] if previous_charge_key is not None else [],
                        annotation,
                        annotation_charge_key,
                    ):
                        charge_keys[position] = annotation_charge_key
                        target["annotations"] = annotations
                    else:
                        self._mark_truncated(storage_key)
        if event_type in {"response.output_item.done", "response.content_part.done"}:
            if not self._capture_output:
                return
            item = _field(event, "item") or _field(event, "part")
            if item is not None:
                item_key = self._item_key(event)
                old_truncated = self._truncated_items.copy()
                old_truncated_keys = {
                    key: values.copy() for key, values in self._truncated_item_keys.items()
                }
                old_retained_item_count = self._retained_item_count
                old_truncated_key_count = self._truncated_key_count
                old_overflow = self._truncation_overflow
                old_index = self._item_indexes.get(item_key)
                if event_type == "response.output_item.done":
                    old_completed = self._completed_items.get(item_key)
                    old_call = self._pop_partial_call(item_key)
                    old_content = self._pop_partial_content(item_key)
                    content_charge_keys = [
                        charge_key
                        for key in old_content
                        for charge_key in self._charge_keys((item_key, key))
                    ]
                    charge_keys = [*self._charge_keys(item_key), *content_charge_keys]
                    accepted, converted = self._replace_raw_charge(charge_keys, item, item_key)
                    if accepted:
                        self._clear_annotation_charges(item_key)
                        for key in old_content:
                            self._clear_annotation_charges((item_key, key))
                        self._set_completed_item(item_key, converted)
                        self._remember_item_index(event, item_key)
                    self._resolve_truncated(item_key)
                    self._budget.truncated = (
                        bool(self._truncated_items) or self._truncation_overflow
                    )
                    if not accepted:
                        if old_completed is None:
                            self._pop_completed_item(item_key)
                        else:
                            self._set_completed_item(item_key, old_completed)
                        if old_call is not None:
                            self._partial_calls[item_key] = old_call
                        if old_content:
                            self._partial_content[item_key] = old_content
                        self._truncated_items = old_truncated
                        self._truncated_item_keys = old_truncated_keys
                        self._retained_item_count = old_retained_item_count
                        self._truncated_key_count = old_truncated_key_count
                        self._truncation_overflow = old_overflow
                        if old_index is None:
                            self._item_indexes.pop(item_key, None)
                        else:
                            self._item_indexes[item_key] = old_index
                        self._mark_truncated(item_key)
                else:
                    content_key = (item_key, self._content_key(event))
                    content_id = self._content_key(event)
                    item_content = self._partial_content.setdefault(item_key, {})
                    old_content = item_content.get(content_id)
                    accepted, converted = self._replace_raw_charge(
                        self._charge_keys(content_key), item, content_key
                    )
                    if accepted:
                        self._clear_annotation_charges(content_key)
                        self._set_partial_content(
                            item_key, content_id, cast(dict[str, Any], converted)
                        )
                        self._remember_item_index(event, item_key)
                        self._resolve_content_truncated(item_key, content_key)
                    self._budget.truncated = (
                        bool(self._truncated_items) or self._truncation_overflow
                    )
                    if not accepted:
                        if old_content is None:
                            item_content.pop(content_id, None)
                            if not item_content:
                                self._partial_content.pop(item_key, None)
                        else:
                            item_content[content_id] = old_content
                        self._truncated_items = old_truncated
                        self._truncated_item_keys = old_truncated_keys
                        self._retained_item_count = old_retained_item_count
                        self._truncated_key_count = old_truncated_key_count
                        self._truncation_overflow = old_overflow
                        if old_index is None:
                            self._item_indexes.pop(item_key, None)
                        else:
                            self._item_indexes[item_key] = old_index
                        self._mark_truncated(content_key)
        if event_type in {"response.completed", "response.incomplete", "response.failed"}:
            self._finish(response=response)
        elif event_type == "error":
            raw_error = _field(event, "error")
            message = (
                _string(_field(raw_error, "message"))
                or _string(_field(event, "message"))
                or "LiteLLM Responses API stream error"
            )
            code = _string(_field(raw_error, "code")) or _string(_field(event, "code"))
            if code is not None:
                message = f"{message} ({code})"
            self._finish(error=RuntimeError(message))

    def _finish(self, response: Any = None, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        fields = (
            _safe_response_fields(_responses_response_without_output, response)
            if response is not None
            else {}
        )
        if response is not None and self._capture_output:
            terminal_output, terminal_budget = _bounded_responses_native(_field(response, "output"))
            if terminal_output is not None and not terminal_budget.truncated:
                fields["output"] = terminal_output
                self._budget = terminal_budget
                self._retained_item_count -= len(self._truncated_items)
                self._truncated_items.clear()
                self._truncated_item_keys.clear()
                self._truncated_key_count = 0
                self._truncation_overflow = False
            elif _field(response, "output") is not None or terminal_budget.truncated:
                self._budget.truncated = True
        if "output" not in fields:
            partial_output = self._partial_output()
            if partial_output:
                fields["output"] = partial_output
        if self._budget.truncated:
            attributes = fields.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
                fields["attributes"] = attributes
            attributes["telemetry.dev.capture.truncated"] = True
        if self._request_provider is not None:
            fields.pop("provider", None)
        if error is not None:
            fields["error"] = error
            if self._request_provider is None:
                fields["provider"] = _provider_from_error(error)
        self._end(**fields)

    def close(self) -> None:
        self._finish()
        first_error: BaseException | None = None
        close = getattr(self._inner, "close", None)
        try:
            if callable(close):
                _run_sync_awaitable(close())
            else:
                aclose = getattr(self._inner, "aclose", None)
                if callable(aclose):
                    _run_sync_awaitable(aclose())
        except BaseException as exc:
            first_error = exc
        response = _field(self._inner, "response")
        close = getattr(response, "close", None)
        try:
            if callable(close):
                _run_sync_awaitable(close())
            else:
                aclose = getattr(response, "aclose", None)
                if callable(aclose):
                    _run_sync_awaitable(aclose())
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error

    async def aclose(self) -> None:
        self._finish()
        first_error: BaseException | None = None
        aclose = getattr(self._inner, "aclose", None)
        try:
            if callable(aclose):
                await _maybe_await(aclose())
            else:
                close = getattr(self._inner, "close", None)
                if callable(close):
                    await _maybe_await(close())
        except BaseException as exc:
            first_error = exc
        response = _field(self._inner, "response")
        aclose = getattr(response, "aclose", None)
        try:
            if callable(aclose):
                await _maybe_await(aclose())
            else:
                close = getattr(response, "close", None)
                if callable(close):
                    await _maybe_await(close())
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error


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
        if op == "responses" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedResponsesStream(result, handle, started_at, request_provider)
        if op == "chat" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedStream(
                result,
                handle,
                messages=_arg(args, mapped_kwargs, "messages", 1),
                started_at=started_at,
                request_provider=request_provider,
            )
        client = telemetry_dev.get_client()
        active_mapper = response_mapper
        if client is None or not client.capture_output:
            if op == "responses":
                active_mapper = _responses_response_without_output
            elif op == "rerank":
                active_mapper = _rerank_response_without_output
        try:
            fields = _safe_response_fields(active_mapper, result)
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
        if op == "responses" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedResponsesStream(result, handle, started_at, request_provider)
        if op == "chat" and _stream_enabled(args, mapped_kwargs, stream_index):
            return _InstrumentedStream(
                result,
                handle,
                messages=_arg(args, mapped_kwargs, "messages", 1),
                started_at=started_at,
                request_provider=request_provider,
            )
        client = telemetry_dev.get_client()
        active_mapper = response_mapper
        if client is None or not client.capture_output:
            if op == "responses":
                active_mapper = _responses_response_without_output
            elif op == "rerank":
                active_mapper = _rerank_response_without_output
        try:
            fields = _safe_response_fields(active_mapper, result)
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
    elif name in {"embedding", "aembedding"}:
        mapper = _embedding_request
        response_mapper = _embedding_response
        op = "embeddings"
    elif name in {"responses", "aresponses"}:
        mapper = _responses_request
        response_mapper = _responses_response
        op = "responses"
    else:
        mapper = _rerank_request
        response_mapper = _rerank_response
        op = "rerank"
    stream_index = _stream_index(original)
    if name in {"acompletion", "aembedding", "arerank", "aresponses"}:
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


def rerank(*args: Any, **kwargs: Any) -> Any:
    return _dropin("rerank", *args, **kwargs)


async def arerank(*args: Any, **kwargs: Any) -> Any:
    return await _dropin("arerank", *args, **kwargs)


def responses(*args: Any, **kwargs: Any) -> Any:
    return _dropin("responses", *args, **kwargs)


async def aresponses(*args: Any, **kwargs: Any) -> Any:
    return await _dropin("aresponses", *args, **kwargs)


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
        for name in (
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "rerank",
            "arerank",
            "responses",
            "aresponses",
        ):
            if hasattr(litellm, name):
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
    for name in (
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "rerank",
        "arerank",
        "responses",
        "aresponses",
    ):
        if hasattr(router, name):
            _patch_router_method(cast(object, router), name)
    setattr(router, _ROUTER_WRAPPED_ATTR, True)
    return router


__all__ = [
    "__version__",
    "acompletion",
    "aembedding",
    "arerank",
    "aresponses",
    "completion",
    "embedding",
    "instrument_litellm",
    "rerank",
    "responses",
    "uninstrument_litellm",
    "wrap_router",
]
