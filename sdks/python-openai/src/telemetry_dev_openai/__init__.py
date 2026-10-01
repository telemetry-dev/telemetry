from __future__ import annotations

import threading
import time
import types
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import wraps
from io import StringIO
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit

import openai
import telemetry_dev
from openai.resources.audio.speech import AsyncSpeech, Speech
from openai.resources.audio.transcriptions import AsyncTranscriptions, Transcriptions
from openai.resources.audio.translations import AsyncTranslations, Translations
from openai.resources.batches import AsyncBatches, Batches
from openai.resources.chat.completions.completions import AsyncCompletions, Completions
from openai.resources.embeddings import AsyncEmbeddings, Embeddings
from openai.resources.images import AsyncImages, Images
from openai.resources.responses.responses import AsyncResponses, Responses

__version__ = "0.1.5"

ProviderResolver = Callable[[object | None], str]
RequestMapper = Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]

_WRAPPED_ATTR = "_telemetry_dev_openai_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_openai_original"
_ORIGINALS: list[tuple[type[Any], str, Any, Any]] = []
_installed = False
_install_lock = threading.Lock()
_T = TypeVar("_T")
_OMIT = object()
_CHAT_STREAM_CAPTURE_MAX_BYTES = 48 * 1024
_TRANSCRIPT_CAPTURE_MAX_BYTES = 64 * 1024
_CHAT_STREAM_CAPTURE_MAX_ITEMS = 1000


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
            size += 6
        elif code <= 0xFFFF:
            size += 3
        else:
            size += 4
        if size > limit:
            return size
    return size


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return mapping.get(name)
    return getattr(value, name, None)


def _own_field(
    value: Any,
    name: str,
    budget: telemetry_dev.CaptureBudget | None = None,
    failure: list[bool] | None = None,
    report_error: Callable[[BaseException], None] | None = None,
) -> Any:
    try:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, Any], value)
            return mapping.get(name)
    except Exception as exc:
        if budget is not None:
            budget.truncated = True
        if failure is not None:
            failure[0] = True
        if report_error is not None:
            report_error(exc)
        return None
    for storage in ("__dict__", "__pydantic_extra__"):
        stored = _stored_namespace(value, storage)
        if stored is not None and name in stored:
            return stored[name]
    return None


_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAMESPACE = type.__dict__["__dict__"]


def _is_storage_descriptor(descriptor: object, name: str) -> bool:
    kind = type(descriptor)
    allowed = kind is types.MemberDescriptorType or (
        name == "__dict__" and kind is types.GetSetDescriptorType
    )
    return allowed and cast(Any, descriptor).__name__ == name


def _stored_namespace(value: Any, name: str) -> dict[str, Any] | None:
    # Read instance storage through C-level slot descriptors only, so a property or a
    # custom mapping on a hostile object never runs inside a bounded field read.
    cls = type(cast(object, value))
    for klass in _TYPE_MRO.__get__(cls):
        namespace = _TYPE_NAMESPACE.__get__(klass)
        if name not in namespace:
            continue
        descriptor = namespace[name]
        if not _is_storage_descriptor(descriptor, name):
            return None
        try:
            stored = descriptor.__get__(value, cls)
        except (AttributeError, TypeError):
            return None
        return cast(dict[str, Any], stored) if type(stored) is dict else None
    return None


def _sequence_items(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return cast(Sequence[Any], value)
    return ()


def _bounded_responses_capture(
    value: Any,
    capture: Literal["input", "output"] = "output",
    parent_type: str | None = None,
    *,
    capture_enabled: bool | None = None,
    max_bytes: int | None = None,
    report_error: Callable[[BaseException], None] | None = None,
) -> tuple[Any | None, bool]:
    client = telemetry_dev.get_client()
    if capture_enabled is None:
        if client is None:
            return None, False
        capture_enabled = client.capture_input if capture == "input" else client.capture_output
    if not capture_enabled:
        return None, False
    budget = (
        telemetry_dev.CaptureBudget(max_bytes=max_bytes)
        if max_bytes is not None
        else telemetry_dev.CaptureBudget()
    )
    ancestors: set[int] = set()

    def reserve(byte_count: int) -> None:
        if (
            budget.bytes_used + byte_count > budget.max_bytes
            or budget.items_used >= budget.max_items
        ):
            raise _CaptureLimit
        budget.bytes_used += byte_count
        budget.items_used += 1

    def convert(item: Any, parent_type: str | None, depth: int) -> Any:
        if depth > 32:
            raise _CaptureLimit
        if isinstance(item, bytes | bytearray | memoryview):
            return _OMIT
        if item is None or isinstance(item, bool | int | float):
            reserve(16)
            return item
        if isinstance(item, str):
            reserve(16 + _bounded_utf8_size(item, budget.max_bytes - budget.bytes_used - 16))
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
                item_type = _string(source.get("type")) or parent_type
                result: dict[str, Any] = {}
                for raw_key, child in source.items():
                    if child is None:
                        continue
                    key = str(raw_key)
                    binary = (
                        key in {"b64_json", "file_data"}
                        or (
                            key == "image_url"
                            and isinstance(child, str)
                            and child[:5].lower() == "data:"
                        )
                        or (
                            key == "url"
                            and item_type == "image_url"
                            and isinstance(child, str)
                            and child[:5].lower() == "data:"
                        )
                        or (
                            key in {"data", "audio"}
                            and item_type in {"input_audio", "output_audio", "audio"}
                        )
                        or (
                            key in {"result", "partial_image_b64"}
                            and item_type == "image_generation_call"
                        )
                    )
                    if binary:
                        continue
                    reserve(16 + _bounded_utf8_size(key, budget.max_bytes - budget.bytes_used - 16))
                    converted = convert(child, item_type or key, depth + 1)
                    if converted is not _OMIT:
                        result[key] = converted
                return result
            if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
                sequence = cast(Sequence[Any], item)
                reserve(16)
                list_result: list[Any] = []
                for child in sequence:
                    converted = convert(child, parent_type, depth + 1)
                    if converted is not _OMIT:
                        list_result.append(converted)
                return list_result
            reserve(16)
            return None
        finally:
            ancestors.remove(item_id)

    try:
        return convert(value, parent_type, 0), False
    except _CaptureLimit:
        return None, True
    except Exception as exc:
        try:
            if report_error is not None:
                report_error(exc)
            elif client is not None:
                client.report("provider instrumentation failed", exc)
        except Exception:
            pass
        return None, True


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


def _stop_sequences(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    strings = [item for item in _sequence_items(value) if isinstance(item, str)]
    return strings or None


def _sent_params(params: Mapping[str, Any]) -> dict[str, Any]:
    sent = dict(params)
    extra_body = params.get("extra_body")
    if isinstance(extra_body, Mapping):
        overrides = cast(Mapping[str, Any], extra_body)
        sent.update(
            (key, value)
            for key, value in overrides.items()
            if not isinstance(value, openai.NotGiven)
        )
    return {
        key: value
        for key, value in sent.items()
        if key != "extra_body" and not isinstance(value, (openai.NotGiven, openai.Omit))
    }


def _transcript_capture_budget() -> telemetry_dev.CaptureBudget:
    return telemetry_dev.CaptureBudget(max_bytes=_TRANSCRIPT_CAPTURE_MAX_BYTES)


def _capture_text(value: str, budget: telemetry_dev.CaptureBudget) -> str:
    if not value or budget.truncated:
        return ""
    if budget.remaining_bytes <= 0:
        budget.truncated = True
        return ""
    byte_count = 0
    retained_characters = 0
    for index, character in enumerate(value):
        code = ord(character)
        if code <= 0x7F:
            character_bytes = 1
        elif code <= 0x7FF:
            character_bytes = 2
        elif 0xD800 <= code <= 0xDFFF:
            character_bytes = 6
        elif code <= 0xFFFF:
            character_bytes = 3
        else:
            character_bytes = 4
        if byte_count + character_bytes > budget.remaining_bytes:
            budget.truncated = True
            break
        byte_count += character_bytes
        retained_characters = index + 1
    budget.bytes_used += byte_count
    if retained_characters < len(value):
        budget.truncated = True
        return value[:retained_characters]
    return value


def _chat_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    captured, truncated = _bounded_responses_capture(params.get("messages"), "input")
    return (
        f"chat {model or 'unknown'}",
        {
            "type": "generation",
            "model": model,
            "input": captured,
            "temperature": _number(params.get("temperature")),
            "top_p": _number(params.get("top_p")),
            "max_tokens": _number(params.get("max_completion_tokens"))
            or _number(params.get("max_tokens")),
            "stop_sequences": _stop_sequences(params.get("stop")),
            "seed": _number(params.get("seed")),
            "frequency_penalty": _number(params.get("frequency_penalty")),
            "presence_penalty": _number(params.get("presence_penalty")),
            "attributes": {"telemetry.dev.capture.truncated": True} if truncated else None,
        },
    )


def _chat_usage(raw: Any) -> dict[str, int | float] | None:
    prompt_details = _field(raw, "prompt_tokens_details")
    completion_details = _field(raw, "completion_tokens_details")
    cached_details = _field(prompt_details, "cached_tokens_details")
    return _usage(
        {
            "input_tokens": _number(_field(raw, "prompt_tokens")),
            "output_tokens": _number(_field(raw, "completion_tokens")),
            "total_tokens": _number(_field(raw, "total_tokens")),
            "cache_read_input_tokens": _number(_field(prompt_details, "cached_tokens")),
            "reasoning_output_tokens": _number(_field(completion_details, "reasoning_tokens")),
            "text_input_tokens": _number(_field(prompt_details, "text_tokens")),
            "audio_input_tokens": _number(_field(prompt_details, "audio_tokens")),
            "image_input_tokens": _number(_field(prompt_details, "image_tokens")),
            "text_cache_read_input_tokens": _number(_field(cached_details, "text_tokens")),
            "audio_cache_read_input_tokens": _number(_field(cached_details, "audio_tokens")),
            "image_cache_read_input_tokens": _number(_field(cached_details, "image_tokens")),
            "text_output_tokens": _number(_field(completion_details, "text_tokens")),
            "audio_output_tokens": _number(_field(completion_details, "audio_tokens")),
            "image_output_tokens": _number(_field(completion_details, "image_tokens")),
        }
    )


def _chat_response(response: Any) -> dict[str, Any]:
    choices = list(_field(response, "choices") or [])
    messages = [_field(choice, "message") for choice in choices]
    output, truncated = _bounded_responses_capture(messages)
    if isinstance(output, list):
        for message, captured in zip(messages, cast(list[Any], output), strict=False):
            if isinstance(captured, dict) and _field(message, "content") is None:
                captured["content"] = None
    finish_reasons = [
        reason
        for choice in choices
        if (reason := _string(_field(choice, "finish_reason"))) is not None
    ]
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "output": output,
        "usage": _chat_usage(_field(response, "usage")),
        # The core SDK maps finish_reason to a single-element array; multi-choice
        # responses need one entry per choice, via the raw-attribute escape hatch.
        "attributes": (
            {
                **(
                    {"gen_ai.response.finish_reasons": finish_reasons}
                    if len(finish_reasons) > 1
                    else {}
                ),
                **({"telemetry.dev.capture.truncated": True} if truncated else {}),
            }
            or None
        ),
    }


def _responses_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    captured, truncated = _bounded_responses_capture(params.get("input"), "input")
    return (
        f"chat {model or 'unknown'}",
        {
            "type": "generation",
            "model": model,
            "input": captured,
            "system_instructions": params.get("instructions"),
            "temperature": _number(params.get("temperature")),
            "top_p": _number(params.get("top_p")),
            "max_tokens": _number(params.get("max_output_tokens")),
            "attributes": {"telemetry.dev.capture.truncated": True} if truncated else None,
        },
    )


def _responses_usage(raw: Any) -> dict[str, int | float] | None:
    input_details = _field(raw, "input_tokens_details")
    output_details = _field(raw, "output_tokens_details")
    cached_details = _field(input_details, "cached_tokens_details")
    return _usage(
        {
            "input_tokens": _number(_field(raw, "input_tokens")),
            "output_tokens": _number(_field(raw, "output_tokens")),
            "total_tokens": _number(_field(raw, "total_tokens")),
            "cache_read_input_tokens": _number(_field(input_details, "cached_tokens")),
            "reasoning_output_tokens": _number(_field(output_details, "reasoning_tokens")),
            "text_input_tokens": _number(_field(input_details, "text_tokens")),
            "audio_input_tokens": _number(_field(input_details, "audio_tokens")),
            "image_input_tokens": _number(_field(input_details, "image_tokens")),
            "text_cache_read_input_tokens": _number(_field(cached_details, "text_tokens")),
            "audio_cache_read_input_tokens": _number(_field(cached_details, "audio_tokens")),
            "image_cache_read_input_tokens": _number(_field(cached_details, "image_tokens")),
            "text_output_tokens": _number(_field(output_details, "text_tokens")),
            "audio_output_tokens": _number(_field(output_details, "audio_tokens")),
            "image_output_tokens": _number(_field(output_details, "image_tokens")),
        }
    )


def _media_request(endpoint: str, output_type: str) -> RequestMapper:
    def mapper(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        model = _string(params.get("model"))
        input_value = params.get("prompt") or params.get("input")
        if not isinstance(input_value, str):
            input_value = None
        return (
            f"{endpoint} {model or 'unknown'}",
            {
                "type": "generation",
                "model": model,
                "input": input_value,
                "output_type": output_type,
                "attributes": {"gen_ai.operation.name": "generate_content"},
            },
        )

    return mapper


def _media_response(response: Any) -> dict[str, Any]:
    usage = _field(response, "usage")
    input_details = _field(usage, "input_tokens_details") or _field(usage, "input_token_details")
    output_details = _field(usage, "output_tokens_details") or _field(usage, "output_token_details")
    output_tokens = _number(_field(usage, "output_tokens"))
    image_output_tokens = _number(_field(output_details, "image_tokens"))
    event_type = _string(_field(response, "type")) or ""
    is_image = (
        _field(response, "data") is not None
        or event_type.startswith("image_generation.")
        or event_type.startswith("image_edit.")
    )
    return {
        "response_id": _string(_field(response, "id")),
        "usage": _usage(
            {
                "input_tokens": _number(_field(usage, "input_tokens")),
                "output_tokens": _number(_field(usage, "output_tokens")),
                "total_tokens": _number(_field(usage, "total_tokens")),
                "text_input_tokens": _number(_field(input_details, "text_tokens")),
                "image_input_tokens": _number(_field(input_details, "image_tokens")),
                "audio_input_tokens": _number(_field(input_details, "audio_tokens")),
                "text_output_tokens": _number(_field(output_details, "text_tokens")),
                "image_output_tokens": image_output_tokens
                if image_output_tokens is not None
                else (output_tokens if is_image else None),
                "audio_output_tokens": _number(_field(output_details, "audio_tokens")),
            }
        ),
    }


def _text_media_response(response: Any, *, capture_output: bool | None = None) -> dict[str, Any]:
    fields = _media_response(response)
    if capture_output is None:
        client = telemetry_dev.get_client()
        capture_output = client is not None and client.capture_output
    if capture_output:
        text = _string(_field(response, "text"))
        if text is not None:
            fields["output"] = text
    return _text_media_usage(response, fields)


def _text_media_usage(response: Any, fields: dict[str, Any]) -> dict[str, Any]:
    usage = _field(response, "usage")
    output_tokens = _number(_field(usage, "output_tokens"))
    output_details = _field(usage, "output_tokens_details") or _field(usage, "output_token_details")
    if output_tokens is not None and _number(_field(output_details, "text_tokens")) is None:
        mapped_usage = fields.get("usage")
        if isinstance(mapped_usage, dict):
            mapped_usage["text_output_tokens"] = output_tokens
    return fields


def _batch_request(action: str) -> RequestMapper:
    def mapper(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        batch_id = _string(params.get("batch_id"))
        return (
            f"openai.batch.{action}",
            {
                "type": "span",
                "input": {
                    key: params.get(key)
                    for key in ("batch_id", "input_file_id", "endpoint")
                    if params.get(key) is not None
                },
                "attributes": {
                    "gen_ai.operation.name": f"openai.batch.{action}",
                    **({"openai.batch.id": batch_id} if batch_id else {}),
                },
            },
        )

    return mapper


def _batch_response(response: Any) -> dict[str, Any]:
    batch_id = _string(_field(response, "id"))
    status = _string(_field(response, "status"))
    attrs: dict[str, Any] = {}
    if batch_id:
        attrs["openai.batch.id"] = batch_id
    if status:
        attrs["openai.batch.status"] = status
    return {"attributes": attrs or None}


def _batch_params(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(kwargs)
    if "batch_id" not in params:
        offset = 1 if args and hasattr(args[0], "_client") else 0
        if len(args) > offset:
            params["batch_id"] = args[offset]
    return params


def _responses_response(
    response: Any, *, include_error: bool = True, include_output: bool = True
) -> dict[str, Any]:
    status = _string(_field(response, "status"))
    incomplete_details = _field(response, "incomplete_details")
    fields: dict[str, Any] = {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "usage": _responses_usage(_field(response, "usage")),
        "finish_reason": "stop"
        if status == "completed"
        else _string(_field(incomplete_details, "reason")) or status,
    }
    if include_output:
        output, truncated = _bounded_responses_capture(_field(response, "output"))
        fields["output"] = output
        if truncated:
            fields["attributes"] = {"telemetry.dev.capture.truncated": True}
    if include_error and status == "failed":
        fields["error"] = _response_failed_error(response)
    return fields


def _embeddings_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    model = _string(params.get("model"))
    return (
        f"embeddings {model or 'unknown'}",
        {"type": "embedding", "model": model, "input": params.get("input")},
    )


def _embeddings_response(response: Any) -> dict[str, Any]:
    raw_usage = _field(response, "usage")
    return {
        "response_model": _string(_field(response, "model")),
        "usage": _usage(
            {
                "input_tokens": _number(_field(raw_usage, "prompt_tokens")),
                "total_tokens": _number(_field(raw_usage, "total_tokens")),
            }
        ),
    }


def _base_url_host(base_url: object) -> str | None:
    host = getattr(base_url, "host", None)
    if isinstance(host, str):
        return host
    if isinstance(base_url, str):
        return urlsplit(base_url).hostname
    return None


def _provider_for_client(client: object | None) -> str:
    if isinstance(client, openai.AzureOpenAI | openai.AsyncAzureOpenAI):
        return "azure.ai.openai"
    host = _base_url_host(getattr(client, "base_url", None))
    if host is not None:
        host = host.lower().removesuffix(".")
    if host == "openrouter.ai" or (host is not None and host.endswith(".openrouter.ai")):
        return "openrouter"
    for domain, provider in (
        ("groq.com", "groq"),
        ("x.ai", "x_ai"),
        ("deepseek.com", "deepseek"),
        ("together.xyz", "together_ai"),
        ("fireworks.ai", "fireworks_ai"),
    ):
        if host == domain or (host is not None and host.endswith(f".{domain}")):
            return provider
    return "openai"


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


class _ChatChoice:
    def __init__(self) -> None:
        self.role: str | None = None
        self.role_resolved = True
        self.content_fragments = StringIO()
        self.refusal_fragments = StringIO()
        self.function_call: dict[str, Any] | None = None
        self.unresolved_function_scalars: set[str] = set()
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.unresolved_tool_scalars: dict[int, set[str]] = {}
        self.terminal = False

    @property
    def content(self) -> str:
        return self.content_fragments.getvalue()

    @property
    def refusal(self) -> str:
        return self.refusal_fragments.getvalue()


class _ChatToolCallDelta:
    def __init__(
        self,
        *,
        index: int | None,
        tool_id: str | None,
        tool_type: str | None,
        function_name: str | None,
        function_arguments: str | None,
        custom_name: str | None,
        custom_input: str | None,
        read_failed: bool,
    ) -> None:
        self.index = index
        self.tool_id = tool_id
        self.tool_type = tool_type
        self.function_name = function_name
        self.function_arguments = function_arguments
        self.custom_name = custom_name
        self.custom_input = custom_input
        self.read_failed = read_failed


def _chat_capture_budget(*, reserve_output_list: bool = False) -> telemetry_dev.CaptureBudget:
    budget = telemetry_dev.CaptureBudget(
        _CHAT_STREAM_CAPTURE_MAX_BYTES,
        _CHAT_STREAM_CAPTURE_MAX_ITEMS,
    )
    if reserve_output_list:
        budget.accept([])
    return budget


def _replace_finish_reason(
    index: int,
    value: str,
    budget: telemetry_dev.CaptureBudget,
    reservations: dict[int, tuple[int, int]],
) -> bool:
    held_bytes, held_items = reservations.get(index, (0, 0))
    base_bytes = budget.bytes_used - held_bytes
    base_items = budget.items_used - held_items
    available_bytes = budget.max_bytes - base_bytes - 98
    if available_bytes < 0 or base_items + 5 > budget.max_items:
        budget.bytes_used = base_bytes
        budget.items_used = base_items
        reservations.pop(index, None)
        return False
    value_bytes = _bounded_json_string_size(value, available_bytes)
    if value_bytes > available_bytes:
        budget.bytes_used = base_bytes
        budget.items_used = base_items
        reservations.pop(index, None)
        return False
    reservation = (98 + value_bytes, 5)
    budget.bytes_used = base_bytes + reservation[0]
    budget.items_used = base_items + reservation[1]
    reservations[index] = reservation
    return True


def _reserve_chat_budget(
    budget: telemetry_dev.CaptureBudget,
    byte_count: int,
    item_count: int = 0,
    *,
    recoverable: bool = False,
) -> bool:
    if (
        budget.truncated
        or budget.bytes_used + byte_count > budget.max_bytes
        or budget.items_used + item_count > budget.max_items
    ):
        if not recoverable:
            budget.truncated = True
        return False
    budget.bytes_used += byte_count
    budget.items_used += item_count
    return True


def _bounded_json_string_size(value: str, limit: int) -> int:
    size = 0
    for character in value:
        code = ord(character)
        if character in {'"', "\\"} or character in {"\b", "\t", "\n", "\f", "\r"}:
            size += 2
        elif code <= 0x1F or 0xD800 <= code <= 0xDFFF:
            size += 6
        elif code <= 0x7F:
            size += 1
        elif code <= 0x7FF:
            size += 2
        elif code <= 0xFFFF:
            size += 3
        else:
            size += 4
        if size > limit:
            return size
    return size


def _capture_chat_string(
    value: str,
    budget: telemetry_dev.CaptureBudget,
    field_name: str | None = None,
    *,
    recoverable: bool = False,
) -> bool:
    structure_bytes = 0
    structure_items = 0
    if field_name is not None:
        structure_bytes = 32 + len(field_name.encode())
        structure_items = 2
    available_bytes = budget.max_bytes - budget.bytes_used - structure_bytes
    if available_bytes < 0:
        if not recoverable:
            budget.truncated = True
        return False
    value_bytes = _bounded_json_string_size(value, available_bytes)
    if value_bytes > available_bytes:
        if not recoverable:
            budget.truncated = True
        return False
    return _reserve_chat_budget(
        budget,
        structure_bytes + value_bytes,
        structure_items,
        recoverable=recoverable,
    )


def _replace_chat_scalar(
    current: str | None, value: str, budget: telemetry_dev.CaptureBudget
) -> bool:
    held_bytes = 16
    if current is not None:
        held_bytes += _bounded_json_string_size(current, budget.max_bytes)
    base_bytes = budget.bytes_used - held_bytes
    available_bytes = budget.max_bytes - base_bytes - 16
    if available_bytes < 0:
        return False
    value_bytes = _bounded_json_string_size(value, available_bytes)
    if value_bytes > available_bytes:
        return False
    budget.bytes_used = base_bytes + 16 + value_bytes
    return True


def _release_chat_field(field_name: str, value: str, budget: telemetry_dev.CaptureBudget) -> None:
    budget.bytes_used -= (
        32 + len(field_name.encode()) + _bounded_json_string_size(value, budget.max_bytes)
    )
    budget.items_used -= 2


def _read_tool_call_delta(
    delta: Any,
    budget: telemetry_dev.CaptureBudget | None = None,
    report_error: Callable[[BaseException], None] | None = None,
) -> _ChatToolCallDelta:
    failure = [False]
    incoming_function = _own_field(delta, "function", budget, failure, report_error)
    incoming_custom = _own_field(delta, "custom", budget, failure, report_error)
    index = _own_field(delta, "index", budget, failure, report_error)
    return _ChatToolCallDelta(
        index=index if isinstance(index, int) else None,
        tool_id=_string(_own_field(delta, "id", budget, failure, report_error)),
        tool_type=_string(_own_field(delta, "type", budget, failure, report_error)),
        function_name=_string(_own_field(incoming_function, "name", budget, failure, report_error)),
        function_arguments=_string(
            _own_field(incoming_function, "arguments", budget, failure, report_error)
        ),
        custom_name=_string(_own_field(incoming_custom, "name", budget, failure, report_error)),
        custom_input=_string(_own_field(incoming_custom, "input", budget, failure, report_error)),
        read_failed=failure[0],
    )


def _capture_tool_call_payload(
    current: dict[str, Any],
    payload_key: str,
    name: str | None,
    value_key: str,
    value: str | None,
    unresolved: set[str],
    budget: telemetry_dev.CaptureBudget,
) -> bool:
    if name is None and value is None:
        return False
    raw_payload = current.get(payload_key)
    had_payload = isinstance(raw_payload, Mapping)
    payload = dict(cast(Mapping[str, Any], raw_payload)) if had_payload else {}
    unresolved_name = f"{payload_key}.name"
    changed = False
    if not had_payload:
        if not _reserve_chat_budget(budget, 32 + len(payload_key.encode()), 2):
            return False
        current[payload_key] = payload
        changed = True
    if name is not None:
        current_name = _string(payload.get("name"))
        captured = (
            _capture_chat_string(name, budget, "name", recoverable=True)
            if current_name is None
            else _replace_chat_scalar(current_name, name, budget)
        )
        if captured:
            payload["name"] = name
            unresolved.discard(unresolved_name)
            changed = True
        else:
            if current_name is not None:
                _release_chat_field("name", current_name, budget)
                payload.pop("name", None)
                changed = True
            unresolved.add(unresolved_name)
    if not budget.truncated and value is not None:
        fragments = payload.get(value_key)
        had_value = isinstance(fragments, StringIO)
        value_fragments = fragments if had_value else StringIO()
        if value or not had_value:
            if _capture_chat_string(
                value,
                budget,
                None if had_value else value_key,
            ):
                value_fragments.write(value)
                payload[value_key] = value_fragments
                changed = True
    if changed:
        current[payload_key] = payload
    return changed


def _capture_tool_call_delta(
    state: _ChatChoice, delta: _ChatToolCallDelta, budget: telemetry_dev.CaptureBudget
) -> None:
    tool_index = delta.index if delta.index is not None else len(state.tool_calls)
    had_tool_call = tool_index in state.tool_calls
    current = dict(state.tool_calls.get(tool_index, {}))
    unresolved = set(state.unresolved_tool_scalars.get(tool_index, set()))
    tool_id = delta.tool_id
    tool_type = delta.tool_type
    changed = False

    def save() -> None:
        if changed or unresolved or not had_tool_call:
            state.tool_calls[tool_index] = current
        if unresolved:
            state.unresolved_tool_scalars[tool_index] = unresolved
        else:
            state.unresolved_tool_scalars.pop(tool_index, None)

    if (
        tool_id is None
        and tool_type is None
        and delta.function_name is None
        and delta.function_arguments is None
        and delta.custom_name is None
        and delta.custom_input is None
    ):
        return

    if tool_index not in state.tool_calls:
        first_tool_call = not state.tool_calls
        structure_bytes = 16
        structure_items = 1
        if first_tool_call:
            structure_bytes += 42
            structure_items += 2
            if state.content_fragments.tell() == 0 and state.function_call is None:
                structure_bytes += 39
                structure_items += 2
        if not _reserve_chat_budget(budget, structure_bytes, structure_items):
            return

    if tool_id is not None:
        current_id = _string(current.get("id"))
        captured = (
            _capture_chat_string(tool_id, budget, "id", recoverable=True)
            if current_id is None
            else _replace_chat_scalar(current_id, tool_id, budget)
        )
        if captured:
            current["id"] = tool_id
            unresolved.discard("id")
            changed = True
        else:
            if current_id is not None:
                _release_chat_field("id", current_id, budget)
                current.pop("id", None)
                changed = True
            unresolved.add("id")
    if tool_type is not None:
        current_type = _string(current.get("type"))
        captured = (
            _capture_chat_string(tool_type, budget, "type", recoverable=True)
            if current_type is None
            else _replace_chat_scalar(current_type, tool_type, budget)
        )
        if captured:
            current["type"] = tool_type
            unresolved.discard("type")
            changed = True
        else:
            if current_type is not None:
                _release_chat_field("type", current_type, budget)
                current.pop("type", None)
                changed = True
            unresolved.add("type")
    changed = (
        _capture_tool_call_payload(
            current,
            "function",
            delta.function_name,
            "arguments",
            delta.function_arguments,
            unresolved,
            budget,
        )
        or changed
    )
    changed = (
        _capture_tool_call_payload(
            current,
            "custom",
            delta.custom_name,
            "input",
            delta.custom_input,
            unresolved,
            budget,
        )
        or changed
    )
    save()


def _capture_function_call_delta(
    state: _ChatChoice,
    name: str | None,
    arguments: str | None,
    budget: telemetry_dev.CaptureBudget,
) -> None:
    if name is None and arguments is None:
        return
    current = dict(state.function_call or {})
    if state.function_call is None:
        structure_bytes = 45
        structure_items = 2
        if state.content_fragments.tell() == 0 and not state.tool_calls:
            structure_bytes += 39
            structure_items += 2
        if not _reserve_chat_budget(budget, structure_bytes, structure_items):
            return
        state.function_call = current
    if name is not None:
        current_name = _string(current.get("name"))
        captured = (
            _capture_chat_string(name, budget, "name", recoverable=True)
            if current_name is None
            else _replace_chat_scalar(current_name, name, budget)
        )
        if captured:
            current["name"] = name
            state.unresolved_function_scalars.discard("name")
        else:
            if current_name is not None:
                _release_chat_field("name", current_name, budget)
                current.pop("name", None)
            state.unresolved_function_scalars.add("name")
    if not budget.truncated and arguments is not None:
        fragments = current.get("arguments")
        had_arguments = isinstance(fragments, StringIO)
        argument_fragments = fragments if had_arguments else StringIO()
        if arguments or not had_arguments:
            if _capture_chat_string(
                arguments,
                budget,
                None if had_arguments else "arguments",
            ):
                argument_fragments.write(arguments)
                current["arguments"] = argument_fragments
    state.function_call = current


def _record_chat_chunk(
    chunk: Any,
    states: dict[int, _ChatChoice],
    finish_reason_states: dict[int, str],
    finish_reason_reservations: dict[int, tuple[int, int]],
    rejected_finish_reasons: set[int],
    output_budget: telemetry_dev.CaptureBudget,
    finish_reason_budget: telemetry_dev.CaptureBudget,
    capture_output: bool,
    report_error: Callable[[BaseException], None] | None = None,
    unterminated_choices: set[int] | None = None,
) -> dict[str, Any]:
    chunk_has_output = False
    field_read_failed = [False]
    choices = _own_field(chunk, "choices", output_budget, field_read_failed, report_error)
    choice_items = _sequence_items(choices)
    remaining_tool_calls = _CHAT_STREAM_CAPTURE_MAX_ITEMS
    for choice in choice_items[:_CHAT_STREAM_CAPTURE_MAX_ITEMS]:
        capture_budget = output_budget if capture_output else None
        index = _own_field(choice, "index", capture_budget, field_read_failed, report_error)
        choice_index = index if isinstance(index, int) else 0
        delta = _own_field(choice, "delta", capture_budget, field_read_failed, report_error)
        content = _string(
            _own_field(delta, "content", capture_budget, field_read_failed, report_error)
        )
        refusal = _string(
            _own_field(delta, "refusal", capture_budget, field_read_failed, report_error)
        )
        audio = _own_field(delta, "audio", None, field_read_failed, report_error)
        audio_data = _string(_own_field(audio, "data", None, field_read_failed, report_error))
        legacy_function = _own_field(
            delta, "function_call", capture_budget, field_read_failed, report_error
        )
        legacy_name = _string(
            _own_field(legacy_function, "name", capture_budget, field_read_failed, report_error)
        )
        legacy_arguments = _string(
            _own_field(
                legacy_function,
                "arguments",
                capture_budget,
                field_read_failed,
                report_error,
            )
        )
        tool_call_deltas = _sequence_items(
            _own_field(delta, "tool_calls", capture_budget, field_read_failed, report_error)
        )
        chunk_has_output = chunk_has_output or (
            bool(content) or bool(refusal) or bool(audio_data) or bool(legacy_arguments)
        )
        finish_reason = _string(
            _own_field(
                choice,
                "finish_reason",
                finish_reason_budget,
                field_read_failed,
                report_error,
            )
        )
        if finish_reason is not None:
            if unterminated_choices is not None:
                unterminated_choices.discard(choice_index)
            if _replace_finish_reason(
                choice_index,
                finish_reason,
                finish_reason_budget,
                finish_reason_reservations,
            ):
                finish_reason_states[choice_index] = finish_reason
                rejected_finish_reasons.discard(choice_index)
            else:
                finish_reason_states.pop(choice_index, None)
                if (
                    choice_index in rejected_finish_reasons
                    or len(rejected_finish_reasons) < _CHAT_STREAM_CAPTURE_MAX_ITEMS
                ):
                    rejected_finish_reasons.add(choice_index)
                else:
                    finish_reason_budget.truncated = True
        elif (
            unterminated_choices is not None
            and choice_index not in finish_reason_states
            and choice_index not in rejected_finish_reasons
        ):
            if (
                choice_index in unterminated_choices
                or len(unterminated_choices) < _CHAT_STREAM_CAPTURE_MAX_ITEMS
            ):
                unterminated_choices.add(choice_index)
            else:
                finish_reason_budget.truncated = True
        had_state = choice_index in states
        retained_state = states.get(choice_index) if capture_output else None
        if retained_state is not None and finish_reason is not None:
            retained_state.terminal = True
        state = retained_state
        if capture_output and not output_budget.truncated and state is None:
            initial_state = _ChatChoice()
            initial_state.terminal = finish_reason is not None
            if output_budget.accept(_chat_message(initial_state)):
                state = initial_state
                states[choice_index] = state
        role = (
            _string(_own_field(delta, "role", output_budget, field_read_failed, report_error))
            if state is not None
            else None
        )
        if (
            state is not None
            and role is not None
            and (
                not state.role_resolved
                or role != (state.role if state.role is not None else "assistant")
            )
        ):
            captured_role = (
                _replace_chat_scalar(
                    state.role if state.role is not None else "assistant",
                    role,
                    output_budget,
                )
                if state.role_resolved
                else _capture_chat_string(role, output_budget, "role", recoverable=True)
            )
            if captured_role:
                state.role = role
                state.role_resolved = True
            else:
                if state.role_resolved:
                    _release_chat_field(
                        "role", state.role if state.role is not None else "assistant", output_budget
                    )
                state.role = None
                state.role_resolved = False
        if state is not None and not output_budget.truncated and content:
            replacing_null = state.content_fragments.tell() == 0 and (
                state.function_call is not None or bool(state.tool_calls)
            )
            captured_content = (
                _replace_chat_scalar(None, content, output_budget)
                if replacing_null
                else _capture_chat_string(
                    content,
                    output_budget,
                    None if state.content_fragments.tell() > 0 else "content",
                )
            )
            if captured_content:
                state.content_fragments.write(content)
            elif replacing_null:
                output_budget.truncated = True
        if (
            state is not None
            and not output_budget.truncated
            and refusal
            and _capture_chat_string(
                refusal,
                output_budget,
                None if state.refusal_fragments.tell() > 0 else "refusal",
            )
        ):
            state.refusal_fragments.write(refusal)
        if state is not None:
            _capture_function_call_delta(state, legacy_name, legacy_arguments, output_budget)
        retained_tool_call_deltas = tool_call_deltas[:remaining_tool_calls]
        remaining_tool_calls -= len(retained_tool_call_deltas)
        if len(retained_tool_call_deltas) < len(tool_call_deltas):
            chunk_has_output = True
            if capture_output:
                output_budget.truncated = True
        for raw_tool_call in retained_tool_call_deltas:
            if chunk_has_output and state is None:
                break
            tool_call = _read_tool_call_delta(raw_tool_call, capture_budget, report_error)
            if tool_call.read_failed:
                if (
                    not had_state
                    and state is not None
                    and state.role is None
                    and state.content_fragments.tell() == 0
                    and state.refusal_fragments.tell() == 0
                    and state.function_call is None
                    and not state.tool_calls
                ):
                    states.pop(choice_index, None)
                break
            if tool_call.function_arguments or tool_call.custom_input:
                chunk_has_output = True
            if state is not None:
                _capture_tool_call_delta(state, tool_call, output_budget)

    if len(choice_items) > _CHAT_STREAM_CAPTURE_MAX_ITEMS:
        chunk_has_output = True
        if capture_output:
            output_budget.truncated = True
        finish_reason_budget.truncated = True

    response_id = _string(_own_field(chunk, "id", None, field_read_failed, report_error))
    response_model = _string(_own_field(chunk, "model", None, field_read_failed, report_error))
    usage = _chat_usage(_own_field(chunk, "usage", None, field_read_failed, report_error))
    if field_read_failed[0]:
        output_budget.truncated = True
        finish_reason_budget.truncated = True
    return {
        "response_id": response_id,
        "response_model": response_model,
        "usage": usage,
        "has_output": chunk_has_output,
    }


def _chat_message(state: _ChatChoice) -> dict[str, Any]:
    message: dict[str, Any] = {}
    if state.role_resolved:
        message["role"] = state.role if state.role is not None else "assistant"
    if state.content:
        message["content"] = state.content
    elif state.function_call is not None or state.tool_calls:
        message["content"] = None
    if state.refusal:
        message["refusal"] = state.refusal
    if state.function_call is not None:
        captured_function_call = dict(state.function_call)
        arguments = captured_function_call.get("arguments")
        if isinstance(arguments, StringIO):
            captured_function_call["arguments"] = arguments.getvalue()
        if "name" in state.unresolved_function_scalars:
            captured_function_call.pop("name", None)
        message["function_call"] = captured_function_call
    if state.tool_calls:
        tool_calls: list[dict[str, Any]] = []
        for index, tool_call in sorted(state.tool_calls.items()):
            captured_tool_call = dict(tool_call)
            unresolved = state.unresolved_tool_scalars.get(index, set())
            if "id" in unresolved:
                captured_tool_call.pop("id", None)
            if "type" in unresolved:
                captured_tool_call.pop("type", None)
            for payload_key, value_key in (("function", "arguments"), ("custom", "input")):
                raw_payload = tool_call.get(payload_key)
                if isinstance(raw_payload, Mapping):
                    captured_payload = dict(cast(Mapping[str, Any], raw_payload))
                    fragments = captured_payload.get(value_key)
                    if isinstance(fragments, StringIO):
                        captured_payload[value_key] = fragments.getvalue()
                else:
                    captured_payload = {}
                unresolved_name = f"{payload_key}.name"
                if unresolved_name in unresolved:
                    captured_payload.pop("name", None)
                if captured_payload or raw_payload is not None or unresolved_name in unresolved:
                    captured_tool_call[payload_key] = captured_payload
            tool_calls.append(captured_tool_call)
        message["tool_calls"] = tool_calls
    return message


def _chat_output(states: Mapping[int, _ChatChoice]) -> list[dict[str, Any]]:
    return [_chat_message(state) for _, state in sorted(states.items())]


def _chat_partial(
    states: Mapping[int, _ChatChoice],
    finish_reason_states: Mapping[int, str],
    usage: dict[str, int | float] | None,
    capture_output: bool,
    capture_truncated: bool,
) -> dict[str, Any]:
    finish_reasons = [reason for _, reason in sorted(finish_reason_states.items())]
    attributes = {
        **({"gen_ai.response.finish_reasons": finish_reasons} if len(finish_reasons) > 1 else {}),
        **({"telemetry.dev.capture.truncated": True} if capture_truncated else {}),
    }
    return {
        "output": _chat_output(states) if capture_output and states else None,
        "usage": usage,
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "attributes": attributes or None,
    }


def _synthetic_usage_chunk(chunk: Any) -> bool:
    choices = _sequence_items(_field(chunk, "choices"))
    return _field(chunk, "usage") is not None and len(choices) == 0


def _response_event_has_output(event: Any) -> bool:
    event_type = _string(_field(event, "type")) or ""
    if event_type not in {
        "response.output_text.delta",
        "response.refusal.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.custom_tool_call_input.delta",
        "response.code_interpreter_call_code.delta",
        "response.mcp_call_arguments.delta",
        "response.shell_call_command.delta",
        "response.output_audio.delta",
        "response.audio.delta",
        "response.audio.transcript.delta",
    }:
        return False
    delta = _field(event, "delta")
    return isinstance(delta, str) and bool(delta)


def _response_failed_error(response: Any) -> RuntimeError:
    error = _field(response, "error")
    if error is None:
        return RuntimeError("response.failed")
    code = _string(_field(error, "code"))
    message = _string(_field(error, "message"))
    if code and message:
        return RuntimeError(f"response.failed: {code}: {message}")
    if code:
        return RuntimeError(f"response.failed: {code}")
    if message:
        return RuntimeError(f"response.failed: {message}")
    return RuntimeError("response.failed")


def _response_stream_error(event: Any) -> RuntimeError:
    code = _string(_field(event, "code"))
    message = _string(_field(event, "message"))
    if code and message:
        return RuntimeError(f"response.error: {code}: {message}")
    if code:
        return RuntimeError(f"response.error: {code}")
    if message:
        return RuntimeError(f"response.error: {message}")
    return RuntimeError("response.error")


def _responses_event_failure(event: Any) -> RuntimeError | None:
    try:
        event_type = _field(event, "type")
    except Exception:
        return None
    if event_type == "response.failed":
        try:
            return _response_failed_error(_field(event, "response"))
        except Exception:
            return RuntimeError("response.failed")
    if event_type == "error":
        try:
            return _response_stream_error(event)
        except Exception:
            return RuntimeError("response.error")
    return None


def _hook_response_close(inner: Any, finish: Callable[[], None]) -> None:
    """End the span when the transport response is closed behind our back.

    OpenAI's stream managers (``chat.completions.stream()``, ``responses.stream()``)
    wrap the raw stream returned by ``create(stream=True)`` but close
    ``raw_stream.response`` directly on context-manager exit instead of calling the
    raw stream's ``close()``, which would otherwise leave the span open on early
    exit until garbage collection.
    """
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
        aclose_fn = cast(Callable[..., Awaitable[Any]], aclose)

        async def _aclose_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return await aclose_fn(*args, **kwargs)
            finally:
                finish()

        response.aclose = _aclose_hook


def _chat_span_capture_policy(handle: telemetry_dev.SpanHandle) -> tuple[bool, bool]:
    compatible = cast(Any, handle)
    client = getattr(compatible, "_client", None)
    state = getattr(compatible, "_state", None)
    capture_output = getattr(compatible, "capture_output", None)
    if not isinstance(capture_output, bool):
        capture_output = getattr(state, "capture_output", None)
    if not isinstance(capture_output, bool):
        capture_output = getattr(client, "capture_output", False)
    capture_masked = getattr(compatible, "capture_masked", None)
    if not isinstance(capture_masked, bool):
        capture_masked = getattr(client, "mask", _OMIT) is not None if client is not None else True
    return capture_output, capture_masked


def _chat_span_reporter(
    handle: telemetry_dev.SpanHandle,
) -> Callable[[BaseException], None]:
    compatible = cast(Any, handle)
    reporter = getattr(compatible, "report_error", None)
    client = getattr(compatible, "_client", None)
    reported = False

    def report(cause: BaseException) -> None:
        nonlocal reported
        if reported:
            return
        reported = True
        try:
            if callable(reporter):
                reporter(cause)
                return
            client_report = getattr(client, "report", None)
            if callable(client_report):
                client_report("provider instrumentation failed", cause)
        except Exception:
            return

    return report


def _end_mapped_response(
    handle: telemetry_dev.SpanHandle,
    end: Callable[..., None],
    mapper: ResponseMapper,
    response: Any,
) -> None:
    try:
        if mapper is _text_media_response:
            capture_output, _ = _chat_span_capture_policy(handle)
            fields = _text_media_response(response, capture_output=capture_output)
        else:
            fields = mapper(response)
    except Exception as exc:
        _chat_span_reporter(handle)(exc)
        end(attributes={"telemetry.dev.capture.truncated": True})
        return
    end(**fields)


class _InstrumentedStream:
    def __init__(
        self,
        inner: Any,
        handle: telemetry_dev.SpanHandle,
        injected_usage: bool,
        started_at: float,
    ) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._handle = handle
        self._injected_usage = injected_usage
        self._started_at = started_at
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._report_error = _chat_span_reporter(handle)
        self._states: dict[int, _ChatChoice] = {}
        self._finish_reason_states: dict[int, str] = {}
        self._finish_reason_reservations: dict[int, tuple[int, int]] = {}
        self._rejected_finish_reasons: set[int] = set()
        self._unterminated_choices: set[int] = set()
        self._usage: dict[str, int | float] | None = None
        self._saw_first = False
        self._completed_normally = False
        self._consume: Iterator[Any] | None = None
        self._in_next = False
        self._budget = _chat_capture_budget(reserve_output_list=self._capture_output)
        self._finish_reason_budget = _chat_capture_budget()
        _hook_response_close(inner, self._on_response_close)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = next(self._inner)
                    received_at = time.perf_counter()
                except StopIteration:
                    self._completed_normally = True
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record_safely(chunk, received_at)
                if self._injected_usage and self._is_synthetic_usage(chunk):
                    continue
                yield chunk
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
            self._end(**self._partial(), error=exc)
        self.close()

    def _partial(self) -> dict[str, Any]:
        output_incomplete = (
            not self._completed_normally
            or bool(self._unterminated_choices)
            or (
                self._capture_output
                and any(
                    not state.terminal
                    or not state.role_resolved
                    or bool(state.unresolved_function_scalars)
                    or any(state.unresolved_tool_scalars.values())
                    for state in self._states.values()
                )
            )
        )
        return _chat_partial(
            self._states,
            self._finish_reason_states,
            self._usage,
            self._capture_output
            and not (
                self._mask_output_when_incomplete and (self._budget.truncated or output_incomplete)
            ),
            self._budget.truncated
            or output_incomplete
            or self._finish_reason_budget.truncated
            or bool(self._rejected_finish_reasons),
        )

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

    def _record_safely(self, chunk: Any, received_at: float) -> None:
        try:
            self._record(chunk, received_at)
        except Exception as exc:
            self._budget.truncated = True
            self._finish_reason_budget.truncated = True
            self._report_error(exc)

    def _is_synthetic_usage(self, chunk: Any) -> bool:
        try:
            return _synthetic_usage_chunk(chunk)
        except Exception as exc:
            self._budget.truncated = True
            self._finish_reason_budget.truncated = True
            self._report_error(exc)
            return False

    def _record(self, chunk: Any, received_at: float) -> None:
        update = _record_chat_chunk(
            chunk,
            self._states,
            self._finish_reason_states,
            self._finish_reason_reservations,
            self._rejected_finish_reasons,
            self._budget,
            self._finish_reason_budget,
            self._capture_output,
            self._report_error,
            self._unterminated_choices,
        )
        if update["has_output"]:
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
                response_id=update.get("response_id"),
                response_model=update.get("response_model"),
            )
        if update.get("usage") is not None:
            self._usage = update["usage"]


class _InstrumentedAsyncStream:
    def __init__(
        self,
        inner: Any,
        handle: telemetry_dev.SpanHandle,
        injected_usage: bool,
        started_at: float,
    ) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._handle = handle
        self._injected_usage = injected_usage
        self._started_at = started_at
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._report_error = _chat_span_reporter(handle)
        self._states: dict[int, _ChatChoice] = {}
        self._finish_reason_states: dict[int, str] = {}
        self._finish_reason_reservations: dict[int, tuple[int, int]] = {}
        self._rejected_finish_reasons: set[int] = set()
        self._unterminated_choices: set[int] = set()
        self._usage: dict[str, int | float] | None = None
        self._saw_first = False
        self._completed_normally = False
        self._consume: AsyncIterator[Any] | None = None
        self._in_next = False
        self._budget = _chat_capture_budget(reserve_output_list=self._capture_output)
        self._finish_reason_budget = _chat_capture_budget()
        _hook_response_close(inner, self._on_response_close)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = await self._inner.__anext__()
                    received_at = time.perf_counter()
                except StopAsyncIteration:
                    self._completed_normally = True
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record_safely(chunk, received_at)
                if self._injected_usage and self._is_synthetic_usage(chunk):
                    continue
                yield chunk
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
            self._end(**self._partial(), error=exc)
        await self.close()

    def _partial(self) -> dict[str, Any]:
        output_incomplete = (
            not self._completed_normally
            or bool(self._unterminated_choices)
            or (
                self._capture_output
                and any(
                    not state.terminal
                    or not state.role_resolved
                    or bool(state.unresolved_function_scalars)
                    or any(state.unresolved_tool_scalars.values())
                    for state in self._states.values()
                )
            )
        )
        return _chat_partial(
            self._states,
            self._finish_reason_states,
            self._usage,
            self._capture_output
            and not (
                self._mask_output_when_incomplete and (self._budget.truncated or output_incomplete)
            ),
            self._budget.truncated
            or output_incomplete
            or self._finish_reason_budget.truncated
            or bool(self._rejected_finish_reasons),
        )

    def _finish(self) -> None:
        self._end(**self._partial())

    def _on_response_close(self) -> None:
        if not self._in_next:
            self._finish()

    async def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await cast(Awaitable[Any], result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record_safely(self, chunk: Any, received_at: float) -> None:
        try:
            self._record(chunk, received_at)
        except Exception as exc:
            self._budget.truncated = True
            self._finish_reason_budget.truncated = True
            self._report_error(exc)

    def _is_synthetic_usage(self, chunk: Any) -> bool:
        try:
            return _synthetic_usage_chunk(chunk)
        except Exception as exc:
            self._budget.truncated = True
            self._finish_reason_budget.truncated = True
            self._report_error(exc)
            return False

    def _record(self, chunk: Any, received_at: float) -> None:
        update = _record_chat_chunk(
            chunk,
            self._states,
            self._finish_reason_states,
            self._finish_reason_reservations,
            self._rejected_finish_reasons,
            self._budget,
            self._finish_reason_budget,
            self._capture_output,
            self._report_error,
            self._unterminated_choices,
        )
        if update["has_output"]:
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000,
                response_id=update.get("response_id"),
                response_model=update.get("response_model"),
            )
        if update.get("usage") is not None:
            self._usage = update["usage"]


def _mark_responses_capture_incomplete(fields: dict[str, Any], mask_output: bool) -> None:
    fields["attributes"] = {
        **cast(dict[str, Any], fields.get("attributes") or {}),
        "telemetry.dev.capture.truncated": True,
    }
    if mask_output:
        fields.pop("output", None)


class _InstrumentedResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._report_error = _chat_span_reporter(handle)
        self._started_at = started_at
        self._saw_first = False
        self._saw_terminal_snapshot = False
        self._partial: dict[str, Any] = {}
        self._retained_output: Any | None = None
        self._consume: Iterator[Any] | None = None
        self._in_next = False
        _hook_response_close(inner, self._on_response_close)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    event = next(self._inner)
                    received_at = time.perf_counter()
                except StopIteration:
                    break
                except BaseException as exc:
                    _mark_responses_capture_incomplete(
                        self._partial, self._mask_output_when_incomplete
                    )
                    self._end(**self._partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                try:
                    self._record(event, received_at)
                except Exception as exc:
                    self._report_error(exc)
                    _mark_responses_capture_incomplete(
                        self._partial, self._mask_output_when_incomplete
                    )
                    failure = _responses_event_failure(event)
                    if failure is not None:
                        self._end(**self._partial, error=failure)
                yield event
        finally:
            self.close()

    def __iter__(self) -> Iterator[Any]:
        return self._iterate()

    def __next__(self) -> Any:
        if self._consume is None:
            self._consume = self._iterate()
        return next(self._consume)

    def __enter__(self) -> _InstrumentedResponsesStream:
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc is not None:
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=exc)
        self.close()

    def _finish(self) -> None:
        if not self._saw_terminal_snapshot:
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
        self._end(**self._partial)

    def _on_response_close(self) -> None:
        # Mid-iteration closes are part of error/exhaustion unwinding inside
        # next(); those paths must win the end race to record the right status.
        if not self._in_next:
            self._finish()

    def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, event: Any, received_at: float) -> None:
        if _response_event_has_output(event):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        event_type = _field(event, "type")
        response = _field(event, "response")
        if response is not None:
            # Streams keep error out of the partial: their end paths pass an
            # explicit error= kwarg, which must not collide with mapped fields.
            fields = _responses_response(response, include_error=False, include_output=False)
            raw_output = _field(response, "output")
            output = None
            truncated = False
            if raw_output is not None:
                output, truncated = _bounded_responses_capture(
                    raw_output,
                    capture_enabled=self._capture_output,
                    max_bytes=_CHAT_STREAM_CAPTURE_MAX_BYTES,
                    report_error=self._report_error,
                )
                if not truncated and output:
                    self._retained_output = output
            self._partial = fields
            if self._retained_output is not None and not (
                truncated and self._mask_output_when_incomplete
            ):
                self._partial["output"] = self._retained_output
            elif output is not None:
                self._partial["output"] = output
            if truncated:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                self._saw_terminal_snapshot = True
        if event_type == "response.completed":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial)
        elif event_type == "response.failed":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=_response_failed_error(response))
        elif event_type == "response.incomplete":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial)
        elif event_type == "error":
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=_response_stream_error(event))


class _InstrumentedAsyncResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._report_error = _chat_span_reporter(handle)
        self._started_at = started_at
        self._saw_first = False
        self._saw_terminal_snapshot = False
        self._partial: dict[str, Any] = {}
        self._retained_output: Any | None = None
        self._consume: AsyncIterator[Any] | None = None
        self._in_next = False
        _hook_response_close(inner, self._on_response_close)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    event = await self._inner.__anext__()
                    received_at = time.perf_counter()
                except StopAsyncIteration:
                    break
                except BaseException as exc:
                    _mark_responses_capture_incomplete(
                        self._partial, self._mask_output_when_incomplete
                    )
                    self._end(**self._partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                try:
                    self._record(event, received_at)
                except Exception as exc:
                    self._report_error(exc)
                    _mark_responses_capture_incomplete(
                        self._partial, self._mask_output_when_incomplete
                    )
                    failure = _responses_event_failure(event)
                    if failure is not None:
                        self._end(**self._partial, error=failure)
                yield event
        finally:
            await self.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._aiterate()

    async def __anext__(self) -> Any:
        if self._consume is None:
            self._consume = self._aiterate()
        return await self._consume.__anext__()

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
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=exc)
        await self.close()

    def _finish(self) -> None:
        if not self._saw_terminal_snapshot:
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
        self._end(**self._partial)

    def _on_response_close(self) -> None:
        # Mid-iteration closes are part of error/exhaustion unwinding inside
        # __anext__(); those paths must win the end race to record the right status.
        if not self._in_next:
            self._finish()

    async def close(self) -> None:
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await cast(Awaitable[Any], result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _record(self, event: Any, received_at: float) -> None:
        if _response_event_has_output(event):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        event_type = _field(event, "type")
        response = _field(event, "response")
        if response is not None:
            # Streams keep error out of the partial: their end paths pass an
            # explicit error= kwarg, which must not collide with mapped fields.
            fields = _responses_response(response, include_error=False, include_output=False)
            raw_output = _field(response, "output")
            output = None
            truncated = False
            if raw_output is not None:
                output, truncated = _bounded_responses_capture(
                    raw_output,
                    capture_enabled=self._capture_output,
                    max_bytes=_CHAT_STREAM_CAPTURE_MAX_BYTES,
                    report_error=self._report_error,
                )
                if not truncated and output:
                    self._retained_output = output
            self._partial = fields
            if self._retained_output is not None and not (
                truncated and self._mask_output_when_incomplete
            ):
                self._partial["output"] = self._retained_output
            elif output is not None:
                self._partial["output"] = output
            if truncated:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                self._saw_terminal_snapshot = True
        if event_type == "response.completed":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial)
        elif event_type == "response.failed":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=_response_failed_error(response))
        elif event_type == "response.incomplete":
            if not self._saw_terminal_snapshot:
                _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial)
        elif event_type == "error":
            _mark_responses_capture_incomplete(self._partial, self._mask_output_when_incomplete)
            self._end(**self._partial, error=_response_stream_error(event))


def _media_stream_event_fields(
    event: Any,
    response_mapper: ResponseMapper,
    capture_output: bool,
) -> dict[str, Any]:
    if response_mapper is not _text_media_response:
        return response_mapper(event)
    fields = _text_media_usage(event, _media_response(event))
    if capture_output and _string(_field(event, "type")) == "transcript.text.done":
        text = _string(_field(event, "text"))
        if text is not None:
            budget = _transcript_capture_budget()
            fields["output"] = _capture_text(text, budget)
            if budget.truncated:
                fields["attributes"] = {"telemetry.dev.capture.truncated": True}
    return fields


def _media_stream_end_fields(
    fields: Mapping[str, Any],
    response_mapper: ResponseMapper,
    text_parts: Sequence[str],
    budget: telemetry_dev.CaptureBudget,
    mask_output_when_incomplete: bool,
    stream_incomplete: bool,
) -> dict[str, Any]:
    result = dict(fields)
    capture_incomplete = stream_incomplete
    if response_mapper is _text_media_response:
        attributes = result.get("attributes")
        terminal_truncated = (
            isinstance(attributes, Mapping)
            and cast(Mapping[str, Any], attributes).get("telemetry.dev.capture.truncated") is True
        )
        if "output" not in result:
            capture_incomplete = capture_incomplete or budget.truncated
            if text_parts and not (mask_output_when_incomplete and capture_incomplete):
                result["output"] = "".join(text_parts)
        else:
            capture_incomplete = capture_incomplete or terminal_truncated
        if mask_output_when_incomplete and capture_incomplete:
            result.pop("output", None)
    if capture_incomplete:
        attributes = result.get("attributes")
        result["attributes"] = (
            dict(cast(Mapping[str, Any], attributes)) if isinstance(attributes, Mapping) else {}
        ) | {"telemetry.dev.capture.truncated": True}
    return result


def _media_stream_error(event: Any, event_type: str) -> BaseException | None:
    raw_error = _field(event, "error")
    if isinstance(raw_error, BaseException):
        return raw_error
    if raw_error is None and not event_type.endswith(".failed"):
        return None
    message = (
        _string(_field(raw_error, "message"))
        or _string(_field(event, "message"))
        or f"OpenAI media stream ended with {event_type}"
    )
    code = _string(_field(raw_error, "code")) or _string(_field(event, "code"))
    return RuntimeError(f"{code}: {message}" if code else message)


def _media_event_failure(event: Any) -> BaseException | None:
    try:
        event_type = _string(_field(event, "type")) or ""
    except Exception:
        return None
    if event_type not in _MEDIA_TERMINAL_EVENTS:
        return None
    try:
        return _media_stream_error(event, event_type)
    except Exception:
        if event_type.endswith(".failed"):
            return RuntimeError(f"OpenAI media stream ended with {event_type}")
        return None


_MEDIA_TERMINAL_EVENTS = frozenset(
    {
        "image_generation.completed",
        "image_generation.failed",
        "image_edit.completed",
        "image_edit.failed",
        "transcript.text.done",
    }
)


class _InstrumentedMediaStream:
    def __init__(
        self, inner: Any, handle: telemetry_dev.SpanHandle, response_mapper: ResponseMapper
    ) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._fields: dict[str, Any] = {}
        self._response_mapper = response_mapper
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._budget = _transcript_capture_budget()
        self._text_parts: list[str] = []
        self._saw_delta = False
        self._saw_terminal = False
        self._report_error = _chat_span_reporter(handle)
        self._mapping_failed = False

    def __iter__(self) -> _InstrumentedMediaStream:
        return self

    def __enter__(self) -> _InstrumentedMediaStream:
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        exit_method = getattr(self._inner, "__exit__", None)
        if exc is not None:
            self._end(**self._finish_fields(), error=exc)
        try:
            if callable(exit_method):
                suppressed = bool(exit_method(exc_type, exc, tb))
            else:
                close = getattr(self._inner, "close", None)
                if callable(close):
                    close()
                suppressed = False
        except BaseException as exit_error:
            self._end(**self._finish_fields(), error=exit_error)
            raise
        else:
            self._end(**self._finish_fields())
            return suppressed

    def __next__(self) -> Any:
        try:
            event = next(self._inner)
        except StopIteration:
            self.close()
            raise
        except BaseException as exc:
            self._end(**self._finish_fields(), error=exc)
            raise
        self._record_safely(event)
        return event

    def _record_safely(self, event: Any) -> None:
        try:
            self._record(event)
        except Exception as exc:
            self._report_error(exc)
            self._mapping_failed = True
            failure = _media_event_failure(event)
            if failure is not None:
                self._end(**self._finish_fields(), error=failure)

    def _record(self, event: Any) -> None:
        event_type = _string(_field(event, "type")) or ""
        if event_type in _MEDIA_TERMINAL_EVENTS:
            self._saw_terminal = True
        mapped = _media_stream_event_fields(
            event,
            self._response_mapper,
            self._capture_output,
        )
        if "output" in mapped:
            self._fields = _clean_fields(mapped)
        else:
            self._fields.update(_clean_fields(mapped))
        if self._capture_output and self._response_mapper is _text_media_response:
            text: str | None = None
            if event_type == "transcript.text.delta":
                self._saw_delta = True
                text = _string(_field(event, "delta"))
            elif event_type == "transcript.text.segment" and not self._saw_delta:
                text = _string(_field(event, "text"))
            if text:
                captured = _capture_text(text, self._budget)
                if captured:
                    self._text_parts.append(captured)
        if event_type in _MEDIA_TERMINAL_EVENTS:
            error = _media_stream_error(event, event_type)
            self._end(**self._finish_fields(), **({"error": error} if error is not None else {}))

    def _finish_fields(self) -> dict[str, Any]:
        return _media_stream_end_fields(
            self._fields,
            self._response_mapper,
            self._text_parts,
            self._budget,
            self._mask_output_when_incomplete,
            not self._saw_terminal or self._mapping_failed,
        )

    def close(self) -> None:
        self._end(**self._finish_fields())
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InstrumentedAsyncMediaStream:
    def __init__(
        self, inner: Any, handle: telemetry_dev.SpanHandle, response_mapper: ResponseMapper
    ) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._fields: dict[str, Any] = {}
        self._response_mapper = response_mapper
        (
            self._capture_output,
            self._mask_output_when_incomplete,
        ) = _chat_span_capture_policy(handle)
        self._budget = _transcript_capture_budget()
        self._text_parts: list[str] = []
        self._saw_delta = False
        self._saw_terminal = False
        self._report_error = _chat_span_reporter(handle)
        self._mapping_failed = False

    def __aiter__(self) -> _InstrumentedAsyncMediaStream:
        return self

    async def __aenter__(self) -> _InstrumentedAsyncMediaStream:
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            result = enter()
            if hasattr(result, "__await__"):
                await cast(Awaitable[Any], result)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        exit_method = getattr(self._inner, "__aexit__", None)
        if exc is not None:
            self._end(**self._finish_fields(), error=exc)
        try:
            if callable(exit_method):
                result = exit_method(exc_type, exc, tb)
                if hasattr(result, "__await__"):
                    result = await cast(Awaitable[Any], result)
                suppressed = bool(result)
            else:
                close = getattr(self._inner, "close", None)
                if callable(close):
                    result = close()
                    if hasattr(result, "__await__"):
                        await cast(Awaitable[Any], result)
                suppressed = False
        except BaseException as exit_error:
            self._end(**self._finish_fields(), error=exit_error)
            raise
        else:
            self._end(**self._finish_fields())
            return suppressed

    async def __anext__(self) -> Any:
        try:
            event = await self._inner.__anext__()
        except StopAsyncIteration:
            await self.close()
            raise
        except BaseException as exc:
            self._end(**self._finish_fields(), error=exc)
            raise
        self._record_safely(event)
        return event

    def _record_safely(self, event: Any) -> None:
        try:
            self._record(event)
        except Exception as exc:
            self._report_error(exc)
            self._mapping_failed = True
            failure = _media_event_failure(event)
            if failure is not None:
                self._end(**self._finish_fields(), error=failure)

    def _record(self, event: Any) -> None:
        event_type = _string(_field(event, "type")) or ""
        if event_type in _MEDIA_TERMINAL_EVENTS:
            self._saw_terminal = True
        mapped = _media_stream_event_fields(
            event,
            self._response_mapper,
            self._capture_output,
        )
        if "output" in mapped:
            self._fields = _clean_fields(mapped)
        else:
            self._fields.update(_clean_fields(mapped))
        if self._capture_output and self._response_mapper is _text_media_response:
            text: str | None = None
            if event_type == "transcript.text.delta":
                self._saw_delta = True
                text = _string(_field(event, "delta"))
            elif event_type == "transcript.text.segment" and not self._saw_delta:
                text = _string(_field(event, "text"))
            if text:
                captured = _capture_text(text, self._budget)
                if captured:
                    self._text_parts.append(captured)
        if event_type in _MEDIA_TERMINAL_EVENTS:
            error = _media_stream_error(event, event_type)
            self._end(**self._finish_fields(), **({"error": error} if error is not None else {}))

    def _finish_fields(self) -> dict[str, Any]:
        return _media_stream_end_fields(
            self._fields,
            self._response_mapper,
            self._text_parts,
            self._budget,
            self._mask_output_when_incomplete,
            not self._saw_terminal or self._mapping_failed,
        )

    async def close(self) -> None:
        self._end(**self._finish_fields())
        close = getattr(self._inner, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await cast(Awaitable[Any], result)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _inject_chat_usage(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    current = dict(kwargs)
    stream_options = current.get("stream_options")
    options: dict[str, Any] = (
        dict(cast(Mapping[str, Any], stream_options)) if isinstance(stream_options, Mapping) else {}
    )
    if options.get("include_usage") is True:
        return current, False
    options["include_usage"] = True
    current["stream_options"] = options
    return current, True


def _start_span(
    params: Mapping[str, Any], mapper: RequestMapper, provider: str
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float]:
    name, fields = mapper(_sent_params(params))
    handle = telemetry_dev.start_span(name, provider=provider, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter()


def _wrap_sync(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs: dict[str, Any] = dict(kwargs)
        streaming = call_kwargs.get("stream") is True
        injected = False
        if operation == "chat" and streaming and inject_usage:
            call_kwargs, injected = _inject_chat_usage(call_kwargs)
        handle, end, started_at = _start_span(call_kwargs, request_mapper, provider(resource))
        try:
            result = original(*args, **call_kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming and operation == "chat":
            return _InstrumentedStream(result, handle, injected, started_at)
        if streaming and operation == "responses":
            return _InstrumentedResponsesStream(result, handle, started_at)
        if streaming and operation == "media":
            return _InstrumentedMediaStream(result, handle, response_mapper)
        _end_mapped_response(handle, end, response_mapper, result)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        call_kwargs: dict[str, Any] = dict(kwargs)
        streaming = call_kwargs.get("stream") is True
        injected = False
        if operation == "chat" and streaming and inject_usage:
            call_kwargs, injected = _inject_chat_usage(call_kwargs)
        handle, end, started_at = _start_span(call_kwargs, request_mapper, provider(resource))
        try:
            result = await original(*args, **call_kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        if streaming and operation == "chat":
            return _InstrumentedAsyncStream(result, handle, injected, started_at)
        if streaming and operation == "responses":
            return _InstrumentedAsyncResponsesStream(result, handle, started_at)
        if streaming and operation == "media":
            return _InstrumentedAsyncMediaStream(result, handle, response_mapper)
        _end_mapped_response(handle, end, response_mapper, result)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_sync_batch(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    del operation, inject_usage

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args and hasattr(args[0], "_client") else None
        handle, end, _started_at = _start_span(
            _batch_params(args, kwargs), request_mapper, provider(resource)
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        _end_mapped_response(handle, end, response_mapper, result)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_batch(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    del operation, inject_usage

    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args and hasattr(args[0], "_client") else None
        handle, end, _started_at = _start_span(
            _batch_params(args, kwargs), request_mapper, provider(resource)
        )
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        _end_mapped_response(handle, end, response_mapper, result)
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _responses_retrieve_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "chat unknown",
        _clean_fields(
            {
                "type": "generation",
                "response_id": _string(params.get("response_id")),
            }
        ),
    )


def _retrieve_response_id(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if "response_id" in kwargs:
        return kwargs["response_id"]
    if args and hasattr(args[0], "_client"):
        return args[1] if len(args) > 1 else None
    return args[0] if args else None


def _wrap_sync_retrieve(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    del operation, response_mapper, inject_usage

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is not True:
            return original(*args, **kwargs)
        resource = args[0] if args and hasattr(args[0], "_client") else None
        params = {**kwargs, "response_id": _retrieve_response_id(args, kwargs)}
        handle, end, started_at = _start_span(params, request_mapper, provider(resource))
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        return _InstrumentedResponsesStream(result, handle, started_at)

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_retrieve(
    original: Callable[..., Any],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    provider: ProviderResolver,
    inject_usage: bool,
) -> Callable[..., Any]:
    del operation, response_mapper, inject_usage

    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is not True:
            return await original(*args, **kwargs)
        resource = args[0] if args and hasattr(args[0], "_client") else None
        params = {**kwargs, "response_id": _retrieve_response_id(args, kwargs)}
        handle, end, started_at = _start_span(params, request_mapper, provider(resource))
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        return _InstrumentedAsyncResponsesStream(result, handle, started_at)

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _patch_instance(
    resource: object,
    method: str,
    wrapper_factory: Callable[
        [Callable[..., Any], str, RequestMapper, ResponseMapper, ProviderResolver, bool],
        Callable[..., Any],
    ],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    inject_usage: bool,
) -> None:
    current = getattr(resource, method)
    if getattr(current, _WRAPPED_ATTR, False):
        # An instance-level wrapper means this resource is already wrapped. A
        # wrapper inherited from instrument_openai()'s class patch must still be
        # shadowed by an instance wrapper over the underlying original, so the
        # client stays instrumented after uninstrument_openai() restores the class.
        if method in vars(resource):
            return
        original = getattr(current, _ORIGINAL_ATTR, None)
        if original is None:
            return
        current = original.__get__(resource, type(resource))
    wrapped = wrapper_factory(
        current,
        operation,
        request_mapper,
        response_mapper,
        lambda _: _provider_for_resource(resource),
        inject_usage,
    )
    setattr(resource, method, wrapped)


def _patch_class(
    cls: type[Any],
    method: str,
    wrapper_factory: Callable[
        [Callable[..., Any], str, RequestMapper, ResponseMapper, ProviderResolver, bool],
        Callable[..., Any],
    ],
    operation: str,
    request_mapper: RequestMapper,
    response_mapper: ResponseMapper,
    inject_usage: bool,
) -> None:
    original = getattr(cls, method)
    if getattr(original, _WRAPPED_ATTR, False):
        return
    wrapped = wrapper_factory(
        original,
        operation,
        request_mapper,
        response_mapper,
        _provider_for_resource,
        inject_usage,
    )
    _ORIGINALS.append((cls, method, original, wrapped))
    setattr(cls, method, wrapped)


def wrap_openai(client: _T, *, inject_stream_usage: bool = False) -> _T:
    if getattr(client, _WRAPPED_ATTR, False):
        return client
    dynamic_client = cast(Any, client)
    async_client = isinstance(client, openai.AsyncOpenAI)
    wrapper_factory = _wrap_async if async_client else _wrap_sync
    _patch_instance(
        client.chat.completions,  # type: ignore[attr-defined]
        "create",
        wrapper_factory,
        "chat",
        _chat_request,
        _chat_response,
        inject_stream_usage,
    )
    _patch_instance(
        client.chat.completions,  # type: ignore[attr-defined]
        "parse",
        wrapper_factory,
        "chat",
        _chat_request,
        _chat_response,
        inject_stream_usage,
    )
    _patch_instance(
        client.responses,  # type: ignore[attr-defined]
        "create",
        wrapper_factory,
        "responses",
        _responses_request,
        _responses_response,
        inject_stream_usage,
    )
    _patch_instance(
        client.responses,  # type: ignore[attr-defined]
        "retrieve",
        _wrap_async_retrieve if async_client else _wrap_sync_retrieve,
        "responses",
        _responses_retrieve_request,
        _responses_response,
        inject_stream_usage,
    )
    _patch_instance(
        client.responses,  # type: ignore[attr-defined]
        "parse",
        wrapper_factory,
        "responses",
        _responses_request,
        _responses_response,
        inject_stream_usage,
    )
    _patch_instance(
        client.embeddings,  # type: ignore[attr-defined]
        "create",
        wrapper_factory,
        "embeddings",
        _embeddings_request,
        _embeddings_response,
        inject_stream_usage,
    )
    for method in ("generate", "edit", "create_variation"):
        _patch_instance(
            dynamic_client.images,
            method,
            wrapper_factory,
            "media",
            _media_request("image", "image"),
            _media_response,
            False,
        )  # type: ignore[attr-defined]
    _patch_instance(
        dynamic_client.audio.speech,
        "create",
        wrapper_factory,
        "media",
        _media_request("speech", "speech"),
        _media_response,
        False,
    )  # type: ignore[attr-defined]
    for resource, endpoint in (
        (dynamic_client.audio.transcriptions, "transcription"),
        (dynamic_client.audio.translations, "translation"),
    ):
        _patch_instance(
            resource,
            "create",
            wrapper_factory,
            "media",
            _media_request(endpoint, "text"),
            _text_media_response,
            False,
        )
    batch_wrapper = _wrap_async_batch if async_client else _wrap_sync_batch
    for method in ("create", "retrieve", "cancel"):
        _patch_instance(
            dynamic_client.batches,
            method,
            batch_wrapper,
            "batch",
            _batch_request(method),
            _batch_response,
            False,
        )
    setattr(client, _WRAPPED_ATTR, True)
    return client


def instrument_openai(*, inject_stream_usage: bool = False) -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        for cls, wrapper_factory in (
            (Completions, _wrap_sync),
            (AsyncCompletions, _wrap_async),
        ):
            for method in ("create", "parse"):
                _patch_class(
                    cls,
                    method,
                    wrapper_factory,
                    "chat",
                    _chat_request,
                    _chat_response,
                    inject_stream_usage,
                )
        for cls, wrapper_factory, retrieve_factory in (
            (Responses, _wrap_sync, _wrap_sync_retrieve),
            (AsyncResponses, _wrap_async, _wrap_async_retrieve),
        ):
            for method, factory, request_mapper in (
                ("create", wrapper_factory, _responses_request),
                ("retrieve", retrieve_factory, _responses_retrieve_request),
                ("parse", wrapper_factory, _responses_request),
            ):
                _patch_class(
                    cls,
                    method,
                    factory,
                    "responses",
                    request_mapper,
                    _responses_response,
                    inject_stream_usage,
                )
        for cls, wrapper_factory in (
            (Embeddings, _wrap_sync),
            (AsyncEmbeddings, _wrap_async),
        ):
            _patch_class(
                cls,
                "create",
                wrapper_factory,
                "embeddings",
                _embeddings_request,
                _embeddings_response,
                inject_stream_usage,
            )
        for cls, wrapper_factory in ((Images, _wrap_sync), (AsyncImages, _wrap_async)):
            for method in ("generate", "edit", "create_variation"):
                _patch_class(
                    cls,
                    method,
                    wrapper_factory,
                    "media",
                    _media_request("image", "image"),
                    _media_response,
                    False,
                )
        for cls, wrapper_factory, endpoint, output_type in (
            (Speech, _wrap_sync, "speech", "speech"),
            (AsyncSpeech, _wrap_async, "speech", "speech"),
            (Transcriptions, _wrap_sync, "transcription", "text"),
            (AsyncTranscriptions, _wrap_async, "transcription", "text"),
            (Translations, _wrap_sync, "translation", "text"),
            (AsyncTranslations, _wrap_async, "translation", "text"),
        ):
            _patch_class(
                cls,
                "create",
                wrapper_factory,
                "media",
                _media_request(endpoint, output_type),
                _text_media_response if output_type == "text" else _media_response,
                False,
            )
        for cls, wrapper_factory in (
            (Batches, _wrap_sync_batch),
            (AsyncBatches, _wrap_async_batch),
        ):
            for method in ("create", "retrieve", "cancel"):
                _patch_class(
                    cls,
                    method,
                    wrapper_factory,
                    "batch",
                    _batch_request(method),
                    _batch_response,
                    False,
                )
        _installed = True


def uninstrument_openai() -> None:
    global _installed
    with _install_lock:
        while _ORIGINALS:
            cls, method, original, wrapped = _ORIGINALS.pop()
            if getattr(cls, method) is wrapped:
                setattr(cls, method, original)
        _installed = False


__all__ = ["__version__", "instrument_openai", "uninstrument_openai", "wrap_openai"]
