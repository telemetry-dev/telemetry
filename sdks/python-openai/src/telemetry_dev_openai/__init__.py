from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import wraps
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

__version__ = "0.1.4"

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


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[Any], value))
    return []


def _bounded_responses_capture(
    value: Any,
    capture: Literal["input", "output"] = "output",
    parent_type: str | None = None,
) -> tuple[Any | None, bool]:
    client = telemetry_dev.get_client()
    enabled = client is not None and (
        client.capture_input if capture == "input" else client.capture_output
    )
    if not enabled:
        return None, False
    budget = telemetry_dev.CaptureBudget.from_client()
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
    except Exception:
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


def _capture_text(value: str, budget: telemetry_dev.CaptureBudget) -> str:
    if not value or budget.truncated:
        return ""
    if budget.items_used >= budget.max_items or budget.remaining_bytes <= 0:
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
    budget.items_used += 1
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


def _text_media_response(response: Any) -> dict[str, Any]:
    fields = _media_response(response)
    client = telemetry_dev.get_client()
    if client is not None and client.capture_output:
        text = _string(_field(response, "text"))
        if text is not None:
            budget = telemetry_dev.CaptureBudget.from_client()
            fields["output"] = _capture_text(text, budget)
            if budget.truncated:
                fields["attributes"] = {"telemetry.dev.capture.truncated": True}
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
        self.content = ""
        self.refusal = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None


def _choice_state(states: dict[int, _ChatChoice], index: int) -> _ChatChoice:
    if index not in states:
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
        state = _choice_state(states, index if isinstance(index, int) else 0)
        delta = _field(choice, "delta")
        role = _field(delta, "role")
        content = _field(delta, "content")
        refusal = _field(delta, "refusal")
        if isinstance(role, str):
            state.role = role
        if isinstance(content, str) and budget.accept(content):
            state.content += content
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
    }


def _chat_output(states: Mapping[int, _ChatChoice]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _, state in sorted(states.items()):
        message: dict[str, Any] = {"role": state.role or "assistant"}
        if state.content:
            message["content"] = state.content
        elif state.tool_calls:
            message["content"] = None
        if state.refusal:
            message["refusal"] = state.refusal
        if state.tool_calls:
            message["tool_calls"] = [call for _, call in sorted(state.tool_calls.items())]
        output.append(message)
    return output


def _chat_partial(
    states: Mapping[int, _ChatChoice], usage: dict[str, int | float] | None
) -> dict[str, Any]:
    finish_reasons = [
        state.finish_reason
        for _, state in sorted(states.items())
        if state.finish_reason is not None
    ]
    return {
        "output": _chat_output(states) if states else None,
        "usage": usage,
        "finish_reason": finish_reasons[0] if finish_reasons else None,
        "attributes": (
            {"gen_ai.response.finish_reasons": finish_reasons} if len(finish_reasons) > 1 else None
        ),
    }


def _synthetic_usage_chunk(chunk: Any) -> bool:
    choices = _sequence_items(_field(chunk, "choices"))
    return _field(chunk, "usage") is not None and len(choices) == 0


def _chat_chunk_has_output(chunk: Any) -> bool:
    for choice in _sequence_items(_field(chunk, "choices")):
        delta = _field(choice, "delta")
        for key in ("content", "refusal"):
            value = _field(delta, key)
            if isinstance(value, str) and value:
                return True
        audio = _field(delta, "audio")
        if isinstance(_field(audio, "data"), str) and _field(audio, "data"):
            return True
        legacy_arguments = _field(_field(delta, "function_call"), "arguments")
        if isinstance(legacy_arguments, str) and legacy_arguments:
            return True
        for tool_call in _sequence_items(_field(delta, "tool_calls")):
            arguments = _field(_field(tool_call, "function"), "arguments")
            if isinstance(arguments, str) and arguments:
                return True
    return False


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
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
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
                    received_at = time.perf_counter()
                except StopIteration:
                    break
                except BaseException as exc:
                    self._end(**_chat_partial(self._states, self._usage), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(chunk, received_at)
                if self._injected_usage and _synthetic_usage_chunk(chunk):
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
            self._end(**_chat_partial(self._states, self._usage), error=exc)
        self.close()

    def _finish(self) -> None:
        self._end(**_chat_partial(self._states, self._usage))

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

    def _record(self, chunk: Any, received_at: float) -> None:
        update = _record_chat_chunk(chunk, self._states, self._budget)
        if _chat_chunk_has_output(chunk):
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
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
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
                    received_at = time.perf_counter()
                except StopAsyncIteration:
                    break
                except BaseException as exc:
                    self._end(**_chat_partial(self._states, self._usage), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(chunk, received_at)
                if self._injected_usage and _synthetic_usage_chunk(chunk):
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
            self._end(**_chat_partial(self._states, self._usage), error=exc)
        await self.close()

    def _finish(self) -> None:
        self._end(**_chat_partial(self._states, self._usage))

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

    def _record(self, chunk: Any, received_at: float) -> None:
        update = _record_chat_chunk(chunk, self._states, self._budget)
        if _chat_chunk_has_output(chunk):
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


class _InstrumentedResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._saw_first = False
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
                    self._end(**self._partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event, received_at)
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
            self._end(**self._partial, error=exc)
        self.close()

    def _finish(self) -> None:
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
        response = _field(event, "response")
        if response is not None:
            # Streams keep error out of the partial: their end paths pass an
            # explicit error= kwarg, which must not collide with mapped fields.
            fields = _responses_response(response, include_error=False, include_output=False)
            raw_output = _field(response, "output")
            output = None
            truncated = False
            if raw_output is not None:
                output, truncated = _bounded_responses_capture(raw_output)
                if not truncated and output:
                    self._retained_output = output
            self._partial = fields
            if self._retained_output is not None:
                self._partial["output"] = self._retained_output
            elif output is not None:
                self._partial["output"] = output
            if truncated:
                self._partial["attributes"] = {"telemetry.dev.capture.truncated": True}
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._partial)
        elif event_type == "response.failed":
            self._end(**self._partial, error=_response_failed_error(response))
        elif event_type == "response.incomplete":
            self._end(**self._partial)
        elif event_type == "error":
            self._end(**self._partial, error=_response_stream_error(event))


class _InstrumentedAsyncResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._end = _end_once(handle)
        self._started_at = started_at
        self._saw_first = False
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
                    self._end(**self._partial, error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event, received_at)
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
            self._end(**self._partial, error=exc)
        await self.close()

    def _finish(self) -> None:
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
        response = _field(event, "response")
        if response is not None:
            # Streams keep error out of the partial: their end paths pass an
            # explicit error= kwarg, which must not collide with mapped fields.
            fields = _responses_response(response, include_error=False, include_output=False)
            raw_output = _field(response, "output")
            output = None
            truncated = False
            if raw_output is not None:
                output, truncated = _bounded_responses_capture(raw_output)
                if not truncated and output:
                    self._retained_output = output
            self._partial = fields
            if self._retained_output is not None:
                self._partial["output"] = self._retained_output
            elif output is not None:
                self._partial["output"] = output
            if truncated:
                self._partial["attributes"] = {"telemetry.dev.capture.truncated": True}
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._partial)
        elif event_type == "response.failed":
            self._end(**self._partial, error=_response_failed_error(response))
        elif event_type == "response.incomplete":
            self._end(**self._partial)
        elif event_type == "error":
            self._end(**self._partial, error=_response_stream_error(event))


def _media_stream_event_fields(
    event: Any, response_mapper: ResponseMapper, capture_output: bool
) -> dict[str, Any]:
    if response_mapper is not _text_media_response:
        return response_mapper(event)
    fields = _text_media_usage(event, _media_response(event))
    if capture_output and _string(_field(event, "type")) == "transcript.text.done":
        text = _string(_field(event, "text"))
        if text is not None:
            budget = telemetry_dev.CaptureBudget.from_client()
            fields["output"] = _capture_text(text, budget)
            if budget.truncated:
                fields["attributes"] = {"telemetry.dev.capture.truncated": True}
    return fields


def _media_stream_end_fields(
    fields: Mapping[str, Any],
    response_mapper: ResponseMapper,
    text_parts: Sequence[str],
    budget: telemetry_dev.CaptureBudget,
) -> dict[str, Any]:
    result = dict(fields)
    if response_mapper is _text_media_response and "output" not in result:
        if text_parts:
            result["output"] = "".join(text_parts)
        if budget.truncated:
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


class _InstrumentedMediaStream:
    def __init__(
        self, inner: Any, handle: telemetry_dev.SpanHandle, response_mapper: ResponseMapper
    ) -> None:
        self._inner = inner
        self._end = _end_once(handle)
        self._fields: dict[str, Any] = {}
        self._response_mapper = response_mapper
        client = telemetry_dev.get_client()
        self._capture_output = client is not None and client.capture_output
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._text_parts: list[str] = []
        self._saw_delta = False

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
        self._record(event)
        return event

    def _record(self, event: Any) -> None:
        event_type = _string(_field(event, "type")) or ""
        mapped = _media_stream_event_fields(event, self._response_mapper, self._capture_output)
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
        if event_type in {
            "image_generation.completed",
            "image_generation.failed",
            "image_edit.completed",
            "image_edit.failed",
            "transcript.text.done",
        }:
            error = _media_stream_error(event, event_type)
            self._end(**self._finish_fields(), **({"error": error} if error is not None else {}))

    def _finish_fields(self) -> dict[str, Any]:
        return _media_stream_end_fields(
            self._fields, self._response_mapper, self._text_parts, self._budget
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
        client = telemetry_dev.get_client()
        self._capture_output = client is not None and client.capture_output
        self._budget = telemetry_dev.CaptureBudget.from_client()
        self._text_parts: list[str] = []
        self._saw_delta = False

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
        event_type = _string(_field(event, "type")) or ""
        mapped = _media_stream_event_fields(event, self._response_mapper, self._capture_output)
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
        if event_type in {
            "image_generation.completed",
            "image_generation.failed",
            "image_edit.completed",
            "image_edit.failed",
            "transcript.text.done",
        }:
            error = _media_stream_error(event, event_type)
            self._end(**self._finish_fields(), **({"error": error} if error is not None else {}))
        return event

    def _finish_fields(self) -> dict[str, Any]:
        return _media_stream_end_fields(
            self._fields, self._response_mapper, self._text_parts, self._budget
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
        end(**response_mapper(result))
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
        _handle, end, _started_at = _start_span(
            _batch_params(args, kwargs), request_mapper, provider(resource)
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        end(**response_mapper(result))
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
        _handle, end, _started_at = _start_span(
            _batch_params(args, kwargs), request_mapper, provider(resource)
        )
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        end(**response_mapper(result))
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
