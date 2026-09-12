# Google GenAI accepts both pydantic models and plain dicts; the defensive readers below
# intentionally traverse dynamic shapes without trusting SDK internals.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

import contextvars
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, cast

import telemetry_dev
from google.genai import _extra_utils
from google.genai.models import AsyncModels, Models

__version__ = "0.1.0"

ProviderResolver = Callable[[object | None], str]
RequestMapper = Callable[[Any, Any, Any], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]

_WRAPPED_ATTR = "_telemetry_dev_google_genai_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_google_genai_original"
_ORIGINALS: list[tuple[type[Any], str, Any]] = []
_installed = False
_install_lock = threading.Lock()
_T = TypeVar("_T")
_AFC_USAGE_STATE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "telemetry_dev_google_genai_afc_usage_state", default=None
)

_BUILTIN_TOOL_FIELDS = (
    ("google_search", "googleSearch"),
    ("google_search_retrieval", "googleSearchRetrieval"),
    ("code_execution", "codeExecution"),
    ("url_context", "urlContext"),
    ("computer_use", "computerUse"),
    ("file_search", "fileSearch"),
    ("retrieval", "retrieval"),
    ("google_maps", "googleMaps"),
    ("enterprise_web_search", "enterpriseWebSearch"),
    ("parallel_ai_search", "parallelAiSearch"),
    ("mcp_servers", "mcpServers"),
)


def _camel_name(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _field(value: Any, name: str) -> Any:
    camel = _camel_name(name)
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
        return value.get(camel)
    for key in (name, camel):
        if hasattr(value, key):
            return getattr(value, key, None)
    return None


def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(cast(Sequence[Any], value))
    return []


def _native(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", by_alias=True, exclude_none=True)
        except Exception:
            try:
                return value.model_dump(by_alias=True, exclude_none=True)
            except Exception:
                return str(value)
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_native(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, str | int | float | bool):
        return _native(value.value)
    return value


def _normalized_content_input(contents: Any) -> list[Any]:
    if contents is None:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "parts": [{"text": contents}]}]
    items = _sequence_items(contents)
    if items:
        if any(
            _string(_field(item, "role")) is not None or _sequence_items(_field(item, "parts"))
            for item in items
        ):
            return [_native(item) for item in items]
        parts: list[Any] = []
        for item in items:
            if isinstance(item, str):
                parts.append({"text": item})
            else:
                parts.append(_native(item))
        return [{"role": "user", "parts": parts}]
    if _string(_field(contents, "role")) is None and not _sequence_items(_field(contents, "parts")):
        return [{"role": "user", "parts": [_native(contents)]}]
    return [_native(contents)]


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, str):
        raw = value.value
        return raw if isinstance(raw, str) else str(raw)
    return _string(value)


def _usage(fields: dict[str, int | float | None]) -> dict[str, int | float] | None:
    usage = {key: value for key, value in fields.items() if value is not None}
    return usage or None


def _sum_usage(
    total: dict[str, int | float] | None, part: dict[str, int | float] | None
) -> dict[str, int | float] | None:
    if not total:
        return part
    if not part:
        return total
    merged = dict(total)
    for key, value in part.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _afc_usage_state(config: Any) -> dict[str, Any] | None:
    try:
        if _extra_utils.should_disable_afc(config):
            return None
        tools = _sequence_items(_field(config, "tools"))
        if not any(
            callable(tool)
            or callable(getattr(tool, "call_tool", None))
            or callable(getattr(tool, "callTool", None))
            for tool in tools
        ):
            return None
        return {}
    except Exception:
        return None


def _record_afc_usage(response: Any) -> Any:
    state = _AFC_USAGE_STATE.get()
    if state is None:
        return response
    try:
        fields = _generate_response_fields(response)
        usage = fields.get("usage")
        if isinstance(usage, dict):
            state["usage"] = _sum_usage(
                cast(dict[str, int | float] | None, state.get("usage")), usage
            )
        attrs = fields.get("attributes")
        if isinstance(attrs, Mapping):
            tool_use = _number(attrs.get("google_genai.usage.tool_use_prompt_tokens"))
            if tool_use is not None:
                state["tool_use_prompt_tokens"] = int(state.get("tool_use_prompt_tokens", 0)) + int(
                    tool_use
                )
    except Exception:
        return response
    return response


def _fields_with_afc_usage(fields: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    usage = state.get("usage") if state else None
    if isinstance(usage, dict):
        fields["usage"] = usage
    tool_use = _number(state.get("tool_use_prompt_tokens")) if state else None
    if tool_use is not None:
        attrs = fields.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
            fields["attributes"] = attrs
        attrs["google_genai.usage.tool_use_prompt_tokens"] = int(tool_use)
    return fields


def _stop_sequences(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    strings = [item for item in (_string(part) for part in _sequence_items(value)) if item]
    return strings or None


def _json_attr(value: Any) -> str | None:
    try:
        return json.dumps(_native(value))
    except (TypeError, ValueError):
        return None


def _clean_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _safe_fields(mapper: ResponseMapper, response: Any) -> dict[str, Any]:
    try:
        return mapper(response)
    except Exception:
        return {}


def _end_once(handle: telemetry_dev.SpanHandle) -> Callable[..., None]:
    ended = False

    def end(**fields: Any) -> None:
        nonlocal ended
        if ended:
            return
        ended = True
        handle.end(**_clean_fields(fields))

    return end


def _output_type(config: Any) -> str | None:
    if config is None:
        return None
    mime = _string(_field(config, "response_mime_type"))
    if mime == "application/json":
        return "json"
    if (
        _field(config, "response_schema") is not None
        or _field(config, "response_json_schema") is not None
    ):
        return "json"
    if mime == "text/plain":
        return "text"
    return None


def _tool_definitions(tools: Any) -> str | None:
    items = _sequence_items(tools)
    if not items:
        return None
    definitions: list[Any] = []
    for tool in items:
        if callable(tool):
            tool_name = getattr(tool, "__name__", None)
            if tool_name:
                definitions.append({"type": "function", "name": tool_name})
            continue
        if callable(getattr(tool, "call_tool", None)) or callable(getattr(tool, "callTool", None)):
            tool_name = getattr(tool, "name", None)
            if tool_name:
                definitions.append({"type": "function", "name": tool_name})
            continue
        native = _native(tool)
        tool_dict = native if isinstance(native, Mapping) else {}
        decls = tool_dict.get("function_declarations") or tool_dict.get("functionDeclarations")
        if decls is None:
            decls = _field(tool, "function_declarations")
        for decl in _sequence_items(decls):
            decl_native = _native(decl)
            if not isinstance(decl_native, Mapping):
                continue
            entry: dict[str, Any] = {"type": "function", "name": decl_native.get("name")}
            if decl_native.get("description") is not None:
                entry["description"] = decl_native.get("description")
            parameters = decl_native.get("parameters")
            if parameters is None:
                parameters = decl_native.get("parameters_json_schema")
            if parameters is None:
                parameters = decl_native.get("parametersJsonSchema")
            if parameters is not None:
                entry["parameters"] = parameters
            if entry.get("name"):
                definitions.append(entry)
        for field_name, type_name in _BUILTIN_TOOL_FIELDS:
            if _field(tool, field_name) is not None or (
                isinstance(tool_dict, Mapping) and tool_dict.get(field_name) is not None
            ):
                definitions.append({"type": type_name})
    if not definitions:
        return None
    return _json_attr(definitions)


def _request_attributes(config: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    candidate_count = _number(_field(config, "candidate_count"))
    if candidate_count is not None:
        attrs["gen_ai.request.choice.count"] = int(candidate_count)
    tools = _tool_definitions(_field(config, "tools"))
    if tools is not None:
        attrs["gen_ai.tool.definitions"] = tools
    for field_name, attr_name in (
        ("tool_config", "google_genai.request.tool_config"),
        ("safety_settings", "google_genai.request.safety_settings"),
        ("thinking_config", "google_genai.request.thinking_config"),
        ("labels", "google_genai.request.labels"),
        ("response_modalities", "google_genai.request.response_modalities"),
    ):
        value = _field(config, field_name)
        if value is None:
            continue
        encoded = _json_attr(value)
        if encoded is not None:
            attrs[attr_name] = encoded
    cached_content = _field(config, "cached_content")
    if cached_content is not None:
        cached = _string(cached_content) or _string(_field(cached_content, "name"))
        if cached is not None:
            attrs["google_genai.request.cached_content"] = cached
    return attrs


def _generate_request_fields(model: Any, contents: Any, config: Any) -> tuple[str, dict[str, Any]]:
    model_name = _string(model) or "unknown"
    fields: dict[str, Any] = {
        "type": "generation",
        "model": _string(model),
        "input": _normalized_content_input(contents),
    }
    system_instruction = _field(config, "system_instruction")
    if system_instruction is not None:
        fields["system_instructions"] = _native(system_instruction)
    for field_name in (
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = _number(_field(config, field_name))
        if value is not None:
            fields[field_name] = value
    max_output_tokens = _number(_field(config, "max_output_tokens"))
    if max_output_tokens is not None:
        fields["max_tokens"] = int(max_output_tokens)
    stop_sequences = _stop_sequences(_field(config, "stop_sequences"))
    if stop_sequences is not None:
        fields["stop_sequences"] = stop_sequences
    output_type = _output_type(config)
    if output_type is not None:
        fields["output_type"] = output_type
    attrs = _request_attributes(config)
    if attrs:
        fields["attributes"] = attrs
    return f"chat {model_name}", fields


def _generate_usage(raw: Any, attrs: dict[str, Any]) -> dict[str, int | float] | None:
    tool_use = _number(_field(raw, "tool_use_prompt_token_count"))
    if tool_use is not None:
        attrs["google_genai.usage.tool_use_prompt_tokens"] = int(tool_use)
    return _usage(
        {
            "input_tokens": _number(_field(raw, "prompt_token_count")),
            "output_tokens": _number(_field(raw, "candidates_token_count")),
            "total_tokens": _number(_field(raw, "total_token_count")),
            "cache_read_input_tokens": _number(_field(raw, "cached_content_token_count")),
            "reasoning_output_tokens": _number(_field(raw, "thoughts_token_count")),
        }
    )


def _content_output(content: Any) -> dict[str, Any] | None:
    if content is None:
        return None
    parts = _sequence_items(_field(content, "parts"))
    if not parts:
        return None
    role = _string(_field(content, "role")) or "model"
    return {"role": role, "parts": [_native(part) for part in parts]}


def _candidate_index(candidate: Any, fallback: int) -> int:
    index = _number(_field(candidate, "index"))
    return int(index) if index is not None else fallback


def _record_candidate_metadata(
    candidate: Any,
    index: int,
    attrs: dict[str, Any],
    budget: telemetry_dev.CaptureBudget | None = None,
) -> None:
    ratings = _field(candidate, "safety_ratings")
    if ratings:
        existing = attrs.get("google_genai.response.safety_ratings")
        entries: list[Any]
        if isinstance(existing, str):
            try:
                parsed = json.loads(existing)
                entries = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        entries.append({"candidateIndex": index, "ratings": _native(ratings)})
        encoded = _json_attr(entries)
        if encoded is not None and (budget is None or budget.accept(encoded)):
            attrs["google_genai.response.safety_ratings"] = encoded
    if index == 0:
        grounding = _field(candidate, "grounding_metadata")
        if grounding is not None:
            encoded = _json_attr(grounding)
            if encoded is not None and (budget is None or budget.accept(encoded)):
                attrs["google_genai.response.grounding_metadata"] = encoded
        url_context = _field(candidate, "url_context_metadata")
        if url_context is not None:
            encoded = _json_attr(url_context)
            if encoded is not None and (budget is None or budget.accept(encoded)):
                attrs["google_genai.response.url_context_metadata"] = encoded


def _generate_response_fields(response: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    response_model = _string(_field(response, "model_version"))
    if response_model:
        fields["response_model"] = response_model
    response_id = _string(_field(response, "response_id"))
    if response_id:
        fields["response_id"] = response_id
    candidates = _sequence_items(_field(response, "candidates"))
    finish_reasons: list[str] = []
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_index = _candidate_index(candidate, index)
        finish = _enum_value(_field(candidate, "finish_reason"))
        if finish is not None:
            finish_reasons.append(finish)
        content_output = _content_output(_field(candidate, "content"))
        if content_output is not None:
            output.append(content_output)
        _record_candidate_metadata(candidate, candidate_index, attrs)
    if finish_reasons:
        fields["finish_reason"] = finish_reasons[0]
        if len(finish_reasons) > 1:
            attrs["gen_ai.response.finish_reasons"] = finish_reasons
    if output:
        fields["output"] = output
    usage = _generate_usage(_field(response, "usage_metadata"), attrs)
    if usage:
        fields["usage"] = usage
    feedback = _field(response, "prompt_feedback")
    block_reason = _enum_value(_field(feedback, "block_reason"))
    if block_reason:
        attrs["google_genai.response.block_reason"] = block_reason
    block_message = _string(_field(feedback, "block_reason_message"))
    if block_message:
        attrs["google_genai.response.block_reason_message"] = block_message
    prompt_ratings = _sequence_items(_field(feedback, "safety_ratings"))
    if prompt_ratings:
        encoded = _json_attr([_native(rating) for rating in prompt_ratings])
        if encoded is not None:
            attrs["google_genai.response.prompt_safety_ratings"] = encoded
    history = _field(response, "automatic_function_calling_history")
    if history:
        native_history = _native(history)
        if native_history:
            fields["input"] = native_history
            attrs["google_genai.automatic_function_calling"] = True
    if attrs:
        fields["attributes"] = attrs
    return fields


def _embed_request_fields(model: Any, contents: Any, config: Any) -> tuple[str, dict[str, Any]]:
    model_name = _string(model) or "unknown"
    fields: dict[str, Any] = {
        "type": "embedding",
        "model": _string(model),
        "input": _native(contents),
    }
    attrs: dict[str, Any] = {}
    task_type = _string(_field(config, "task_type"))
    if task_type:
        attrs["google_genai.request.task_type"] = task_type
    output_dimensionality = _number(_field(config, "output_dimensionality"))
    if output_dimensionality is not None:
        attrs["google_genai.request.output_dimensionality"] = int(output_dimensionality)
    if attrs:
        fields["attributes"] = attrs
    return f"embeddings {model_name}", fields


def _embed_response_fields(response: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    embeddings = _sequence_items(_field(response, "embeddings"))
    if embeddings:
        attrs["google_genai.response.embedding_count"] = len(embeddings)
        values = _field(embeddings[0], "values")
        if values is not None:
            attrs["google_genai.response.embedding_dimensions"] = len(list(values))
    total_tokens = 0
    saw_tokens = False
    for embedding in embeddings:
        token_count = _number(_field(_field(embedding, "statistics"), "token_count"))
        if token_count is not None:
            total_tokens += int(token_count)
            saw_tokens = True
    usage = _usage({"input_tokens": total_tokens if saw_tokens else None})
    if usage:
        fields["usage"] = usage
    billable = _number(_field(_field(response, "metadata"), "billable_character_count"))
    if billable is not None:
        attrs["google_genai.usage.billable_characters"] = int(billable)
    if attrs:
        fields["attributes"] = attrs
    return fields


class _CandidateState:
    def __init__(self) -> None:
        self.role = "model"
        self.parts: list[Any] = []


class _StreamState:
    def __init__(self) -> None:
        self.candidates: dict[int, _CandidateState] = {}
        self.finish_reasons: dict[int, str] = {}
        self.usage: dict[str, int | float] | None = None
        self.response_id: str | None = None
        self.response_model: str | None = None
        self.attrs: dict[str, Any] = {}
        self.committed_usage: dict[str, int | float] | None = None
        self.prior_contents: list[dict[str, Any]] = []
        self.afc_history: Any = None
        self.budget = telemetry_dev.CaptureBudget.from_client()


def _append_part(parts: list[Any], native: Any) -> None:
    if not isinstance(native, dict):
        parts.append(native)
        return
    text = native.get("text")
    if text is not None:
        thought = native.get("thought", False)
        if parts:
            previous = parts[-1]
            if isinstance(previous, dict) and previous.get("text") is not None:
                if previous.get("thought", False) == thought:
                    previous["text"] = f"{previous.get('text', '')}{text}"
                    return
    parts.append(native)


def _fold_turn(state: _StreamState) -> None:
    # An AFC turn ended: archive its output and fold its usage so the next
    # internal model call does not overwrite what this turn produced.
    for index in sorted(state.candidates):
        candidate_state = state.candidates[index]
        if candidate_state.parts:
            state.prior_contents.append(
                {"role": candidate_state.role, "parts": candidate_state.parts}
            )
    state.candidates.clear()
    state.finish_reasons.clear()
    state.committed_usage = _sum_usage(state.committed_usage, state.usage)
    state.usage = None


def _record_chunk(chunk: Any, state: _StreamState) -> None:
    try:
        _record_chunk_inner(chunk, state)
    except Exception:
        return


def _record_chunk_inner(chunk: Any, state: _StreamState) -> None:
    response_id = _string(_field(chunk, "response_id"))
    if response_id:
        if state.response_id and response_id != state.response_id:
            _fold_turn(state)
        state.response_id = response_id
    response_model = _string(_field(chunk, "model_version"))
    if response_model:
        state.response_model = response_model
    for index, candidate in enumerate(_sequence_items(_field(chunk, "candidates"))):
        candidate_index = _candidate_index(candidate, index)
        candidate_state = state.candidates.setdefault(candidate_index, _CandidateState())
        content = _field(candidate, "content")
        role = _string(_field(content, "role"))
        if role:
            candidate_state.role = role
        for part in _sequence_items(_field(content, "parts")):
            native = _native(part)
            if state.budget.accept(native):
                _append_part(candidate_state.parts, native)
        finish = _enum_value(_field(candidate, "finish_reason"))
        if finish is not None:
            state.finish_reasons[candidate_index] = finish
        _record_candidate_metadata(candidate, candidate_index, state.attrs, state.budget)
    usage = _generate_usage(_field(chunk, "usage_metadata"), state.attrs)
    if usage:
        state.usage = usage
    feedback = _field(chunk, "prompt_feedback")
    block_reason = _enum_value(_field(feedback, "block_reason"))
    if block_reason:
        state.attrs["google_genai.response.block_reason"] = block_reason
    block_message = _string(_field(feedback, "block_reason_message"))
    if block_message:
        state.attrs["google_genai.response.block_reason_message"] = block_message
    prompt_ratings = _sequence_items(_field(feedback, "safety_ratings"))
    if prompt_ratings:
        encoded = _json_attr([_native(rating) for rating in prompt_ratings])
        if encoded is not None and state.budget.accept(encoded):
            state.attrs["google_genai.response.prompt_safety_ratings"] = encoded
    history = _field(chunk, "automatic_function_calling_history")
    if history:
        native_history = _native(history)
        if native_history and state.budget.accept(native_history):
            state.afc_history = native_history


def _stream_fields(state: _StreamState) -> dict[str, Any]:
    try:
        return _stream_fields_inner(state)
    except Exception:
        return {}


def _stream_fields_inner(state: _StreamState) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    attrs = dict(state.attrs)
    if state.response_id:
        fields["response_id"] = state.response_id
    if state.response_model:
        fields["response_model"] = state.response_model
    output: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    for index in sorted(state.candidates):
        candidate_state = state.candidates[index]
        if candidate_state.parts:
            output.append({"role": candidate_state.role, "parts": candidate_state.parts})
        if index in state.finish_reasons:
            finish_reasons.append(state.finish_reasons[index])
    if state.afc_history:
        # AFC turns are fully represented in the history (function calls and
        # responses included), mirroring the non-streaming AFC mapping.
        fields["input"] = state.afc_history
        attrs["google_genai.automatic_function_calling"] = True
    elif state.prior_contents:
        output = [*state.prior_contents, *output]
    if finish_reasons:
        fields["finish_reason"] = finish_reasons[0]
        if len(finish_reasons) > 1:
            attrs["gen_ai.response.finish_reasons"] = finish_reasons
    if output:
        fields["output"] = output
    usage = _sum_usage(state.committed_usage, state.usage)
    if usage:
        fields["usage"] = usage
    if attrs:
        fields["attributes"] = attrs
    return fields


def _provider_for_client(client: object | None) -> str:
    if client is not None and getattr(client, "vertexai", False):
        return "gcp.vertex_ai"
    return "gcp.gemini"


def _provider_for_resource(resource: object | None) -> str:
    api_client = getattr(resource, "_api_client", None) if resource is not None else None
    if api_client is not None and getattr(api_client, "vertexai", False):
        return "gcp.vertex_ai"
    return "gcp.gemini"


def _start_generate_span(
    model: Any, contents: Any, config: Any, provider: str
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float]:
    try:
        name, fields = _generate_request_fields(model, contents, config)
    except Exception:
        name, fields = f"chat {_string(model) or 'unknown'}", {"type": "generation"}
    handle = telemetry_dev.start_span(name, provider=provider, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter()


def _start_embed_span(
    model: Any, contents: Any, config: Any, provider: str
) -> tuple[telemetry_dev.SpanHandle, Callable[..., None], float]:
    try:
        name, fields = _embed_request_fields(model, contents, config)
    except Exception:
        name, fields = f"embeddings {_string(model) or 'unknown'}", {"type": "embedding"}
    handle = telemetry_dev.start_span(name, provider=provider, **_clean_fields(fields))
    return handle, _end_once(handle), time.perf_counter()


def _first_chunk_update(chunk: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
    try:
        update: dict[str, Any] = {
            "time_to_first_chunk_ms": (time.perf_counter() - started_at) * 1000,
        }
        response_id = _string(_field(chunk, "response_id"))
        if response_id:
            update["response_id"] = response_id
        response_model = _string(_field(chunk, "model_version"))
        if response_model:
            update["response_model"] = response_model
        handle.update(**_clean_fields(update))
    except Exception:
        return


def _chunk_has_output(chunk: Any) -> bool:
    for candidate in _sequence_items(_field(chunk, "candidates")):
        content = _field(candidate, "content")
        for part in _sequence_items(_field(content, "parts")):
            text = _field(part, "text")
            function_call = _field(part, "function_call") or _field(part, "functionCall")
            args = _field(function_call, "args")
            partial_args = _field(function_call, "partial_args") or _field(
                function_call, "partialArgs"
            )
            inline_data = _field(part, "inline_data") or _field(part, "inlineData")
            mime_type = _string(_field(inline_data, "mime_type")) or _string(
                _field(inline_data, "mimeType")
            )
            data = _field(inline_data, "data")
            has_partial_arg_value = any(
                isinstance(_field(partial_arg, "bool_value"), bool)
                or isinstance(_field(partial_arg, "boolValue"), bool)
                or isinstance(_field(partial_arg, "number_value"), int | float)
                or isinstance(_field(partial_arg, "numberValue"), int | float)
                or bool(
                    _string(_field(partial_arg, "string_value"))
                    or _string(_field(partial_arg, "stringValue"))
                )
                or _field(partial_arg, "null_value") == "NULL_VALUE"
                or _field(partial_arg, "nullValue") == "NULL_VALUE"
                for partial_arg in _sequence_items(partial_args)
            )
            if (
                (isinstance(text, str) and bool(text))
                or args not in (None, "", [], {})
                or has_partial_arg_value
                or (
                    mime_type is not None
                    and mime_type.startswith("audio/")
                    and isinstance(data, str | bytes)
                    and bool(data)
                )
            ):
                return True
    return False


class _ObservedStream:
    def __init__(
        self,
        inner: Iterator[Any],
        handle: telemetry_dev.SpanHandle,
        end: Callable[..., None],
        started_at: float,
        afc_usage_state: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._handle = handle
        self._end = end
        self._started_at = started_at
        self._state = _StreamState()
        self._saw_first = False
        self._afc_usage_state = afc_usage_state

    def _fields(self) -> dict[str, Any]:
        return _fields_with_afc_usage(_stream_fields(self._state), self._afc_usage_state)

    def __iter__(self) -> Iterator[Any]:
        try:
            while True:
                try:
                    yield self.__next__()
                except StopIteration:
                    return
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                self._end(**_clean_fields(self._fields()), error=exc)
            raise
        finally:
            self.close()

    def __next__(self) -> Any:
        return self._advance(self._inner.__next__)

    def send(self, value: Any) -> Any:
        def pull() -> Any:
            return self._inner.send(value)

        return self._advance(pull)

    def throw(self, *args: Any) -> Any:
        def pull() -> Any:
            return self._inner.throw(*args)

        return self._advance(pull)

    def _advance(self, pull: Callable[[], Any]) -> Any:
        token = (
            _AFC_USAGE_STATE.set(self._afc_usage_state)
            if self._afc_usage_state is not None
            else None
        )
        try:
            chunk = pull()
            received_at = time.perf_counter()
        except StopIteration:
            self._end(**_clean_fields(self._fields()))
            raise
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                self._end(**_clean_fields(self._fields()), error=exc)
            raise
        finally:
            if token is not None:
                _AFC_USAGE_STATE.reset(token)
        if _chunk_has_output(chunk):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            _first_chunk_update(chunk, self._handle, self._started_at)
        _record_chunk(chunk, self._state)
        return chunk

    def close(self) -> None:
        self._end(**_clean_fields(self._fields()))
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


class _ObservedAsyncStream:
    def __init__(
        self,
        inner: AsyncIterator[Any],
        handle: telemetry_dev.SpanHandle,
        end: Callable[..., None],
        started_at: float,
        afc_usage_state: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._handle = handle
        self._end = end
        self._started_at = started_at
        self._state = _StreamState()
        self._saw_first = False
        self._afc_usage_state = afc_usage_state

    def _fields(self) -> dict[str, Any]:
        return _fields_with_afc_usage(_stream_fields(self._state), self._afc_usage_state)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                try:
                    yield await self.__anext__()
                except StopAsyncIteration:
                    return
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                self._end(**_clean_fields(self._fields()), error=exc)
            raise
        finally:
            await self.aclose()

    def __anext__(self) -> Any:
        return self._advance(self._inner.__anext__)

    def asend(self, value: Any) -> Any:
        def pull() -> Awaitable[Any]:
            return self._inner.asend(value)

        return self._advance(pull)

    def athrow(self, *args: Any) -> Any:
        def pull() -> Awaitable[Any]:
            return self._inner.athrow(*args)

        return self._advance(pull)

    async def _advance(self, pull: Callable[[], Awaitable[Any]]) -> Any:
        token = (
            _AFC_USAGE_STATE.set(self._afc_usage_state)
            if self._afc_usage_state is not None
            else None
        )
        try:
            chunk = await pull()
            received_at = time.perf_counter()
        except StopAsyncIteration:
            self._end(**_clean_fields(self._fields()))
            raise
        except BaseException as exc:
            if not isinstance(exc, GeneratorExit):
                self._end(**_clean_fields(self._fields()), error=exc)
            raise
        finally:
            if token is not None:
                _AFC_USAGE_STATE.reset(token)
        if _chunk_has_output(chunk):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            _first_chunk_update(chunk, self._handle, self._started_at)
        _record_chunk(chunk, self._state)
        return chunk

    async def aclose(self) -> None:
        self._end(**_clean_fields(self._fields()))
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def _wrap_sync_collect_generate(
    original: Callable[..., Any], _provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _record_afc_usage(original(*args, **kwargs))

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_sync_collect_stream(
    original: Callable[..., Any], _provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
        for chunk in original(*args, **kwargs):
            yield _record_afc_usage(chunk)

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_collect_generate(
    original: Callable[..., Any], _provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _record_afc_usage(await original(*args, **kwargs))

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_collect_stream(
    original: Callable[..., Any], _provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        inner = await original(*args, **kwargs)

        async def observed() -> AsyncIterator[Any]:
            async for chunk in inner:
                yield _record_afc_usage(chunk)

        return observed()

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_sync_generate(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        _handle, end, _started_at = _start_generate_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        afc_state = _afc_usage_state(kwargs.get("config"))
        token = _AFC_USAGE_STATE.set(afc_state) if afc_state is not None else None
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        finally:
            if token is not None:
                _AFC_USAGE_STATE.reset(token)
        end(**_fields_with_afc_usage(_safe_fields(_generate_response_fields, result), afc_state))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_sync_stream(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        handle, end, started_at = _start_generate_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        afc_state = _afc_usage_state(kwargs.get("config"))
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        return _ObservedStream(result, handle, end, started_at, afc_state)

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_sync_embed(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        _handle, end, _started_at = _start_embed_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        end(**_safe_fields(_embed_response_fields, result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_generate(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        _handle, end, _started_at = _start_generate_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        afc_state = _afc_usage_state(kwargs.get("config"))
        token = _AFC_USAGE_STATE.set(afc_state) if afc_state is not None else None
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        finally:
            if token is not None:
                _AFC_USAGE_STATE.reset(token)
        end(**_fields_with_afc_usage(_safe_fields(_generate_response_fields, result), afc_state))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_stream(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        handle, end, started_at = _start_generate_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        afc_state = _afc_usage_state(kwargs.get("config"))
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        return _ObservedAsyncStream(result, handle, end, started_at, afc_state)

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _wrap_async_embed(
    original: Callable[..., Any], provider_resolver: ProviderResolver
) -> Callable[..., Any]:
    @wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resource = args[0] if args else None
        _handle, end, _started_at = _start_embed_span(
            kwargs.get("model"),
            kwargs.get("contents"),
            kwargs.get("config"),
            provider_resolver(resource),
        )
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            end(error=exc)
            raise
        end(**_safe_fields(_embed_response_fields, result))
        return result

    setattr(wrapper, _WRAPPED_ATTR, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _patch_instance(
    resource: object,
    method: str,
    wrapper_factory: Callable[[Callable[..., Any], ProviderResolver], Callable[..., Any]],
    provider_name: str,
) -> None:
    current = getattr(resource, method)
    if getattr(current, _WRAPPED_ATTR, False):
        if method in vars(resource):
            return
        original = getattr(current, _ORIGINAL_ATTR, None)
        if original is None:
            return
        current = original.__get__(resource, type(resource))
    wrapped = wrapper_factory(current, lambda _: provider_name)
    setattr(resource, method, wrapped)


def _patch_class(
    cls: type[Any],
    method: str,
    wrapper_factory: Callable[[Callable[..., Any], ProviderResolver], Callable[..., Any]],
) -> None:
    original = getattr(cls, method)
    if getattr(original, _WRAPPED_ATTR, False):
        return
    _ORIGINALS.append((cls, method, original))
    setattr(cls, method, wrapper_factory(original, _provider_for_resource))


def _patch_models(resource: object, provider_name: str) -> None:
    _patch_instance(resource, "_generate_content", _wrap_sync_collect_generate, provider_name)
    _patch_instance(resource, "_generate_content_stream", _wrap_sync_collect_stream, provider_name)
    _patch_instance(resource, "generate_content", _wrap_sync_generate, provider_name)
    _patch_instance(resource, "generate_content_stream", _wrap_sync_stream, provider_name)
    _patch_instance(resource, "embed_content", _wrap_sync_embed, provider_name)


def _patch_async_models(resource: object, provider_name: str) -> None:
    _patch_instance(resource, "_generate_content", _wrap_async_collect_generate, provider_name)
    _patch_instance(resource, "_generate_content_stream", _wrap_async_collect_stream, provider_name)
    _patch_instance(resource, "generate_content", _wrap_async_generate, provider_name)
    _patch_instance(resource, "generate_content_stream", _wrap_async_stream, provider_name)
    _patch_instance(resource, "embed_content", _wrap_async_embed, provider_name)


def wrap_google_genai(client: _T) -> _T:
    if getattr(client, _WRAPPED_ATTR, False):
        return client
    provider_name = _provider_for_client(client)
    models = getattr(client, "models", None)
    if models is not None:
        _patch_models(models, provider_name)
    aio = getattr(client, "aio", None)
    async_models = getattr(aio, "models", None) if aio is not None else None
    if async_models is not None:
        _patch_async_models(async_models, provider_name)
    setattr(client, _WRAPPED_ATTR, True)
    return client


def instrument_google_genai() -> None:
    global _installed
    with _install_lock:
        if _installed:
            return
        for method, factory in (
            ("_generate_content", _wrap_sync_collect_generate),
            ("_generate_content_stream", _wrap_sync_collect_stream),
            ("generate_content", _wrap_sync_generate),
            ("generate_content_stream", _wrap_sync_stream),
            ("embed_content", _wrap_sync_embed),
        ):
            _patch_class(Models, method, factory)
        for method, factory in (
            ("_generate_content", _wrap_async_collect_generate),
            ("_generate_content_stream", _wrap_async_collect_stream),
            ("generate_content", _wrap_async_generate),
            ("generate_content_stream", _wrap_async_stream),
            ("embed_content", _wrap_async_embed),
        ):
            _patch_class(AsyncModels, method, factory)
        _installed = True


def uninstrument_google_genai() -> None:
    global _installed
    with _install_lock:
        while _ORIGINALS:
            cls, method, original = _ORIGINALS.pop()
            current = getattr(cls, method, None)
            if (
                getattr(current, _WRAPPED_ATTR, False)
                and getattr(current, _ORIGINAL_ATTR, None) is original
            ):
                setattr(cls, method, original)
        _installed = False


__all__ = [
    "__version__",
    "instrument_google_genai",
    "uninstrument_google_genai",
    "wrap_google_genai",
]
