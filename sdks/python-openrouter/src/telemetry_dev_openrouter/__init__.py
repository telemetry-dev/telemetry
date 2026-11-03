from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from io import StringIO
from typing import Any, TypeVar, cast

import telemetry_dev
from openrouter.chat import Chat
from openrouter.embeddings import Embeddings
from openrouter.responses import Responses

__version__ = "0.2.3"

RequestMapper = Callable[[Mapping[str, Any]], tuple[str, dict[str, Any]]]
ResponseMapper = Callable[[Any], dict[str, Any]]

_PROVIDER = "openrouter"
_WRAPPED_ATTR = "_telemetry_dev_openrouter_wrapped"
_ORIGINAL_ATTR = "_telemetry_dev_openrouter_original"
_CAPTURE_TRUNCATED_ATTRIBUTE = "telemetry.dev.capture.truncated"
# The core applies max_attribute_length after the mask. More choices than this are not kept.
_MAX_STREAM_CHOICES = 1024
_FINISH_REASON_MAX_LENGTH = 128
_TERMINAL_RESPONSE_EVENTS = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
_OMIT = object()
_ORIGINALS: list[tuple[type[Any], str, Any]] = []
_installed = False
_install_lock = threading.Lock()


def _reset_install_lock_in_child() -> None:
    """A thread of the parent can hold the lock at a fork, and only this thread exists here."""
    global _install_lock
    _install_lock = threading.Lock()


# A hook is enough: a fork that runs no hooks, as uWSGI's fork of its workers by default,
# starts from its master, which serves no requests. os.register_at_fork cannot remove a hook,
# and importlib.reload runs this module again with the same globals, so the hook of the first
# run stays. Windows has no fork.
if hasattr(os, "register_at_fork") and "_install_fork_hook" not in globals():
    _install_fork_hook = True
    os.register_at_fork(after_in_child=_reset_install_lock_in_child)

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
    # `cost` is the OpenRouter charge. For a request with the user's own provider key
    # (BYOK), the provider bills the inference, and the charge is only the OpenRouter
    # fee. The cost of such a request is the fee and the provider's inference cost.
    # Without the inference cost, the span has no cost.
    cost = _number(_field(raw_usage, "cost"))
    upstream_cost = _number(_field(_field(raw_usage, "cost_details"), "upstream_inference_cost"))
    if _field(raw_usage, "is_byok") is True:
        return None if upstream_cost is None else upstream_cost + (cost or 0)
    return cost if cost is not None else upstream_cost


def _prompt_attributes(messages: object) -> dict[str, Any]:
    # Older cores have no prompt_attributes_for and record a prompt only inside with_prompt.
    read = getattr(telemetry_dev, "prompt_attributes_for", None)
    return cast("dict[str, Any]", read(messages)) if callable(read) else {}


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
            "attributes": _prompt_attributes(params.get("messages")) or None,
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
            "audio_input_tokens": _number(_field(prompt_details, "audio_tokens")),
            "audio_output_tokens": _number(_field(completion_details, "audio_tokens")),
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
            "attributes": _prompt_attributes(params.get("input")) or None,
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
    prompt_details = _field(raw_usage, "prompt_tokens_details")
    return {
        "response_model": _string(_field(response, "model")),
        "response_id": _string(_field(response, "id")),
        "usage": _usage(
            {
                "input_tokens": _number(_field(raw_usage, "prompt_tokens")),
                "total_tokens": _number(_field(raw_usage, "total_tokens")),
                "text_input_tokens": _number(_field(prompt_details, "text_tokens")),
                "image_input_tokens": _number(_field(prompt_details, "image_tokens")),
                "audio_input_tokens": _number(_field(prompt_details, "audio_tokens")),
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


# On Python 3.10 and 3.11, StringIO keeps up to 100,000 written strings in a list before it
# joins them, about 60 bytes each that the stream bound does not charge: 15 bytes a character
# for 4-character deltas. Deltas are joined 256 at a time, so that those waiting hold under
# about 22 KB, onto a string that is written once it has 4,096 characters, so that what is
# written costs about 1.5% more than its text. Python 3.12 dropped the list.
_DELTA_BATCH = 256
_WRITE_CHARS = 4096


class _DeltaBuffer:
    # A write takes several steps, unlike one StringIO call, so without the lock a read on
    # another thread, such as one that closes the stream, could copy a batch twice or drop it.
    # A fork copies the lock held if another thread was writing, and that thread does not run
    # in the child to release it. Every fork, also one that runs no at-fork hooks, changes the
    # process ID, so a child takes a new lock. The check is inline because it runs per delta.
    # A signal handler runs on the thread it interrupts, so one that reads the stream, such as
    # one that closes it, can read inside a write. So the lock is reentrant, and each step takes
    # the deltas and the tail out of the buffer before it joins or writes them, so that a read
    # inside the step does not write them as well; that read returns a prefix, which can leave
    # out the deltas being joined. A handler that writes inside a step is not supported. An
    # exception inside a step, such as KeyboardInterrupt from a handler, puts back the text the
    # step took unless the StringIO has it, so that a later read still has every delta.
    __slots__ = ("_buffer", "_lock", "_pending", "_pid", "_tail")

    def __init__(self) -> None:
        self._buffer = StringIO()
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._pending: list[str] = []
        self._tail = ""

    def write(self, text: str) -> None:
        pid = os.getpid()
        if self._pid != pid:
            self._lock = threading.RLock()
            self._pid = pid
        with self._lock:
            self._pending.append(text)
            if len(self._pending) >= _DELTA_BATCH:
                self._join(_WRITE_CHARS)

    def getvalue(self) -> str:
        pid = os.getpid()
        if self._pid != pid:
            self._lock = threading.RLock()
            self._pid = pid
        with self._lock:
            self._join(1)
            return self._buffer.getvalue()

    def _join(self, write_at: int) -> None:
        if not self._pending and len(self._tail) < write_at:
            return
        written = self._buffer.tell()
        pending: list[str] = []
        tail = ""
        try:
            pending, self._pending = self._pending, []
            tail, self._tail = self._tail, ""
            joined = "".join([tail, *pending])
            if len(joined) >= write_at:
                self._buffer.write(joined)
            else:
                self._tail = joined
        except BaseException:
            if self._buffer.tell() == written:
                self._pending[:0] = pending
                self._tail = tail + self._tail
            raise


class _ChatChoice:
    def __init__(self) -> None:
        self.role: str | None = None
        # Deltas are written to buffers and joined when output is read: repeated concatenation
        # copies the whole text for each delta.
        self.content_fragments = _DeltaBuffer()
        self.reasoning_fragments = _DeltaBuffer()
        self.reasoning_details: list[Any] = []
        self.refusal_fragments = _DeltaBuffer()
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None

    @property
    def content(self) -> str:
        return self.content_fragments.getvalue()

    @property
    def reasoning(self) -> str:
        return self.reasoning_fragments.getvalue()

    @property
    def refusal(self) -> str:
        return self.refusal_fragments.getvalue()


def _choice_state(
    states: dict[int, _ChatChoice], index: int, budget: telemetry_dev.CaptureBudget
) -> _ChatChoice | None:
    if index not in states:
        if len(states) >= _MAX_STREAM_CHOICES:
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
            fragments = current_function.get("arguments")
            if not isinstance(fragments, _DeltaBuffer):
                fragments = current_function["arguments"] = _DeltaBuffer()
            fragments.write(arguments)
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
            state.content_fragments.write(content)
        if isinstance(reasoning, str) and budget.accept(reasoning):
            state.reasoning_fragments.write(reasoning)
        for detail in _sequence_items(_field(delta, "reasoning_details")):
            native_detail = _native(detail)
            if native_detail is not None and budget.accept(native_detail):
                state.reasoning_details.append(native_detail)
        if isinstance(refusal, str) and budget.accept(refusal):
            state.refusal_fragments.write(refusal)
        for tool_call in _sequence_items(_field(delta, "tool_calls")):
            if budget.accept(tool_call):
                _merge_tool_call(state, tool_call)
        finish_reason = _field(choice, "finish_reason")
        if isinstance(finish_reason, str):
            state.finish_reason = finish_reason[:_FINISH_REASON_MAX_LENGTH]
    return {
        "response_id": _string(_field(chunk, "id")),
        "response_model": _string(_field(chunk, "model")),
        "usage": _chat_usage(_field(chunk, "usage")),
        "cost_usd": _cost_usd(_field(chunk, "usage")),
    }


def _chat_chunk_has_output(chunk: Any) -> bool:
    for choice in _sequence_items(_field(chunk, "choices")):
        delta = _field(choice, "delta")
        if any(
            isinstance(_field(delta, key), str) and _field(delta, key)
            for key in ("content", "reasoning", "refusal")
        ):
            return True
        audio = _field(delta, "audio")
        if isinstance(_field(audio, "data"), str) and _field(audio, "data"):
            return True
        reasoning_details = [
            *_sequence_items(_field(delta, "reasoning_details")),
            *_sequence_items(_field(delta, "reasoningDetails")),
        ]
        if any(
            isinstance(_field(detail, key), str) and _field(detail, key)
            for detail in reasoning_details
            for key in ("text", "summary")
        ):
            return True
        for tool_call in _sequence_items(_field(delta, "tool_calls")):
            arguments = _field(_field(tool_call, "function"), "arguments")
            if isinstance(arguments, str) and arguments:
                return True
    return False


def _response_event_has_output(event: Any) -> bool:
    event = _response_event(event)
    event_type = _string(_field(event, "type")) or ""
    return (
        event_type
        in {
            "response.output_text.delta",
            "response.refusal.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
            "response.code_interpreter_call_code.delta",
            "response.mcp_call_arguments.delta",
            "response.apply_patch_call_operation_diff.delta",
            "response.fusion_call.panel.delta",
            "response.fusion_call.panel.reasoning.delta",
            "response.output_audio.delta",
            "response.audio.delta",
            "response.audio.transcript.delta",
        }
        and isinstance(_field(event, "delta"), str)
        and bool(_field(event, "delta"))
    )


def _joined_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        return call
    arguments = cast(dict[str, Any], function).get("arguments")
    if not isinstance(arguments, _DeltaBuffer):
        return call
    return {**call, "function": {**function, "arguments": arguments.getvalue()}}


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
            message["tool_calls"] = [
                _joined_tool_call(call) for _, call in sorted(state.tool_calls.items())
            ]
        output.append(message)
    return output


def _stream_capture_budget() -> telemetry_dev.CaptureBudget:
    # Older cores have no for_stream and bound each budget at 64 KiB on its own.
    for_stream = getattr(telemetry_dev.CaptureBudget, "for_stream", None)
    budget = for_stream() if callable(for_stream) else telemetry_dev.CaptureBudget()
    return cast(telemetry_dev.CaptureBudget, budget)


def _release_budget(budget: telemetry_dev.CaptureBudget) -> None:
    release = getattr(budget, "release", None)
    if callable(release):
        release()


def _measuring_budget(stream: telemetry_dev.CaptureBudget) -> telemetry_dev.CaptureBudget:
    """An empty budget that measures a value replacing everything ``stream`` holds, within the
    bound ``stream`` has now, without holding bytes itself."""
    return telemetry_dev.CaptureBudget(stream.max_bytes, stream.max_items)


def _adopt_budget(
    stream: telemetry_dev.CaptureBudget, measured: telemetry_dev.CaptureBudget
) -> bool:
    """Takes the counts of a budget that measured a replacement into the stream's own budget.
    Returns False, and takes nothing, when another stream took the room after the measure."""
    # Older cores have no reserve_bytes; there the measure against max_bytes stands.
    reserve = getattr(stream, "reserve_bytes", None)
    if callable(reserve):
        if not reserve(measured.bytes_used):
            return False
    else:
        stream.bytes_used = measured.bytes_used
    stream.items_used = measured.items_used
    stream.truncated = measured.truncated
    return True


def _ending_stream(end: Callable[..., None], release: Callable[[], None]) -> Callable[..., None]:
    def end_stream(**fields: Any) -> None:
        try:
            end(**fields)
        finally:
            release()

    return end_stream


def _capture_masked(handle: telemetry_dev.SpanHandle) -> bool:
    compatible = cast(Any, handle)
    masked = getattr(compatible, "capture_masked", None)
    if isinstance(masked, bool):
        return masked
    client = getattr(compatible, "_client", None)
    return getattr(client, "mask", _OMIT) is not None if client is not None else True


def _mark_capture_incomplete(fields: dict[str, Any], mask_output: bool) -> dict[str, Any]:
    fields["attributes"] = {**(fields.get("attributes") or {}), _CAPTURE_TRUNCATED_ATTRIBUTE: True}
    if mask_output:
        fields.pop("output", None)
    return fields


def _chat_partial(
    states: Mapping[int, _ChatChoice],
    usage: dict[str, int | float] | None,
    cost_usd: float | None,
    budget: telemetry_dev.CaptureBudget,
    *,
    completed: bool = True,
    mask_output: bool = False,
) -> dict[str, Any]:
    finish_reasons = [
        state.finish_reason
        for _, state in sorted(states.items())
        if state.finish_reason is not None
    ]
    fields = _mark_capture_truncated(
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
    if budget.truncated or not completed:
        _mark_capture_incomplete(fields, mask_output)
    return fields


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
    # ``finish`` is a method of the stream wrapper, which holds ``response``: held weakly, the hook
    # adds no reference cycle, so a dropped wrapper and its stream budget are freed at once.
    finish_method = weakref.WeakMethod(finish)

    def finish_if_alive() -> None:
        method = finish_method()
        if method is not None:
            method()

    close = getattr(response, "close", None)
    if callable(close):

        def _close_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return close(*args, **kwargs)
            finally:
                finish_if_alive()

        response.close = _close_hook
    aclose = getattr(response, "aclose", None)
    if callable(aclose):
        aclose_fn = cast(Callable[..., Any], aclose)

        async def _aclose_hook(*args: Any, **kwargs: Any) -> Any:
            try:
                return await aclose_fn(*args, **kwargs)
            finally:
                finish_if_alive()

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
        self._handle = handle
        self._started_at = started_at
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
        self._cost_usd: float | None = None
        self._saw_first = False
        self._consume: Iterator[Any] | None = None
        self._in_next = False
        self._completed = False
        self._mask_output_when_incomplete = _capture_masked(handle)
        self._budget = _stream_capture_budget()
        # Bound to the budget, not self, so the callback adds no reference cycle.
        self._end_stream = _ending_stream(_end_once(handle), partial(_release_budget, self._budget))
        self._ended = False
        _hook_response_close(inner, self._on_response_close)

    def _iterate(self) -> Iterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = next(self._inner)
                    received_at = time.perf_counter()
                except StopIteration:
                    self._completed = True
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                if not self._record(chunk, received_at):
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
        return _chat_partial(
            self._states,
            self._usage,
            self._cost_usd,
            self._budget,
            completed=self._completed,
            mask_output=self._mask_output_when_incomplete,
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

    def _end(self, **fields: Any) -> None:
        # The span has ended and the stream budget no longer counts bytes, so later events are
        # not kept.
        self._ended = True
        self._end_stream(**fields)

    def _record(self, chunk: Any, received_at: float) -> bool:
        if self._ended:
            return True
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
        if update.get("cost_usd") is not None:
            self._cost_usd = update["cost_usd"]
        return _chunk_error(chunk) is None


class _InstrumentedAsyncStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._started_at = started_at
        self._states: dict[int, _ChatChoice] = {}
        self._usage: dict[str, int | float] | None = None
        self._cost_usd: float | None = None
        self._saw_first = False
        self._consume: AsyncIterator[Any] | None = None
        self._in_next = False
        self._completed = False
        self._mask_output_when_incomplete = _capture_masked(handle)
        self._budget = _stream_capture_budget()
        # Bound to the budget, not self, so the callback adds no reference cycle.
        self._end_stream = _ending_stream(_end_once(handle), partial(_release_budget, self._budget))
        self._ended = False
        _hook_response_close(inner, self._on_response_close)

    async def _aiterate(self) -> AsyncIterator[Any]:
        try:
            while True:
                self._in_next = True
                try:
                    chunk = await self._inner.__anext__()
                    received_at = time.perf_counter()
                except StopAsyncIteration:
                    self._completed = True
                    break
                except BaseException as exc:
                    self._end(**self._partial(), error=exc)
                    raise
                finally:
                    self._in_next = False
                if not self._record(chunk, received_at):
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
        return _chat_partial(
            self._states,
            self._usage,
            self._cost_usd,
            self._budget,
            completed=self._completed,
            mask_output=self._mask_output_when_incomplete,
        )

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

    def _end(self, **fields: Any) -> None:
        # The span has ended and the stream budget no longer counts bytes, so later events are
        # not kept.
        self._ended = True
        self._end_stream(**fields)

    def _record(self, chunk: Any, received_at: float) -> bool:
        if self._ended:
            return True
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
        if update.get("cost_usd") is not None:
            self._cost_usd = update["cost_usd"]
        return _chunk_error(chunk) is None


def _write_fragment(target: dict[str, Any], key: str, value: str) -> None:
    """Appends a delta to ``target[key]`` in a buffer, since repeated concatenation copies the
    whole text for each delta. _joined_item reads the text back."""
    fragments = target.get(key)
    if not isinstance(fragments, _DeltaBuffer):
        fragments = _DeltaBuffer()
        fragments.write(f"{target.get(key, '')}")
        target[key] = fragments
    fragments.write(value)


def _joined_value(value: Any) -> Any:
    return value.getvalue() if isinstance(value, _DeltaBuffer) else value


def _joined_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    joined: dict[str, Any] = {}
    for key, value in cast(dict[str, Any], item).items():
        if key in ("content", "summary") and isinstance(value, list):
            value = [
                {name: _joined_value(field) for name, field in cast(dict[str, Any], part).items()}
                if isinstance(part, dict)
                else part
                for part in cast(list[Any], value)
            ]
        joined[key] = _joined_value(value)
    return joined


class _ResponsesStreamState:
    def __init__(self) -> None:
        self.partial: dict[str, Any] = {}
        self.retained_output: Any | None = None
        self.saw_terminal_snapshot = False
        self._budget = _stream_capture_budget()
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
        return [
            *[_joined_item(value) for _, value in sorted(items.items())],
            *self._synthetic_events,
        ]

    @staticmethod
    def _can_accept(budget: telemetry_dev.CaptureBudget, value: Any) -> bool:
        candidate = telemetry_dev.CaptureBudget(budget.max_bytes, budget.max_items)
        candidate.bytes_used = budget.bytes_used
        candidate.items_used = budget.items_used
        candidate.truncated = budget.truncated
        return candidate.accept(value)

    def _replace_output(self, output: Any) -> bool:
        replacement = _measuring_budget(self._budget)
        if not replacement.accept(output):
            self._budget.truncated = True
            return False
        replacement.truncated = self._provider_events_truncated
        if not _adopt_budget(self._budget, replacement):
            self._budget.truncated = True
            return False
        return True

    def release(self) -> None:
        _release_budget(self._budget)

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
            preflight = _measuring_budget(self._budget) if replacement else self._budget
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
            preflight = _measuring_budget(self._budget) if replacement else self._budget
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
                _write_fragment(item, field, value)
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
            preflight = _measuring_budget(self._budget) if replacement else self._budget
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
                _write_fragment(summary, "text", value)
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
            _write_fragment(part, "refusal", value)
        else:
            _write_fragment(part, "text", value)
        return True

    def record(self, event: Any) -> None:
        event = _response_event(event)
        response = _field(event, "response")
        if response is not None:
            fields = _responses_response(response, include_error=False, include_output=False)
            response_budget = _measuring_budget(self._budget)
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
                        adopted = False
                        if response_budget.accept(response_artifact):
                            response_budget.truncated = self._provider_events_truncated
                            adopted = _adopt_budget(self._budget, response_budget)
                        if adopted:
                            self._items.clear()
                            self._content.clear()
                            self._summary.clear()
                            for index, item in enumerate(response_items):
                                if isinstance(item, dict):
                                    self._hydrate_item(index, cast(dict[str, Any], item))
                            self.retained_output = response_artifact
                        else:
                            self._budget.truncated = True
            self.partial = fields
            if self.retained_output is not None:
                self.partial["output"] = self.retained_output
            if _field(event, "type") in _TERMINAL_RESPONSE_EVENTS:
                self.saw_terminal_snapshot = True
        elif self._record_event(event):
            self._sync_output()
        _mark_capture_truncated(self.partial, self._budget)

    def fields(self, mask_output: bool, *, failed: bool = False) -> dict[str, Any]:
        fields = dict(self.partial)
        output = fields.get("output")
        if isinstance(output, list):
            fields["output"] = [_joined_item(item) for item in cast(list[Any], output)]
        if failed or self._budget.truncated or not self.saw_terminal_snapshot:
            _mark_capture_incomplete(fields, mask_output)
        return fields


class _InstrumentedResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._started_at = started_at
        self._saw_first = False
        self._state = _ResponsesStreamState()
        # Bound to the state, not self, so the callback adds no reference cycle.
        self._end_stream = _ending_stream(_end_once(handle), self._state.release)
        self._ended = False
        self._mask_output_when_incomplete = _capture_masked(handle)
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
                    self._end(**self._failed_fields(), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event, received_at)
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
            self._end(**self._failed_fields(), error=exc)
        self.close()

    def _fields(self) -> dict[str, Any]:
        return self._state.fields(self._mask_output_when_incomplete)

    def _failed_fields(self) -> dict[str, Any]:
        return self._state.fields(self._mask_output_when_incomplete, failed=True)

    def _finish(self) -> None:
        self._end(**self._fields())

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

    def _end(self, **fields: Any) -> None:
        # The span has ended and the stream budget no longer counts bytes, so later events are
        # not kept.
        self._ended = True
        self._end_stream(**fields)

    def _record(self, event: Any, received_at: float) -> None:
        if self._ended:
            return
        if _response_event_has_output(event):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        self._state.record(event)
        event = _response_event(event)
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._fields())
        elif event_type == "response.failed":
            self._end(**self._fields(), error=_response_failed_error(_field(event, "response")))
        elif event_type == "response.incomplete":
            self._end(**self._fields())
        elif event_type == "error":
            self._end(**self._failed_fields(), error=_response_stream_error(event))


class _InstrumentedAsyncResponsesStream:
    def __init__(self, inner: Any, handle: telemetry_dev.SpanHandle, started_at: float) -> None:
        self._inner = inner
        self._handle = handle
        self._started_at = started_at
        self._saw_first = False
        self._state = _ResponsesStreamState()
        # Bound to the state, not self, so the callback adds no reference cycle.
        self._end_stream = _ending_stream(_end_once(handle), self._state.release)
        self._ended = False
        self._mask_output_when_incomplete = _capture_masked(handle)
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
                    self._end(**self._failed_fields(), error=exc)
                    raise
                finally:
                    self._in_next = False
                self._record(event, received_at)
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
            self._end(**self._failed_fields(), error=exc)
        await self.close()

    def _fields(self) -> dict[str, Any]:
        return self._state.fields(self._mask_output_when_incomplete)

    def _failed_fields(self) -> dict[str, Any]:
        return self._state.fields(self._mask_output_when_incomplete, failed=True)

    def _finish(self) -> None:
        self._end(**self._fields())

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

    def _end(self, **fields: Any) -> None:
        # The span has ended and the stream budget no longer counts bytes, so later events are
        # not kept.
        self._ended = True
        self._end_stream(**fields)

    def _record(self, event: Any, received_at: float) -> None:
        if self._ended:
            return
        if _response_event_has_output(event):
            record_output_chunk = getattr(self._handle, "record_output_chunk", None)
            if callable(record_output_chunk):
                record_output_chunk(received_at * 1000)
        if not self._saw_first:
            self._saw_first = True
            self._handle.update(
                time_to_first_chunk_ms=(time.perf_counter() - self._started_at) * 1000
            )
        self._state.record(event)
        event = _response_event(event)
        event_type = _field(event, "type")
        if event_type == "response.completed":
            self._end(**self._fields())
        elif event_type == "response.failed":
            self._end(**self._fields(), error=_response_failed_error(_field(event, "response")))
        elif event_type == "response.incomplete":
            self._end(**self._fields())
        elif event_type == "error":
            self._end(**self._failed_fields(), error=_response_stream_error(event))


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
