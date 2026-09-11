from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import telemetry_dev

from ._fields import clean, error_fields, merge_fields, usage_from_converse
from ._messages import normalize_content_block


class StreamState:
    def feed(self, event: Any) -> bool | None:
        raise NotImplementedError

    def finish(self, *, partial: bool) -> dict[str, Any]:
        raise NotImplementedError


def _event_has_output(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    output = event.get("output") if isinstance(event.get("output"), dict) else {}
    if isinstance(output.get("text"), str) and output["text"]:
        return True
    block = (
        event.get("contentBlockDelta") if isinstance(event.get("contentBlockDelta"), dict) else {}
    )
    delta = block.get("delta") if isinstance(block.get("delta"), dict) else {}
    reasoning = (
        delta.get("reasoningContent") if isinstance(delta.get("reasoningContent"), dict) else {}
    )
    tool_use = delta.get("toolUse") if isinstance(delta.get("toolUse"), dict) else {}
    if any(
        isinstance(value, str) and bool(value)
        for value in (delta.get("text"), reasoning.get("text"), tool_use.get("input"))
    ):
        return True
    chunk = event.get("chunk") if isinstance(event.get("chunk"), dict) else {}
    raw = chunk.get("bytes")
    if isinstance(raw, bytes | bytearray):
        text = bytes(raw).decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        return False
    if not text:
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return True
    if isinstance(parsed, dict) and isinstance(parsed.get("delta"), dict):
        if any(
            isinstance(value, str) and bool(value)
            for value in (parsed["delta"].get("thinking"), parsed["delta"].get("partial_json"))
        ):
            return True
    return bool(_text_from_provider_chunk(parsed))


class ConverseStreamState(StreamState):
    def __init__(self) -> None:
        self.budget = telemetry_dev.CaptureBudget.from_client()
        self.role = "assistant"
        self.blocks: dict[int, dict[str, Any]] = {}
        self.stop_reason: str | None = None
        self.usage: dict[str, int] | None = None
        self.metadata: dict[str, Any] = {}
        self.response_model: str | None = None

    def feed(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        start = event.get("messageStart")
        if isinstance(start, dict) and isinstance(start.get("role"), str):
            self.role = start["role"]
        block_start = event.get("contentBlockStart")
        if isinstance(block_start, dict):
            index = block_start.get("contentBlockIndex")
            idx = index if isinstance(index, int) else len(self.blocks)
            start_payload = (
                block_start.get("start") if isinstance(block_start.get("start"), dict) else {}
            )
            if self.budget.accept(start_payload):
                tool_use = (
                    start_payload.get("toolUse")
                    if isinstance(start_payload.get("toolUse"), dict)
                    else None
                )
                tool_result = (
                    start_payload.get("toolResult")
                    if isinstance(start_payload.get("toolResult"), dict)
                    else None
                )
                image = (
                    start_payload.get("image")
                    if isinstance(start_payload.get("image"), dict)
                    else None
                )
                if tool_use:
                    self.blocks[idx] = {
                        "kind": "tool",
                        "id": tool_use.get("toolUseId"),
                        "name": tool_use.get("name"),
                        "input": "",
                    }
                elif tool_result:
                    self.blocks[idx] = {
                        "kind": "content",
                        "content": {
                            "toolResult": {
                                "toolUseId": tool_result.get("toolUseId"),
                                "status": tool_result.get("status"),
                                "content": [],
                            }
                        },
                    }
                elif image:
                    self.blocks[idx] = {"kind": "content", "content": {"image": image}}
                else:
                    self.blocks[idx] = {"kind": "text", "text": ""}
        delta_event = event.get("contentBlockDelta")
        if isinstance(delta_event, dict):
            index = delta_event.get("contentBlockIndex")
            idx = index if isinstance(index, int) else 0
            delta = delta_event.get("delta") if isinstance(delta_event.get("delta"), dict) else {}
            if self.budget.accept(delta):
                block = self.blocks.setdefault(idx, {"kind": "text", "text": ""})
                if isinstance(delta.get("text"), str):
                    block["kind"] = "text"
                    block["text"] = f"{block.get('text', '')}{delta['text']}"
                tool_delta = (
                    delta.get("toolUse") if isinstance(delta.get("toolUse"), dict) else None
                )
                if tool_delta and isinstance(tool_delta.get("input"), str):
                    block["kind"] = "tool"
                    block["input"] = f"{block.get('input', '')}{tool_delta['input']}"
                reasoning = (
                    delta.get("reasoningContent")
                    if isinstance(delta.get("reasoningContent"), dict)
                    else None
                )
                if reasoning:
                    block["kind"] = "reasoning"
                    text = reasoning.get("text") if isinstance(reasoning.get("text"), str) else ""
                    if not text and reasoning.get("redactedContent") is not None:
                        text = "[redacted]"
                    block["text"] = f"{block.get('text', '')}{text}"
                image = delta.get("image")
                if isinstance(image, dict):
                    content_value = block.get("content")
                    content: dict[str, Any] = (
                        content_value if isinstance(content_value, dict) else {}
                    )
                    current_value = content.get("image")
                    current: dict[str, Any] = (
                        current_value if isinstance(current_value, dict) else {}
                    )
                    block["kind"] = "content"
                    block["content"] = {"image": {**current, **image}}
                tool_result_value = delta.get("toolResult")
                tool_result: list[Any] | None = (
                    tool_result_value if isinstance(tool_result_value, list) else None
                )
                if tool_result:
                    content_value = block.get("content")
                    content = content_value if isinstance(content_value, dict) else {}
                    current_value = content.get("toolResult")
                    current = current_value if isinstance(current_value, dict) else {}
                    current_content_value = current.get("content")
                    current_content: list[Any] = (
                        current_content_value if isinstance(current_content_value, list) else []
                    )
                    block["kind"] = "content"
                    block["content"] = {
                        "toolResult": {**current, "content": [*current_content, *tool_result]}
                    }
                citation = delta.get("citation")
                if isinstance(citation, dict):
                    citations_value = block.get("citations")
                    citations: list[Any] = (
                        citations_value if isinstance(citations_value, list) else []
                    )
                    block["citations"] = [*citations, citation]
        stop = event.get("messageStop")
        if isinstance(stop, dict) and isinstance(stop.get("stopReason"), str):
            self.stop_reason = stop["stopReason"]
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            self.usage = usage_from_converse(metadata.get("usage")) or self.usage
            metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
            trace = metadata.get("trace") if isinstance(metadata.get("trace"), dict) else {}
            router = (
                trace.get("promptRouter") if isinstance(trace.get("promptRouter"), dict) else {}
            )
            guard = trace.get("guardrail") if isinstance(trace.get("guardrail"), dict) else {}
            if isinstance(router.get("invokedModelId"), str):
                self.response_model = router["invokedModelId"]
            self.metadata.update(
                clean(
                    {
                        "server_latency_ms": metrics.get("latencyMs"),
                        "guardrail_action": guard.get("action"),
                        "guardrail_action_reason": guard.get("actionReason"),
                    }
                )
            )

    def finish(self, *, partial: bool) -> dict[str, Any]:
        parts = [self._part_for(self.blocks[index]) for index in sorted(self.blocks)]
        message = {"role": self.role, "parts": parts}
        if not partial and self.stop_reason is not None:
            message["finish_reason"] = self.stop_reason
        return clean(
            {
                "output": [message] if parts else None,
                "finish_reason": None if partial else self.stop_reason,
                "usage": self.usage,
                "response_model": self.response_model,
                "metadata": self.metadata or None,
            }
        )

    def _part_for(self, block: dict[str, Any]) -> dict[str, Any]:
        if block.get("kind") == "tool":
            raw = block.get("input") if isinstance(block.get("input"), str) else ""
            try:
                arguments: Any = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                arguments = raw
            return clean(
                {
                    "type": "tool_call",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": arguments,
                }
            )
        if block.get("kind") == "reasoning":
            return {"type": "reasoning", "content": block.get("text", "")}
        if block.get("kind") == "content" and isinstance(block.get("content"), dict):
            parts = normalize_content_block(block["content"])
            return parts[0] if parts else {}
        part = {"type": "text", "content": block.get("text", "")}
        if isinstance(block.get("citations"), list) and block["citations"]:
            part["citations"] = block["citations"]
        return part


class InvokeModelStreamState(StreamState):
    def __init__(self) -> None:
        self.budget = telemetry_dev.CaptureBudget.from_client()
        self.chunks: list[Any] = []
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None

    def feed(self, event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        chunk = event.get("chunk") if isinstance(event.get("chunk"), dict) else {}
        raw = chunk.get("bytes")
        if isinstance(raw, bytes | bytearray):
            text = bytes(raw).decode("utf-8")
        elif isinstance(raw, str):
            text = raw
        else:
            return False
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return bool(text)
        if self.budget.accept(parsed):
            self.chunks.append(parsed)
        if isinstance(parsed, dict):
            self._collect_anthropic_usage(parsed)
            if isinstance(parsed.get("amazon-bedrock-invocationMetrics"), dict):
                metrics = parsed["amazon-bedrock-invocationMetrics"]
                self.usage.update(
                    clean(
                        {
                            "input_tokens": metrics.get("inputTokenCount"),
                            "output_tokens": metrics.get("outputTokenCount"),
                        }
                    )
                )
            billed_units = (
                parsed.get("meta", {}).get("billed_units")
                if isinstance(parsed.get("meta"), dict)
                and isinstance(parsed.get("meta", {}).get("billed_units"), dict)
                else {}
            )
            self.finish_reason = (
                parsed.get("stop_reason")
                or parsed.get("finish_reason")
                or (
                    parsed.get("outputs", [{}])[0].get("stop_reason")
                    if isinstance(parsed.get("outputs"), list)
                    and parsed.get("outputs")
                    and isinstance(parsed.get("outputs", [None])[0], dict)
                    else None
                )
                or (
                    parsed.get("generations", [{}])[0].get("finish_reason")
                    if isinstance(parsed.get("generations"), list)
                    and parsed.get("generations")
                    and isinstance(parsed.get("generations", [None])[0], dict)
                    else None
                )
                or parsed.get("completionReason")
                or self.finish_reason
            )
            self.usage.update(
                clean(
                    {
                        "input_tokens": parsed.get("inputTextTokenCount")
                        or parsed.get("prompt_token_count")
                        or billed_units.get("input_tokens"),
                        "output_tokens": parsed.get("totalOutputTextTokenCount")
                        or parsed.get("generation_token_count")
                        or billed_units.get("output_tokens"),
                    }
                )
            )
        if isinstance(parsed, dict) and isinstance(parsed.get("delta"), dict):
            delta = parsed["delta"]
            if any(
                isinstance(value, str) and bool(value)
                for value in (delta.get("thinking"), delta.get("partial_json"))
            ):
                return True
        return bool(_text_from_provider_chunk(parsed))

    def finish(self, *, partial: bool) -> dict[str, Any]:
        del partial
        text = "".join(_text_from_provider_chunk(chunk) for chunk in self.chunks)
        output: Any = self.chunks
        if text:
            output = [
                clean(
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": text}],
                        "finish_reason": self.finish_reason,
                    }
                )
            ]
        return clean(
            {
                "output": output if output else None,
                "usage": self.usage or None,
                "finish_reason": self.finish_reason,
            }
        )

    def _collect_anthropic_usage(self, parsed: dict[str, Any]) -> None:
        message = parsed.get("message") if isinstance(parsed.get("message"), dict) else {}
        delta = parsed.get("delta") if isinstance(parsed.get("delta"), dict) else {}
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        message_usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        self.finish_reason = delta.get("stop_reason") or self.finish_reason
        self.usage.update(
            clean(
                {
                    "input_tokens": message_usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "cache_read_input_tokens": message_usage.get("cache_read_input_tokens"),
                    "cache_creation_input_tokens": message_usage.get("cache_creation_input_tokens"),
                }
            )
        )


class AgentStreamState(StreamState):
    def __init__(self, *, capture_trace: bool = False) -> None:
        self.budget = telemetry_dev.CaptureBudget.from_client()
        self.capture_trace = capture_trace
        self.text = ""
        self.usage: dict[str, int] = {}
        self.trace_count = 0
        self.failure_reason: str | None = None
        self.guardrail_action: str | None = None
        self.return_control: Any = None
        self.traces: list[Any] = []

    def feed(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        chunk = event.get("chunk") if isinstance(event.get("chunk"), dict) else {}
        raw = chunk.get("bytes")
        if isinstance(raw, bytes | bytearray):
            piece = bytes(raw).decode("utf-8")
            if self.budget.accept(piece):
                self.text += piece
        elif isinstance(raw, str) and self.budget.accept(raw):
            self.text += raw
        return_control = (
            event.get("returnControl") if isinstance(event.get("returnControl"), dict) else None
        )
        if return_control is not None:
            self.return_control = return_control
        trace = event.get("trace") if isinstance(event.get("trace"), dict) else None
        if trace:
            self.trace_count += 1
            if self.capture_trace and self.budget.accept(trace):
                self.traces.append(trace)
            self._collect_trace(trace)

    def finish(self, *, partial: bool) -> dict[str, Any]:
        del partial
        metadata = clean(
            {
                "trace_event_count": self.trace_count or None,
                "failure_reason": self.failure_reason,
                "guardrail_action": self.guardrail_action,
                "return_control": True if self.return_control is not None else None,
                "agent_trace": self.traces if self.capture_trace and self.traces else None,
            }
        )
        output: Any = None
        if self.return_control is not None:
            output = {"returnControl": self.return_control}
        elif self.text:
            output = [{"role": "assistant", "parts": [{"type": "text", "content": self.text}]}]
        return clean({"output": output, "usage": self.usage or None, "metadata": metadata or None})

    def _collect_trace(self, value: Any) -> None:
        if isinstance(value, dict):
            output = value.get("modelInvocationOutput")
            if isinstance(output, dict):
                metadata = (
                    output.get("metadata") if isinstance(output.get("metadata"), dict) else {}
                )
                usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else None
                if usage:
                    input_tokens = _int_value(usage.get("inputTokens"))
                    output_tokens = _int_value(usage.get("outputTokens"))
                    total_tokens = _int_value(usage.get("totalTokens"))
                    if input_tokens is not None:
                        self.usage["input_tokens"] = (
                            self.usage.get("input_tokens", 0) + input_tokens
                        )
                    if output_tokens is not None:
                        self.usage["output_tokens"] = (
                            self.usage.get("output_tokens", 0) + output_tokens
                        )
                    if total_tokens is not None:
                        self.usage["total_tokens"] = (
                            self.usage.get("total_tokens", 0) + total_tokens
                        )
                    elif input_tokens is not None or output_tokens is not None:
                        self.usage["total_tokens"] = self.usage.get(
                            "input_tokens", 0
                        ) + self.usage.get("output_tokens", 0)
            failure = value.get("failureTrace")
            if isinstance(failure, dict) and isinstance(failure.get("failureReason"), str):
                self.failure_reason = failure["failureReason"]
            guard = value.get("guardrailTrace")
            if isinstance(guard, dict) and isinstance(guard.get("action"), str):
                self.guardrail_action = guard["action"]
            for child in value.values():
                self._collect_trace(child)
        elif isinstance(value, list):
            for child in value:
                self._collect_trace(child)


class RagStreamState(StreamState):
    def __init__(self) -> None:
        self.budget = telemetry_dev.CaptureBudget.from_client()
        self.text = ""
        self.citation_count = 0
        self.guardrail_action: str | None = None

    def feed(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        output = event.get("output") if isinstance(event.get("output"), dict) else {}
        if isinstance(output.get("text"), str) and self.budget.accept(output["text"]):
            self.text += output["text"]
        if "citation" in event:
            self.citation_count += 1
        guard = event.get("guardrail") if isinstance(event.get("guardrail"), dict) else {}
        if isinstance(guard.get("action"), str):
            self.guardrail_action = guard["action"]

    def finish(self, *, partial: bool) -> dict[str, Any]:
        del partial
        return clean(
            {
                "output": [{"role": "assistant", "parts": [{"type": "text", "content": self.text}]}]
                if self.text
                else None,
                "metadata": clean(
                    {
                        "citation_count": self.citation_count or None,
                        "guardrail_action": self.guardrail_action,
                    }
                )
                or None,
            }
        )


class FlowStreamState(StreamState):
    def __init__(self) -> None:
        self.budget = telemetry_dev.CaptureBudget.from_client()
        self.outputs: list[Any] = []
        self.completion_reason: str | None = None

    def feed(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        output = (
            event.get("flowOutputEvent") if isinstance(event.get("flowOutputEvent"), dict) else None
        )
        if output:
            content = output.get("content")
            if self.budget.accept(content):
                self.outputs.append(content)
        input_request = (
            event.get("flowMultiTurnInputRequestEvent")
            if isinstance(event.get("flowMultiTurnInputRequestEvent"), dict)
            else None
        )
        if input_request:
            content = input_request.get("content")
            if self.budget.accept(content):
                self.outputs.append(content)
        completion = (
            event.get("flowCompletionEvent")
            if isinstance(event.get("flowCompletionEvent"), dict)
            else None
        )
        if completion and isinstance(completion.get("completionReason"), str):
            self.completion_reason = completion["completionReason"]

    def finish(self, *, partial: bool) -> dict[str, Any]:
        del partial
        return clean({"output": self.outputs or None, "finish_reason": self.completion_reason})


class InstrumentedEventStream:
    def __init__(
        self,
        stream: Any,
        state: StreamState,
        finish: Callable[[dict[str, Any]], None],
        started_at: float,
        handle: Any,
    ) -> None:
        self._stream = stream
        self._state = state
        self._finish_callback = finish
        self._started_at = started_at
        self._handle = handle
        self._ended = False
        self._first = False
        self._track_output_chunks = isinstance(state, ConverseStreamState | InvokeModelStreamState)

    def __iter__(self) -> Iterator[Any]:
        completed = False
        try:
            for event in self._stream:
                received_at = time.perf_counter()
                if not self._first:
                    self._first = True
                    self._finish_callback(
                        {"time_to_first_chunk_ms": (received_at - self._started_at) * 1000}
                    )
                try:
                    state_has_output = self._state.feed(event)
                except Exception:
                    state_has_output = None
                if self._track_output_chunks and (
                    state_has_output if state_has_output is not None else _event_has_output(event)
                ):
                    try:
                        record_output_chunk = getattr(self._handle, "record_output_chunk", None)
                        if callable(record_output_chunk):
                            record_output_chunk(received_at * 1000)
                    except Exception:
                        pass
                yield event
            completed = True
        except GeneratorExit:
            self._finish(partial=True)
            raise
        except BaseException as exc:
            self._finish(partial=True, error=exc)
            raise
        finally:
            if completed:
                self._finish(partial=False)

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()
        self._finish(partial=True)

    def __del__(self) -> None:
        self._finish(partial=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _finish(self, *, partial: bool, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        fields = self._state.finish(partial=partial)
        if error is not None:
            fields = merge_fields(fields, error_fields(error))
        self._finish_callback(fields)


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text_from_provider_chunk(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    delta = value.get("delta") if isinstance(value.get("delta"), dict) else {}
    block_delta = (
        value.get("content_block_delta")
        if isinstance(value.get("content_block_delta"), dict)
        else {}
    )
    nested = block_delta.get("delta") if isinstance(block_delta.get("delta"), dict) else {}
    for candidate in (
        value.get("outputText"),
        value.get("generation"),
        value.get("completion"),
        value.get("text"),
        delta.get("text"),
        nested.get("text"),
        (
            value.get("outputs", [{}])[0].get("text")
            if isinstance(value.get("outputs"), list)
            and value.get("outputs")
            and isinstance(value.get("outputs", [None])[0], dict)
            else None
        ),
        (
            value.get("generations", [{}])[0].get("text")
            if isinstance(value.get("generations"), list)
            and value.get("generations")
            and isinstance(value.get("generations", [None])[0], dict)
            else None
        ),
    ):
        if isinstance(candidate, str):
            return candidate
    return ""
