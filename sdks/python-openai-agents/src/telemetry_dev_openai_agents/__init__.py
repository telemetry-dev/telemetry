"""OpenAI Agents SDK integration for telemetry.dev.

``TelemetryDevProcessor`` is an Agents SDK tracing processor. It maps each Agents trace to one root
span and each Agents span to a telemetry.dev span under it, following the OpenAI Agents section of
the SDK conformance doc, which the TypeScript package ``@telemetry-dev/openai-agents`` follows too.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, cast

import telemetry_dev
from agents.tracing import Span, Trace, TracingProcessor
from opentelemetry import trace as otel_trace

__version__ = "0.0.0"

__all__ = ["TelemetryDevProcessor", "__version__"]

# Bounds the processor's memory when a host never ends its spans; past the cap new spans are
# dropped, not exported.
MAX_OPEN_SPANS = 10_000

# Bounds the trace roots kept so that spans arriving after their trace ended still find their root.
# Past the cap the root of the trace whose spans started or ended least recently is forgotten, and
# ended first if it is still open.
MAX_TRACES = 10_000

# The sandbox shell tools write this line in the header of their output, above `Output:`, once the
# command has exited. A command still running gets a session id line instead, and the command's own
# output below the header can contain any text.
_EXIT_CODE_LINE = re.compile(r"^Process exited with code (-?\d+)$", re.MULTILINE)
_OUTPUT_MARKER = "\nOutput:"

_SANDBOX_COMMAND_TOOLS = frozenset({"exec_command", "write_stdin"})
_SANDBOX_SPAN_PREFIX = "sandbox."

_Attributes = dict[str, Any]


@dataclass(eq=False)
class _TraceState:
    root: telemetry_dev.SpanHandle
    ended: bool = False
    # Latest observed span end, used to timestamp the root when the trace never ends.
    last_end: datetime | None = None


@dataclass(eq=False)
class _OpenSpan:
    handle: telemetry_dev.SpanHandle
    # The root state the span started under; a later trace can reuse the same trace id.
    trace: _TraceState | None


@dataclass
class _ToolCall:
    name: str
    id: str
    arguments: str | None


@dataclass
class _Fields:
    """Keyword arguments of ``SpanHandle.end``; ``attributes`` merge instead of replacing."""

    values: dict[str, Any] = field(default_factory=lambda: {})
    attributes: _Attributes = field(default_factory=lambda: {})

    def merge(self, other: _Fields) -> None:
        self.values.update(other.values)
        self.attributes.update(other.attributes)

    def content(self, input: Any, output: Any) -> None:
        """Sets input and output that the span recorded: the core SDK would send None as "null"."""
        if input is not None:
            self.values["input"] = _plain(input)
        if output is not None:
            self.values["output"] = _plain(output)


def _get(value: object, key: str) -> Any:
    """A key of a mapping or an attribute of an object, such as a pydantic model."""
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(key)
    return getattr(value, key, None)


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _list(value: object) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, (list, tuple)) else []


def _plain(value: Any) -> Any:
    """Pydantic models as dicts, so capture serializes their fields instead of their repr."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in cast(list[Any], value)]
    return value


def _time(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def _usage(value: object) -> dict[str, int] | None:
    """Usage of a model call: input and output token details as the Agents SDK records them."""
    if not isinstance(value, Mapping):
        return None
    usage = cast(Mapping[str, Any], value)
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    out = {
        key: v
        for key, v in (
            ("input_tokens", _int(usage.get("input_tokens"))),
            ("output_tokens", _int(usage.get("output_tokens"))),
            ("total_tokens", _int(usage.get("total_tokens"))),
            ("cache_read_input_tokens", _int(_get(input_details, "cached_tokens"))),
            ("reasoning_output_tokens", _int(_get(output_details, "reasoning_tokens"))),
        )
        if v is not None
    }
    return out or None


def _primitive_metadata(value: object) -> dict[str, str | int | float | bool] | None:
    if not isinstance(value, Mapping):
        return None
    out = {
        key: v
        for key, v in cast(Mapping[str, Any], value).items()
        if isinstance(v, (str, int, float, bool))
    }
    return out or None


def _start_of(data: Any) -> tuple[str, dict[str, Any]]:
    """Name and ``start_span`` keyword arguments known when the span starts."""
    kind: str = data.type
    if kind == "agent":
        return f"invoke_agent {data.name}", {"type": "agent", "agent_name": data.name}
    if kind == "task":
        return f"task {data.name}", {}
    if kind == "turn":
        return f"turn {data.turn}", {"attributes": {"openai_agents.turn": data.turn}}
    if kind == "response":
        return "chat", {"type": "generation", "provider": "openai"}
    if kind == "generation":
        model = _str(data.model)
        return (f"chat {model}" if model else "chat"), {"type": "generation", "model": model}
    if kind == "function":
        return f"execute_tool {data.name}", {"type": "tool", "tool_name": data.name}
    if kind == "handoff":
        return "handoff", {}
    if kind == "guardrail":
        return f"guardrail {data.name}", {"attributes": {"openai_agents.guardrail.name": data.name}}
    if kind == "mcp_tools":
        server = _str(data.server)
        return (f"mcp_tools {server}" if server else "mcp_tools"), {}
    if kind == "custom":
        return data.name, {}
    return kind, {}


def _end_of(data: Any) -> _Fields:
    """Fields known when the span ends, apart from sandbox attributes."""
    kind: str = data.type
    fields = _Fields()
    if kind == "agent":
        # The SDK fills the tool list once the agent's tools are resolved, after the span starts.
        for key, value in (
            ("openai_agents.agent.tools", data.tools),
            ("openai_agents.agent.handoffs", data.handoffs),
            ("openai_agents.agent.output_type", data.output_type),
        ):
            if value:
                fields.attributes[key] = value
    elif kind == "response":
        response = data.response
        model = _str(_get(response, "model"))
        if model:
            fields.values.update(name=f"chat {model}", model=model)
        fields.values.update(
            response_model=model,
            # Without sensitive data the SDK keeps only the response id, in a private field.
            response_id=_str(_get(response, "id")) or _str(getattr(data, "_response_id", None)),
            usage=_usage(data.usage),
        )
        fields.content(data.input, _get(response, "output"))
    elif kind == "generation":
        # The Chat Completions model records the usage with the same token details as a Responses
        # call. Its output is the assistant message, or, for a streamed call, the Responses API
        # response it builds from the stream, which has the response id and model.
        model = _str(data.model)
        if model:
            fields.values.update(name=f"chat {model}", model=model)
        streamed = next(iter(_list(data.output)), None)
        fields.values.update(
            usage=_usage(data.usage),
            response_model=_str(_get(streamed, "model")),
            response_id=_str(_get(streamed, "id")),
        )
        fields.content(data.input, data.output)
    elif kind == "function":
        fields.content(data.input, data.output)
        server = _str(_get(data.mcp_data, "server"))
        if server:
            fields.attributes["openai_agents.mcp.server"] = server
    elif kind == "handoff":
        if data.from_agent and data.to_agent:
            fields.values["name"] = f"handoff {data.from_agent} -> {data.to_agent}"
        if data.from_agent:
            fields.attributes["openai_agents.handoff.from"] = data.from_agent
        if data.to_agent:
            fields.attributes["openai_agents.handoff.to"] = data.to_agent
    elif kind == "guardrail":
        fields.attributes["openai_agents.guardrail.triggered"] = bool(data.triggered)
    elif kind == "mcp_tools":
        if data.result is not None:
            fields.attributes["openai_agents.mcp.tool_count"] = len(data.result)
    elif kind == "custom" and not data.name.startswith(_SANDBOX_SPAN_PREFIX):
        fields.values["metadata"] = _primitive_metadata(data.data)
    return fields


def _tool_calls_of(data: Any) -> list[_ToolCall]:
    """The tool calls in the output of a model call: Responses API ``function_call`` items, or the
    ``tool_calls`` of a Chat Completions message. The Agents SDK records the output only when
    ``trace_include_sensitive_data`` is on, which is its default."""
    calls: list[_ToolCall] = []

    def add(name: object, call_id: object, arguments: object) -> None:
        call_name, call_id_str = _str(name), _str(call_id)
        if call_name and call_id_str:
            calls.append(_ToolCall(call_name, call_id_str, _str(arguments)))

    if data.type == "response":
        for item in _list(_get(data.response, "output")):
            if _get(item, "type") == "function_call":
                add(_get(item, "name"), _get(item, "call_id"), _get(item, "arguments"))
    elif data.type == "generation":
        for message in _list(data.output):
            for call in _list(_get(message, "tool_calls")):
                function = _get(call, "function")
                add(_get(function, "name"), _get(call, "id"), _get(function, "arguments"))
            # A streamed call records a Responses API response instead of a message.
            for item in _list(_get(message, "output")):
                if _get(item, "type") == "function_call":
                    add(_get(item, "name"), _get(item, "call_id"), _get(item, "arguments"))
    return calls


@lru_cache(maxsize=64)
def _error_class(name: str) -> type[Exception]:
    """An exception class with the given name, which telemetry.dev records as ``error.type``."""
    return type(name, (Exception,), {})


def _exit_code_of(output: object) -> int | None:
    if not isinstance(output, str):
        return None
    marker = output.find(_OUTPUT_MARKER)
    header = output if marker < 0 else output[:marker]
    match = _EXIT_CODE_LINE.search(header)
    return int(match[1]) if match else None


def _is_exited_command(data: Any) -> bool:
    """A ``sandbox.exec`` span of a command that ran to its exit, which the SDK marks as an error
    when the exit code is not zero. An exec that raised has no exit code."""
    return (
        data.type == "custom"
        and data.name == "sandbox.exec"
        and _int(_get(data.data, "process.exit.code")) is not None
    )


def _error_of(span: Span[Any]) -> Exception | None:
    error = span.error
    if not error or _is_exited_command(span.span_data):
        return None
    data = span.span_data
    sandbox_type = _str(_get(data.data, "error.type")) if data.type == "custom" else None
    return _error_class(sandbox_type or "Error")(_str(error.get("message")) or "")


def _sandbox_span_end(data: Any) -> _Fields:
    fields = _Fields()
    if data.type != "custom" or not data.name.startswith(_SANDBOX_SPAN_PREFIX):
        return fields
    fields.attributes["sandbox.operation"] = data.name[len(_SANDBOX_SPAN_PREFIX) :]
    backend = _str(_get(data.data, "backend_id")) or _str(_get(data.data, "sandbox.backend"))
    if backend:
        fields.attributes["sandbox.backend"] = backend
    exit_code = _int(_get(data.data, "process.exit.code"))
    if data.name == "sandbox.exec" and exit_code is not None:
        fields.attributes["process.exit.code"] = exit_code
    return fields


def _command_tool_end(data: Any) -> _Fields:
    fields = _Fields()
    if data.type == "function" and data.name in _SANDBOX_COMMAND_TOOLS:
        exit_code = _exit_code_of(data.output)
        # A non-zero exit is recorded, not raised: `grep` exits 1 when nothing matches.
        if exit_code is not None:
            fields.attributes["process.exit.code"] = exit_code
    return fields


def _report(exc: BaseException) -> None:
    try:
        client = telemetry_dev.get_client()
        if client is not None:
            client.report("openai agents tracing failed", exc)
    except Exception:
        pass


class TelemetryDevProcessor(TracingProcessor):
    """Sends OpenAI Agents SDK traces to telemetry.dev.

    Call ``telemetry_dev.init()`` first, then register the processor with
    ``agents.add_trace_processor``, which keeps the SDK's default OpenAI exporter, or
    ``agents.set_trace_processors([...])``, which replaces it.

    Each Agents trace gets one root span, and its spans keep their parents. Model calls are
    ``chat`` spans and the only spans with token usage; task, turn, and agent spans repeat the
    totals of their calls and carry none. Sandbox spans keep their ``sandbox.*`` names and get
    ``sandbox.operation`` and, when the SDK records it, ``sandbox.backend``. ``sandbox.exec`` spans
    and the ``exec_command`` and ``write_stdin`` tool spans get ``process.exit.code`` once the
    command has exited; a non-zero exit is not an error.

    The root span ends when the trace ends, which the Agents SDK does even when a run raises.
    ``force_flush`` and ``shutdown`` also end the root of every trace with no open span, at the end
    of its last span, such as a trace that a caller started and never finished. Spans that end
    after the root still nest under it while the processor holds the root (the ``MAX_TRACES`` most
    recently active traces): telemetry.dev takes a trace's start and end from all of its spans.

    The Agents SDK calls processors from any thread, so all state is guarded by one lock. The lock
    is reentrant because starting and ending spans can run the core SDK's ``mask`` and
    ``on_error``, which may start Agents spans on the same thread; loops over the state iterate
    over copies for the same reason.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Least recently active first: span starts and ends move a trace to the end.
        self._traces: dict[str, _TraceState] = {}
        self._spans: dict[str, _OpenSpan] = {}
        # Tool calls that a model call asked for and no tool span has claimed yet, by the Agents
        # span id of the model call's parent (its turn or agent span), under which the SDK also runs
        # the tools. The Agents SDK puts no call id on its tool spans, so a tool span takes it here.
        self._pending_calls: dict[str, list[_ToolCall]] = {}
        self._reported_cap = False

    def _touch(self, trace_id: str, state: _TraceState) -> None:
        self._traces.pop(trace_id, None)
        self._traces[trace_id] = state

    def _end_root(self, state: _TraceState, end_time: datetime | None = None) -> None:
        if state.ended:
            return
        state.ended = True
        state.root.end(end_time=end_time)

    def _end_idle_roots(self) -> None:
        """Ends, at their last span's end, the roots of traces that have no open span."""
        busy = {id(open_span.trace) for open_span in self._spans.values()}
        for state in list(self._traces.values()):
            if id(state) not in busy:
                self._end_root(state, state.last_end)

    def _claim_call(self, span: Span[Any]) -> str | None:
        """Takes the call a tool span ran: the first with its name and arguments, else its name."""
        data = span.span_data
        if data.type != "function" or not span.parent_id:
            return None
        calls = self._pending_calls.get(span.parent_id)
        if not calls:
            return None
        named = [call for call in calls if call.name == data.name]
        match = next((call for call in named if call.arguments == data.input), None)
        match = match or (named[0] if named else None)
        if match is None:
            return None
        calls.remove(match)
        return match.id

    def on_trace_start(self, trace: Trace) -> None:
        try:
            with self._lock:
                # A trace id seen again, such as a resumed run's or a fixed `trace_id`, keeps its
                # root while the processor still holds it, even when that root has already ended.
                if trace.trace_id in self._traces:
                    return
                if len(self._traces) >= MAX_TRACES:
                    oldest_id = next(iter(self._traces))
                    oldest = self._traces.pop(oldest_id)
                    self._end_root(oldest, oldest.last_end)
                attributes: _Attributes = {"openai_agents.trace_id": trace.trace_id}
                group_id = _str(getattr(trace, "group_id", None))
                if group_id:
                    attributes["gen_ai.conversation.id"] = group_id
                root = telemetry_dev.start_span(
                    trace.name,
                    attributes=attributes,
                    metadata=_primitive_metadata(getattr(trace, "metadata", None)),
                )
                self._traces[trace.trace_id] = _TraceState(root)
        except Exception as exc:
            _report(exc)

    def on_trace_end(self, trace: Trace) -> None:
        try:
            with self._lock:
                state = self._traces.get(trace.trace_id)
                if state is not None:
                    self._end_root(state)
        except Exception as exc:
            _report(exc)

    def on_span_start(self, span: Span[Any]) -> None:
        try:
            with self._lock:
                if len(self._spans) >= MAX_OPEN_SPANS:
                    if not self._reported_cap:
                        self._reported_cap = True
                        _report(
                            RuntimeError(
                                f"more than {MAX_OPEN_SPANS} open Agents spans; dropping new spans"
                            )
                        )
                    return
                state = self._traces.get(span.trace_id)
                if state is not None:
                    self._touch(span.trace_id, state)
                parent_span = self._spans.get(span.parent_id) if span.parent_id else None
                parent = parent_span.handle if parent_span else (state.root if state else None)
                name, options = _start_of(span.span_data)
                handle = telemetry_dev.start_span(
                    name,
                    parent=otel_trace.set_span_in_context(parent.span) if parent else None,
                    start_time=_time(span.started_at),
                    **options,
                )
                self._spans[span.span_id] = _OpenSpan(handle, state)
        except Exception as exc:
            _report(exc)

    def on_span_end(self, span: Span[Any]) -> None:
        try:
            with self._lock:
                # A turn or agent span ends after the tools it ran; its unclaimed calls are done.
                self._pending_calls.pop(span.span_id, None)
                open_span = self._spans.pop(span.span_id, None)
                if open_span is None:
                    return
                data = span.span_data
                calls = _tool_calls_of(data)
                if calls and span.parent_id:
                    self._pending_calls[span.parent_id] = calls
                tool_call_id = self._claim_call(span)

                fields = _end_of(data)
                fields.merge(_sandbox_span_end(data))
                fields.merge(_command_tool_end(data))
                if tool_call_id:
                    fields.values["tool_call_id"] = tool_call_id
                end_time = _time(span.ended_at)
                open_span.handle.end(
                    **fields.values,
                    attributes=fields.attributes or None,
                    error=_error_of(span),
                    end_time=end_time,
                )

                state = open_span.trace
                if state is None:
                    return
                end = end_time or datetime.now(timezone.utc)
                if state.last_end is None or end > state.last_end:
                    state.last_end = end
                if self._traces.get(span.trace_id) is state:
                    self._touch(span.trace_id, state)
        except Exception as exc:
            _report(exc)

    def force_flush(self) -> None:
        try:
            with self._lock:
                self._end_idle_roots()
        except Exception as exc:
            _report(exc)
        telemetry_dev.flush()

    def shutdown(self, timeout: float | None = None) -> None:
        """Ends every open span and flushes, within ``timeout`` seconds when the Agents SDK gives
        one (it passes the rest of its shutdown budget)."""
        # Roots stay known: spans of a server that keeps draining requests still join their trace.
        try:
            with self._lock:
                self._end_idle_roots()
                open_spans = list(self._spans.values())
                self._spans.clear()
                self._pending_calls.clear()
                for open_span in open_spans:
                    open_span.handle.end()
                for state in list(self._traces.values()):
                    self._end_root(state)
        except Exception as exc:
            _report(exc)
        if timeout is None:
            telemetry_dev.flush()
        else:
            telemetry_dev.flush(timeout_s=timeout)
