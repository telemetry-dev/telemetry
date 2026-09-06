from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping, Sequence
from contextvars import Token
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict
from weakref import WeakKeyDictionary

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.trace import Span, SpanContext, Status, StatusCode
from opentelemetry.util import types as otel_types

from ._client import Client, get_client
from ._config import logger
from ._context import (
    context_from_parent,
    context_with_propagated_attributes,
    propagated_attributes,
    traceparent_of,
    with_session_parent,
)
from ._semconv import (
    ATTR_AGENT_ID,
    ATTR_AGENT_NAME,
    ATTR_COST,
    ATTR_ERROR_TYPE,
    ATTR_FINISH_REASONS,
    ATTR_OPERATION,
    ATTR_OUTPUT_TYPE,
    ATTR_PROVIDER,
    ATTR_REQUEST_MODEL,
    ATTR_RESPONSE_ID,
    ATTR_RESPONSE_MODEL,
    ATTR_SESSION_ID,
    ATTR_SYSTEM_INSTRUCTIONS,
    ATTR_TIME_TO_FIRST_CHUNK,
    ATTR_TOOL_CALL_ID,
    ATTR_TOOL_DESCRIPTION,
    ATTR_TOOL_NAME,
    METADATA_PREFIX,
    RESERVED_METADATA_KEYS,
    SAMPLING_ATTRS,
    SEVERITY,
    SPAN_TYPE_TO_OPERATION,
    USAGE_ATTRS,
    SpanType,
    input_key,
    output_key,
)
from ._serialize import AttributeValue, coerce_attr_value


class _NotGiven:
    def __repr__(self) -> str:
        return "NOT_GIVEN"

    def __bool__(self) -> bool:
        return False


NOT_GIVEN: Any = _NotGiven()


class Usage(TypedDict, total=False):
    """Token usage — exactly these six fields are emitted; unknown keys are dropped."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    reasoning_output_tokens: int


ParentType = str | Context | SpanContext | None
TimeInput = datetime | int | None


@dataclass
class _SpanState:
    operation: str
    capture_input: bool
    capture_output: bool


_SPAN_STATES: WeakKeyDictionary[Span, _SpanState] = WeakKeyDictionary()


def _to_ns(value: TimeInput) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000_000_000)
    return int(value)


def _record_error_on_span(span: Span, error: BaseException) -> None:
    error_type = type(error).__name__
    stacktrace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    span.set_attribute(ATTR_ERROR_TYPE, error_type)
    span.add_event(
        "exception",
        {
            "exception.type": error_type,
            "exception.message": str(error),
            "exception.stacktrace": stacktrace,
            "log.severity_number": SEVERITY["error"],
        },
    )
    span.set_status(Status(StatusCode.ERROR))


def _metadata_attrs(
    metadata: Mapping[str, Any],
    max_len: int,
    on_error: Callable[[BaseException], None] | None,
) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {}
    for key, value in metadata.items():
        if key in RESERVED_METADATA_KEYS:
            logger.debug(
                "telemetry-dev: dropping reserved metadata key %r "
                "(use propagate_attributes user_id/session_id instead)",
                key,
            )
            continue
        attr = coerce_attr_value(
            value,
            max_len=max_len,
            key=f"{METADATA_PREFIX}{key}",
            on_error=on_error,
        )
        if attr is not None:
            attrs[f"{METADATA_PREFIX}{key}"] = attr
    return attrs


def _usage_attrs(usage: Mapping[str, Any]) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {}
    for key, value in usage.items():
        attr = USAGE_ATTRS.get(key)
        if attr is None:
            logger.debug("telemetry-dev: dropping unknown usage key %r", key)
            continue
        if isinstance(value, int | float) and not isinstance(value, bool):
            attrs[attr] = value
    return attrs


def _fields_to_attributes(
    client: Client, state: _SpanState, fields: dict[str, Any]
) -> dict[str, otel_types.AttributeValue]:
    """Map the shared snake_case field set onto gen_ai.* attributes; raw `attributes` merge last."""
    attrs: dict[str, otel_types.AttributeValue] = {}

    for field, attr in (
        ("model", ATTR_REQUEST_MODEL),
        ("provider", ATTR_PROVIDER),
        ("response_model", ATTR_RESPONSE_MODEL),
        ("response_id", ATTR_RESPONSE_ID),
        ("output_type", ATTR_OUTPUT_TYPE),
        ("tool_name", ATTR_TOOL_NAME),
        ("tool_call_id", ATTR_TOOL_CALL_ID),
        ("tool_description", ATTR_TOOL_DESCRIPTION),
        ("agent_name", ATTR_AGENT_NAME),
        ("agent_id", ATTR_AGENT_ID),
    ):
        value = fields.get(field)
        if value is not None:
            attrs[attr] = value

    finish_reason = fields.get("finish_reason")
    if finish_reason:
        attrs[ATTR_FINISH_REASONS] = [finish_reason]

    usage = fields.get("usage")
    if usage is not None:
        attrs.update(_usage_attrs(usage))

    cost_usd = fields.get("cost_usd")
    if cost_usd is not None:
        attrs[ATTR_COST] = float(cost_usd)

    for field, attr in SAMPLING_ATTRS.items():
        value = fields.get(field)
        if value is not None:
            attrs[attr] = list(value) if field == "stop_sequences" else value

    time_to_first_chunk_ms = fields.get("time_to_first_chunk_ms")
    if time_to_first_chunk_ms is not None:
        attrs[ATTR_TIME_TO_FIRST_CHUNK] = time_to_first_chunk_ms / 1000

    metadata = fields.get("metadata")
    if metadata is not None:
        attrs.update(_metadata_attrs(metadata, client.max_attribute_length, client.on_error))

    if "input" in fields and state.capture_input:
        key = input_key(state.operation)
        serialized = client.serialize(fields["input"], key)
        if serialized is not None:
            attrs[key] = serialized
    system_instructions = fields.get("system_instructions")
    if system_instructions is not None and state.capture_input:
        serialized = client.serialize(system_instructions, ATTR_SYSTEM_INSTRUCTIONS)
        if serialized is not None:
            attrs[ATTR_SYSTEM_INSTRUCTIONS] = serialized
    if "output" in fields and state.capture_output:
        key = output_key(state.operation)
        serialized = client.serialize(fields["output"], key)
        if serialized is not None:
            attrs[key] = serialized

    raw = fields.get("attributes")
    if raw is not None:
        for key, value in raw.items():
            attr = coerce_attr_value(
                value,
                max_len=client.max_attribute_length,
                key=key,
                on_error=client.on_error,
            )
            if attr is not None:
                attrs[key] = attr

    return attrs


def _apply_fields(client: Client, span: Span, state: _SpanState, fields: dict[str, Any]) -> None:
    name = fields.get("name")
    if name is not None:
        span.update_name(name)
    attrs = _fields_to_attributes(client, state, fields)
    if attrs:
        span.set_attributes(attrs)

    error = fields.get("error")
    if error is not None:
        _record_error_on_span(span, error)


def _collect_fields(
    *,
    input: Any = NOT_GIVEN,
    output: Any = NOT_GIVEN,
    name: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    system_instructions: Any = None,
    response_model: str | None = None,
    response_id: str | None = None,
    output_type: str | None = None,
    finish_reason: str | None = None,
    usage: Usage | Mapping[str, int] | None = None,
    cost_usd: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    time_to_first_chunk_ms: float | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_description: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if input is not NOT_GIVEN:
        fields["input"] = input
    if output is not NOT_GIVEN:
        fields["output"] = output
    for key, value in (
        ("name", name),
        ("model", model),
        ("provider", provider),
        ("system_instructions", system_instructions),
        ("response_model", response_model),
        ("response_id", response_id),
        ("output_type", output_type),
        ("finish_reason", finish_reason),
        ("usage", usage),
        ("cost_usd", cost_usd),
        ("temperature", temperature),
        ("top_p", top_p),
        ("top_k", top_k),
        ("max_tokens", max_tokens),
        ("stop_sequences", stop_sequences),
        ("seed", seed),
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
        ("time_to_first_chunk_ms", time_to_first_chunk_ms),
        ("tool_name", tool_name),
        ("tool_call_id", tool_call_id),
        ("tool_description", tool_description),
        ("agent_name", agent_name),
        ("agent_id", agent_id),
        ("metadata", metadata),
        ("attributes", attributes),
        ("error", error),
    ):
        if value is not None:
            fields[key] = value
    return fields


class SpanHandle:
    """Handle around an OTel span. Use as a context manager to activate the span in the
    current context, or keep it detached and call .end() manually."""

    def __init__(
        self,
        span: Span,
        client: Client | None,
        state: _SpanState | None,
        context: Context | None = None,
    ) -> None:
        self.span = span
        self._client = client
        self._state = state
        self._context = context
        self._context_token: Token[Context] | None = None
        self._ended = False

    def _recording(self) -> bool:
        return (
            self._client is not None
            and self._state is not None
            and not self._ended
            and self.span.is_recording()
        )

    def update(
        self,
        *,
        name: str | None = None,
        input: Any = NOT_GIVEN,
        output: Any = NOT_GIVEN,
        model: str | None = None,
        provider: str | None = None,
        system_instructions: Any = None,
        response_model: str | None = None,
        response_id: str | None = None,
        output_type: str | None = None,
        finish_reason: str | None = None,
        usage: Usage | Mapping[str, int] | None = None,
        cost_usd: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        time_to_first_chunk_ms: float | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_description: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        error: BaseException | None = None,
    ) -> SpanHandle:
        if not self._recording():
            return self
        assert self._client is not None and self._state is not None
        try:
            fields = _collect_fields(
                name=name,
                input=input,
                output=output,
                model=model,
                provider=provider,
                system_instructions=system_instructions,
                response_model=response_model,
                response_id=response_id,
                output_type=output_type,
                finish_reason=finish_reason,
                usage=usage,
                cost_usd=cost_usd,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                seed=seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                time_to_first_chunk_ms=time_to_first_chunk_ms,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_description=tool_description,
                agent_name=agent_name,
                agent_id=agent_id,
                metadata=metadata,
                attributes=attributes,
                error=error,
            )
            _apply_fields(self._client, self.span, self._state, fields)
        except BaseException as exc:
            self._client.report("SpanHandle.update failed", exc)
        return self

    def end(
        self,
        *,
        name: str | None = None,
        input: Any = NOT_GIVEN,
        output: Any = NOT_GIVEN,
        model: str | None = None,
        provider: str | None = None,
        system_instructions: Any = None,
        response_model: str | None = None,
        response_id: str | None = None,
        output_type: str | None = None,
        finish_reason: str | None = None,
        usage: Usage | Mapping[str, int] | None = None,
        cost_usd: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        time_to_first_chunk_ms: float | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_description: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        error: BaseException | None = None,
        end_time: TimeInput = None,
    ) -> None:
        if not self._recording():
            return
        self.update(
            name=name,
            input=input,
            output=output,
            model=model,
            provider=provider,
            system_instructions=system_instructions,
            response_model=response_model,
            response_id=response_id,
            output_type=output_type,
            finish_reason=finish_reason,
            usage=usage,
            cost_usd=cost_usd,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            time_to_first_chunk_ms=time_to_first_chunk_ms,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_description=tool_description,
            agent_name=agent_name,
            agent_id=agent_id,
            metadata=metadata,
            attributes=attributes,
            error=error,
        )
        self._ended = True
        self.span.end(_to_ns(end_time))

    def traceparent(self) -> str | None:
        return traceparent_of(self.span.get_span_context())

    def __enter__(self) -> SpanHandle:
        if self._client is not None and self._state is not None:
            self._context_token = otel_context.attach(
                trace.set_span_in_context(self.span, self._context)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if self._context_token is not None:
            otel_context.detach(self._context_token)
            self._context_token = None
        if not self._ended and self._recording():
            if isinstance(exc, BaseException):
                self.end(error=exc)
            else:
                self.end()
        return False


def _noop_handle() -> SpanHandle:
    return SpanHandle(trace.INVALID_SPAN, None, None)


def start_span(
    name: str,
    *,
    type: SpanType = "span",
    input: Any = NOT_GIVEN,
    output: Any = NOT_GIVEN,
    model: str | None = None,
    provider: str | None = None,
    system_instructions: Any = None,
    response_model: str | None = None,
    response_id: str | None = None,
    output_type: str | None = None,
    finish_reason: str | None = None,
    usage: Usage | Mapping[str, int] | None = None,
    cost_usd: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    time_to_first_chunk_ms: float | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_description: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
    parent: ParentType = None,
    start_time: TimeInput = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> SpanHandle:
    """Start a telemetry.dev span. Entering the returned handle (``with``) activates it in the
    current context; without ``with`` it is a detached handle that must be ended via .end()."""
    client = get_client()
    if client is None or not client.enabled or client.tracer is None:
        return _noop_handle()
    try:
        operation = SPAN_TYPE_TO_OPERATION.get(type)
        if operation is None:
            logger.debug("telemetry-dev: unknown span type %r; using 'span'", type)
            operation = SPAN_TYPE_TO_OPERATION["span"]
        state = _SpanState(
            operation=operation,
            capture_input=client.capture_input if capture_input is None else capture_input,
            capture_output=client.capture_output if capture_output is None else capture_output,
        )
        fields = _collect_fields(
            input=input,
            output=output,
            model=model,
            provider=provider,
            system_instructions=system_instructions,
            response_model=response_model,
            response_id=response_id,
            output_type=output_type,
            finish_reason=finish_reason,
            usage=usage,
            cost_usd=cost_usd,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            time_to_first_chunk_ms=time_to_first_chunk_ms,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_description=tool_description,
            agent_name=agent_name,
            agent_id=agent_id,
            metadata=metadata,
            attributes=attributes,
        )
        attrs = _fields_to_attributes(client, state, fields)
        initial: dict[str, otel_types.AttributeValue] = {ATTR_OPERATION: operation}
        if operation == "execute_tool":
            initial[ATTR_TOOL_NAME] = name
        elif operation == "invoke_agent":
            initial[ATTR_AGENT_NAME] = name
        ctx = context_from_parent(parent)
        propagated = propagated_attributes(ctx)
        explicit_session_id = (attributes or {}).get(ATTR_SESSION_ID)
        session_id = (
            explicit_session_id
            if isinstance(explicit_session_id, str)
            else propagated.get(ATTR_SESSION_ID)
        )
        if not isinstance(session_id, str):
            session_id = client._process_session_id  # pyright: ignore[reportPrivateUsage]
        if isinstance(session_id, str):
            propagated[ATTR_SESSION_ID] = session_id
            ctx = context_with_propagated_attributes(propagated, ctx)
        ctx = with_session_parent(
            ctx, session_id if isinstance(session_id, str) else None, client.config.api_key
        )
        initial.update(propagated)
        initial.update(attrs)
        span = client.tracer.start_span(
            name,
            context=ctx,
            attributes=initial,
            start_time=_to_ns(start_time),
        )
        _SPAN_STATES[span] = state
        handle = SpanHandle(span, client, state, ctx)
        # The processor stamps propagation on start; explicit fields still win.
        if attrs:
            span.set_attributes(attrs)
        return handle
    except BaseException as exc:
        client.report("start_span failed", exc)
        return _noop_handle()


def update_current_span(
    *,
    name: str | None = None,
    input: Any = NOT_GIVEN,
    output: Any = NOT_GIVEN,
    model: str | None = None,
    provider: str | None = None,
    system_instructions: Any = None,
    response_model: str | None = None,
    response_id: str | None = None,
    output_type: str | None = None,
    finish_reason: str | None = None,
    usage: Usage | Mapping[str, int] | None = None,
    cost_usd: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: Sequence[str] | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    time_to_first_chunk_ms: float | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_description: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
    error: BaseException | None = None,
) -> None:
    """Apply the update field set to the currently active span; no-ops when there is none."""
    client = get_client()
    if client is None or not client.enabled:
        return
    span = trace.get_current_span()
    if not span.is_recording():
        return
    state = _SPAN_STATES.get(span)
    if state is None:
        operation = "function"
        if isinstance(span, SdkSpan):
            current = (span.attributes or {}).get(ATTR_OPERATION)
            if isinstance(current, str):
                operation = current
        state = _SpanState(
            operation=operation,
            capture_input=client.capture_input,
            capture_output=client.capture_output,
        )
    try:
        fields = _collect_fields(
            name=name,
            input=input,
            output=output,
            model=model,
            provider=provider,
            system_instructions=system_instructions,
            response_model=response_model,
            response_id=response_id,
            output_type=output_type,
            finish_reason=finish_reason,
            usage=usage,
            cost_usd=cost_usd,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            time_to_first_chunk_ms=time_to_first_chunk_ms,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_description=tool_description,
            agent_name=agent_name,
            agent_id=agent_id,
            metadata=metadata,
            attributes=attributes,
            error=error,
        )
        _apply_fields(client, span, state, fields)
    except BaseException as exc:
        client.report("update_current_span failed", exc)
