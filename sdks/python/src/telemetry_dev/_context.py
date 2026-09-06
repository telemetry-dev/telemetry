from __future__ import annotations

import hashlib
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    TraceFlags,
    TraceState,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.types import Attributes

from ._config import logger
from ._semconv import (
    ATTR_SESSION_ID,
    ATTR_USER_ID,
    METADATA_PREFIX,
    RESERVED_METADATA_KEYS,
)
from ._serialize import AttributeValue, coerce_attr_value

_PROPAGATED_KEY = otel_context.create_key("telemetry_dev.propagated")

_PROPAGATOR = TraceContextTextMapPropagator()

ParentInput = str | Context | SpanContext | None


class _SessionParent(NonRecordingSpan):
    def __init__(self, span_context: SpanContext, context: Context) -> None:
        super().__init__(span_context)
        self.parent_context = context


class SessionSampler(Sampler):
    """Pair with with_session_parent to sample roots using the provider's real policy."""

    def __init__(self, inner: Sampler) -> None:
        self.inner = inner

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        parent = trace.get_current_span(parent_context)
        if isinstance(parent, _SessionParent):
            parent_context = trace.set_span_in_context(
                trace.get_current_span(parent.parent_context), parent_context
            )
        return self.inner.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"SessionSampler{{{self.inner.get_description()}}}"


def propagated_attributes(context: Context | None = None) -> dict[str, AttributeValue]:
    value = otel_context.get_value(_PROPAGATED_KEY, context=context)
    if isinstance(value, Mapping):
        return dict(cast("Mapping[str, AttributeValue]", value))
    return {}


def context_with_propagated_attributes(
    attributes: Mapping[str, AttributeValue], context: Context | None = None
) -> Context:
    base = context if context is not None else otel_context.get_current()
    return otel_context.set_value(_PROPAGATED_KEY, attributes, context=base)


@contextmanager
def propagate_attributes(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Generator[None]:
    """Stamp user.id / gen_ai.conversation.id / td.metadata.* on every span and log record
    started inside this context. Nestable; inner values win on key conflicts."""
    attrs = propagated_attributes()
    if user_id is not None:
        attrs[ATTR_USER_ID] = user_id
    if session_id is not None:
        attrs[ATTR_SESSION_ID] = session_id
    if metadata is not None:
        for key, value in metadata.items():
            if key in RESERVED_METADATA_KEYS:
                logger.debug(
                    "telemetry-dev: dropping reserved metadata key %r "
                    "(use user_id/session_id instead)",
                    key,
                )
                continue
            attr = coerce_attr_value(value, key=f"{METADATA_PREFIX}{key}")
            if attr is not None:
                attrs[f"{METADATA_PREFIX}{key}"] = attr
    token = otel_context.attach(otel_context.set_value(_PROPAGATED_KEY, attrs))
    try:
        yield
    finally:
        otel_context.detach(token)


def get_traceparent() -> str | None:
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier.get("traceparent")


def traceparent_of(span_context: SpanContext) -> str | None:
    if not span_context.is_valid:
        return None
    return (
        f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}"
        f"-{span_context.trace_flags:02x}"
    )


def context_from_parent(parent: ParentInput) -> Context | None:
    """Resolve the `parent=` option into an OTel Context (None = current context)."""
    if parent is None:
        return None
    if isinstance(parent, str):
        # Anchor extraction on the current context: an invalid traceparent then falls back to
        # the active span instead of silently starting a new root trace.
        ctx = _PROPAGATOR.extract({"traceparent": parent}, context=otel_context.get_current())
    elif isinstance(parent, SpanContext):
        ctx = trace.set_span_in_context(NonRecordingSpan(parent))
    else:
        ctx = parent
    # Preserve ambient propagated attributes when the parent is overridden explicitly.
    if otel_context.get_value(_PROPAGATED_KEY, context=ctx) is None:
        ambient = propagated_attributes()
        if ambient:
            ctx = otel_context.set_value(_PROPAGATED_KEY, ambient, context=ctx)
    return ctx


def session_span_context(api_key: str | None, session_id: str) -> SpanContext:
    """Deterministic remote parent for a session, byte-identical to the TS `sessionSpanContext`:
    trace id = SHA-256(api_key ‖ 0x00 ‖ session_id)[0:16], parent span id = digest[16:24].
    The flags are not a sampling decision; pair with_session_parent with SessionSampler."""
    text = f"{api_key or ''}\0{session_id}"
    # Match TextEncoder: surrogate pairs combine, lone surrogates become U+FFFD.
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    digest = hashlib.sha256(text.encode()).digest()
    return SpanContext(
        trace_id=int.from_bytes(digest[:16], "big"),
        span_id=int.from_bytes(digest[16:24], "big"),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.DEFAULT),
    )


def with_session_parent(
    context: Context | None, session_id: str | None, api_key: str | None
) -> Context | None:
    """Parent a root under the session; requires SessionSampler on the provider."""
    if not api_key or not session_id:
        return context
    if trace.get_current_span(context).get_span_context().is_valid:
        return context
    base = context if context is not None else otel_context.get_current()
    return trace.set_span_in_context(
        _SessionParent(session_span_context(api_key, session_id), base), base
    )
