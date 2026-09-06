"""telemetry.dev SDK for Python — OpenTelemetry-native GenAI tracing, logs, and metrics.

Quickstart::

    import telemetry_dev

    telemetry_dev.init()  # reads TELEMETRY_DEV_API_KEY

    with telemetry_dev.start_span("chat gpt-4o", type="generation", model="gpt-4o",
                                  provider="openai", input=messages) as span:
        ...
        span.update(output=completion, usage={"input_tokens": 11, "output_tokens": 7})
"""

from ._capture import CaptureBudget
from ._client import Client, flush, get_client, init, shutdown
from ._config import SDK_VERSION as __version__
from ._config import LogLevelOption
from ._context import get_traceparent, propagate_attributes
from ._logs import log
from ._observe import observe
from ._semconv import LogLevel, SpanType
from ._serialize import AttributeValue, Mask, MaskContext
from ._spans import (
    NOT_GIVEN,
    SpanHandle,
    Usage,
    start_span,
    update_current_span,
)
from .otel import TelemetrySpanProcessor

__all__ = [
    "NOT_GIVEN",
    "AttributeValue",
    "CaptureBudget",
    "Client",
    "LogLevel",
    "LogLevelOption",
    "Mask",
    "MaskContext",
    "SpanHandle",
    "SpanType",
    "TelemetrySpanProcessor",
    "Usage",
    "__version__",
    "flush",
    "get_client",
    "get_traceparent",
    "init",
    "log",
    "observe",
    "propagate_attributes",
    "shutdown",
    "start_span",
    "update_current_span",
]
