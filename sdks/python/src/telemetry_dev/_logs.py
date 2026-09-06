from __future__ import annotations

from collections.abc import Mapping

from opentelemetry import context as otel_context
from opentelemetry._logs import SeverityNumber

from ._client import get_client
from ._config import logger
from ._context import context_with_propagated_attributes, propagated_attributes
from ._semconv import ATTR_SESSION_ID, SEVERITY, LogLevel
from ._serialize import AttributeValue, coerce_attr_value


def log(
    message: str,
    *,
    level: LogLevel = "info",
    event_name: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> None:
    """Emit an OTLP log record to /v1/logs, correlated with the current trace context.
    Levels: debug | info | warn | error ("warning" is accepted as an alias of "warn")."""
    client = get_client()
    if client is None or not client.enabled or client.otel_logger is None:
        return
    try:
        normalized = "warn" if level == "warning" else level
        severity = SEVERITY.get(normalized)
        if severity is None:
            logger.debug("telemetry-dev: unknown log level %r; using 'info'", level)
            normalized, severity = "info", SEVERITY["info"]
        attrs: dict[str, AttributeValue] = {}
        if attributes is not None:
            for key, value in attributes.items():
                attr = coerce_attr_value(
                    value,
                    max_len=client.max_attribute_length,
                    key=key,
                    on_error=client.on_error,
                )
                if attr is not None:
                    attrs[key] = attr
        propagated = propagated_attributes()
        process_session_id = client._process_session_id  # pyright: ignore[reportPrivateUsage]
        explicit_session = attributes is not None and ATTR_SESSION_ID in attributes
        if (
            not explicit_session
            and ATTR_SESSION_ID not in propagated
            and process_session_id is not None
        ):
            propagated[ATTR_SESSION_ID] = process_session_id
        context = otel_context.get_current()
        if propagated:
            context = context_with_propagated_attributes(propagated, context)
        # Propagated correlation attrs win on key collisions, matching the span-processor
        # stamping model and the TypeScript SDK.
        for key, value in propagated.items():
            attr = coerce_attr_value(
                value,
                max_len=client.max_attribute_length,
                key=key,
                on_error=client.on_error,
            )
            if attr is not None:
                attrs[key] = attr
        body = client.serialize(message, "log.message") or ""
        client.otel_logger.emit(
            context=context,
            severity_number=SeverityNumber(severity),
            severity_text=normalized.upper(),
            body=body,
            attributes=attrs or None,
            event_name=event_name,
        )
    except BaseException as exc:
        client.report("log() failed", exc)
