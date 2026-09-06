from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ._config import report_error

TRUNCATION_MARKER = "...[truncated]"
DEFAULT_MAX_ATTRIBUTE_LENGTH = 65536

AttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True)
class MaskContext:
    key: str


Mask = Callable[[Any, MaskContext], Any]


def truncate(text: str, max_len: int) -> str:
    # The cap counts UTF-16 code units (JavaScript's String.length) so both SDKs truncate
    # identical payloads at the same point; a slice landing mid-surrogate-pair backs off one unit.
    if len(text) * 2 <= max_len:
        return text
    encoded = text.encode("utf-16-le")
    if len(encoded) <= max_len * 2:
        return text
    # Total stays within max_len so the provider's attribute_value_length_limit backstop,
    # set to the same cap, never slices the marker off.
    head = encoded[: max(max_len - len(TRUNCATION_MARKER), 0) * 2]
    try:
        return head.decode("utf-16-le") + TRUNCATION_MARKER
    except UnicodeDecodeError:
        return head[:-2].decode("utf-16-le") + TRUNCATION_MARKER


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=repr, ensure_ascii=False)


def serialize_content(
    value: Any,
    *,
    key: str,
    mask: Mask | None,
    max_len: int,
    on_error: Callable[[BaseException], None] | None = None,
) -> str | None:
    """The single content funnel: mask -> JSON stringify -> truncate.

    Returns None (content dropped, matching the TypeScript SDK) when the mask hook raises
    or the value cannot be stringified.
    """
    if mask is not None:
        try:
            value = mask(value, MaskContext(key=key))
        except BaseException as exc:
            report_error(on_error, f"mask hook raised for attribute '{key}'", exc)
            return None
    try:
        text = stringify(value)
    except BaseException as exc:
        report_error(on_error, f"failed to serialize attribute '{key}'", exc)
        return None
    return truncate(text, max_len)


def coerce_attr_value(
    value: Any,
    *,
    max_len: int = DEFAULT_MAX_ATTRIBUTE_LENGTH,
    key: str | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> AttributeValue | None:
    """Pass scalars and homogeneous scalar sequences through; JSON-stringify everything else."""
    try:
        if isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return truncate(value, max_len)
        if isinstance(value, Sequence) and not isinstance(value, Mapping):
            items: list[Any] = list(cast("Sequence[Any]", value))
            if items and all(isinstance(item, str) for item in items):
                return [truncate(item, max_len) for item in items]
            if items and all(isinstance(item, bool) for item in items):
                return items
            if items and all(
                isinstance(item, int | float) and not isinstance(item, bool) for item in items
            ):
                return items
        return truncate(stringify(value), max_len)
    except BaseException as exc:
        suffix = f" '{key}'" if key is not None else ""
        report_error(on_error, f"failed to serialize attribute{suffix}", exc)
        return None
