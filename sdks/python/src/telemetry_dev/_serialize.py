from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import rfc8785

from ._config import report_error

TRUNCATION_MARKER = "...[truncated]"
# Deprecated: the old default of max_attribute_length. Values are now sent whole by default.
DEFAULT_MAX_ATTRIBUTE_LENGTH = 65536
_MAX_SAFE_INTEGER = 2**53 - 1
_MAX_ARRAY_INDEX = 2**32 - 2
_SURROGATES = re.compile("[\ud800-\udfff]")

AttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True)
class MaskContext:
    key: str


Mask = Callable[[Any, MaskContext], Any]


def utf16_length(text: str) -> int:
    """The length of ``text`` in UTF-16 units, as JavaScript and the telemetry.dev API count it."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def truncate(text: str, max_len: int, *, keep_end: bool = False) -> str:
    """Cap ``text`` at ``max_len`` UTF-16 units, marking the cut.

    ``keep_end`` keeps the tail and puts the marker first, for text whose most useful part
    comes last.
    """
    # Count UTF-16 units like JavaScript, but back off at a split surrogate pair
    # rather than emitting the lone surrogate that String.slice would retain.
    if len(text) * 2 <= max_len:
        return text
    encoded = text.encode("utf-16-le")
    if len(encoded) <= max_len * 2:
        return text
    if max_len < len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[: max(max_len, 0)]
    kept = (max_len - len(TRUNCATION_MARKER)) * 2
    if keep_end:
        tail = encoded[len(encoded) - kept :]
        try:
            return TRUNCATION_MARKER + tail.decode("utf-16-le")
        except UnicodeDecodeError:
            return TRUNCATION_MARKER + tail[2:].decode("utf-16-le")
    head = encoded[:kept]
    try:
        return head.decode("utf-16-le") + TRUNCATION_MARKER
    except UnicodeDecodeError:
        return head[:-2].decode("utf-16-le") + TRUNCATION_MARKER


def _json_number(value: int | float) -> str:
    if isinstance(value, int) and -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
        # int subclasses such as IntEnum on Python 3.10 may override __str__ with non-JSON text.
        return int.__repr__(value)
    try:
        number = float(value)
    except OverflowError:
        return "null"
    return rfc8785.dumps(number).decode("ascii") if math.isfinite(number) else "null"


def _json_string(value: str) -> str:
    if _SURROGATES.search(value):
        value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
        return _SURROGATES.sub(
            lambda match: f"\\u{ord(match[0]):04x}", json.dumps(value, ensure_ascii=False)
        )
    return json.dumps(value, ensure_ascii=False)


def _json_key_order(key: str) -> tuple[int, int]:
    if key.isascii() and key.isdigit() and len(key) <= 10:
        index = int(key)
        if index <= _MAX_ARRAY_INDEX and str(index) == key:
            return 0, index
    return 1, 0


def json_dumps(value: Any, *, default: Callable[[Any], Any] | None = repr) -> str:
    """JSON.stringify spelling and key order for JSON values; default handles Python-only values."""
    active: set[int] = set()
    chunks: list[str] = []

    def encode(item: Any) -> None:
        if item is None:
            chunks.append("null")
            return
        if isinstance(item, bool):
            chunks.append("true" if item else "false")
            return
        if isinstance(item, str):
            chunks.append(_json_string(item))
            return
        if isinstance(item, int | float):
            chunks.append(_json_number(item))
            return
        identity = id(item)
        if identity in active:
            raise ValueError("Circular reference detected")
        active.add(identity)
        try:
            if isinstance(item, dict):
                properties: dict[str, Any] = {}
                for key, child in cast("dict[Any, Any]", item).items():
                    if not isinstance(key, str):
                        if key is None:
                            key = "null"
                        elif isinstance(key, bool):
                            key = "true" if key else "false"
                        elif isinstance(key, float) and not math.isfinite(key):
                            key = json.dumps(key)
                        elif isinstance(key, int | float):
                            key = _json_number(key)
                        else:
                            raise TypeError("JSON object keys must be strings or scalars")
                    if _SURROGATES.search(key):
                        key = key.encode("utf-16-le", "surrogatepass").decode(
                            "utf-16-le", "surrogatepass"
                        )
                    properties[key] = child
                chunks.append("{")
                for index, key in enumerate(sorted(properties, key=_json_key_order)):
                    if index:
                        chunks.append(",")
                    chunks.extend((_json_string(key), ":"))
                    encode(properties[key])
                chunks.append("}")
            elif isinstance(item, list | tuple):
                chunks.append("[")
                for index, child in enumerate(cast("list[Any] | tuple[Any, ...]", item)):
                    if index:
                        chunks.append(",")
                    encode(child)
                chunks.append("]")
            elif default is not None:
                encode(default(item))
            else:
                raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")
        finally:
            active.remove(identity)

    encode(value)
    return "".join(chunks)


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_dumps(value)


def serialize_content(
    value: Any,
    *,
    key: str,
    mask: Mask | None,
    max_len: int,
    on_error: Callable[[BaseException], None] | None = None,
    keep_end: bool = False,
) -> str | None:
    """The single content funnel: mask -> JSON stringify -> truncate.

    Returns None (content dropped, matching the TypeScript SDK) when the mask hook raises
    or the value cannot be stringified. ``keep_end`` is passed to ``truncate``.
    """
    if mask is not None:
        try:
            value = mask(value, MaskContext(key=key))
        except BaseException:
            # Raised hook errors can retain unmasked values and the application exception chain.
            message = "mask hook raised; content dropped"
            report_error(on_error, message, RuntimeError(message))
            return None
    try:
        return truncate(stringify(value), max_len, keep_end=keep_end)
    except BaseException:
        message = "failed to serialize attribute; content dropped"
        report_error(on_error, message, ValueError(message))
        return None


def fit_object_values(text: str, max_len: int) -> str:
    """Cap the JSON of an object at ``max_len`` UTF-16 units and keep it an object with every key.

    The values share the room: a short value stays whole, and each long value gets the same share
    and becomes a text that ends with the marker. A text whose escapes pass its share becomes the
    marker alone. A value that is not text becomes the cut text of its JSON. JSON that is no
    object, whose keys alone pass the cap, or whose values get a share under the marker, is cut
    as a whole. The TypeScript SDK cuts the same way.
    """
    if utf16_length(text) <= max_len:
        return text
    try:
        parsed = json.loads(text)
    except ValueError:
        return _cut_text(text, max_len)
    if not isinstance(parsed, dict):
        return _cut_text(text, max_len)
    values = cast("dict[str, Any]", parsed)
    skeleton = utf16_length(json_dumps(dict.fromkeys(values, "")))

    def extra(value: Any) -> int:
        return utf16_length(json_dumps(value)) - 2

    sizes = {key: extra(value) for key, value in values.items()}
    room = max_len - skeleton
    if room < 0:
        return _cut_text(text, max_len)
    fitted = dict(values)
    by_size = sorted(values, key=sizes.__getitem__)
    marker = len(TRUNCATION_MARKER)
    for index, key in enumerate(by_size):
        share = room // (len(by_size) - index)
        if sizes[key] <= share:
            room -= sizes[key]
            continue
        # No marker fits the share of this value, so the JSON is cut as a whole.
        if share < marker:
            return _cut_text(text, max_len)
        value = values[key]
        whole = value if isinstance(value, str) else json_dumps(value)
        cut = _cut_text(whole, share)
        keep = share
        # Escapes make the JSON of a text longer than the text.
        while extra(cut) > share and keep > marker:
            keep = max(keep - (extra(cut) - share), marker)
            cut = _cut_text(whole, keep)
        # A share of the marker's length leaves no room to cut an escaped text further.
        if extra(cut) > share:
            cut = TRUNCATION_MARKER
        fitted[key] = cut
        room -= extra(cut)
    result = json_dumps(fitted)
    return result if utf16_length(result) <= max_len else _cut_text(text, max_len)


def _cut_text(text: str, max_len: int) -> str:
    """``truncate`` for a text that can hold lone surrogates, as ``json.loads`` gives them back.

    Each surrogate counts as one UTF-16 unit, and a high surrogate right before the marker goes,
    as the TypeScript SDK cuts. The JSON of the result escapes any lone surrogate that stays.
    """
    encoded = text.encode("utf-16-le", "surrogatepass")
    if len(encoded) <= max_len * 2:
        return text
    if max_len < len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[: max(max_len, 0)]
    head = encoded[: (max_len - len(TRUNCATION_MARKER)) * 2].decode("utf-16-le", "surrogatepass")
    if head and "\ud800" <= head[-1] <= "\udbff":
        head = head[:-1]
    return head + TRUNCATION_MARKER


def coerce_attr_value(
    value: Any,
    *,
    max_len: int = sys.maxsize,
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
    except BaseException:
        message = "failed to serialize attribute; content dropped"
        report_error(on_error, message, ValueError(message))
        return None
