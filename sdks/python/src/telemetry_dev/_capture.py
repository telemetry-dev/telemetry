"""Bounded retention for provider streaming instrumentation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Final, cast

_DEFAULT_MAX_BYTES: Final = 64 * 1024
_DEFAULT_MAX_ITEMS: Final = 1024
_MAX_DEPTH: Final = 32
_BASE_ITEM_BYTES: Final = 16

# Code points whose JSON form is longer than their UTF-8 form, mapped to the extra bytes.
_SHORT_ESCAPE_EXTRA: Final = 1  # '"', '\\', \b \t \n \f \r: 1 byte raw, 2 escaped
_CONTROL_ESCAPE_EXTRA: Final = 5  # other code points below 0x20: 1 byte raw, 6 as \u00XX
_SURROGATE_ESCAPE_EXTRA: Final = 3  # a surrogate code point: 3 bytes raw, 6 as \uXXXX
_JSON_ESCAPED: Final = re.compile('["\\\\\x00-\x1f\ud800-\udfff]')
_JSON_STRING_CHUNK: Final = 4096


def _escape_extra(character: str) -> int:
    code = ord(character)
    if character in '"\\\b\t\n\f\r':
        return _SHORT_ESCAPE_EXTRA
    if code < 0x20:
        return _CONTROL_ESCAPE_EXTRA
    return _SURROGATE_ESCAPE_EXTRA


def _json_string_bytes(value: str, limit: int | None) -> int:
    # Every code point costs at least one byte, so a string longer than the limit needs no scan.
    if limit is not None and len(value) > limit:
        return len(value)
    size = 0
    for start in range(0, len(value), _JSON_STRING_CHUNK):
        chunk = value[start : start + _JSON_STRING_CHUNK]
        size += len(chunk.encode("utf-8", "surrogatepass"))
        size += sum(_escape_extra(match.group()) for match in _JSON_ESCAPED.finditer(chunk))
        if limit is not None and size > limit:
            return size
    return size


class CaptureBudget:
    """Track a hard byte and item budget before retaining streamed provider data."""

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES, max_items: int = _DEFAULT_MAX_ITEMS):
        self.max_bytes: int = max(0, min(max_bytes, _DEFAULT_MAX_BYTES))
        self.max_items: int = max(0, min(max_items, _DEFAULT_MAX_ITEMS))
        self.bytes_used: int = 0
        self.items_used: int = 0
        self.truncated: bool = False

    @classmethod
    def from_client(cls) -> CaptureBudget:
        """Use the active client's attribute limit without exceeding the SDK ceiling."""
        from ._client import get_client

        client = get_client()
        return cls(
            max_bytes=client.max_attribute_length if client is not None else _DEFAULT_MAX_BYTES
        )

    @staticmethod
    def json_string_bytes(value: str, limit: int | None = None) -> int:
        """Return the UTF-8 bytes ``value`` occupies inside a JSON string, without the quotes.

        Matches the TypeScript SDK's measure: ``"``, ``\\``, and ``\\b \\t \\n \\f \\r`` cost 2
        bytes, other control characters cost 6 (``\\u00XX``), and every surrogate code point
        costs 6 (``\\uXXXX``); anything else costs its UTF-8 width. The budget charges a string
        this plus its per-item cost, which covers the quotes and separators. Counting stops
        once the total passes ``limit``, so a result above ``limit`` is only a lower bound.
        """
        return _json_string_bytes(value, limit)

    @property
    def remaining_bytes(self) -> int:
        """Return bytes still available for retained capture state."""
        return self.max_bytes - self.bytes_used

    def accept(self, value: object) -> bool:
        """Reserve space for a value, returning false without retaining it when over budget."""
        if self.truncated:
            return False
        measured = self._measure(
            value,
            remaining_bytes=self.remaining_bytes,
            remaining_items=self.max_items - self.items_used,
            depth=0,
            seen=set(),
        )
        if measured is None:
            self.truncated = True
            return False
        byte_count, item_count = measured
        self.bytes_used += byte_count
        self.items_used += item_count
        return True

    def capture_bytes(self, value: bytes | bytearray | memoryview) -> bytes:
        """Retain at most the remaining byte prefix while marking a partial capture."""
        if self.truncated:
            return b""
        view = memoryview(value)
        try:
            byte_view = view.cast("B")
        except TypeError:
            byte_view = memoryview(bytes(view))
        byte_length = len(byte_view)
        if byte_length == 0:
            return b""
        if self.items_used >= self.max_items or self.remaining_bytes <= 0:
            self.truncated = True
            return b""
        retained_length = min(byte_length, self.remaining_bytes)
        self.items_used += 1
        self.bytes_used += retained_length
        if retained_length < byte_length:
            self.truncated = True
        if isinstance(value, bytes) and retained_length == byte_length:
            return value
        return bytes(byte_view[:retained_length])

    @classmethod
    def _measure(
        cls,
        value: object,
        *,
        remaining_bytes: int,
        remaining_items: int,
        depth: int,
        seen: set[int],
    ) -> tuple[int, int] | None:
        if remaining_items <= 0 or remaining_bytes < _BASE_ITEM_BYTES or depth > _MAX_DEPTH:
            return None

        byte_count = _BASE_ITEM_BYTES
        item_count = 1
        if value is None or isinstance(value, bool | int | float):
            return byte_count, item_count
        if isinstance(value, str):
            # Mapping keys come through here too, so they are charged at their escaped size.
            byte_count += _json_string_bytes(value, remaining_bytes - byte_count)
            return (byte_count, item_count) if byte_count <= remaining_bytes else None
        if isinstance(value, bytes | bytearray | memoryview):
            length = value.nbytes if isinstance(value, memoryview) else len(value)
            # Structured attributes serialize binary values as base64 text.
            byte_count += 4 * ((length + 2) // 3)
            return (byte_count, item_count) if byte_count <= remaining_bytes else None

        value_id = id(value)
        if value_id in seen:
            return byte_count, item_count
        seen.add(value_id)
        try:
            mapping: Mapping[object, object] | None = None
            children: Iterable[object] = ()
            if isinstance(value, Mapping):
                mapping = cast("Mapping[object, object]", value)
            elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
                children = cast("Iterable[object]", value)
            elif isinstance(value, set | frozenset):
                children = cast("Iterable[object]", value)
            else:
                attributes = getattr(value, "__dict__", None)
                if isinstance(attributes, Mapping):
                    mapping = cast("Mapping[object, object]", attributes)

            if mapping is not None:
                children = chain.from_iterable(mapping.items())

            for child in children:
                measured = cls._measure(
                    child,
                    remaining_bytes=remaining_bytes - byte_count,
                    remaining_items=remaining_items - item_count,
                    depth=depth + 1,
                    seen=seen,
                )
                if measured is None:
                    return None
                child_bytes, child_items = measured
                byte_count += child_bytes
                item_count += child_items
            return byte_count, item_count
        finally:
            seen.remove(value_id)
