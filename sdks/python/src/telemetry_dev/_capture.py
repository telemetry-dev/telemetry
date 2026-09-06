"""Bounded retention for provider streaming instrumentation."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Final, cast

_DEFAULT_MAX_BYTES: Final = 64 * 1024
_DEFAULT_MAX_ITEMS: Final = 1024
_MAX_DEPTH: Final = 32
_BASE_ITEM_BYTES: Final = 16


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
            for character in value:
                codepoint = ord(character)
                byte_count += (
                    1
                    if codepoint <= 0x7F
                    else 2
                    if codepoint <= 0x7FF
                    else 3
                    if codepoint <= 0xFFFF
                    else 4
                )
                if byte_count > remaining_bytes:
                    return None
            return byte_count, item_count
        if isinstance(value, bytes | bytearray | memoryview):
            length = value.nbytes if isinstance(value, memoryview) else len(value)
            # Structured attributes serialize binary values as base64 text.
            byte_count += 4 * ((length + 2) // 3)
            return (byte_count, item_count) if byte_count <= remaining_bytes else None

        value_id = id(value)
        if value_id in seen:
            return byte_count, item_count
        seen.add(value_id)
        mapping: dict[object, object] | None = None
        children: Iterable[object] = ()
        if isinstance(value, dict):
            mapping = cast("dict[object, object]", value)
        elif isinstance(value, list | tuple | set | frozenset):
            children = cast("Iterable[object]", value)
        else:
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                mapping = cast("dict[object, object]", attributes)

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
