"""Bounded retention for provider streaming instrumentation."""

from __future__ import annotations

import os
import re
import sys
import threading
import weakref
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from itertools import chain
from typing import Final, cast

from ._export import MAX_RECORD_BYTES

DEFAULT_MAX_STREAM_BYTES: Final = 64 * 1024 * 1024
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


class _StreamBudget:
    """Bounds the bytes that provider integrations keep, across the process, for streams that
    have not ended, since each stream may keep up to MAX_RECORD_BYTES. While no other stream
    holds bytes, a stream may grow to MAX_RECORD_BYTES even past max_bytes, so that one large
    stream is not cut by the shared bound. Streams run on many threads, so the total holds a
    lock.

    A stream holds its bytes in a ``[bytes, generation, given_back]`` list. Garbage collection
    can run at any allocation, also while this thread holds the lock, and it runs finalizers and
    the ``finally`` blocks of dropped stream generators, which end the stream. So no path that
    ends a stream waits for the lock on the thread that holds it: limit() reads without waiting,
    give_back() and a move made under the lock are queued, and the next call that takes the
    lock applies them, and a reserve() made under the lock is refused. The collector calls the
    finalizer of a dropped stream's budget before the ``finally`` block that ends the stream, so
    once a list is given back, later moves of it count as nothing. A forked child starts a new
    generation, and lists of the parent's streams count as holding nothing in it.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # Set on a thread from before it takes the lock until after it lets go of it.
        self._local = threading.local()
        self._held = 0
        self._generation = 0
        self._queued: deque[tuple[list[int], int]] = deque()
        if hasattr(os, "register_at_fork"):
            # Streams of the parent do not run in a forked child.
            os.register_at_fork(after_in_child=self._reset)

    def _reset(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._held = 0
        self._generation += 1
        self._queued.clear()

    def new_held(self) -> list[int]:
        return [0, self._generation, 0]

    def give_back(self, held: list[int]) -> None:
        """Gives the bytes of ``held`` back at the next call that takes the lock. Safe to call
        from a finalizer."""
        held[2] = 1
        self._queued.append((held, 0))

    def _move(self, held: list[int], to: int) -> None:
        if held[2]:
            to = 0
        if held[1] != self._generation:
            held[0] = 0
            held[1] = self._generation
        self._held = max(self._held + to - held[0], 0)
        held[0] = to

    def _locked_by_this_thread(self) -> bool:
        return getattr(self._local, "locking", False)

    def _apply_queued(self) -> None:
        while self._queued:
            self._move(*self._queued.popleft())

    @contextmanager
    def _locked(self) -> Generator[None]:
        self._local.locking = True
        try:
            with self._lock:
                self._apply_queued()
                yield
        finally:
            self._local.locking = False

    def _limit(self, held: list[int]) -> int:
        own = held[0] if held[1] == self._generation else 0
        others = self._held - own
        room = self.max_bytes - others if others > 0 else MAX_RECORD_BYTES
        return max(own, min(MAX_RECORD_BYTES, room))

    def limit(self, held: list[int]) -> int:
        """The total the stream that holds ``held`` may hold now. It applies the queued moves
        only when the lock is free, so while another thread moves a stream the result can be
        that move behind."""
        if not self._locked_by_this_thread():
            self._local.locking = True
            try:
                if self._lock.acquire(blocking=False):
                    try:
                        self._apply_queued()
                    finally:
                        self._lock.release()
            finally:
                self._local.locking = False
        return self._limit(held)

    def reserve(self, held: list[int], to: int) -> bool:
        """Moves a stream to ``to`` bytes if its limit allows it, checked under the lock, so that
        streams measuring on other threads cannot pass the bound together."""
        if self._locked_by_this_thread():
            return False
        with self._locked():
            if to > self._limit(held):
                return False
            self._move(held, to)
            return True

    def move(self, held: list[int], to: int) -> None:
        if self._locked_by_this_thread():
            self._queued.append((held, to))
            return
        with self._locked():
            self._move(held, to)


# Sized by max_stream_bytes when a span processor starts.
stream_budget = _StreamBudget(DEFAULT_MAX_STREAM_BYTES)


class CaptureBudget:
    """Track a byte and item budget before retaining provider data.

    ``max_bytes`` is capped at MAX_RECORD_BYTES (16 MiB), and ``max_items`` has no limit by
    default. A budget from for_stream() also holds its bytes in the process-wide stream budget,
    so its ``max_bytes`` shrinks while other streams hold bytes, until release().
    """

    def __init__(self, max_bytes: int = MAX_RECORD_BYTES, max_items: int = sys.maxsize):
        self._max_bytes: int = max(0, min(max_bytes, MAX_RECORD_BYTES))
        self.max_items: int = max(0, max_items)
        self._bytes_used: int = 0
        self.items_used: int = 0
        self.truncated: bool = False
        self._held: list[int] | None = None
        self._release: Callable[[], object] | None = None
        self._released_limit: int | None = None

    @classmethod
    def for_stream(cls) -> CaptureBudget:
        """A budget for the output one stream keeps, held in the process-wide stream budget.

        Call release() when the stream ends; a budget dropped without it releases its bytes
        when it is garbage collected.
        """
        budget = cls()
        budget._held = stream_budget.new_held()
        budget._release = weakref.finalize(budget, stream_budget.give_back, budget._held)
        return budget

    @classmethod
    def from_client(cls) -> CaptureBudget:
        """Use the active client's attribute limit without exceeding the SDK ceiling."""
        from ._client import get_client

        client = get_client()
        return cls(
            max_bytes=client.max_attribute_length if client is not None else MAX_RECORD_BYTES
        )

    @property
    def max_bytes(self) -> int:
        if self._released_limit is not None:
            return self._released_limit
        if self._held is None:
            return self._max_bytes
        return min(self._max_bytes, stream_budget.limit(self._held))

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        self._max_bytes = max(0, min(value, MAX_RECORD_BYTES))

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    @bytes_used.setter
    def bytes_used(self, value: int) -> None:
        """Sets the bytes without a check; use reserve_bytes() to grow a stream budget."""
        self._bytes_used = value
        if self._held is not None and self._released_limit is None:
            stream_budget.move(self._held, value)

    def reserve_bytes(self, total: int) -> bool:
        """Sets ``bytes_used`` to ``total`` if ``max_bytes`` allows it, and returns whether it
        did. For a stream budget, the shared bound is checked under the same lock as the move,
        so streams that measured on other threads at the same time cannot pass it together.
        Does not mark the budget truncated."""
        if total <= self._bytes_used:
            self.bytes_used = total
            return True
        if total > self._max_bytes:
            return False
        if self._held is not None and self._released_limit is None:
            if not stream_budget.reserve(self._held, total):
                return False
        elif total > self.max_bytes:
            return False
        self._bytes_used = total
        return True

    def release(self) -> None:
        """Returns the bytes of a stream budget. After it, ``max_bytes`` stays at the bytes held at
        release, so a stream that is still read after it ended keeps no more outside the bound."""
        if self._release is None or self._released_limit is not None:
            return
        self._released_limit = min(self.max_bytes, self._bytes_used)
        self._release()

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
        if not self.reserve_bytes(self._bytes_used + byte_count):
            self.truncated = True
            return False
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
        if not self.reserve_bytes(self._bytes_used + retained_length):
            self.truncated = True
            return b""
        self.items_used += 1
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
