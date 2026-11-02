from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
import sys
import time
from array import array
from pathlib import Path
from typing import Any

import pytest

import telemetry_dev._capture
from telemetry_dev import MAX_RECORD_BYTES, CaptureBudget

# A bound for the tests that measure what a budget charges; the default holds a whole record.
_LIMIT = 64 * 1024
# The per-item cost a string pays on top of its JSON bytes.
_ITEM = 16


def test_capture_budget_defaults_to_one_record_with_no_item_limit() -> None:
    fits = "x" * (MAX_RECORD_BYTES - _ITEM)
    budget = CaptureBudget()

    assert budget.max_bytes == MAX_RECORD_BYTES
    assert CaptureBudget(max_bytes=sys.maxsize).max_bytes == MAX_RECORD_BYTES
    assert budget.accept(fits) is True
    assert budget.bytes_used == MAX_RECORD_BYTES
    assert budget.truncated is False

    over = CaptureBudget()
    assert over.accept(fits + "x") is False
    assert over.bytes_used == 0
    assert over.truncated is True

    items = CaptureBudget()
    assert items.max_items == sys.maxsize
    assert items.accept([None] * 5_000) is True
    for _ in range(2_000):
        assert items.accept(None) is True
    assert items.items_used == 7_001
    assert items.truncated is False


def test_capture_budget_explicit_max_items_caps_items() -> None:
    budget = CaptureBudget(max_items=5_000)

    assert budget.max_items == 5_000
    for _ in range(5_000):
        assert budget.accept(None) is True

    assert budget.accept(None) is False
    assert budget.truncated is True
    assert budget.items_used == 5_000


@pytest.fixture
def stream_bound(monkeypatch: pytest.MonkeyPatch) -> int:
    # A stream budget an earlier test dropped gives its bytes back when it is collected.
    gc.collect()
    monkeypatch.setattr(telemetry_dev._capture.stream_budget, "max_bytes", 10_000)
    return 10_000


def _text(byte_count: int) -> str:
    """A string the budget charges exactly ``byte_count`` bytes."""
    return "x" * (byte_count - _ITEM)


def test_stream_budget_alone_may_reach_max_record_bytes(stream_bound: int) -> None:
    budget = CaptureBudget.for_stream()

    assert budget.max_bytes == MAX_RECORD_BYTES
    assert budget.accept(_text(stream_bound + 5_000)) is True
    assert budget.max_bytes == MAX_RECORD_BYTES
    assert budget.truncated is False
    budget.release()


def test_stream_budgets_share_the_room_left_by_other_streams(stream_bound: int) -> None:
    first = CaptureBudget.for_stream()
    assert first.accept(_text(3_000)) is True

    over = CaptureBudget.for_stream()
    assert over.max_bytes == stream_bound - 3_000
    assert over.accept(_text(stream_bound - 3_000 + 1)) is False
    assert over.truncated is True

    second = CaptureBudget.for_stream()
    assert second.accept(_text(stream_bound - 3_000)) is True
    assert second.max_bytes == stream_bound - 3_000
    # The bound is full: the first stream keeps what it holds and gets no more.
    assert first.max_bytes == 3_000
    assert first.accept(None) is False

    full = CaptureBudget.for_stream()
    assert full.max_bytes == 0
    assert full.accept(None) is False

    for budget in (first, over, second, full):
        budget.release()


def test_stream_budget_release_returns_its_bytes_and_freezes_max_bytes(stream_bound: int) -> None:
    first = CaptureBudget.for_stream()
    assert first.accept(_text(2_000)) is True
    second = CaptureBudget.for_stream()
    assert second.accept(_text(7_000)) is True
    assert first.max_bytes == stream_bound - 7_000

    held = first.bytes_used
    first.release()

    assert CaptureBudget.for_stream().max_bytes == stream_bound - 7_000
    # A stream that is still read after it ended keeps no more, as it holds nothing in the bound.
    assert first.accept(_text(500)) is False
    assert first.reserve_bytes(held + 1) is False

    second.release()

    assert CaptureBudget.for_stream().max_bytes == MAX_RECORD_BYTES
    assert first.max_bytes == held
    assert second.max_bytes == second.bytes_used


def test_stream_budget_dropped_without_release_returns_its_bytes(stream_bound: int) -> None:
    dropped = CaptureBudget.for_stream()
    assert dropped.accept(_text(9_000)) is True
    other = CaptureBudget.for_stream()
    assert other.max_bytes == stream_bound - 9_000

    del dropped
    gc.collect()

    assert other.max_bytes == MAX_RECORD_BYTES
    other.release()


def test_stream_budgets_measuring_at_once_cannot_pass_the_bound_together(
    stream_bound: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = CaptureBudget.for_stream()
    second = CaptureBudget.for_stream()
    # Wrapping the measure is the only way to run the other stream between this stream's
    # measure and its reserve.
    measure = CaptureBudget._measure  # pyright: ignore[reportPrivateUsage]
    interleaved: list[bool] = []

    def measure_while_another_thread_accepts(*args: Any, **kwargs: Any) -> Any:
        measured = measure(*args, **kwargs)
        if not interleaved:
            interleaved.append(False)
            # Another stream takes its bytes after this one read its room.
            interleaved[0] = second.accept(_text(MAX_RECORD_BYTES - 100))
        return measured

    monkeypatch.setattr(
        CaptureBudget, "_measure", staticmethod(measure_while_another_thread_accepts)
    )

    # Both read the room of a stream alone, so only one may keep it.
    assert first.accept(_text(MAX_RECORD_BYTES - 100)) is False
    assert interleaved == [True]
    assert first.truncated is True
    assert first.bytes_used == 0
    assert CaptureBudget.for_stream().max_bytes == 0
    second.release()
    first.release()


def test_a_stream_collected_while_the_stream_lock_is_held_does_not_deadlock(
    stream_bound: int,
) -> None:
    # The cyclic collector runs at allocations, also inside the lock that the stream bound holds
    # while it measures. A collected stream budget gives its bytes back from its finalizer, and
    # a dropped provider stream ends from the finally block of its generator, as Stream.__del__
    # does here: it moves its bytes, measures, and releases them.
    script = """
import faulthandler, gc, sys
from telemetry_dev import CaptureBudget
from telemetry_dev._capture import stream_budget
faulthandler.dump_traceback_later(20, exit=True)
stream_budget.max_bytes = 100_000
gc.set_threshold(1, 1, 1)

class Stream:
    def __init__(self):
        self.budget = CaptureBudget.for_stream()
        self.budget.accept("x")
        self.cycle = self

    def __del__(self):
        self.budget.bytes_used += 1
        self.budget.accept("y")
        self.budget.release()

kept = CaptureBudget.for_stream()
for _ in range(2_000):
    dropped = CaptureBudget.for_stream()
    dropped.accept("x")
    dropped.cycle = dropped
    del dropped
    Stream()
    kept.accept("x")
gc.collect()
assert CaptureBudget.for_stream().max_bytes == 100_000 - kept.bytes_used, "bytes not given back"
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, timeout=60, check=False
    )

    assert result.returncode == 0, result.stderr.decode()


def test_a_stream_cannot_take_room_that_another_stream_took_after_it_measured(
    stream_bound: int,
) -> None:
    holder = CaptureBudget.for_stream()
    holder.bytes_used = 1
    first = CaptureBudget.for_stream()
    second = CaptureBudget.for_stream()
    # Both measure the same room, as streams on two threads can at the same time.
    room = first.max_bytes
    assert second.max_bytes == room == stream_bound - 1

    assert second.reserve_bytes(6_000) is True
    assert first.reserve_bytes(6_000) is False
    assert first.bytes_used == 0
    assert CaptureBudget.for_stream().max_bytes == stream_bound - 6_001


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_does_not_count_the_bytes_of_its_parents_streams(
    stream_bound: int,
) -> None:
    parent = CaptureBudget.for_stream()
    assert parent.accept(_text(4_000)) is True
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            child = CaptureBudget.for_stream()
            room = child.max_bytes
            assert child.accept(_text(3_000)) is True
            # A stream of the parent that ends in the child gives back nothing there.
            parent.release()
            other = CaptureBudget.for_stream()
            code = 0 if room == MAX_RECORD_BYTES and other.max_bytes == stream_bound - 3_000 else 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 10
    while (waited := os.waitpid(pid, os.WNOHANG)) == (0, 0):
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("The forked process did not end in 10 s")
        time.sleep(0.01)
    assert os.waitstatus_to_exitcode(waited[1]) == 0
    parent.release()


def test_capture_budget_rejects_oversized_values_without_retaining_them() -> None:
    budget = CaptureBudget(max_bytes=1024)

    assert budget.accept("x" * 1024) is False
    assert budget.bytes_used == 0
    assert budget.items_used == 0
    assert budget.truncated is True

    assert budget.accept(None) is False
    assert budget.capture_bytes(b"x") == b""
    assert budget.bytes_used == 0
    assert budget.items_used == 0


def test_capture_budget_measures_nested_dict_values_without_skipping_subtrees() -> None:
    short_value = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
    large_value = {"candidates": [{"content": {"parts": [{"text": "x" * 1_000_000}]}}]}

    assert CaptureBudget(_LIMIT).accept(short_value) is True
    assert CaptureBudget(_LIMIT).accept(large_value) is False

    class Model:
        def __init__(self, text: str) -> None:
            self.payload = [{"text": text}]

    assert CaptureBudget(_LIMIT).accept(Model("x")) is True
    assert CaptureBudget(_LIMIT).accept(Model("x" * 1_000_000)) is False

    short_dicts = [{"text": "x"} for _ in range(100)]
    long_dicts = [{"text": "x" * 32} for _ in range(100)]
    short_budget = CaptureBudget()
    long_budget = CaptureBudget()

    assert short_budget.accept(short_dicts) is True
    assert long_budget.accept(long_dicts) is True
    assert long_budget.bytes_used - short_budget.bytes_used == 100 * 31


def test_capture_budget_charges_repeated_aliases_for_each_serialized_occurrence() -> None:
    shared = {"text": "x" * 1_000}

    assert CaptureBudget(_LIMIT).accept([shared] * 100) is False


def test_capture_budget_measures_serialized_binary_footprint() -> None:
    assert CaptureBudget(_LIMIT).accept({"data": b"x" * 30_000}) is True
    assert CaptureBudget(_LIMIT).accept({"data": b"x" * 49_200}) is False


def test_capture_budget_handles_cycles_and_bounded_byte_prefixes() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    structured_budget = CaptureBudget(max_bytes=1024)

    assert structured_budget.accept(cyclic) is True
    assert structured_budget.bytes_used <= structured_budget.max_bytes

    byte_budget = CaptureBudget(max_bytes=8, max_items=2)
    assert byte_budget.capture_bytes(b"0123456789") == b"01234567"
    assert byte_budget.truncated is True
    assert byte_budget.bytes_used == 8
    assert byte_budget.capture_bytes(b"x") == b""


def test_capture_budget_measures_memoryview_in_bytes() -> None:
    value = memoryview(array("I", range(20_000)))
    budget = CaptureBudget(_LIMIT)

    retained = budget.capture_bytes(value)

    assert retained == bytes(value)[: budget.max_bytes]
    assert len(retained) == budget.max_bytes
    assert budget.bytes_used == budget.max_bytes
    assert budget.truncated is True


# Shared with packages/sdk/tests/capture.test.ts so both SDKs are held to the same numbers.
_JSON_STRING_BYTES_FIXTURE = Path(__file__).parent / "fixtures/json-string-bytes.json"
_JSON_STRING_BYTES_CASES: list[dict[str, Any]] = json.loads(
    _JSON_STRING_BYTES_FIXTURE.read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case", _JSON_STRING_BYTES_CASES, ids=[case["name"] for case in _JSON_STRING_BYTES_CASES]
)
def test_capture_budget_charges_strings_and_keys_at_their_json_escaped_size(
    case: dict[str, Any],
) -> None:
    value: str = case["value"]
    expected: int = case["bytes"]

    assert CaptureBudget.json_string_bytes(value) == expected

    as_value = CaptureBudget()
    assert as_value.accept(value) is True
    assert as_value.bytes_used == 16 + expected

    as_key = CaptureBudget()
    assert as_key.accept({value: None}) is True
    assert as_key.bytes_used == 3 * 16 + expected


def test_capture_budget_charges_each_surrogate_code_point_separately() -> None:
    # Python can hold a high and a low surrogate as two code points. json.dumps escapes each
    # one, so each costs six bytes; TypeScript would pair the same UTF-16 units into four.
    assert CaptureBudget.json_string_bytes("\ud83d\ude00") == 12
    assert CaptureBudget.json_string_bytes("\U0001f600") == 4


def test_capture_budget_rejects_escape_heavy_strings_and_keys_over_the_limit() -> None:
    # 1,000 raw bytes fit a 2 KiB budget, 6,000 escaped bytes do not.
    assert CaptureBudget(max_bytes=2048).accept("\x01" * 1000) is False
    assert CaptureBudget(max_bytes=2048).accept({"\x01" * 1000: 1}) is False
    assert CaptureBudget(max_bytes=8192).accept({"\x01" * 1000: 1}) is True


def test_json_string_measure_stops_once_past_the_limit() -> None:
    # Longer than the limit in code points: rejected without a scan.
    assert CaptureBudget.json_string_bytes("x" * 100_000, 10) == 100_000
    # Short enough to scan but escape-heavy: counting stops before the full 30,000 bytes.
    oversized = "\x01" * 5_000
    assert CaptureBudget.json_string_bytes(oversized) == 30_000
    assert 10_000 < CaptureBudget.json_string_bytes(oversized, 10_000) < 30_000
