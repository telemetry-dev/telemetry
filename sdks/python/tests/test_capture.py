from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Any

import pytest

from telemetry_dev import CaptureBudget


def test_capture_budget_caps_bytes_and_items() -> None:
    budget = CaptureBudget()

    for _ in range(1024):
        assert budget.accept(None) is True

    assert budget.accept(None) is False
    assert budget.truncated is True
    assert budget.items_used == budget.max_items
    assert budget.bytes_used <= budget.max_bytes


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

    assert CaptureBudget().accept(short_value) is True
    assert CaptureBudget().accept(large_value) is False

    class Model:
        def __init__(self, text: str) -> None:
            self.payload = [{"text": text}]

    assert CaptureBudget().accept(Model("x")) is True
    assert CaptureBudget().accept(Model("x" * 1_000_000)) is False

    short_dicts = [{"text": "x"} for _ in range(100)]
    long_dicts = [{"text": "x" * 32} for _ in range(100)]
    short_budget = CaptureBudget()
    long_budget = CaptureBudget()

    assert short_budget.accept(short_dicts) is True
    assert long_budget.accept(long_dicts) is True
    assert long_budget.bytes_used - short_budget.bytes_used == 100 * 31


def test_capture_budget_charges_repeated_aliases_for_each_serialized_occurrence() -> None:
    shared = {"text": "x" * 1_000}

    assert CaptureBudget().accept([shared] * 100) is False


def test_capture_budget_measures_serialized_binary_footprint() -> None:
    assert CaptureBudget().accept({"data": b"x" * 30_000}) is True
    assert CaptureBudget().accept({"data": b"x" * 49_200}) is False


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
    budget = CaptureBudget()

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
