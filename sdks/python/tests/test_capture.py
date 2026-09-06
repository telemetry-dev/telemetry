from __future__ import annotations

from array import array

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
