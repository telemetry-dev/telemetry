from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import pytest
import telemetry_dev
from agents import set_trace_processors
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from telemetry_dev_openai_agents import TelemetryDevProcessor

ENV_VARS = (
    "TELEMETRY_DEV_API_KEY",
    "TELEMETRY_DEV_BASE_URL",
    "TELEMETRY_DEV_ENVIRONMENT",
    "OTEL_SERVICE_NAME",
)


@pytest.fixture(autouse=True)
def isolate_sdk(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    set_trace_processors([])
    telemetry_dev.shutdown()


MakeProcessor = Callable[..., tuple[TelemetryDevProcessor, InMemorySpanExporter]]


@pytest.fixture
def make() -> MakeProcessor:
    """Initializes the core SDK with an in-memory exporter and registers a fresh processor."""

    def _make(**kwargs: Any) -> tuple[TelemetryDevProcessor, InMemorySpanExporter]:
        exporter = InMemorySpanExporter()
        options: dict[str, Any] = {
            "span_exporter": exporter,
            "export_mode": "immediate",
            "service_name": "t",
            "environment": "test",
            "log_level": "silent",
            "disable_atexit": True,
        }
        options.update(kwargs)
        telemetry_dev.init(**options)
        processor = TelemetryDevProcessor()
        set_trace_processors([processor])
        return processor, exporter

    return _make
