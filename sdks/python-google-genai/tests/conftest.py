from __future__ import annotations

import gc
from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Any

import pytest
import telemetry_dev
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import Histogram
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry_dev_google_genai

ENV_VARS = (
    "TELEMETRY_DEV_API_KEY",
    "TELEMETRY_DEV_BASE_URL",
    "TELEMETRY_DEV_ENVIRONMENT",
    "OTEL_SERVICE_NAME",
)


# Another stream holds a byte of the shared stream budget, and make() sizes that budget at
# STREAM_LIMIT + 1, so each test stream keeps at most STREAM_LIMIT, the old fixed stream bound.
STREAM_LIMIT = 48 * 1024
other_stream: telemetry_dev.CaptureBudget = telemetry_dev.CaptureBudget()


@pytest.fixture(autouse=True)
def isolate_sdk(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    telemetry_dev_google_genai.uninstrument_google_genai()
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # A stream that an earlier test left unfinished holds its bytes until it is collected.
    gc.collect()
    global other_stream
    other_stream = telemetry_dev.CaptureBudget.for_stream()
    other_stream.bytes_used = 1
    yield
    other_stream.release()
    telemetry_dev_google_genai.uninstrument_google_genai()
    telemetry_dev.shutdown()


MakeClient = Callable[..., SimpleNamespace]


@pytest.fixture
def make() -> MakeClient:
    def _make(**kwargs: Any) -> SimpleNamespace:
        span_exporter = InMemorySpanExporter()
        metric_reader = InMemoryMetricReader(
            preferred_temporality={Histogram: AggregationTemporality.DELTA}
        )
        log_exporter = InMemoryLogRecordExporter()
        options: dict[str, Any] = {
            "span_exporter": span_exporter,
            "metric_reader": metric_reader,
            "log_exporter": log_exporter,
            "export_mode": "immediate",
            "service_name": "t",
            "environment": "test",
            "log_level": "silent",
            "disable_atexit": True,
            "max_stream_bytes": STREAM_LIMIT + 1,
        }
        options.update(kwargs)
        client = telemetry_dev.init(**options)
        return SimpleNamespace(
            client=client,
            span_exporter=span_exporter,
            metric_reader=metric_reader,
            log_exporter=log_exporter,
        )

    return _make


@pytest.fixture
def memory(make: MakeClient) -> SimpleNamespace:
    return make()
