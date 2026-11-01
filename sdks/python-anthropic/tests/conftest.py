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

# Tests that build a stream state without a client size the process-wide stream budget, which
# only init() and the span processor set; the core's own tests use the same private name.
from telemetry_dev._capture import stream_budget  # pyright: ignore[reportPrivateUsage]

import telemetry_dev_anthropic

ENV_VARS = (
    "TELEMETRY_DEV_API_KEY",
    "TELEMETRY_DEV_BASE_URL",
    "TELEMETRY_DEV_ENVIRONMENT",
    "OTEL_SERVICE_NAME",
)

# The stream bound most tests check. Another stream holds a byte in the shared stream budget,
# so a test stream gets max_stream_bytes minus that byte: STREAM_LIMIT, unless the test
# releases other_stream or passes its own max_stream_bytes.
STREAM_LIMIT = 48 * 1024
other_stream: telemetry_dev.CaptureBudget = telemetry_dev.CaptureBudget()


@pytest.fixture(autouse=True)
def isolate_sdk(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    telemetry_dev_anthropic.uninstrument_anthropic()
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Tests that build a stream state without make() get the same bound; monkeypatch restores
    # the process-wide value that make() also sets.
    monkeypatch.setattr(stream_budget, "max_bytes", STREAM_LIMIT + 1)
    # A stream state an earlier test left in a reference cycle still holds its bytes in the
    # shared budget until it is collected, and would shrink the bound of this test's streams.
    gc.collect()
    global other_stream
    other_stream = telemetry_dev.CaptureBudget.for_stream()
    other_stream.bytes_used = 1
    yield
    other_stream.release()
    gc.collect()
    telemetry_dev_anthropic.uninstrument_anthropic()
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
