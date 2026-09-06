from __future__ import annotations

from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Any

import pytest
import telemetry_dev
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import Histogram
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry_dev_bedrock

MakeClient = Callable[..., SimpleNamespace]


@pytest.fixture(autouse=True)
def isolate_sdk(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    for var in (
        "TELEMETRY_DEV_API_KEY",
        "TELEMETRY_DEV_BASE_URL",
        "TELEMETRY_DEV_ENVIRONMENT",
        "OTEL_SERVICE_NAME",
    ):
        monkeypatch.delenv(var, raising=False)
    telemetry_dev_bedrock.uninstrument_bedrock()
    yield
    telemetry_dev_bedrock.uninstrument_bedrock()
    telemetry_dev.shutdown()


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
            "service_name": "svc",
            "environment": "test",
            "log_level": "silent",
            "disable_atexit": True,
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
