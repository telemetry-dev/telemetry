from __future__ import annotations

import time
from types import SimpleNamespace

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    HistogramDataPoint,
    Metric,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import ReadableSpan

import telemetry_dev
from telemetry_dev._metrics import GuardedOTLPMetricExporter
from tests.conftest import MakeClient
from tests.otlp_capture import otlp_capture_server


def collect_metrics(env: SimpleNamespace) -> dict[str, Metric]:
    data = env.metric_reader.get_metrics_data()
    out: dict[str, Metric] = {}
    if data is None:
        return out
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                out[metric.name] = metric
    return out


def histogram_points(metric: Metric) -> list[HistogramDataPoint]:
    return [point for point in metric.data.data_points]  # type: ignore[union-attr]


DURATION_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)
TOKEN_BUCKETS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)


def test_generation_records_duration_and_tokens(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span(
        "gen",
        type="generation",
        model="gpt-4o",
        provider="openai",
        response_model="gpt-4o-mini",
    ).end(usage={"input_tokens": 11, "output_tokens": 7})
    metrics = collect_metrics(memory)

    duration = metrics["gen_ai.client.operation.duration"]
    assert duration.unit == "s"
    assert duration.data.aggregation_temporality == AggregationTemporality.DELTA  # type: ignore[union-attr]
    (duration_point,) = histogram_points(duration)
    assert duration_point.explicit_bounds == DURATION_BUCKETS
    assert dict(duration_point.attributes or {}) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.response.model": "gpt-4o-mini",
    }

    tokens = metrics["gen_ai.client.token.usage"]
    assert tokens.unit == "{token}"
    assert tokens.data.aggregation_temporality == AggregationTemporality.DELTA  # type: ignore[union-attr]
    points = {
        dict(point.attributes or {})["gen_ai.token.type"]: point
        for point in histogram_points(tokens)
    }
    assert points["input"].sum == 11
    assert points["output"].sum == 7
    assert points["input"].explicit_bounds == TOKEN_BUCKETS


def test_agent_and_embedding_record_both_histograms(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("agent", type="agent").end(usage={"input_tokens": 3})
    telemetry_dev.start_span("embed", type="embedding").end(usage={"input_tokens": 4})
    metrics = collect_metrics(memory)
    duration_ops = {
        dict(p.attributes or {})["gen_ai.operation.name"]
        for p in histogram_points(metrics["gen_ai.client.operation.duration"])
    }
    assert duration_ops == {"invoke_agent", "embeddings"}
    token_ops = {
        dict(p.attributes or {})["gen_ai.operation.name"]
        for p in histogram_points(metrics["gen_ai.client.token.usage"])
    }
    assert token_ops == {"invoke_agent", "embeddings"}


def test_failed_span_adds_error_type_to_duration_only(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("gen", type="generation").end(
        error=ValueError("boom"), usage={"input_tokens": 4, "output_tokens": 2}
    )
    metrics = collect_metrics(memory)
    (duration_point,) = histogram_points(metrics["gen_ai.client.operation.duration"])
    assert dict(duration_point.attributes or {})["error.type"] == "ValueError"
    token_points = histogram_points(metrics["gen_ai.client.token.usage"])
    assert len(token_points) == 2
    for point in token_points:
        assert "error.type" not in dict(point.attributes or {})


def test_tool_records_duration_only(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("tool", type="tool").end(usage={"input_tokens": 5})
    metrics = collect_metrics(memory)
    (duration_point,) = histogram_points(metrics["gen_ai.client.operation.duration"])
    assert dict(duration_point.attributes or {})["gen_ai.operation.name"] == "execute_tool"
    assert "gen_ai.client.token.usage" not in metrics


def test_plain_span_records_nothing(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("plain").end(usage={"input_tokens": 5})
    metrics = collect_metrics(memory)
    assert "gen_ai.client.operation.duration" not in metrics
    assert "gen_ai.client.token.usage" not in metrics


def test_quiet_interval_emits_zero_metric_posts() -> None:
    # Empty-datapoint guard: collections with no data points are never POSTed.
    # Same exporter + periodic reader wiring the client uses, against a real local server.
    with otlp_capture_server() as server:
        exporter = GuardedOTLPMetricExporter(endpoint=f"{server.url}/v1/metrics")
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=50)
        provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
        meter = provider.get_meter("quiet")
        histogram = meter.create_histogram("gen_ai.client.operation.duration", unit="s")

        time.sleep(0.3)  # several quiet collection intervals
        assert server.requests_for("/v1/metrics") == []

        histogram.record(0.5, {"gen_ai.operation.name": "chat"})
        provider.force_flush()
        assert len(server.requests_for("/v1/metrics")) >= 1
        provider.shutdown()


def test_filter_rejected_spans_record_no_metrics(make: MakeClient) -> None:
    def keep(span: ReadableSpan) -> bool:
        return span.name != "rejected"

    env = make(span_filter=keep)
    telemetry_dev.start_span("rejected", type="generation", usage={"input_tokens": 5}).end()
    telemetry_dev.start_span("kept", type="generation", usage={"input_tokens": 3}).end()
    metrics = collect_metrics(env)
    points = histogram_points(metrics["gen_ai.client.operation.duration"])
    assert len(points) == 1
    assert env.span_exporter.get_finished_spans()[0].name == "kept"
    token_points = histogram_points(metrics["gen_ai.client.token.usage"])
    assert {point.sum for point in token_points} == {3}
