from __future__ import annotations

import gc
import math
import threading
import time
from types import SimpleNamespace
from weakref import ref

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    Histogram,
    HistogramDataPoint,
    Metric,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import telemetry_dev
from telemetry_dev._metrics import GuardedOTLPMetricExporter, OutputChunkAggregation
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


def test_streaming_first_chunk_seconds(memory: SimpleNamespace) -> None:
    telemetry_dev.start_span("stream", type="generation", model="gpt-4o", provider="openai").end(
        time_to_first_chunk_ms=250, error=ValueError("failed after first chunk")
    )
    telemetry_dev.start_span("instant", type="generation", time_to_first_chunk_ms=0).end()
    telemetry_dev.start_span("nonstream", type="generation").end()
    for timing in [-1, float("nan"), float("inf")]:
        telemetry_dev.start_span("invalid", type="generation", time_to_first_chunk_ms=timing).end()
    telemetry_dev.start_span("tool", type="tool", time_to_first_chunk_ms=900).end()
    telemetry_dev.start_span("agent", type="agent", time_to_first_chunk_ms=800).end()
    metric = collect_metrics(memory)["gen_ai.client.operation.time_to_first_chunk"]
    assert metric.unit == "s"
    assert isinstance(metric.data, Histogram)
    assert metric.data.aggregation_temporality == AggregationTemporality.DELTA
    points = histogram_points(metric)
    assert len(points) == 2
    streamed = next(
        p for p in points if (p.attributes or {}).get("gen_ai.request.model") == "gpt-4o"
    )
    assert streamed.count == 1
    assert streamed.sum == 0.25
    assert streamed.explicit_bounds == DURATION_BUCKETS
    assert dict(streamed.attributes or {}) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.provider.name": "openai",
    }
    assert next(p for p in points if p is not streamed).sum == 0


def test_custom_reader_preserves_ordinary_metrics_and_reports_chunk_limitation_once(
    make: MakeClient,
) -> None:
    errors: list[BaseException] = []
    env = make(on_error=errors.append)
    for name in ("first", "second"):
        handle = telemetry_dev.start_span(name, type="generation")
        for timestamp in (1000, 1010, 1050, 1210):
            handle.record_output_chunk(timestamp)
        handle.end(usage={"input_tokens": 3}, time_to_first_chunk_ms=250)

    metrics = collect_metrics(env)
    assert "gen_ai.client.operation.duration" in metrics
    assert "gen_ai.client.token.usage" in metrics
    assert "gen_ai.client.operation.time_to_first_chunk" in metrics
    assert "gen_ai.client.operation.time_per_output_chunk" not in metrics
    assert env.client._output_chunks is None
    assert len(errors) == 1
    assert "requires the built-in OTLP metric exporter" in str(errors[0])


@pytest.mark.parametrize("ending", ["complete", "failed", "concurrent", "nonmonotonic"])
def test_builtin_exporter_merges_exact_output_chunk_histogram(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    errors: list[BaseException] = []

    def keep(span: ReadableSpan) -> bool:
        return span.name != "filtered"

    with otlp_capture_server() as server:
        client = telemetry_dev.init(
            api_key="td_test_x",
            base_url=server.url,
            export_mode="immediate",
            span_exporter=InMemorySpanExporter(),
            log_exporter=InMemoryLogRecordExporter(),
            disable_atexit=True,
            span_filter=keep,
            on_error=errors.append,
        )
        telemetry_dev.start_span("filtered", type="generation").record_output_chunk(
            1000
        ).record_output_chunk(9000).end(error=RuntimeError("filtered failure"))
        handle = telemetry_dev.start_span(
            "stream", type="generation", model="gpt-4o", provider="openai"
        )
        timestamps = (
            (1000, 1000, 990, 1010) if ending == "nonmonotonic" else (1000, 1010, 1050, 1210)
        )
        for timestamp in timestamps[:-1]:
            handle.record_output_chunk(timestamp)
        if ending == "concurrent":
            import telemetry_dev._spans as span_module

            bucket_started = threading.Event()
            resume_bucket = threading.Event()
            close_attempted = threading.Event()
            close_finished = threading.Event()
            close_acquired: list[bool] = []
            lock = threading.Lock()
            original_bisect = span_module.bisect_left

            class ObservedLock:
                def __enter__(self) -> None:
                    acquired = lock.acquire(blocking=False)
                    if threading.current_thread() is closer:
                        close_acquired.append(acquired)
                        close_attempted.set()
                    if not acquired:
                        lock.acquire()

                def __exit__(self, *args: object) -> None:
                    lock.release()

            def paused_bucket(boundaries: tuple[float, ...], interval: float) -> int:
                bucket_started.set()
                assert resume_bucket.wait(5)
                return original_bisect(boundaries, interval)

            def close() -> None:
                handle.end()
                close_finished.set()

            recorder = threading.Thread(target=handle.record_output_chunk, args=(timestamps[-1],))
            closer = threading.Thread(target=close)
            monkeypatch.setattr(span_module, "_OUTPUT_CHUNK_STATES_LOCK", ObservedLock())
            monkeypatch.setattr(span_module, "bisect_left", paused_bucket)
            try:
                recorder.start()
                assert bucket_started.wait(5)
                closer.start()
                assert close_attempted.wait(5)
                if close_acquired[0]:
                    assert close_finished.wait(5)
            finally:
                resume_bucket.set()
                recorder.join(timeout=5)
                if closer.ident is not None:
                    closer.join(timeout=5)
            assert not recorder.is_alive() and not closer.is_alive()
            assert close_finished.is_set()
        else:
            handle.record_output_chunk(timestamps[-1])
            handle.end(error=RuntimeError("stream failed") if ending == "failed" else None)
        handle.record_output_chunk(9999)
        client.flush()

        (request,) = server.requests_for("/v1/metrics")
        decoded = ExportMetricsServiceRequest()
        decoded.ParseFromString(request.decompressed())
        matching = [
            (scope_metrics.scope, metric)
            for resource_metrics in decoded.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
            if metric.name == "gen_ai.client.operation.time_per_output_chunk"
        ]
        [(scope, metric)] = matching
        assert scope.name == "telemetry_dev"
        assert len(metric.histogram.data_points) == 1
        point = metric.histogram.data_points[0]
        if ending == "nonmonotonic":
            assert point.count == 2
            assert math.isclose(point.sum, 0.01)
            assert point.min == 0
            assert point.max == 0.01
            assert tuple(point.bucket_counts[:5]) == (2, 0, 0, 0, 0)
        else:
            assert point.count == 3
            assert math.isclose(point.sum, 0.21)
            assert point.min == 0.01
            assert point.max == 0.16
            assert tuple(point.bucket_counts[:5]) == (1, 0, 1, 0, 1)
        assert tuple(point.explicit_bounds) == DURATION_BUCKETS
        assert point.exemplars == []
        assert point.count == sum(point.bucket_counts)
        assert errors == []
        client.shutdown()


def test_output_chunk_zero_or_one_and_filter_rejection_emit_nothing(make: MakeClient) -> None:
    def keep(span: ReadableSpan) -> bool:
        return span.name != "rejected"

    errors: list[BaseException] = []
    env = make(span_filter=keep, on_error=errors.append)
    telemetry_dev.start_span("zero", type="generation").end()
    one = telemetry_dev.start_span("one", type="generation")
    one.record_output_chunk(1000).end()
    rejected = telemetry_dev.start_span("rejected", type="generation")
    rejected.record_output_chunk(1000).record_output_chunk(1020).end()

    assert env.client._output_chunks is None
    assert errors == []


def test_rejected_chunk_state_does_not_retain_abandoned_handles(make: MakeClient) -> None:
    def reject(span: ReadableSpan) -> bool:
        return False

    make(span_filter=reject)
    handle = telemetry_dev.start_span("rejected", type="generation")
    state = ref(vars(handle)["_state"])
    handle.record_output_chunk(0).end()
    del handle
    gc.collect()
    assert state() is None


def test_output_chunk_attribute_cardinality_is_bounded() -> None:
    aggregation = OutputChunkAggregation(max_attribute_sets=2)
    value = (1, 0.01, 0.01, 0.01, (1, *([0] * len(DURATION_BUCKETS))))
    for model in ("a", "b", "c", "d"):
        aggregation.add({"gen_ai.request.model": model}, value)

    points = aggregation.drain_points()
    assert len(points) == 2
    overflow = next(point for point in points if point.attributes == {"otel.metric.overflow": True})
    assert overflow.count == 3


def test_ended_child_metrics_reference_child_not_active_parent(memory: SimpleNamespace) -> None:
    with telemetry_dev.start_span("parent", type="agent"):
        telemetry_dev.start_span("child", type="generation").end(
            usage={"input_tokens": 11, "output_tokens": 7}, time_to_first_chunk_ms=250
        )
        child = memory.span_exporter.get_finished_spans()[0].context
        assert child is not None
        metrics = collect_metrics(memory)
        for metric in metrics.values():
            for point in histogram_points(metric):
                assert len(point.exemplars) == 1
                exemplar = point.exemplars[0]
                assert exemplar.trace_id == child.trace_id
                assert exemplar.span_id == child.span_id


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
    telemetry_dev.start_span(
        "rejected", type="generation", usage={"input_tokens": 5}, time_to_first_chunk_ms=300
    ).end()
    telemetry_dev.start_span("kept", type="generation", usage={"input_tokens": 3}).end()
    metrics = collect_metrics(env)
    points = histogram_points(metrics["gen_ai.client.operation.duration"])
    assert len(points) == 1
    assert env.span_exporter.get_finished_spans()[0].name == "kept"
    token_points = histogram_points(metrics["gen_ai.client.token.usage"])
    assert {point.sum for point in token_points} == {3}
    assert "gen_ai.client.operation.time_to_first_chunk" not in metrics
