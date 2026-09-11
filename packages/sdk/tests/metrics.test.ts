import {
  AggregationTemporality,
  DataPointType,
  type HistogramMetricData,
} from "@opentelemetry/sdk-metrics";
import { afterEach, expect, test } from "vitest";

import { flush, shutdown, startSpan } from "../src/index.ts";
import { makeMetricCapture, setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("streaming generations record first-chunk seconds without inventing missing timings", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("stream", { type: "generation", model: "gpt-4o", provider: "openai" }).end({
    timeToFirstChunkMs: 250,
    error: new Error("failed after first chunk"),
  });
  startSpan("instant", { type: "generation", timeToFirstChunkMs: 0 }).end();
  startSpan("nonstream", { type: "generation" }).end();

  for (const timeToFirstChunkMs of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
    startSpan("invalid", { type: "generation", timeToFirstChunkMs }).end();
  }

  startSpan("tool", { type: "tool", timeToFirstChunkMs: 900 }).end();
  startSpan("agent", { type: "agent", timeToFirstChunkMs: 800 }).end();
  await flush();

  const points = histogramPoints(batches, "gen_ai.client.operation.time_to_first_chunk");
  expect(points).toHaveLength(2);
  const streamed = points.find((p) => p.attributes["gen_ai.request.model"] === "gpt-4o")!;
  expect(streamed.value.count).toBe(1);
  expect(streamed.value.sum).toBe(0.25);
  expect(streamed.attributes).toEqual({
    "gen_ai.operation.name": "chat",
    "gen_ai.request.model": "gpt-4o",
    "gen_ai.provider.name": "openai",
  });
  expect(points.find((p) => p !== streamed)!.value.sum).toBe(0);
});

test("output chunk intervals export their bounded aggregate only after an accepted span ends", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({ spanFilter: (span) => span.name !== "rejected" }, { metricExporter: exporter });
  const kept = startSpan("kept", { type: "generation", model: "model", provider: "provider" });
  kept.recordOutputChunk(0);
  kept.recordOutputChunk(15);
  kept.recordOutputChunk(55);
  await flush();
  expect(histogramPoints(batches, "gen_ai.client.operation.time_per_output_chunk")).toHaveLength(0);
  kept.end();

  const rejected = startSpan("rejected", { type: "generation" });
  rejected.recordOutputChunk(100);
  rejected.recordOutputChunk(200);
  rejected.end();
  startSpan("zero", { type: "generation" }).end();
  const one = startSpan("one", { type: "generation" });
  one.recordOutputChunk();
  one.end();
  await flush();

  const points = histogramPoints(batches, "gen_ai.client.operation.time_per_output_chunk");
  expect(points).toHaveLength(1);
  expect(points[0]!.attributes).toEqual({
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "provider",
    "gen_ai.request.model": "model",
  });
  expect(points[0]!.value).toEqual({
    count: 2,
    sum: 0.055,
    min: 0.015,
    max: 0.04,
    buckets: {
      boundaries: [
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
      ],
      counts: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
  });
});

test("output chunks merge series, cap cardinality, and reset after collection", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });

  for (let index = 0; index < 2001; index++) {
    const span = startSpan("stream", { type: "generation", model: `model-${index}` });
    span.recordOutputChunk(100);
    span.recordOutputChunk(110);
    span.end();
    span.recordOutputChunk(1000);
  }

  const same = startSpan("same", { type: "generation", model: "model-0" });

  for (const timestamp of [100, 100, Number.NaN, 90, Number.POSITIVE_INFINITY, 140]) {
    same.recordOutputChunk(timestamp);
  }

  same.end({ error: new Error("stream interrupted") });
  await flush();
  const points = histogramPoints(batches, "gen_ai.client.operation.time_per_output_chunk");
  expect(points).toHaveLength(2000);
  const merged = points.find((point) => point.attributes["gen_ai.request.model"] === "model-0")!;
  expect(merged.value.count).toBe(3);
  expect(merged.value.sum).toBeCloseTo(0.05);
  expect(merged.value.min).toBe(0);
  expect(merged.value.max).toBe(0.04);
  expect(merged.value.buckets.counts.slice(0, 3)).toEqual([2, 0, 1]);
  expect(merged.attributes["error.type"]).toBeUndefined();
  expect(points.find((point) => point.attributes["otel.metric.overflow"])!.value.count).toBe(2);
  batches.length = 0;
  await flush();
  expect(histogramPoints(batches, "gen_ai.client.operation.time_per_output_chunk")).toHaveLength(0);
});

function histogramPoints(batches: ReturnType<typeof makeMetricCapture>["batches"], name: string) {
  return batches
    .flatMap((rm) => rm.scopeMetrics)
    .flatMap((sm) => sm.metrics)
    .filter((m): m is HistogramMetricData => m.descriptor.name === name)
    .flatMap((m) => m.dataPoints);
}

test("generation spans record duration and both token histograms", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("gen", {
    type: "generation",
    model: "gpt-4o",
    provider: "openai",
    usage: { inputTokens: 10, outputTokens: 5 },
  }).end();
  await flush();

  const durations = histogramPoints(batches, "gen_ai.client.operation.duration");
  expect(durations).toHaveLength(1);
  expect(durations[0]!.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(durations[0]!.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(durations[0]!.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(durations[0]!.value.count).toBe(1);

  const tokens = histogramPoints(batches, "gen_ai.client.token.usage");
  expect(tokens).toHaveLength(2);
  const input = tokens.find((p) => p.attributes["gen_ai.token.type"] === "input")!;
  const output = tokens.find((p) => p.attributes["gen_ai.token.type"] === "output")!;
  expect(input.value.sum).toBe(10);
  expect(output.value.sum).toBe(5);
});

test("failed spans add error.type to the duration histogram only", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("gen", { type: "generation", usage: { inputTokens: 4, outputTokens: 2 } }).end({
    error: new TypeError("boom"),
  });
  await flush();
  const durations = histogramPoints(batches, "gen_ai.client.operation.duration");
  expect(durations).toHaveLength(1);
  expect(durations[0]!.attributes["error.type"]).toBe("TypeError");
  const tokens = histogramPoints(batches, "gen_ai.client.token.usage");
  expect(tokens).toHaveLength(2);
  for (const point of tokens) {
    expect(point.attributes["error.type"]).toBeUndefined();
  }
});

test("tool spans record duration only", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("tool", { type: "tool", usage: { inputTokens: 99 } }).end();
  await flush();
  expect(histogramPoints(batches, "gen_ai.client.operation.duration")).toHaveLength(1);
  expect(histogramPoints(batches, "gen_ai.client.token.usage")).toHaveLength(0);
});

test("agent and embedding spans record duration and tokens", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("agent", { type: "agent", usage: { inputTokens: 7 } }).end();
  startSpan("embed", { type: "embedding", usage: { inputTokens: 3 } }).end();
  await flush();
  const durations = histogramPoints(batches, "gen_ai.client.operation.duration");
  const operations = durations.map((d) => String(d.attributes["gen_ai.operation.name"]));
  expect(operations.sort((a, b) => a.localeCompare(b))).toEqual(["embeddings", "invoke_agent"]);
  const tokens = histogramPoints(batches, "gen_ai.client.token.usage");
  expect(tokens).toHaveLength(2);
});

test("plain function spans record no metrics", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("fn", { usage: { inputTokens: 4 } }).end();
  await flush();
  expect(histogramPoints(batches, "gen_ai.client.operation.duration")).toHaveLength(0);
  expect(histogramPoints(batches, "gen_ai.client.token.usage")).toHaveLength(0);
});

test("histograms use DELTA temporality and the GenAI bucket boundaries", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  startSpan("gen", { type: "generation", usage: { inputTokens: 10 } }).end();
  await flush();
  const metric = batches
    .flatMap((rm) => rm.scopeMetrics)
    .flatMap((sm) => sm.metrics)
    .find((m) => m.descriptor.name === "gen_ai.client.token.usage")!;
  expect(metric.aggregationTemporality).toBe(AggregationTemporality.DELTA);
  expect(metric.dataPointType).toBe(DataPointType.HISTOGRAM);
  const point = metric.dataPoints[0]!;
  expect(
    point.value instanceof Object && "buckets" in point.value ? point.value.buckets.boundaries : [],
  ).toEqual([
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
  ]);
});

test("filter-rejected spans record no metrics", async () => {
  const { batches, exporter } = makeMetricCapture();
  const { spans } = setup(
    { spanFilter: (span) => span.name !== "rejected" },
    { metricExporter: exporter },
  );

  startSpan("rejected", {
    type: "generation",
    usage: { inputTokens: 5 },
    timeToFirstChunkMs: 300,
  }).end();
  startSpan("kept", { type: "generation", usage: { inputTokens: 3 } }).end();
  await flush();
  expect(spans.getFinishedSpans().map((s) => s.name)).toEqual(["kept"]);
  const tokens = histogramPoints(batches, "gen_ai.client.token.usage");
  expect(tokens).toHaveLength(1);
  expect(tokens[0]!.value.sum).toBe(3);
  expect(histogramPoints(batches, "gen_ai.client.operation.time_to_first_chunk")).toHaveLength(0);
});

test("quiet intervals collect zero data points", async () => {
  const { batches, exporter } = makeMetricCapture();
  setup({}, { metricExporter: exporter });
  await flush();
  const pointCount = batches
    .flatMap((rm) => rm.scopeMetrics)
    .flatMap((sm) => sm.metrics)
    .reduce((total, metric) => total + metric.dataPoints.length, 0);
  expect(pointCount).toBe(0);
});
