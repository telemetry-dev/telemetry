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
  startSpan("rejected", { type: "generation", usage: { inputTokens: 5 } }).end();
  startSpan("kept", { type: "generation", usage: { inputTokens: 3 } }).end();
  await flush();
  expect(spans.getFinishedSpans().map((s) => s.name)).toEqual(["kept"]);
  const tokens = histogramPoints(batches, "gen_ai.client.token.usage");
  expect(tokens).toHaveLength(1);
  expect(tokens[0]!.value.sum).toBe(3);
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
