import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import {
  AggregationTemporality,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";

import { type ClientOverrides, init, type TelemetryClient } from "../src/client.ts";
import type { TelemetryOptions } from "../src/config.ts";

export interface Setup {
  client: TelemetryClient;
  spans: InMemorySpanExporter;
  logs: InMemoryLogRecordExporter;
}

export function setup(options?: TelemetryOptions, overrides?: ClientOverrides): Setup {
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();

  const client = init(
    {
      apiKey: "td_live_test",
      serviceName: "svc",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
      ...options,
    },
    { spanExporter: spans, logRecordExporter: logs, ...overrides },
  );

  return { client, spans, logs };
}

export function makeMetricCapture() {
  const batches: ResourceMetrics[] = [];

  const exporter: PushMetricExporter = {
    export: (resourceMetrics, resultCallback) => {
      batches.push(resourceMetrics);
      resultCallback({ code: 0 });
    },
    selectAggregationTemporality: () => AggregationTemporality.DELTA,
    forceFlush: () => Promise.resolve(),
    shutdown: () => Promise.resolve(),
  };

  return { batches, exporter };
}
