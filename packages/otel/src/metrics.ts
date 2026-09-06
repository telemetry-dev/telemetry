import type { Attributes } from "@opentelemetry/api";
import type { Resource } from "@opentelemetry/resources";
import {
  MeterProvider,
  PeriodicExportingMetricReader,
  type PushMetricExporter,
} from "@opentelemetry/sdk-metrics";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { omitUndefined, SCOPE_NAME, SCOPE_VERSION } from "./attrs.ts";

// Histogram bucket boundaries from the OTel GenAI semantic-convention recommendations for
// gen_ai.client.operation.duration (seconds) and gen_ai.client.token.usage ({token}).
export const DURATION_BUCKETS = [
  0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];
export const TOKEN_BUCKETS = [
  1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
];

const DURATION_OPERATIONS = new Set(["chat", "invoke_agent", "embeddings", "execute_tool"]);
const TOKEN_OPERATIONS = new Set(["chat", "invoke_agent", "embeddings"]);

// Dormant interval for immediate mode, where flush()/shutdown() drive the only exports.
export const DORMANT_INTERVAL_MS = 2 ** 31 - 1;
export const BATCHED_METRIC_INTERVAL_MS = 60_000;

export interface MetricsPipeline {
  record(span: ReadableSpan): void;
  forceFlush(): Promise<void>;
  shutdown(): Promise<void>;
}

function stringAttr(value: Attributes[string]): string | undefined {
  return value?.constructor === String ? `${value}` : undefined;
}

export function createMetricsPipeline({
  resource,
  exporter,
  exportIntervalMillis,
}: {
  resource: Resource;
  exporter: PushMetricExporter;
  exportIntervalMillis: number;
}): MetricsPipeline {
  const reader = new PeriodicExportingMetricReader({ exporter, exportIntervalMillis });
  const meterProvider = new MeterProvider({ resource, readers: [reader] });
  const meter = meterProvider.getMeter(SCOPE_NAME, SCOPE_VERSION);

  const durationHistogram = meter.createHistogram("gen_ai.client.operation.duration", {
    unit: "s",
    advice: { explicitBucketBoundaries: DURATION_BUCKETS },
  });
  const tokenHistogram = meter.createHistogram("gen_ai.client.token.usage", {
    unit: "{token}",
    advice: { explicitBucketBoundaries: TOKEN_BUCKETS },
  });

  const record = (span: ReadableSpan): void => {
    const operation = span.attributes["gen_ai.operation.name"];
    if (operation?.constructor !== String || !DURATION_OPERATIONS.has(`${operation}`)) return;
    const attrs: Attributes = omitUndefined({
      "gen_ai.operation.name": operation,
      "gen_ai.provider.name": stringAttr(span.attributes["gen_ai.provider.name"]),
      "gen_ai.request.model": stringAttr(span.attributes["gen_ai.request.model"]),
      "gen_ai.response.model": stringAttr(span.attributes["gen_ai.response.model"]),
    });
    const durationSec = span.duration[0] + span.duration[1] / 1e9;
    const errorType = stringAttr(span.attributes["error.type"]);
    durationHistogram.record(
      durationSec,
      errorType !== undefined ? { ...attrs, "error.type": errorType } : attrs,
    );
    if (!TOKEN_OPERATIONS.has(operation)) return;
    const inputTokens = span.attributes["gen_ai.usage.input_tokens"];
    if (inputTokens?.constructor === Number) {
      tokenHistogram.record(inputTokens, { ...attrs, "gen_ai.token.type": "input" });
    }
    const outputTokens = span.attributes["gen_ai.usage.output_tokens"];
    if (outputTokens?.constructor === Number) {
      tokenHistogram.record(outputTokens, { ...attrs, "gen_ai.token.type": "output" });
    }
  };

  return {
    record,
    forceFlush: () => reader.forceFlush(),
    shutdown: () => meterProvider.shutdown(),
  };
}
