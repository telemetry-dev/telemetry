import { ValueType } from "@opentelemetry/api";
import { ExportResultCode } from "@opentelemetry/core";
import {
  ProtobufLogsSerializer,
  ProtobufMetricsSerializer,
  ProtobufTraceSerializer,
} from "@opentelemetry/otlp-transformer";
import { resourceFromAttributes } from "@opentelemetry/resources";
import type { LogRecordExporter, ReadableLogRecord } from "@opentelemetry/sdk-logs";
import {
  AggregationTemporality,
  type DataPoint,
  DataPointType,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
  type SpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  createMetricExporter,
  createLogExporter,
  createTraceExporter,
  maybeGzip,
  otlpHeaders,
  postOtlp,
} from "../src/transport.ts";

interface Call {
  url: string;
  headers: Record<string, string>;
  body: Uint8Array;
}

function makeFetch(statuses: number[], headers?: Record<string, string>) {
  const calls: Call[] = [];

  const fetchImpl: typeof fetch = (input, init) => {
    if (!init) throw new Error("expected request init");
    calls.push({
      url: input instanceof URL ? input.href : input instanceof Request ? input.url : input,
      headers: init.headers as Record<string, string>,
      body: init.body as Uint8Array,
    });
    const status = statuses[Math.min(calls.length - 1, statuses.length - 1)]!;

    return Promise.resolve(new Response(null, { status, headers }));
  };

  return { calls, fetchImpl };
}

function makeSpans(count: number, attrBytes = 0): ReadableSpan[] {
  const exporter = new InMemorySpanExporter();

  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
    spanLimits: { attributeValueLengthLimit: 10_000_000 },
  });

  const tracer = provider.getTracer("test");

  for (let i = 0; i < count; i += 1) {
    const span = tracer.startSpan(`span-${i}`);

    if (attrBytes > 0) span.setAttribute("payload", "p".repeat(attrBytes));
    span.end();
  }

  return exporter.getFinishedSpans();
}

const exportSpans = (exporter: SpanExporter, spans: ReadableSpan[]): Promise<ExportResultCode> =>
  new Promise((resolve) => exporter.export(spans, (result) => resolve(result.code)));

function makeGaugeMetrics(dataPoints: DataPoint<number>[]): ResourceMetrics {
  return {
    resource: resourceFromAttributes({}),
    scopeMetrics: [
      {
        scope: { name: "s" },
        metrics: [
          {
            descriptor: { name: "gauge", description: "", unit: "", valueType: ValueType.DOUBLE },
            aggregationTemporality: AggregationTemporality.CUMULATIVE,
            dataPointType: DataPointType.GAUGE,
            dataPoints,
          },
        ],
      },
    ],
  };
}

const GAUGE_POINT: DataPoint<number> = {
  attributes: {},
  startTime: [1, 0],
  endTime: [2, 0],
  value: 1,
};

const LOG_RECORD: ReadableLogRecord = {
  hrTime: [1, 0],
  hrTimeObserved: [1, 0],
  body: "log",
  resource: resourceFromAttributes({}),
  instrumentationScope: { name: "test" },
  attributes: {},
  droppedAttributesCount: 0,
};

const exportMetrics = (
  exporter: PushMetricExporter,
  metrics: ResourceMetrics,
): Promise<ExportResultCode> =>
  new Promise((resolve) => exporter.export(metrics, (result) => resolve(result.code)));

test("retries on 429/503 then succeeds", async () => {
  const { calls, fetchImpl } = makeFetch([429, 503, 200]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl },
  );

  const code = await exportSpans(exporter, makeSpans(1));
  expect(code).toBe(ExportResultCode.SUCCESS);
  expect(calls).toHaveLength(3);
  expect(calls[0]!.headers.authorization).toBe("Bearer td_live_x");
  expect(calls[0]!.headers["content-type"]).toBe("application/x-protobuf");
  expect(calls[0]!.headers["x-telemetry-dev-sdk"]).toBe("@telemetry-dev/otel");
});

test("does not retry non-retryable statuses and reports the failure", async () => {
  const errors: unknown[] = [];
  const { calls, fetchImpl } = makeFetch([400]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl, onError: (e) => errors.push(e) },
  );

  const code = await exportSpans(exporter, makeSpans(1));
  expect(code).toBe(ExportResultCode.FAILED);
  expect(calls).toHaveLength(1);
  expect(errors).toHaveLength(1);
});

test("gives up after exhausting retries", async () => {
  const { calls, fetchImpl } = makeFetch([503, 503, 503]);

  const res = await postOtlp({
    fetchImpl,
    url: "https://ingest.example/v1/traces",
    headers: {},
    body: new Uint8Array([1]),
  });

  expect(res.status).toBe(503);
  expect(calls).toHaveLength(3);
});

test("exhausted retries surface as a failed export with onError", async () => {
  const errors: unknown[] = [];
  const { calls, fetchImpl } = makeFetch([503, 503, 503]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl, onError: (e) => errors.push(e) },
  );

  const code = await exportSpans(exporter, makeSpans(1));
  expect(code).toBe(ExportResultCode.FAILED);
  expect(calls).toHaveLength(3);
  expect(String(errors[0])).toContain("503");
});

test("reports retry-after values beyond the local cap", async () => {
  const errors: unknown[] = [];
  const { calls, fetchImpl } = makeFetch([429, 200], { "retry-after": "60" });

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl, onError: (error) => errors.push(error) },
  );

  const code = await exportSpans(exporter, makeSpans(1));
  expect(code).toBe(ExportResultCode.FAILED);
  expect(calls).toHaveLength(1);
  expect(String(errors[0])).toContain("retry-after 60 exceeds 2s retry cap");
});

describe("retry-after", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const startPost = (statuses: number[], headers: Record<string, string>) => {
    const { calls, fetchImpl } = makeFetch(statuses, headers);

    const pending = postOtlp({
      fetchImpl,
      url: "https://ingest.example/v1/traces",
      headers: {},
      body: new Uint8Array([1]),
    });

    return { calls, pending };
  };

  test("sets the delay before the next attempt", async () => {
    const { calls, pending } = startPost([503, 200], { "retry-after": "1" });
    await vi.advanceTimersByTimeAsync(999);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls).toHaveLength(2);
    expect((await pending).status).toBe(200);
  });

  test("waits when retry-after fits under the cap", async () => {
    const { calls, pending } = startPost([503, 200], { "retry-after": "2" });
    await vi.advanceTimersByTimeAsync(1999);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls).toHaveLength(2);
    expect((await pending).status).toBe(200);
  });

  test("gives up at once when retry-after exceeds the cap", async () => {
    const { calls, pending } = startPost([429, 200], { "retry-after": "60" });
    expect((await pending).status).toBe(429);
    expect(calls).toHaveLength(1);
  });

  test("an HTTP date sets the delay relative to now", async () => {
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));

    const { calls, pending } = startPost([503, 200], {
      "retry-after": new Date(Date.now() + 1000).toUTCString(),
    });

    await vi.advanceTimersByTimeAsync(999);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls).toHaveLength(2);
    expect((await pending).status).toBe(200);
  });

  test.each(["Thursday, 01-Jan-26 00:00:01 GMT", "Thu Jan  1 00:00:01 2026"])(
    "accepts obsolete HTTP date %s",
    async (retryAfter) => {
      vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
      const { calls, pending } = startPost([503, 200], { "retry-after": retryAfter });
      await vi.advanceTimersByTimeAsync(999);
      expect(calls).toHaveLength(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toHaveLength(2);
      expect((await pending).status).toBe(200);
    },
  );

  test("falls back to the fixed schedule when unparseable", async () => {
    const { calls, pending } = startPost([503, 200], { "retry-after": "soon" });
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toHaveLength(2);
    expect((await pending).status).toBe(200);
  });

  test.each([
    "+1",
    "1.5",
    "2026-01-01T00:00:01Z",
    "Thu, 31 Feb 2026 00:00:01 GMT",
    "Fri, 01 Jan 2026 00:00:01 GMT",
    "Friday, 01-Jan-26 00:00:01 GMT",
    "Thursday, 31-Feb-26 00:00:01 GMT",
    "Fri Jan  1 00:00:01 2026",
    "Thu Feb 31 00:00:01 2026",
    "January 1, 2026 00:00:01 GMT",
  ])("falls back to the fixed schedule for invalid value %s", async (retryAfter) => {
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    const { calls, pending } = startPost([503, 200], { "retry-after": retryAfter });
    await vi.advanceTimersByTimeAsync(99);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls).toHaveLength(2);
    expect((await pending).status).toBe(200);
  });
});

test("a missing CompressionStream ships large bodies uncompressed", async () => {
  vi.stubGlobal("CompressionStream", undefined);

  try {
    const original = new TextEncoder().encode("a".repeat(5000));
    const { body, contentEncoding } = await maybeGzip(original);
    expect(contentEncoding).toBeUndefined();
    expect(body).toBe(original);
  } finally {
    vi.unstubAllGlobals();
  }
});

test("bodies above the threshold are gzipped and round-trip", async () => {
  const original = new TextEncoder().encode("a".repeat(5000));
  const { body, contentEncoding } = await maybeGzip(original);
  expect(contentEncoding).toBe("gzip");
  expect(body.byteLength).toBeLessThan(original.byteLength);

  const decompressed = new Uint8Array(
    await new Response(
      new Blob([body as Uint8Array<ArrayBuffer>])
        .stream()
        .pipeThrough(new DecompressionStream("gzip")),
    ).arrayBuffer(),
  );

  expect(decompressed).toEqual(original);
});

test("small bodies are not compressed", async () => {
  const original = new TextEncoder().encode("tiny");
  const { body, contentEncoding } = await maybeGzip(original);
  expect(contentEncoding).toBeUndefined();
  expect(body).toBe(original);
});

test("exported batches send a content-encoding gzip header for large payloads", async () => {
  const { calls, fetchImpl } = makeFetch([200]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl },
  );

  await exportSpans(exporter, makeSpans(1, 10_000));
  expect(calls[0]!.headers["content-encoding"]).toBe("gzip");
});

test("oversized batches split into multiple POSTs", async () => {
  const { calls, fetchImpl } = makeFetch([200]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl },
  );

  // Two ~2 MB spans: combined serialization exceeds the 3.5 MB guard, each half fits.
  const code = await exportSpans(exporter, makeSpans(2, 2_000_000));
  expect(code).toBe(ExportResultCode.SUCCESS);
  expect(calls).toHaveLength(2);
});

test("a single span beyond the limit is dropped with onError, not wedged", async () => {
  const errors: unknown[] = [];
  const { calls, fetchImpl } = makeFetch([200]);

  const exporter = createTraceExporter(
    { url: "https://ingest.example/v1/traces", headers: otlpHeaders("td_live_x") },
    { fetchImpl, onError: (e) => errors.push(e) },
  );

  const code = await exportSpans(exporter, makeSpans(1, 4_000_000));
  expect(code).toBe(ExportResultCode.SUCCESS);
  expect(calls).toHaveLength(0);
  expect(String(errors[0])).toContain("exceeds max export size");
});

test("metric exporter skips POSTs when every data point set is empty", async () => {
  const { calls, fetchImpl } = makeFetch([200]);

  const exporter = createMetricExporter(
    { url: "https://ingest.example/v1/metrics", headers: otlpHeaders("td_live_x") },
    { fetchImpl },
  );

  const code = await exportMetrics(exporter, makeGaugeMetrics([]));
  expect(code).toBe(ExportResultCode.SUCCESS);
  expect(calls).toHaveLength(0);
});

test.each([429, 503])("metric exporter retries status %s", async (status) => {
  const { calls, fetchImpl } = makeFetch([status, 200]);

  const exporter = createMetricExporter(
    { url: "https://ingest.example/v1/metrics", headers: otlpHeaders("td_live_x") },
    { fetchImpl },
  );

  const code = await exportMetrics(exporter, makeGaugeMetrics([GAUGE_POINT]));
  expect(code).toBe(ExportResultCode.SUCCESS);
  expect(calls).toHaveLength(2);
  expect(calls[1]!.body).toEqual(calls[0]!.body);
});

test("metric exporter exhausts retries for a network failure", async () => {
  const errors: unknown[] = [];
  let calls = 0;

  const fetchImpl: typeof fetch = () => {
    calls += 1;

    return Promise.reject(new Error("connection reset"));
  };

  const exporter = createMetricExporter(
    { url: "https://ingest.example/v1/metrics", headers: otlpHeaders("td_live_x") },
    { fetchImpl, onError: (e) => errors.push(e) },
  );

  const code = await exportMetrics(exporter, makeGaugeMetrics([GAUGE_POINT]));
  expect(code).toBe(ExportResultCode.FAILED);
  expect(calls).toBe(3);
  expect(String(errors[0])).toContain("connection reset");
});

const target = { url: "https://ingest.example/v1/telemetry", headers: otlpHeaders("td_live_x") };

const lifecycleCases = [
  {
    name: "trace",
    create: (fetchImpl: typeof fetch) => createTraceExporter(target, { fetchImpl }),
    start: (exporter: SpanExporter, callback: (code: ExportResultCode) => void) =>
      exporter.export(makeSpans(1), (result) => callback(result.code)),
  },
  {
    name: "log",
    create: (fetchImpl: typeof fetch) => createLogExporter(target, { fetchImpl }),
    start: (exporter: LogRecordExporter, callback: (code: ExportResultCode) => void) =>
      exporter.export([LOG_RECORD], (result) => callback(result.code)),
  },
  {
    name: "metric",
    create: (fetchImpl: typeof fetch) => createMetricExporter(target, { fetchImpl }),
    start: (exporter: PushMetricExporter, callback: (code: ExportResultCode) => void) =>
      exporter.export(makeGaugeMetrics([GAUGE_POINT]), (result) => callback(result.code)),
  },
] as const;

describe.each(lifecycleCases)("$name exporter lifecycle", ({ create, start }) => {
  test("shutdown waits for active sends and rejects later exports", async () => {
    let resolveFetch: ((response: Response) => void) | undefined;

    const fetchImpl: typeof fetch = () =>
      new Promise((resolve) => {
        resolveFetch = resolve;
      });

    const exporter = create(fetchImpl);
    const events: string[] = [];
    start(exporter as never, (code) => events.push(`active:${code}`));
    await vi.waitFor(() => expect(resolveFetch).toBeDefined());

    const shutdown = exporter.shutdown().then(() => {
      events.push("shutdown");
    });

    await Promise.resolve();
    expect(events).toEqual([]);
    start(exporter as never, (code) => events.push(`late:${code}`));
    expect(events).toEqual([`late:${ExportResultCode.FAILED}`]);
    resolveFetch!(new Response(null, { status: 200 }));
    await shutdown;
    expect(events).toEqual([
      `late:${ExportResultCode.FAILED}`,
      `active:${ExportResultCode.SUCCESS}`,
      "shutdown",
    ]);
  });

  test("forceFlush resolves after every send active when it begins", async () => {
    const resolvers: Array<(response: Response) => void> = [];

    const fetchImpl: typeof fetch = () =>
      new Promise((resolve) => {
        resolvers.push(resolve);
      });

    const exporter = create(fetchImpl);
    const events: string[] = [];
    start(exporter as never, (code) => events.push(`first:${code}`));
    start(exporter as never, (code) => events.push(`second:${code}`));
    await vi.waitFor(() => expect(resolvers).toHaveLength(2));

    const flush = exporter.forceFlush!().then(() => {
      events.push("flush");
    });

    resolvers[0]!(new Response(null, { status: 200 }));
    await vi.waitFor(() => expect(events).toEqual([`first:${ExportResultCode.SUCCESS}`]));
    resolvers[1]!(new Response(null, { status: 200 }));
    await flush;
    expect(events).toEqual([
      `first:${ExportResultCode.SUCCESS}`,
      `second:${ExportResultCode.SUCCESS}`,
      "flush",
    ]);
  });
});

test("forceFlush waits through a retry delay", async () => {
  vi.useFakeTimers();

  try {
    const { calls, fetchImpl } = makeFetch([503, 200]);
    const exporter = createTraceExporter(target, { fetchImpl });
    exporter.export(makeSpans(1), () => undefined);
    let flushed = false;

    const flush = exporter.forceFlush!().then(() => {
      flushed = true;
    });

    await vi.advanceTimersByTimeAsync(99);
    expect(calls).toHaveLength(1);
    expect(flushed).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await flush;
    expect(calls).toHaveLength(2);
  } finally {
    vi.useRealTimers();
  }
});

test.each([
  ["trace", ProtobufTraceSerializer],
  ["log", ProtobufLogsSerializer],
  ["metric", ProtobufMetricsSerializer],
] as const)(
  "%s serializer throws produce a failed result without stranding flush",
  async (name, serializer) => {
    vi.spyOn(serializer, "serializeRequest").mockImplementationOnce(() => {
      throw new Error("serialize failed");
    });
    const { fetchImpl } = makeFetch([200]);
    const lifecycle = lifecycleCases.find((entry) => entry.name === name)!;
    const exporter = lifecycle.create(fetchImpl);
    const results: ExportResultCode[] = [];
    lifecycle.start(exporter as never, (code) => results.push(code));
    await expect(exporter.forceFlush!()).resolves.toBeUndefined();
    expect(results).toEqual([ExportResultCode.FAILED]);
  },
);

test("callback and onError throws do not strand lifecycle state", async () => {
  const exporter = createTraceExporter(target, {
    fetchImpl: () => Promise.resolve(new Response(null, { status: 400 })),
    onError: () => {
      throw new Error("onError failed");
    },
  });

  exporter.export(makeSpans(1), () => {
    throw new Error("callback failed");
  });
  await expect(exporter.forceFlush!()).resolves.toBeUndefined();
  await expect(exporter.shutdown()).resolves.toBeUndefined();
});

test("never-settling sends time out once and release lifecycle state", async () => {
  const spans = makeSpans(1);
  vi.useFakeTimers();

  try {
    const signals: AbortSignal[] = [];
    const results: ExportResultCode[][] = [[], []];

    const fetchImpl: typeof fetch = (_input, init) => {
      signals.push(init!.signal!);

      return new Promise(() => undefined);
    };

    const exporter = createTraceExporter(target, {
      fetchImpl,
      exportTimeoutMillis: 100,
      onError: () => {
        throw new Error("onError failed");
      },
    });

    exporter.export(spans, (result) => {
      results[0]!.push(result.code);
      throw new Error("callback failed");
    });
    exporter.export(spans, (result) => results[1]!.push(result.code));
    await Promise.resolve();
    await Promise.resolve();
    expect(signals).toHaveLength(2);

    const flush = exporter.forceFlush!();
    const shutdown = exporter.shutdown();
    await vi.advanceTimersByTimeAsync(100);
    await expect(flush).resolves.toBeUndefined();
    await expect(shutdown).resolves.toBeUndefined();

    expect(results).toEqual([[ExportResultCode.FAILED], [ExportResultCode.FAILED]]);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
    expect(vi.getTimerCount()).toBe(0);

    exporter.export(spans, (result) => results[0]!.push(result.code));
    expect(results[0]).toEqual([ExportResultCode.FAILED, ExportResultCode.FAILED]);
    expect(signals).toHaveLength(2);
    expect(vi.getTimerCount()).toBe(0);
  } finally {
    vi.useRealTimers();
  }
});

test.each([-1, Number.NaN, Number.POSITIVE_INFINITY, 2_147_483_648])(
  "invalid export timeout %s uses the default",
  async (exportTimeoutMillis) => {
    vi.useFakeTimers();

    try {
      const signals: AbortSignal[] = [];
      const results: ExportResultCode[] = [];

      const exporter = createTraceExporter(target, {
        fetchImpl: (_input, init) => {
          signals.push(init!.signal!);

          return new Promise(() => undefined);
        },
        exportTimeoutMillis,
      });

      exporter.export(makeSpans(1), (result) => results.push(result.code));
      await Promise.resolve();
      await Promise.resolve();

      await vi.advanceTimersByTimeAsync(29_999);
      expect(results).toEqual([]);
      expect(signals[0]!.aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(1);
      expect(results).toEqual([ExportResultCode.FAILED]);
      expect(signals[0]!.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  },
);
