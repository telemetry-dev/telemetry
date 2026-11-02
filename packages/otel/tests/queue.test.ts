import { runInNewContext } from "node:vm";
import { setFlagsFromString } from "node:v8";

import { BasicTracerProvider, InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { DEFAULT_BATCH } from "../src/config.ts";
import { StampingSpanProcessor } from "../src/processor.ts";
import { MAX_RECORD_BYTES, StreamCapture, streamCaptureBudget } from "../src/queue.ts";

const MAX_STREAM_BYTES = 10_000;
const open: StreamCapture[] = [];
let previousMaxBytes: number;

function capture(bytes = 0): StreamCapture {
  const stream = new StreamCapture(bytes);
  open.push(stream);

  return stream;
}

beforeEach(() => {
  previousMaxBytes = streamCaptureBudget.maxBytes;
  streamCaptureBudget.maxBytes = MAX_STREAM_BYTES;
});

afterEach(() => {
  for (const stream of open.splice(0)) stream.release();
  streamCaptureBudget.maxBytes = previousMaxBytes;
});

test.each([
  ["alone", 0, 0, MAX_RECORD_BYTES],
  ["alone past the shared bound", 20_000, 0, MAX_RECORD_BYTES],
  ["beside another stream", 0, 3_000, MAX_STREAM_BYTES - 3_000],
  ["holding part of the room left", 2_000, 3_000, MAX_STREAM_BYTES - 3_000],
  ["after others filled the bound", 0, MAX_STREAM_BYTES, 0],
  ["after others passed the bound", 0, 12_000, 0],
  ["holding more than the room left", 5_000, 8_000, 5_000],
])("a stream %s may hold up to its limit", (_case, own, others, limit) => {
  if (others > 0) capture(others);

  expect(capture(own).limit).toBe(limit);
});

test("a shared bound of 0 leaves a stream beside another one no room", () => {
  streamCaptureBudget.maxBytes = 0;
  capture(1);

  expect(capture().limit).toBe(0);
});

test("release gives the bytes back once and freezes the limit", () => {
  const first = capture(4_000);
  const second = capture(1_000);

  expect(second.limit).toBe(MAX_STREAM_BYTES - 4_000);
  first.release();
  first.release();
  // A stream still read after it ended may keep no more than it held at release.
  expect(first.limit).toBe(4_000);
  // A capture that keeps writing after release holds nothing in the bound.
  first.bytes = 9_000;

  expect(first.limit).toBe(4_000);
  expect(second.limit).toBe(MAX_RECORD_BYTES);
  expect(capture().limit).toBe(MAX_STREAM_BYTES - 1_000);
});

test("a stream dropped without release gives its bytes back when it is collected", async () => {
  setFlagsFromString("--expose-gc");
  // With --expose-gc set, a new context defines gc().
  const gc = runInNewContext("gc") as () => void;
  // The capture is only reachable from this closure, which ends before the collection.
  (() => void new StreamCapture(6_000))();
  const probe = capture();

  for (let attempt = 0; attempt < 20 && probe.limit !== MAX_RECORD_BYTES; attempt += 1) {
    gc();
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  expect(probe.limit).toBe(MAX_RECORD_BYTES);
});

test("a span processor keeps the timing options it was given beside the byte bounds", async () => {
  vi.useFakeTimers();

  try {
    const exporter = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [
        new StampingSpanProcessor({
          exporter,
          exportMode: "batched",
          batch: {
            maxExportBatchSize: 64,
            scheduledDelayMillis: 60_000,
            maxQueueSize: 2048,
            exportTimeoutMillis: 30000,
            maxStreamBytes: 5_000,
          },
        }),
      ],
    });

    provider.getTracer("queue").startSpan("waits").end();
    // Past the default delay of 1,000 ms, before the 60,000 ms that was given.
    await vi.advanceTimersByTimeAsync(5_000);
    expect(exporter.getFinishedSpans()).toEqual([]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(exporter.getFinishedSpans().map((span) => span.name)).toEqual(["waits"]);
    await provider.shutdown();
  } finally {
    vi.useRealTimers();
  }
});

test("a span processor sizes the bound, with the default for an absent or invalid value", () => {
  const errors: Error[] = [];
  const batch = { maxExportBatchSize: 64, scheduledDelayMillis: 1000, maxQueueSize: 2048 };
  const processor = (maxStreamBytes?: number, maxQueueBytes?: number) =>
    new StampingSpanProcessor({
      exporter: new InMemorySpanExporter(),
      exportMode: "immediate",
      batch: { ...batch, exportTimeoutMillis: 30000, maxStreamBytes, maxQueueBytes },
      onError: (error) => errors.push(error),
    });

  processor(5_000);
  expect(streamCaptureBudget.maxBytes).toBe(5_000);
  // Callers of earlier releases build batch without the byte bounds.
  processor();
  expect(streamCaptureBudget.maxBytes).toBe(DEFAULT_BATCH.maxStreamBytes);
  processor(5_000);
  processor(Number.NaN);
  expect(streamCaptureBudget.maxBytes).toBe(DEFAULT_BATCH.maxStreamBytes);
  // An invalid queue bound leaves a valid stream bound in place.
  processor(6_000, -1);
  expect(streamCaptureBudget.maxBytes).toBe(6_000);
  expect(errors).toHaveLength(2);
  expect(errors.every((error) => error instanceof RangeError)).toBe(true);
});
