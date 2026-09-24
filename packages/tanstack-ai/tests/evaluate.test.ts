import { SpanKind, SpanStatusCode, type Attributes } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { boolean, decide, type EvaluateAdapter } from "@tanstack/ai";
import { expect, test } from "vitest";

import { telemetryDev } from "../src/index.ts";

function capture() {
  const spanBatches: ReadableSpan[][] = [];
  const metrics: Array<{ type: string; value: number; attributes: Attributes }> = [];

  const middleware = telemetryDev(
    { apiKey: "td_live_test" },
    {
      sendSpans: async (spans) => {
        spanBatches.push(spans);
      },
      recordDuration: (value, attributes) => metrics.push({ type: "duration", value, attributes }),
      recordTokens: (type, value, attributes) => metrics.push({ type, value, attributes }),
    },
  );

  return { middleware, metrics, spanBatches };
}

function adapter(
  name: string,
  evaluate: EvaluateAdapter["evaluate"],
): EvaluateAdapter<string, Record<string, never>> {
  return {
    kind: "evaluate",
    name,
    model: `${name}-model`,
    "~types": { providerOptions: {} },
    evaluate,
  };
}

const questions = { accepted: boolean({ instructions: "Accept?" }) };

const answer = (model: string, promptTokens: number, completionTokens: number) => ({
  model,
  answers: { accepted: { type: "noul" as const, noul: 0.9 } },
  usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
});

test("decide emits isolated evaluation spans and one set of usage metrics for concurrent calls", async () => {
  const { middleware, metrics, spanBatches } = capture();
  let releaseFirst!: () => void;
  const firstBlocked = new Promise<void>((resolve) => (releaseFirst = resolve));

  const first = adapter("first", async () => {
    await firstBlocked;

    return answer("first-resolved", 11, 3);
  });

  const second = adapter("second", async () => answer("second-resolved", 7, 2));

  const firstRun = decide({ adapter: first, state: "one", questions, middleware: [middleware] });
  const secondRun = decide({ adapter: second, state: "two", questions, middleware: [middleware] });
  await secondRun;
  releaseFirst();
  await firstRun;

  expect(spanBatches).toHaveLength(2);
  const spans = spanBatches.flat();
  expect(spans).toHaveLength(2);
  expect(spans.every((span) => span.kind === SpanKind.CLIENT)).toBe(true);
  expect(
    spans
      .map((span) => String(span.attributes["gen_ai.provider.name"]))
      .sort((a, b) => a.localeCompare(b)),
  ).toEqual(["first", "second"]);
  expect(
    spans
      .map((span) => String(span.attributes["gen_ai.request.model"]))
      .sort((a, b) => a.localeCompare(b)),
  ).toEqual(["first-model", "second-model"]);
  expect(
    spans.map((span) => Number(span.attributes["gen_ai.usage.input_tokens"])).sort((a, b) => a - b),
  ).toEqual([7, 11]);
  expect(spans.every((span) => span.attributes["gen_ai.output.messages"] === undefined)).toBe(true);
  expect(metrics.filter((metric) => metric.type === "duration")).toHaveLength(2);
  expect(metrics.filter((metric) => metric.type === "input")).toHaveLength(2);
  expect(metrics.filter((metric) => metric.type === "output")).toHaveLength(2);
});

test.each([
  { error: new TypeError("provider failed"), type: "TypeError", message: "provider failed" },
  {
    error: Object.assign(Object.create(null), { message: "provider failed" }),
    type: "Error",
    message: "provider failed",
  },
  { error: Object.create(null), type: "Error", message: "Unknown error" },
  {
    error: Object.defineProperty({}, "message", {
      get() {
        throw new Error("getter failed");
      },
    }),
    type: "Error",
    message: "Unknown error",
  },
  { error: "provider failed", type: "Error", message: "provider failed" },
])(
  "decide preserves rejection and flushes one terminal span: $type / $message",
  async ({ error, type, message }) => {
    const { middleware, metrics, spanBatches } = capture();

    const failing = adapter("broken", async () => {
      throw error;
    });

    await expect(
      decide({ adapter: failing, state: "bad", questions, middleware: [middleware] }),
    ).rejects.toBe(error);

    expect(spanBatches).toHaveLength(1);
    expect(spanBatches[0]).toHaveLength(1);
    expect(spanBatches[0]![0]!.ended).toBe(true);
    expect(spanBatches[0]![0]!.status.code).toBe(SpanStatusCode.ERROR);
    expect(spanBatches[0]![0]!.attributes["error.type"]).toBe(type);
    expect(spanBatches[0]![0]!.status.message).toBe(message);
    expect(metrics.filter((metric) => metric.type === "duration")).toHaveLength(1);
    expect(metrics.filter((metric) => metric.type === "input")).toHaveLength(0);
  },
);

test("decide abort closes and flushes one cancelled terminal span", async () => {
  const { middleware, spanBatches } = capture();
  const controller = new AbortController();

  const aborting = adapter("aborting", async ({ abortSignal }) => {
    controller.abort();
    abortSignal?.throwIfAborted();

    return answer("unused", 0, 0);
  });

  await expect(
    decide({
      adapter: aborting,
      state: "cancel",
      questions,
      abortSignal: controller.signal,
      middleware: [middleware],
    }),
  ).rejects.toBeDefined();

  expect(spanBatches).toHaveLength(1);
  expect(spanBatches[0]).toHaveLength(1);
  expect(spanBatches[0]![0]!.status.code).toBe(SpanStatusCode.ERROR);
  expect(spanBatches[0]![0]!.attributes["error.type"]).toBe("cancelled");
});
