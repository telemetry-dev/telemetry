import { afterEach, expect, test } from "vitest";

import { TRUNCATION_MARKER } from "../src/capture.ts";
import { flush, observe, shutdown, startSpan } from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("mask receives the structured value and the attribute key before stringification", async () => {
  const calls: Array<{ key: string; value: unknown }> = [];

  const { spans } = setup({
    mask: (value, ctx) => {
      calls.push({ key: ctx.key, value });

      if (ctx.key === "gen_ai.input.messages") return { redacted: true };

      return value;
    },
  });

  startSpan("masked", {
    type: "generation",
    input: { prompt: "secret", ssn: "123-45-6789" },
    output: "visible",
  }).end();
  await flush();
  const inputCall = calls.find((c) => c.key === "gen_ai.input.messages")!;
  // Structured object, not a JSON string.
  expect(inputCall.value).toEqual({ prompt: "secret", ssn: "123-45-6789" });
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify({ redacted: true }));
  expect(span.attributes["gen_ai.output.messages"]).toBe("visible");
});

test("mask returning undefined drops the content attribute", async () => {
  const { spans } = setup({ mask: () => undefined });
  startSpan("dropped", { type: "generation", input: "secret" }).end();
  await flush();
  expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]).toBeUndefined();
});

test("a throwing mask drops content but never the span", async () => {
  const { spans } = setup({
    mask: () => {
      throw new Error("mask broke");
    },
    onError: () => {},
  });

  startSpan("survives", { input: "content" }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(span.name).toBe("survives");
});

test("content is truncated to the cap with the marker, marker surviving the span limit", async () => {
  const { spans } = setup();
  startSpan("big", { input: "x".repeat(70000) }).end();
  await flush();
  const value = String(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]);
  expect(value.length).toBe(65536);
  expect(value.endsWith(TRUNCATION_MARKER)).toBe(true);
});

test("truncation counts UTF-16 code units, matching the Python SDK on astral-plane content", async () => {
  const { spans } = setup();
  startSpan("emoji", { input: "🤖".repeat(40000) }).end();
  await flush();
  const value = String(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]);
  expect(value.length).toBe(65536);
  expect(value).toBe("🤖".repeat(32761) + TRUNCATION_MARKER);
});

test("custom maxAttributeLength applies", async () => {
  const { spans } = setup({ maxAttributeLength: 100 });
  startSpan("small-cap", { input: "y".repeat(500) }).end();
  await flush();
  const value = String(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]);
  expect(value.length).toBe(100);
  expect(value.endsWith(TRUNCATION_MARKER)).toBe(true);
});

test("global captureInput:false drops inputs everywhere; per-call override re-enables", async () => {
  const { spans } = setup({ captureInput: false });
  startSpan("default-off", { input: "hidden" }).end();
  startSpan("explicit-on", { input: "shown", captureInput: true }).end();
  await flush();
  const exported = spans.getFinishedSpans();
  expect(
    exported.find((s) => s.name === "default-off")!.attributes["gen_ai.input.messages"],
  ).toBeUndefined();
  expect(exported.find((s) => s.name === "explicit-on")!.attributes["gen_ai.input.messages"]).toBe(
    "shown",
  );
});

test("global captureOutput:false drops outputs including observe results", async () => {
  const { spans } = setup({ captureOutput: false });
  startSpan("no-output", { output: "hidden" }).end();
  const observed = observe(() => "also hidden", { name: "observed-no-output" });
  expect(observed()).toBe("also hidden");
  await flush();
  const exported = spans.getFinishedSpans();
  expect(
    exported.find((span) => span.name === "no-output")!.attributes["gen_ai.output.messages"],
  ).toBeUndefined();
  expect(
    exported.find((span) => span.name === "observed-no-output")!.attributes[
      "gen_ai.output.messages"
    ],
  ).toBeUndefined();
});

test("metadata values flow through truncation", async () => {
  const { spans } = setup({ maxAttributeLength: 50 });
  startSpan("meta", { metadata: { blob: "z".repeat(200) } }).end();
  await flush();
  const value = String(spans.getFinishedSpans()[0]!.attributes["td.metadata.blob"]);
  expect(value.length).toBe(50);
  expect(value.endsWith(TRUNCATION_MARKER)).toBe(true);
});
