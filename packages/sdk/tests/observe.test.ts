import { SpanStatusCode } from "@opentelemetry/api";
import { afterEach, expect, test } from "vitest";

import { flush, observe, shutdown } from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("sync function: name, single-arg input, return output", async () => {
  const { spans } = setup();
  const double = observe(function double(n: number) {
    return n * 2;
  });
  expect(double(21)).toBe(42);
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.name).toBe("double");
  expect(span.attributes["gen_ai.operation.name"]).toBe("function");
  expect(span.attributes["gen_ai.input.messages"]).toBe("21");
  expect(span.attributes["gen_ai.output.messages"]).toBe("42");
});

test("multiple args are captured as an array; zero args capture nothing", async () => {
  const { spans } = setup();
  const add = observe((a: number, b: number) => a + b, { name: "add" });
  const nothing = observe(() => "ok", { name: "nothing" });
  add(1, 2);
  nothing();
  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported.find((s) => s.name === "add")!.attributes["gen_ai.input.messages"]).toBe(
    JSON.stringify([1, 2]),
  );
  expect(
    exported.find((s) => s.name === "nothing")!.attributes["gen_ai.input.messages"],
  ).toBeUndefined();
});

test("async function ends after resolution with output", async () => {
  const { spans } = setup();
  const fetchData = observe(
    async function fetchData() {
      await Promise.resolve();
      return { items: [1, 2] };
    },
    { type: "tool", toolName: "fetch-data" },
  );
  await fetchData();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.operation.name"]).toBe("execute_tool");
  expect(span.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ items: [1, 2] }));
});

test("sync throw is recorded and rethrown", async () => {
  const { spans } = setup();
  const boom = observe(function boom(): never {
    throw new RangeError("sync fail");
  });
  expect(() => boom()).toThrow("sync fail");
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("RangeError");
});

test("async rejection is recorded and rethrown", async () => {
  const { spans } = setup();
  const boom = observe(async function asyncBoom() {
    await Promise.resolve();
    throw new Error("async fail");
  });
  await expect(boom()).rejects.toThrow("async fail");
  await flush();
  expect(spans.getFinishedSpans()[0]!.status.code).toBe(SpanStatusCode.ERROR);
});

test("nested observes parent correctly across await", async () => {
  const { spans } = setup();
  const inner = observe(async function inner() {
    await Promise.resolve();
    return "inner-done";
  });
  const outer = observe(async function outer() {
    await Promise.resolve();
    return inner();
  });
  await outer();
  await flush();
  const exported = spans.getFinishedSpans();
  const outerSpan = exported.find((s) => s.name === "outer")!;
  const innerSpan = exported.find((s) => s.name === "inner")!;
  expect(innerSpan.parentSpanContext?.spanId).toBe(outerSpan.spanContext().spanId);
  expect(innerSpan.spanContext().traceId).toBe(outerSpan.spanContext().traceId);
});

test("wrapping before init works; the client is resolved at call time", async () => {
  const wrapped = observe(function early(x: string) {
    return x.toUpperCase();
  });
  expect(wrapped("pre-init")).toBe("PRE-INIT");
  const { spans } = setup();
  wrapped("post-init");
  await flush();
  expect(spans.getFinishedSpans()).toHaveLength(1);
  // String values pass through the capture funnel unquoted.
  expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]).toBe("post-init");
});

test("anonymous functions fall back to the anonymous name", async () => {
  const { spans } = setup();
  observe(() => 1)();
  await flush();
  // Arrow functions assigned inline get an inferred name only when bound; the wrapper
  // falls back to "anonymous" when fn.name is empty.
  expect(spans.getFinishedSpans()[0]!.name).toBe("anonymous");
});
