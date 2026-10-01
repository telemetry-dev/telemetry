import { afterEach, describe, expect, test } from "vitest";

import {
  boundedCapture,
  boundedCaptureDetails,
  truncate,
  TRUNCATION_MARKER,
} from "../src/capture.ts";
import { flush, observe, shutdown, startSpan } from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

describe("boundedCapture", () => {
  test("captures repeated acyclic references independently", () => {
    const shared = { text: "same" };

    expect(boundedCapture([shared, shared])).toEqual({
      value: [{ text: "same" }, { text: "same" }],
      truncated: false,
    });
  });

  test("reports the retained JSON size without changing the public result", () => {
    expect(boundedCaptureDetails({ text: "same" })).toEqual({
      value: { text: "same" },
      truncated: false,
      bytes: 15,
      items: 2,
    });
  });

  test("restores tentative object and array charges after rejecting a value", () => {
    expect(boundedCapture({ a: "oversized", b: 0 }, { maxBytes: 7 })).toEqual({
      value: { b: 0 },
      truncated: true,
    });
    expect(boundedCapture(["oversized", 0], { maxBytes: 3 })).toEqual({
      value: [0],
      truncated: true,
    });
  });

  test("caps traversal attempts when rejected values retain no bytes", () => {
    const input = Array.from({ length: 1_100 }, () => "oversized");
    input.push("kept");

    expect(boundedCapture(input, { maxBytes: 10 })).toEqual({
      value: [],
      truncated: true,
    });

    const skipped = Object.fromEntries(
      Array.from({ length: 1_100 }, (_, index) => [`skip${index}`, index]),
    );

    skipped.kept = 1;

    expect(boundedCapture(skipped, { skip: (key) => key.startsWith("skip") })).toEqual({
      value: {},
      truncated: true,
    });
  });

  test("counts inherited enumerable properties toward the traversal cap", () => {
    const prototype = Object.fromEntries(
      Array.from({ length: 1_100 }, (_, index) => [`inherited${index}`, index]),
    );

    const input: object = Object.create(prototype);

    expect(boundedCapture(input, { maxItems: 10 })).toEqual({
      value: {},
      truncated: true,
    });
  });

  test("preserves __proto__ as data without changing captured object prototypes", () => {
    const input = JSON.parse('{"__proto__":{"index":7},"kept":true}');
    const captured = boundedCapture(input);

    const value = captured.value as {
      __proto__?: { index: number };
      index?: unknown;
      kept?: boolean;
    };

    expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
    expect(Object.hasOwn(value, "__proto__")).toBe(true);
    expect(value["__proto__"]).toEqual({ index: 7 });
    expect(value.index).toBeUndefined();
    expect(value.kept).toBe(true);
    expect(captured.truncated).toBe(false);
  });

  test.each([
    ["maxBytes", Number.NaN],
    ["maxBytes", Number.POSITIVE_INFINITY],
    ["maxDepth", -1],
    ["maxDepth", 1.5],
    ["maxItems", Number.NaN],
    ["maxItems", Number.MAX_VALUE],
  ] as const)("rejects unsafe %s limit %s", (name, value) => {
    expect(() => boundedCapture({ value: "kept" }, { [name]: value })).toThrow(RangeError);
  });

  test("measures escaped and non-BMP strings by their JSON UTF-8 size", () => {
    expect(boundedCapture("😀", { maxBytes: 6 })).toEqual({ value: "😀", truncated: false });
    expect(boundedCapture("😀", { maxBytes: 5 })).toEqual({
      value: undefined,
      truncated: true,
    });
    expect(boundedCapture("\n", { maxBytes: 4 })).toEqual({ value: "\n", truncated: false });
    expect(boundedCapture("\n", { maxBytes: 3 })).toEqual({
      value: undefined,
      truncated: true,
    });
  });
});

test.each([
  [0, ""],
  [TRUNCATION_MARKER.length - 1, TRUNCATION_MARKER.slice(0, -1)],
  [TRUNCATION_MARKER.length, TRUNCATION_MARKER],
])("truncate stays within a %i-character limit", (maxLength, expected) => {
  expect(truncate("x".repeat(TRUNCATION_MARKER.length + 1), maxLength)).toBe(expected);
});

describe("capture integration", () => {
  test("passes structured values and attribute keys to the mask", async () => {
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

    const inputCall = calls.find((call) => call.key === "gen_ai.input.messages")!;
    expect(inputCall.value).toEqual({ prompt: "secret", ssn: "123-45-6789" });
    const span = spans.getFinishedSpans()[0]!;
    expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify({ redacted: true }));
    expect(span.attributes["gen_ai.output.messages"]).toBe("visible");
  });

  test("fails closed when the mask returns undefined or throws", async () => {
    const dropped = setup({ mask: () => undefined });
    startSpan("dropped", { type: "generation", input: "secret" }).end();
    await flush();
    expect(
      dropped.spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"],
    ).toBeUndefined();

    await shutdown();

    const throwing = setup({
      mask: () => {
        throw new Error("mask broke");
      },
      onError: () => {},
    });

    startSpan("survives", { input: "content" }).end();
    await flush();

    const span = throwing.spans.getFinishedSpans()[0]!;
    expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(span.name).toBe("survives");
  });

  test("preserves truncation markers at default and custom span limits", async () => {
    const defaults = setup();
    startSpan("big", { input: "x".repeat(70_000) }).end();
    await flush();

    const defaultValue = String(
      defaults.spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"],
    );

    expect(defaultValue.length).toBe(65_536);
    expect(defaultValue.endsWith(TRUNCATION_MARKER)).toBe(true);

    await shutdown();

    const custom = setup({ maxAttributeLength: 100 });
    startSpan("small-cap", { input: "y".repeat(500) }).end();
    await flush();

    const customValue = String(
      custom.spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"],
    );

    expect(customValue.length).toBe(100);
    expect(customValue.endsWith(TRUNCATION_MARKER)).toBe(true);
  });

  test("counts non-BMP content in UTF-16 code units", async () => {
    const { spans } = setup();
    startSpan("emoji", { input: "🤖".repeat(40_000) }).end();
    await flush();

    const value = String(spans.getFinishedSpans()[0]!.attributes["gen_ai.input.messages"]);
    expect(value.length).toBe(65_536);
    expect(value).toBe("🤖".repeat(32_761) + TRUNCATION_MARKER);
  });

  test("honors global input and output capture controls", async () => {
    const inputCapture = setup({ captureInput: false });
    startSpan("default-off", { input: "hidden" }).end();
    startSpan("explicit-on", { input: "shown", captureInput: true }).end();
    await flush();

    const inputSpans = inputCapture.spans.getFinishedSpans();
    expect(
      inputSpans.find((span) => span.name === "default-off")!.attributes["gen_ai.input.messages"],
    ).toBeUndefined();
    expect(
      inputSpans.find((span) => span.name === "explicit-on")!.attributes["gen_ai.input.messages"],
    ).toBe("shown");

    await shutdown();

    const outputCapture = setup({ captureOutput: false });
    startSpan("no-output", { output: "hidden" }).end();
    const observed = observe(() => "also hidden", { name: "observed-no-output" });
    expect(observed()).toBe("also hidden");
    await flush();

    const outputSpans = outputCapture.spans.getFinishedSpans();
    expect(
      outputSpans.find((span) => span.name === "no-output")!.attributes["gen_ai.output.messages"],
    ).toBeUndefined();
    expect(
      outputSpans.find((span) => span.name === "observed-no-output")!.attributes[
        "gen_ai.output.messages"
      ],
    ).toBeUndefined();
  });

  test("applies the attribute limit to metadata", async () => {
    const { spans } = setup({ maxAttributeLength: 50 });
    startSpan("meta", { metadata: { blob: "z".repeat(200) } }).end();
    await flush();

    const value = String(spans.getFinishedSpans()[0]!.attributes["td.metadata.blob"]);
    expect(value.length).toBe(50);
    expect(value.endsWith(TRUNCATION_MARKER)).toBe(true);
  });
});
