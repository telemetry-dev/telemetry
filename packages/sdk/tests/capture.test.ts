import { readFileSync } from "node:fs";

import { SpanStatusCode } from "@opentelemetry/api";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  boundedCapture,
  boundedCaptureDetails,
  truncate,
  TRUNCATION_MARKER,
} from "../src/capture.ts";
import {
  flush,
  observe,
  shutdown,
  startActiveSpan,
  startSpan,
  updateActiveSpan,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

test.each([
  [false, true],
  [true, false],
  [false, false],
])(
  "exception content respects captureInput=%s captureOutput=%s and per-span overrides",
  async (captureInput, captureOutput) => {
    const { spans } = setup({ captureInput, captureOutput });
    const error = new TypeError("PRIVATE_EXCEPTION");
    startSpan("initial", { error }).end();
    startSpan("ended").end({ error });
    startActiveSpan("active", () => updateActiveSpan({ error }));
    const failed = observe(async () => {
      throw error;
    });
    await expect(failed()).rejects.toBe(error);
    startSpan("override", { captureInput: true, captureOutput: true }).end({ error });
    await flush();
    expect(spans.getFinishedSpans()).toHaveLength(5);
    for (const span of spans.getFinishedSpans()) {
      expect(span.status.code).toBe(SpanStatusCode.ERROR);
      expect(span.attributes["error.type"]).toBe("TypeError");
      const event = span.events.find((event) => event.name === "exception")!;
      expect(event.attributes?.["exception.type"]).toBe("TypeError");
      if (span.name === "override") {
        expect(event.attributes?.["exception.message"]).toBe("PRIVATE_EXCEPTION");
      } else {
        expect(event.attributes?.["exception.message"]).toBeUndefined();
        expect(event.attributes?.["exception.stacktrace"]).toBeUndefined();
        expect(JSON.stringify(span.events)).not.toContain("PRIVATE_EXCEPTION");
      }
    }
  },
);

test.each(["redact", "drop", "throw", "truncate"] as const)(
  "exception masking: %s",
  async (mode) => {
    const { spans } = setup({
      maxAttributeLength: 50,
      mask: (value, { key }) => {
        if (mode === "throw") throw new Error("mask failed");
        if (mode === "drop") return undefined;
        if (mode === "truncate") return "x".repeat(100);
        return key === "exception.message"
          ? "safe message"
          : key === "exception.stacktrace"
            ? "safe stack"
            : value;
      },
    });
    const error = new Error("PRIVATE_EXCEPTION");
    const failure = observe(() => {
      throw error;
    });
    expect(failure).toThrow(error);
    startSpan("private", { captureOutput: false }).end({ error: "PRIVATE_EXCEPTION" });
    await flush();
    const [span, privateSpan] = spans.getFinishedSpans();
    expect(span!.status.code).toBe(SpanStatusCode.ERROR);
    expect(span!.attributes["error.type"]).toBe("Error");
    const attributes = span!.events[0]!.attributes;
    expect(attributes?.["exception.message"]).toBe(
      mode === "redact"
        ? "safe message"
        : mode === "truncate"
          ? "x".repeat(36) + TRUNCATION_MARKER
          : undefined,
    );
    expect(attributes?.["exception.stacktrace"]).toBe(
      mode === "redact"
        ? "safe stack"
        : mode === "truncate"
          ? "x".repeat(36) + TRUNCATION_MARKER
          : undefined,
    );
    expect(privateSpan!.events[0]!.attributes?.["exception.message"]).toBeUndefined();
    expect(JSON.stringify([span!.events, privateSpan!.events])).not.toContain("PRIVATE_EXCEPTION");
  },
);

test.each([false, true])(
  "mask failure diagnostics hide content (callback throws: %s)",
  async (throws) => {
    const diagnostic = vi.spyOn(console, "error").mockImplementation(() => {});
    const errors: Error[] = [];
    const original = new Error("PRIVATE_APPLICATION");
    try {
      const { spans } = setup({
        logLevel: "debug",
        mask: () => {
          throw new Error("PRIVATE_MASK", { cause: original });
        },
        onError: (error) => {
          errors.push(error);
          if (throws) throw new Error("PRIVATE_CALLBACK", { cause: original });
        },
      });
      const failure = observe(() => {
        throw original;
      });
      expect(failure).toThrow(original);
      await flush();
      expect(errors.length).toBeGreaterThan(0);
      for (const error of errors) {
        expect(error.message).toBe("mask hook raised; content dropped");
        expect(error.cause).toBeUndefined();
        expect(error.stack).not.toContain("PRIVATE_");
      }
      expect(diagnostic).toHaveBeenCalled();
      for (const args of diagnostic.mock.calls)
        expect(args.map(String).join(" ")).not.toContain("PRIVATE_");
      expect(JSON.stringify(spans.getFinishedSpans()[0]!.events)).not.toContain("PRIVATE_");
    } finally {
      diagnostic.mockRestore();
    }
  },
);

// Shared with the Python SDK's capture tests. Read at runtime: the JSON module loader rejects
// the lone surrogate escapes these cases need.
const jsonStringBytesCases = JSON.parse(
  readFileSync(
    new URL("../../../sdks/python/tests/fixtures/json-string-bytes.json", import.meta.url),
    "utf8",
  ),
) as { name: string; value: string; bytes: number }[];

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

  // Shared with sdks/python/tests/test_capture.py so both SDKs are held to the same numbers.
  test.each(jsonStringBytesCases.map((entry) => [entry.name, entry] as const))(
    "charges %s at its JSON-escaped size as a value and as a key",
    (_name, { value, bytes }) => {
      expect(boundedCaptureDetails(value).bytes).toBe(bytes + 2);
      // Braces, the quoted key, the colon, and the value 0.
      expect(boundedCaptureDetails({ [value]: 0 }).bytes).toBe(2 + bytes + 2 + 1 + 1);
    },
  );
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

  test("serializeContent masks, stringifies and truncates integration content", () => {
    const rewrite = (value: unknown): unknown =>
      typeof value === "string"
        ? value.replaceAll("secret", "[masked]")
        : Array.isArray(value)
          ? value.map(rewrite)
          : value !== null && typeof value === "object"
            ? Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewrite(item)]))
            : value;

    const keys: string[] = [];

    setup({
      maxAttributeLength: 40,
      mask: (value, ctx) => {
        keys.push(ctx.key);

        return rewrite(value);
      },
    });

    const span = startSpan("integration", { captureInput: false, captureOutput: false });

    expect(span.serializeContent?.("tool failed: secret path", "exception.message")).toBe(
      "tool failed: [masked] path",
    );
    expect(span.serializeContent?.({ chunks: [{ text: "secret doc" }] }, "grounding")).toBe(
      '{"chunks":[{"text":"[masked] doc"}]}',
    );
    expect(span.serializeContent?.(`secret ${"x".repeat(60)}`, "long")).toBe(
      `[masked] ${"x".repeat(17)}...[truncated]`,
    );
    expect(keys).toEqual(["exception.message", "grounding", "long"]);
    span.end();
  });

  test("serializeContent withholds content the mask cannot handle", () => {
    expect(startSpan("before-init").serializeContent?.("secret", "key")).toBeUndefined();

    const errors: Error[] = [];

    setup({
      mask: (value, ctx) => {
        if (ctx.key === "dropped") return undefined;

        if (ctx.key === "broken") throw new Error("mask broke");

        if (ctx.key === "odd-throw") throw Object.create(null);

        return value;
      },
      onError: (error) => errors.push(error),
    });

    const span = startSpan("integration");
    const circular: Record<string, unknown> = { text: "secret" };
    circular.self = circular;

    expect(span.serializeContent?.("secret", "dropped")).toBeUndefined();
    expect(span.serializeContent?.("secret", "broken")).toBeUndefined();
    expect(span.serializeContent?.("secret", "odd-throw")).toBeUndefined();
    expect(span.serializeContent?.(circular, "circular")).toBeUndefined();
    expect(errors.map((error) => error.message)).toEqual([
      "mask hook raised; content dropped",
      "mask hook raised; content dropped",
    ]);
    span.end();
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
