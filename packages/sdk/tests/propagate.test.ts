import { ROOT_CONTEXT, SpanKind, TraceFlags } from "@opentelemetry/api";
import {
  AlwaysOffSampler,
  AlwaysOnSampler,
  ParentBasedSampler,
  SamplingDecision,
  TraceIdRatioBasedSampler,
} from "@opentelemetry/sdk-trace-base";
import { sessionSpanContext } from "@telemetry-dev/otel";
import { afterEach, expect, test } from "vitest";
import { TRUNCATION_MARKER } from "../src/capture.ts";

import {
  flush,
  log,
  observe,
  propagateAttributes,
  shutdown,
  startActiveSpan,
  startSpan,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("stamps user.id, gen_ai.conversation.id, and td.metadata.* on every span in scope", async () => {
  const { spans } = setup();
  const traced = observe(function worker() {
    startSpan("manual-in-scope").end();
    return "ok";
  });
  propagateAttributes(
    { userId: "u_1", sessionId: "conv_9", metadata: { plan: "pro", flags: { beta: true } } },
    () => {
      startActiveSpan("outer", () => {
        traced();
      });
    },
  );
  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported).toHaveLength(3);
  for (const span of exported) {
    expect(span.attributes["user.id"]).toBe("u_1");
    expect(span.attributes["gen_ai.conversation.id"]).toBe("conv_9");
    expect(span.attributes["td.metadata.plan"]).toBe("pro");
    expect(span.attributes["td.metadata.flags"]).toBe(JSON.stringify({ beta: true }));
  }
});

test("propagation survives await boundaries", async () => {
  const { spans } = setup();
  await propagateAttributes({ userId: "u_async" }, async () => {
    await Promise.resolve();
    startSpan("after-await").end();
  });
  await flush();
  expect(spans.getFinishedSpans()[0]!.attributes["user.id"]).toBe("u_async");
});

test("nested scopes merge with inner values winning per key", async () => {
  const { spans } = setup();
  propagateAttributes({ userId: "u_outer", sessionId: "s_outer" }, () => {
    propagateAttributes({ sessionId: "s_inner", metadata: { depth: 2 } }, () => {
      startSpan("nested").end();
    });
    startSpan("outer-only").end();
  });
  await flush();
  const nested = spans.getFinishedSpans().find((s) => s.name === "nested")!;
  expect(nested.attributes["user.id"]).toBe("u_outer");
  expect(nested.attributes["gen_ai.conversation.id"]).toBe("s_inner");
  expect(nested.attributes["td.metadata.depth"]).toBe("2");
  const outer = spans.getFinishedSpans().find((s) => s.name === "outer-only")!;
  expect(outer.attributes["gen_ai.conversation.id"]).toBe("s_outer");
  expect(outer.attributes["td.metadata.depth"]).toBeUndefined();
});

test("reserved metadata keys are dropped", async () => {
  const { spans } = setup();
  propagateAttributes({ userId: "u_1", metadata: { userId: "spoof", sessionId: "spoof" } }, () => {
    startSpan("clean").end();
  });
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["td.metadata.userId"]).toBeUndefined();
  expect(span.attributes["td.metadata.sessionId"]).toBeUndefined();
  expect(span.attributes["user.id"]).toBe("u_1");
});

test("log records carry propagated attributes", async () => {
  const { logs } = setup();
  propagateAttributes({ userId: "u_log", sessionId: "conv_log" }, () => {
    log("inside scope");
  });
  await flush();
  const record = logs.getFinishedLogRecords()[0]!;
  expect(record.attributes["user.id"]).toBe("u_log");
  expect(record.attributes["gen_ai.conversation.id"]).toBe("conv_log");
});

test("log records cap propagated attributes with the client limit", async () => {
  const { logs } = setup({ maxAttributeLength: 30 });
  propagateAttributes({ metadata: { blob: "z".repeat(100) } }, () => {
    log("inside scope");
  });
  await flush();
  const value = logs.getFinishedLogRecords()[0]!.attributes["td.metadata.blob"];
  expect(value).toBeTypeOf("string");
  expect(value).toBe("z".repeat(30 - TRUNCATION_MARKER.length) + TRUNCATION_MARKER);
});

test("works before init without throwing", () => {
  const result = propagateAttributes({ userId: "u_preinit" }, () => "ran");
  expect(result).toBe("ran");
});

test("unserializable metadata never throws and keeps serializable keys", async () => {
  const { spans } = setup();
  const circular = { self: null };
  Object.defineProperty(circular, "self", { value: circular, enumerable: true });
  propagateAttributes({ userId: "safe", metadata: { circular, ok: "v" } }, () => {
    startSpan("survives").end();
  });
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["user.id"]).toBe("safe");
  expect(span.attributes["td.metadata.ok"]).toBe("v");
  expect(span.attributes["td.metadata.circular"]).toBeUndefined();
});

test("explicit handle parents keep ambient propagated attributes", async () => {
  const { spans } = setup();
  const detached = startSpan("detached-parent");
  propagateAttributes({ userId: "ambient" }, () => {
    startSpan("explicit-child", { parent: detached }).end();
  });
  detached.end();
  await flush();
  const child = spans.getFinishedSpans().find((s) => s.name === "explicit-child")!;
  expect(child.attributes["user.id"]).toBe("ambient");
  expect(child.parentSpanContext?.spanId).toBe(detached.spanId);
});

test("root spans of one session share the deterministic session trace", async () => {
  const { spans } = setup();
  const session = sessionSpanContext("td_live_test", "s1");
  propagateAttributes({ sessionId: "s1" }, () => {
    startSpan("a").end();
    startSpan("b").end();
  });
  startSpan("c", { attributes: { "gen_ai.conversation.id": "s1" } }).end();
  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported).toHaveLength(3);
  for (const span of exported) {
    expect(span.spanContext().traceId).toBe(session.traceId);
    expect(span.parentSpanContext?.spanId).toBe(session.spanId);
  }
});

test("nested spans keep their real parent inside a session", async () => {
  const { spans } = setup();
  propagateAttributes({ sessionId: "s1" }, () => {
    startActiveSpan("root", () => {
      startSpan("child").end();
    });
  });
  await flush();
  const root = spans.getFinishedSpans().find((s) => s.name === "root")!;
  const child = spans.getFinishedSpans().find((s) => s.name === "child")!;
  expect(child.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
  expect(root.parentSpanContext?.spanId).toBe(sessionSpanContext("td_live_test", "s1").spanId);
});

test("spans without a session id keep separate random traces", async () => {
  const { spans } = setup();
  startSpan("a").end();
  startSpan("b").end();
  await flush();
  const [a, b] = spans.getFinishedSpans();
  expect(a!.spanContext().traceId).not.toBe(b!.spanContext().traceId);
  expect(a!.parentSpanContext).toBeUndefined();
});

test("an empty explicit session id overrides the propagated session", async () => {
  const { spans } = setup();
  propagateAttributes({ sessionId: "from-context" }, () => {
    startSpan("empty-explicit", {
      attributes: { "gen_ai.conversation.id": "" },
    }).end();
  });
  await flush();
  expect(spans.getFinishedSpans()[0]!.parentSpanContext).toBeUndefined();
});

test("session roots obey configured root sampling and retain deterministic trace IDs", async () => {
  for (const root of [new AlwaysOffSampler(), new TraceIdRatioBasedSampler(0.5)]) {
    const sampler = new ParentBasedSampler({ root });
    const { spans } = setup({ sampler });
    const expected: string[] = [];
    for (let i = 0; i < 20; i++) {
      const sessionId = `session-${i}`;
      const traceId = sessionSpanContext("td_live_test", sessionId).traceId;
      const sampled =
        sampler.shouldSample(ROOT_CONTEXT, traceId, "turn", SpanKind.INTERNAL, {}, []).decision ===
        SamplingDecision.RECORD_AND_SAMPLED;
      propagateAttributes({ sessionId }, () => {
        startActiveSpan("turn", (span) => {
          expect(span.isRecording).toBe(sampled);
          expect(span.traceId).toBe(traceId);
          const child = startSpan("child");
          expect(child.isRecording).toBe(sampled);
          expect(child.traceId).toBe(traceId);
          child.end();
        });
      });
      if (sampled) expected.push(traceId, traceId);
    }
    await flush();
    expect(spans.getFinishedSpans().map((span) => span.spanContext().traceId)).toEqual(expected);
    await shutdown();
  }
});

test("an unsampled explicit parent stays active for children inside a session", async () => {
  const { spans } = setup({ sampler: new ParentBasedSampler({ root: new AlwaysOnSampler() }) });
  const parent = {
    traceId: "12345678901234567890123456789012",
    spanId: "1234567890123456",
    traceFlags: TraceFlags.NONE,
    isRemote: true,
  };
  propagateAttributes({ sessionId: "s1" }, () => {
    startActiveSpan("turn", { parent }, (span) => {
      expect(span.isRecording).toBe(false);
      const child = startSpan("child");
      expect(child.isRecording).toBe(false);
      expect(child.traceId).toBe(parent.traceId);
      child.end();
    });
  });
  await flush();
  expect(spans.getFinishedSpans()).toEqual([]);
});

test.each([
  { sessionId: undefined, captureInput: true },
  { sessionId: "s1", captureInput: true },
  { sessionId: "s1", captureInput: false },
])(
  "root sampling sees safe initial attributes ($sessionId, capture $captureInput)",
  async ({ sessionId, captureInput }) => {
    let masked = 0;
    const { spans } = setup({
      captureInput: false,
      captureOutput: false,
      mask: () => `masked-${++masked}`,
      sampler: new ParentBasedSampler({
        root: {
          shouldSample(_ctx, _traceId, _name, _kind, attributes) {
            const accepted =
              attributes["gen_ai.operation.name"] === "chat" &&
              attributes["gen_ai.request.model"] === "sampled-model" &&
              attributes["gen_ai.provider.name"] === "explicit-provider" &&
              attributes["user.id"] === "inherited-user" &&
              attributes["td.metadata.plan"] === "explicit-plan" &&
              attributes["gen_ai.conversation.id"] === sessionId &&
              attributes["gen_ai.input.messages"] === (captureInput ? "masked-1" : undefined) &&
              attributes["gen_ai.output.messages"] === undefined;
            return {
              decision: accepted
                ? SamplingDecision.RECORD_AND_SAMPLED
                : SamplingDecision.NOT_RECORD,
            };
          },
          toString: () => "AttributeSampler",
        },
      }),
    });
    propagateAttributes(
      { userId: "inherited-user", sessionId, metadata: { plan: "inherited-plan" } },
      () => {
        startSpan("accepted", {
          type: "generation",
          model: "sampled-model",
          provider: "field-provider",
          metadata: { plan: "field-plan" },
          attributes: {
            "gen_ai.provider.name": "explicit-provider",
            "td.metadata.plan": "explicit-plan",
          },
          input: "private input",
          output: "private output",
          captureInput,
        }).end();
        startSpan("rejected", { type: "generation", model: "other-model", captureInput }).end();
      },
    );
    await flush();
    const exported = spans.getFinishedSpans();
    expect(exported.map((span) => span.name)).toEqual(["accepted"]);
    expect(exported[0]?.attributes).toMatchObject({
      "gen_ai.request.model": "sampled-model",
      "gen_ai.provider.name": "explicit-provider",
      "user.id": "inherited-user",
      "td.metadata.plan": "explicit-plan",
    });
    expect(exported[0]?.attributes["gen_ai.input.messages"]).toBe(
      captureInput ? "masked-1" : undefined,
    );
    expect(exported[0]?.attributes["gen_ai.output.messages"]).toBeUndefined();
  },
);
