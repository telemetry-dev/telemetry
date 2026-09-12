import { propagation, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { afterEach, expect, test } from "vitest";

import {
  activeContext,
  extractW3cContext,
  flush,
  getTraceparent,
  injectW3cContext,
  shutdown,
  type SpanType,
  startActiveSpan,
  startSpan,
  updateActiveSpan,
  withContext,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

afterEach(async () => {
  await shutdown();
});

test("each span type maps to its gen_ai.operation.name", async () => {
  const { spans } = setup();

  const expected = {
    span: "function",
    generation: "chat",
    tool: "execute_tool",
    agent: "invoke_agent",
    embedding: "embeddings",
  } satisfies Record<SpanType, string>;

  for (const type of ["span", "generation", "tool", "agent", "embedding"] satisfies SpanType[]) {
    startSpan(type, { type }).end();
  }

  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported).toHaveLength(5);

  for (const [type, operation] of Object.entries(expected)) {
    const span = exported.find((s) => s.name === type)!;
    expect(span.attributes["gen_ai.operation.name"]).toBe(operation);
  }
});

test("kind can be overridden without changing defaults", async () => {
  const { spans } = setup();
  startSpan("client", { kind: SpanKind.CLIENT }).end();
  startSpan("server", { kind: SpanKind.SERVER }).end();
  startSpan("default").end();
  await flush();
  const exported = spans.getFinishedSpans();
  expect(exported.find((span) => span.name === "client")?.kind).toBe(SpanKind.CLIENT);
  expect(exported.find((span) => span.name === "client")?.attributes["gen_ai.operation.name"]).toBe(
    "function",
  );
  expect(exported.find((span) => span.name === "server")?.kind).toBe(SpanKind.SERVER);
  expect(exported.find((span) => span.name === "server")?.attributes["gen_ai.operation.name"]).toBe(
    "function",
  );
  expect(exported.find((span) => span.name === "default")?.kind).toBe(SpanKind.INTERNAL);
  expect(
    exported.find((span) => span.name === "default")?.attributes["gen_ai.operation.name"],
  ).toBe("function");
});

test("passes span links through unchanged", async () => {
  const { spans } = setup();

  const linked = {
    traceId: "0af7651916cd43dd8448eb211c80319c",
    spanId: "b7ad6b7169203331",
    traceFlags: 1,
  };

  startSpan("linked", { links: [{ context: linked, attributes: { reason: "ambient" } }] }).end();
  await flush();
  expect(spans.getFinishedSpans()[0]?.links).toEqual([
    { context: linked, attributes: { reason: "ambient" } },
  ]);
  expect(trace.isSpanContextValid(linked)).toBe(true);
});

test("generation fields map to gen_ai.* attributes", async () => {
  const { spans } = setup();
  startSpan("gen", {
    type: "generation",
    model: "gpt-4o",
    provider: "openai",
    systemInstructions: "be helpful",
    responseModel: "gpt-4o-2024-11-20",
    responseId: "resp_1",
    input: [{ role: "user", content: "hi" }],
    output: { role: "assistant", content: "hello" },
    usage: {
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
      cacheReadInputTokens: 3,
      cacheCreationInputTokens: 2,
      reasoningOutputTokens: 4,
    },
    costUsd: 0.0123,
    finishReason: "stop",
    outputType: "text",
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    maxTokens: 256,
    stopSequences: ["END"],
    seed: 7,
    frequencyPenalty: 0.1,
    presencePenalty: 0.2,
    timeToFirstChunkMs: 1234,
    metadata: { plan: "pro", nested: { a: 1 } },
    attributes: { "custom.raw": "kept" },
  }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  const a = span.attributes;
  expect(a["gen_ai.operation.name"]).toBe("chat");
  expect(a["gen_ai.request.model"]).toBe("gpt-4o");
  expect(a["gen_ai.provider.name"]).toBe("openai");
  expect(a["gen_ai.system_instructions"]).toBe("be helpful");
  expect(a["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(a["gen_ai.response.id"]).toBe("resp_1");
  expect(a["gen_ai.input.messages"]).toBe(JSON.stringify([{ role: "user", content: "hi" }]));
  expect(a["gen_ai.output.messages"]).toBe(JSON.stringify({ role: "assistant", content: "hello" }));
  expect(a["gen_ai.usage.input_tokens"]).toBe(10);
  expect(a["gen_ai.usage.output_tokens"]).toBe(5);
  expect(a["gen_ai.usage.total_tokens"]).toBe(15);
  expect(a["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(a["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(a["gen_ai.usage.reasoning.output_tokens"]).toBe(4);
  expect(a["gen_ai.usage.cost"]).toBe(0.0123);
  expect(a["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(a["gen_ai.output.type"]).toBe("text");
  expect(a["gen_ai.request.temperature"]).toBe(0.7);
  expect(a["gen_ai.request.top_p"]).toBe(0.9);
  expect(a["gen_ai.request.top_k"]).toBe(40);
  expect(a["gen_ai.request.max_tokens"]).toBe(256);
  expect(a["gen_ai.request.stop_sequences"]).toEqual(["END"]);
  expect(a["gen_ai.request.seed"]).toBe(7);
  expect(a["gen_ai.request.frequency_penalty"]).toBe(0.1);
  expect(a["gen_ai.request.presence_penalty"]).toBe(0.2);
  // milliseconds in, seconds on the wire
  expect(a["gen_ai.response.time_to_first_chunk"]).toBeCloseTo(1.234);
  expect(a["td.metadata.plan"]).toBe("pro");
  expect(a["td.metadata.nested"]).toBe(JSON.stringify({ a: 1 }));
  expect(a["custom.raw"]).toBe("kept");
});

test("tool spans capture input/output under tool call keys", async () => {
  const { spans } = setup();
  startSpan("search", {
    type: "tool",
    toolName: "web-search",
    toolCallId: "call_1",
    toolDescription: "searches the web",
    input: { query: "otel" },
    output: ["result"],
  }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.tool.name"]).toBe("web-search");
  expect(span.attributes["gen_ai.tool.call.id"]).toBe("call_1");
  expect(span.attributes["gen_ai.tool.description"]).toBe("searches the web");
  expect(span.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ query: "otel" }));
  expect(span.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify(["result"]));
  expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
});

test("agent fields map and update() applies later fields", async () => {
  const { spans } = setup();
  const handle = startSpan("agent-run", { type: "agent", agentName: "support" });
  handle.update({ agentId: "agent_7", output: "done" });
  handle.end({ usage: { inputTokens: 1 } });
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(span.attributes["gen_ai.agent.name"]).toBe("support");
  expect(span.attributes["gen_ai.agent.id"]).toBe("agent_7");
  expect(span.attributes["gen_ai.output.messages"]).toBe("done");
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(1);
});

test("parent accepts a W3C traceparent string and joins the trace", async () => {
  const { spans } = setup();
  const remoteTraceId = "0af7651916cd43dd8448eb211c80319c";
  const remoteSpanId = "b7ad6b7169203331";
  startSpan("joined", { parent: `00-${remoteTraceId}-${remoteSpanId}-01` }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.spanContext().traceId).toBe(remoteTraceId);
  expect(span.parentSpanContext?.spanId).toBe(remoteSpanId);
});

test("an invalid traceparent string falls back to the active context", async () => {
  const { spans } = setup();
  startActiveSpan("active-root", (root) => {
    startSpan("fallback", { parent: "garbage" }).end();
    void root;
  });
  await flush();
  const exported = spans.getFinishedSpans();
  const rootSpan = exported.find((s) => s.name === "active-root")!;
  const fallback = exported.find((s) => s.name === "fallback")!;
  // Falls back to the active span — same trace, parented to it — instead of a new root.
  expect(fallback.spanContext().traceId).toBe(rootSpan.spanContext().traceId);
  expect(fallback.parentSpanContext?.spanId).toBe(rootSpan.spanContext().spanId);
});

test("handle.traceparent and getTraceparent round-trip", async () => {
  setup();
  const handle = startSpan("outer");
  expect(handle.traceparent).toBe(`00-${handle.traceId}-${handle.spanId}-01`);
  startActiveSpan("inner", (inner) => {
    expect(getTraceparent()).toBe(inner.traceparent);
  });
  handle.end();
  expect(getTraceparent()).toBeNull();
});

test("W3C propagation omits baggage unless enabled", () => {
  setup();
  let parentTraceId = "";

  const carrier = {
    traceparent: undefined as string | undefined,
    tracestate: undefined as string | undefined,
    baggage: "tenant=stale" as string | undefined,
    Baggage: "tenant=uppercase" as string | undefined,
  };

  const baggageCarrier = { ...carrier };
  startActiveSpan("parent", (parent) => {
    parentTraceId = parent.traceId;
    const baggage = propagation.createBaggage({ tenant: { value: "acme" } });
    withContext(propagation.setBaggage(activeContext(), baggage), () => {
      injectW3cContext(activeContext(), carrier);
      injectW3cContext(activeContext(), baggageCarrier, { includeBaggage: true });
    });
  });

  expect(carrier.traceparent).toBeTypeOf("string");
  expect(carrier.baggage).toBeUndefined();
  expect(carrier.Baggage).toBeUndefined();
  expect(baggageCarrier.baggage).toBe("tenant=acme");
  expect(baggageCarrier.Baggage).toBeUndefined();
  const remoteCarrier = { ...carrier, baggage: "tenant=remote" };
  const extracted = extractW3cContext(remoteCarrier);
  expect(trace.getSpanContext(extracted)?.traceId).toBe(parentTraceId);
  expect(propagation.getBaggage(extracted)).toBeUndefined();
  expect(
    propagation
      .getBaggage(extractW3cContext(remoteCarrier, { includeBaggage: true }))
      ?.getEntry("tenant")?.value,
  ).toBe("remote");

  const staleCarrier = {
    TraceParent: "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
    TRACESTATE: "vendor=stale",
    Baggage: "tenant=stale",
  };

  injectW3cContext(extractW3cContext({}), staleCarrier);
  expect(staleCarrier).toEqual({});

  const sealedCarrier = {};
  Object.defineProperty(sealedCarrier, "TraceParent", {
    configurable: false,
    enumerable: true,
    value: "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
  });
  expect(() => injectW3cContext(extractW3cContext({}), sealedCarrier)).toThrow(
    "Cannot delete stale W3C carrier property: TraceParent",
  );
});

test("W3C extraction does not inherit ambient context", () => {
  setup();
  const remoteTraceId = "0af7651916cd43dd8448eb211c80319c";
  const remoteSpanId = "b7ad6b7169203331";
  const remoteCarrier = { traceparent: `00-${remoteTraceId}-${remoteSpanId}-01` };

  startActiveSpan("ambient", () => {
    const baggage = propagation.createBaggage({ tenant: { value: "local" } });
    withContext(propagation.setBaggage(activeContext(), baggage), () => {
      const extracted = extractW3cContext(remoteCarrier);
      expect(trace.getSpanContext(extracted)?.traceId).toBe(remoteTraceId);
      expect(propagation.getBaggage(extracted)).toBeUndefined();

      const missing = extractW3cContext({}, { includeBaggage: true });
      expect(trace.getSpanContext(missing)).toBeUndefined();
      expect(propagation.getBaggage(missing)).toBeUndefined();
    });
  });
});

test("startSpan does not activate context; startActiveSpan does", async () => {
  const { spans } = setup();
  const detached = startSpan("detached");
  // A span started while `detached` exists does NOT parent to it.
  startSpan("sibling").end();
  startActiveSpan("active-parent", () => {
    startSpan("child").end();
  });
  detached.end();
  await flush();
  const exported = spans.getFinishedSpans();
  const sibling = exported.find((s) => s.name === "sibling")!;
  const parent = exported.find((s) => s.name === "active-parent")!;
  const child = exported.find((s) => s.name === "child")!;
  expect(sibling.parentSpanContext).toBeUndefined();
  expect(child.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
  expect(child.spanContext().traceId).toBe(parent.spanContext().traceId);
});

test("manual parenting via handle context", async () => {
  const { spans } = setup();
  const parent = startSpan("manual-parent");
  startSpan("manual-child", { parent }).end();
  parent.end();
  await flush();
  const exported = spans.getFinishedSpans();
  const child = exported.find((s) => s.name === "manual-child")!;
  expect(child.parentSpanContext?.spanId).toBe(parent.spanId);
});

test("error capture: status, error.type, exception event", async () => {
  const { spans } = setup();
  const failure = new TypeError("boom");
  startSpan("fails").end({ error: failure });
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("TypeError");
  const event = span.events.find((e) => e.name === "exception")!;
  expect(event.attributes?.["exception.type"]).toBe("TypeError");
  expect(event.attributes?.["exception.message"]).toBe("boom");
  expect(event.attributes?.["log.severity_number"]).toBe(17);
});

test("async startActiveSpan ends after the promise settles and records rejections", async () => {
  const { spans } = setup();
  await expect(
    startActiveSpan("async-fail", async () => {
      await Promise.resolve();
      throw new Error("nope");
    }),
  ).rejects.toThrow("nope");
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.events.some((e) => e.name === "exception")).toBe(true);
});

test("updateActiveSpan maps fields using the active span's type", async () => {
  const { spans } = setup();
  startActiveSpan("gen", { type: "generation" }, () => {
    updateActiveSpan({ output: { content: "streamed" }, usage: { outputTokens: 9 } });
  });
  updateActiveSpan({ output: "ignored outside spans" });
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.output.messages"]).toBe(JSON.stringify({ content: "streamed" }));
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(9);
});

test("spanLimits backstop truncates raw attributes set outside the capture funnel", async () => {
  const { spans } = setup();
  const handle = startSpan("raw");
  handle.span.setAttribute("raw.dump", "x".repeat(70000));
  handle.end();
  await flush();
  const value = String(spans.getFinishedSpans()[0]!.attributes["raw.dump"]);
  expect(value.length).toBe(65536);
});

test("per-call captureInput:false drops input but keeps output", async () => {
  const { spans } = setup();
  startSpan("private", {
    type: "generation",
    input: "secret prompt",
    output: "public answer",
    captureInput: false,
  }).end();
  await flush();
  const span = spans.getFinishedSpans()[0]!;
  expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(span.attributes["gen_ai.output.messages"]).toBe("public answer");
});

test("rename via update({name})", async () => {
  const { spans } = setup();
  startSpan("old-name").end({ name: "new-name" });
  await flush();
  expect(spans.getFinishedSpans()[0]!.name).toBe("new-name");
});
