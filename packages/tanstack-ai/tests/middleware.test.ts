import { type Attributes, ROOT_CONTEXT, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import {
  AlwaysOffSampler,
  AlwaysOnSampler,
  ParentBasedSampler,
  type ReadableSpan,
  type Sampler,
  SamplingDecision,
  TraceIdRatioBasedSampler,
} from "@opentelemetry/sdk-trace-base";
import { type ChatMiddlewareContext } from "@tanstack/ai";
import { sessionSpanContext } from "@telemetry-dev/otel";
import { expect, test } from "vitest";

import { telemetryDev } from "../src/index.ts";

type TestValue =
  | string
  | number
  | bigint
  | boolean
  | null
  | TestValue[]
  | { [key: string]: TestValue };

interface MetricRecord {
  metric: "duration" | "tokens";
  tokenType?: "input" | "output";
  value: number;
  attributes: Attributes;
}

// Inject a capturing transport via the otel.ts test seam: spans are captured as the real
// ReadableSpans the emitter would serialize, and metric points are captured in-process.
function makeCapture() {
  const spanBatches: ReadableSpan[][] = [];
  const metrics: MetricRecord[] = [];
  const overrides = {
    sendSpans: async (spans: ReadableSpan[]) => {
      spanBatches.push(spans);
    },
    recordDuration: (value: number, attributes: Attributes) => {
      metrics.push({ metric: "duration", value, attributes });
    },
    recordTokens: (tokenType: "input" | "output", value: number, attributes: Attributes) => {
      metrics.push({ metric: "tokens", tokenType, value, attributes });
    },
  };
  return { spanBatches, metrics, overrides };
}

// Mutable stand-in for the engine's per-chat middleware context. The middleware only reads
// identity fields, `phase`, `options.metadata`, `modelOptions`, and `accumulatedContent`.
function makeCtx(over?: Record<string, TestValue>): ChatMiddlewareContext {
  return {
    requestId: "req_1",
    streamId: "stream_1",
    runId: "run_1",
    threadId: "thread_1",
    phase: "beforeModel",
    iteration: 0,
    chunkIndex: 0,
    provider: "openai",
    model: "gpt-4o",
    signal: undefined,
    abort: () => {},
    context: undefined,
    source: "server",
    streaming: true,
    systemPrompts: [],
    options: undefined,
    modelOptions: undefined,
    messages: [],
    messageCount: 0,
    hasTools: false,
    currentMessageId: null,
    accumulatedContent: "",
    createId: (prefix) => `${prefix}_1`,
    defer: () => {},
    ...over,
  } as ChatMiddlewareContext;
}

const chatConfig = (over?: Record<string, TestValue>) =>
  ({
    messages: [{ role: "user", content: "hi" }],
    systemPrompts: [],
    tools: [],
    ...over,
  }) as never;

const spanId = (s: ReadableSpan) => s.spanContext().spanId;
const traceId = (s: ReadableSpan) => s.spanContext().traceId;
const byOperation = (spans: ReadableSpan[], operation: string) =>
  spans.filter((s) => s.attributes["gen_ai.operation.name"] === operation);
const event = (s: ReadableSpan, name: string) => s.events.find((ev) => ev.name === name);

const middleware = (overrides: ReturnType<typeof makeCapture>["overrides"]) =>
  telemetryDev({ apiKey: "td_live_test", environment: "test", serviceName: "svc" }, overrides);

test("single-iteration chat emits a root + CLIENT iteration with all five token kinds and cost", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ options: { metadata: { userId: "u1", feature: "support" } } });

  await mw.onStart?.(ctx);
  await mw.onConfig?.(
    ctx,
    chatConfig({
      systemPrompts: ["be helpful"],
      modelOptions: { temperature: 0.7, top_p: 0.9, max_output_tokens: 256 },
    }),
  );
  (ctx as { accumulatedContent: string }).accumulatedContent = "hello there";
  await mw.onChunk?.(ctx, {
    type: "RUN_FINISHED",
    finishReason: "stop",
    model: "gpt-4o-2024-11-20",
  } as never);
  await mw.onUsage?.(ctx, {
    promptTokens: 10,
    completionTokens: 5,
    totalTokens: 15,
    promptTokensDetails: { cachedTokens: 3, cacheWriteTokens: 2 },
    completionTokensDetails: { reasoningTokens: 4 },
    cost: 0.012,
  } as never);
  await mw.onFinish?.(ctx, {
    finishReason: "stop",
    duration: 120,
    content: "hello there",
  } as never);

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;

  // Resource + scope ride along on every span (proves they serialize as gen_ai OTLP).
  expect(root.resource.attributes["service.name"]).toBe("svc");
  expect(root.resource.attributes["deployment.environment.name"]).toBe("test");
  expect(root.instrumentationScope.name).toBe("@telemetry-dev/tanstack-ai");

  expect(root.name).toBe("chat");
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(root.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(root.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(root.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(root.attributes["user.id"]).toBe("u1");
  expect(root.attributes["td.metadata.feature"]).toBe("support");
  expect(root.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(root.attributes["gen_ai.output.messages"]).toBe("hello there");
  expect(root.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(root.status.code).toBe(SpanStatusCode.UNSET);
  expect(root.parentSpanContext?.spanId).toBe(
    sessionSpanContext("td_live_test", "thread_1").spanId,
  );
  // Rolled-up usage lives on the summary event only — root usage attrs would double-count in
  // the ingest's per-trace sum.
  expect(root.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(root.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();

  expect(iteration.name).toBe("chat");
  expect(iteration.parentSpanContext?.spanId).toBe(spanId(root));
  expect(traceId(iteration)).toBe(traceId(root));
  expect(iteration.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(iteration.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(iteration.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(iteration.attributes["gen_ai.request.max_tokens"]).toBe(256);
  expect(JSON.parse(String(iteration.attributes["gen_ai.input.messages"]))).toEqual([
    { role: "system", content: "be helpful" },
    { role: "user", content: "hi" },
  ]);
  expect(iteration.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(iteration.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(iteration.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(iteration.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(iteration.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(4);
  expect(iteration.attributes["gen_ai.usage.cost"]).toBe(0.012);
  expect(iteration.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(iteration.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(iteration.attributes["gen_ai.output.messages"]).toBe("hello there");

  // No explicit sessionId: conversation.id falls back to the chat's threadId.
  for (const s of spans) {
    expect(s.attributes["gen_ai.conversation.id"]).toBe("thread_1");
  }

  const durations = metrics.filter((m) => m.metric === "duration");
  const tokens = metrics.filter((m) => m.metric === "tokens");
  expect(durations).toHaveLength(1);
  expect(tokens).toHaveLength(2);
  expect(durations[0]!.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(durations[0]!.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(durations[0]!.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(durations[0]!.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(tokens.find((m) => m.tokenType === "input")?.value).toBe(10);
  expect(tokens.find((m) => m.tokenType === "output")?.value).toBe(5);

  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["log.severity_number"]).toBe(9);
  expect(String(summary?.attributes?.["log.message"])).toContain("10 in / 5 out");
  expect(summary?.attributes?.["gen_ai.usage.input_tokens"]).toBe(10);
  expect(summary?.attributes?.["gen_ai.usage.output_tokens"]).toBe(5);
});

test("bigint metadata IDs populate user and conversation attributes", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ options: { metadata: { userId: 2n, sessionId: 9n } } });

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  const spans = spanBatches[0]!;
  for (const s of spans) {
    expect(s.attributes["gen_ai.conversation.id"]).toBe("9");
  }
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  expect(root.attributes["user.id"]).toBe("2");
  // Reserved keys never leak into td.metadata.*.
  expect(root.attributes["td.metadata.userId"]).toBeUndefined();
  expect(root.attributes["td.metadata.sessionId"]).toBeUndefined();
});

test("string metadata IDs override threadId and populate user attributes", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ options: { metadata: { userId: "u2", sessionId: "sess_9" } } });

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  const spans = spanBatches[0]!;
  for (const span of spans) {
    expect(span.attributes["gen_ai.conversation.id"]).toBe("sess_9");
  }
  const root = spans.find((span) => span.kind === SpanKind.INTERNAL)!;
  expect(root.attributes["user.id"]).toBe("u2");
});

test("multi-iteration run with a tool yields invoke_agent root and parents the tool to its iteration", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ hasTools: true });

  await mw.onStart?.(ctx);
  // Iteration 0: model asks for a tool.
  await mw.onConfig?.(ctx, chatConfig({ messages: [{ role: "user", content: "weather?" }] }));
  await mw.onChunk?.(ctx, {
    type: "RUN_FINISHED",
    finishReason: "tool_calls",
    model: "gpt-4o",
  } as never);
  await mw.onUsage?.(ctx, { promptTokens: 8, completionTokens: 2, totalTokens: 10 } as never);
  await mw.onBeforeToolCall?.(ctx, {
    toolName: "getWeather",
    toolCallId: "call_1",
    args: { city: "SF" },
  } as never);
  await mw.onAfterToolCall?.(ctx, {
    toolName: "getWeather",
    toolCallId: "call_1",
    ok: true,
    duration: 42,
    result: { tempF: 70 },
  } as never);
  // Iteration 1: final answer. Opening it closes iteration 0's span.
  (ctx as { iteration: number }).iteration = 1;
  await mw.onConfig?.(ctx, chatConfig({ messages: [{ role: "user", content: "weather?" }] }));
  (ctx as { accumulatedContent: string }).accumulatedContent = "It is 70F in SF.";
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop", model: "gpt-4o" } as never);
  await mw.onUsage?.(ctx, { promptTokens: 12, completionTokens: 6, totalTokens: 18 } as never);
  await mw.onFinish?.(ctx, {
    finishReason: "stop",
    duration: 90,
    content: "It is 70F in SF.",
  } as never);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(4);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const iterations = byOperation(spans, "chat").filter((s) => s.kind === SpanKind.CLIENT);
  const tool = byOperation(spans, "execute_tool")[0]!;
  expect(iterations).toHaveLength(2);

  // Tools push the root operation to invoke_agent.
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");

  // The tool span is parented to iteration 0 (the one whose model call requested it).
  const iteration0 = iterations.find((s) => s.attributes["gen_ai.usage.input_tokens"] === 8)!;
  expect(tool.parentSpanContext?.spanId).toBe(spanId(iteration0));
  expect(tool.attributes["gen_ai.tool.name"]).toBe("getWeather");
  expect(tool.attributes["gen_ai.tool.call.id"]).toBe("call_1");
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ city: "SF" }));
  expect(tool.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ tempF: 70 }));
  expect(tool.status.code).toBe(SpanStatusCode.UNSET);
  expect(iteration0.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);

  const durations = metrics.filter((m) => m.metric === "duration");
  expect(durations.filter((m) => m.attributes["gen_ai.operation.name"] === "chat")).toHaveLength(2);
  expect(
    durations.filter((m) => m.attributes["gen_ai.operation.name"] === "execute_tool"),
  ).toHaveLength(1);
  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(4);
});

test("a failed tool call records ERROR status and an exception event", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onBeforeToolCall?.(ctx, { toolName: "t1", toolCallId: "a", args: { n: 1 } } as never);
  await mw.onBeforeToolCall?.(ctx, { toolName: "t2", toolCallId: "b", args: { n: 2 } } as never);
  await mw.onAfterToolCall?.(ctx, {
    toolName: "t1",
    toolCallId: "a",
    ok: true,
    duration: 10,
    result: "ra",
  } as never);
  await mw.onAfterToolCall?.(ctx, {
    toolName: "t2",
    toolCallId: "b",
    ok: false,
    duration: 20,
    error: new Error("kaboom"),
  } as never);
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 50, content: "done" } as never);

  const spans = spanBatches[0]!;
  const tools = byOperation(spans, "execute_tool");
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(tools).toHaveLength(2);
  for (const t of tools) {
    expect(t.parentSpanContext?.spanId).toBe(spanId(iteration));
  }
  const failed = tools.find((t) => t.attributes["gen_ai.tool.name"] === "t2")!;
  expect(failed.status.code).toBe(SpanStatusCode.ERROR);
  expect(failed.attributes["error.type"]).toBe("Error");
  expect(failed.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  const exc = event(failed, "exception");
  expect(exc?.attributes?.["exception.message"]).toBe("kaboom");
  expect(exc?.attributes?.["log.severity_number"]).toBe(17);
});

test("onError marks open iteration and root ERROR and still flushes the batch", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onError?.(ctx, { error: new TypeError("boom"), duration: 30 } as never);

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(root.attributes["error.type"]).toBe("TypeError");
  expect(iteration.status.code).toBe(SpanStatusCode.ERROR);
  expect(iteration.attributes["error.type"]).toBe("TypeError");
  const exc = event(root, "exception");
  expect(exc?.attributes?.["exception.message"]).toBe("boom");
  expect(exc?.attributes?.["log.severity_number"]).toBe(17);
  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["log.severity_number"]).toBe(17);
});

test("onAbort closes everything as cancelled", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onBeforeToolCall?.(ctx, { toolName: "slow", toolCallId: "c", args: {} } as never);
  await mw.onAbort?.(ctx, { reason: "user navigated away", duration: 15 } as never);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(3);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(root.attributes["error.type"]).toBe("cancelled");
  expect(root.attributes["gen_ai.response.finish_reasons"]).toEqual(["cancelled"]);
  const tool = byOperation(spans, "execute_tool")[0]!;
  expect(tool.status.code).toBe(SpanStatusCode.ERROR);
  expect(tool.attributes["error.type"]).toBe("cancelled");
});

test("no apiKey is a complete no-op (transport never invoked)", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = telemetryDev({ environment: "test", serviceName: "svc" }, overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "y" } as never);

  expect(spanBatches).toHaveLength(0);
  expect(metrics).toHaveLength(0);
});

test("one middleware instance handles concurrent chats independently", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctxA = makeCtx({ threadId: "thread_a" });
  const ctxB = makeCtx({ threadId: "thread_b", model: "gpt-4o-mini" });

  // Interleave two runs through the SAME instance: per-ctx WeakMap state keeps them apart.
  await mw.onStart?.(ctxA);
  await mw.onStart?.(ctxB);
  await mw.onConfig?.(ctxA, chatConfig());
  await mw.onConfig?.(ctxB, chatConfig());
  await mw.onChunk?.(ctxA, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onUsage?.(ctxA, { promptTokens: 1, completionTokens: 1, totalTokens: 2 } as never);
  await mw.onChunk?.(ctxB, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onUsage?.(ctxB, { promptTokens: 7, completionTokens: 3, totalTokens: 10 } as never);
  await mw.onFinish?.(ctxA, { finishReason: "stop", duration: 5, content: "a" } as never);
  await mw.onFinish?.(ctxB, { finishReason: "stop", duration: 5, content: "b" } as never);

  expect(spanBatches).toHaveLength(2);
  const [batchA, batchB] = [spanBatches[0]!, spanBatches[1]!];
  expect(traceId(batchA[0]!)).not.toBe(traceId(batchB[0]!));
  const rootA = batchA.find((s) => s.kind === SpanKind.INTERNAL)!;
  const rootB = batchB.find((s) => s.kind === SpanKind.INTERNAL)!;
  expect(rootA.attributes["gen_ai.conversation.id"]).toBe("thread_a");
  expect(rootB.attributes["gen_ai.conversation.id"]).toBe("thread_b");
  expect(rootB.attributes["gen_ai.request.model"]).toBe("gpt-4o-mini");
  const iterB = batchB.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(iterB.attributes["gen_ai.usage.input_tokens"]).toBe(7);
});

test("chats on one thread share one trace under the session parent", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const run = async (ctx: ChatMiddlewareContext) => {
    await mw.onStart?.(ctx);
    await mw.onConfig?.(ctx, chatConfig());
    await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
    await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);
  };
  await run(makeCtx({ runId: "run_1" }));
  await run(makeCtx({ runId: "run_2" }));
  await run(makeCtx({ threadId: "thread_other" }));

  const session = sessionSpanContext("td_live_test", "thread_1");
  const [batch1, batch2, batch3] = spanBatches;
  for (const s of [...batch1!, ...batch2!]) expect(s.spanContext().traceId).toBe(session.traceId);
  for (const batch of [batch1!, batch2!]) {
    const root = batch.find((s) => s.kind === SpanKind.INTERNAL)!;
    expect(root.parentSpanContext?.spanId).toBe(session.spanId);
  }
  expect(batch3![0]!.spanContext().traceId).not.toBe(session.traceId);
});

test("export failure reaches onError even on the waitUntil path", async () => {
  const errors: unknown[] = [];
  let handed: Promise<unknown> | undefined;
  const mw = telemetryDev(
    {
      apiKey: "td_live_test",
      environment: "test",
      serviceName: "svc",
      waitUntil: (p) => {
        handed = p;
      },
      onError: (e) => {
        errors.push(e);
      },
    },
    {
      sendSpans: async () => {
        throw new Error("trace export boom");
      },
      recordDuration: () => {},
      recordTokens: () => {},
    },
  );
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  // waitUntil received the handed-off flush promise, which never rejects.
  expect(handed).toBeDefined();
  await expect(handed).resolves.toBeUndefined();
  // The trace-export rejection still surfaced through onError.
  expect(errors).toHaveLength(1);
  expect((errors[0] as Error).message).toBe("trace export boom");
});

test("metric export retries a transient ingest failure", async () => {
  const errors: unknown[] = [];
  let metricCalls = 0;
  const mw = telemetryDev({
    apiKey: "td_live_metric_retry",
    environment: "test",
    serviceName: "metric-retry",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (String(url).endsWith("/v1/metrics")) {
        metricCalls += 1;
        return new Response(null, { status: metricCalls === 1 ? 503 : 200 });
      }
      return new Response(null, { status: 200 });
    },
    onError: (e) => {
      errors.push(e);
    },
  });
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onUsage?.(ctx, { promptTokens: 10, completionTokens: 5, totalTokens: 15 } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  expect(metricCalls).toBe(2);
  expect(errors).toHaveLength(0);
});

test("Ollama-style nested sampling options populate gen_ai.request attrs", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(
    ctx,
    chatConfig({ modelOptions: { options: { temperature: 0.2, num_predict: 64 } } }),
  );
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  const iteration = spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(iteration.attributes["gen_ai.request.temperature"]).toBe(0.2);
  expect(iteration.attributes["gen_ai.request.max_tokens"]).toBe(64);
});

test("an iteration with no usage records no token attributes or metrics (not zeros)", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(0);
  expect(metrics.filter((m) => m.metric === "duration")).toHaveLength(1);

  const spans = spanBatches[0]!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(iteration.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(iteration.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(summary?.attributes?.["gen_ai.usage.output_tokens"]).toBeUndefined();
  expect(String(summary?.attributes?.["log.message"])).not.toContain("tokens");
});

test("onConfig during init opens no iteration span", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  // The engine may fire onConfig at init before any model call; init must not open an iteration span.
  Object.assign(ctx, { phase: "init" });
  await mw.onConfig?.(ctx, chatConfig());
  Object.assign(ctx, { phase: "beforeModel" });
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);
  expect(spans.filter((s) => s.kind === SpanKind.CLIENT)).toHaveLength(1);
});

test("structured output finalization records a separate iteration span", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  Object.assign(ctx, { accumulatedContent: "agent loop text" });
  await mw.onChunk?.(ctx, {
    type: "RUN_FINISHED",
    finishReason: "stop",
    model: "gpt-4o-2024-11-20",
  } as never);
  await mw.onUsage?.(ctx, {
    promptTokens: 10,
    completionTokens: 5,
    totalTokens: 15,
    cost: 0.01,
  } as never);

  Object.assign(ctx, { phase: "structuredOutput" });
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, {
    type: "CUSTOM",
    name: "structured-output.complete",
    value: { object: { a: 1 }, raw: '{"a":1}' },
  } as never);
  await mw.onUsage?.(ctx, { promptTokens: 7, completionTokens: 3, totalTokens: 10 } as never);
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 50, content: '{"a":1}' } as never);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(3);

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const iterations = byOperation(spans, "chat").filter((s) => s.kind === SpanKind.CLIENT);
  expect(iterations).toHaveLength(2);

  const agentIteration = iterations.find((s) => s.attributes["gen_ai.output.type"] === "text")!;
  expect(agentIteration.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(agentIteration.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(agentIteration.attributes["gen_ai.usage.cost"]).toBe(0.01);
  expect(agentIteration.attributes["gen_ai.output.messages"]).toBe("agent loop text");

  const structuredIteration = iterations.find(
    (s) => s.attributes["gen_ai.output.type"] === "json",
  )!;
  expect(structuredIteration.attributes["gen_ai.usage.input_tokens"]).toBe(7);
  expect(structuredIteration.attributes["gen_ai.usage.output_tokens"]).toBe(3);
  expect(structuredIteration.attributes["gen_ai.output.messages"]).toBe('{"a":1}');

  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["gen_ai.usage.input_tokens"]).toBe(17);
  expect(summary?.attributes?.["gen_ai.usage.output_tokens"]).toBe(8);

  const inputTokens = metrics
    .filter((m) => m.metric === "tokens" && m.tokenType === "input")
    .map((m) => m.value);
  expect(inputTokens).toEqual([10, 7]);
});

test("RUN_FINISHED usage is kept as a fallback when onUsage never fires", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, {
    type: "RUN_FINISHED",
    finishReason: "stop",
    usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
  } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);

  const iteration = spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(iteration.attributes["gen_ai.usage.input_tokens"]).toBe(4);
  expect(iteration.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(2);
});

test("tool wait finalization flushes an invoke_agent run once and ignores later finish", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  Object.assign(ctx, { accumulatedContent: "Waiting on external tools." });
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "tool_calls" } as never);
  await mw.onToolPhaseComplete?.(ctx, {
    toolCalls: [],
    results: [],
    needsApproval: [
      {
        toolCallId: "call_approve",
        toolName: "approveTransfer",
        input: { amount: 25 },
        approvalId: "approval_1",
      },
    ],
    needsClientExecution: [
      { toolCallId: "call_client", toolName: "clientLookup", input: { id: "cust_1" } },
    ],
  } as never);

  expect(spanBatches).toHaveLength(1);
  expect(metrics.filter((m) => m.metric === "duration")).toHaveLength(1);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.endTime[0]).toBeGreaterThan(0);
  expect(iteration.endTime[0]).toBeGreaterThan(0);
  expect(iteration.parentSpanContext?.spanId).toBe(spanId(root));
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(root.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);
  expect(root.attributes["gen_ai.output.messages"]).toBe("Waiting on external tools.");
  expect(root.status.code).toBe(SpanStatusCode.UNSET);

  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["log.severity_number"]).toBe(9);
  expect(String(summary?.attributes?.["log.message"])).toContain(
    "Generation paused awaiting tools",
  );
  expect(String(summary?.attributes?.["log.message"])).toContain("approveTransfer");
  expect(String(summary?.attributes?.["log.message"])).toContain("clientLookup");
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "late finish" } as never);
  expect(spanBatches).toHaveLength(1);
});

test("tool phase completion with no pending work is a no-op and later finish flushes normally", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onToolPhaseComplete?.(ctx, {
    toolCalls: [],
    results: [],
    needsApproval: [],
    needsClientExecution: [],
  } as never);
  expect(spanBatches).toHaveLength(0);

  Object.assign(ctx, { accumulatedContent: "normal final answer" });
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, {
    finishReason: "stop",
    duration: 5,
    content: "normal final answer",
  } as never);

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(root.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(root.attributes["gen_ai.output.messages"]).toBe("normal final answer");
  expect(iteration.attributes["gen_ai.output.messages"]).toBe("normal final answer");
  expect(String(event(root, "generation.summary")?.attributes?.["log.message"])).toContain(
    "Generation completed (stop)",
  );
});

test("tool wait finalization ends and flushes a dangling tool span", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ hasTools: true });

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "tool_calls" } as never);
  await mw.onBeforeToolCall?.(ctx, {
    toolName: "serverApprovalTool",
    toolCallId: "call_open",
    args: { id: "abc" },
  } as never);
  await mw.onToolPhaseComplete?.(ctx, {
    toolCalls: [],
    results: [],
    needsApproval: [
      {
        toolCallId: "call_waiting",
        toolName: "approveServerTool",
        input: { id: "abc" },
        approvalId: "approval_2",
      },
    ],
    needsClientExecution: [],
  } as never);

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(3);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const iteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  const tool = byOperation(spans, "execute_tool")[0]!;
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(tool.parentSpanContext?.spanId).toBe(spanId(iteration));
  expect(tool.attributes["gen_ai.tool.name"]).toBe("serverApprovalTool");
  expect(tool.attributes["gen_ai.tool.call.id"]).toBe("call_open");
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ id: "abc" }));
  expect(tool.status.code).toBe(SpanStatusCode.UNSET);
  expect(tool.endTime[0]).toBeGreaterThan(0);
});

test("structured output root prefers finalization JSON over stale finish content", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx();

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  Object.assign(ctx, { accumulatedContent: "agent loop text" });
  await mw.onChunk?.(ctx, {
    type: "RUN_FINISHED",
    finishReason: "stop",
    model: "gpt-4o-2024-11-20",
  } as never);

  Object.assign(ctx, { phase: "structuredOutput" });
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, {
    type: "CUSTOM",
    name: "structured-output.complete",
    value: { object: { final: true }, raw: '{"final":true}' },
  } as never);
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, {
    finishReason: "stop",
    duration: 50,
    content: "agent loop text",
  } as never);

  const root = spanBatches[0]!.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  expect(root.attributes["gen_ai.output.messages"]).toBe('{"final":true}');
  expect(root.attributes["gen_ai.output.messages"]).not.toBe("agent loop text");
});

test("structured output root records finalization JSON when finish content is empty", async () => {
  const { spanBatches, overrides } = makeCapture();
  const mw = middleware(overrides);
  const ctx = makeCtx({ phase: "structuredOutput" });

  await mw.onStart?.(ctx);
  await mw.onConfig?.(ctx, chatConfig());
  await mw.onChunk?.(ctx, {
    type: "CUSTOM",
    name: "structured-output.complete",
    value: { object: { skipped: true }, raw: '{"skipped":true}' },
  } as never);
  await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
  await mw.onFinish?.(ctx, { finishReason: "stop", duration: 50, content: "" } as never);

  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const structuredIteration = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(structuredIteration.attributes["gen_ai.output.type"]).toBe("json");
  expect(structuredIteration.attributes["gen_ai.output.messages"]).toBe('{"skipped":true}');
  expect(root.attributes["gen_ai.output.messages"]).toBe('{"skipped":true}');
});

test("session chats export spans only when the configured sampler selects them", async () => {
  const roots: Sampler[] = [
    new AlwaysOnSampler(),
    new AlwaysOffSampler(),
    new TraceIdRatioBasedSampler(0.5),
    { shouldSample: () => ({ decision: SamplingDecision.RECORD }), toString: () => "RecordOnly" },
  ];
  for (const root of roots) {
    const sampler = new ParentBasedSampler({ root });
    const { spanBatches, overrides } = makeCapture();
    const errors: unknown[] = [];
    const mw = telemetryDev(
      {
        apiKey: "td_live_test",
        serviceName: "sampling",
        sampler,
        onError: (error) => errors.push(error),
      },
      overrides,
    );
    const expected: string[] = [];
    for (let i = 0; i < 20; i++) {
      const threadId = `session-${i}`;
      const id = sessionSpanContext("td_live_test", threadId).traceId;
      if (
        sampler.shouldSample(ROOT_CONTEXT, id, "chat", SpanKind.INTERNAL, {}, []).decision ===
        SamplingDecision.RECORD_AND_SAMPLED
      )
        expected.push(id, id);
      const ctx = makeCtx({ threadId });
      await mw.onStart?.(ctx);
      await mw.onConfig?.(ctx, chatConfig());
      await mw.onChunk?.(ctx, { type: "RUN_FINISHED", finishReason: "stop" } as never);
      await mw.onFinish?.(ctx, { finishReason: "stop", duration: 5, content: "ok" } as never);
    }
    expect(spanBatches.flat().map(traceId)).toEqual(expected);
    expect(errors).toEqual([]);
  }
});
