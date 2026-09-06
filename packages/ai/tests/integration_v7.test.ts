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
import { sessionSpanContext } from "@telemetry-dev/otel";
import { expect, test } from "vitest";

import { telemetryDev } from "../src/index.ts";

interface MetricRecord {
  metric: "duration" | "tokens";
  tokenType?: "input" | "output";
  value: number;
  attributes: Attributes;
}

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

const model = { provider: "openai", modelId: "gpt-4o" } as const;
const baseOptions = { apiKey: "td_live_test", environment: "test", serviceName: "svc" } as const;

const spanId = (s: ReadableSpan) => s.spanContext().spanId;
const traceId = (s: ReadableSpan) => s.spanContext().traceId;
const byOperation = (spans: ReadableSpan[], operation: string) =>
  spans.filter((s) => s.attributes["gen_ai.operation.name"] === operation);
const event = (s: ReadableSpan, name: string) => s.events.find((ev) => ev.name === name);

test("single-step generateText flushes root+child with aggregated onEnd usage and responseTime duration", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-text-1";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    functionId: "answer",
    instructions: "be helpful",
    messages: [{ role: "user", content: "hi" }],
    runtimeContext: { userId: "u1", sessionId: "s1", feature: "chat" },
    temperature: 0.7,
    topP: 0.9,
    maxOutputTokens: 512,
  });
  integ.onStepStart?.({
    callId,
    stepNumber: 0,
    messages: [{ role: "user", content: "hi" }],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "hello there",
    finishReason: "stop",
    response: { id: "resp_1", modelId: "gpt-4o-2024-11-20" },
    usage: {
      inputTokens: 10,
      outputTokens: 5,
      inputTokenDetails: { cacheReadTokens: 3, cacheWriteTokens: 2 },
      outputTokenDetails: { reasoningTokens: 4 },
    },
    performance: { responseTimeMs: 2500 },
  });
  await integ.onEnd?.({
    callId,
    text: "hello there",
    finishReason: "stop",
    usage: { inputTokens: 100, outputTokens: 50 },
  });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const child = spans.find((s) => s.kind === SpanKind.CLIENT)!;

  expect(root.name).toBe("answer");
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(root.attributes["user.id"]).toBe("u1");
  expect(root.attributes["gen_ai.conversation.id"]).toBe("s1");
  expect(root.attributes["td.metadata.feature"]).toBe("chat");
  expect(root.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(root.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(root.attributes["gen_ai.request.max_tokens"]).toBe(512);
  expect(child.parentSpanContext?.spanId).toBe(spanId(root));
  expect(child.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(child.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(child.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(child.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(child.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(4);

  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["gen_ai.usage.input_tokens"]).toBe(100);
  expect(summary?.attributes?.["gen_ai.usage.output_tokens"]).toBe(50);
  expect(String(summary?.attributes?.["log.message"])).toContain("100 in / 50 out");
  expect(String(summary?.attributes?.["log.message"])).not.toContain("10 in / 5 out");

  const duration = metrics.find((m) => m.metric === "duration")!;
  expect(duration.value).toBe(2.5);
  expect(duration.attributes["gen_ai.operation.name"]).toBe("chat");
});

test("runtimeContext applies identity metadata to the root and conversation id to child spans only", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-runtime";
  const toolCall = { toolCallId: "rt-tool", toolName: "lookup", input: { q: "x" } };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    functionId: "runtime-test",
    messages: [{ role: "user", content: "hi" }],
    runtimeContext: { userId: "user-123", sessionId: "session-456", plan: "pro" },
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall,
    toolExecutionMs: 25,
    toolOutput: { type: "tool-result", output: { ok: true } },
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "done",
    finishReason: "stop",
    response: { id: "runtime-response", modelId: "gpt-4o" },
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.name === "runtime-test")!;
  const step = byOperation(spans, "chat").find((s) => s.kind === SpanKind.CLIENT)!;
  const tool = byOperation(spans, "execute_tool")[0]!;

  expect(root.attributes["user.id"]).toBe("user-123");
  expect(root.attributes["gen_ai.conversation.id"]).toBe("session-456");
  expect(root.attributes["td.metadata.plan"]).toBe("pro");

  for (const child of [step, tool]) {
    expect(child.parentSpanContext?.spanId).toBe(child === step ? spanId(root) : spanId(step));
    expect(child.attributes["gen_ai.conversation.id"]).toBe("session-456");
    expect(child.attributes["user.id"]).toBeUndefined();
    expect(child.attributes["td.metadata.plan"]).toBeUndefined();
  }
});

test("runtimeContext refreshes user metadata but keeps the starting session", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-runtime-refresh";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    functionId: "runtime-refresh-test",
    messages: [{ role: "user", content: "hi" }],
    runtimeContext: { userId: "user-start", sessionId: "session-start", plan: "starter" },
  });
  integ.onStepStart?.({
    callId,
    stepNumber: 0,
    runtimeContext: { userId: "user-step", sessionId: "session-step", plan: "pro" },
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "done",
    finishReason: "stop",
    response: { id: "runtime-refresh-response", modelId: "gpt-4o" },
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.name === "runtime-refresh-test")!;
  const session = sessionSpanContext("td_live_test", "session-start");

  expect(root.parentSpanContext?.spanId).toBe(session.spanId);
  for (const span of spans) {
    expect(traceId(span)).toBe(session.traceId);
    expect(span.attributes["gen_ai.conversation.id"]).toBe("session-start");
  }
  expect(root.attributes["user.id"]).toBe("user-step");
  expect(root.attributes["td.metadata.plan"]).toBe("pro");
});

test("missing or empty runtimeContext emits no identity or metadata attributes", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const cases = [
    { callId: "call-runtime-absent", runtimeContext: undefined },
    { callId: "call-runtime-empty", runtimeContext: {} },
  ] as const;

  for (const { callId, runtimeContext } of cases) {
    const toolCall = { toolCallId: `${callId}-tool`, toolName: "lookup", input: { q: callId } };
    const startEvent = {
      callId,
      operationId: "ai.generateText",
      provider: "openai",
      modelId: "gpt-4o",
      messages: [{ role: "user", content: callId }],
      runtimeContext,
    };
    integ.onStart?.(startEvent);
    integ.onStepStart?.({ callId, stepNumber: 0 });
    integ.onToolExecutionEnd?.({
      callId,
      toolCall,
      toolExecutionMs: 10,
      toolOutput: { type: "tool-result", output: { ok: true } },
    });
    integ.onStepEnd?.({
      callId,
      stepNumber: 0,
      model,
      text: "done",
      finishReason: "stop",
      response: { id: `${callId}-response`, modelId: "gpt-4o" },
      usage: { inputTokens: 1, outputTokens: 1 },
    });
    await integ.onEnd?.({
      callId,
      text: "done",
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  }

  expect(spanBatches).toHaveLength(2);
  for (const spans of spanBatches) {
    expect(spans).toHaveLength(3);
    for (const span of spans) {
      expect(span.attributes["user.id"]).toBeUndefined();
      expect(span.attributes["gen_ai.conversation.id"]).toBeUndefined();
      expect(Object.keys(span.attributes).some((key) => key.startsWith("td.metadata."))).toBe(
        false,
      );
    }
  }
});

test("onStepFinish is not implemented, so ai@7's step-end fan-out cannot double spans", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-dedupe";
  const stepResult = {
    callId,
    stepNumber: 0,
    model,
    text: "once",
    finishReason: "stop",
    response: { id: "r1", modelId: "gpt-4o" },
    usage: { inputTokens: 4, outputTokens: 2 },
    performance: { responseTimeMs: 1000 },
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages: [{ role: "user", content: "x" }],
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onStepEnd?.(stepResult);
  // ai@7 fans step-end out to BOTH onStepEnd and the deprecated onStepFinish, calling each
  // implemented hook once. The integration must not declare onStepFinish, or every step would be
  // recorded twice.
  expect(Object.keys(integ)).not.toContain("onStepFinish");
  await integ.onEnd?.({
    callId,
    text: "once",
    finishReason: "stop",
    usage: stepResult.usage,
  });

  expect(spanBatches).toHaveLength(1);
  expect(
    byOperation(spanBatches[0]!, "chat").filter((s) => s.kind === SpanKind.CLIENT),
  ).toHaveLength(1);
  expect(metrics.filter((m) => m.metric === "duration")).toHaveLength(1);
});

test("two interleaved callIds on one integration produce disjoint traces per flush", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callA = "call-a";
  const callB = "call-b";

  const start = (callId: string, fn: string) =>
    integ.onStart?.({
      callId,
      operationId: "ai.generateText",
      provider: "openai",
      modelId: "gpt-4o",
      functionId: fn,
    });
  const stepStart = (callId: string) => integ.onStepStart?.({ callId, stepNumber: 0 });
  const stepEnd = (callId: string, text: string) =>
    integ.onStepEnd?.({
      callId,
      stepNumber: 0,
      model,
      text,
      finishReason: "stop",
      response: { id: `resp-${callId}`, modelId: "gpt-4o" },
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  const end = (callId: string, text: string) =>
    integ.onEnd?.({
      callId,
      text,
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
    });

  start(callA, "trace-a");
  start(callB, "trace-b");
  stepStart(callA);
  stepStart(callB);
  stepEnd(callB, "b");
  stepEnd(callA, "a");
  await end(callB, "b");
  await end(callA, "a");

  expect(spanBatches).toHaveLength(2);

  const batchFor = (fn: string) => {
    const batch = spanBatches.find((spans) => spans.some((s) => s.name === fn));
    if (!batch) throw new Error(`missing batch for ${fn}`);
    return batch;
  };

  const batchA = batchFor("trace-a");
  const batchB = batchFor("trace-b");
  expect(batchA.every((s) => s.name === "trace-a" || s.name === "chat")).toBe(true);
  expect(batchB.every((s) => s.name === "trace-b" || s.name === "chat")).toBe(true);
  expect(batchA.some((s) => s.name === "trace-b")).toBe(false);
  expect(batchB.some((s) => s.name === "trace-a")).toBe(false);

  const traceA = traceId(batchA[0]!);
  const traceB = traceId(batchB[0]!);
  expect(traceA).not.toBe(traceB);
  for (const s of batchA) expect(traceId(s)).toBe(traceA);
  for (const s of batchB) expect(traceId(s)).toBe(traceB);
});

test("calls that share a sessionId share one trace under the session parent", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const run = async (callId: string, sessionId?: string) => {
    integ.onStart?.({
      callId,
      operationId: "ai.generateText",
      provider: "openai",
      modelId: "gpt-4o",
      functionId: callId,
      runtimeContext: sessionId ? { sessionId } : undefined,
    });
    integ.onStepStart?.({ callId, stepNumber: 0 });
    integ.onStepEnd?.({
      callId,
      stepNumber: 0,
      model,
      text: "ok",
      finishReason: "stop",
      response: { id: `resp-${callId}`, modelId: "gpt-4o" },
      usage: { inputTokens: 1, outputTokens: 1 },
    });
    await integ.onEnd?.({
      callId,
      text: "ok",
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  };
  await run("turn-1", "s1");
  await run("turn-2", "s1");
  await run("solo");

  const session = sessionSpanContext(baseOptions.apiKey, "s1");
  const [batch1, batch2, batch3] = spanBatches;
  for (const s of [...batch1!, ...batch2!]) expect(traceId(s)).toBe(session.traceId);
  for (const batch of [batch1!, batch2!]) {
    const root = batch.find((s) => s.kind === SpanKind.INTERNAL)!;
    expect(root.parentSpanContext?.spanId).toBe(session.spanId);
  }
  expect(traceId(batch3![0]!)).not.toBe(session.traceId);
  expect(batch3!.find((s) => s.kind === SpanKind.INTERNAL)!.parentSpanContext).toBeUndefined();
});

test("timeToFirstOutputMs maps to gen_ai.client.operation.time_to_first_chunk in seconds", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const withTtft = "call-ttft";
  const withoutTtft = "call-no-ttft";

  integ.onStart?.({
    callId: withTtft,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId: withTtft, stepNumber: 0 });
  integ.onStepEnd?.({
    callId: withTtft,
    stepNumber: 0,
    model,
    text: "fast",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
    performance: { timeToFirstOutputMs: 125 },
  });
  await integ.onEnd?.({
    callId: withTtft,
    text: "fast",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  integ.onStart?.({
    callId: withoutTtft,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId: withoutTtft, stepNumber: 0 });
  integ.onStepEnd?.({
    callId: withoutTtft,
    stepNumber: 0,
    model,
    text: "slow",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId: withoutTtft,
    text: "slow",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const ttftChild = spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(ttftChild.attributes["gen_ai.client.operation.time_to_first_chunk"]).toBe(0.125);

  const noTtftChild = spanBatches[1]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(noTtftChild.attributes["gen_ai.client.operation.time_to_first_chunk"]).toBeUndefined();
});

test("tool success and tool error spans parent to the open step with execute_tool metrics", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-tools";
  const okTool = { toolCallId: "t-ok", toolName: "lookup", input: { q: "x" } };
  const badTool = { toolCallId: "t-bad", toolName: "lookup", input: { q: "y" } };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages: [{ role: "user", content: "go" }],
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onToolExecutionStart?.({ callId, toolCall: okTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: okTool,
    toolExecutionMs: 800,
    toolOutput: { type: "tool-result", output: { ok: true } },
  });
  integ.onToolExecutionStart?.({ callId, toolCall: badTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: badTool,
    toolExecutionMs: 400,
    toolOutput: { type: "tool-error", error: new Error("tool blew up") },
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    response: { id: "r-tools", modelId: "gpt-4o" },
    usage: { inputTokens: 3, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 3, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const step = byOperation(spans, "chat").find((s) => s.kind === SpanKind.CLIENT)!;
  const tools = byOperation(spans, "execute_tool");
  expect(tools).toHaveLength(2);

  const ok = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "t-ok")!;
  const bad = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "t-bad")!;
  expect(ok.parentSpanContext?.spanId).toBe(spanId(step));
  expect(bad.parentSpanContext?.spanId).toBe(spanId(step));
  expect(ok.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ ok: true }));
  expect(ok.status.code).toBe(SpanStatusCode.UNSET);
  expect(bad.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(bad.status.code).toBe(SpanStatusCode.ERROR);
  expect(bad.attributes["error.type"]).toBe("Error");
  expect(event(bad, "exception")?.attributes?.["log.severity_number"]).toBe(17);

  const toolDurations = metrics.filter(
    (m) => m.metric === "duration" && m.attributes["gen_ai.operation.name"] === "execute_tool",
  );
  expect(toolDurations.map((m) => m.value).sort((a, b) => a - b)).toEqual([0.4, 0.8]);
});

test("onError marks root and open step failed and a later onEnd for the same callId is a no-op", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-err";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, provider: "openai", modelId: "gpt-4o", stepNumber: 0 });
  await integ.onError?.({ callId, error: new Error("stream died") });
  await integ.onEnd?.({
    callId,
    text: "ignored",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(step.status.code).toBe(SpanStatusCode.ERROR);
  expect(event(root, "exception")?.attributes?.["exception.type"]).toBe("Error");
  expect(event(root, "exception")?.attributes?.["exception.message"]).toBe("stream died");
  expect(event(step, "exception")?.attributes?.["exception.message"]).toBe("stream died");
  expect(step.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(step.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(step.attributes["gen_ai.request.model"]).toBe("gpt-4o");
});

test("onAbort ends the open step, keeps root UNSET, and emits generation.summary aborted", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-abort";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, provider: "openai", modelId: "gpt-4o", stepNumber: 0 });
  await integ.onAbort?.({ callId, steps: [] });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(step).toBeDefined();
  expect(step.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(step.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(step.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(root.status.code).toBe(SpanStatusCode.UNSET);
  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["log.message"]).toBe("Generation aborted");
  expect(summary?.attributes?.["log.severity_number"]).toBe(9);
});

test("generateObject emits json root, one chat child, and a step metric", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-object";

  integ.onStart?.({
    callId,
    operationId: "ai.generateObject",
    provider: "openai",
    modelId: "gpt-4o",
    system: "extract",
    prompt: "name?",
    messages: [{ role: "user", content: "Ada" }],
    output: "object",
    schema: { type: "object", properties: { name: { type: "string" } } },
  });
  integ.onObjectStepStart?.({
    callId,
    provider: "openai",
    modelId: "gpt-4o",
    promptMessages: [{ role: "user", content: "Ada" }],
  });
  integ.onObjectStepEnd?.({
    callId,
    finishReason: "stop",
    response: { id: "obj-1", modelId: "gpt-4o-2024-11-20" },
    usage: { inputTokens: 7, outputTokens: 3 },
    objectText: '{"name":"Ada"}',
    msToFirstChunk: 125,
  });
  await integ.onEnd?.({
    callId,
    object: { name: "Ada" },
    finishReason: "stop",
    usage: { inputTokens: 7, outputTokens: 3 },
  });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const child = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(root.attributes["gen_ai.output.type"]).toBe("json");
  expect(child.attributes["gen_ai.usage.input_tokens"]).toBe(7);
  expect(child.attributes["gen_ai.usage.output_tokens"]).toBe(3);
  expect(child.attributes["gen_ai.output.messages"]).toBe('{"name":"Ada"}');
  expect(child.attributes["gen_ai.client.operation.time_to_first_chunk"]).toBe(0.125);

  expect(metrics.filter((m) => m.metric === "duration")).toHaveLength(1);
  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(2);
});

test("embedMany flushes two embedding children and aggregated root usage on onEnd only", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-embed";

  integ.onStart?.({
    callId,
    operationId: "ai.embedMany",
    provider: "openai",
    modelId: "text-embedding-3-small",
    value: ["a", "b"],
  });
  integ.onEmbedStart?.({
    callId,
    embedCallId: "e1",
    provider: "openai",
    modelId: "text-embedding-3-small",
  });
  integ.onEmbedEnd?.({ callId, embedCallId: "e1", usage: { tokens: 3 } });
  integ.onEmbedStart?.({
    callId,
    embedCallId: "e2",
    provider: "openai",
    modelId: "text-embedding-3-small",
  });
  integ.onEmbedEnd?.({ callId, embedCallId: "e2", usage: { tokens: 5 } });

  expect(spanBatches).toHaveLength(0);

  await integ.onEnd?.({ callId, usage: { tokens: 8 } });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const children = spans.filter((s) => s.kind === SpanKind.CLIENT);
  expect(children).toHaveLength(2);
  expect(root.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(root.attributes["gen_ai.usage.input_tokens"]).toBe(8);
  for (const child of children) {
    expect(child.name).toBe("embeddings");
    expect(child.attributes["gen_ai.operation.name"]).toBe("embeddings");
    expect(child.attributes["gen_ai.usage.input_tokens"]).toBeDefined();
  }
  expect(
    children.map((s) => Number(s.attributes["gen_ai.usage.input_tokens"])).sort((a, b) => a - b),
  ).toEqual([3, 5]);

  const embedTokens = metrics.filter(
    (m) =>
      m.metric === "tokens" &&
      m.tokenType === "input" &&
      m.attributes["gen_ai.operation.name"] === "embeddings",
  );
  expect(embedTokens).toHaveLength(2);
  expect(embedTokens.map((m) => m.value).sort((a, b) => a - b)).toEqual([3, 5]);
});

test("rerank success flushes root and child spans with rerank duration metrics", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-rerank";

  integ.onStart?.({
    callId,
    operationId: "ai.rerank",
    provider: "cohere",
    modelId: "rerank-english-v3.0",
    documents: ["alpha", "beta"],
    query: "best document",
  });
  integ.onRerankStart?.({ callId });
  integ.onRerankEnd?.({ callId });
  await integ.onEnd?.({
    callId,
    ranking: [
      { index: 1, score: 0.9 },
      { index: 0, score: 0.2 },
    ],
    response: { modelId: "rerank-english-v3.0-2026-01-01" },
  });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);

  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const rerank = byOperation(spans, "rerank").find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.attributes["gen_ai.operation.name"]).toBe("rerank");
  expect(root.attributes["gen_ai.response.model"]).toBe("rerank-english-v3.0-2026-01-01");
  expect(rerank.parentSpanContext?.spanId).toBe(spanId(root));
  expect(rerank.attributes["gen_ai.operation.name"]).toBe("rerank");
  expect(rerank.attributes["gen_ai.provider.name"]).toBe("cohere");
  expect(rerank.attributes["gen_ai.request.model"]).toBe("rerank-english-v3.0");

  const duration = metrics.find(
    (m) => m.metric === "duration" && m.attributes["gen_ai.operation.name"] === "rerank",
  )!;
  expect(duration.value).toBeGreaterThanOrEqual(0);
  expect(duration.attributes["gen_ai.provider.name"]).toBe("cohere");
  expect(duration.attributes["gen_ai.request.model"]).toBe("rerank-english-v3.0");
});

test("rerank onError preserves rerank root operation and fails the open rerank child", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-rerank-error";

  integ.onStart?.({
    callId,
    operationId: "ai.rerank",
    provider: "cohere",
    modelId: "rerank-english-v3.0",
    documents: ["alpha", "beta"],
    query: "best document",
  });
  integ.onRerankStart?.({ callId });
  await integ.onError?.({ callId, error: new Error("rerank failed") });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const rerank = spans.find((s) => s.kind === SpanKind.CLIENT && s.name === "rerank")!;

  expect(root.attributes["gen_ai.operation.name"]).toBe("rerank");
  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(root.attributes["error.type"]).toBe("Error");
  expect(rerank.parentSpanContext?.spanId).toBe(spanId(root));
  expect(rerank.status.code).toBe(SpanStatusCode.ERROR);
  expect(rerank.attributes["error.type"]).toBe("Error");
  expect(event(rerank, "exception")?.attributes?.["exception.message"]).toBe("rerank failed");
  expect(rerank.attributes["gen_ai.operation.name"]).toBe("rerank");
  expect(rerank.attributes["gen_ai.provider.name"]).toBe("cohere");
  expect(rerank.attributes["gen_ai.request.model"]).toBe("rerank-english-v3.0");
});

test("onError preserves open object and embed child classification attrs", async () => {
  const objectCapture = makeCapture();
  const objectInteg = telemetryDev(baseOptions, objectCapture.overrides);
  objectInteg.onStart?.({
    callId: "call-object-error",
    operationId: "ai.generateObject",
    provider: "openai",
    modelId: "gpt-4o",
  });
  objectInteg.onObjectStepStart?.({
    callId: "call-object-error",
    provider: "openai",
    modelId: "gpt-4o",
  });
  await objectInteg.onError?.({
    callId: "call-object-error",
    error: new Error("object died"),
  });
  const objectChild = objectCapture.spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(objectChild.status.code).toBe(SpanStatusCode.ERROR);
  expect(objectChild.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(objectChild.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(objectChild.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(objectChild.attributes["gen_ai.output.type"]).toBe("json");

  const embedCapture = makeCapture();
  const embedInteg = telemetryDev(baseOptions, embedCapture.overrides);
  embedInteg.onStart?.({
    callId: "call-embed-error",
    operationId: "ai.embed",
    provider: "openai",
    modelId: "text-embedding-3-small",
  });
  embedInteg.onEmbedStart?.({
    callId: "call-embed-error",
    embedCallId: "embed-err",
    provider: "openai",
    modelId: "text-embedding-3-small",
  });
  await embedInteg.onError?.({
    callId: "call-embed-error",
    error: new Error("embed died"),
  });
  const embedChild = embedCapture.spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(embedChild.status.code).toBe(SpanStatusCode.ERROR);
  expect(embedChild.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(embedChild.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(embedChild.attributes["gen_ai.request.model"]).toBe("text-embedding-3-small");
});

test("recordInputs and recordOutputs false omit message and tool payload attrs but keep ids and tokens", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-redact";
  const toolCall = { toolCallId: "tc-1", toolName: "search", input: { q: "secret-in" } };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages: [{ role: "user", content: "secret root in" }],
    recordInputs: false,
    recordOutputs: false,
  });
  integ.onStepStart?.({
    callId,
    stepNumber: 0,
    messages: [{ role: "user", content: "secret step in" }],
  });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall,
    toolExecutionMs: 50,
    toolOutput: { type: "tool-result", output: { secret: "out" } },
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "secret step out",
    finishReason: "stop",
    response: { id: "redact-r", modelId: "gpt-4o" },
    usage: { inputTokens: 9, outputTokens: 4 },
  });
  await integ.onEnd?.({
    callId,
    text: "secret root out",
    finishReason: "stop",
    usage: { inputTokens: 9, outputTokens: 4 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const step = byOperation(spans, "chat").find((s) => s.kind === SpanKind.CLIENT)!;
  const tool = byOperation(spans, "execute_tool")[0]!;

  expect(root.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(root.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(step.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(step.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
  expect(tool.attributes["gen_ai.tool.call.result"]).toBeUndefined();

  expect(root.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(step.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(step.attributes["gen_ai.usage.output_tokens"]).toBe(4);
  expect(step.attributes["gen_ai.response.id"]).toBe("redact-r");
  expect(tool.attributes["gen_ai.tool.call.id"]).toBe("tc-1");
  expect(tool.attributes["gen_ai.tool.name"]).toBe("search");
});

test("v7 exports session spans only when the configured sampler selects them", async () => {
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
    const integ = telemetryDev(
      { ...baseOptions, sampler, onError: (error) => errors.push(error) },
      overrides,
    );
    const expected: string[] = [];
    for (let i = 0; i < 20; i++) {
      const callId = `sample-${i}`;
      const sessionId = `session-${i}`;
      const id = sessionSpanContext("td_live_test", sessionId).traceId;
      if (
        sampler.shouldSample(ROOT_CONTEXT, id, "chat", SpanKind.INTERNAL, {}, []).decision ===
        SamplingDecision.RECORD_AND_SAMPLED
      )
        expected.push(id, id);
      integ.onStart?.({
        callId,
        operationId: "ai.generateText",
        provider: "openai",
        modelId: "gpt-4o",
        messages: [{ role: "user", content: "hi" }],
        runtimeContext: { sessionId },
      });
      integ.onStepStart?.({ callId, stepNumber: 0 });
      integ.onStepEnd?.({
        callId,
        stepNumber: 0,
        model,
        text: "ok",
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1 },
      });
      await integ.onEnd?.({ callId, text: "ok", finishReason: "stop" });
    }
    expect(spanBatches.flat().map(traceId)).toEqual(expected);
    expect(errors).toEqual([]);
  }
});
