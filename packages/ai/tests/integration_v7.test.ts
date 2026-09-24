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
import { unknownErrorMessage } from "../src/shared.ts";

test("unknown errors do not expose structured payloads", () => {
  expect(unknownErrorMessage({ code: "tool_failed", secret: "private" })).toBe("[object Object]");
});

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
  const privateTool = { toolCallId: "t-private", toolName: "lookup", input: { q: "z" } };
  const arrayTool = { toolCallId: "t-array", toolName: "lookup", input: { q: "a" } };
  const customTool = { toolCallId: "t-custom", toolName: "lookup", input: { q: "c" } };
  const emptyErrorTool = { toolCallId: "t-empty", toolName: "lookup", input: { q: "e" } };
  const primitiveErrorTool = { toolCallId: "t-primitive", toolName: "lookup", input: { q: "p" } };
  let customToStringCalls = 0;

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
  integ.onToolExecutionStart?.({ callId, toolCall: privateTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: privateTool,
    toolExecutionMs: 200,
    toolOutput: { type: "tool-error", error: { code: "tool_failed", secret: "private" } },
  });
  integ.onToolExecutionStart?.({ callId, toolCall: arrayTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: arrayTool,
    toolExecutionMs: 150,
    toolOutput: { type: "tool-error", error: ["private-array"] },
  });
  integ.onToolExecutionStart?.({ callId, toolCall: customTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: customTool,
    toolExecutionMs: 100,
    toolOutput: {
      type: "tool-error",
      error: {
        get message() {
          throw new Error("poisoned getter");
        },
        toString() {
          customToStringCalls += 1;

          return "private-custom";
        },
      },
    },
  });
  integ.onToolExecutionStart?.({ callId, toolCall: emptyErrorTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: emptyErrorTool,
    toolExecutionMs: 50,
    toolOutput: { type: "tool-error", error: new Error() },
  });
  integ.onToolExecutionStart?.({ callId, toolCall: primitiveErrorTool });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: primitiveErrorTool,
    toolExecutionMs: 25,
    toolOutput: { type: "tool-error", error: 404 },
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
  expect(tools).toHaveLength(7);

  const ok = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "t-ok")!;
  const bad = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "t-bad")!;
  const privateFailure = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "t-private")!;
  expect(ok.parentSpanContext?.spanId).toBe(spanId(step));
  expect(bad.parentSpanContext?.spanId).toBe(spanId(step));
  expect(ok.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ ok: true }));
  expect(ok.status.code).toBe(SpanStatusCode.UNSET);
  expect(bad.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(bad.status.code).toBe(SpanStatusCode.ERROR);
  expect(bad.attributes["error.type"]).toBe("Error");
  expect(event(bad, "exception")?.attributes?.["log.severity_number"]).toBe(17);
  expect(privateFailure.attributes["error.type"]).toBe("tool_error");
  expect(event(privateFailure, "exception")?.attributes?.["exception.message"]).toBe(
    "Tool execution failed",
  );
  expect(
    tools
      .filter((tool) =>
        ["t-array", "t-custom", "t-empty"].includes(String(tool.attributes["gen_ai.tool.call.id"])),
      )
      .map((tool) => event(tool, "exception")?.attributes?.["exception.message"]),
  ).toEqual(["Tool execution failed", "Tool execution failed", "Tool execution failed"]);
  expect(
    event(
      tools.find((tool) => tool.attributes["gen_ai.tool.call.id"] === "t-primitive")!,
      "exception",
    )?.attributes?.["exception.message"],
  ).toBe("404");
  expect(customToStringCalls).toBe(0);

  const toolDurations = metrics.filter(
    (m) => m.metric === "duration" && m.attributes["gen_ai.operation.name"] === "execute_tool",
  );

  expect(toolDurations.map((m) => m.value).sort((a, b) => a - b)).toEqual([
    0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.8,
  ]);
});

test("provider-executed tools use the last normalized result for extension spans", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-tools";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages: [{ role: "user", content: "search" }],
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "provider-ok",
        toolName: "web_search",
        input: { query: "OpenTelemetry" },
        providerExecuted: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-ok",
        toolName: "web_search",
        input: { query: "OpenTelemetry" },
        output: { result: "preview" },
        providerExecuted: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-ok",
        toolName: "web_search",
        input: { query: "OpenTelemetry" },
        output: { result: "found" },
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "provider-error",
        toolName: "code_interpreter",
        input: { code: "fail()" },
        providerExecuted: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-error",
        toolName: "code_interpreter",
        input: { code: "fail()" },
        output: { status: "running" },
        providerExecuted: true,
      },
      {
        type: "tool-error",
        toolCallId: "provider-error",
        toolName: "code_interpreter",
        input: { code: "fail()" },
        error: new RangeError("sandbox failed"),
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "client-tool",
        toolName: "lookup",
        input: { query: "local" },
        providerExecuted: false,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 3, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 3, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name !== "execute_tool")!;
  const step = byOperation(spans, "chat").find((s) => s.kind === SpanKind.CLIENT)!;
  const tools = byOperation(spans, "execute_tool");

  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(tools).toHaveLength(2);

  const succeeded = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "provider-ok")!;
  const failed = tools.find((s) => s.attributes["gen_ai.tool.call.id"] === "provider-error")!;

  for (const tool of tools) {
    expect(tool.parentSpanContext?.spanId).toBe(spanId(step));
    expect(tool.attributes["gen_ai.tool.type"]).toBe("extension");
    expect(tool.duration).toEqual([0, 0]);
  }

  expect(succeeded.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "OpenTelemetry" }),
  );
  expect(succeeded.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ result: "found" }));
  expect(failed.status.code).toBe(SpanStatusCode.ERROR);
  expect(failed.attributes["error.type"]).toBe("RangeError");
  expect(event(failed, "exception")?.attributes?.["exception.message"]).toBe("sandbox failed");
  expect(
    metrics.filter(
      (m) => m.metric === "duration" && m.attributes["gen_ai.operation.name"] === "execute_tool",
    ),
  ).toHaveLength(0);
});

test("provider tool observations snapshot callback payloads", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-snapshot";

  const toolCall = {
    type: "tool-call",
    toolCallId: "snapshot-tool",
    toolName: "web_search",
    input: { query: "original" },
    providerExecuted: true,
  };

  const toolResult = {
    type: "tool-result",
    toolCallId: "snapshot-tool",
    toolName: "web_search",
    output: { result: "original" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: [toolCall, toolResult] });

  toolCall.toolCallId = "mutated-call";
  toolCall.toolName = "mutated_tool";
  toolCall.input.query = "mutated";
  toolResult.toolCallId = "mutated-result";
  toolResult.toolName = "mutated_tool";
  toolResult.output.result = "mutated";

  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tool = byOperation(spanBatches[0]!, "execute_tool")[0]!;

  expect(tool.attributes["gen_ai.tool.call.id"]).toBe("snapshot-tool");
  expect(tool.attributes["gen_ai.tool.name"]).toBe("web_search");
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ query: "original" }));
  expect(tool.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ result: "original" }));
});

test("provider-executed tools preserve structured errors and classify invalid MCP result errors", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-error-shapes";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "structured-error",
        toolName: "code_interpreter",
        input: { code: "run()" },
        providerExecuted: true,
      },
      {
        type: "tool-error",
        toolCallId: "structured-error",
        toolName: "code_interpreter",
        error: { code: "sandbox_failed", message: "quota exceeded" },
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "mcp-error",
        toolName: "mcp.delete_file",
        input: undefined,
        error: new Error("invalid arguments"),
        invalid: true,
        providerExecuted: true,
        dynamic: true,
      },
      {
        type: "tool-result",
        toolCallId: "mcp-error",
        toolName: "mcp.delete_file",
        output: {
          type: "call",
          serverLabel: "files",
          name: "delete_file",
          arguments: '{"path":"/protected"}',
          error: "permission denied",
        },
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "mcp-health",
        toolName: "mcp.healthcheck",
        input: undefined,
        providerExecuted: true,
        dynamic: true,
      },
      {
        type: "tool-result",
        toolCallId: "mcp-health",
        toolName: "mcp.healthcheck",
        output: {
          type: "call",
          serverLabel: "health",
          name: "healthcheck",
          arguments: "{}",
          error: false,
        },
        providerExecuted: true,
        dynamic: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 3, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 3, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  const structured = tools.find(
    (span) => span.attributes["gen_ai.tool.call.id"] === "structured-error",
  )!;

  const mcp = tools.find((span) => span.attributes["gen_ai.tool.call.id"] === "mcp-error")!;
  const health = tools.find((span) => span.attributes["gen_ai.tool.call.id"] === "mcp-health")!;

  expect(tools).toHaveLength(3);

  for (const tool of [structured, mcp]) {
    expect(tool.status.code).toBe(SpanStatusCode.ERROR);
    expect(tool.attributes["error.type"]).toBe("tool_error");
  }

  expect(event(structured, "exception")?.attributes?.["exception.message"]).toBe("quota exceeded");
  expect(event(mcp, "exception")?.attributes?.["exception.message"]).toBe("permission denied");
  expect(mcp.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(health.status.code).toBe(SpanStatusCode.UNSET);
  expect(health.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({
      type: "call",
      serverLabel: "health",
      name: "healthcheck",
      arguments: "{}",
      error: false,
    }),
  );
});

test("provider-executed tools correlate deferred final results across model calls", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-deferred";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "deferred-tool",
        toolName: "code_execution",
        input: { code: "print(1)" },
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  integ.onStepStart?.({ callId, stepNumber: 1 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-result",
        toolCallId: "deferred-tool",
        toolName: "code_execution",
        input: undefined,
        output: { stdout: "1" },
        providerExecuted: true,
        preliminary: true,
      },
    ],
  });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-result",
        toolCallId: "deferred-tool",
        toolName: "code_execution",
        input: undefined,
        output: { stdout: "1\n" },
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 1,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const steps = byOperation(spans, "chat").filter((span) => span.kind === SpanKind.CLIENT);
  const tools = byOperation(spans, "execute_tool");

  expect(steps).toHaveLength(2);
  expect(tools).toHaveLength(1);
  expect(tools[0]!.parentSpanContext?.spanId).toBe(spanId(steps[0]!));
  expect(tools[0]!.parentSpanContext?.spanId).not.toBe(spanId(steps[1]!));
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ code: "print(1)" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ stdout: "1\n" }));
  expect(tools[0]!.duration).toEqual([0, 0]);
});

test("provider results recover the latest matching input from a later generation history", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const firstCallId = "call-provider-history-source";
  const secondCallId = "call-provider-history-result";

  const providerCall = {
    type: "tool-call",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    input: { query: "latest" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId: firstCallId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId: firstCallId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId: firstCallId, content: [providerCall] });
  integ.onStepEnd?.({
    callId: firstCallId,
    stepNumber: 0,
    model,
    content: [providerCall],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId: firstCallId,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const messages = [
    {
      role: "assistant",
      content: [
        {
          ...providerCall,
          input: { query: "stale" },
        },
        providerCall,
      ],
    },
  ];

  const providerResult = {
    type: "tool-result",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    output: { result: "latest result" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId: secondCallId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages,
  });
  integ.onStepStart?.({ callId: secondCallId, stepNumber: 0, messages });
  integ.onLanguageModelCallEnd?.({ callId: secondCallId, content: [providerResult] });
  integ.onStepEnd?.({
    callId: secondCallId,
    stepNumber: 0,
    model,
    content: [providerResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId: secondCallId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });

  const resultTool = byOperation(spanBatches.flat(), "execute_tool").find(
    (span) => span.attributes["gen_ai.tool.call.result"] !== undefined,
  )!;

  expect(resultTool.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "latest" }),
  );
  expect(resultTool.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "latest result" }),
  );
});

test("provider results recover inputs from prepareStep message replacements", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-prepare-step-history";

  const providerCall = {
    type: "tool-call",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    input: { query: "initial" },
    providerExecuted: true,
  };

  const messages = [{ role: "assistant", content: [providerCall] }];

  const providerResult = {
    type: "tool-result",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    output: { result: "prepared result" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages,
  });
  providerCall.input = { query: "prepared" };
  integ.onStepStart?.({ callId, stepNumber: 0, messages });
  integ.onLanguageModelCallEnd?.({ callId, content: [providerResult] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content: [providerResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "prepared" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "prepared result" }),
  );
});

test("provider results recover inputs from nested history mutations", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-history-nested-mutation";

  const providerCall = {
    type: "tool-call",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    input: { query: "initial" },
    providerExecuted: true,
  };

  const messages = [{ role: "assistant", content: [providerCall] }];

  const providerResult = {
    type: "tool-result",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    output: { result: "mutated result" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages,
  });
  providerCall.input.query = "mutated";
  integ.onStepStart?.({ callId, stepNumber: 0, messages });
  integ.onLanguageModelCallEnd?.({ callId, content: [providerResult] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content: [providerResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "mutated" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "mutated result" }),
  );
});

test("provider result history is not inspected when input recording is disabled", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-history-redacted";
  let inputReads = 0;

  const providerCall = {
    type: "tool-call",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    providerExecuted: true,
    get input() {
      inputReads++;

      return { query: "history-secret" };
    },
  };

  const messages = [{ role: "assistant", content: [providerCall] }];

  const providerResult = {
    type: "tool-result",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    output: { result: "public" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages,
    recordInputs: false,
  });
  integ.onStepStart?.({ callId, stepNumber: 0, messages });
  integ.onLanguageModelCallEnd?.({ callId, content: [providerResult] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content: [providerResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(inputReads).toBe(0);
  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
  expect(JSON.stringify(spanBatches)).not.toContain("history-secret");
});

test("a poisoned latest history input does not reuse stale provider-tool arguments", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-history-poisoned";
  let inputReads = 0;

  const messages = [
    {
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolCallId: "deferred-tool",
          toolName: "web_search",
          input: { query: "stale" },
          providerExecuted: true,
        },
        {
          type: "tool-call",
          toolCallId: "deferred-tool",
          toolName: "web_search",
          providerExecuted: true,
          get input() {
            inputReads++;

            throw new Error("poisoned input");
          },
        },
      ],
    },
  ];

  const providerResult = {
    type: "tool-result",
    toolCallId: "deferred-tool",
    toolName: "web_search",
    output: { result: "latest result" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    messages,
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: [providerResult] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content: [providerResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(inputReads).toBeGreaterThan(0);
  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "latest result" }),
  );
});

test("provider-executed tools keep repeated call ids distinct across steps", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-repeated-id";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });

  for (let stepNumber = 0; stepNumber < 2; stepNumber++) {
    const input = { query: `query-${stepNumber}` };
    const output = { result: `result-${stepNumber}` };

    integ.onStepStart?.({ callId, stepNumber });
    integ.onLanguageModelCallEnd?.({
      callId,
      content: [
        {
          type: "tool-call",
          toolCallId: "reused-id",
          toolName: "web_search",
          input,
          providerExecuted: true,
        },
        {
          type: "tool-result",
          toolCallId: "reused-id",
          toolName: "web_search",
          output,
          providerExecuted: true,
        },
      ],
    });
    integ.onStepEnd?.({
      callId,
      stepNumber,
      model,
      text: "",
      finishReason: stepNumber === 0 ? "tool-calls" : "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  }

  const approvalToolCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "requires-approval" },
    providerExecuted: true,
  };

  const approvalRequest = {
    type: "tool-approval-request",
    approvalId: "reused-id-approval",
    toolCall: approvalToolCall,
  };

  integ.onStepStart?.({ callId, stepNumber: 2 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [approvalToolCall, approvalRequest],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 2,
    model,
    content: [approvalToolCall, approvalRequest],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 2 },
  });

  const spans = spanBatches[0]!;
  const steps = byOperation(spans, "chat").filter((span) => span.kind === SpanKind.CLIENT);
  const tools = byOperation(spans, "execute_tool");

  expect(steps).toHaveLength(3);
  expect(tools).toHaveLength(2);

  for (let stepNumber = 0; stepNumber < 2; stepNumber++) {
    const tool = tools.find(
      (span) =>
        span.attributes["gen_ai.tool.call.arguments"] ===
        JSON.stringify({ query: `query-${stepNumber}` }),
    )!;

    expect(tool.parentSpanContext?.spanId).toBe(spanId(steps[stepNumber]!));
    expect(tool.attributes["gen_ai.tool.call.id"]).toBe("reused-id");
    expect(tool.attributes["gen_ai.tool.call.result"]).toBe(
      JSON.stringify({ result: `result-${stepNumber}` }),
    );
  }
});

test("provider results preceding reused calls stay with the older invocation", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-result-before-reuse";

  const oldCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "old" },
    providerExecuted: true,
  };

  const oldResult = {
    type: "tool-result",
    toolCallId: "reused-id",
    toolName: "web_search",
    output: { result: "old result" },
    providerExecuted: true,
  };

  const newCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "new" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: [oldCall] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content: [oldCall],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  integ.onStepStart?.({ callId, stepNumber: 1 });
  integ.onLanguageModelCallEnd?.({ callId, content: [oldResult, newCall] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 1,
    model,
    content: [oldResult, newCall],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 2 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  const oldTool = tools.find(
    (span) => span.attributes["gen_ai.tool.call.arguments"] === JSON.stringify({ query: "old" }),
  )!;

  const newTool = tools.find(
    (span) => span.attributes["gen_ai.tool.call.arguments"] === JSON.stringify({ query: "new" }),
  )!;

  expect(tools).toHaveLength(2);
  expect(oldTool.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "old result" }),
  );
  expect(event(oldTool, "tool.result_unobserved")).toBeUndefined();
  expect(newTool.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(event(newTool, "tool.result_unobserved")).toBeDefined();
});

test("provider call and result pairs with reused ids remain distinct in one model call", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-pairs-reused-id";

  const content = [
    {
      type: "tool-call",
      toolCallId: "reused-id",
      toolName: "web_search",
      input: { query: "first" },
      providerExecuted: true,
    },
    {
      type: "tool-result",
      toolCallId: "reused-id",
      toolName: "web_search",
      output: { result: "first result" },
      providerExecuted: true,
    },
    {
      type: "tool-call",
      toolCallId: "reused-id",
      toolName: "web_search",
      input: { query: "second" },
      providerExecuted: true,
    },
    {
      type: "tool-result",
      toolCallId: "reused-id",
      toolName: "web_search",
      output: { result: "second result" },
      providerExecuted: true,
    },
  ];

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content,
    text: "",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(2);

  for (const value of ["first", "second"]) {
    const tool = tools.find(
      (span) => span.attributes["gen_ai.tool.call.arguments"] === JSON.stringify({ query: value }),
    )!;

    expect(tool.attributes["gen_ai.tool.call.result"]).toBe(
      JSON.stringify({ result: `${value} result` }),
    );
    expect(event(tool, "tool.result_unobserved")).toBeUndefined();
  }
});

test("approval fallback omits only the blocked occurrence when ids are reused", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-reused-approval-fallback";

  const completedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "completed" },
    providerExecuted: true,
  };

  const completedResult = {
    type: "tool-result",
    toolCallId: "reused-id",
    toolName: "web_search",
    output: { result: "completed result" },
    providerExecuted: true,
  };

  const blockedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "blocked" },
    providerExecuted: true,
  };

  const content = [
    completedCall,
    completedResult,
    blockedCall,
    {
      type: "tool-approval-request",
      approvalId: "approval-reused-id",
      toolCall: blockedCall,
    },
  ];

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "completed" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "completed result" }),
  );
  expect(event(tools[0]!, "tool.result_unobserved")).toBeUndefined();
});

test("approval cleanup preserves an unresolved same-step occurrence with a reused identity", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-reused-approval-same-step";

  const unresolvedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "unresolved" },
    providerExecuted: true,
  };

  const blockedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "blocked" },
    providerExecuted: true,
  };

  const content = [
    unresolvedCall,
    blockedCall,
    {
      type: "tool-approval-request",
      approvalId: "approval-reused-id",
      toolCall: blockedCall,
    },
  ];

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "unresolved" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(event(tools[0]!, "tool.result_unobserved")).toBeDefined();
});

test("mixed blocked and approved requests bind to reused tool-call occurrences", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-mixed-approval";

  const blockedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "blocked" },
    providerExecuted: true,
  };

  const approvedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "approved" },
    providerExecuted: true,
  };

  const approvedResult = {
    type: "tool-result",
    toolCallId: "reused-id",
    toolName: "web_search",
    output: { result: "approved result" },
    providerExecuted: true,
  };

  const modelContent = [blockedCall, approvedCall, approvedResult];
  const blockedRequestCall = { ...blockedCall };
  const approvedRequestCall = { ...approvedCall };

  const content = [
    ...modelContent,
    {
      type: "tool-approval-request",
      approvalId: "blocked-approval",
      toolCall: blockedRequestCall,
    },
    {
      type: "tool-approval-request",
      approvalId: "approved-approval",
      toolCall: approvedRequestCall,
      isAutomatic: true,
    },
    {
      type: "tool-approval-response",
      approvalId: "approved-approval",
      toolCall: approvedRequestCall,
      approved: true,
      providerExecuted: true,
    },
  ];

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: modelContent });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "approved" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "approved result" }),
  );
});

test("automatic approval responses are consumed once when approval ids are reused", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-reused-approval-id";

  const deniedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "denied" },
    providerExecuted: true,
  };

  const approvedCall = {
    type: "tool-call",
    toolCallId: "reused-id",
    toolName: "web_search",
    input: { query: "approved" },
    providerExecuted: true,
  };

  const approvedResult = {
    type: "tool-result",
    toolCallId: "reused-id",
    toolName: "web_search",
    output: { result: "approved result" },
    providerExecuted: true,
  };

  const modelContent = [deniedCall, approvedCall, approvedResult];

  const content = [
    ...modelContent,
    {
      type: "tool-approval-request",
      approvalId: "reused-approval",
      toolCall: deniedCall,
      isAutomatic: true,
    },
    {
      type: "tool-approval-request",
      approvalId: "reused-approval",
      toolCall: approvedCall,
      isAutomatic: true,
    },
    {
      type: "tool-approval-response",
      approvalId: "reused-approval",
      toolCall: deniedCall,
      approved: false,
      providerExecuted: true,
    },
    {
      type: "tool-approval-response",
      approvalId: "reused-approval",
      toolCall: approvedCall,
      approved: true,
      providerExecuted: true,
    },
  ];

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: modelContent });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    content,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tools = byOperation(spanBatches[0]!, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "approved" }),
  );
  expect(tools[0]!.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ result: "approved result" }),
  );
});

test("approval cleanup preserves older unresolved calls with reused ids", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-reused-approval";

  const unresolved = {
    type: "tool-call",
    toolCallId: "shared-id",
    toolName: "code_execution",
    input: { code: "wait()" },
    providerExecuted: true,
  };

  const awaitingApproval = {
    type: "tool-call",
    toolCallId: "shared-id",
    toolName: "web_search",
    input: { query: "latest" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: [unresolved] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [unresolved],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  integ.onStepStart?.({ callId, stepNumber: 1 });
  integ.onLanguageModelCallEnd?.({ callId, content: [awaitingApproval] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 1,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [
      awaitingApproval,
      {
        type: "tool-approval-request",
        approvalId: "shared-approval",
        toolCall: awaitingApproval,
      },
    ],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 2, outputTokens: 2 },
  });

  const spans = spanBatches[0]!;
  const steps = byOperation(spans, "chat").filter((span) => span.kind === SpanKind.CLIENT);
  const tools = byOperation(spans, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(tools[0]!.attributes["gen_ai.tool.name"]).toBe("code_execution");
  expect(tools[0]!.parentSpanContext?.spanId).toBe(spanId(steps[0]!));
  expect(event(tools[0]!, "tool.result_unobserved")).toBeDefined();
});

test("provider results match tool names when call ids are reused", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-reused-result";

  const codeCall = {
    type: "tool-call",
    toolCallId: "shared-id",
    toolName: "code_execution",
    input: { code: "print(1)" },
    providerExecuted: true,
  };

  const searchCall = {
    type: "tool-call",
    toolCallId: "shared-id",
    toolName: "web_search",
    input: { query: "latest" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });

  for (const [stepNumber, toolCall] of [codeCall, searchCall].entries()) {
    integ.onStepStart?.({ callId, stepNumber });
    integ.onLanguageModelCallEnd?.({ callId, content: [toolCall] });
    integ.onStepEnd?.({
      callId,
      stepNumber,
      model,
      content: [toolCall],
      text: "",
      finishReason: "tool-calls",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  }

  integ.onStepStart?.({ callId, stepNumber: 2 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-result",
        toolCallId: "shared-id",
        toolName: "code_execution",
        output: { stdout: "1\n" },
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 2,
    model,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 3, outputTokens: 3 },
  });

  const spans = spanBatches[0]!;
  const steps = byOperation(spans, "chat").filter((span) => span.kind === SpanKind.CLIENT);
  const tools = byOperation(spans, "execute_tool");
  const code = tools.find((span) => span.attributes["gen_ai.tool.name"] === "code_execution")!;
  const search = tools.find((span) => span.attributes["gen_ai.tool.name"] === "web_search")!;

  expect(tools).toHaveLength(2);
  expect(code.parentSpanContext?.spanId).toBe(spanId(steps[0]!));
  expect(code.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ stdout: "1\n" }));
  expect(search.parentSpanContext?.spanId).toBe(spanId(steps[1]!));
  expect(search.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  expect(event(search, "tool.result_unobserved")).toBeDefined();
});

test("provider tools awaiting approval are not recorded as failed executions", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-approval";

  const toolCall = {
    type: "tool-call",
    toolCallId: "approval-tool",
    toolName: "computer_use",
    input: { action: "click" },
    providerExecuted: true,
  };

  const deniedToolCall = {
    type: "tool-call",
    toolCallId: "denied-tool",
    toolName: "computer_use",
    input: { action: "delete" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [toolCall, deniedToolCall],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [
      toolCall,
      {
        type: "tool-approval-request",
        approvalId: "user-approval",
        toolCall,
      },
      deniedToolCall,
      {
        type: "tool-approval-request",
        approvalId: "denied-approval",
        toolCall: deniedToolCall,
        isAutomatic: true,
      },
      {
        type: "tool-approval-response",
        approvalId: "denied-approval",
        toolCall: deniedToolCall,
        approved: false,
        providerExecuted: true,
      },
    ],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((span) => span.kind === SpanKind.INTERNAL)!;

  expect(byOperation(spans, "execute_tool")).toHaveLength(0);
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
});

test("automatically approved provider tools retain deferred results", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-auto-approved";

  const toolCall = {
    type: "tool-call",
    toolCallId: "approved-tool",
    toolName: "computer_use",
    input: { action: "click" },
    providerExecuted: true,
  };

  const approvalRequest = {
    type: "tool-approval-request",
    approvalId: "approval-1",
    toolCall,
    isAutomatic: true,
  };

  const approvalResponse = {
    type: "tool-approval-response",
    approvalId: "approval-1",
    toolCall,
    approved: true,
    providerExecuted: true,
  };

  const toolResult = {
    type: "tool-result",
    toolCallId: "approved-tool",
    toolName: "computer_use",
    output: { status: "clicked" },
    providerExecuted: true,
  };

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({ callId, content: [toolCall] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [toolCall, approvalRequest, approvalResponse],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  integ.onStepStart?.({ callId, stepNumber: 1 });
  integ.onLanguageModelCallEnd?.({ callId, content: [toolResult] });
  integ.onStepEnd?.({
    callId,
    stepNumber: 1,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [toolResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    content: [toolCall, approvalRequest, approvalResponse, toolResult],
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 3, outputTokens: 2 },
  });

  const spans = spanBatches[0]!;

  const root = spans.find(
    (span) => span.kind === SpanKind.INTERNAL && span.name !== "execute_tool",
  )!;

  const steps = byOperation(spans, "chat").filter((span) => span.kind === SpanKind.CLIENT);
  const tool = byOperation(spans, "execute_tool")[0]!;

  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(byOperation(spans, "execute_tool")).toHaveLength(1);
  expect(tool.parentSpanContext?.spanId).toBe(spanId(steps[0]!));
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ action: "click" }));
  expect(tool.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ status: "clicked" }));
});

test("invalid provider tool calls remain observable and retain deferred results", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-invalid";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "invalid-tool",
        toolName: "web_search",
        input: { query: "latest release" },
        error: new Error("invalid input"),
        invalid: true,
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "invalid-unresolved",
        toolName: "code_execution",
        input: { code: "wait()" },
        error: new Error("refinement failed"),
        invalid: true,
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "openai", modelId: "gpt-4o" },
    text: "",
    finishReason: "error",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  integ.onStepStart?.({ callId, stepNumber: 1 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-result",
        toolCallId: "invalid-tool",
        toolName: "web_search",
        input: undefined,
        output: { results: ["release"] },
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 1,
    model: { provider: "openai", modelId: "gpt-4o" },
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "done",
    finishReason: "stop",
    usage: { inputTokens: 2, outputTokens: 1 },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((span) => span.kind === SpanKind.INTERNAL)!;
  const tools = byOperation(spans, "execute_tool");

  const completed = tools.find(
    (span) => span.attributes["gen_ai.tool.call.id"] === "invalid-tool",
  )!;

  const unresolved = tools.find(
    (span) => span.attributes["gen_ai.tool.call.id"] === "invalid-unresolved",
  )!;

  expect(tools).toHaveLength(2);
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(completed.attributes["gen_ai.tool.call.arguments"]).toBe(
    JSON.stringify({ query: "latest release" }),
  );
  expect(completed.attributes["gen_ai.tool.call.result"]).toBe(
    JSON.stringify({ results: ["release"] }),
  );
  expect(completed.status.code).toBe(SpanStatusCode.UNSET);
  expect(unresolved.status.code).toBe(SpanStatusCode.ERROR);
  expect(unresolved.attributes["error.type"]).toBe("Error");
  expect(event(unresolved, "exception")?.attributes?.["exception.message"]).toBe(
    "refinement failed",
  );
  expect(event(unresolved, "tool.result_unobserved")).toBeUndefined();
});

test("invalid provider tool details honor disabled input capture", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-invalid-redacted";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "openai",
    modelId: "gpt-4o",
    recordInputs: false,
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "invalid-redacted",
        toolName: "web_search",
        input: { query: "private rejected input" },
        error: new Error("invalid input: private rejected input"),
        invalid: true,
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tool = byOperation(spanBatches[0]!, "execute_tool")[0]!;

  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
  expect(tool.attributes["error.type"]).toBe("Error");
  expect(event(tool, "exception")?.attributes?.["exception.message"]).toBe("Tool execution failed");
  expect(JSON.stringify({ attributes: tool.attributes, events: tool.events })).not.toContain(
    "private rejected input",
  );
});

test("pending provider tools remain incomplete when a call ends without a final result", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-unresolved";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "pending-result",
        toolName: "code_execution",
        input: { code: "wait()" },
        providerExecuted: true,
      },
    ],
  });
  integ.onStepEnd?.({
    callId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    text: "",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onEnd?.({
    callId,
    text: "",
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 1 },
  });

  const tool = byOperation(spanBatches[0]!, "execute_tool")[0]!;

  expect(tool.status.code).toBe(SpanStatusCode.UNSET);
  expect(tool.attributes["error.type"]).toBeUndefined();
  expect(event(tool, "exception")).toBeUndefined();
  expect(event(tool, "tool.result_unobserved")?.attributes?.["log.message"]).toBe(
    "Provider tool result was not observed",
  );
});

test("terminal errors retain concrete provider results and suppress provisional calls", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-approval-error";

  integ.onStart?.({
    callId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "approval-pending",
        toolName: "computer_use",
        input: { action: "click" },
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "provider-completed",
        toolName: "web_search",
        input: { query: "latest" },
        providerExecuted: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-completed",
        toolName: "web_search",
        output: { result: "found" },
        providerExecuted: true,
      },
    ],
  });
  await integ.onError?.({ callId, error: new Error("approval callback failed") });

  const spans = spanBatches[0]!;

  const root = spans.find(
    (span) => span.kind === SpanKind.INTERNAL && span.name !== "execute_tool",
  )!;

  const tools = byOperation(spans, "execute_tool");

  expect(tools).toHaveLength(1);
  expect(
    tools.find((span) => span.attributes["gen_ai.tool.call.id"] === "approval-pending"),
  ).toBeUndefined();

  const completed = tools.find(
    (span) => span.attributes["gen_ai.tool.call.id"] === "provider-completed",
  )!;

  expect(completed.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ result: "found" }));
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
});

test("confirmed pending provider tools survive error and abort termination", async () => {
  const errorCapture = makeCapture();
  const errorInteg = telemetryDev(baseOptions, errorCapture.overrides);
  const errorCallId = "call-provider-pending-error";

  const errorToolCall = {
    type: "tool-call",
    toolCallId: "pending-error",
    toolName: "code_execution",
    input: { code: "fail()" },
    providerExecuted: true,
  };

  errorInteg.onStart?.({
    callId: errorCallId,
    operationId: "ai.generateText",
    provider: "anthropic",
    modelId: "claude-sonnet",
    recordOutputs: false,
  });
  errorInteg.onStepStart?.({ callId: errorCallId, stepNumber: 0 });
  errorInteg.onLanguageModelCallEnd?.({
    callId: errorCallId,
    content: [errorToolCall],
  });
  errorInteg.onStepEnd?.({
    callId: errorCallId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [errorToolCall],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await errorInteg.onError?.({ callId: errorCallId, error: new Error("secret failure") });

  const errorSpans = errorCapture.spanBatches[0]!;

  const errorRoot = errorSpans.find(
    (span) => span.kind === SpanKind.INTERNAL && span.name !== "execute_tool",
  )!;

  const errorStep = byOperation(errorSpans, "chat").find((span) => span.kind === SpanKind.CLIENT)!;
  const incompleteTool = byOperation(errorSpans, "execute_tool")[0]!;

  expect(errorRoot.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(event(errorRoot, "exception")?.attributes?.["exception.message"]).toBe(
    "Generation failed",
  );
  expect(incompleteTool.parentSpanContext?.spanId).toBe(spanId(errorStep));
  expect(incompleteTool.status.code).toBe(SpanStatusCode.UNSET);
  expect(incompleteTool.attributes["error.type"]).toBeUndefined();
  expect(event(incompleteTool, "exception")).toBeUndefined();
  expect(event(incompleteTool, "tool.result_unobserved")?.attributes?.["log.message"]).toBe(
    "Provider tool result was not observed",
  );
  expect(JSON.stringify(errorSpans.map((span) => span.events))).not.toContain("secret failure");

  const abortCapture = makeCapture();
  const abortInteg = telemetryDev(baseOptions, abortCapture.overrides);
  const abortCallId = "call-provider-pending-abort";

  const abortToolCall = {
    type: "tool-call",
    toolCallId: "pending-abort",
    toolName: "code_execution",
    input: { code: "wait()" },
    providerExecuted: true,
  };

  abortInteg.onStart?.({
    callId: abortCallId,
    operationId: "ai.streamText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  abortInteg.onStepStart?.({ callId: abortCallId, stepNumber: 0 });
  abortInteg.onLanguageModelCallEnd?.({
    callId: abortCallId,
    content: [abortToolCall],
  });
  abortInteg.onStepEnd?.({
    callId: abortCallId,
    stepNumber: 0,
    model: { provider: "anthropic", modelId: "claude-sonnet" },
    content: [abortToolCall],
    text: "",
    finishReason: "tool-calls",
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await abortInteg.onAbort?.({ callId: abortCallId });

  const abortSpans = abortCapture.spanBatches[0]!;

  const abortRoot = abortSpans.find(
    (span) => span.kind === SpanKind.INTERNAL && span.name !== "execute_tool",
  )!;

  const abortStep = byOperation(abortSpans, "chat").find((span) => span.kind === SpanKind.CLIENT)!;
  const abortedTool = byOperation(abortSpans, "execute_tool")[0]!;

  expect(abortRoot.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(abortedTool.parentSpanContext?.spanId).toBe(spanId(abortStep));
  expect(abortedTool.status.code).toBe(SpanStatusCode.UNSET);
  expect(abortedTool.attributes["error.type"]).toBeUndefined();
  expect(event(abortedTool, "exception")).toBeUndefined();
  expect(event(abortedTool, "tool.result_unobserved")?.attributes?.["log.message"]).toBe(
    "Provider tool result was not observed",
  );
  expect(abortSpans.every((span) => span.status.code !== SpanStatusCode.ERROR)).toBe(true);
});

test("abort before step confirmation omits provisional provider calls", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-provider-provisional-abort";

  integ.onStart?.({
    callId,
    operationId: "ai.streamText",
    provider: "anthropic",
    modelId: "claude-sonnet",
  });
  integ.onStepStart?.({ callId, stepNumber: 0 });
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "provisional-abort",
        toolName: "computer_use",
        input: { action: "click" },
        providerExecuted: true,
      },
    ],
  });
  await integ.onAbort?.({ callId });

  const spans = spanBatches[0]!;
  const root = spans.find((span) => span.kind === SpanKind.INTERNAL)!;

  expect(byOperation(spans, "execute_tool")).toHaveLength(0);
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
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

test("streamObject redacts terminal error details when output capture is disabled", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-object-redacted-error";

  integ.onStart?.({
    callId,
    operationId: "ai.streamObject",
    provider: "openai",
    modelId: "gpt-4o",
    recordOutputs: false,
  });
  await integ.onEnd?.({
    callId,
    error: new Error("private object failure"),
    finishReason: "stop",
    usage: { inputTokens: 1, outputTokens: 0 },
  });

  const root = spanBatches[0]!.find((span) => span.kind === SpanKind.INTERNAL)!;

  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(event(root, "exception")?.attributes?.["exception.message"]).toBe(
    "Structured output parsing or validation failed",
  );
  expect(JSON.stringify(root.events)).not.toContain("private object failure");
});

test("embedMany records usage only on provider-call children", async () => {
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

test("model warning details are redacted when payload capture is disabled", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);

  const warnings = [
    { type: "sensitive-warning-type", message: "secret-warning-detail" },
    { message: "secret-only-message" },
  ];

  for (const [callId, recordInputs, recordOutputs] of [
    ["call-warning-redacted", false, false],
    ["call-warning-open", true, true],
    ["call-warning-inputs-off", false, true],
    ["call-warning-outputs-off", true, false],
  ] as const) {
    integ.onStart?.({
      callId,
      operationId: "ai.generateText",
      provider: "openai",
      modelId: "gpt-4o",
      recordInputs,
      recordOutputs,
    });
    integ.onStepStart?.({ callId, stepNumber: 0 });
    integ.onStepEnd?.({
      callId,
      stepNumber: 0,
      model,
      text: "done",
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
      warnings,
    });
    await integ.onEnd?.({
      callId,
      text: "done",
      finishReason: "stop",
      usage: { inputTokens: 1, outputTokens: 1 },
    });
  }

  const redactedStep = spanBatches[0]!.find((span) => span.kind === SpanKind.CLIENT)!;
  const openStep = spanBatches[1]!.find((span) => span.kind === SpanKind.CLIENT)!;

  const stepWarnings = (span: typeof redactedStep) =>
    span.events.filter((ev) => ev.name === "model.warning");

  expect(stepWarnings(redactedStep)).toHaveLength(2);

  for (const warningEvent of stepWarnings(redactedStep)) {
    expect(warningEvent.attributes?.["log.message"]).toBe("Model warning");
  }

  expect(stepWarnings(redactedStep)[0]!.attributes?.["warning.type"]).toBe(
    "sensitive-warning-type",
  );
  expect(JSON.stringify(spanBatches[0])).not.toContain("secret-warning-detail");
  expect(JSON.stringify(spanBatches[0])).not.toContain("secret-only-message");

  expect(stepWarnings(openStep)).toHaveLength(2);
  expect(stepWarnings(openStep)[0]!.attributes?.["log.message"]).toBe(
    "Model warning: secret-warning-detail",
  );
  expect(stepWarnings(openStep)[1]!.attributes?.["log.message"]).toBe(
    "Model warning: secret-only-message",
  );

  for (const mixedBatch of [spanBatches[2]!, spanBatches[3]!]) {
    const mixedStep = mixedBatch.find((span) => span.kind === SpanKind.CLIENT)!;
    expect(stepWarnings(mixedStep)).toHaveLength(2);

    for (const warningEvent of stepWarnings(mixedStep)) {
      expect(warningEvent.attributes?.["log.message"]).toBe("Model warning");
    }

    expect(stepWarnings(mixedStep)[0]!.attributes?.["warning.type"]).toBe("sensitive-warning-type");
    expect(JSON.stringify(mixedBatch)).not.toContain("secret-warning-detail");
    expect(JSON.stringify(mixedBatch)).not.toContain("secret-only-message");
  }
});

test("recordInputs and recordOutputs false omit message and tool payload attrs but keep ids and tokens", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "call-redact";
  const toolCall = { toolCallId: "tc-1", toolName: "search", input: { q: "secret-in" } };

  const failedToolCall = {
    toolCallId: "tc-error",
    toolName: "search",
    input: { q: "local-error-secret-in" },
  };

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
  integ.onLanguageModelCallEnd?.({
    callId,
    content: [
      {
        type: "tool-call",
        toolCallId: "provider-redacted",
        toolName: "web_search",
        input: { q: "provider-secret-in" },
        providerExecuted: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-redacted",
        toolName: "web_search",
        input: { q: "provider-secret-in" },
        output: { secret: "provider-secret-out" },
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "provider-string-error",
        toolName: "web_search",
        input: { q: "provider-string-secret-in" },
        providerExecuted: true,
      },
      {
        type: "tool-error",
        toolCallId: "provider-string-error",
        toolName: "web_search",
        input: { q: "provider-string-secret-in" },
        error: "provider-string-secret-out",
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "provider-error-object",
        toolName: "web_search",
        input: { q: "provider-error-secret-in" },
        providerExecuted: true,
      },
      {
        type: "tool-error",
        toolCallId: "provider-error-object",
        toolName: "web_search",
        input: { q: "provider-error-secret-in" },
        error: new RangeError("provider-error-secret-out"),
        providerExecuted: true,
      },
      {
        type: "tool-call",
        toolCallId: "provider-mcp-error",
        toolName: "mcp.delete_file",
        input: '{"path":"provider-mcp-secret-in"}',
        providerExecuted: true,
        dynamic: true,
      },
      {
        type: "tool-result",
        toolCallId: "provider-mcp-error",
        toolName: "mcp.delete_file",
        output: {
          type: "call",
          serverLabel: "files",
          name: "delete_file",
          arguments: '{"path":"provider-mcp-secret-in"}',
          error: "provider-mcp-secret-out",
        },
        providerExecuted: true,
      },
    ],
  });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall,
    toolExecutionMs: 50,
    toolOutput: { type: "tool-result", output: { secret: "out" } },
  });
  integ.onToolExecutionEnd?.({
    callId,
    toolCall: failedToolCall,
    toolExecutionMs: 25,
    toolOutput: { type: "tool-error", error: new Error("local-error-secret-out") },
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
  const tools = byOperation(spans, "execute_tool");

  expect(root.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(root.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(step.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(step.attributes["gen_ai.output.messages"]).toBeUndefined();

  expect(tools).toHaveLength(6);

  for (const tool of tools) {
    expect(tool.attributes["gen_ai.tool.call.arguments"]).toBeUndefined();
    expect(tool.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  }

  const failures = tools.filter((tool) => tool.status.code === SpanStatusCode.ERROR);

  expect(failures).toHaveLength(4);
  expect(
    failures
      .map((failure) => failure.attributes["error.type"])
      .sort((left, right) => String(left).localeCompare(String(right))),
  ).toEqual(["Error", "RangeError", "tool_error", "tool_error"]);

  for (const failure of failures) {
    expect(event(failure, "exception")?.attributes?.["exception.message"]).toBe(
      "Tool execution failed",
    );
  }

  expect(
    JSON.stringify(spans.map((span) => ({ attributes: span.attributes, events: span.events }))),
  ).not.toContain("secret");

  expect(root.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(step.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(step.attributes["gen_ai.usage.output_tokens"]).toBe(4);
  expect(step.attributes["gen_ai.response.id"]).toBe("redact-r");
  expect(
    tools
      .map((tool) => [
        String(tool.attributes["gen_ai.tool.call.id"]),
        String(tool.attributes["gen_ai.tool.name"]),
      ])
      .sort((a, b) => a[0]!.localeCompare(b[0]!)),
  ).toEqual([
    ["provider-error-object", "web_search"],
    ["provider-mcp-error", "mcp.delete_file"],
    ["provider-redacted", "web_search"],
    ["provider-string-error", "web_search"],
    ["tc-1", "search"],
    ["tc-error", "search"],
  ]);
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

test.each([
  [true, true],
  [true, false],
  [false, true],
  [false, false],
])(
  "AI SDK evaluate respects recordInputs=%s and recordOutputs=%s",
  async (recordInputs, recordOutputs) => {
    const { experimental_evaluate: evaluate } = await import("ai");
    const { spanBatches, overrides } = makeCapture();
    const integration = telemetryDev(baseOptions, overrides);

    await evaluate({
      model: {
        specificationVersion: "v4",
        provider: "typesafe",
        modelId: "jev",
        supportedQuestionTypes: ["boolean"],
        doEvaluate: async () => ({
          answers: { safe: { type: "boolean", probability: 0.73 } },
          usage: { inputTokens: 19, outputTokens: 2 },
          warnings: [],
        }),
      },
      state: "private-action",
      questions: { safe: { type: "boolean", instructions: "private-policy" } },
      telemetry: { integrations: [integration], recordInputs, recordOutputs },
    });

    expect(spanBatches).toHaveLength(1);

    for (const batch of spanBatches) {
      expect(batch).toHaveLength(2);
      expect(batch.every((span) => span.attributes["gen_ai.operation.name"] === "evaluate")).toBe(
        true,
      );
      expect(
        batch.reduce(
          (sum, span) => sum + Number(span.attributes["gen_ai.usage.input_tokens"] ?? 0),
          0,
        ),
      ).toBe(19);
      expect(
        batch.reduce(
          (sum, span) => sum + Number(span.attributes["gen_ai.usage.output_tokens"] ?? 0),
          0,
        ),
      ).toBe(2);
    }

    for (const span of spanBatches[0]!) {
      expect(span.attributes["gen_ai.provider.name"]).toBe("typesafe");
      expect(span.attributes["gen_ai.request.model"]).toBe("jev");
      expect(span.attributes["gen_ai.input.messages"]).toBe(
        recordInputs
          ? JSON.stringify({
              state: "private-action",
              questions: { safe: { type: "boolean", instructions: "private-policy" } },
            })
          : undefined,
      );
      expect(span.attributes["gen_ai.output.messages"]).toBe(
        recordOutputs
          ? JSON.stringify({
              safe: { type: "boolean", probability: 0.73 },
            })
          : undefined,
      );
    }
  },
);

test.each([true, false])(
  "evaluate preserves arbitrary rejections and cleans up with recordOutputs=%s",
  async (recordOutputs) => {
    const { experimental_evaluate: evaluate } = await import("ai");
    const { spanBatches, overrides } = makeCapture();
    const integration = telemetryDev(baseOptions, overrides);

    for (const error of [
      Object.create(null),
      Object.assign(Object.create(null), { message: "private failure" }),
      Object.defineProperty({}, "message", {
        get() {
          throw new Error("getter failed");
        },
      }),
    ]) {
      let callId: string | undefined;
      await expect(
        evaluate({
          model: {
            specificationVersion: "v4",
            provider: "typesafe",
            modelId: "jev",
            supportedQuestionTypes: ["boolean"],
            doEvaluate: async () => {
              throw error;
            },
          },
          maxRetries: 0,
          state: "action",
          questions: { safe: { type: "boolean", instructions: "policy" } },
          telemetry: {
            integrations: [
              {
                ...integration,
                experimental_onEvaluateStart(event) {
                  callId = event.callId;

                  return integration.experimental_onEvaluateStart?.(event);
                },
              },
            ],
            recordOutputs,
          },
        }),
      ).rejects.toBe(error);

      const batch = spanBatches.at(-1)!;
      expect(batch).toHaveLength(2);

      for (const span of batch) {
        expect(span.ended).toBe(true);
        expect(span.status.code).toBe(SpanStatusCode.ERROR);
        expect(span.attributes["gen_ai.operation.name"]).toBe("evaluate");

        if (!recordOutputs) expect(JSON.stringify(span.events)).not.toContain("private failure");
      }

      expect(callId).toBeDefined();
      const count = spanBatches.length;
      await integration.onError?.({ callId, error });
      expect(spanBatches).toHaveLength(count);
    }

    expect(spanBatches).toHaveLength(3);
  },
);

test("evaluation hooks emit correlated root and model spans without duplicate token metrics", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "evaluation-1";

  const input = {
    state: { response: "Paris" },
    questions: { correct: { type: "boolean", question: "Is this correct?" } },
  };

  const answers = { correct: { type: "boolean", probability: 0.98 } };

  integ.experimental_onEvaluateStart?.({
    callId,
    operationId: "ai.evaluate",
    provider: "vercel",
    modelId: "judge-1",
    functionId: "quality-check",
    runtimeContext: { userId: "reviewer", sessionId: "eval-session", suite: "facts" },
    ...input,
  });
  integ.experimental_onEvaluationModelCallStart?.({
    callId,
    operationId: "ai.evaluate.doEvaluate",
    provider: "vercel",
    modelId: "judge-1",
    ...input,
  });
  integ.experimental_onEvaluationModelCallEnd?.({
    callId,
    operationId: "ai.evaluate.doEvaluate",
    provider: "vercel",
    modelId: "judge-1",
    ...input,
    answers,
    usage: { inputTokens: 12, outputTokens: 3 },
    response: { id: "eval-response", modelId: "judge-1.1", timestamp: new Date() },
  });
  await integ.experimental_onEvaluateEnd?.({
    callId,
    operationId: "ai.evaluate",
    provider: "vercel",
    modelId: "judge-1",
    runtimeContext: { userId: "reviewer", sessionId: "eval-session", suite: "facts" },
    ...input,
    answers,
    usage: { inputTokens: 12, outputTokens: 3 },
    response: { id: "eval-response", modelId: "judge-1.1", timestamp: new Date() },
  });

  const spans = spanBatches[0]!;
  const root = spans.find((span) => span.name === "quality-check")!;
  const modelCall = spans.find((span) => span.kind === SpanKind.CLIENT)!;
  expect(spans).toHaveLength(2);
  expect(modelCall.parentSpanContext?.spanId).toBe(spanId(root));
  expect(root.attributes["gen_ai.operation.name"]).toBe("evaluate");
  expect(root.attributes["gen_ai.response.model"]).toBe("judge-1.1");
  expect(root.attributes["gen_ai.response.id"]).toBe("eval-response");
  expect(root.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(input));
  expect(root.attributes["gen_ai.output.messages"]).toBe(JSON.stringify(answers));
  expect(root.attributes["user.id"]).toBe("reviewer");
  expect(root.attributes["td.metadata.suite"]).toBe("facts");
  expect(modelCall.attributes["gen_ai.usage.input_tokens"]).toBe(12);
  expect(modelCall.attributes["gen_ai.usage.output_tokens"]).toBe(3);
  expect(root.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(root.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
  expect(metrics.filter((metric) => metric.metric === "tokens")).toEqual([
    expect.objectContaining({ tokenType: "input", value: 12 }),
    expect.objectContaining({ tokenType: "output", value: 3 }),
  ]);
});

test("evaluation capture flags redact payloads while preserving identity and usage", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);
  const callId = "evaluation-redacted";

  integ.experimental_onEvaluateStart?.({
    callId,
    provider: "openai",
    modelId: "judge",
    state: { secret: "input" },
    questions: { secret: { type: "score", question: "private" } },
    recordInputs: false,
    recordOutputs: false,
  });
  integ.experimental_onEvaluationModelCallStart?.({ callId });
  integ.experimental_onEvaluationModelCallEnd?.({
    callId,
    answers: { secret: { type: "score", score: 1 } },
    usage: { inputTokens: 4, outputTokens: 2 },
  });
  await integ.experimental_onEvaluateEnd?.({
    callId,
    provider: "openai",
    modelId: "judge",
    answers: { secret: { type: "score", score: 1 } },
    usage: { inputTokens: 4, outputTokens: 2 },
  });

  const spans = spanBatches[0]!;
  expect(JSON.stringify(spans.map((span) => span.attributes))).not.toContain("secret");
  expect(spans.every((span) => span.attributes["gen_ai.input.messages"] === undefined)).toBe(true);
  expect(spans.every((span) => span.attributes["gen_ai.output.messages"] === undefined)).toBe(true);
  expect(
    spans.find((span) => span.kind === SpanKind.CLIENT)?.attributes["gen_ai.usage.input_tokens"],
  ).toBe(4);
});

test("overlapping evaluations stay isolated and errors clean up their call state", async () => {
  const { spanBatches, overrides } = makeCapture();
  const integ = telemetryDev(baseOptions, overrides);

  for (const callId of ["evaluation-a", "evaluation-b"]) {
    integ.experimental_onEvaluateStart?.({
      callId,
      provider: "provider",
      modelId: callId,
      state: callId,
      questions: {},
    });
    integ.experimental_onEvaluationModelCallStart?.({ callId });
  }

  await integ.onError?.({ callId: "evaluation-a", error: new Error("judge failed") });
  integ.experimental_onEvaluationModelCallEnd?.({
    callId: "evaluation-b",
    answers: { result: { type: "boolean", probability: 1 } },
    usage: { inputTokens: 7, outputTokens: 1 },
  });
  await integ.experimental_onEvaluateEnd?.({
    callId: "evaluation-b",
    provider: "provider",
    modelId: "evaluation-b",
    answers: { result: { type: "boolean", probability: 1 } },
    usage: { inputTokens: 7, outputTokens: 1 },
  });
  await integ.experimental_onEvaluateEnd?.({ callId: "evaluation-a" });

  expect(spanBatches).toHaveLength(2);
  const failed = spanBatches.flat().filter((span) => span.status.code === SpanStatusCode.ERROR);
  expect(failed).toHaveLength(2);
  expect(new Set(spanBatches[0]!.map(traceId)).size).toBe(1);
  expect(new Set(spanBatches[1]!.map(traceId)).size).toBe(1);
  expect(traceId(spanBatches[0]![0]!)).not.toBe(traceId(spanBatches[1]![0]!));
});
