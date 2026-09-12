import { type Attributes, ROOT_CONTEXT, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { loggingErrorHandler, setGlobalErrorHandler } from "@opentelemetry/core";
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

import { telemetryDev } from "../src/v6.ts";

interface MetricRecord {
  metric: "duration" | "tokens";
  tokenType?: "input" | "output";
  value: number;
  attributes: Attributes;
}

const requestUrl = (input: string | Request | URL): string =>
  input instanceof Request ? input.url : input.toString();

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

const model = { provider: "openai", modelId: "gpt-4o" } as const;

const spanId = (s: ReadableSpan) => s.spanContext().spanId;
const traceId = (s: ReadableSpan) => s.spanContext().traceId;

const byOperation = (spans: ReadableSpan[], operation: string) =>
  spans.filter((s) => s.attributes["gen_ai.operation.name"] === operation);

const eventNames = (s: ReadableSpan) => s.events.map((ev) => ev.name);
const event = (s: ReadableSpan, name: string) => s.events.find((ev) => ev.name === name);

test("single-step generateText emits a chat root + chat child with all five token kinds", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({
    model,
    system: "be helpful",
    prompt: "hi",
    temperature: 0.7,
    functionId: "answer",
    metadata: { userId: "u1", sessionId: "s1", feature: "chat" },
  });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "hello there",
    finishReason: "stop",
    rawFinishReason: "stop",
    response: { id: "resp_1", modelId: "gpt-4o-2024-11-20", timestamp: new Date() },
    providerMetadata: { openai: { systemFingerprint: "fp_1" } },
    usage: {
      inputTokens: 10,
      outputTokens: 5,
      inputTokenDetails: { cacheReadTokens: 3, cacheWriteTokens: 2 },
      outputTokenDetails: { reasoningTokens: 4 },
    },
  });
  await integ.onFinish?.({ text: "hello there", finishReason: "stop" });

  expect(spanBatches).toHaveLength(1);
  const spans = spanBatches[0]!;
  expect(spans).toHaveLength(2);

  const [root] = byOperation(spans, "chat").filter((s) => s.kind === SpanKind.INTERNAL);
  const child = spans.find((s) => s.kind === SpanKind.CLIENT);

  if (!root || !child) throw new Error("missing spans");

  // Resource + scope ride along on every span (proves they serialize as gen_ai OTLP).
  expect(root.resource.attributes["service.name"]).toBe("svc");
  expect(root.resource.attributes["deployment.environment.name"]).toBe("test");
  expect(root.instrumentationScope.name).toBe("@telemetry-dev/ai-sdk");

  // Root is a `chat` operation (no tools) and names itself from functionId.
  expect(root.name).toBe("answer");
  expect(root.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(root.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(root.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(root.attributes["user.id"]).toBe("u1");
  expect(root.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(root.attributes["td.metadata.feature"]).toBe("chat");
  expect(root.status.code).toBe(SpanStatusCode.UNSET);
  expect(root.parentSpanContext?.spanId).toBe(sessionSpanContext("td_live_test", "s1").spanId);

  // Child step hangs off the root and carries all five token kinds.
  expect(child.name).toBe("chat");
  expect(child.parentSpanContext?.spanId).toBe(spanId(root));
  expect(child.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(child.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(child.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(child.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(child.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(4);
  expect(child.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(child.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(child.attributes["gen_ai.response.id"]).toBe("resp_1");
  expect(child.attributes["gen_ai.output.messages"]).toBe("hello there");

  // conversation.id == sessionId on every span.
  for (const s of spans) {
    expect(s.attributes["gen_ai.conversation.id"]).toBe("s1");
    expect(traceId(s)).toMatch(/^[0-9a-f]{32}$/);
    expect(spanId(s)).toMatch(/^[0-9a-f]{16}$/);
  }

  // One trace per call: every span shares the root trace id.
  expect(traceId(child)).toBe(traceId(root));

  // Two histograms: one duration + two token points (input/output) for the single step.
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
});

test("multi-step run with a tool yields invoke_agent root and parents the tool to its step", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const toolCall = { toolCallId: "call_1", toolName: "getWeather", input: { city: "SF" } };

  integ.onStart?.({
    model,
    messages: [{ role: "user", content: "weather?" }],
    metadata: { sessionId: "s9" },
  });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onToolCallStart?.({ stepNumber: 0, toolCall });
  integ.onToolCallFinish?.({
    stepNumber: 0,
    toolCall,
    durationMs: 42,
    success: true,
    output: { tempF: 70 },
  });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    response: { id: "r0", modelId: "gpt-4o" },
    usage: { inputTokens: 8, outputTokens: 2 },
  });
  integ.onStepStart?.({ stepNumber: 1, model });
  integ.onStepFinish?.({
    stepNumber: 1,
    model,
    text: "It is 70F in SF.",
    finishReason: "stop",
    response: { id: "r1", modelId: "gpt-4o" },
    usage: { inputTokens: 12, outputTokens: 6 },
  });
  await integ.onFinish?.({ text: "It is 70F in SF.", finishReason: "stop" });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL && s.name === "chat")!;
  const chatSteps = byOperation(spans, "chat").filter((s) => s.kind === SpanKind.CLIENT);
  const tool = byOperation(spans, "execute_tool")[0];
  expect(chatSteps).toHaveLength(2);

  if (!tool) throw new Error("missing tool span");

  // Tools push the root operation to invoke_agent.
  expect(root.attributes["gen_ai.operation.name"]).toBe("invoke_agent");

  // The tool span is parented to its step span (step 0), not to the root.
  const step0 = chatSteps.find((s) => s.attributes["gen_ai.response.id"] === "r0")!;
  expect(tool.parentSpanContext?.spanId).toBe(spanId(step0));
  expect(tool.attributes["gen_ai.tool.name"]).toBe("getWeather");
  expect(tool.attributes["gen_ai.tool.call.id"]).toBe("call_1");
  expect(tool.attributes["gen_ai.tool.call.arguments"]).toBe(JSON.stringify({ city: "SF" }));
  expect(tool.attributes["gen_ai.tool.call.result"]).toBe(JSON.stringify({ tempF: 70 }));
  expect(tool.status.code).toBe(SpanStatusCode.UNSET);

  for (const s of spans) {
    expect(s.attributes["gen_ai.conversation.id"]).toBe("s9");
  }

  // Per-step duration + tokens (x2 steps) and a per-tool duration with operation=execute_tool.
  const durations = metrics.filter((m) => m.metric === "duration");
  expect(durations.filter((m) => m.attributes["gen_ai.operation.name"] === "chat")).toHaveLength(2);

  const toolDurations = durations.filter(
    (m) => m.attributes["gen_ai.operation.name"] === "execute_tool",
  );

  expect(toolDurations).toHaveLength(1);
  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(4);
});

test("error finishReason marks root and step ERROR and emits an exception event", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({ model, prompt: "boom" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "",
    finishReason: "error",
    response: { id: "r", modelId: "gpt-4o" },
    usage: {},
  });
  await integ.onFinish?.({ text: "", finishReason: "error" });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.status.code).toBe(SpanStatusCode.ERROR);
  expect(step.status.code).toBe(SpanStatusCode.ERROR);
  expect(root.attributes["error.type"]).toBe("error");
  expect(eventNames(root)).toContain("exception");
  const exc = event(root, "exception");
  expect(exc?.attributes?.["log.severity_number"]).toBe(17);
  // Missing usage emits no token attributes (not zeros).
  expect(step.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(step.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
});

test("a failed tool call records ERROR status and an exception event", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const callA = { toolCallId: "a", toolName: "t1", input: { n: 1 } };
  const callB = { toolCallId: "b", toolName: "t2", input: { n: 2 } };

  integ.onStart?.({ model, prompt: "use tools" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onToolCallStart?.({ stepNumber: 0, toolCall: callA });
  integ.onToolCallStart?.({ stepNumber: 0, toolCall: callB });
  integ.onToolCallFinish?.({
    stepNumber: 0,
    toolCall: callA,
    durationMs: 10,
    success: true,
    output: "ra",
  });
  integ.onToolCallFinish?.({
    stepNumber: 0,
    toolCall: callB,
    durationMs: 20,
    success: false,
    error: new Error("kaboom"),
  });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "done",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "done", finishReason: "stop" });

  const spans = spanBatches[0]!;
  const tools = byOperation(spans, "execute_tool");
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(tools).toHaveLength(2);

  // Both tools parent to their step.
  for (const t of tools) {
    expect(t.parentSpanContext?.spanId).toBe(spanId(step));
  }

  const failed = tools.find((t) => t.attributes["gen_ai.tool.name"] === "t2")!;
  expect(failed.status.code).toBe(SpanStatusCode.ERROR);
  expect(failed.attributes["error.type"]).toBe("Error");
  expect(failed.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  const exc = event(failed, "exception");
  expect(exc?.attributes?.["exception.message"]).toBe("kaboom");
  expect(exc?.attributes?.["log.severity_number"]).toBe(17);
});

test("gateway provider is relabeled to Vercel AI Gateway on every span", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const gatewayModel = { provider: "gateway", modelId: "openai/gpt-4o" } as const;
  const toolCall = { toolCallId: "c", toolName: "search", input: {} };
  integ.onStart?.({ model: gatewayModel, prompt: "hi", metadata: { sessionId: "g1" } });
  integ.onStepStart?.({ stepNumber: 0, model: gatewayModel });
  integ.onToolCallStart?.({ stepNumber: 0, toolCall });
  integ.onToolCallFinish?.({
    stepNumber: 0,
    toolCall,
    durationMs: 5,
    success: true,
    output: {},
  });
  integ.onStepFinish?.({
    stepNumber: 0,
    model: gatewayModel,
    text: "hello",
    finishReason: "stop",
    response: { id: "r", modelId: "openai/gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "hello", finishReason: "stop" });

  const spans = spanBatches[0]!;
  // Every span that carries a provider carries the product name, never the bare slug.
  const withProvider = spans.filter((s) => "gen_ai.provider.name" in s.attributes);
  expect(withProvider.length).toBeGreaterThanOrEqual(2);

  for (const s of withProvider) {
    expect(s.attributes["gen_ai.provider.name"]).toBe("Vercel AI Gateway");
  }

  // The upstream model slug is left untouched (it still drives pricing lookups server-side).
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(step.attributes["gen_ai.request.model"]).toBe("openai/gpt-4o");
});

test("non-gateway providers pass through unchanged", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "hello",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "hello", finishReason: "stop" });

  const spans = spanBatches[0]!;
  const root = spans.find((s) => s.kind === SpanKind.INTERNAL)!;
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(root.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(step.attributes["gen_ai.provider.name"]).toBe("openai");
});

test("model warnings become model.warning events at severity 13", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
    warnings: [{ type: "unsupported-setting", setting: "seed", message: "seed is not supported" }],
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  const spans = spanBatches[0]!;
  const step = spans.find((s) => s.kind === SpanKind.CLIENT)!;
  const warning = event(step, "model.warning");
  expect(warning).toBeDefined();
  expect(warning?.attributes?.["log.severity_number"]).toBe(13);
  expect(String(warning?.attributes?.["log.message"])).toContain("seed is not supported");
});

test("no apiKey is a complete no-op (transport never invoked)", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();
  const integ = telemetryDev({ environment: "test", serviceName: "svc" }, overrides);

  integ.onStart?.({ model, prompt: "x" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "y",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: {},
  });
  await integ.onFinish?.({ text: "y", finishReason: "stop" });

  expect(spanBatches).toHaveLength(0);
  expect(metrics).toHaveLength(0);
});

test("a step with missing usage records no token metric and no summary usage attrs", async () => {
  const { spanBatches, metrics, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: {},
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  // Absent usage must never be reported as a 0-token observation: no token point on either side.
  expect(metrics.filter((m) => m.metric === "tokens")).toHaveLength(0);
  // The step duration is still recorded.
  expect(metrics.filter((m) => m.metric === "duration")).toHaveLength(1);

  const root = spanBatches[0]!.find((s) => s.kind === SpanKind.INTERNAL)!;
  const summary = event(root, "generation.summary");
  expect(summary?.attributes?.["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(summary?.attributes?.["gen_ai.usage.output_tokens"]).toBeUndefined();
  expect(String(summary?.attributes?.["log.message"])).not.toContain("tokens");
});

test("export failure reaches onError even on the waitUntil path", async () => {
  const errors: unknown[] = [];
  let handed: Promise<unknown> | undefined;

  const integ = telemetryDev(
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
      // Span send rejects; the metrics path is a no-op so only the trace export fails.
      sendSpans: async () => {
        throw new Error("trace export boom");
      },
      recordDuration: () => {},
      recordTokens: () => {},
    },
  );

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  // waitUntil received the handed-off flush promise, which never rejects.
  expect(handed).toBeDefined();
  await expect(handed).resolves.toBeUndefined();
  // The trace-export rejection still surfaced through onError.
  expect(errors).toHaveLength(1);
  expect(errors[0] instanceof Error ? errors[0].message : undefined).toBe("trace export boom");
});

test("spans use each emitter's own transport (no cross-talk on a shared cached core)", async () => {
  const traceCalls: string[] = [];
  const aErrors: unknown[] = [];
  const bErrors: unknown[] = [];

  const makeFetch =
    (tag: string): typeof fetch =>
    async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) traceCalls.push(tag);

      return new Response(null, { status: 200 });
    };

  // First emitter caches a core under identity "dup" with fetch A.
  const integA = telemetryDev({
    apiKey: "td_live_dup",
    environment: "test",
    serviceName: "dup",
    fetch: makeFetch("A"),
    onError: (e) => aErrors.push(e),
  });

  // Second emitter reuses that cached core but supplies fetch B.
  telemetryDev({
    apiKey: "td_live_dup",
    environment: "test",
    serviceName: "dup",
    fetch: makeFetch("B"),
    onError: (e) => bErrors.push(e),
  });

  // Run a generation on the FIRST (older) emitter after the second was created: its spans
  // must still go through ITS OWN fetch (A), proving span transport is per-call, not shared.
  integA.onStart?.({ model, prompt: "hi" });
  integA.onStepStart?.({ stepNumber: 0, model });
  integA.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integA.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls.length).toBeGreaterThan(0);
  expect(traceCalls.every((c) => c === "A")).toBe(true);
  expect(aErrors).toHaveLength(0);
  expect(bErrors).toHaveLength(0);
});

test("trace export retries a transient ingest failure", async () => {
  const errors: unknown[] = [];
  let traceCalls = 0;

  const integ = telemetryDev({
    apiKey: "td_live_trace_retry",
    environment: "test",
    serviceName: "trace-retry",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) {
        traceCalls += 1;

        return new Response(null, { status: traceCalls === 1 ? 503 : 200 });
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls).toBe(2);
  expect(errors).toHaveLength(0);
});

test("trace export retries a transient network error", async () => {
  const errors: unknown[] = [];
  let traceCalls = 0;

  const integ = telemetryDev({
    apiKey: "td_live_trace_network_retry",
    environment: "test",
    serviceName: "trace-network-retry",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) {
        traceCalls += 1;

        if (traceCalls === 1) throw new Error("socket closed");
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls).toBe(2);
  expect(errors).toHaveLength(0);
});

test("trace export reports one error after exhausting network failures", async () => {
  const errors: unknown[] = [];
  let traceCalls = 0;
  const networkError = new Error("socket closed");

  const integ = telemetryDev({
    apiKey: "td_live_trace_network_exhausted",
    environment: "test",
    serviceName: "trace-network-exhausted",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) {
        traceCalls += 1;
        throw networkError;
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls).toBe(3);
  expect(errors).toHaveLength(1);
  expect(errors[0]).toBe(networkError);
});

test("trace export does not retry non-retryable ingest failures", async () => {
  const errors: unknown[] = [];
  let traceCalls = 0;

  const integ = telemetryDev({
    apiKey: "td_live_trace_no_retry",
    environment: "test",
    serviceName: "trace-no-retry",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) {
        traceCalls += 1;

        return new Response(null, { status: 400 });
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls).toBe(1);
  expect(errors).toHaveLength(1);
  expect(errors[0] instanceof Error ? errors[0].message : undefined).toBe(
    "telemetry.dev trace ingest failed: 400",
  );
});

test("trace export reports one error after exhausting retryable failures", async () => {
  const errors: unknown[] = [];
  let traceCalls = 0;

  const integ = telemetryDev({
    apiKey: "td_live_trace_retry_exhausted",
    environment: "test",
    serviceName: "trace-retry-exhausted",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/traces")) {
        traceCalls += 1;

        return new Response(null, { status: 503 });
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(traceCalls).toBe(3);
  expect(errors).toHaveLength(1);
  expect(errors[0] instanceof Error ? errors[0].message : undefined).toBe(
    "telemetry.dev trace ingest failed: 503",
  );
});

test("step metrics attribute tokens and duration to each step's own model, not the root", async () => {
  const { metrics, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const modelB = { provider: "anthropic", modelId: "claude-3-5-sonnet" } as const;

  // Root/step 0 run on openai/gpt-4o; step 1 falls back to anthropic/claude (mixed-model run).
  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "a",
    finishReason: "stop",
    response: { id: "r0", modelId: "gpt-4o-2024-11-20" },
    usage: { inputTokens: 10, outputTokens: 5 },
  });
  integ.onStepStart?.({ stepNumber: 1, model: modelB });
  integ.onStepFinish?.({
    stepNumber: 1,
    model: modelB,
    text: "b",
    finishReason: "stop",
    response: { id: "r1", modelId: "claude-3-5-sonnet-20241022" },
    usage: { inputTokens: 20, outputTokens: 7 },
  });
  await integ.onFinish?.({ text: "b", finishReason: "stop" });

  const durations = metrics.filter((m) => m.metric === "duration");
  expect(durations).toHaveLength(2);

  // Each step's metric carries the model/provider that actually ran it — not the root's. Before the
  // fix every step inherited the root metricBase (gpt-4o), so the claude point below would not exist.
  const openaiDur = durations.find((m) => m.attributes["gen_ai.request.model"] === "gpt-4o");

  const claudeDur = durations.find(
    (m) => m.attributes["gen_ai.request.model"] === "claude-3-5-sonnet",
  );

  if (!openaiDur || !claudeDur) throw new Error("missing per-model duration metric");
  expect(openaiDur.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(openaiDur.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(claudeDur.attributes["gen_ai.provider.name"]).toBe("anthropic");
  expect(claudeDur.attributes["gen_ai.response.model"]).toBe("claude-3-5-sonnet-20241022");

  // Tokens line up with their owning model too (20 in belongs to claude, 5 out to openai).
  const tokens = metrics.filter((m) => m.metric === "tokens");

  const claudeInput = tokens.find(
    (m) => m.tokenType === "input" && m.attributes["gen_ai.request.model"] === "claude-3-5-sonnet",
  );

  expect(claudeInput?.value).toBe(20);

  const openaiOutput = tokens.find(
    (m) => m.tokenType === "output" && m.attributes["gen_ai.request.model"] === "gpt-4o",
  );

  expect(openaiOutput?.value).toBe(5);
});

test("metrics use each emitter's own transport (no cross-talk on a shared cached core)", async () => {
  const metricCalls: string[] = [];
  const aErrors: unknown[] = [];
  const bErrors: unknown[] = [];

  const makeFetch =
    (tag: string): typeof fetch =>
    async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/metrics")) metricCalls.push(tag);

      return new Response(null, { status: 200 });
    };

  // First emitter caches a core under identity "dupm" with fetch A (real metrics pipeline).
  const integA = telemetryDev({
    apiKey: "td_live_dupm",
    environment: "test",
    serviceName: "dupm",
    fetch: makeFetch("A"),
    onError: (e) => aErrors.push(e),
  });

  // Second emitter reuses that cached core but supplies fetch B.
  telemetryDev({
    apiKey: "td_live_dupm",
    environment: "test",
    serviceName: "dupm",
    fetch: makeFetch("B"),
    onError: (e) => bErrors.push(e),
  });

  // Run a generation on the FIRST (older) emitter after the second was created: its metrics must
  // flush through ITS OWN fetch (A). Before the fix the shared core transport was last-writer (B).
  integA.onStart?.({ model, prompt: "hi" });
  integA.onStepStart?.({ stepNumber: 0, model });
  integA.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integA.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(metricCalls.length).toBeGreaterThan(0);
  expect(metricCalls.every((c) => c === "A")).toBe(true);
  expect(aErrors).toHaveLength(0);
  expect(bErrors).toHaveLength(0);
});

test("metric export retries a transient ingest failure", async () => {
  const errors: unknown[] = [];
  let metricCalls = 0;

  const integ = telemetryDev({
    apiKey: "td_live_metric_retry",
    environment: "test",
    serviceName: "metric-retry",
    fetch: async (url: string | Request | URL, _init?: RequestInit) => {
      if (requestUrl(url).endsWith("/v1/metrics")) {
        metricCalls += 1;

        return new Response(null, { status: metricCalls === 1 ? 503 : 200 });
      }

      return new Response(null, { status: 200 });
    },
    onError: (e) => errors.push(e),
  });

  integ.onStart?.({ model, prompt: "hi" });
  integ.onStepStart?.({ stepNumber: 0, model });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  expect(metricCalls).toBe(2);
  expect(errors).toHaveLength(0);
});

test("a non-OK /v1/metrics response is reported as a failed export, not a success", async () => {
  const errors: unknown[] = [];
  const exportErrors: unknown[] = [];
  let metricCalls = 0;
  // Capture OTel's global error handler: the metric reader only routes here when the export result
  // is FAILED (code 1). A success (code 0) report — the pre-fix behavior — never reaches it.
  setGlobalErrorHandler((e) => exportErrors.push(e));

  try {
    const integ = telemetryDev({
      apiKey: "td_live_metricfail",
      environment: "test",
      serviceName: "metricfail",
      fetch: async (url: string | Request | URL, _init?: RequestInit) => {
        if (requestUrl(url).endsWith("/v1/metrics")) metricCalls += 1;

        return new Response(null, {
          status: requestUrl(url).endsWith("/v1/metrics") ? 503 : 200,
        });
      },
      onError: (e) => errors.push(e),
    });

    integ.onStart?.({ model, prompt: "hi" });
    integ.onStepStart?.({ stepNumber: 0, model });
    integ.onStepFinish?.({
      stepNumber: 0,
      model,
      text: "ok",
      finishReason: "stop",
      response: { id: "r", modelId: "gpt-4o" },
      usage: { inputTokens: 1, outputTokens: 1 },
    });
    await integ.onFinish?.({ text: "ok", finishReason: "stop" });

    // Our own callback sees the 503 once the retries are exhausted.
    expect(
      errors.some((e) => e instanceof Error && /metric ingest failed: 503/.test(e.message)),
    ).toBe(true);
    expect(metricCalls).toBe(3);
    // The export is marked FAILED, so OTel rethrows it to the global handler — this is the bit the
    // fix changes (code 0 → code 1).
    expect(
      exportErrors.some((e) => e instanceof Error && /metrics export failed/.test(e.message)),
    ).toBe(true);
  } finally {
    setGlobalErrorHandler(loggingErrorHandler());
  }
});

test("each step span records its own per-step input messages, not the root input", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const toolCall = { toolCallId: "call_1", toolName: "getWeather", input: { city: "SF" } };
  const step0Messages = [{ role: "user", content: "weather in SF?" }];

  // The tool loop appends the assistant tool-call and the tool result before the next step runs, so
  // step 1's request differs from both step 0 and the root input.
  const step1Messages = [
    { role: "user", content: "weather in SF?" },
    { role: "assistant", content: [{ type: "tool-call", toolCallId: "call_1" }] },
    {
      role: "tool",
      content: [{ type: "tool-result", toolCallId: "call_1", output: { tempF: 70 } }],
    },
  ];

  integ.onStart?.({ model, messages: step0Messages });
  integ.onStepStart?.({ stepNumber: 0, model, messages: step0Messages });
  integ.onToolCallStart?.({ stepNumber: 0, toolCall });
  integ.onToolCallFinish?.({
    stepNumber: 0,
    toolCall,
    durationMs: 5,
    success: true,
    output: { tempF: 70 },
  });
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "",
    finishReason: "tool-calls",
    response: { id: "r0", modelId: "gpt-4o" },
    usage: { inputTokens: 8, outputTokens: 2 },
  });
  integ.onStepStart?.({ stepNumber: 1, model, messages: step1Messages });
  integ.onStepFinish?.({
    stepNumber: 1,
    model,
    text: "It is 70F in SF.",
    finishReason: "stop",
    response: { id: "r1", modelId: "gpt-4o" },
    usage: { inputTokens: 12, outputTokens: 6 },
  });
  await integ.onFinish?.({ text: "It is 70F in SF.", finishReason: "stop" });

  const spans = spanBatches[0]!;
  const chatSteps = byOperation(spans, "chat").filter((s) => s.kind === SpanKind.CLIENT);
  const step0 = chatSteps.find((s) => s.attributes["gen_ai.response.id"] === "r0")!;
  const step1 = chatSteps.find((s) => s.attributes["gen_ai.response.id"] === "r1")!;

  expect(JSON.parse(String(step0.attributes["gen_ai.input.messages"]))).toEqual(step0Messages);
  expect(JSON.parse(String(step1.attributes["gen_ai.input.messages"]))).toEqual(step1Messages);
  expect(step0.attributes["gen_ai.input.messages"]).not.toBe(
    step1.attributes["gen_ai.input.messages"],
  );
});

test("a step finishing without onStepStart omits input messages", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  integ.onStart?.({ model, messages: [{ role: "user", content: "hi" }] });
  // No onStepStart: the fallback step span has no per-step request to record.
  integ.onStepFinish?.({
    stepNumber: 0,
    model,
    text: "ok",
    finishReason: "stop",
    response: { id: "r", modelId: "gpt-4o" },
    usage: { inputTokens: 1, outputTokens: 1 },
  });
  await integ.onFinish?.({ text: "ok", finishReason: "stop" });

  const step = spanBatches[0]!.find((s) => s.kind === SpanKind.CLIENT)!;
  expect(step.attributes["gen_ai.input.messages"]).toBeUndefined();
});

test("calls that share a sessionId share one trace under the session parent", async () => {
  const { spanBatches, overrides } = makeCapture();

  const integ = telemetryDev(
    { apiKey: "td_live_test", environment: "test", serviceName: "svc" },
    overrides,
  );

  const run = async (sessionId?: string) => {
    integ.onStart?.({ model, prompt: "hi", metadata: sessionId ? { sessionId } : undefined });
    integ.onStepStart?.({ stepNumber: 0, model });
    integ.onStepFinish?.({
      stepNumber: 0,
      model,
      text: "ok",
      finishReason: "stop",
      response: { id: "r", modelId: "gpt-4o" },
      usage: { inputTokens: 1, outputTokens: 1 },
    });
    await integ.onFinish?.({ text: "ok", finishReason: "stop" });
  };

  await run("s1");
  await run("s1");
  await run();

  const session = sessionSpanContext("td_live_test", "s1");
  const [batch1, batch2, batch3] = spanBatches;

  for (const s of [...batch1!, ...batch2!]) expect(s.spanContext().traceId).toBe(session.traceId);

  for (const batch of [batch1!, batch2!]) {
    const root = batch.find((s) => s.kind === SpanKind.INTERNAL)!;
    expect(root.parentSpanContext?.spanId).toBe(session.spanId);
  }

  expect(batch3![0]!.spanContext().traceId).not.toBe(session.traceId);
});

test("v6 exports session spans only when the configured sampler selects them", async () => {
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
      const sessionId = `session-${i}`;
      const id = sessionSpanContext("td_live_test", sessionId).traceId;

      if (
        sampler.shouldSample(ROOT_CONTEXT, id, "chat", SpanKind.INTERNAL, {}, []).decision ===
        SamplingDecision.RECORD_AND_SAMPLED
      )
        expected.push(id, id);
      integ.onStart?.({ model, prompt: "hi", metadata: { sessionId } });
      integ.onStepStart?.({ stepNumber: 0, model });
      integ.onStepFinish?.({
        stepNumber: 0,
        model,
        text: "ok",
        finishReason: "stop",
        usage: { inputTokens: 1, outputTokens: 1 },
      });
      await integ.onFinish?.({ text: "ok", finishReason: "stop" });
    }

    expect(spanBatches.flat().map(traceId)).toEqual(expected);
    expect(errors).toEqual([]);
  }
});
