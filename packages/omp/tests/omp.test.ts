import { SpanStatusCode } from "@opentelemetry/api";
import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { flush, type ClientOverrides } from "@telemetry-dev/sdk";
import type { ExtensionContext, ExtensionFactory } from "@oh-my-pi/pi-coding-agent";
import { afterEach, expect, test } from "vitest";

import { resetForTesting } from "../src/config.ts";
import { telemetryDevExtension, type TelemetryDevExtensionOptions } from "../src/extension.ts";

interface Exporters extends ClientOverrides {
  logs: InMemoryLogRecordExporter;
  spans: InMemorySpanExporter;
}

function makeExporters(): Exporters {
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();
  return { logRecordExporter: logs, logs, spanExporter: spans, spans };
}

function telemetryOptions(options?: TelemetryDevExtensionOptions): TelemetryDevExtensionOptions {
  const fetch = Object.assign(async () => new Response(null, { status: 200 }), {
    preconnect: () => undefined,
  });
  return {
    environment: "test",
    exportMode: "immediate" as const,
    logLevel: "silent" as const,
    fetch,
    ...options,
  };
}

interface TestEvent {
  type: string;
  [key: string]: TestValue;
}

type TestValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | TestValue[]
  | { [key: string]: TestValue };

type EmitFn = (event: TestEvent, ctx?: ExtensionContext) => Promise<void>;

interface Harness {
  emit: EmitFn;
  exporters: Exporters;
}

function makeContext(sessionId = "session-1"): ExtensionContext {
  return {
    cwd: "/tmp/project",
    model: { id: "claude-fable-5", provider: "anthropic" },
    sessionManager: {
      getSessionId: () => sessionId,
      getCwd: () => "/tmp/project",
      getSessionName: () => undefined,
      getSessionFile: () => undefined,
    },
  } as ExtensionContext;
}

function makeHarness(options?: TelemetryDevExtensionOptions): Harness {
  const exporters = makeExporters();
  const handlers = new Map<
    string,
    ((event: TestEvent, ctx: ExtensionContext) => void | Promise<void>)[]
  >();
  const api = {
    on(type: string, handler: (event: TestEvent, ctx: ExtensionContext) => void | Promise<void>) {
      const list = handlers.get(type) ?? [];
      list.push(handler);
      handlers.set(type, list);
    },
  };

  const factory = telemetryDevExtension(telemetryOptions(options), exporters);
  factory(api as never);

  const defaultCtx = makeContext();
  const emit: EmitFn = async (event, ctx = defaultCtx) => {
    for (const handler of handlers.get(event.type) ?? []) {
      await handler(event, ctx);
    }
  };
  return { emit, exporters };
}

function assistantMessage(overrides?: { [key: string]: TestValue }) {
  return {
    role: "assistant",
    content: [{ type: "text", text: "All done." }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-fable-5",
    responseId: "resp-1",
    usage: {
      input: 120,
      output: 40,
      cacheRead: 300,
      cacheWrite: 25,
      totalTokens: 485,
      reasoningTokens: 10,
    },
    stopReason: "stop",
    timestamp: 1_700_000_000_000,
    duration: 2_000,
    ttft: 250,
    ...overrides,
  };
}

async function runAgentLoop(harness: Harness, message = assistantMessage()): Promise<void> {
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await flush();
}

function spanByName(exporters: Exporters, name: string): ReadableSpan {
  const span = exporters.spans.getFinishedSpans().find((candidate) => candidate.name === name);
  expect(span, `expected span ${name}`).toBeDefined();
  return span as ReadableSpan;
}

afterEach(async () => {
  await resetForTesting();
});

test("willContinue agent_end keeps one invoke_agent span until the terminal agent_end", async () => {
  const harness = makeHarness();
  const failed = assistantMessage({ stopReason: "error", errorMessage: "overloaded", content: [] });
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_end", message: failed });
  await harness.emit({ type: "agent_end", messages: [failed], willContinue: true });
  // Auto-retry: the host runs another loop for the same prompt.
  await harness.emit({ type: "agent_start" });
  const message = assistantMessage();
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await flush();

  const agents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(agents).toHaveLength(1);
  const agent = agents[0] as ReadableSpan;
  expect(agent.status.code).not.toBe(SpanStatusCode.ERROR);
  expect(agent.attributes).toMatchObject({
    "gen_ai.input.messages": "fix the bug",
    "gen_ai.output.messages": "All done.",
    "gen_ai.response.finish_reasons": ["stop"],
  });
});

test("continuation agent_start arriving before its willContinue agent_end keeps one span", async () => {
  const harness = makeHarness();
  const failed = assistantMessage({ stopReason: "error", errorMessage: "overloaded", content: [] });
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_end", message: failed });
  // The host launches the agent_end notification without awaiting it, so the
  // continuation's agent_start can reach this extension first.
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "agent_end", messages: [failed], willContinue: true });
  const message = assistantMessage();
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await flush();

  const agents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(agents).toHaveLength(1);
  const agent = agents[0] as ReadableSpan;
  expect(agent.attributes["gen_ai.input.messages"]).toBe("fix the bug");
  expect(agent.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
});

test("a fresh prompt after a missing agent_end closes the dangling span instead of merging", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "before_agent_start", prompt: "first prompt" });
  await harness.emit({ type: "agent_start" });
  // The first loop dies without emitting agent_end; a new prompt starts.
  await harness.emit({ type: "before_agent_start", prompt: "second prompt" });
  await harness.emit({ type: "agent_start" });
  const message = assistantMessage();
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await flush();

  const agents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(agents).toHaveLength(2);
  const [dangling, fresh] = agents as [ReadableSpan, ReadableSpan];
  expect(dangling.attributes["gen_ai.input.messages"]).toBe("first prompt");
  expect(dangling.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  expect(fresh.attributes["gen_ai.input.messages"]).toBe("second prompt");
  expect(fresh.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
});

test("a delayed terminal agent_end does not close or clear the fresh prompt span", async () => {
  const harness = makeHarness();
  const oldMessage = assistantMessage({
    content: [{ type: "text", text: "Old output." }],
    responseId: "resp-old",
  });
  await harness.emit({ type: "before_agent_start", prompt: "first prompt" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_end", message: oldMessage });

  await harness.emit({ type: "before_agent_start", prompt: "second prompt" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "tool_execution_start",
    toolCallId: "call-fresh",
    toolName: "bash",
    args: { command: "pwd" },
  });

  // This terminal notification was launched for the first loop, but an earlier
  // extension delayed its delivery until after the second loop started.
  await harness.emit({ type: "agent_end", messages: [oldMessage] });
  await flush();

  const endedAgents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(endedAgents).toHaveLength(1);
  expect(endedAgents[0]?.attributes["gen_ai.input.messages"]).toBe("first prompt");
  expect(endedAgents[0]?.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);

  await harness.emit({
    type: "tool_execution_end",
    toolCallId: "call-fresh",
    toolName: "bash",
    result: { content: [{ type: "text", text: "/tmp/project" }] },
    isError: false,
  });
  const freshMessage = assistantMessage({
    content: [{ type: "text", text: "Fresh output." }],
    responseId: "resp-fresh",
  });
  await harness.emit({ type: "message_end", message: freshMessage });
  await harness.emit({ type: "agent_end", messages: [freshMessage] });
  await flush();

  const agents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(agents).toHaveLength(2);
  const fresh = agents.find((span) => span.attributes["gen_ai.input.messages"] === "second prompt");
  expect(fresh?.attributes["gen_ai.output.messages"]).toBe("Fresh output.");
  expect(fresh?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  const tool = spanByName(harness.exporters, "execute_tool bash");
  expect(tool.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("agent_end closes the span when message_end receives a copied message", async () => {
  const harness = makeHarness();
  const original = assistantMessage();
  const displayed = {
    ...original,
    content: [{ type: "text", text: "All done." }],
  };
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_end", message: displayed });
  await harness.emit({ type: "agent_end", messages: [original] });
  await flush();

  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(agent.attributes["gen_ai.output.messages"]).toBe("All done.");
  expect(agent.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(agent.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("factory stays assignable to the host ExtensionFactory without host imports in src", () => {
  const factory: ExtensionFactory = telemetryDevExtension();
  expect(factory).toBeInstanceOf(Function);
});

test("agent loop produces an invoke_agent span with prompt input and conversation id", async () => {
  const harness = makeHarness();
  await runAgentLoop(harness);

  const span = spanByName(harness.exporters, "invoke_agent");
  expect(span.attributes).toMatchObject({
    "gen_ai.agent.name": "omp",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.input.messages": "fix the bug",
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.output.messages": "All done.",
    "gen_ai.response.finish_reasons": ["stop"],
  });
  expect(span.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("assistant message_end produces a chat span with usage, timing, and parenting", async () => {
  const harness = makeHarness();
  await runAgentLoop(harness);

  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(chat.attributes).toMatchObject({
    "gen_ai.conversation.id": "session-1",
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-fable-5",
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.response.id": "resp-1",
    "gen_ai.response.time_to_first_chunk": 0.25,
    "gen_ai.usage.cache_creation.input_tokens": 25,
    "gen_ai.usage.cache_read.input_tokens": 300,
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 40,
    "gen_ai.usage.reasoning.output_tokens": 10,
    "gen_ai.usage.total_tokens": 485,
  });
  // startTime = message.timestamp, endTime = timestamp + duration.
  expect(chat.startTime[0]).toBe(1_700_000_000);
  expect(chat.endTime[0]).toBe(1_700_000_002);
});

test("failed assistant message marks the chat and agent spans as errors", async () => {
  const harness = makeHarness();
  await runAgentLoop(
    harness,
    assistantMessage({ stopReason: "error", errorMessage: "overloaded", content: [] }),
  );

  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  expect(chat.status.code).toBe(SpanStatusCode.ERROR);
  expect(chat.attributes["error.type"]).toBe("error");

  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(agent.status.code).toBe(SpanStatusCode.ERROR);
  expect(agent.attributes["error.type"]).toBe("error");
});

test("tool execution produces an execute_tool span with arguments and result", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "tool_execution_start",
    toolCallId: "call-1",
    toolName: "bash",
    args: { command: "ls" },
  });
  await harness.emit({
    type: "tool_execution_end",
    toolCallId: "call-1",
    toolName: "bash",
    result: { content: [{ type: "text", text: "README.md" }] },
    isError: false,
  });
  await harness.emit({ type: "agent_end", messages: [] });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(span.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(span.attributes).toMatchObject({
    "gen_ai.conversation.id": "session-1",
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.call.arguments": JSON.stringify({ command: "ls" }),
    "gen_ai.tool.call.id": "call-1",
    "gen_ai.tool.name": "bash",
  });
  expect(span.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("failing tool execution marks the execute_tool span as an error", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "tool_execution_start",
    toolCallId: "call-2",
    toolName: "bash",
    args: { command: "false" },
  });
  await harness.emit({
    type: "tool_execution_end",
    toolCallId: "call-2",
    toolName: "bash",
    result: { content: [{ type: "text", text: "command failed" }] },
    isError: true,
  });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("ToolExecutionError");
});

test("dangling tool spans are closed when the agent loop ends", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "tool_execution_start",
    toolCallId: "call-3",
    toolName: "bash",
    args: {},
  });
  await harness.emit({ type: "agent_end", messages: [] });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("incomplete");
});

test("session lifecycle events are logged with conversation id and event name", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "session_start" });
  await harness.emit({ type: "session_shutdown" });
  await flush();

  const records = harness.exporters.logs.getFinishedLogRecords();
  expect(records).toHaveLength(2);
  const [start, shutdown] = records;
  expect(start?.eventName).toBe("session_start");
  expect(start?.body).toBe("Session started");
  expect(start?.attributes).toMatchObject({
    "gen_ai.agent.name": "omp",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-fable-5",
    "omp.cwd": "/tmp/project",
  });
  expect(shutdown?.eventName).toBe("session_shutdown");
  expect(shutdown?.severityText).toBe("INFO");
});

test("auto retry and compaction events are logged with severities", async () => {
  const harness = makeHarness();
  await harness.emit({
    type: "auto_retry_start",
    attempt: 2,
    maxAttempts: 5,
    delayMs: 1000,
    errorMessage: "overloaded",
  });
  await harness.emit({
    type: "auto_retry_end",
    success: false,
    attempt: 5,
    finalError: "still overloaded",
  });
  await harness.emit({
    type: "auto_compaction_start",
    reason: "threshold",
    action: "context-full",
  });
  await harness.emit({
    type: "auto_compaction_end",
    action: "context-full",
    result: undefined,
    aborted: false,
    willRetry: false,
  });
  await flush();

  const records = harness.exporters.logs.getFinishedLogRecords();
  expect(records.map((record) => [record.eventName, record.severityText])).toEqual([
    ["auto_retry_start", "WARN"],
    ["auto_retry_end", "ERROR"],
    ["auto_compaction_start", "INFO"],
    ["auto_compaction_end", "INFO"],
  ]);
  expect(records[0]?.attributes).toMatchObject({
    "omp.retry.attempt": 2,
    "omp.retry.max_attempts": 5,
  });
});

test("malformed events never throw and report through onError", async () => {
  const errors: unknown[] = [];
  const harness = makeHarness({ onError: (error) => errors.push(error) });
  const poisoned = {
    type: "message_end",
    get message(): never {
      throw new Error("poisoned event");
    },
  };
  await expect(harness.emit(poisoned as never)).resolves.toBeUndefined();
  expect(errors).toHaveLength(1);

  // The rest of the pipeline still works after a poisoned event.
  await runAgentLoop(harness);
  expect(harness.exporters.spans.getFinishedSpans().length).toBeGreaterThan(0);
});

test("events for a different session carry that session's conversation id", async () => {
  const harness = makeHarness();
  const other = makeContext("session-2");
  await harness.emit({ type: "session_start" }, other);
  await flush();

  const record = harness.exporters.logs.getFinishedLogRecords()[0];
  expect(record?.attributes["gen_ai.conversation.id"]).toBe("session-2");
});

test("tool spans nest under the chat span that issued the tool call", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "message_end",
    message: assistantMessage({
      stopReason: "toolUse",
      content: [{ type: "toolCall", id: "call-9", name: "bash", arguments: { command: "ls" } }],
    }),
  });
  await harness.emit({
    type: "tool_execution_start",
    toolCallId: "call-9",
    toolName: "bash",
    args: { command: "ls" },
  });
  await harness.emit({
    type: "tool_execution_end",
    toolCallId: "call-9",
    toolName: "bash",
    result: { content: [{ type: "text", text: "README.md" }] },
    isError: false,
  });
  await harness.emit({ type: "agent_end", messages: [] });
  await flush();

  const agent = spanByName(harness.exporters, "invoke_agent");
  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  const tool = spanByName(harness.exporters, "execute_tool bash");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(tool.parentSpanContext?.spanId).toBe(chat.spanContext().spanId);
});
