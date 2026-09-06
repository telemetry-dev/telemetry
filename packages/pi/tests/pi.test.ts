import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SpanStatusCode } from "@opentelemetry/api";
import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { flush, type ClientOverrides } from "@telemetry-dev/sdk";
import {
  convertToLlm,
  createExtensionRuntime,
  createSyntheticSourceInfo,
  ExtensionRunner,
  ModelRegistry,
  ModelRuntime,
  SessionManager,
  type Extension,
  type ExtensionContext,
  type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
import { afterEach, expect, test, vi } from "vitest";

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
  return {
    environment: "test",
    exportMode: "immediate" as const,
    logLevel: "silent" as const,
    fetch: async () => new Response("{}", { status: 200 }),
    ...options,
  };
}

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
type JsonRecord = { [key: string]: JsonValue };
type Event = { type: string; [key: string]: JsonValue };
type EmitFn = (event: Event, ctx?: ExtensionContext) => Promise<void>;

interface Harness {
  emit: EmitFn;
  exporters: Exporters;
}

function makeContext(sessionId = "session-1"): ExtensionContext {
  const context = {
    cwd: "/tmp/project",
    model: { id: "claude-fable-5", provider: "anthropic" },
    getSystemPrompt: () => "You are pi.",
    sessionManager: {
      getSessionId: () => sessionId,
      getCwd: () => "/tmp/project",
      getSessionName: () => undefined,
      getSessionFile: () => undefined,
    },
  };
  return context as ExtensionContext;
}

function makeHarness(options?: TelemetryDevExtensionOptions): Harness {
  const exporters = makeExporters();
  type Handler = (event: Event, ctx: ExtensionContext) => void | Promise<void>;
  const handlers = new Map<string, Handler[]>();
  const api = {
    on(type: string, handler: Handler) {
      const list = handlers.get(type) ?? [];
      list.push(handler);
      handlers.set(type, list);
    },
  };

  const factory = telemetryDevExtension(telemetryOptions(options), exporters);
  factory(api);

  const defaultCtx = makeContext();
  const emit: EmitFn = async (event, ctx = defaultCtx) => {
    for (const handler of handlers.get(event.type) ?? []) {
      await handler(event, ctx);
    }
  };
  return { emit, exporters };
}

function assistantMessage(overrides?: JsonRecord) {
  const message = {
    role: "assistant",
    content: [{ type: "text", text: "All done." }],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-fable-5",
    responseModel: "claude-fable-5-20260701",
    responseId: "resp-1",
    usage: {
      input: 120,
      output: 40,
      cacheRead: 300,
      cacheWrite: 25,
      reasoning: 10,
      totalTokens: 485,
      cost: {
        input: 0.001,
        output: 0.002,
        cacheRead: 0.0003,
        cacheWrite: 0.0004,
        total: 0.0037,
      },
    },
    stopReason: "stop",
    timestamp: 1_700_000_000_000,
    ...overrides,
  };
  return message;
}

async function runAgentLoop(
  harness: Harness,
  message = assistantMessage(),
  ctx?: ExtensionContext,
): Promise<void> {
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" }, ctx);
  await harness.emit({ type: "agent_start" }, ctx);
  await harness.emit({ type: "message_start", message }, ctx);
  await harness.emit({ type: "message_end", message }, ctx);
  await harness.emit({ type: "agent_end", messages: [message] }, ctx);
  await harness.emit({ type: "agent_settled" }, ctx);
  await flush();
}

function spanByName(exporters: Exporters, name: string): ReadableSpan {
  const span = exporters.spans.getFinishedSpans().find((candidate) => candidate.name === name);
  expect(span, `expected span ${name}`).toBeDefined();
  return span as ReadableSpan;
}

afterEach(async () => {
  vi.restoreAllMocks();
  await resetForTesting();
});

test("agent loop produces an invoke_agent span with prompt input and conversation id", async () => {
  const harness = makeHarness();
  await runAgentLoop(harness);

  const span = spanByName(harness.exporters, "invoke_agent");
  expect(span.attributes).toMatchObject({
    "gen_ai.agent.name": "pi",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.input.messages": "fix the bug",
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.output.messages": JSON.stringify([{ type: "text", text: "All done." }]),
    "gen_ai.response.finish_reasons": ["stop"],
  });
  expect(span.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("automatic retries keep one invoke_agent span until the run settles", async () => {
  const harness = makeHarness();
  const failed = assistantMessage({ stopReason: "error", errorMessage: "overloaded", content: [] });
  await harness.emit({ type: "before_agent_start", prompt: "fix the bug" });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_start", message: failed });
  await harness.emit({ type: "message_end", message: failed });
  await harness.emit({ type: "agent_end", messages: [failed] });
  // Auto-retry: pi removes the failure and runs another loop before settling.
  await harness.emit({ type: "agent_start" });
  const message = assistantMessage();
  await harness.emit({ type: "message_start", message });
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await harness.emit({ type: "agent_settled" });
  await flush();

  const agents = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "invoke_agent");
  expect(agents).toHaveLength(1);
  const agent = agents[0] as ReadableSpan;
  expect(agent.status.code).not.toBe(SpanStatusCode.ERROR);
  expect(agent.attributes).toMatchObject({
    "gen_ai.input.messages": "fix the bug",
    "gen_ai.output.messages": JSON.stringify([{ type: "text", text: "All done." }]),
    "gen_ai.response.finish_reasons": ["stop"],
  });
});

test("factory stays assignable to the host ExtensionFactory without host imports in src", () => {
  const factory: ExtensionFactory = telemetryDevExtension();
  expect(factory).toBeInstanceOf(Function);
});

test("all prompts of a pi session share one trace under a session root span", async () => {
  const networkFetch = vi
    .spyOn(globalThis, "fetch")
    .mockRejectedValue(new Error("unexpected network"));
  const harness = makeHarness({ apiKey: "td_test_key" });
  await harness.emit({ type: "session_start", reason: "startup" });
  await runAgentLoop(harness);
  await runAgentLoop(harness);
  await harness.emit({ type: "session_shutdown", reason: "quit" });
  await flush();
  expect(networkFetch).not.toHaveBeenCalled();

  const spans = harness.exporters.spans.getFinishedSpans();
  const agents = spans.filter((span) => span.name === "invoke_agent");
  const root = spanByName(harness.exporters, "session");
  expect(agents).toHaveLength(2);
  expect(new Set(spans.map((span) => span.spanContext().traceId)).size).toBe(1);
  expect(agents.map((span) => span.parentSpanContext?.spanId)).toEqual([
    root.spanContext().spanId,
    root.spanContext().spanId,
  ]);
  // The session root hangs off the never-stored session parent:
  // sha256("td_test_key\0session-1")[16:24].
  expect(root.parentSpanContext?.spanId).toBe("687ec1e59f5a78cc");
  expect(root.attributes["gen_ai.conversation.id"]).toBe("session-1");
});

test("a session switch closes the trace and the next prompt starts a new one", async () => {
  const harness = makeHarness();
  await runAgentLoop(harness);
  const next = makeContext("session-2");
  await harness.emit({ type: "session_start", reason: "new" }, next);
  await runAgentLoop(harness, assistantMessage(), next);
  await harness.emit({ type: "session_shutdown", reason: "quit" }, next);
  await flush();

  const roots = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "session");
  expect(roots).toHaveLength(2);
  expect(roots[0]?.spanContext().traceId).not.toBe(roots[1]?.spanContext().traceId);
});

test("assistant lifecycle produces a timed chat span with model, usage, and cost", async () => {
  const networkFetch = vi
    .spyOn(globalThis, "fetch")
    .mockRejectedValue(new Error("unexpected network"));
  let now = 1_700_000_000_000;
  vi.spyOn(Date, "now").mockImplementation(() => now);
  const harness = makeHarness();
  const message = assistantMessage();
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "message_start", message });
  now += 2_000;
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await harness.emit({ type: "agent_settled" });
  await flush();
  expect(networkFetch).not.toHaveBeenCalled();

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
    "gen_ai.response.model": "claude-fable-5-20260701",
    "gen_ai.usage.cache_creation.input_tokens": 25,
    "gen_ai.usage.cache_read.input_tokens": 300,
    "gen_ai.usage.cost": 0.0037,
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 40,
    "gen_ai.usage.reasoning.output_tokens": 10,
    "gen_ai.usage.total_tokens": 485,
  });
  expect(chat.startTime[0]).toBe(1_700_000_000);
  expect(chat.endTime[0]).toBe(1_700_000_002);
});

test("chat spans carry the system prompt, the model context, and every reply block", async () => {
  const harness = makeHarness();
  const context: JsonValue[] = [{ role: "user", content: [{ type: "text", text: "fix the bug" }] }];
  const content: JsonValue[] = [
    { type: "thinking", thinking: "Look first." },
    { type: "text", text: "Reading." },
    { type: "toolCall", id: "call-1", name: "read", arguments: { path: "a.ts" } },
  ];
  const message = assistantMessage({ stopReason: "toolUse", content });
  await harness.emit({
    type: "before_agent_start",
    prompt: "fix the bug",
    images: [{ type: "image", data: "aGk=", mimeType: "image/png" }],
    systemPrompt: "You are pi.",
  });
  await harness.emit({ type: "agent_start" });
  await harness.emit({ type: "before_provider_request", payload: { messages: context } });
  await harness.emit({ type: "message_start", message });
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await harness.emit({ type: "agent_settled" });
  await flush();

  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  expect(chat.attributes).toMatchObject({
    "gen_ai.system_instructions": "You are pi.",
    "gen_ai.input.messages": JSON.stringify({ messages: context }),
    "gen_ai.output.messages": JSON.stringify(content),
  });
  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(agent.attributes["gen_ai.input.messages"]).toBe(
    JSON.stringify([
      { type: "text", text: "fix the bug" },
      { type: "image", data: "aGk=", mimeType: "image/png" },
    ]),
  );
  expect(agent.attributes["gen_ai.output.messages"]).toBe(JSON.stringify(content));
});

test("request capture follows host conversion and later context and system-prompt replacements", async () => {
  const directory = await mkdtemp(join(tmpdir(), "telemetry-pi-capture-"));
  try {
    const exporters = makeExporters();
    const runtime = createExtensionRuntime();
    function extension(path: string): Extension {
      return {
        path,
        resolvedPath: path,
        sourceInfo: createSyntheticSourceInfo(path, { source: "test" }),
        handlers: new Map(),
        tools: new Map(),
        messageRenderers: new Map(),
        commands: new Map(),
        flags: new Map(),
        shortcuts: new Map(),
      };
    }
    const telemetry = extension("telemetry");
    telemetryDevExtension(
      telemetryOptions(),
      exporters,
    )({
      on(type, handler) {
        const handlers = telemetry.handlers.get(type) ?? [];
        handlers.push(handler);
        telemetry.handlers.set(type, handlers);
      },
    });
    const original: Parameters<typeof convertToLlm>[0] = [
      { role: "user", content: "obsolete context", timestamp: 1 },
      {
        role: "bashExecution",
        command: "printenv PRIVATE_TOKEN",
        output: "private-shell-output",
        exitCode: 0,
        cancelled: false,
        truncated: false,
        excludeFromContext: true,
        timestamp: 3,
      },
      {
        role: "bashExecution",
        command: "pwd",
        output: "/project",
        exitCode: 0,
        cancelled: false,
        truncated: false,
        timestamp: 4,
      },
      {
        role: "custom",
        customType: "private-extension-state",
        content: "public note",
        display: false,
        details: { secret: "private-custom-details" },
        timestamp: 5,
      },
    ];
    const later = extension("later");
    later.handlers.set("context", [
      async () => ({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "final context" },
              { type: "image", data: "aGk=", mimeType: "image/png" },
            ],
            timestamp: 2,
          },
          ...original.slice(1),
        ] satisfies Parameters<typeof convertToLlm>[0],
      }),
    ]);
    later.handlers.set("before_agent_start", [
      async () => ({ systemPrompt: "Effective system instructions." }),
    ]);
    const runner = new ExtensionRunner(
      [telemetry, later],
      runtime,
      directory,
      SessionManager.inMemory(directory),
      new ModelRegistry(
        await ModelRuntime.create({ authPath: join(directory, "auth.json"), modelsPath: null }),
      ),
    );
    let systemPrompt = "Base system instructions.";
    runner.bindCore(runtime, {
      getModel: () => undefined,
      isIdle: () => false,
      isProjectTrusted: () => true,
      getSignal: () => undefined,
      abort: () => {},
      hasPendingMessages: () => false,
      shutdown: () => {},
      getContextUsage: () => undefined,
      compact: () => {},
      getSystemPrompt: () => systemPrompt,
    });
    const beforeStart = await runner.emitBeforeAgentStart("fix the bug", undefined, systemPrompt, {
      cwd: directory,
    });
    systemPrompt = beforeStart?.systemPrompt ?? systemPrompt;
    await runner.emit({ type: "agent_start" });
    const messages = convertToLlm(await runner.emitContext(original));
    await runner.emitBeforeProviderRequest({ messages });
    const message = {
      ...assistantMessage(),
      role: "assistant" as const,
      content: [{ type: "text" as const, text: "All done." }],
      stopReason: "stop" as const,
    };
    await runner.emit({ type: "message_start", message });
    await runner.emitMessageEnd({ type: "message_end", message });
    await runner.emit({ type: "agent_end", messages: [message] });
    await runner.emit({ type: "agent_settled" });
    await runner.emit({ type: "session_shutdown", reason: "quit" });
    await flush();

    const chat = spanByName(exporters, "chat claude-fable-5");
    expect(chat.attributes["gen_ai.input.messages"]).toBe(
      JSON.stringify({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "final context" },
              { type: "image", data: "aGk=", mimeType: "image/png" },
            ],
            timestamp: 2,
          },
          {
            role: "user",
            content: [{ type: "text", text: "Ran `pwd`\n```\n/project\n```" }],
            timestamp: 4,
          },
          {
            role: "user",
            content: [{ type: "text", text: "public note" }],
            timestamp: 5,
          },
        ],
      }),
    );
    expect(chat.attributes["gen_ai.system_instructions"]).toBe("Effective system instructions.");
    const captured = JSON.stringify(
      exporters.spans.getFinishedSpans().map((span) => span.attributes),
    );
    expect(captured).not.toContain("PRIVATE_TOKEN");
    expect(captured).not.toContain("private-shell-output");
    expect(captured).not.toContain("private-custom-details");
    expect(captured).not.toContain("obsolete context");
    expect(captured).not.toContain("Base system instructions.");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("provider request and effective system instructions obey captureInput", async () => {
  const harness = makeHarness({ captureInput: false });
  await harness.emit({
    type: "before_agent_start",
    prompt: "private prompt",
    images: [{ type: "image", data: "private-image", mimeType: "image/png" }],
  });
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "before_provider_request",
    payload: { messages: [{ role: "user", content: "private request" }] },
  });
  const message = assistantMessage();
  await harness.emit({ type: "message_start", message });
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await harness.emit({ type: "agent_settled" });
  await flush();

  for (const name of ["invoke_agent", "chat claude-fable-5"]) {
    const span = spanByName(harness.exporters, name);
    expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(span.attributes["gen_ai.system_instructions"]).toBeUndefined();
    expect(span.attributes["gen_ai.output.messages"]).toBe(
      JSON.stringify([{ type: "text", text: "All done." }]),
    );
  }
});

test("provider request and effective system instructions pass through SDK masking", async () => {
  const harness = makeHarness({
    mask: (_value, { key }) =>
      key === "gen_ai.system_instructions" ? "[masked instructions]" : "[masked content]",
  });
  await harness.emit({ type: "agent_start" });
  await harness.emit({
    type: "before_provider_request",
    payload: { messages: [{ role: "user", content: "private request" }] },
  });
  const message = assistantMessage();
  await harness.emit({ type: "message_start", message });
  await harness.emit({ type: "message_end", message });
  await harness.emit({ type: "agent_end", messages: [message] });
  await harness.emit({ type: "agent_settled" });
  await flush();

  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  expect(chat.attributes["gen_ai.input.messages"]).toBe("[masked content]");
  expect(chat.attributes["gen_ai.system_instructions"]).toBe("[masked instructions]");
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

test("tool executions produce successful and failed execute_tool spans", async () => {
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
  await harness.emit({ type: "agent_end", messages: [] });
  await harness.emit({ type: "agent_settled" });
  await flush();

  const spans = harness.exporters.spans
    .getFinishedSpans()
    .filter((span) => span.name === "execute_tool bash");
  expect(spans).toHaveLength(2);
  expect(spans[0]?.attributes).toMatchObject({
    "gen_ai.conversation.id": "session-1",
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.call.arguments": JSON.stringify({ command: "ls" }),
    "gen_ai.tool.call.id": "call-1",
    "gen_ai.tool.name": "bash",
  });
  expect(spans[0]?.status.code).not.toBe(SpanStatusCode.ERROR);
  expect(spans[1]?.status.code).toBe(SpanStatusCode.ERROR);
  expect(spans[1]?.attributes["error.type"]).toBe("ToolExecutionError");
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
  await harness.emit({ type: "agent_settled" });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes["error.type"]).toBe("incomplete");
});

test("session lifecycle logs include reasons, model context, and conversation id", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "session_start", reason: "startup" });
  await harness.emit({ type: "session_shutdown", reason: "quit" });
  await flush();

  const records = harness.exporters.logs.getFinishedLogRecords();
  expect(records).toHaveLength(2);
  const [start, shutdown] = records;
  expect(start?.eventName).toBe("session_start");
  expect(start?.body).toBe("Session started");
  expect(start?.attributes).toMatchObject({
    "gen_ai.agent.name": "pi",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-fable-5",
    "pi.cwd": "/tmp/project",
    "pi.session.reason": "startup",
  });
  expect(shutdown?.eventName).toBe("session_shutdown");
  expect(shutdown?.severityText).toBe("INFO");
  expect(shutdown?.attributes["pi.session.reason"]).toBe("quit");
});

test("model_select logs the previous and selected model", async () => {
  const harness = makeHarness();
  await harness.emit({
    type: "model_select",
    model: { id: "gpt-5.6" },
    previousModel: { id: "claude-fable-5" },
    source: "set",
  });
  await flush();

  const [record] = harness.exporters.logs.getFinishedLogRecords();
  expect(record?.eventName).toBe("model_select");
  expect(record?.body).toBe("Model changed");
  expect(record?.attributes).toMatchObject({
    "pi.model.from": "claude-fable-5",
    "pi.model.source": "set",
    "pi.model.to": "gpt-5.6",
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
  await expect(harness.emit(poisoned)).resolves.toBeUndefined();
  expect(errors).toHaveLength(1);

  await runAgentLoop(harness);
  expect(harness.exporters.spans.getFinishedSpans().length).toBeGreaterThan(0);
});

test("each event reads the current session id", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "session_start", reason: "startup" }, makeContext("session-2"));
  await harness.emit({ type: "model_select", model: { id: "gpt-5.6" }, source: "set" });
  await flush();

  const records = harness.exporters.logs.getFinishedLogRecords();
  expect(records[0]?.attributes["gen_ai.conversation.id"]).toBe("session-2");
  expect(records[1]?.attributes["gen_ai.conversation.id"]).toBe("session-1");
});

test("tool spans nest under the chat span that issued the tool call", async () => {
  const harness = makeHarness();
  await harness.emit({ type: "agent_start" });
  const message = assistantMessage({
    stopReason: "toolUse",
    content: [{ type: "toolCall", id: "call-9", name: "bash", arguments: { command: "ls" } }],
  });
  await harness.emit({ type: "message_start", message });
  await harness.emit({ type: "message_end", message });
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
  await harness.emit({ type: "agent_settled" });
  await flush();

  const agent = spanByName(harness.exporters, "invoke_agent");
  const chat = spanByName(harness.exporters, "chat claude-fable-5");
  const tool = spanByName(harness.exporters, "execute_tool bash");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(tool.parentSpanContext?.spanId).toBe(chat.spanContext().spanId);
});
