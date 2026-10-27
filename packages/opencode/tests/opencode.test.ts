import { SpanStatusCode, trace } from "@opentelemetry/api";
import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { Hooks, Plugin, PluginModule } from "@opencode-ai/plugin";
import { createOpencodeClient } from "@opencode-ai/sdk/client";
import { flush, type ClientOverrides } from "@telemetry-dev/sdk";
import { afterEach, describe, expect, test, vi } from "vitest";

import { resetForTesting } from "../src/config.ts";
import { telemetryDevPlugin, type TelemetryDevPluginOptions } from "../src/plugin.ts";
import serverModule from "../src/server.ts";

interface Exporters extends ClientOverrides {
  logs: InMemoryLogRecordExporter;
  spans: InMemorySpanExporter;
}

interface Harness {
  exporters: Exporters;
  hooks: Hooks;
}

function makeExporters(): Exporters {
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();

  return { logRecordExporter: logs, logs, spanExporter: spans, spans };
}

function telemetryOptions(options?: TelemetryDevPluginOptions): TelemetryDevPluginOptions {
  return {
    environment: "test",
    exportMode: "immediate" as const,
    logLevel: "silent" as const,
    fetch: async () => new Response(null, { status: 200 }),
    ...options,
  };
}

const pluginInput = {
  client: {},
  project: { id: "proj-1" },
  directory: "/tmp/project",
  worktree: "/tmp/project",
  $: undefined,
  serverUrl: new URL("http://localhost:4096"),
  experimental_workspace: {
    register() {},
  },
};

async function makeHarness(options?: TelemetryDevPluginOptions): Promise<Harness> {
  const exporters = makeExporters();
  const plugin = telemetryDevPlugin(telemetryOptions(options), exporters);
  const hooks = await plugin(pluginInput);

  return { exporters, hooks };
}

async function chatMessage(
  hooks: Hooks,
  sessionID = "session-1",
  parts: JsonValue[] = [{ type: "text", text: "fix the bug" }],
): Promise<void> {
  await hooks["chat.message"]?.({ agent: "build", sessionID }, { message: {}, parts } as never);
}

async function event(hooks: Hooks, type: string, properties: JsonValue): Promise<void> {
  await hooks.event?.({ event: { type, properties } as never });
}

type JsonObject = { [key: string]: JsonValue };

type JsonValue = string | number | boolean | null | JsonValue[] | JsonObject;

function assistantMessage(overrides?: JsonObject) {
  return {
    id: "message-1",
    sessionID: "session-1",
    role: "assistant",
    time: {
      created: 1_700_000_000_000,
      completed: 1_700_000_002_000,
    },
    modelID: "claude-sonnet-4-5",
    providerID: "anthropic",
    finish: "stop",
    cost: 0.0037,
    tokens: {
      input: 120,
      output: 30,
      reasoning: 10,
      cache: { read: 300, write: 25 },
    },
    ...overrides,
  };
}

function spansByName(exporters: Exporters, name: string): ReadableSpan[] {
  return exporters.spans.getFinishedSpans().filter((span) => span.name === name);
}

function spanByName(exporters: Exporters, name: string): ReadableSpan {
  const spans = spansByName(exporters, name);
  expect(spans, `expected one span named ${name}`).toHaveLength(1);

  return spans[0]!;
}

afterEach(async () => {
  vi.restoreAllMocks();
  await resetForTesting();
});

test("chat.message starts an invoke_agent span with prompt text and conversation id", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks, "session-1", [
    { type: "text", text: "fix the bug" },
    { type: "image", url: "image.png" },
    { type: "text", text: "keep the diff small" },
  ]);
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const span = spanByName(harness.exporters, "invoke_agent");
  expect(span.attributes).toMatchObject({
    "gen_ai.agent.name": "opencode",
    "gen_ai.conversation.id": "session-1",
    "gen_ai.input.messages": "fix the bug\nkeep the diff small",
    "gen_ai.operation.name": "invoke_agent",
    "opencode.agent": "build",
  });
  expect(span.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("completed assistant message creates one normalized, retroactive chat span", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  const info = assistantMessage();

  await event(harness.hooks, "message.updated", { info });
  await event(harness.hooks, "message.updated", { info });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const chat = spanByName(harness.exporters, "chat claude-sonnet-4-5");
  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(chat.attributes).toMatchObject({
    "gen_ai.conversation.id": "session-1",
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-sonnet-4-5",
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.usage.cache_creation.input_tokens": 25,
    "gen_ai.usage.cache_read.input_tokens": 300,
    "gen_ai.usage.cost": 0.0037,
    "gen_ai.usage.input_tokens": 120 + 300 + 25,
    "gen_ai.usage.output_tokens": 40,
    "gen_ai.usage.reasoning.output_tokens": 10,
    "gen_ai.usage.total_tokens": 485,
  });
  expect(chat.startTime[0]).toBe(1_700_000_000);
  expect(chat.endTime[0]).toBe(1_700_000_002);
  expect(spansByName(harness.exporters, "chat claude-sonnet-4-5")).toHaveLength(1);
});

test("duplicate chat.message calls do not start a second agent span", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  await chatMessage(harness.hooks);
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  expect(spansByName(harness.exporters, "invoke_agent")).toHaveLength(1);
});

test("tool before and after create an execute_tool span with arguments, result, and start time", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  const now = vi.spyOn(Date, "now").mockReturnValue(1_700_000_010_000);

  await harness.hooks["tool.execute.before"]?.(
    { callID: "call-1", sessionID: "session-1", tool: "bash" },
    { args: { command: "ls" } },
  );
  await harness.hooks["tool.execute.after"]?.(
    { args: { command: "ls" }, callID: "call-1", sessionID: "session-1", tool: "bash" },
    { metadata: { exit: 0 }, output: "README.md", title: "Shell" },
  );
  now.mockRestore();
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(span.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(span.attributes).toMatchObject({
    "gen_ai.conversation.id": "session-1",
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.call.arguments": JSON.stringify({ command: "ls" }),
    "gen_ai.tool.call.id": "call-1",
    "gen_ai.tool.call.result": JSON.stringify({
      title: "Shell",
      output: "README.md",
      metadata: { exit: 0 },
    }),
    "gen_ai.tool.name": "bash",
  });
  expect(span.startTime[0]).toBe(1_700_000_010);
});

test.each([
  { name: "default capture", options: {}, message: "grep: .env:\nTOKEN=private-tool-output" },
  { name: "captureOutput: false", options: { captureOutput: false }, message: undefined },
  {
    name: "a mask",
    options: {
      mask: (value: unknown) =>
        JSON.parse(JSON.stringify(value).replaceAll("private-tool-output", "[redacted]")),
    },
    message: "tool bash failed",
  },
])("message.part.updated reconstructs tool error spans ($name)", async ({ options, message }) => {
  const harness = await makeHarness(options);
  await chatMessage(harness.hooks);
  vi.spyOn(Date, "now").mockReturnValue(1_700_000_020_000);
  await harness.hooks["tool.execute.before"]?.(
    { callID: "call-2", sessionID: "session-1", tool: "bash" },
    { args: { command: "grep TOKEN .env" } },
  );

  await event(harness.hooks, "message.part.updated", {
    part: {
      type: "tool",
      sessionID: "session-1",
      callID: "call-2",
      tool: "bash",
      state: {
        status: "error",
        input: { command: "grep TOKEN .env" },
        error: "grep: .env:\nTOKEN=private-tool-output",
        time: { start: 1_700_000_020_000, end: 1_700_000_021_000 },
      },
    },
  });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool bash");
  const exception = span.events.find((entry) => entry.name === "exception")?.attributes;
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes).toMatchObject({
    "error.type": "ToolExecutionError",
    "gen_ai.tool.call.arguments": JSON.stringify({ command: "grep TOKEN .env" }),
    "gen_ai.tool.call.id": "call-2",
  });
  expect(exception?.["exception.type"]).toBe("ToolExecutionError");
  // The host error text is the tool result: only captured, unmasked output may reach the exception.
  expect(exception?.["exception.message"]).toBe(message);
  if (message === undefined) {
    expect(exception?.["exception.stacktrace"]).toBeUndefined();
  } else {
    const stack = String(exception?.["exception.stacktrace"]);
    expect(stack.startsWith(`ToolExecutionError: ${message}\n`)).toBe(true);
    expect(stack.includes("private-tool-output")).toBe(message.includes("private-tool-output"));
  }
  expect(span.startTime[0]).toBe(1_700_000_020);
  expect(span.endTime[0]).toBe(1_700_000_021);
});

test("dispose ends every dangling agent span as incomplete and flushes", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks, "session-1");
  await chatMessage(harness.hooks, "session-2");

  await harness.hooks.dispose?.();

  const spans = spansByName(harness.exporters, "invoke_agent");
  expect(spans).toHaveLength(2);

  for (const span of spans) {
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  }
});

describe.each([
  {
    name: "default capture",
    options: {},
    message: "File not found: a.ts\nprivate-attachment-listing",
    logMessage: "File not found: a.ts\nprivate-attachment-listing",
  },
  {
    name: "captureOutput: false",
    options: { captureOutput: false },
    message: undefined,
    logMessage: undefined,
  },
  {
    name: "a mask",
    options: {
      mask: (value: unknown) =>
        JSON.parse(JSON.stringify(value).replaceAll("private-attachment-listing", "[redacted]")),
    },
    message: "session error",
    logMessage: "File not found: a.ts\n[redacted]",
  },
])("session.error text follows output capture ($name)", ({ options, message, logMessage }) => {
  test.each(["settle", "dispose"])("on %s", async (ending) => {
    const harness = await makeHarness(options);
    await chatMessage(harness.hooks);
    await event(harness.hooks, "message.updated", {
      info: assistantMessage({ time: { created: 1_700_000_000_000 } }),
    });
    await harness.hooks["tool.execute.before"]?.(
      { callID: "call-5", sessionID: "session-1", tool: "task" },
      { args: { description: "run subtask" } },
    );
    // A failed prompt file attachment publishes the Read tool's error text.
    await event(harness.hooks, "session.error", {
      sessionID: "session-1",
      error: {
        name: "UnknownError",
        data: { message: "File not found: a.ts\nprivate-attachment-listing" },
      },
    });

    if (ending === "settle") {
      await event(harness.hooks, "session.idle", { sessionID: "session-1" });
      await flush();
    } else {
      await harness.hooks.dispose?.();
    }

    for (const name of ["invoke_agent", "chat claude-sonnet-4-5", "execute_tool task"]) {
      const span = spanByName(harness.exporters, name);
      const exception = span.events.find((entry) => entry.name === "exception")?.attributes;
      expect(span.status.code, name).toBe(SpanStatusCode.ERROR);
      expect(span.attributes["error.type"], name).toBe("UnknownError");
      expect(exception?.["exception.message"], name).toBe(message);
      if (message === undefined) {
        expect(exception?.["exception.stacktrace"], name).toBeUndefined();
      } else {
        const stack = String(exception?.["exception.stacktrace"]);
        expect(stack.startsWith(`UnknownError: ${message}\n`), name).toBe(true);
        expect(stack.includes("private-attachment-listing"), name).toBe(
          message !== "session error",
        );
      }
      expect(span.attributes["gen_ai.response.finish_reasons"], name).toBeUndefined();
    }

    const records = harness.exporters.logs.getFinishedLogRecords();
    expect(records).toHaveLength(1);
    expect(records[0]?.body).toBe("Session error");
    expect(records[0]?.eventName).toBe("session.error");
    expect(records[0]?.severityText).toBe("ERROR");
    expect(records[0]?.attributes).toMatchObject({
      "gen_ai.agent.name": "opencode",
      "gen_ai.conversation.id": "session-1",
      "opencode.error.name": "UnknownError",
    });
    expect(records[0]?.attributes["opencode.error.message"]).toBe(logMessage);
  });
});

describe("synthetic Task correlation", () => {
  test("uses the tool part callID when an errored session settles", async () => {
    const harness = await makeHarness();
    await chatMessage(harness.hooks);
    vi.spyOn(Date, "now").mockReturnValue(1_700_000_040_000);
    await event(harness.hooks, "message.part.updated", {
      part: {
        type: "tool",
        id: "part-4",
        sessionID: "session-1",
        callID: "01K1EXAMPLETOOLCALLID",
        tool: "task",
        state: {
          status: "running",
          input: { description: "run subtask" },
          time: { start: 1_700_000_040_000 },
        },
      },
    });
    // Missing-agent synthetic Task: the hook receives part.id, then the host
    // emits session.error and throws before the after hook or a terminal part.
    await harness.hooks["tool.execute.before"]?.(
      { callID: "part-4", sessionID: "session-1", tool: "task" },
      { args: { description: "run subtask" } },
    );
    await event(harness.hooks, "session.error", {
      sessionID: "session-1",
      error: { name: "AgentNotFoundError", data: { message: "agent not found" } },
    });
    await event(harness.hooks, "session.status", {
      sessionID: "session-1",
      status: { type: "idle" },
    });
    await event(harness.hooks, "session.idle", { sessionID: "session-1" });
    await flush();

    const spans = spansByName(harness.exporters, "execute_tool task");
    expect(spans).toHaveLength(1);
    const span = spans[0]!;
    expect(span.status.code).toBe(SpanStatusCode.ERROR);
    expect(span.attributes["error.type"]).toBe("AgentNotFoundError");
    expect(span.attributes["gen_ai.tool.call.id"]).toBe("01K1EXAMPLETOOLCALLID");
    expect(span.startTime[0]).toBe(1_700_000_040);
  });
});

test("poisoned events report through onError without throwing", async () => {
  const errors: unknown[] = [];
  const harness = await makeHarness({ onError: (error) => errors.push(error) });

  const poisoned = {
    get type(): never {
      throw "poisoned event";
    },
  };

  await expect(harness.hooks.event?.({ event: poisoned as never })).resolves.toBeUndefined();
  expect(errors).toHaveLength(1);
  expect(errors[0]).toEqual(new Error("poisoned event"));

  await chatMessage(harness.hooks);
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();
  expect(spansByName(harness.exporters, "invoke_agent")).toHaveLength(1);
});

test("errors go to opencode's log through the plugin client, not to the console", async () => {
  const consoleSpies = (["error", "warn", "info", "debug", "log"] as const).map((method) =>
    vi.spyOn(console, method).mockImplementation(() => {}),
  );
  const logged: { method: string; path: string; body: Record<string, string> }[] = [];

  const client = createOpencodeClient({
    baseUrl: "http://localhost:4096",
    fetch: async (request: Request) => {
      const { method, url } = request;
      const body = (await request.json()) as Record<string, string>;
      logged.push({ method, path: new URL(url).pathname, body });

      return Response.json(true);
    },
  });

  const hooks = await telemetryDevPlugin({
    apiKey: "td_live_test",
    exportMode: "immediate",
    fetch: async () => new Response(null, { status: 400 }),
  })({ ...pluginInput, client });

  await event(hooks, "session.created", { info: { id: "session-1" } });
  await flush();
  await hooks.event?.({
    event: {
      get type(): never {
        throw new Error("poisoned event");
      },
    },
  });

  await vi.waitFor(() => expect(logged).toHaveLength(2));
  expect(logged.map(({ method, path, body }) => [method, path, body.service, body.level])).toEqual([
    ["POST", "/log", "telemetry-dev", "error"],
    ["POST", "/log", "telemetry-dev", "error"],
  ]);
  expect(logged[0]?.body.message).toContain("telemetry.dev log ingest failed: 400");
  expect(logged[1]?.body.message).toContain("poisoned event");
  expect(consoleSpies.flatMap((spy) => spy.mock.calls)).toEqual([]);
});

test("tool spans nest under the session's open chat span", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  // First step finished: the assistant message exists but is not completed yet.
  await event(harness.hooks, "message.updated", {
    info: assistantMessage({ time: { created: 1_700_000_000_000 } }),
  });
  await harness.hooks["tool.execute.before"]?.(
    { tool: "bash", sessionID: "session-1", callID: "call-1" },
    { args: { command: "ls" } },
  );
  await harness.hooks["tool.execute.after"]?.(
    { tool: "bash", sessionID: "session-1", callID: "call-1", args: { command: "ls" } },
    { title: "ls", output: "README.md", metadata: {} },
  );
  await event(harness.hooks, "message.updated", { info: assistantMessage() });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const agent = spanByName(harness.exporters, "invoke_agent");
  const chat = spanByName(harness.exporters, "chat claude-sonnet-4-5");
  const tool = spanByName(harness.exporters, "execute_tool bash");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
  expect(tool.parentSpanContext?.spanId).toBe(chat.spanContext().spanId);
  // Completion fields still land on the eagerly opened span.
  expect(chat.attributes).toMatchObject({
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.usage.cost": 0.0037,
    "gen_ai.usage.input_tokens": 120 + 300 + 25,
  });
  expect(chat.endTime[0]).toBe(1_700_000_002);
});

test("subagent sessions nest under the parent session's span", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks, "session-1");
  await event(harness.hooks, "session.created", {
    info: { id: "session-2", parentID: "session-1" },
  });
  await chatMessage(harness.hooks, "session-2", [{ type: "text", text: "review the diff" }]);
  await event(harness.hooks, "session.idle", { sessionID: "session-2" });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const agents = spansByName(harness.exporters, "invoke_agent");
  expect(agents).toHaveLength(2);
  const parent = agents.find((s) => s.attributes["gen_ai.conversation.id"] === "session-1");
  const child = agents.find((s) => s.attributes["gen_ai.conversation.id"] === "session-2");
  expect(parent).toBeDefined();
  expect(child?.parentSpanContext?.spanId).toBe(parent?.spanContext().spanId);
});

test("open chat spans are closed when the session settles without completing", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  await event(harness.hooks, "message.updated", {
    info: assistantMessage({ time: { created: 1_700_000_000_000 } }),
  });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const chat = spanByName(harness.exporters, "chat claude-sonnet-4-5");
  expect(chat.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
});

test("nonterminal session.error keeps the trace alive until the session settles", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  await event(harness.hooks, "session.error", {
    sessionID: "session-1",
    error: { name: "ContextOverflowError", data: { message: "context window exceeded" } },
  });
  await event(harness.hooks, "session.compacted", { sessionID: "session-1" });
  // The host compacts and continues the same prompt.
  await event(harness.hooks, "message.updated", { info: assistantMessage() });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const agent = spanByName(harness.exporters, "invoke_agent");
  expect(agent.status.code).not.toBe(SpanStatusCode.ERROR);
  // The continued generation stays parented under the original prompt span.
  const chat = spanByName(harness.exporters, "chat claude-sonnet-4-5");
  expect(chat.parentSpanContext?.spanId).toBe(agent.spanContext().spanId);
});

test("failed Task produces an execute_tool error span despite undefined after-hook output", async () => {
  const harness = await makeHarness();
  await chatMessage(harness.hooks);
  vi.spyOn(Date, "now").mockReturnValue(1_700_000_030_000);
  await harness.hooks["tool.execute.before"]?.(
    // The synthetic Task path keys hook calls by part id, not by LLM callID.
    { callID: "part-1", sessionID: "session-1", tool: "task" },
    { args: { description: "run subtask" } },
  );
  await harness.hooks["tool.execute.after"]?.(
    {
      args: { description: "run subtask" },
      callID: "part-1",
      sessionID: "session-1",
      tool: "task",
    },
    undefined as never,
  );

  await event(harness.hooks, "message.part.updated", {
    part: {
      type: "tool",
      id: "part-1",
      sessionID: "session-1",
      callID: "call-9",
      tool: "task",
      state: {
        status: "error",
        input: { description: "run subtask" },
        error: "Tool execution failed: subtask crashed",
        time: { start: 1_700_000_030_000, end: 1_700_000_031_000 },
      },
    },
  });
  await event(harness.hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  const span = spanByName(harness.exporters, "execute_tool task");
  expect(span.status.code).toBe(SpanStatusCode.ERROR);
  expect(span.attributes).toMatchObject({
    "error.type": "ToolExecutionError",
    "gen_ai.tool.call.id": "call-9",
  });
  expect(span.startTime[0]).toBe(1_700_000_030);
  expect(span.endTime[0]).toBe(1_700_000_031);
});

test("configured tuple options cannot enable global tracer registration", async () => {
  const setGlobal = vi.spyOn(trace, "setGlobalTracerProvider");
  const exporters = makeExporters();
  const plugin = telemetryDevPlugin(telemetryOptions(), exporters);
  const hooks: Hooks = await plugin(pluginInput, { registerGlobal: true });
  await chatMessage(hooks);
  await event(hooks, "session.idle", { sessionID: "session-1" });
  await flush();

  expect(setGlobal).not.toHaveBeenCalled();
  expect(spansByName(exporters, "invoke_agent")).toHaveLength(1);
});

test("plugin and server module stay assignable to the host plugin types", () => {
  const plugin: Plugin = telemetryDevPlugin();
  const module: PluginModule = serverModule;
  expect(plugin).toBeInstanceOf(Function);
  expect(module.id).toBe("telemetry-dev");
});
