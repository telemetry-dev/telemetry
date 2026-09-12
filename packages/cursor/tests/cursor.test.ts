import { execFileSync } from "node:child_process";
import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { createConnection } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SpanStatusCode } from "@opentelemetry/api";
import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { flush, type ClientOverrides } from "@telemetry-dev/sdk";
import { afterEach, describe, expect, expectTypeOf, test, vi } from "vitest";
import { sessionSpanContext } from "../../otel/src/index.ts";

import { fileConfig, resetForTesting, type TelemetryDevCursorOptions } from "../src/config.ts";
import { listen } from "../src/daemon.ts";
import { runInstall, runUninstall } from "../src/install.ts";
import { createCursorTelemetry, type CursorTelemetry } from "../src/telemetry.ts";
import { createTitleLookup } from "../src/titles.ts";

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue;
}

interface Harness {
  spans: InMemorySpanExporter;
  logs: InMemoryLogRecordExporter;
  telemetry: CursorTelemetry;
}

/** Nonexistent by design: title lookups must never touch the real ~/.cursor. */
const isolatedChatsDir = join(tmpdir(), "telemetry-dev-cursor-tests-no-chats");
const apiKey = "td_live_cursor_test";

function makeHarness(options: TelemetryDevCursorOptions = {}): Harness {
  const spans = new InMemorySpanExporter();
  const logs = new InMemoryLogRecordExporter();
  const overrides: ClientOverrides = { spanExporter: spans, logRecordExporter: logs };

  const telemetry = createCursorTelemetry(
    {
      apiKey,
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      chatsDir: isolatedChatsDir,
      fetch: async () => new Response(null, { status: 200 }),
      ...options,
    },
    overrides,
  );

  return { spans, logs, telemetry };
}

/**
 * Redirects homedir() to a temp dir on every platform, and the daemon socket
 * directory with it so install/uninstall tests cannot signal a real daemon.
 * Returns a restore fn.
 */
function setHome(home: string): () => void {
  const previous = {
    HOME: process.env.HOME,
    USERPROFILE: process.env.USERPROFILE,
    XDG_RUNTIME_DIR: process.env.XDG_RUNTIME_DIR,
  };

  process.env.HOME = home;
  process.env.USERPROFILE = home;
  process.env.XDG_RUNTIME_DIR = home;

  return () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };
}

function event(name: string, fields: JsonRecord = {}) {
  return {
    hook_event_name: name,
    conversation_id: "conv-1",
    generation_id: "gen-1",
    model: "claude-opus-4-7-thinking-max",
    model_id: "claude-opus-4-7",
    cursor_version: "3.12.0",
    workspace_roots: ["/tmp/project"],
    user_email: "dev@example.com",
    ...fields,
  };
}

function finished(spans: InMemorySpanExporter): ReadableSpan[] {
  return spans.getFinishedSpans();
}

/** Turn spans for one agent name; span names now prefer titles/prompts/tasks. */
function agentSpans(spans: InMemorySpanExporter, name: string): ReadableSpan[] {
  return finished(spans).filter(
    (s) =>
      s.attributes["gen_ai.operation.name"] === "invoke_agent" &&
      s.attributes["gen_ai.agent.name"] === name,
  );
}

afterEach(async () => {
  await resetForTesting();
});

describe("turn lifecycle", () => {
  test("a turn without a chat title takes the prompt as its name", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(
      event("beforeSubmitPrompt", { prompt: "fix the login bug\nwith more detail" }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const agent = finished(spans).find((s) => s.attributes["gen_ai.conversation.id"] === "conv-1");
    expect(agent!.name).toBe("fix the login bug");
  });

  test("beforeSubmitPrompt then stop produces one invoke_agent span", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "fix the bug" }));
    telemetry.handle(event("stop", { status: "completed", loop_count: 0 }));
    await flush();

    const agent = agentSpans(spans, "cursor")[0];
    expect(agent).toBeDefined();
    expect(agent!.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
    expect(agent!.attributes["gen_ai.conversation.id"]).toBe("conv-1");
    expect(agent!.attributes["gen_ai.agent.name"]).toBe("cursor");
    expect(agent!.attributes["gen_ai.input.messages"]).toContain("fix the bug");
    expect(agent!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
    expect(agent!.attributes["cursor.user_email"]).toBe("dev@example.com");
    expect(agent!.attributes["cursor.workspace"]).toBe("/tmp/project");
  });

  test("stop with error status marks the turn span failed", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "do it" }));
    telemetry.handle(event("stop", { status: "error", loop_count: 0 }));
    await flush();

    const agent = agentSpans(spans, "cursor")[0];
    expect(agent!.status.code).toBe(SpanStatusCode.ERROR);
    expect(agent!.attributes["error.type"]).toBe("AgentError");
  });

  test("a second beforeSubmitPrompt ends the stale turn as incomplete", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "first" }));
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "second", generation_id: "gen-2" }));
    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-2" }));
    await flush();

    const agents = agentSpans(spans, "cursor");
    expect(agents).toHaveLength(2);
    expect(agents[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  });

  test("generation_id drift within a turn does not split it", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "one turn" }));
    telemetry.handle(
      event("afterAgentThought", { text: "t", duration_ms: 5, generation_id: "gen-2" }),
    );
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Shell",
        tool_use_id: "t1",
        tool_output: "{}",
        duration: 10,
        generation_id: "gen-3",
      }),
    );
    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-3" }));
    await flush();

    const agents = agentSpans(spans, "cursor");
    expect(agents).toHaveLength(1);
    expect(agents[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
  });

  test("events without beforeSubmitPrompt still open a turn (CLI flows)", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Shell",
        tool_input: { command: "ls" },
        tool_output: '{"exitCode":0}',
        tool_use_id: "t1",
        duration: 50,
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const names = finished(spans).map((s) => s.name);
    expect(names).toContain("invoke_agent cursor");
    expect(names).toContain("execute_tool Shell");
  });
});

describe("tool spans", () => {
  test("preToolUse + postToolUse produce a nested execute_tool span", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "run tests" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Shell",
        tool_input: { command: "npm test" },
        tool_use_id: "t1",
      }),
    );
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Shell",
        tool_input: { command: "npm test" },
        tool_output: '{"exitCode":0,"stdout":"ok"}',
        tool_use_id: "t1",
        duration: 1234,
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const all = finished(spans);
    const tool = all.find((s) => s.name === "execute_tool Shell");
    const agent = agentSpans(spans, "cursor")[0];
    expect(tool).toBeDefined();
    expect(tool!.parentSpanContext?.spanId).toBe(agent!.spanContext().spanId);
    expect(tool!.spanContext().traceId).toBe(agent!.spanContext().traceId);
    expect(tool!.attributes["gen_ai.tool.name"]).toBe("Shell");
    expect(tool!.attributes["gen_ai.tool.call.id"]).toBe("t1");
    expect(tool!.attributes["gen_ai.tool.call.result"]).toContain("exitCode");
    // Start time comes from the paired preToolUse (milliseconds ago), not the
    // reported duration fallback (which would make the span ~1234ms long).
    const durationMs = tool!.duration[0] * 1000 + tool!.duration[1] / 1e6;
    expect(durationMs).toBeLessThan(1000);
  });

  test("postToolUseFailure records an error tool span", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
    telemetry.handle(
      event("postToolUseFailure", {
        tool_name: "Shell",
        tool_input: { command: "npm test" },
        tool_use_id: "t1",
        error_message: "Command timed out after 30s",
        failure_type: "timeout",
        duration: 30000,
      }),
    );
    await flush();

    const tool = finished(spans).find((s) => s.name === "execute_tool Shell");
    expect(tool!.status.code).toBe(SpanStatusCode.ERROR);
    expect(tool!.attributes["error.type"]).toBe("timeout");
  });
});

describe("responses and thoughts", () => {
  test("afterAgentResponse emits a chat span and updates turn output", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "hello" }));
    telemetry.handle(event("afterAgentResponse", { text: "done, all tests pass" }));
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const all = finished(spans);
    const chat = all.find((s) => s.name === "chat claude-opus-4-7");
    const agent = agentSpans(spans, "cursor")[0];
    expect(chat).toBeDefined();
    expect(chat!.attributes["gen_ai.operation.name"]).toBe("chat");
    expect(chat!.attributes["gen_ai.request.model"]).toBe("claude-opus-4-7");
    expect(chat!.attributes["gen_ai.output.messages"]).toContain("all tests pass");
    expect(chat!.parentSpanContext?.spanId).toBe(agent!.spanContext().spanId);
    expect(agent!.attributes["gen_ai.output.messages"]).toContain("all tests pass");
  });

  test("afterAgentThought emits a duration-backed thought span", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "hello" }));
    telemetry.handle(
      event("afterAgentThought", { text: "considering options", duration_ms: 5000 }),
    );
    await flush();

    const thought = finished(spans).find((s) => s.name === "thought");
    expect(thought).toBeDefined();
    const durationMs = thought!.duration[0] * 1000 + thought!.duration[1] / 1e6;
    expect(durationMs).toBeGreaterThanOrEqual(4900);
  });
});

describe("subagents", () => {
  test("parallel workers share one Task call id and one result closes both", async () => {
    // Headless workers use their conversation id as the generation id. One
    // Task call can fan out parallel workers with the same tool_use_id.
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));

    for (let i = 0; i < 2; i++) {
      telemetry.handle(
        event("preToolUse", {
          tool_name: "Task",
          tool_use_id: "task-shared",
          tool_input: { prompt: "explore in parallel", subagent_type: "code-explorer" },
        }),
      );
    }

    for (const sub of ["sub-a", "sub-b"]) {
      telemetry.handle(
        event("preToolUse", {
          conversation_id: sub,
          generation_id: sub,
          tool_name: "Shell",
          tool_use_id: `${sub}-t1`,
          tool_input: { command: "ls" },
        }),
      );
      telemetry.handle(
        event("postToolUse", {
          conversation_id: sub,
          generation_id: sub,
          tool_name: "Shell",
          tool_use_id: `${sub}-t1`,
          tool_output: "files",
        }),
      );
    }

    telemetry.handle(
      event("postToolUse", {
        tool_name: "Task",
        tool_use_id: "task-shared",
        tool_output: "both done",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const parent = agentSpans(spans, "cursor")[0]!;
    const subs = agentSpans(spans, "code-explorer");
    expect(subs).toHaveLength(2);

    for (const sub of subs) {
      expect(sub.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
      expect(sub.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
    }
  });

  test("a parent turn that ends closes its open subagent turns", async () => {
    // Headless sessions send no postToolUse for Task; sessionEnd/stop on the
    // parent must not leave subagent spans open.
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-1",
        tool_input: { prompt: "look around", subagent_type: "explore" },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "sub-1",
        generation_id: "sub-1",
        tool_name: "Shell",
        tool_use_id: "s1",
        tool_input: { command: "ls" },
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const sub = agentSpans(spans, "explore")[0];
    expect(sub).toBeDefined();
    expect(sub!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  });

  test("a subagent conversation with no subagentStart claims the pending Task call and nests", async () => {
    // Real cursor-agent CLI shape: no subagentStart/subagentStop events; the
    // parent emits preToolUse/postToolUse for Task, and the subagent runs as
    // its own conversation with no parent-linking fields.
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_input: {
          description: "List files",
          prompt: "List all files",
          subagent_type: "code-explorer",
        },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "sub-conv-1",
        generation_id: "sub-conv-1",
        tool_name: "Shell",
        tool_use_id: "shell-1",
        tool_input: { command: "ls" },
      }),
    );
    telemetry.handle(
      event("postToolUse", {
        conversation_id: "sub-conv-1",
        generation_id: "sub-conv-1",
        tool_name: "Shell",
        tool_use_id: "shell-1",
        tool_input: { command: "ls" },
        tool_output: "files",
      }),
    );
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_output: "the listing",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const parent = agentSpans(spans, "cursor")[0]!;
    const sub = agentSpans(spans, "code-explorer")[0]!;
    expect(sub.spanContext().traceId).toBe(parent.spanContext().traceId);
    expect(sub.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
    expect(sub.attributes["gen_ai.input.messages"]).toContain("List all files");
    expect(sub.attributes["gen_ai.output.messages"]).toContain("the listing");
    expect(sub.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);

    const shell = finished(spans).find((s) => s.name === "execute_tool Shell")!;
    expect(shell.parentSpanContext?.spanId).toBe(sub.spanContext().spanId);
    expect(shell.spanContext().traceId).toBe(parent.spanContext().traceId);
  });

  test("a new conversation opened by beforeSubmitPrompt never claims a pending Task call", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_input: { prompt: "List all files", subagent_type: "code-explorer" },
      }),
    );
    telemetry.handle(
      event("beforeSubmitPrompt", { conversation_id: "conv-2", prompt: "unrelated chat" }),
    );
    telemetry.handle(event("stop", { conversation_id: "conv-2", status: "completed" }));
    await flush();

    const other = finished(spans).find((s) => s.attributes["gen_ai.conversation.id"] === "conv-2")!;
    expect(other.name).toBe("unrelated chat");
    const session = sessionSpanContext(apiKey, "conv-2");
    expect(other.spanContext().traceId).toBe(session.traceId);
    expect(other.parentSpanContext?.spanId).toBe(session.spanId);
  });

  test("an unrelated hook-only conversation never claims a pending Task call", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_input: { prompt: "List all files", subagent_type: "code-explorer" },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "conv-2",
        generation_id: "gen-2",
        tool_name: "Shell",
        tool_use_id: "shell-2",
        tool_input: { command: "pwd" },
      }),
    );
    telemetry.handle(event("stop", { conversation_id: "conv-2", status: "completed" }));
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_output: "done",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const other = finished(spans).find(
      (span) =>
        span.attributes["gen_ai.conversation.id"] === "conv-2" &&
        span.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    expect(other!.attributes["gen_ai.agent.name"]).toBe("cursor");
    const session = sessionSpanContext(apiKey, "conv-2");
    expect(other!.spanContext().traceId).toBe(session.traceId);
    expect(other!.parentSpanContext?.spanId).toBe(session.spanId);
  });

  test("an independent headless session never claims a pending Task call", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_input: { prompt: "List all files", subagent_type: "code-explorer" },
      }),
    );
    telemetry.handle(
      event("sessionStart", {
        conversation_id: "headless-2",
        generation_id: "headless-2",
        session_id: "headless-2",
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "headless-2",
        generation_id: "headless-2",
        session_id: "headless-2",
        tool_name: "Shell",
        tool_use_id: "shell-2",
        tool_input: { command: "pwd" },
      }),
    );
    telemetry.handle(
      event("sessionEnd", {
        conversation_id: "headless-2",
        generation_id: "headless-2",
        session_id: "headless-2",
        reason: "completed",
      }),
    );
    telemetry.handle(
      event("postToolUse", {
        tool_name: "Task",
        tool_use_id: "task-call-1",
        tool_output: "done",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const headless = finished(spans).find(
      (span) =>
        span.attributes["gen_ai.conversation.id"] === "headless-2" &&
        span.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    expect(headless!.attributes["gen_ai.agent.name"]).toBe("cursor");
    const session = sessionSpanContext(apiKey, "headless-2");
    expect(headless!.spanContext().traceId).toBe(session.traceId);
    expect(headless!.parentSpanContext?.spanId).toBe(session.spanId);
  });

  test("parallel Task calls claim waits in order", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-a",
        tool_input: { prompt: "task a", subagent_type: "explore" },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-b",
        tool_input: { prompt: "task b", subagent_type: "shell" },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "sub-a",
        generation_id: "sub-a",
        tool_name: "Shell",
        tool_use_id: "s1",
        tool_input: { command: "ls" },
      }),
    );
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "sub-b",
        generation_id: "sub-b",
        tool_name: "Shell",
        tool_use_id: "s2",
        tool_input: { command: "pwd" },
      }),
    );
    telemetry.handle(
      event("postToolUse", { tool_name: "Task", tool_use_id: "task-a", tool_output: "done a" }),
    );
    telemetry.handle(
      event("postToolUse", { tool_name: "Task", tool_use_id: "task-b", tool_output: "done b" }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const parent = agentSpans(spans, "cursor")[0]!;
    const subA = agentSpans(spans, "explore")[0]!;
    const subB = agentSpans(spans, "shell")[0]!;

    for (const sub of [subA, subB]) {
      expect(sub.spanContext().traceId).toBe(parent.spanContext().traceId);
      expect(sub.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
    }

    expect(subA.attributes["gen_ai.output.messages"]).toContain("done a");
    expect(subB.attributes["gen_ai.output.messages"]).toContain("done b");
  });

  test("subagentStop emits a nested invoke_agent span with status", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("subagentStop", {
        subagent_type: "explore",
        status: "completed",
        task: "Explore the auth flow",
        summary: "Auth uses JWT",
        duration_ms: 4500,
        message_count: 12,
        tool_call_count: 8,
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const sub = agentSpans(spans, "explore")[0];
    const parent = agentSpans(spans, "cursor")[0];
    expect(sub).toBeDefined();
    expect(sub!.attributes["gen_ai.agent.name"]).toBe("explore");
    expect(sub!.attributes["cursor.subagent_status"]).toBe("completed");
    expect(sub!.attributes["gen_ai.input.messages"]).toContain("auth flow");
    expect(sub!.attributes["gen_ai.output.messages"]).toContain("JWT");
    // Nested in the turn's trace, not a synthetic root trace.
    expect(sub!.spanContext().traceId).toBe(parent!.spanContext().traceId);
    expect(sub!.parentSpanContext?.spanId).toBe(parent!.spanContext().spanId);
  });

  test("subagent conversation events nest inside the parent turn's trace", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("subagentStart", {
        subagent_id: "sub-1",
        parent_conversation_id: "conv-1",
        subagent_type: "explore",
        subagent_model: "gpt-5",
        task: "Explore the auth flow",
        tool_call_id: "tc-1",
      }),
    );
    telemetry.handle(
      event("postToolUse", {
        conversation_id: "sub-1",
        tool_name: "Grep",
        tool_use_id: "t9",
        tool_output: '{"matches":3}',
        duration: 20,
      }),
    );
    telemetry.handle(
      event("subagentStop", {
        subagent_type: "explore",
        status: "completed",
        task: "Explore the auth flow",
        summary: "Auth uses JWT",
        duration_ms: 4500,
        agent_transcript_path: "/tmp/cursor/sub-1/transcript.txt",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const parent = agentSpans(spans, "cursor")[0]!;
    const subs = agentSpans(spans, "explore");
    expect(subs).toHaveLength(1);
    const sub = subs[0]!;
    expect(sub.spanContext().traceId).toBe(parent.spanContext().traceId);
    expect(sub.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
    expect(sub.attributes["gen_ai.agent.name"]).toBe("explore");
    expect(sub.attributes["gen_ai.input.messages"]).toContain("auth flow");
    expect(sub.attributes["gen_ai.output.messages"]).toContain("JWT");
    expect(sub.attributes["cursor.subagent_status"]).toBe("completed");
    expect(sub.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);

    const tool = finished(spans).find((s) => s.name === "execute_tool Grep")!;
    expect(tool.parentSpanContext?.spanId).toBe(sub.spanContext().spanId);
    expect(tool.spanContext().traceId).toBe(parent.spanContext().traceId);
  });

  test("a subagent turn ended by its own stop is not duplicated by subagentStop", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));
    telemetry.handle(
      event("subagentStart", {
        subagent_id: "sub-1",
        parent_conversation_id: "conv-1",
        subagent_type: "explore",
        task: "Explore the auth flow",
      }),
    );
    telemetry.handle(event("afterAgentResponse", { conversation_id: "sub-1", text: "found it" }));
    telemetry.handle(event("stop", { conversation_id: "sub-1", status: "completed" }));
    telemetry.handle(
      event("subagentStop", {
        subagent_type: "explore",
        status: "completed",
        summary: "found it",
        agent_transcript_path: "/tmp/cursor/sub-1/transcript.txt",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const subs = agentSpans(spans, "explore");
    expect(subs).toHaveLength(1);
    expect(subs[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
  });
  test("subagent transcript paths match complete path segments", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "explore" }));

    for (const subagentId of ["sub-1", "sub-10"]) {
      telemetry.handle(
        event("subagentStart", {
          subagent_id: subagentId,
          parent_conversation_id: "conv-1",
          subagent_type: "explore",
          task: subagentId,
        }),
      );
      telemetry.handle(
        event("afterAgentResponse", { conversation_id: subagentId, text: `${subagentId} done` }),
      );
    }

    telemetry.handle(
      event("subagentStop", {
        conversation_id: "conv-1",
        subagent_type: "explore",
        status: "completed",
        summary: "sub-10 done",
        agent_transcript_path: "/tmp/cursor/sub-10/transcript.txt",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const sub1 = finished(spans).find(
      (span) =>
        span.attributes["gen_ai.conversation.id"] === "sub-1" &&
        span.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    const sub10 = finished(spans).find(
      (span) =>
        span.attributes["gen_ai.conversation.id"] === "sub-10" &&
        span.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    expect(sub1!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
    expect(sub10!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
  });
});

describe("titles", () => {
  test("turn spans take Cursor chat titles as their names", async () => {
    const chats = mkdtempSync(join(tmpdir(), "cursor-chats-"));
    mkdirSync(join(chats, "hash1", "conv-1"), { recursive: true });
    writeFileSync(
      join(chats, "hash1", "conv-1", "meta.json"),
      JSON.stringify({ schemaVersion: 1, title: "Title from metadata" }),
    );
    mkdirSync(join(chats, "hash1", "sub-1"), { recursive: true });
    writeFileSync(
      join(chats, "hash1", "sub-1", "meta.json"),
      JSON.stringify({ schemaVersion: 1, title: "Subagent title from metadata" }),
    );
    const { spans, telemetry } = makeHarness({ chatsDir: chats });

    try {
      telemetry.handle(event("beforeSubmitPrompt", { prompt: "fix login" }));
      telemetry.handle(
        event("subagentStart", {
          subagent_id: "sub-1",
          parent_conversation_id: "conv-1",
          subagent_type: "explore",
          task: "Explore the auth flow",
        }),
      );
      telemetry.handle(event("afterAgentResponse", { conversation_id: "sub-1", text: "ok" }));
      telemetry.handle(
        event("subagentStop", {
          subagent_type: "explore",
          status: "completed",
          summary: "ok",
          agent_transcript_path: "/tmp/cursor/sub-1/transcript.txt",
        }),
      );
      telemetry.handle(event("stop", { status: "completed" }));
      await flush();

      const names = finished(spans).map((s) => s.name);
      expect(names).toContain("Title from metadata");
      expect(names).toContain("Subagent title from metadata");
    } finally {
      rmSync(chats, { recursive: true, force: true });
    }
  });

  test("a missing title falls back to the prompt as the span name", async () => {
    const chats = mkdtempSync(join(tmpdir(), "cursor-chats-"));
    const { spans, telemetry } = makeHarness({ chatsDir: chats });

    try {
      telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
      telemetry.handle(event("stop", { status: "completed" }));
      await flush();

      expect(finished(spans).map((s) => s.name)).toContain("x");
    } finally {
      rmSync(chats, { recursive: true, force: true });
    }
  });
  test("title lookup rejects path escapes and dot segments", () => {
    const cases = [
      { conversationId: ".", metaDir: ["hash1"] },
      { conversationId: "..", metaDir: [] },
      { conversationId: "../outside", metaDir: ["outside"] },
      { conversationId: "bad\\id", metaDir: ["hash1", "bad\\id"] },
    ];

    for (const { conversationId, metaDir } of cases) {
      const chats = mkdtempSync(join(tmpdir(), "cursor-chats-"));
      mkdirSync(join(chats, "hash1"), { recursive: true });
      const titleDir = join(chats, ...metaDir);
      mkdirSync(titleDir, { recursive: true });
      writeFileSync(join(titleDir, "meta.json"), JSON.stringify({ title: "Outside title" }));

      try {
        expect(createTitleLookup(chats)(conversationId)).toBeUndefined();
      } finally {
        rmSync(chats, { recursive: true, force: true });
      }
    }
  });
});

describe("lifecycle logs", () => {
  test("session, file edit, and compaction events emit correlated logs", async () => {
    const { logs, telemetry } = makeHarness();
    telemetry.handle(event("sessionStart", { session_id: "conv-1", composer_mode: "agent" }));
    telemetry.handle(event("afterFileEdit", { file_path: "/tmp/project/a.ts", edits: [{}, {}] }));
    telemetry.handle(event("preCompact", { trigger: "auto", context_usage_percent: 85 }));
    telemetry.handle(
      event("sessionEnd", { session_id: "conv-1", reason: "completed", duration_ms: 100 }),
    );
    await flush();

    const records = logs.getFinishedLogRecords();
    const names = records.map((r) => r.eventName);
    expect(names).toEqual(["sessionStart", "afterFileEdit", "preCompact", "sessionEnd"]);

    for (const record of records) {
      expect(record.attributes["gen_ai.conversation.id"]).toBe("conv-1");
      expect(record.attributes["gen_ai.agent.name"]).toBe("cursor");
    }

    const edit = records.find((r) => r.eventName === "afterFileEdit");
    expect(edit!.attributes["cursor.file_path"]).toBe("/tmp/project/a.ts");
    expect(edit!.attributes["cursor.edit_count"]).toBe(2);
  });

  test("sessionEnd with reason error fails the open turn", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
    telemetry.handle(event("sessionEnd", { reason: "error", error_message: "model overloaded" }));
    await flush();

    const agent = agentSpans(spans, "cursor")[0];
    expect(agent!.status.code).toBe(SpanStatusCode.ERROR);
    expect(agent!.attributes["error.type"]).toBe("SessionError");
  });
});

describe("settle", () => {
  test("settle ends open turns as incomplete and reports no open work after", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
    expect(telemetry.open()).toBe(true);
    await telemetry.settle();
    expect(telemetry.open()).toBe(false);

    const agent = agentSpans(spans, "cursor")[0];
    expect(agent!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  });

  test("stop without conversation_id closes the sole open turn", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
    telemetry.handle({ hook_event_name: "stop", status: "completed", loop_count: 0 });
    await flush();

    expect(telemetry.open()).toBe(false);
    const agent = agentSpans(spans, "cursor")[0];
    expect(agent!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
  });

  test("malformed events never throw", () => {
    const { telemetry } = makeHarness();

    const eventRecord = {
      hook_event_name: "unknownEvent",
      nested: { arbitrary: new Date() },
    };

    expectTypeOf<CursorTelemetry["handle"]>().parameter(0).toEqualTypeOf<Record<string, unknown>>();
    telemetry.handle({});
    telemetry.handle({ hook_event_name: "postToolUse" });
    telemetry.handle({ hook_event_name: "unknownEvent", conversation_id: "conv-1" });
    telemetry.handle(eventRecord);
    expect(telemetry.open()).toBe(false);
  });
});

describe("fileConfig", () => {
  test("a malformed config file yields empty options instead of a crash", () => {
    const home = mkdtempSync(join(tmpdir(), "cursor-cfg-"));
    mkdirSync(join(home, ".cursor"));
    writeFileSync(join(home, ".cursor", "telemetry-dev.json"), "{not json");
    const restoreHome = setHome(home);

    try {
      expect(fileConfig()).toEqual({});
    } finally {
      restoreHome();
      rmSync(home, { recursive: true, force: true });
    }
  });
});

describe("stale state across turns", () => {
  test("a delayed stop from a closed turn does not end the fresh turn", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "first", generation_id: "gen-1" }));
    telemetry.handle(event("afterAgentResponse", { text: "done", generation_id: "gen-1" }));
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "second", generation_id: "gen-2" }));
    // The first turn's stop arrives late, after its turn was already closed.
    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-1" }));
    expect(telemetry.open()).toBe(true);
    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-2" }));
    await flush();

    const agents = agentSpans(spans, "cursor");
    expect(agents).toHaveLength(2);
    expect(agents[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
    expect(agents[1]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
    expect(telemetry.open()).toBe(false);
  });

  test("a late stop remains stale after 100 other conversations close", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "first", generation_id: "gen-1" }));
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "second", generation_id: "gen-2" }));

    for (let index = 0; index < 100; index++) {
      const id = `other-${index}`;
      telemetry.handle(
        event("beforeSubmitPrompt", { conversation_id: id, generation_id: id, prompt: id }),
      );
      telemetry.handle(
        event("stop", { conversation_id: id, generation_id: id, status: "completed" }),
      );
    }

    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-1" }));
    expect(telemetry.open()).toBe(true);
    telemetry.handle(event("stop", { status: "completed", generation_id: "gen-2" }));
    await flush();

    const agents = finished(spans).filter(
      (span) =>
        span.attributes["gen_ai.conversation.id"] === "conv-1" &&
        span.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    expect(agents).toHaveLength(2);
    expect(agents[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
    expect(agents[1]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
  });

  test("a Task wait from an ended turn is not claimed by a later conversation", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "spawn" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Task",
        tool_use_id: "task-1",
        tool_input: { prompt: "never finishes", subagent_type: "explore" },
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    // A later unrelated CLI conversation must open as its own root turn.
    telemetry.handle(
      event("preToolUse", {
        conversation_id: "conv-9",
        tool_name: "Shell",
        tool_use_id: "s9",
        tool_input: { command: "ls" },
      }),
    );
    telemetry.handle(event("stop", { conversation_id: "conv-9", status: "completed" }));
    await flush();

    const later = finished(spans).find(
      (s) =>
        s.attributes["gen_ai.conversation.id"] === "conv-9" &&
        s.attributes["gen_ai.operation.name"] === "invoke_agent",
    );

    expect(later).toBeDefined();
    expect(later!.attributes["gen_ai.agent.name"]).toBe("cursor");
    const session = sessionSpanContext(apiKey, "conv-9");
    expect(later!.spanContext().traceId).toBe(session.traceId);
    expect(later!.parentSpanContext?.spanId).toBe(session.spanId);
  });

  test("a post-tool event after its turn ended does not reuse stale pre-tool state", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "x" }));
    telemetry.handle(
      event("preToolUse", {
        tool_name: "Shell",
        tool_use_id: "t1",
        tool_input: { command: "stale-input" },
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    telemetry.handle(event("postToolUse", { tool_name: "Shell", tool_use_id: "t1", duration: 10 }));
    await telemetry.settle();

    const tool = finished(spans).find((s) => s.name === "execute_tool Shell")!;
    expect(JSON.stringify(tool.attributes)).not.toContain("stale-input");
  });

  test("sessionEnd closes the turns of its session, tracked by session_id", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "a", session_id: "sess-1" }));
    telemetry.handle(
      event("beforeSubmitPrompt", { conversation_id: "conv-2", prompt: "b", session_id: "sess-2" }),
    );
    telemetry.handle({ hook_event_name: "sessionEnd", session_id: "sess-1", reason: "completed" });
    await flush();

    // Only the sess-1 turn closed; the sess-2 turn stays open.
    expect(telemetry.open()).toBe(true);
    const agents = agentSpans(spans, "cursor");
    expect(agents).toHaveLength(1);
    expect(agents[0]!.attributes["gen_ai.conversation.id"]).toBe("conv-1");
    await telemetry.settle();
  });

  test("a nested subagent's stop in its parent's context ends the child, not the parent", async () => {
    const { spans, telemetry } = makeHarness();
    telemetry.handle(event("beforeSubmitPrompt", { prompt: "root" }));
    telemetry.handle(
      event("subagentStart", {
        subagent_id: "sub-1",
        parent_conversation_id: "conv-1",
        subagent_type: "explore",
        task: "outer task",
      }),
    );
    telemetry.handle(
      event("subagentStart", {
        subagent_id: "sub-2",
        parent_conversation_id: "sub-1",
        subagent_type: "digger",
        task: "inner task",
      }),
    );
    telemetry.handle(event("afterAgentResponse", { conversation_id: "sub-2", text: "dug" }));
    // The child's stop is delivered in the subagent parent's context: the
    // conversation_id names sub-1, the transcript path names the child.
    telemetry.handle(
      event("subagentStop", {
        conversation_id: "sub-1",
        subagent_type: "digger",
        status: "completed",
        summary: "dug",
        agent_transcript_path: "/tmp/cursor/sub-2/transcript.txt",
      }),
    );
    telemetry.handle(event("stop", { status: "completed" }));
    await flush();

    const child = agentSpans(spans, "digger")[0];
    const outer = agentSpans(spans, "explore")[0];
    expect(child).toBeDefined();
    expect(child!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
    // The outer subagent was not the stop's target: it closes as incomplete
    // when the root turn ends.
    expect(outer).toBeDefined();
    expect(outer!.attributes["gen_ai.response.finish_reasons"]).toEqual(["incomplete"]);
  });
});

describe.skipIf(process.platform === "win32")("daemon socket protocol", () => {
  function sendChunks(path: string, chunks: string[], acks: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = createConnection(path);
      let seen = 0;
      socket.setEncoding("utf8");
      socket.on("data", (chunk: string) => {
        seen += chunk.split("\n").filter((l) => l.length > 0).length;

        if (seen >= acks) {
          socket.end();
          resolve();
        }
      });
      socket.once("error", reject);
      socket.once("connect", () => {
        for (const chunk of chunks) socket.write(chunk);
      });
    });
  }

  test("splits concatenated lines and drops replayed deliveries", async () => {
    const { spans, telemetry } = makeHarness();
    const dir = mkdtempSync(join(tmpdir(), "tdc-sock-"));
    const path = join(dir, "d.sock");
    const server = await listen(path, telemetry);

    if (!server) throw new Error("failed to start test daemon");

    try {
      const post = {
        ...event("postToolUse", {
          tool_name: "Shell",
          tool_use_id: "t1",
          tool_output: "ok",
          duration: 5,
        }),
        hook_delivery_id: "d-1",
      };

      const lines = [
        { ...event("beforeSubmitPrompt", { prompt: "x" }), hook_delivery_id: "d-0" },
        post,
        post, // retry after a lost ack: must not double-emit
        { ...event("stop", { status: "completed" }), hook_delivery_id: "d-2" },
      ]
        .map((l) => `${JSON.stringify(l)}\n`)
        .join("");

      // Split mid-line to exercise framing across chunk boundaries.
      await sendChunks(path, [lines.slice(0, 25), lines.slice(25)], 4);
      await flush();

      const tools = finished(spans).filter((s) => s.name === "execute_tool Shell");
      expect(tools).toHaveLength(1);
      const agents = agentSpans(spans, "cursor");
      expect(agents).toHaveLength(1);
      expect(agents[0]!.attributes["gen_ai.response.finish_reasons"]).toEqual(["completed"]);
    } finally {
      await new Promise((resolve) => server.close(resolve));
      rmSync(dir, { recursive: true, force: true });
    }
  });
  test("a stale startup lock yields one socket owner", async () => {
    const { telemetry } = makeHarness();
    const dir = mkdtempSync(join(tmpdir(), "tdc-lock-"));
    const path = join(dir, "d.sock");
    const lock = `${path}.lock`;
    const held = join(lock, "stale-owner");
    mkdirSync(lock);
    writeFileSync(held, "");
    utimesSync(held, 0, 0);

    const servers = (await Promise.all([listen(path, telemetry), listen(path, telemetry)])).filter(
      (server) => server !== undefined,
    );

    try {
      expect(servers).toHaveLength(1);
    } finally {
      await Promise.all(
        servers.map((server) => new Promise<void>((resolve) => server.close(() => resolve()))),
      );
      await telemetry.settle();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("install and uninstall", () => {
  test("install preserves foreign hooks and reinstall replaces its own entries across path changes", async () => {
    const home = mkdtempSync(join(tmpdir(), "cursor-home-"));
    const restoreHome = setHome(home);
    const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);

    try {
      const hooksPath = join(home, ".cursor", "hooks.json");
      mkdirSync(join(home, ".cursor"), { recursive: true });
      writeFileSync(
        hooksPath,
        JSON.stringify({ version: 1, hooks: { stop: [{ command: "my-other-hook" }] } }),
      );
      await runInstall("/opt/telemetry/cli.js", ["--api-key", "td_live_test"]);
      // Reinstall from a moved checkout: the marker, not the embedded path,
      // identifies our old entries.
      await runInstall("/moved/elsewhere/cli.js", ["--api-key", "td_live_test"]);

      const hooks = JSON.parse(readFileSync(hooksPath, "utf8")) as {
        hooks: Record<string, { command: string; telemetryDev?: boolean }[]>;
      };

      const stopHooks = hooks.hooks.stop!;
      expect(stopHooks.filter((h) => h.command === "my-other-hook")).toHaveLength(1);
      const oursEntries = stopHooks.filter((h) => h.telemetryDev === true);
      expect(oursEntries).toHaveLength(1);
      expect(oursEntries[0]!.command).toContain("/moved/elsewhere/cli.js");

      const config = JSON.parse(
        readFileSync(join(home, ".cursor", "telemetry-dev.json"), "utf8"),
      ) as { apiKey: string };

      expect(config.apiKey).toBe("td_live_test");
      expect(lstatSync(join(home, ".cursor", "telemetry-dev.json")).mode & 0o777).toBe(0o600);
    } finally {
      stdout.mockRestore();
      restoreHome();
      rmSync(home, { recursive: true, force: true });
    }
  });

  test("uninstall removes only our entries", async () => {
    const home = mkdtempSync(join(tmpdir(), "cursor-home-"));
    const restoreHome = setHome(home);
    const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);

    try {
      const hooksPath = join(home, ".cursor", "hooks.json");
      mkdirSync(join(home, ".cursor"), { recursive: true });
      writeFileSync(
        hooksPath,
        JSON.stringify({
          version: 1,
          hooks: {
            stop: [
              { command: "my-other-hook" },
              { command: "my-telemetry-dev-cursor-backup" },
              { command: "echo /opt/telemetry/cli.js" },
            ],
          },
        }),
      );
      await runInstall("/opt/telemetry/cli.js", ["--api-key", "td_live_test"]);
      await runUninstall();

      const hooks = JSON.parse(readFileSync(hooksPath, "utf8")) as {
        hooks: Record<string, { command: string }[]>;
      };

      expect(hooks.hooks.stop).toEqual([
        { command: "my-other-hook" },
        { command: "my-telemetry-dev-cursor-backup" },
        { command: "echo /opt/telemetry/cli.js" },
      ]);
    } finally {
      stdout.mockRestore();
      restoreHome();
      rmSync(home, { recursive: true, force: true });
    }
  });

  test.skipIf(process.platform === "win32")(
    "install commands preserve shell metacharacters in paths",
    async () => {
      const home = mkdtempSync(join(tmpdir(), "cursor-home-"));
      const restoreHome = setHome(home);
      const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);

      try {
        const cliDir = join(home, "cli '$HOME' `tick`");
        const cliPath = join(cliDir, "hook.mjs");
        mkdirSync(cliDir, { recursive: true });
        writeFileSync(cliPath, "process.stdout.write(JSON.stringify(process.argv.slice(2)));");
        await runInstall(cliPath, ["--api-key", "td_live_test"]);

        const hooks = JSON.parse(readFileSync(join(home, ".cursor", "hooks.json"), "utf8")) as {
          hooks: Record<string, { command: string }[]>;
        };

        const command = hooks.hooks.stop![0]!.command;
        const output = execFileSync("/bin/sh", ["-c", command], { encoding: "utf8" });
        expect(JSON.parse(output)).toEqual(["hook"]);
      } finally {
        stdout.mockRestore();
        restoreHome();
        rmSync(home, { recursive: true, force: true });
      }
    },
  );

  test.skipIf(process.platform === "win32")(
    "install rejects a symlink config without changing its target",
    async () => {
      const home = mkdtempSync(join(tmpdir(), "cursor-home-"));
      const restoreHome = setHome(home);

      try {
        const cursorDir = join(home, ".cursor");
        const target = join(home, "target.json");
        mkdirSync(cursorDir, { recursive: true });
        writeFileSync(target, "keep");
        symlinkSync(target, join(cursorDir, "telemetry-dev.json"));

        await expect(
          runInstall("/opt/telemetry/cli.js", ["--api-key", "td_live_test"]),
        ).rejects.toThrow(/regular file/);
        expect(readFileSync(target, "utf8")).toBe("keep");
        expect(lstatSync(join(cursorDir, "telemetry-dev.json")).isSymbolicLink()).toBe(true);
      } finally {
        restoreHome();
        rmSync(home, { recursive: true, force: true });
      }
    },
  );

  test("an option flag without a value aborts instead of consuming the next flag", async () => {
    const home = mkdtempSync(join(tmpdir(), "cursor-home-"));
    const restoreHome = setHome(home);

    const exit = vi.spyOn(process, "exit").mockImplementation((() => {
      throw new Error("exit(1)");
    }) as never);

    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    try {
      await expect(
        runInstall("/opt/telemetry/cli.js", ["--api-key", "--base-url", "https://example.test"]),
      ).rejects.toThrow("exit(1)");
      expect(stderr).toHaveBeenCalledWith("Missing value for --api-key.\n");
    } finally {
      exit.mockRestore();
      stderr.mockRestore();
      restoreHome();
      rmSync(home, { recursive: true, force: true });
    }
  });
});
