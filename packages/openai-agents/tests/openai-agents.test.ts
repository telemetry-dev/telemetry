import {
  Agent,
  type Model,
  OpenAIChatCompletionsModel,
  OpenAIResponsesModel,
  Runner,
  Span,
  setTraceProcessors,
  setTracingDisabled,
  Trace,
  type TracingProcessor,
  withCustomSpan,
  withTrace,
} from "@openai/agents";
import { SandboxAgent, shell } from "@openai/agents/sandbox";
import { UnixLocalSandboxClient } from "@openai/agents/sandbox/local";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { flush, propagateAttributes, startActiveSpan } from "@telemetry-dev/sdk";
import OpenAI from "openai";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { resetForTesting } from "../src/config.ts";
import { instrumentSandboxClient, telemetryDevProcessor } from "../src/index.ts";

type Json = Record<string, unknown>;

let exporter: InMemorySpanExporter;

type ProcessorOptions = Parameters<typeof telemetryDevProcessor>[0];

function makeProcessor(
  options: ProcessorOptions = {},
  before: TracingProcessor[] = [],
): TracingProcessor {
  exporter = new InMemorySpanExporter();
  const processor = telemetryDevProcessor(
    {
      apiKey: "td_live_test",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
      ...options,
    },
    { spanExporter: exporter },
  );
  setTraceProcessors([...before, processor]);

  return processor;
}

beforeEach(() => {
  // The Agents SDK turns tracing off when NODE_ENV is "test".
  setTracingDisabled(false);
});

afterEach(async () => {
  setTraceProcessors([]);
  await resetForTesting();
});

function functionCall(command: string, callId: string): Json {
  return {
    type: "function_call",
    id: `fc_${callId}`,
    call_id: callId,
    name: "exec_command",
    arguments: JSON.stringify({ cmd: command, login: false }),
    status: "completed",
  };
}

function message(text: string): Json {
  return {
    type: "message",
    id: "msg_1",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
}

function response(id: string, output: Json[], usage: Json): Json {
  return {
    id,
    object: "response",
    created_at: 0,
    model: "gpt-5-mini",
    status: "completed",
    output,
    usage,
  };
}

/** An OpenAI client whose HTTP calls return `bodies` in order. */
function scriptedClient(bodies: Json[], status = 200): OpenAI {
  const queue = [...bodies];

  return new OpenAI({
    apiKey: "sk-test",
    maxRetries: 0,
    fetch: async () => {
      const body = queue.shift();

      if (!body) throw new Error("no scripted response left");

      return new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      });
    },
  });
}

/** A Responses API model: the model asks for `command`, then answers "done". */
function responsesModel(command: string): Model {
  return new OpenAIResponsesModel(
    scriptedClient([
      response("resp_1", [functionCall(command, "call_1")], {
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 120,
        input_tokens_details: { cached_tokens: 40 },
        output_tokens_details: { reasoning_tokens: 5 },
      }),
      response("resp_2", [message("done")], {
        input_tokens: 50,
        output_tokens: 7,
        total_tokens: 57,
      }),
    ]),
    "gpt-5-mini",
  );
}

function completion(id: string, message: Json, usage: Json): Json {
  return {
    id,
    object: "chat.completion",
    created: 0,
    model: "gpt-4.1-mini-2025-04-14",
    choices: [{ index: 0, message, finish_reason: "stop", logprobs: null }],
    usage,
  };
}

/** A Chat Completions model with the same script as `responsesModel`. */
function chatCompletionsModel(command: string): Model {
  return new OpenAIChatCompletionsModel(
    scriptedClient([
      completion(
        "chatcmpl_1",
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "exec_command", arguments: JSON.stringify({ cmd: command }) },
            },
          ],
        },
        {
          prompt_tokens: 100,
          completion_tokens: 20,
          total_tokens: 120,
          prompt_tokens_details: { cached_tokens: 40 },
          completion_tokens_details: { reasoning_tokens: 5 },
        },
      ),
      completion(
        "chatcmpl_2",
        { role: "assistant", content: "done" },
        { prompt_tokens: 50, completion_tokens: 7, total_tokens: 57 },
      ),
    ]),
    "gpt-4.1-mini",
  );
}

/** One run of a sandbox agent with `model`, by default one that runs `command`. */
async function runCommand(
  command: string,
  options: {
    sandboxClient?: UnixLocalSandboxClient;
    workflowName?: string;
    model?: Model;
  } = {},
): Promise<void> {
  const agent = new SandboxAgent({
    name: "coder",
    instructions: "Run the command.",
    model: options.model ?? responsesModel(command),
    capabilities: [shell()],
  });

  await new Runner({ workflowName: options.workflowName ?? "workflow" }).run(agent, "go", {
    sandbox: { client: options.sandboxClient ?? new UnixLocalSandboxClient() },
  });
}

async function exported(): Promise<ReadableSpan[]> {
  await flush();

  return exporter.getFinishedSpans();
}

const parentOf = (span: ReadableSpan): string | undefined => span.parentSpanContext?.spanId;

function only(spans: ReadableSpan[], predicate: (span: ReadableSpan) => boolean): ReadableSpan {
  const matches = spans.filter(predicate);
  expect(matches).toHaveLength(1);

  return matches[0]!;
}

const named = (name: string) => (span: ReadableSpan) => span.name === name;

test("a sandbox command becomes a tool span with its exit code, nested under its agent", async () => {
  makeProcessor();
  await runCommand("sh -c 'exit 3'");
  const spans = await exported();

  const root = only(spans, named("workflow"));
  const tool = only(spans, named("exec_command"));
  const exec = only(spans, named("sandbox.exec"));
  const turn = spans.find((span) => span.spanContext().spanId === parentOf(tool))!;
  const agent = spans.find((span) => span.spanContext().spanId === parentOf(turn))!;

  expect(new Set(spans.map((span) => span.spanContext().traceId)).size).toBe(1);
  expect(tool.attributes["gen_ai.operation.name"]).toBe("execute_tool");
  expect(tool.attributes["process.exit.code"]).toBe(3);
  // A non-zero exit is not a failed tool call.
  expect(tool.status.code).not.toBe(2);
  expect(parentOf(exec)).toBe(tool.spanContext().spanId);
  expect(exec.attributes["sandbox.operation"]).toBe("exec");
  expect(exec.attributes["sandbox.backend"]).toBe("unix_local");
  expect(exec.attributes["process.command_line"]).toBe("sh -c 'exit 3'");
  expect(turn.name).toMatch(/^turn /);
  expect(agent.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(agent.attributes["gen_ai.agent.name"]).toBe("coder");
  expect(spans.every((span) => span === root || parentOf(span) !== undefined)).toBe(true);
});

test("a command that exits 0 records exit code 0", async () => {
  makeProcessor();
  await runCommand("true");

  const tool = only(await exported(), named("exec_command"));
  expect(tool.attributes["process.exit.code"]).toBe(0);
});

test("only model-call spans carry token usage", async () => {
  makeProcessor();
  await runCommand("true");
  const spans = await exported();

  const chats = spans.filter((span) => span.attributes["gen_ai.operation.name"] === "chat");
  expect(chats.map((span) => span.name)).toEqual(["chat gpt-5-mini", "chat gpt-5-mini"]);
  const sum = (key: string) =>
    chats.reduce((total, span) => total + Number(span.attributes[key] ?? 0), 0);
  expect(sum("gen_ai.usage.input_tokens")).toBe(150);
  expect(sum("gen_ai.usage.output_tokens")).toBe(27);
  expect(sum("gen_ai.usage.cache_read.input_tokens")).toBe(40);
  expect(sum("gen_ai.usage.reasoning.output_tokens")).toBe(5);
  expect(new Set(chats.map((span) => span.attributes["gen_ai.response.id"]))).toEqual(
    new Set(["resp_1", "resp_2"]),
  );

  const withUsage = spans.filter((span) =>
    Object.keys(span.attributes).some((key) => key.startsWith("gen_ai.usage.")),
  );
  expect(withUsage).toEqual(chats);
});

test("with captureInput off, the command is not on the sandbox or tool spans", async () => {
  makeProcessor({ captureInput: false });
  await runCommand("echo hello");
  const spans = await exported();

  expect(only(spans, named("sandbox.exec")).attributes["process.command_line"]).toBeUndefined();
  expect(
    only(spans, named("exec_command")).attributes["gen_ai.tool.call.arguments"],
  ).toBeUndefined();
});

test("with capture off, no attribute contains the command or its output", async () => {
  // The command is also the model's output (its function call) and the tool's output (the echo).
  makeProcessor({ captureInput: false, captureOutput: false });
  const marker = "MARKER_4f2a9c";
  await runCommand(`echo ${marker}`);
  const spans = await exported();

  const leaked = spans.filter((span) =>
    Object.values(span.attributes).some((value) => JSON.stringify(value).includes(marker)),
  );
  expect(leaked.map((span) => span.name)).toEqual([]);
  expect(only(spans, named("exec_command")).attributes["process.exit.code"]).toBe(0);
});

/** A local sandbox client whose sessions carry `sandboxId`, as a hosted backend such as E2B's do. */
function clientWithId(sandboxId: string): UnixLocalSandboxClient {
  const client = new UnixLocalSandboxClient();
  const create = client.create.bind(client);
  client.create = (async (...args: Parameters<typeof create>) => {
    const session = await create(...args);
    session.state.sandboxId = sandboxId;

    return session;
  }) as typeof client.create;

  return instrumentSandboxClient(client);
}

test("the provider's sandbox id reaches the session, command, and tool spans", async () => {
  makeProcessor();
  await runCommand("true", { sandboxClient: clientWithId("sbx_test") });
  const spans = await exported();

  for (const name of ["sandbox.create_session", "sandbox.exec", "exec_command"]) {
    expect(only(spans, named(name)).attributes["sandbox.id"], name).toBe("sbx_test");
  }
});

test("two concurrent runs produce two traces whose spans do not cross", async () => {
  makeProcessor();
  await Promise.all([
    runCommand("echo run_a", { workflowName: "run a" }),
    runCommand("echo run_b", { workflowName: "run b" }),
  ]);
  const spans = await exported();

  for (const run of ["a", "b"]) {
    const root = only(spans, named(`run ${run}`));
    const traceId = root.spanContext().traceId;
    const commands = spans
      .filter((span) => span.spanContext().traceId === traceId && span.name === "sandbox.exec")
      .map((span) => span.attributes["process.command_line"]);
    expect(commands).toEqual([`echo run_${run}`]);
  }
});

/** One run of an agent that answers "done", with its root span named `workflowName`. */
async function runAgent(workflowName: string, groupId?: string): Promise<void> {
  const model = new OpenAIResponsesModel(
    scriptedClient([
      response("resp_1", [message("done")], { input_tokens: 1, output_tokens: 1, total_tokens: 2 }),
    ]),
    "gpt-5-mini",
  );

  await new Runner({ workflowName, groupId }).run(new Agent({ name: "assistant", model }), "go");
}

const traceOf = (span: ReadableSpan): string => span.spanContext().traceId;
const sessionOf = (span: ReadableSpan): unknown => span.attributes["gen_ai.conversation.id"];

test("runs that share a groupId land in one session trace", async () => {
  makeProcessor();
  await runAgent("a1", "g1");
  await runAgent("a2", "g1");
  await runAgent("b", "g2");
  await runAgent("c");
  const spans = await exported();
  const [a1, a2, b, c] = ["a1", "a2", "b", "c"].map((name) => only(spans, named(name)));

  expect(traceOf(a2)).toBe(traceOf(a1));
  expect(new Set([a1, b, c].map(traceOf)).size).toBe(3);
  expect([a1, a2, b, c].map(sessionOf)).toEqual(["g1", "g1", "g2", undefined]);
  // Each run of a session is a root under one parent that is never exported.
  expect(parentOf(a1)).toEqual(expect.any(String));
  expect(parentOf(a2)).toBe(parentOf(a1));
  expect(spans.map((span) => span.spanContext().spanId)).not.toContain(parentOf(a1));
  expect(parentOf(c)).toBeUndefined();
});

test("a run inside an active span joins that span's trace, even with a groupId", async () => {
  makeProcessor();
  await startActiveSpan("request", async () => {
    await runAgent("a", "g1");
    await runAgent("b", "g1");
  });
  const spans = await exported();
  const request = only(spans, named("request"));

  for (const root of [only(spans, named("a")), only(spans, named("b"))]) {
    expect(traceOf(root)).toBe(traceOf(request));
    expect(parentOf(root)).toBe(request.spanContext().spanId);
    expect(sessionOf(root)).toBe("g1");
  }
});

test("runs share a propagateAttributes session, and a groupId takes precedence", async () => {
  makeProcessor();
  await propagateAttributes({ sessionId: "s1" }, async () => {
    await runAgent("a");
    await runAgent("b");
    await runAgent("c", "g1");
    await runAgent("d", "g1");
  });
  const spans = await exported();
  const [a, b, c, d] = ["a", "b", "c", "d"].map((name) => only(spans, named(name)));

  expect(traceOf(b)).toBe(traceOf(a));
  expect(traceOf(d)).toBe(traceOf(c));
  expect(traceOf(c)).not.toBe(traceOf(a));
  expect([a, b, c, d].map(sessionOf)).toEqual(["s1", "s1", "g1", "g1"]);
  const groupTraceSpans = spans.filter((span) => traceOf(span) === traceOf(c));
  expect(new Set(groupTraceSpans.map(sessionOf))).toEqual(new Set(["g1"]));
});

test("process sessions group the runs that have no session of their own", async () => {
  makeProcessor({ sessionMode: "process" });
  await runAgent("a");
  await runAgent("b");
  await runAgent("c", "g1");
  await runAgent("d", "g1");
  const spans = await exported();
  const [a, b, c, d] = ["a", "b", "c", "d"].map((name) => only(spans, named(name)));

  expect(traceOf(b)).toBe(traceOf(a));
  expect(traceOf(d)).toBe(traceOf(c));
  expect(traceOf(c)).not.toBe(traceOf(a));
  expect(sessionOf(a)).toEqual(expect.any(String));
  expect(sessionOf(b)).toBe(sessionOf(a));
  expect(sessionOf(c)).toBe("g1");
});

test("a span whose parent is unknown nests under its trace root", async () => {
  const processor = makeProcessor();
  const trace = new Trace({ traceId: "trace_0123456789abcdef0123456789abcdef", name: "manual" });
  await processor.onTraceStart(trace);
  const span = new Span(
    {
      traceId: trace.traceId,
      parentId: "span_missing",
      data: { type: "custom", name: "step", data: { attempt: 2 } },
    },
    processor,
  );
  span.start();
  span.end();
  await processor.onTraceEnd(trace);
  const spans = await exported();

  const root = only(spans, named("manual"));
  const step = only(spans, named("step"));
  expect(parentOf(step)).toBe(root.spanContext().spanId);
  expect(step.attributes["td.metadata.attempt"]).toBe("2");
});

test("two runs under one trace keep their own sandbox ids", async () => {
  makeProcessor();
  // Both agents are named "coder", so a lookup by name alone would mix the sandboxes up.
  await withTrace("batch", () =>
    Promise.all([
      runCommand("echo a", { sandboxClient: clientWithId("sbx_a") }),
      runCommand("echo b", { sandboxClient: clientWithId("sbx_b") }),
    ]),
  );
  const spans = await exported();

  const execs = spans.filter(named("sandbox.exec"));
  expect(
    Object.fromEntries(
      execs.map((span) => [span.attributes["process.command_line"], span.attributes["sandbox.id"]]),
    ),
  ).toEqual({ "echo a": "sbx_a", "echo b": "sbx_b" });

  for (const exec of execs) {
    const tool = spans.find((span) => span.spanContext().spanId === parentOf(exec))!;
    expect(tool.attributes["sandbox.id"]).toBe(exec.attributes["sandbox.id"]);
  }
});

test("instrumentSandboxClient records the id of a resumed session", async () => {
  makeProcessor();
  const client = new UnixLocalSandboxClient();
  client.resume = (async () => ({ state: { sandboxId: "sbx_resumed" } })) as never;
  instrumentSandboxClient(client);

  await withTrace("resume", () =>
    withCustomSpan(async () => await client.resume!({} as never), {
      data: {
        name: "sandbox.resume_session",
        data: { agent_name: "coder", backend_id: "unix_local" },
      },
    }),
  );
  const span = only(await exported(), named("sandbox.resume_session"));

  expect(span.attributes["sandbox.id"]).toBe("sbx_resumed");
  expect(span.attributes["sandbox.backend"]).toBe("unix_local");
});

test("a mask applies to the command before it is sent", async () => {
  makeProcessor({
    mask: (value, { key }) => (key === "process.command_line" ? "[masked]" : value),
  });
  await runCommand("echo secret");

  const exec = only(await exported(), named("sandbox.exec"));
  expect(exec.attributes["process.command_line"]).toBe("[masked]");
});

test("Chat Completions model calls carry their model and usage", async () => {
  makeProcessor();
  await runCommand("true", { model: chatCompletionsModel("true") });
  const spans = await exported();

  const chats = spans.filter((span) => span.attributes["gen_ai.operation.name"] === "chat");
  expect(chats.map((span) => span.name)).toEqual(["chat gpt-4.1-mini", "chat gpt-4.1-mini"]);
  const first = chats.find((span) => span.attributes["gen_ai.response.id"] === "chatcmpl_1")!;
  expect(first.attributes).toMatchObject({
    "gen_ai.request.model": "gpt-4.1-mini",
    "gen_ai.response.model": "gpt-4.1-mini-2025-04-14",
    "gen_ai.usage.input_tokens": 100,
    "gen_ai.usage.output_tokens": 20,
    "gen_ai.usage.cache_read.input_tokens": 40,
    "gen_ai.usage.reasoning.output_tokens": 5,
  });
});

/** Every exported span except the root has an exported parent. */
function expectConnected(spans: ReadableSpan[], rootName: string): void {
  const ids = new Set(spans.map((span) => span.spanContext().spanId));
  const root = only(spans, named(rootName));
  const orphans = spans.filter((span) => span !== root && !ids.has(parentOf(span) ?? ""));
  expect(orphans.map((span) => span.name)).toEqual([]);
}

test("a run that throws still exports its root once flushed", async () => {
  const processor = makeProcessor();
  const failing = new OpenAIResponsesModel(
    scriptedClient([{ error: { message: "bad request", type: "invalid_request_error" } }], 400),
    "gpt-5-mini",
  );
  await expect(runCommand("true", { model: failing })).rejects.toThrow();

  expect((await exported()).filter(named("workflow"))).toEqual([]);
  await processor.forceFlush();
  const spans = await exported();

  expectConnected(spans, "workflow");
  expect(
    only(spans, (span) => span.attributes["gen_ai.operation.name"] === "chat").status.code,
  ).toBe(2);
});

test("spans that end after their trace still reach the root", async () => {
  // A processor registered first that takes time to handle each span end, such as the default
  // OpenAI exporter during an export, delays this processor's span ends past the trace end.
  const slow: TracingProcessor = {
    async onTraceStart() {},
    async onTraceEnd() {},
    async onSpanStart() {},
    async onSpanEnd() {
      await new Promise((resolve) => setTimeout(resolve, 20));
    },
    async forceFlush() {},
    async shutdown() {},
  };
  makeProcessor({}, [slow]);
  await runCommand("true");

  await vi.waitFor(async () =>
    expect((await exported()).filter(named("workflow"))).toHaveLength(1),
  );
  const spans = await exported();

  expectConnected(spans, "workflow");
  expect(spans.filter((span) => span.attributes["gen_ai.operation.name"] === "chat")).toHaveLength(
    2,
  );
});

test("an exit line in the output of a running command is not an exit code", async () => {
  const processor = makeProcessor();
  const trace = new Trace({ traceId: "trace_fedcba9876543210fedcba9876543210", name: "manual" });
  await processor.onTraceStart(trace);
  const span = new Span(
    {
      traceId: trace.traceId,
      data: {
        type: "function",
        name: "write_stdin",
        input: "{}",
        output: [
          "Chunk ID: abc123",
          "Wall time: 1.0000 seconds",
          "Process running with session ID 3",
          "Output:",
          "Process exited with code 0",
        ].join("\n"),
      },
    },
    processor,
  );
  span.start();
  span.end();
  await processor.onTraceEnd(trace);

  const tool = only(await exported(), named("write_stdin"));
  expect(tool.attributes["process.exit.code"]).toBeUndefined();
});

/** Starts and ends a custom span of `trace` through `processor`, the way the SDK does. */
function customSpan(
  processor: TracingProcessor,
  trace: Trace,
  name: string,
  parent?: Span<never>,
): Span<never> {
  const span = new Span(
    {
      traceId: trace.traceId,
      parentId: parent?.spanId,
      data: { type: "custom", name, data: {} },
    },
    processor,
  ) as Span<never>;
  span.start();

  return span;
}

test("a flush between two runs of one trace keeps them under one root", async () => {
  const processor = makeProcessor();
  await withTrace("two runs", async () => {
    await runCommand("echo first");
    await processor.forceFlush();
    await runCommand("echo second");
  });
  const spans = await exported();

  expectConnected(spans, "two runs");
  expect(new Set(spans.map((span) => span.spanContext().traceId)).size).toBe(1);
});

test("spans that start after shutdown still join their trace", async () => {
  const processor = makeProcessor();
  const trace = new Trace({ traceId: "trace_11111111111111111111111111111111", name: "draining" });
  await processor.onTraceStart(trace);
  const inFlight = customSpan(processor, trace, "in flight");

  // A server that drains on SIGTERM keeps running after the SDK calls shutdown.
  await processor.shutdown();
  const late = customSpan(processor, trace, "late", inFlight);
  late.end();
  inFlight.end();
  await processor.onTraceEnd(trace);
  const spans = await exported();

  expectConnected(spans, "draining");
  const parent = only(spans, named("in flight"));
  expect(parentOf(only(spans, named("late")))).toBe(parent.spanContext().spanId);
});

test("a span after its trace ended joins the ended root", async () => {
  const processor = makeProcessor();
  const trace = new Trace({ traceId: "trace_22222222222222222222222222222222", name: "ended" });
  await processor.onTraceStart(trace);
  await processor.onTraceEnd(trace);
  // Such as the memory spans that a sandbox agent writes after its run.
  customSpan(processor, trace, "memory flush").end();
  const spans = await exported();

  expectConnected(spans, "ended");
  expect(new Set(spans.map((span) => span.spanContext().traceId)).size).toBe(1);
});

test("the root of a trace that never ends is exported after the idle wait", async () => {
  vi.useFakeTimers();

  try {
    const processor = makeProcessor();
    const trace = new Trace({ traceId: "trace_33333333333333333333333333333333", name: "threw" });
    await processor.onTraceStart(trace);
    customSpan(processor, trace, "step").end();
    await vi.advanceTimersByTimeAsync(59_000);
    expect((await exported()).filter(named("threw"))).toEqual([]);

    await vi.advanceTimersByTimeAsync(2_000);
    expectConnected(await exported(), "threw");
  } finally {
    vi.useRealTimers();
  }
});
