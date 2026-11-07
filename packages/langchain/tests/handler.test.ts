import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { flush, init, shutdown, startActiveSpan } from "@telemetry-dev/sdk";
import { afterEach, expect, test } from "vitest";
import { z } from "zod";

import { TelemetryDevCallbackHandler } from "../src/index.ts";

class TestChatModel extends BaseChatModel {
  constructor(
    private readonly reply: AIMessage,
    private readonly tools: unknown[] = [],
  ) {
    super({});
  }

  override invocationParams() {
    return { tools: this.tools };
  }

  _llmType(): string {
    return "test";
  }

  override getLsParams() {
    return {
      ls_provider: "google_genai",
      ls_model_name: "gemini-test",
      ls_model_type: "chat" as const,
      ls_temperature: 0.2,
      ls_max_tokens: 64,
      ls_stop: ["END"],
    };
  }

  async _generate(): Promise<ChatResult> {
    return { generations: [{ message: this.reply, text: this.reply.text }] };
  }

  override async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    for (const text of ["Hel", "lo"]) {
      // As @langchain/anthropic streams, the last chunk carries the model and stop reason.
      const additional_kwargs =
        text === "lo" ? { model: "claude-streamed", stop_reason: "end_turn" } : {};
      const message = new AIMessageChunk({ content: text, additional_kwargs });
      const chunk = new ChatGenerationChunk({ message, text });
      yield chunk;
      await runManager?.handleLLMNewToken(text, undefined, undefined, undefined, undefined, {
        chunk,
      });
    }
  }
}

const reply = new AIMessage({
  content: "Hello",
  id: "msg_123",
  response_metadata: { model_name: "gemini-test-001", finish_reason: "stop" },
  usage_metadata: {
    input_tokens: 12,
    output_tokens: 3,
    total_tokens: 15,
    input_token_details: { cache_read: 4 },
    output_token_details: { reasoning: 1 },
  },
});

function setup(): InMemorySpanExporter {
  const spans = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "langchain-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter: spans },
  );

  return spans;
}

async function finished(spans: InMemorySpanExporter): Promise<ReadableSpan[]> {
  await flush();

  return spans.getFinishedSpans();
}

function byName(spans: ReadableSpan[], name: string): ReadableSpan {
  const span = spans.find((s) => s.name === name);
  expect(span, `span ${name}`).toBeDefined();

  return span!;
}

function parentOf(span: ReadableSpan): string | undefined {
  return span.parentSpanContext?.spanId;
}

afterEach(async () => {
  await shutdown();
});

test("records a chain with its prompt and model call as one span tree", async () => {
  const spans = setup();
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Be brief."],
    ["human", "{question}"],
  ]);

  await prompt.pipe(new TestChatModel(reply)).invoke(
    { question: "Hi?" },
    {
      callbacks: [new TelemetryDevCallbackHandler()],
      metadata: { userId: "user-1", session_id: "session-1", plan: "pro" },
    },
  );

  const all = await finished(spans);
  const root = byName(all, "RunnableSequence");
  const promptSpan = byName(all, "ChatPromptTemplate");
  const chat = byName(all, "chat gemini-test");

  expect(all).toHaveLength(3);
  expect(parentOf(promptSpan)).toBe(root.spanContext().spanId);
  expect(parentOf(chat)).toBe(root.spanContext().spanId);
  expect(new Set(all.map((s) => s.spanContext().traceId)).size).toBe(1);
  expect(new Set(all.map((s) => s.attributes["langchain.run.id"])).size).toBe(3);

  expect(root.attributes).toMatchObject({
    "gen_ai.operation.name": "function",
    "gen_ai.input.messages": '{"question":"Hi?"}',
    "gen_ai.conversation.id": "session-1",
    "user.id": "user-1",
    "td.metadata.plan": "pro",
  });
  expect(root.attributes["gen_ai.output.messages"]).toBe('{"role":"assistant","content":"Hello"}');
  expect(chat.attributes["td.metadata.plan"]).toBeUndefined();

  expect(chat.attributes).toMatchObject({
    "gen_ai.operation.name": "chat",
    "gen_ai.request.model": "gemini-test",
    "gen_ai.provider.name": "gcp.gen_ai",
    "gen_ai.request.temperature": 0.2,
    "gen_ai.request.max_tokens": 64,
    "gen_ai.request.stop_sequences": ["END"],
    "gen_ai.response.model": "gemini-test-001",
    "gen_ai.response.id": "msg_123",
    "gen_ai.response.finish_reasons": ["stop"],
    "gen_ai.usage.input_tokens": 12,
    "gen_ai.usage.output_tokens": 3,
    "gen_ai.usage.total_tokens": 15,
    "gen_ai.usage.cache_read.input_tokens": 4,
    "gen_ai.usage.reasoning.output_tokens": 1,
    "gen_ai.conversation.id": "session-1",
    "user.id": "user-1",
    "gen_ai.input.messages":
      '[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi?"}]',
    "gen_ai.output.messages": '[{"role":"assistant","content":"Hello"}]',
  });
});

test("records a LangGraph run as an agent span with node spans and skips hidden runs", async () => {
  const spans = setup();
  const State = Annotation.Root({ question: Annotation<string>, answer: Annotation<string> });
  const model = new TestChatModel(reply);

  const graph = new StateGraph(State)
    .addNode("respond", async (state) => ({
      answer: (await model.invoke(state.question)).text,
    }))
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile();

  await graph.invoke(
    { question: "Hi?" },
    { callbacks: [new TelemetryDevCallbackHandler()], configurable: { thread_id: "thread-1" } },
  );

  const all = await finished(spans);
  const agent = byName(all, "invoke_agent LangGraph");
  const node = byName(all, "respond");
  const chat = byName(all, "chat gemini-test");

  expect(all.map((s) => s.name)).not.toContain("__start__");
  expect(agent.attributes).toMatchObject({
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.agent.name": "LangGraph",
  });
  expect(parentOf(node)).toBe(agent.spanContext().spanId);
  expect(parentOf(chat)).toBe(node.spanContext().spanId);

  for (const span of all) {
    expect(span.attributes["gen_ai.conversation.id"]).toBe("thread-1");
    expect(span.attributes["td.metadata.thread_id"]).toBeUndefined();
  }
});

test("records the answer of a graph with a message state as its output", async () => {
  const spans = setup();
  const State = Annotation.Root({
    messages: Annotation<BaseMessage[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  });
  const model = new TestChatModel(reply);

  const graph = new StateGraph(State)
    .addNode("respond", async (state) => ({ messages: [await model.invoke(state.messages)] }))
    .addEdge(START, "respond")
    .addEdge("respond", END)
    .compile();

  await graph.invoke(
    { messages: [new HumanMessage("Hi?")] },
    { callbacks: [new TelemetryDevCallbackHandler()] },
  );

  const agent = byName(await finished(spans), "invoke_agent LangGraph");

  expect(agent.attributes["gen_ai.input.messages"]).toBe(
    '{"messages":[{"role":"user","content":"Hi?"}]}',
  );
  expect(agent.attributes["gen_ai.output.messages"]).toBe(
    '[{"role":"assistant","content":"Hello"}]',
  );
});

test("records the offered tools and each tool call of a reply once", async () => {
  const spans = setup();

  const toolReply = new AIMessage({
    content: [
      { type: "text", text: "Adding." },
      { type: "tool_use", id: "call_1", name: "add", input: { a: 1, b: 2 } },
    ],
    tool_calls: [{ id: "call_1", name: "add", args: { a: 1, b: 2 }, type: "tool_call" }],
  });

  const tools = [
    { type: "function", function: { name: "add", description: "Add.", parameters: {} } },
    { name: "multiply", description: "Multiply.", input_schema: {} },
    { type: "web_search_20250305", name: "web_search" },
    { functionDeclarations: [{ name: "divide" }, { name: "subtract" }] },
  ];

  await new TestChatModel(toolReply, tools).invoke("Add 1 and 2.", {
    callbacks: [new TelemetryDevCallbackHandler()],
  });

  const chat = byName(await finished(spans), "chat gemini-test");

  expect(JSON.parse(String(chat.attributes["gen_ai.tool.definitions"]))).toEqual([
    { type: "function", name: "add" },
    { type: "function", name: "multiply" },
    { type: "web_search_20250305", name: "web_search" },
    { type: "function", name: "divide" },
    { type: "function", name: "subtract" },
  ]);
  expect(JSON.parse(String(chat.attributes["gen_ai.output.messages"]))).toEqual([
    {
      role: "assistant",
      content: [{ type: "text", text: "Adding." }],
      tool_calls: [
        { id: "call_1", type: "function", function: { name: "add", arguments: '{"a":1,"b":2}' } },
      ],
    },
  ]);
});

test("records a tool call with its arguments, call id, and result", async () => {
  const spans = setup();

  const add = tool(async ({ a, b }: { a: number; b: number }) => String(a + b), {
    name: "add",
    description: "Add two numbers.",
    schema: z.object({ a: z.number(), b: z.number() }),
  });

  await add.invoke(
    { id: "call_1", name: "add", args: { a: 1, b: 2 }, type: "tool_call" },
    { callbacks: [new TelemetryDevCallbackHandler()] },
  );

  const span = byName(await finished(spans), "execute_tool add");

  expect(span.attributes).toMatchObject({
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.name": "add",
    "gen_ai.tool.call.id": "call_1",
    "gen_ai.tool.call.arguments": '{"a":1,"b":2}',
    "gen_ai.tool.call.result": "3",
  });
});

test("marks the failed run and each run that it fails", async () => {
  const spans = setup();

  const fail = RunnableLambda.from((): string => {
    throw new TypeError("bad input");
  }).withConfig({ runName: "fail" });

  const chain = RunnableLambda.from((input: string) => input)
    .pipe(fail)
    .withConfig({ runName: "outer" });

  await expect(
    chain.invoke("x", { callbacks: [new TelemetryDevCallbackHandler()] }),
  ).rejects.toThrow("bad input");

  const all = await finished(spans);

  for (const name of ["outer", "fail"]) {
    const span = byName(all, name);
    expect(span.status.code).toBe(SpanStatusCode.ERROR);
    expect(span.attributes["error.type"]).toBe("TypeError");
  }
});

test("nests the top run in the active span of the app", async () => {
  const spans = setup();

  await startActiveSpan("handle-request", () =>
    new TestChatModel(reply).invoke("Hi?", { callbacks: [new TelemetryDevCallbackHandler()] }),
  );

  const all = await finished(spans);

  expect(parentOf(byName(all, "chat gemini-test"))).toBe(
    byName(all, "handle-request").spanContext().spanId,
  );
});

test("nests concurrent runs each in its own app span", async () => {
  const spans = setup();
  const handler = new TelemetryDevCallbackHandler();
  const slow = RunnableLambda.from(async (input: string) => {
    await new Promise((resolve) => setTimeout(resolve, 5));

    return input;
  });

  await Promise.all(
    ["first", "second", "third"].map((name) =>
      startActiveSpan(name, () =>
        slow.withConfig({ runName: `run ${name}` }).invoke(name, { callbacks: [handler] }),
      ),
    ),
  );

  const all = await finished(spans);

  for (const name of ["first", "second", "third"]) {
    expect(parentOf(byName(all, `run ${name}`))).toBe(byName(all, name).spanContext().spanId);
  }
});

test("records the time to the first chunk and the response of a streamed call", async () => {
  const spans = setup();
  const stream = await new TestChatModel(reply).stream("Hi?", {
    callbacks: [new TelemetryDevCallbackHandler()],
  });

  for await (const _chunk of stream) {
    // Read the stream to its end.
  }

  const chat = byName(await finished(spans), "chat gemini-test");

  expect(chat.attributes["gen_ai.response.time_to_first_chunk"]).toBeGreaterThanOrEqual(0);
  expect(chat.attributes["gen_ai.response.model"]).toBe("claude-streamed");
  expect(chat.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);
  expect(chat.attributes["gen_ai.output.messages"]).toBe(
    '[{"role":"assistant","content":"Hello"}]',
  );
});
