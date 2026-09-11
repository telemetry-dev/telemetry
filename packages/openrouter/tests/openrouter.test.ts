import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { HTTPClient, OpenRouter, type Fetcher } from "@openrouter/sdk";
import { Chat } from "@openrouter/sdk/sdk/chat.js";
import { Embeddings } from "@openrouter/sdk/sdk/embeddings.js";
import { EventStream } from "@openrouter/sdk/lib/event-streams.js";
import { Responses } from "@openrouter/sdk/sdk/responses.js";
import { flush, init, type MaskFn, shutdown, startActiveSpan } from "@telemetry-dev/sdk";
import { afterEach, expect, test, vi } from "vitest";

import { instrumentOpenRouter, uninstrumentOpenRouter, wrapOpenRouter } from "../src/index.ts";

const SPAN_STATUS_UNSET = 0;
const SPAN_STATUS_ERROR = 2;

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;
interface JsonRecord {
  [key: string]: JsonValue;
}
interface CapturedRequest {
  method: string;
  path: string;
  body: unknown;
}

function jsonResponse<T>(body: T): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function sseResponse(events: unknown[]): Response {
  const body = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function openSseResponse(events: unknown[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      }
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function createFakeFetcher(...responses: Response[]) {
  const requests: CapturedRequest[] = [];
  const fetcher: Fetcher = async (input, init) => {
    const request = input instanceof Request ? input : new Request(input, init);
    const text = await request.clone().text();
    requests.push({
      method: request.method,
      path: new URL(request.url).pathname,
      body: text.length > 0 ? JSON.parse(text) : undefined,
    });
    const response = responses.shift();
    if (!response) throw new Error(`unexpected request to ${request.url}`);
    return response;
  };
  return { fetcher, requests };
}

function setupSpans(mask?: MaskFn): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "openrouter-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      mask,
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter },
  );
  return spanExporter;
}

function clientWith(fetcher: Fetcher, wrapped = true): OpenRouter {
  const client = new OpenRouter({
    apiKey: "test",
    httpClient: new HTTPClient({ fetcher }),
    retryConfig: { strategy: "none" },
  });
  return wrapped ? wrapOpenRouter(client) : client;
}

async function finishedSpans(
  exporter: InMemorySpanExporter,
  expectedCount: number,
): Promise<ReadableSpan[]> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await flush();
    const spans = exporter.getFinishedSpans();
    if (spans.length === expectedCount) return spans;
    if (spans.length > expectedCount) expect(spans).toHaveLength(expectedCount);
    await Promise.resolve();
  }
  expect(exporter.getFinishedSpans()).toHaveLength(expectedCount);
  return exporter.getFinishedSpans();
}

async function exportedSpan(exporter: InMemorySpanExporter): Promise<ReadableSpan> {
  const spans = await finishedSpans(exporter, 1);
  return spans[0]!;
}

function jsonAttr<T>(span: ReadableSpan, key: string): T {
  const value = span.attributes[key];
  expect(String(value)).toBe(value);
  return JSON.parse(String(value)) as T;
}

function chatBody(
  id: string,
  options: {
    model?: string;
    cost?: number;
    upstreamCost?: number;
    content?: string;
  } = {},
) {
  const usage = {
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
    prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 6 },
    completion_tokens_details: { reasoning_tokens: 2 },
  };
  if (options.cost !== undefined) Object.assign(usage, { cost: options.cost });
  if (options.upstreamCost !== undefined) {
    Object.assign(usage, {
      cost_details: {
        upstream_inference_completions_cost: 0.006,
        upstream_inference_cost: options.upstreamCost,
        upstream_inference_prompt_cost: 0.003,
      },
    });
  }
  return {
    id,
    object: "chat.completion",
    created: 1,
    model: options.model ?? "openai/gpt-4o-2024-11-20",
    system_fingerprint: null,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: options.content ?? "Hello." },
        finish_reason: "stop",
      },
    ],
    usage,
  };
}

function chatChunk(
  id: string,
  delta: JsonRecord,
  options: {
    finishReason?: string | null;
    usage?: unknown;
    error?: unknown;
    model?: string;
  } = {},
) {
  const chunk = {
    id,
    object: "chat.completion.chunk",
    created: 1,
    model: options.model ?? "openai/gpt-4o-2024-11-20",
    choices: [
      {
        index: 0,
        delta,
        finish_reason: options.finishReason ?? null,
      },
    ],
  };
  if (options.usage !== undefined) Object.assign(chunk, { usage: options.usage });
  if (options.error !== undefined) Object.assign(chunk, { error: options.error });
  return chunk;
}

function responsesBody(
  id: string,
  status: "completed" | "failed" | "incomplete" | "in_progress" = "completed",
) {
  const output = Array<JsonValue>();
  return {
    completed_at: status === "completed" ? 2 : null,
    created_at: 1,
    error: status === "failed" ? { code: "server_error", message: "provider failed" } : null,
    frequency_penalty: null,
    id,
    incomplete_details: status === "incomplete" ? { reason: "max_output_tokens" } : null,
    instructions: null,
    metadata: null,
    model: "openai/gpt-4o-2024-11-20",
    object: "response",
    output,
    parallel_tool_calls: false,
    presence_penalty: null,
    status,
    temperature: null,
    tool_choice: "auto",
    tools: [],
    top_p: null,
    usage: {
      input_tokens: 12,
      output_tokens: 7,
      total_tokens: 19,
      input_tokens_details: { cached_tokens: 4, cache_write_tokens: 9 },
      output_tokens_details: { reasoning_tokens: 3 },
      cost: 0.021,
      cost_details: {
        upstream_inference_cost: 0.02,
        upstream_inference_input_cost: 0.008,
        upstream_inference_output_cost: 0.012,
      },
    },
  };
}

function embeddingsBody(id: string) {
  return {
    id,
    object: "list",
    model: "openai/text-embedding-3-small-2024",
    data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
    usage: {
      prompt_tokens: 8,
      total_tokens: 8,
      cost: 0.0004,
      cost_details: {
        upstream_inference_completions_cost: 0,
        upstream_inference_cost: 0.0003,
        upstream_inference_prompt_cost: 0.0003,
      },
    },
  };
}

async function collectStream(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const chunks: unknown[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

function prototypeMethod<T extends object>(target: T, key: string) {
  const value: unknown = Object.getOwnPropertyDescriptor(target, key)?.value;
  return value instanceof Function ? value : undefined;
}

afterEach(async () => {
  uninstrumentOpenRouter();
  await shutdown();
  vi.restoreAllMocks();
});

test("chat maps request, response, usage, primary cost, provider, and sampling fields", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    jsonResponse(
      chatBody("chat_1", {
        cost: 0.012,
        upstreamCost: 0.011,
      }),
    ),
  );
  const client = clientWith(fake.fetcher);
  const messages = [
    { role: "system" as const, content: "Be terse" },
    { role: "user" as const, content: "Say hello" },
  ];

  await client.chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      maxCompletionTokens: 64,
      stop: "END",
      seed: 7,
      frequencyPenalty: 0.1,
      presencePenalty: 0.2,
    },
  });

  const span = await exportedSpan(spans);
  expect(fake.requests).toEqual([
    {
      method: "POST",
      path: "/api/v1/chat/completions",
      body: {
        frequency_penalty: 0.1,
        max_completion_tokens: 64,
        messages,
        model: "openai/gpt-4o",
        presence_penalty: 0.2,
        seed: 7,
        stop: "END",
        stream: false,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
      },
    },
  ]);
  expect(span.name).toBe("chat openai/gpt-4o");
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("openrouter");
  expect(span.attributes["gen_ai.request.model"]).toBe("openai/gpt-4o");
  expect(span.attributes["gen_ai.response.model"]).toBe("openai/gpt-4o-2024-11-20");
  expect(span.attributes["gen_ai.response.id"]).toBe("chat_1");
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(messages));
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Hello." },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(15);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.012);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.top_k"]).toBe(40);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(64);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["END"]);
  expect(span.attributes["gen_ai.request.seed"]).toBe(7);
  expect(span.attributes["gen_ai.request.frequency_penalty"]).toBe(0.1);
  expect(span.attributes["gen_ai.request.presence_penalty"]).toBe(0.2);
});

test("cost falls back to upstream inference cost when usage cost is absent", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    jsonResponse(chatBody("chat_cost_fallback", { upstreamCost: 0.009 })),
  );

  await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Cost" }],
    },
  });

  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.009);
});

test("chat output chunks recognize decoded reasoning text but not opaque metadata", async () => {
  const spans = setupSpans();

  const fake = createFakeFetcher(
    sseResponse([
      chatChunk("tool_timing", { role: "assistant", content: "" }),
      chatChunk("tool_timing", {
        reasoning_details: [{ type: "reasoning.text", text: "Inspect" }],
      }),
      chatChunk("tool_timing", {
        content: "once",
        reasoning_details: [{ type: "reasoning.summary", summary: "Summary" }],
      }),
      chatChunk("tool_timing", {
        reasoning_details: [{ type: "reasoning.summary", summary: "Summary only" }],
      }),
      chatChunk("tool_timing", {
        reasoning_details: [{ type: "reasoning.summary", summary: "" }],
      }),
      chatChunk("tool_timing", {
        reasoning_details: [
          { type: "reasoning.encrypted", data: "opaque" },
          { type: "reasoning.text", signature: "signed", text: "" },
        ],
      }),
      chatChunk("tool_timing", {
        tool_calls: [
          {
            index: 0,
            id: "call_1",
            type: "function",
            function: { name: "weather", arguments: '{"city":' },
          },
        ],
      }),
      chatChunk("tool_timing", { tool_calls: [{ index: 0, function: { arguments: "" } }] }),
      chatChunk("tool_timing", { content: null }),
      chatChunk("tool_timing", { tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }] }),
    ]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Weather" }],
      stream: true,
    },
  });

  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  expect(await collectStream(stream)).toHaveLength(10);
  const span = await exportedSpan(spans);
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 4 },
  });
});

test("Responses output timing accepts code, MCP, and audio transcript deltas but excludes done snapshots", async () => {
  const spans = setupSpans();

  const outputEvents = [
    { type: "response.code_interpreter_call_code.delta", delta: "print(1)" },
    { type: "response.code_interpreter_call_code.done", code: "print(1)" },
    { type: "response.mcp_call_arguments.delta", delta: '{"city":"Paris"}' },
    { type: "response.mcp_call_arguments.delta", delta: "" },
    { type: "response.mcp_call_arguments.done", arguments: '{"city":"Paris"}' },
    { type: "response.audio.transcript.delta", delta: "Hello" },
    { type: "response.audio.transcript.delta", delta: "" },
    { type: "response.audio.transcript.delta", delta: " world" },
    { type: "response.audio.transcript.done", transcript: "Hello world" },
  ];

  const fake = createFakeFetcher(
    sseResponse([
      ...outputEvents,
      { type: "response.completed", response: responsesBody("resp_metric") },
    ]),
    openSseResponse(outputEvents),
  );

  const client = clientWith(fake.fetcher);

  const completed = await client.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Run", stream: true },
  });

  if (!(completed instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(completed);

  const interrupted = await client.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Run", stream: true },
  });

  if (!(interrupted instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = interrupted.getReader();

  for (const _event of outputEvents) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  const finished = await finishedSpans(spans, 2);
  expect(finished).toHaveLength(2);

  for (const span of finished) {
    expect(span).toMatchObject({
      [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 3 },
    });
  }
});

test.each(["chat", "responses"])("%s timestamps precede telemetry mapping", async (operation) => {
  const spans = setupSpans();
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => now);

  const timings = [
    [100, 30],
    [240, 90],
  ] as const;

  let index = 0;

  const source = new ReadableStream(
    {
      pull(controller) {
        const timing = timings[index++];

        if (!timing) {
          controller.close();

          return;
        }

        const [receivedAt, mappingMs] = timing;
        now = receivedAt;
        controller.enqueue(
          operation === "chat"
            ? {
                get choices() {
                  now = receivedAt + mappingMs;

                  return [{ index: 0, delta: { content: "A" } }];
                },
              }
            : {
                get type() {
                  now = receivedAt + mappingMs;

                  return "response.output_text.delta";
                },
                delta: "A",
              },
        );
      },
    },
    { highWaterMark: 0 },
  );

  const client = wrapOpenRouter({
    chat: { send: async (_params: unknown) => source },
    responses: { send: async (_params: unknown) => source },
    embeddings: {},
  });

  const stream =
    operation === "chat"
      ? await client.chat.send({
          chatRequest: { model: "openai/gpt-4o", messages: [], stream: true },
        })
      : await client.responses.send({
          responsesRequest: { model: "openai/gpt-4o", input: "Hi", stream: true },
        });

  expect(await collectStream(stream)).toHaveLength(2);
  expect(await exportedSpan(spans)).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: {
      count: 1,
      sum: 0.14,
      min: 0.14,
      max: 0.14,
    },
  });
});

test("consumed chat streams remain readable, reconstruct tool calls, and use final usage without request mutation", async () => {
  const spans = setupSpans();
  const finalUsage = {
    prompt_tokens: 11,
    completion_tokens: 6,
    total_tokens: 17,
    prompt_tokens_details: { cached_tokens: 2 },
    completion_tokens_details: { reasoning_tokens: 4 },
    cost: 0.015,
    cost_details: {
      upstream_inference_completions_cost: 0.008,
      upstream_inference_cost: 0.014,
      upstream_inference_prompt_cost: 0.006,
    },
  };
  const fake = createFakeFetcher(
    sseResponse([
      chatChunk("stream_1", { role: "assistant", content: "" }),
      chatChunk("stream_1", {
        reasoning: "Inspecting weather.",
        reasoning_details: [{ type: "reasoning.text", index: 0, text: "Inspecting weather." }],
        tool_calls: [
          {
            index: 0,
            id: "call_1",
            type: "function",
            function: { name: "weather", arguments: '{"city"' },
          },
        ],
      }),
      chatChunk(
        "stream_1",
        { tool_calls: [{ index: 0, function: { arguments: ':"Paris"}' } }] },
        { finishReason: "tool_calls", usage: finalUsage },
      ),
    ]),
  );
  const client = clientWith(fake.fetcher);
  const request = {
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user" as const, content: "Weather" }],
      stream: true as const,
    },
  };

  const stream = await client.chat.send(request);
  expect(stream).toBeInstanceOf(ReadableStream);
  expect(stream).toBeInstanceOf(EventStream);
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const chunks = await collectStream(stream);

  expect(chunks).toHaveLength(3);
  expect(request).toEqual({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Weather" }],
      stream: true,
    },
  });
  expect(fake.requests[0]?.body).toEqual({
    messages: [{ role: "user", content: "Weather" }],
    model: "openai/gpt-4o",
    stream: true,
  });
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(span.attributes["gen_ai.response.id"]).toBe("stream_1");
  expect(span.attributes["gen_ai.response.model"]).toBe("openai/gpt-4o-2024-11-20");
  expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toEqual(expect.any(Number));
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: null,
      reasoning: "Inspecting weather.",
      reasoningDetails: [{ type: "reasoning.text", index: 0, text: "Inspecting weather." }],
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "weather", arguments: '{"city":"Paris"}' },
        },
      ],
    },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(11);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(17);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.015);
});

test("chat stream capture is bounded without dropping chunks or terminal metadata", async () => {
  const spans = setupSpans();
  const content = "x".repeat(1024);
  const deltas = Array.from({ length: 70 }, (_, index) =>
    chatChunk("stream_bounded", index === 0 ? { role: "assistant", content } : { content }),
  );
  const usage = { prompt_tokens: 3, completion_tokens: 70, total_tokens: 73, cost: 0.04 };
  const events = [...deltas, chatChunk("stream_bounded", {}, { finishReason: "stop", usage })];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Long answer" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const chunks = await collectStream(stream);

  expect(chunks).toHaveLength(events.length);
  const span = await exportedSpan(spans);
  const output = jsonAttr<Array<{ content?: string }>>(span, "gen_ai.output.messages");
  expect(output[0]?.content?.length).toBeGreaterThan(0);
  expect(output[0]?.content?.length).toBeLessThan(content.length * deltas.length);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(73);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.04);
});

test("chat stream chunk errors end one error span with partial output", async () => {
  const spans = setupSpans();
  const errorChunk = {
    id: "stream_error",
    object: "chat.completion.chunk",
    created: 1,
    model: "openai/gpt-4o-2024-11-20",
    choices: [],
    error: { code: 502, message: "upstream disconnected" },
  };
  const fake = createFakeFetcher(
    openSseResponse([
      chatChunk("stream_error", { role: "assistant", content: "Partial" }),
      errorChunk,
    ]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Stream" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  expect((await reader.read()).done).toBe(false);
  expect((await reader.read()).done).toBe(false);

  const span = await exportedSpan(spans);
  await reader.cancel();
  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("Error");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Partial" },
  ]);
  expect(span.events[0]?.attributes?.["exception.message"]).toContain("upstream disconnected");
  expect(span.events[0]?.attributes?.["exception.message"]).toContain("502");
});

test("early chat stream cancellation captures partial output and ends once", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    sseResponse([
      chatChunk("stream_cancel", { role: "assistant", content: "First" }),
      chatChunk("stream_cancel", { content: " ignored" }, { finishReason: "stop" }),
    ]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Cancel" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  const first = await reader.read();
  expect(first.done).toBe(false);
  await reader.cancel("caller stopped");

  const span = await exportedSpan(spans);
  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "First" },
  ]);
});

test("early Responses stream cancellation captures final text, reasoning, summary, and refusal", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.output_text.delta",
      sequence_number: 1,
      item_id: "msg_partial",
      output_index: 0,
      content_index: 0,
      delta: "Partial",
      logprobs: [],
    },
    {
      type: "response.refusal.delta",
      sequence_number: 2,
      item_id: "msg_partial",
      output_index: 0,
      content_index: 1,
      delta: "No",
    },
    {
      type: "response.reasoning_text.delta",
      sequence_number: 3,
      item_id: "reason_partial",
      output_index: 1,
      content_index: 0,
      delta: "Thinking",
    },
    {
      type: "response.reasoning_summary_text.delta",
      sequence_number: 4,
      item_id: "reason_partial",
      output_index: 1,
      summary_index: 0,
      delta: "Summary",
    },
    {
      type: "response.output_text.done",
      sequence_number: 5,
      item_id: "msg_partial",
      output_index: 0,
      content_index: 0,
      text: "Final answer",
      logprobs: [],
    },
    {
      type: "response.refusal.done",
      sequence_number: 6,
      item_id: "msg_partial",
      output_index: 0,
      content_index: 1,
      refusal: "Final refusal",
    },
    {
      type: "response.reasoning_text.done",
      sequence_number: 7,
      item_id: "reason_partial",
      output_index: 1,
      content_index: 0,
      text: "Final reasoning",
    },
    {
      type: "response.reasoning_summary_text.done",
      sequence_number: 8,
      item_id: "reason_partial",
      output_index: 1,
      summary_index: 0,
      text: "Final summary",
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Cancel", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  for (const _event of events) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  const span = await exportedSpan(spans);
  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      id: "msg_partial",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [
        { type: "output_text", text: "Final answer", annotations: [] },
        { type: "refusal", refusal: "Final refusal" },
      ],
    },
    {
      id: "reason_partial",
      type: "reasoning",
      status: "in_progress",
      summary: [{ type: "summary_text", text: "Final summary" }],
      content: [{ type: "reasoning_text", text: "Final reasoning" }],
    },
  ]);
});

test("early Responses stream cancellation retains partial text, reasoning, summary, and refusal", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.output_text.delta",
      sequence_number: 1,
      item_id: "msg_partial_only",
      output_index: 0,
      content_index: 0,
      delta: "Partial",
      logprobs: [],
    },
    {
      type: "response.refusal.delta",
      sequence_number: 2,
      item_id: "msg_partial_only",
      output_index: 0,
      content_index: 1,
      delta: "No",
    },
    {
      type: "response.reasoning_text.delta",
      sequence_number: 3,
      item_id: "reason_partial_only",
      output_index: 1,
      content_index: 0,
      delta: "Thinking",
    },
    {
      type: "response.reasoning_summary_text.delta",
      sequence_number: 4,
      item_id: "reason_partial_only",
      output_index: 1,
      summary_index: 0,
      delta: "Summary",
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Cancel", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  for (const _event of events) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  expect(jsonAttr(await exportedSpan(spans), "gen_ai.output.messages")).toEqual([
    {
      id: "msg_partial_only",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [
        { type: "output_text", text: "Partial", annotations: [] },
        { type: "refusal", refusal: "No" },
      ],
    },
    {
      id: "reason_partial_only",
      type: "reasoning",
      status: "in_progress",
      summary: [{ type: "summary_text", text: "Summary" }],
      content: [{ type: "reasoning_text", text: "Thinking" }],
    },
  ]);
});

test.each(["response.created", "response.in_progress"])(
  "%s output survives Responses stream cancellation",
  async (type) => {
    const spans = setupSpans();
    const response = responsesBody(`resp_${type}`, "in_progress");
    response.output = [
      {
        id: "msg_response_event",
        type: "message",
        status: "in_progress",
        role: "assistant",
        content: [{ type: "output_text", text: "Retained output", annotations: [] }],
      },
    ];
    const events = [
      { type, sequence_number: 1, response },
      {
        type: "response.debug",
        sequence_number: 2,
        debug: {
          timings: { epoch_ms: 10, event: "adapter_request", start_ms: 2 },
          echo_upstream_body: { prompt: "Sensitive prompt" },
        },
      },
    ];
    const fake = createFakeFetcher(sseResponse(events));

    const stream = await clientWith(fake.fetcher).responses.send({
      responsesRequest: { model: "openai/gpt-4o", input: "Cancel", stream: true },
    });
    if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
    const reader = stream.getReader();
    for (const _event of events) expect((await reader.read()).done).toBe(false);
    await reader.cancel("caller stopped");

    const span = await exportedSpan(spans);
    expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
      ...(response.output as unknown[]),
      {
        type: "telemetry.dev.response_stream_event",
        event_type: "response.debug",
        payload: {
          type: "response.debug",
          sequence_number: 2,
          debug: { timings: { epoch_ms: 10, event: "adapter_request", start_ms: 2 } },
        },
      },
    ]);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  },
);

test("terminal Responses output retains previously consumed provider events", async () => {
  const spans = setupSpans();
  const terminal = responsesBody("resp_provider_terminal");
  terminal.output = [
    {
      id: "msg_terminal_provider",
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: "Terminal output", annotations: [] }],
    },
  ];
  const providerEvent = {
    type: "response.image_generation_call.completed",
    sequence_number: 1,
    item_id: "image_terminal",
    output_index: 1,
  };
  const events = [
    providerEvent,
    { type: "response.completed", sequence_number: 2, response: terminal },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Complete", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    ...(terminal.output as unknown[]),
    {
      type: "telemetry.dev.response_stream_event",
      event_type: providerEvent.type,
      payload: providerEvent,
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("terminal Responses output preserves provider event truncation", async () => {
  const spans = setupSpans();
  const terminal = responsesBody("resp_provider_truncated");
  terminal.output = [
    {
      id: "msg_terminal_after_provider_truncation",
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: "Terminal output", annotations: [] }],
    },
  ];
  const events = [
    {
      type: "response.image_generation_call.partial_image",
      sequence_number: 1,
      item_id: "image_oversized",
      output_index: 1,
      partial_image_index: 0,
      partial_image_b64: "x".repeat(70 * 1024),
    },
    { type: "response.completed", sequence_number: 2, response: terminal },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Complete", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(terminal.output);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("early Responses stream cancellation retains hydrated items, content, annotations, and tool arguments", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.output_item.added",
      sequence_number: 1,
      output_index: 0,
      item: {
        id: "function_1",
        type: "function_call",
        call_id: "call_1",
        name: "weather",
        arguments: "",
        status: "in_progress",
      },
    },
    {
      type: "response.function_call_arguments.delta",
      sequence_number: 2,
      item_id: "function_1",
      output_index: 0,
      delta: '{"city":"Par',
    },
    {
      type: "response.function_call_arguments.done",
      sequence_number: 3,
      item_id: "function_1",
      output_index: 0,
      name: "weather",
      arguments: '{"city":"Paris"}',
    },
    {
      type: "response.output_item.added",
      sequence_number: 4,
      output_index: 1,
      item: {
        id: "message_1",
        type: "message",
        role: "assistant",
        status: "in_progress",
        content: [],
      },
    },
    {
      type: "response.content_part.added",
      sequence_number: 5,
      item_id: "message_1",
      output_index: 1,
      content_index: 0,
      part: { type: "output_text", text: "Forecast", annotations: [] },
    },
    {
      type: "response.output_text.annotation.added",
      sequence_number: 6,
      item_id: "message_1",
      output_index: 1,
      content_index: 0,
      annotation_index: 0,
      annotation: {
        type: "url_citation",
        start_index: 0,
        end_index: 8,
        title: "Forecast",
        url: "https://example.test/forecast",
      },
    },
    {
      type: "response.content_part.done",
      sequence_number: 7,
      item_id: "message_1",
      output_index: 1,
      content_index: 0,
      part: {
        type: "output_text",
        text: "Forecast ready",
        annotations: [
          {
            type: "url_citation",
            start_index: 0,
            end_index: 8,
            title: "Forecast",
            url: "https://example.test/forecast",
          },
        ],
      },
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Weather", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  for (const _event of events) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  const output = jsonAttr<JsonRecord[]>(await exportedSpan(spans), "gen_ai.output.messages");
  expect(output).toEqual([
    {
      id: "function_1",
      type: "function_call",
      call_id: "call_1",
      name: "weather",
      arguments: '{"city":"Paris"}',
      status: "completed",
    },
    {
      id: "message_1",
      type: "message",
      role: "assistant",
      status: "in_progress",
      content: [
        {
          type: "output_text",
          text: "Forecast ready",
          annotations: [
            {
              type: "url_citation",
              start_index: 0,
              end_index: 8,
              title: "Forecast",
              url: "https://example.test/forecast",
            },
          ],
        },
      ],
    },
  ]);
});

test("Responses stream capture rejects sparse annotation indexes", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.content_part.added",
      sequence_number: 1,
      item_id: "message_sparse_annotation",
      output_index: 0,
      content_index: 0,
      part: { type: "output_text", text: "Retained", annotations: [] },
    },
    {
      type: "response.output_text.annotation.added",
      sequence_number: 2,
      item_id: "message_sparse_annotation",
      output_index: 0,
      content_index: 0,
      annotation_index: 1_000_000_000,
      annotation: {
        type: "url_citation",
        start_index: 0,
        end_index: 8,
        title: "Unsafe",
        url: "https://example.test/unsafe",
      },
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Annotate", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  for (const _event of events) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      id: "message_sparse_annotation",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [{ type: "output_text", text: "Retained", annotations: [] }],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("Responses stream cancellation retains custom tools and bounded provider event payloads", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.output_item.added",
      sequence_number: 1,
      output_index: 0,
      item: {
        id: "custom_1",
        type: "custom_tool_call",
        call_id: "call_custom",
        name: "patch",
        input: "",
      },
    },
    {
      type: "response.custom_tool_call_input.delta",
      sequence_number: 2,
      item_id: "custom_1",
      output_index: 0,
      delta: "*** Begin",
    },
    {
      type: "response.custom_tool_call_input.done",
      sequence_number: 3,
      item_id: "custom_1",
      output_index: 0,
      input: "*** Begin Patch",
    },
    {
      type: "response.image_generation_call.partial_image",
      sequence_number: 4,
      item_id: "image_1",
      output_index: 1,
      partial_image_index: 0,
      partial_image_b64: "aW1hZ2U=",
    },
    {
      type: "response.apply_patch_call_operation_diff.delta",
      sequence_number: 5,
      item_id: "patch_1",
      output_index: 2,
      delta: "*** Begin",
    },
    {
      type: "response.apply_patch_call_operation_diff.done",
      sequence_number: 6,
      item_id: "patch_1",
      output_index: 2,
      diff: "*** Begin Patch",
    },
    {
      type: "response.fusion_call.panel.added",
      sequence_number: 7,
      item_id: "fusion_1",
      output_index: 3,
      model: "openai/gpt-4o",
    },
    {
      type: "response.fusion_call.panel.delta",
      sequence_number: 8,
      item_id: "fusion_1",
      output_index: 3,
      model: "openai/gpt-4o",
      delta: "Panel",
    },
    {
      type: "response.fusion_call.panel.reasoning.delta",
      sequence_number: 9,
      item_id: "fusion_1",
      output_index: 3,
      model: "openai/gpt-4o",
      delta: "Reasoning",
    },
    {
      type: "response.fusion_call.panel.completed",
      sequence_number: 10,
      item_id: "fusion_1",
      output_index: 3,
      model: "openai/gpt-4o",
      content: "Panel complete",
    },
    {
      type: "response.fusion_call.panel.failed",
      sequence_number: 11,
      item_id: "fusion_2",
      output_index: 3,
      model: "anthropic/claude-sonnet-4",
      error: "provider failed",
      status_code: 502,
    },
    {
      type: "response.reasoning_summary_part.added",
      sequence_number: 12,
      item_id: "reason_1",
      output_index: 4,
      summary_index: 0,
      part: { type: "summary_text", text: "Initial summary" },
    },
    {
      type: "response.reasoning_summary_part.done",
      sequence_number: 13,
      item_id: "reason_1",
      output_index: 4,
      summary_index: 0,
      part: { type: "summary_text", text: "Final summary" },
    },
    {
      type: "response.image_generation_call.in_progress",
      sequence_number: 14,
      item_id: "image_1",
      output_index: 1,
    },
    {
      type: "response.image_generation_call.generating",
      sequence_number: 15,
      item_id: "image_1",
      output_index: 1,
    },
    {
      type: "response.image_generation_call.completed",
      sequence_number: 16,
      item_id: "image_1",
      output_index: 1,
    },
    {
      type: "response.fusion_call.in_progress",
      sequence_number: 17,
      item_id: "fusion_1",
      output_index: 3,
    },
    {
      type: "response.fusion_call.analysis.in_progress",
      sequence_number: 18,
      item_id: "fusion_1",
      output_index: 3,
      judge_model: "openai/gpt-4o",
    },
    {
      type: "response.fusion_call.analysis.completed",
      sequence_number: 19,
      item_id: "fusion_1",
      output_index: 3,
      analysis: {
        blind_spots: [],
        consensus: ["agreed"],
        contradictions: [],
        partial_coverage: [],
        unique_insights: [],
      },
    },
    {
      type: "response.fusion_call.completed",
      sequence_number: 20,
      item_id: "fusion_1",
      output_index: 3,
    },
    {
      type: "response.web_search_call.in_progress",
      sequence_number: 21,
      item_id: "search_1",
      output_index: 5,
    },
    {
      type: "response.web_search_call.searching",
      sequence_number: 22,
      item_id: "search_1",
      output_index: 5,
    },
    {
      type: "response.web_search_call.completed",
      sequence_number: 23,
      item_id: "search_1",
      output_index: 5,
    },
    {
      type: "response.debug",
      sequence_number: 24,
      debug: {
        timings: { epoch_ms: 10, event: "adapter_request", start_ms: 2 },
        echo_upstream_body: { prompt: "Sensitive prompt" },
      },
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Tools", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  for (const _event of events) expect((await reader.read()).done).toBe(false);
  await reader.cancel("caller stopped");

  const output = jsonAttr<JsonRecord[]>(await exportedSpan(spans), "gen_ai.output.messages");
  expect(output[0]).toMatchObject({
    id: "custom_1",
    type: "custom_tool_call",
    input: "*** Begin Patch",
  });
  expect(output[1]).toMatchObject({
    id: "reason_1",
    type: "reasoning",
    summary: [{ type: "summary_text", text: "Final summary" }],
  });
  const payloadEvents = output.slice(2);
  expect(payloadEvents.map((event) => event.event_type)).toEqual([
    "response.image_generation_call.partial_image",
    "response.apply_patch_call_operation_diff.delta",
    "response.apply_patch_call_operation_diff.done",
    "response.fusion_call.panel.added",
    "response.fusion_call.panel.delta",
    "response.fusion_call.panel.reasoning.delta",
    "response.fusion_call.panel.completed",
    "response.fusion_call.panel.failed",
    "response.image_generation_call.in_progress",
    "response.image_generation_call.generating",
    "response.image_generation_call.completed",
    "response.fusion_call.in_progress",
    "response.fusion_call.analysis.in_progress",
    "response.fusion_call.analysis.completed",
    "response.fusion_call.completed",
    "response.web_search_call.in_progress",
    "response.web_search_call.searching",
    "response.web_search_call.completed",
    "response.debug",
  ]);
  expect(payloadEvents[0]?.payload).toMatchObject({
    partial_image_b64: "aW1hZ2U=",
    partial_image_index: 0,
  });
  expect(payloadEvents.at(-1)?.payload).toEqual({
    type: "response.debug",
    sequence_number: 24,
    debug: { timings: { epoch_ms: 10, event: "adapter_request", start_ms: 2 } },
  });
});

test("Responses streams map completed and failed terminal events", async () => {
  const spans = setupSpans();
  const completed = responsesBody("resp_completed", "completed");
  const failed = responsesBody("resp_failed", "failed");
  const fake = createFakeFetcher(
    sseResponse([{ type: "response.completed", sequence_number: 1, response: completed }]),
    sseResponse([{ type: "response.failed", sequence_number: 1, response: failed }]),
  );
  const client = clientWith(fake.fetcher);

  const completedStream = await client.responses.send({
    responsesRequest: {
      model: "openai/gpt-4o",
      input: "Complete",
      instructions: "Be terse",
      temperature: 0.4,
      topP: 0.8,
      maxOutputTokens: 32,
      stream: true,
    },
  });
  if (!(completedStream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(completedStream);

  const failedStream = await client.responses.send({
    responsesRequest: {
      model: "openai/gpt-4o",
      input: "Fail",
      stream: true,
    },
  });
  if (!(failedStream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(failedStream);

  const finished = await finishedSpans(spans, 2);
  const completedSpan = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "resp_completed",
  );
  const failedSpan = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "resp_failed",
  );
  expect(completedSpan?.name).toBe("chat openai/gpt-4o");
  expect(completedSpan?.status.code).toBe(SPAN_STATUS_UNSET);
  expect(completedSpan?.attributes["gen_ai.provider.name"]).toBe("openrouter");
  expect(completedSpan?.attributes["gen_ai.system_instructions"]).toBe("Be terse");
  expect(completedSpan?.attributes["gen_ai.request.temperature"]).toBe(0.4);
  expect(completedSpan?.attributes["gen_ai.request.top_p"]).toBe(0.8);
  expect(completedSpan?.attributes["gen_ai.request.max_tokens"]).toBe(32);
  expect(completedSpan?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(completedSpan?.attributes["gen_ai.usage.input_tokens"]).toBe(12);
  expect(completedSpan?.attributes["gen_ai.usage.output_tokens"]).toBe(7);
  expect(completedSpan?.attributes["gen_ai.usage.total_tokens"]).toBe(19);
  expect(completedSpan?.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(4);
  expect(completedSpan?.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(9);
  expect(completedSpan?.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(3);
  expect(completedSpan?.attributes["gen_ai.usage.cost"]).toBe(0.021);
  expect(failedSpan?.status.code).toBe(SPAN_STATUS_ERROR);
  expect(failedSpan?.events[0]?.attributes?.["exception.message"]).toContain("provider failed");
});

test("Responses streams capture SDK unknown-event wrappers", async () => {
  const spans = setupSpans();
  const usage = {
    input_tokens: 3,
    output_tokens: 2,
    total_tokens: 5,
    cost_details: { upstream_inference_cost: 0.004 },
  };
  const fake = createFakeFetcher(
    sseResponse([
      {
        type: "response.output_text.delta",
        sequence_number: 1,
        item_id: "msg_raw",
        output_index: 0,
        content_index: 0,
        delta: "Raw output",
      },
      {
        type: "response.completed",
        sequence_number: 2,
        response: {
          id: "resp_raw_completed",
          model: "openai/gpt-4o-2024-11-20",
          status: "completed",
          usage,
        },
      },
    ]),
    sseResponse([
      {
        type: "response.failed",
        sequence_number: 1,
        response: {
          id: "resp_raw_failed",
          model: "openai/gpt-4o-2024-11-20",
          status: "failed",
          error: { code: "server_error", message: "raw provider failed" },
        },
      },
    ]),
  );
  const client = clientWith(fake.fetcher);

  const completedStream = await client.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Complete", stream: true },
  });
  if (!(completedStream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const completedEvents = await collectStream(completedStream);

  const failedStream = await client.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Fail", stream: true },
  });
  if (!(failedStream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(failedStream);

  expect(completedEvents).toEqual([
    expect.objectContaining({ type: "UNKNOWN", isUnknown: true }),
    expect.objectContaining({ type: "UNKNOWN", isUnknown: true }),
  ]);
  const finished = await finishedSpans(spans, 2);
  const completedSpan = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "resp_raw_completed",
  );
  const failedSpan = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "resp_raw_failed",
  );
  expect(jsonAttr(completedSpan!, "gen_ai.output.messages")).toEqual([
    {
      id: "msg_raw",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [{ type: "output_text", text: "Raw output", annotations: [] }],
    },
  ]);
  expect(completedSpan?.attributes["gen_ai.usage.total_tokens"]).toBe(5);
  expect(completedSpan?.attributes["gen_ai.usage.cost"]).toBe(0.004);
  expect(failedSpan?.status.code).toBe(SPAN_STATUS_ERROR);
  expect(failedSpan?.events[0]?.attributes?.["exception.message"]).toContain("raw provider failed");
});

test("stream spans are parented by the invocation context, not the consumer context", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    sseResponse([chatChunk("stream_parented", { role: "assistant", content: "Parented" })]),
  );
  const client = clientWith(fake.fetcher);

  const invocation = await startActiveSpan("invocation", async (parent) => {
    const stream = await client.chat.send({
      chatRequest: {
        model: "openai/gpt-4o",
        messages: [{ role: "user", content: "Parent" }],
        stream: true,
      },
    });
    return { stream, parentSpanId: parent.spanId };
  });
  const invocationStream = invocation.stream;
  if (!(invocationStream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await startActiveSpan("consumer", async () => collectStream(invocationStream));

  const finished = await finishedSpans(spans, 3);
  const chatSpan = finished.find((span) => span.name === "chat openai/gpt-4o");
  expect(chatSpan?.attributes["gen_ai.response.id"]).toBe("stream_parented");
  expect(chatSpan?.parentSpanContext?.spanId).toBe(invocation.parentSpanId);
});

test("unconsumed streams export no span until the caller cancels", async () => {
  let captureCount = 0;
  const spans = setupSpans((value) => {
    captureCount += 1;
    return value;
  });
  const fake = createFakeFetcher(
    sseResponse([chatChunk("stream_unconsumed", { content: "Not consumed" })]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Wait" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await flush();
  expect(spans.getFinishedSpans()).toHaveLength(0);
  expect(captureCount).toBe(0);

  const reader = stream.getReader();
  await reader.cancel("not consumed");
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(captureCount).toBeGreaterThan(0);
});

test.each([
  ["control characters", "\u0001"],
  ["lone surrogates", "\ud800"],
])("chat stream capture accounts for JSON escape overhead from %s", async (_, character) => {
  const spans = setupSpans();
  const escaped = character.repeat(1024);
  const events = Array.from({ length: 70 }, (_, index) =>
    chatChunk(
      "stream_escaped",
      index === 0 ? { role: "assistant", content: escaped } : { content: escaped },
    ),
  );
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Escapes" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  expect(await collectStream(stream)).toHaveLength(events.length);

  const span = await exportedSpan(spans);
  const serialized = String(span.attributes["gen_ai.output.messages"]);
  expect(serialized).not.toContain("...[truncated]");
  expect(serialized.length).toBeLessThanOrEqual(65536);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  const output = jsonAttr<Array<{ content?: string }>>(span, "gen_ai.output.messages");
  expect(output[0]?.content?.length).toBeGreaterThan(0);
  expect(output[0]?.content).toBe(escaped.repeat(output[0]!.content!.length / escaped.length));
});

test("chat streams bound choice-state creation without dropping chunks", async () => {
  const spans = setupSpans();
  const choices = Array.from({ length: 1200 }, (_, index) => ({
    index,
    delta: index === 0 ? { role: "assistant", content: "Hi" } : {},
    finish_reason: null,
  }));
  const fake = createFakeFetcher(
    sseResponse([
      {
        id: "stream_choices",
        object: "chat.completion.chunk",
        created: 1,
        model: "openai/gpt-4o-2024-11-20",
        choices,
      },
    ]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Many choices" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  expect(await collectStream(stream)).toHaveLength(1);

  const span = await exportedSpan(spans);
  const output = jsonAttr<Array<{ role: string; content?: string }>>(
    span,
    "gen_ai.output.messages",
  );
  expect(output).toHaveLength(1024);
  expect(output[0]).toEqual({ role: "assistant", content: "Hi" });
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("multi-choice chat streams reconstruct every choice", async () => {
  const spans = setupSpans();
  const chunk = (deltas: Array<{ index: number; delta: JsonRecord; finish?: string }>) => ({
    id: "stream_multi",
    object: "chat.completion.chunk",
    created: 1,
    model: "openai/gpt-4o-2024-11-20",
    choices: deltas.map(({ index, delta, finish }) => ({
      index,
      delta,
      finish_reason: finish ?? null,
    })),
  });
  const fake = createFakeFetcher(
    sseResponse([
      chunk([
        { index: 0, delta: { role: "assistant", content: "First" } },
        { index: 1, delta: { role: "assistant", content: "Second" } },
      ]),
      chunk([
        { index: 0, delta: { content: " one" }, finish: "stop" },
        { index: 1, delta: { content: " two" }, finish: "length" },
      ]),
    ]),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Two choices" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "First one" },
    { role: "assistant", content: "Second two" },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop", "length"]);
});

test("non-2xx chat responses end one error span and propagate the failure", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    new Response(JSON.stringify({ error: { code: 500, message: "provider exploded" } }), {
      status: 500,
      headers: { "content-type": "application/json" },
    }),
  );

  await expect(
    clientWith(fake.fetcher).chat.send({
      chatRequest: {
        model: "openai/gpt-4o",
        messages: [{ role: "user", content: "Boom" }],
      },
    }),
  ).rejects.toThrow();

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events[0]?.name).toBe("exception");
});

test("rejected transport requests end one error span and rethrow", async () => {
  const spans = setupSpans();
  const failure = new Error("network down");
  const fetcher: Fetcher = () => Promise.reject(failure);

  await expect(
    clientWith(fetcher).chat.send({
      chatRequest: {
        model: "openai/gpt-4o",
        messages: [{ role: "user", content: "Offline" }],
      },
    }),
  ).rejects.toThrow("network down");

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events[0]?.attributes?.["exception.message"]).toContain("network down");
});

test("mid-stream body failures end one error span with partial output", async () => {
  const spans = setupSpans();
  const encoder = new TextEncoder();
  let delivered = false;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (delivered) throw new Error("connection reset");
      delivered = true;
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify(chatChunk("stream_broken", { role: "assistant", content: "Partial" }))}\n\n`,
        ),
      );
    },
  });
  const fake = createFakeFetcher(
    new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
  );

  const stream = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Break" }],
      stream: true,
    },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await expect(collectStream(stream)).rejects.toThrow("connection reset");

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events[0]?.attributes?.["exception.message"]).toContain("connection reset");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Partial" },
  ]);
});

test("Responses stream error events record numeric codes and end once", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    openSseResponse([
      {
        type: "response.output_text.delta",
        sequence_number: 1,
        item_id: "msg_error",
        output_index: 0,
        content_index: 0,
        delta: "Partial",
      },
      { type: "error", sequence_number: 2, code: 502, message: "provider disconnected" },
    ]),
  );

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Fail mid-stream", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const reader = stream.getReader();
  expect((await reader.read()).done).toBe(false);
  expect((await reader.read()).done).toBe(false);

  const span = await exportedSpan(spans);
  await reader.cancel();
  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events[0]?.attributes?.["exception.message"]).toBe(
    "response.error: 502: provider disconnected",
  );
});

test("final Responses text replaces delta-truncated output using a fresh budget", async () => {
  const spans = setupSpans();
  const delta = "y".repeat(1024);
  const deltas = Array.from({ length: 70 }, (_, sequenceNumber) => ({
    type: "response.output_text.delta",
    sequence_number: sequenceNumber,
    item_id: "msg_done",
    output_index: 0,
    content_index: 0,
    delta,
    logprobs: [],
  }));
  const events = [
    ...deltas,
    {
      type: "response.output_text.done",
      sequence_number: deltas.length,
      item_id: "msg_done",
      output_index: 0,
      content_index: 0,
      text: "Final answer",
      logprobs: [],
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Long answer", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      id: "msg_done",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [{ type: "output_text", text: "Final answer", annotations: [] }],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("final Responses text preserves provider event truncation", async () => {
  const spans = setupSpans();
  const events = [
    {
      type: "response.image_generation_call.partial_image",
      sequence_number: 1,
      item_id: "image_oversized_before_done",
      output_index: 1,
      partial_image_index: 0,
      partial_image_b64: "x".repeat(70 * 1024),
    },
    {
      type: "response.output_text.done",
      sequence_number: 2,
      item_id: "msg_done_after_provider_truncation",
      output_index: 0,
      content_index: 0,
      text: "Final answer",
      logprobs: [],
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Long answer", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      id: "msg_done_after_provider_truncation",
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [{ type: "output_text", text: "Final answer", annotations: [] }],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("oversized final Responses text preserves bounded partial output", async () => {
  const spans = setupSpans();
  const delta = "y".repeat(1024);
  const deltas = Array.from({ length: 20 }, (_, sequenceNumber) => ({
    type: "response.output_text.delta",
    sequence_number: sequenceNumber,
    item_id: "msg_done_oversized",
    output_index: 0,
    content_index: 0,
    delta,
    logprobs: [],
  }));
  const events = [
    ...deltas,
    {
      type: "response.output_text.done",
      sequence_number: deltas.length,
      item_id: "msg_done_oversized",
      output_index: 0,
      content_index: 0,
      text: "z".repeat(70 * 1024),
      logprobs: [],
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Long answer", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  const output = jsonAttr<Array<{ content: Array<{ text: string }> }>>(
    span,
    "gen_ai.output.messages",
  );
  expect(output[0]?.content[0]?.text).toBe(delta.repeat(deltas.length));
  expect(output[0]?.content[0]?.text).not.toContain("z");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("small terminal Responses output replaces delta-truncated output", async () => {
  const spans = setupSpans();
  const delta = "y".repeat(1024);
  const deltas = Array.from({ length: 70 }, (_, sequenceNumber) => ({
    type: "response.output_text.delta",
    sequence_number: sequenceNumber,
    item_id: "msg_bounded",
    output_index: 0,
    content_index: 0,
    delta,
    logprobs: [],
  }));
  const events = [
    ...deltas,
    {
      type: "response.completed",
      sequence_number: deltas.length,
      response: responsesBody("resp_bounded"),
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Long answer", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  const received = await collectStream(stream);

  expect(received).toHaveLength(events.length);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_bounded");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(19);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.021);
});

test("oversized terminal Responses output preserves bounded partial output", async () => {
  const spans = setupSpans();
  const delta = "y".repeat(1024);
  const deltas = Array.from({ length: 20 }, (_, sequenceNumber) => ({
    type: "response.output_text.delta",
    sequence_number: sequenceNumber,
    item_id: "msg_oversized",
    output_index: 0,
    content_index: 0,
    delta,
    logprobs: [],
  }));
  const terminal = responsesBody("resp_oversized");
  terminal.output = [
    {
      id: "msg_terminal",
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: "z".repeat(70 * 1024), annotations: [] }],
    },
  ];
  const events = [
    ...deltas,
    {
      type: "response.completed",
      sequence_number: deltas.length,
      response: terminal,
    },
  ];
  const fake = createFakeFetcher(sseResponse(events));

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Long answer", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  const output = jsonAttr<Array<{ content: Array<{ text: string }> }>>(
    span,
    "gen_ai.output.messages",
  );
  expect(output[0]?.content[0]?.text).toBe(delta.repeat(deltas.length));
  expect(output[0]?.content[0]?.text).not.toContain("z");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_oversized");
});

test("non-streaming failed Responses record a useful error", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(jsonResponse(responsesBody("resp_failed", "failed")));

  await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Fail" },
  });

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("Error");
  expect(span.events[0]?.attributes?.["exception.message"]).toContain(
    "response.failed: server_error: provider failed",
  );
});

test("failed Responses stream events keep numeric provider error codes", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(
    sseResponse([
      {
        type: "response.failed",
        sequence_number: 1,
        response: {
          id: "resp_failed_numeric",
          model: "openai/gpt-4o-2024-11-20",
          status: "failed",
          error: { code: 502, message: "provider failed" },
        },
      },
    ]),
  );

  const stream = await clientWith(fake.fetcher).responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Fail", stream: true },
  });
  if (!(stream instanceof ReadableStream)) throw new Error("expected a readable stream");
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events[0]?.attributes?.["exception.message"]).toBe(
    "response.failed: 502: provider failed",
  );
});

test("embeddings map input, model, usage, and cost without capturing vectors", async () => {
  const spans = setupSpans();
  const fake = createFakeFetcher(jsonResponse(embeddingsBody("embed_1")));
  const input = ["first", "second"];

  await clientWith(fake.fetcher).embeddings.generate({
    requestBody: { model: "openai/text-embedding-3-small", input },
  });

  const span = await exportedSpan(spans);
  expect(span.name).toBe("embeddings openai/text-embedding-3-small");
  expect(span.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(span.attributes["gen_ai.provider.name"]).toBe("openrouter");
  expect(span.attributes["gen_ai.request.model"]).toBe("openai/text-embedding-3-small");
  expect(span.attributes["gen_ai.response.model"]).toBe("openai/text-embedding-3-small-2024");
  expect(span.attributes["gen_ai.response.id"]).toBe("embed_1");
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(input));
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(8);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(8);
  expect(span.attributes["gen_ai.usage.cost"]).toBe(0.0004);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(JSON.stringify(span.attributes)).not.toContain("0.1");
});

test("wrap and global instrumentation have parity, are idempotent, and restore prototypes", async () => {
  const spans = setupSpans();
  const originalChat = prototypeMethod(Chat.prototype, "send");
  const originalResponses = prototypeMethod(Responses.prototype, "send");
  const originalEmbeddings = prototypeMethod(Embeddings.prototype, "generate");
  instrumentOpenRouter();
  instrumentOpenRouter();
  expect(prototypeMethod(Chat.prototype, "send")).not.toBe(originalChat);
  expect(prototypeMethod(Responses.prototype, "send")).not.toBe(originalResponses);
  expect(prototypeMethod(Embeddings.prototype, "generate")).not.toBe(originalEmbeddings);

  const globalFake = createFakeFetcher(
    jsonResponse(chatBody("global_chat", { cost: 0.012 })),
    jsonResponse(responsesBody("global_response")),
    jsonResponse(embeddingsBody("global_embedding")),
  );
  const globalClient = clientWith(globalFake.fetcher, false);
  await globalClient.chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Global" }],
    },
  });
  await globalClient.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Global" },
  });
  await globalClient.embeddings.generate({
    requestBody: { model: "openai/text-embedding-3-small", input: "Global" },
  });

  const wrappedFake = createFakeFetcher(
    jsonResponse(chatBody("wrapped_chat", { cost: 0.012 })),
    jsonResponse(responsesBody("wrapped_response")),
    jsonResponse(embeddingsBody("wrapped_embedding")),
    jsonResponse(chatBody("wrapped_after_restore", { cost: 0.012 })),
  );
  const wrappedClient = wrapOpenRouter(wrapOpenRouter(clientWith(wrappedFake.fetcher, false)));
  await wrappedClient.chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Wrapped" }],
    },
  });
  await wrappedClient.responses.send({
    responsesRequest: { model: "openai/gpt-4o", input: "Wrapped" },
  });
  await wrappedClient.embeddings.generate({
    requestBody: { model: "openai/text-embedding-3-small", input: "Wrapped" },
  });

  const initial = await finishedSpans(spans, 6);
  for (const id of [
    "global_chat",
    "global_response",
    "global_embedding",
    "wrapped_chat",
    "wrapped_response",
    "wrapped_embedding",
  ]) {
    const span = initial.find((candidate) => candidate.attributes["gen_ai.response.id"] === id);
    expect(span?.attributes["gen_ai.provider.name"]).toBe("openrouter");
  }

  uninstrumentOpenRouter();
  expect(prototypeMethod(Chat.prototype, "send")).toBe(originalChat);
  expect(prototypeMethod(Responses.prototype, "send")).toBe(originalResponses);
  expect(prototypeMethod(Embeddings.prototype, "generate")).toBe(originalEmbeddings);

  await wrappedClient.chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Still wrapped" }],
    },
  });
  await finishedSpans(spans, 7);

  const restoredFake = createFakeFetcher(jsonResponse(chatBody("fresh_restored")));
  await clientWith(restoredFake.fetcher, false).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "Fresh" }],
    },
  });
  await flush();
  expect(spans.getFinishedSpans()).toHaveLength(7);
});

test("global uninstrumentation preserves later prototype patches", () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(Chat.prototype, "send");
  if (!originalDescriptor) throw new Error("Chat.prototype.send is missing");
  instrumentOpenRouter();
  const laterPatch = () => undefined;
  Object.defineProperty(Chat.prototype, "send", {
    ...originalDescriptor,
    value: laterPatch,
  });

  try {
    uninstrumentOpenRouter();
    expect(prototypeMethod(Chat.prototype, "send")).toBe(laterPatch);
  } finally {
    Object.defineProperty(Chat.prototype, "send", originalDescriptor);
  }
});

test("wrapped clients fail open when telemetry is not initialized", async () => {
  await shutdown();
  const fake = createFakeFetcher(
    jsonResponse(chatBody("no_telemetry", { content: "Still works" })),
  );

  const response = await clientWith(fake.fetcher).chat.send({
    chatRequest: {
      model: "openai/gpt-4o",
      messages: [{ role: "user", content: "No telemetry" }],
    },
  });

  expect(fake.requests).toHaveLength(1);
  expect(response).toMatchObject({ id: "no_telemetry" });
});
