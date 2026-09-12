import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { flush, init, shutdown } from "@telemetry-dev/sdk";
import Anthropic from "@anthropic-ai/sdk";
import { afterEach, expect, test, vi } from "vitest";

import { instrumentAnthropic, uninstrumentAnthropic, wrapAnthropic } from "../src/index.ts";

const SPAN_STATUS_UNSET = 0;
const SPAN_STATUS_ERROR = 2;

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

interface JsonRecord {
  [key: string]: JsonValue;
}

interface CapturedRequest {
  method: string | undefined;
  path: string;
  body: JsonValue | undefined;
}

function asRecord<T>(value: T): (T & JsonRecord) | undefined {
  if (value === null || value === undefined || Array.isArray(value)) return undefined;

  return Object(value) === value ? (value as T & JsonRecord) : undefined;
}

function readString<T>(value: T): string | undefined {
  return String(value) === value ? (value as string) : undefined;
}

function jsonResponse(body: JsonValue): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json", "request-id": "req_test" },
  });
}

function jsonErrorResponse(status: number, message: string): Response {
  return new Response(
    JSON.stringify({ type: "error", error: { type: "invalid_request_error", message } }),
    {
      status,
      headers: { "content-type": "application/json", "request-id": "req_error" },
    },
  );
}

function namedSseResponse(events: JsonRecord[]): Response {
  const body = events
    .map(
      (event) =>
        `event: ${typeof event.type === "string" ? event.type : ""}\ndata: ${JSON.stringify(event)}\n\n`,
    )
    .join("");

  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream", "request-id": "req_stream" },
  });
}

interface CountedResponse {
  response: Response;
  pulledChunks: () => number;
  chunkCount: number;
}

function countedSseResponse(events: JsonRecord[]): CountedResponse {
  const encoder = new TextEncoder();

  const parts = events.map(
    (event) =>
      `event: ${typeof event.type === "string" ? event.type : ""}\ndata: ${JSON.stringify(event)}\n\n`,
  );

  let index = 0;
  let pulled = 0;

  const body = new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        const part = parts[index];

        if (part === undefined) {
          controller.close();

          return;
        }

        index += 1;
        pulled += 1;
        controller.enqueue(encoder.encode(part));
      },
    },
    { highWaterMark: 0 },
  );

  return {
    response: new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream", "request-id": "req_stream_counted" },
    }),
    pulledChunks: () => pulled,
    chunkCount: parts.length,
  };
}

async function parseRequestBody(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): Promise<JsonValue | undefined> {
  const initBody = init?.body;

  if (typeof initBody === "string") return JSON.parse(initBody);

  if (input instanceof Request) {
    const text = await input.clone().text();

    return text ? JSON.parse(text) : undefined;
  }

  return undefined;
}

interface FakeFetch {
  fetch: typeof fetch;
  requests: CapturedRequest[];
}

function createFakeFetch(...responses: Response[]): FakeFetch {
  const requests: CapturedRequest[] = [];

  const fetchImpl: typeof fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : String(input);
    requests.push({
      method: init?.method ?? (input instanceof Request ? input.method : undefined),
      path: new URL(url).pathname,
      body: await parseRequestBody(input, init),
    });
    const response = responses.shift();

    if (!response) throw new Error(`unexpected request to ${url}`);

    return response;
  };

  return { fetch: fetchImpl, requests };
}

function setupSpans(): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "anthropic-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter },
  );

  return spanExporter;
}

function clientWith(fetchImpl: typeof fetch): Anthropic {
  return wrapAnthropic(new Anthropic({ apiKey: "test", fetch: fetchImpl, maxRetries: 0 }));
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
  expect(String(value) === value).toBe(true);

  return JSON.parse(String(value)) as T;
}

async function collectStream(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const chunks: unknown[] = [];

  for await (const chunk of stream) chunks.push(chunk);

  return chunks;
}

function messagePayload(overrides: JsonRecord = {}) {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-6",
    content: [{ type: "text", text: "Hello." }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 2,
      cache_read_input_tokens: 3,
      output_tokens_details: { thinking_tokens: 1 },
    },
    ...overrides,
  };
}

function streamEvents(): JsonRecord[] {
  return [
    {
      type: "message_start",
      message: messagePayload({
        id: "msg_stream",
        content: [],
        usage: { input_tokens: 9, output_tokens: 0 },
      }),
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello " } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "world" } },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: 2 },
    },
    { type: "message_stop" },
  ];
}

afterEach(async () => {
  uninstrumentAnthropic();
  await shutdown();
  vi.restoreAllMocks();
});

test("messages.create maps request, response, usage, finish reason, provider, and sampling attributes", async () => {
  const spans = setupSpans();
  const messages = [{ role: "user" as const, content: "Say hello" }];
  const fake = createFakeFetch(jsonResponse(messagePayload()));
  const client = clientWith(fake.fetch);

  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages,
    system: "Be terse",
    temperature: 0.7,
    top_p: 0.9,
    top_k: 40,
    stop_sequences: ["END"],
  });

  const span = await exportedSpan(spans);
  expect(fake.requests).toHaveLength(1);
  expect(fake.requests[0]?.path).toBe("/v1/messages");
  expect(span.name).toBe("chat claude-sonnet-4-6");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("anthropic");
  expect(span.attributes["gen_ai.request.model"]).toBe("claude-sonnet-4-6");
  expect(span.attributes["gen_ai.response.model"]).toBe("claude-sonnet-4-6");
  expect(span.attributes["gen_ai.response.id"]).toBe("msg_1");
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(messages));
  expect(span.attributes["gen_ai.system_instructions"]).toBe("Be terse");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello." }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(1);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.top_k"]).toBe(40);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(64);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["END"]);
});

test("messages.create preserves the Anthropic promise API", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(jsonResponse(messagePayload({ id: "msg_promise" })));
  const client = clientWith(fake.fetch);

  const result = client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
  });

  expect(result).toBeInstanceOf(Promise);
  const { data, response, request_id } = await result.withResponse();

  expect(data.id).toBe("msg_promise");
  expect(response.status).toBe(200);
  expect(request_id).toBe("req_test");
  await finishedSpans(spans, 1);
});

test("messages.create records one error span when the Anthropic API returns 4xx", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(jsonErrorResponse(400, "bad model"));

  await expect(
    clientWith(fake.fetch).messages.create({
      model: "claude-bad",
      max_tokens: 64,
      messages: [{ role: "user", content: "Fail" }],
    }),
  ).rejects.toMatchObject({ status: 400 });

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.request.model"]).toBe("claude-bad");
  expect(String(span.attributes["error.type"]) === span.attributes["error.type"]).toBe(true);
  const exception = span.events.find((event) => event.name === "exception");
  expect(
    String(exception?.attributes?.["exception.type"]) === exception?.attributes?.["exception.type"],
  ).toBe(true);
  expect(String(exception?.attributes?.["exception.message"])).toContain("bad model");
});

test("messages.create preserves request tools and tool-use blocks in output", async () => {
  const spans = setupSpans();

  const toolUse = {
    type: "tool_use",
    id: "toolu_1",
    name: "get_weather",
    input: { location: "Paris" },
  };

  const tool = {
    name: "get_weather",
    description: "Get weather",
    input_schema: { type: "object", properties: { location: { type: "string" } } },
  } as const;

  const fake = createFakeFetch(
    jsonResponse(
      messagePayload({
        id: "msg_tool",
        content: [toolUse],
        stop_reason: "tool_use",
        usage: { input_tokens: 12, output_tokens: 4 },
      }),
    ),
  );

  await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Weather in Paris?" }],
    tools: [tool],
  });

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [toolUse] },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_use"]);
  expect(jsonAttr(span, "gen_ai.input.messages")).toEqual({
    messages: [{ role: "user", content: "Weather in Paris?" }],
    tools: [tool],
  });
});

test("messages.create streaming preserves events and records aggregated text usage and time to first chunk", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(namedSseResponse(streamEvents()));

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  const visibleEvents = await collectStream(stream);

  expect(fake.requests[0]?.body).toMatchObject({ stream: true });
  expect(visibleEvents.map((event) => readString(asRecord(event)?.type))).toEqual(
    streamEvents().map((event) => event.type),
  );
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);
  expect(span.attributes["gen_ai.response.id"]).toBe("msg_stream");
  expect(span.attributes["gen_ai.response.model"]).toBe("claude-sonnet-4-6");
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"],
  ).toBe(true);
});

test("messages.create streaming observes duck-typed stream responses", async () => {
  const spans = setupSpans();

  class AlternateStream implements AsyncIterable<unknown> {
    private abortController = new AbortController();
    controller = {
      abort: (reason?: unknown) => this.abortController.abort(reason),
      signal: this.abortController.signal,
    };

    async *[Symbol.asyncIterator](): AsyncIterator<unknown> {
      for (const event of streamEvents()) yield event;
    }
  }

  const client = wrapAnthropic({
    messages: {
      create: (_params: JsonValue) => Promise.resolve(new AlternateStream()),
    },
  });

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
});

test("message timestamps precede telemetry mapping", async () => {
  const spans = setupSpans();
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => now);

  const source = {
    controller: new AbortController(),
    async *[Symbol.asyncIterator]() {
      for (const [receivedAt, mappingMs] of [
        [100, 30],
        [240, 90],
      ] as const) {
        now = receivedAt;
        yield {
          type: "content_block_delta",
          index: 0,
          get delta() {
            now = receivedAt + mappingMs;

            return { type: "text_delta", text: "A" };
          },
        };
      }
    },
  };

  const client = wrapAnthropic({
    messages: { create: async (_params: unknown) => source },
  });

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
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

test("messages.create streaming preserves citation deltas", async () => {
  const spans = setupSpans();

  const citation = {
    type: "char_location",
    cited_text: "quoted text",
    document_index: 0,
    document_title: "Source",
    start_char_index: 0,
    end_char_index: 11,
  };

  const fake = createFakeFetch(
    namedSseResponse([
      { type: "message_start", message: messagePayload({ id: "msg_cited", content: [] }) },
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
      { type: "content_block_delta", index: 0, delta: { type: "citations_delta", citation } },
      { type: "content_block_stop", index: 0 },
      { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
      { type: "message_stop" },
    ]),
  );

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Cite this" }],
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello", citations: [citation] }] },
  ]);
});

test("messages.create streaming aggregates tool-use input JSON fragments", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    namedSseResponse([
      { type: "message_start", message: messagePayload({ id: "msg_tool_stream", content: [] }) },
      {
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "tool_use",
          id: "toolu_1",
          name: "get_weather",
          input: { unit: "celsius" },
          caller: { type: "direct" },
        },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: '{"loc' },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "input_json_delta", partial_json: 'ation":"Paris"}' },
      },
      { type: "content_block_stop", index: 0 },
      {
        type: "content_block_start",
        index: 1,
        content_block: { type: "tool_use", id: "toolu_2", name: "empty_tool" },
      },
      { type: "content_block_stop", index: 1 },
      { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 6 } },
      { type: "message_stop" },
    ]),
  );

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Weather in Paris?" }],
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_1",
          name: "get_weather",
          input: { unit: "celsius", location: "Paris" },
          caller: { type: "direct" },
        },
        { type: "tool_use", id: "toolu_2", name: "empty_tool", input: {} },
      ],
    },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_use"]);
});

test("messages.create streaming aggregates thinking and signature deltas", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    namedSseResponse([
      { type: "message_start", message: messagePayload({ id: "msg_thinking", content: [] }) },
      { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "I should answer." },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "signature_delta", signature: "sig" },
      },
      { type: "content_block_stop", index: 0 },
      { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } },
      { type: "message_stop" },
    ]),
  );

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Think" }],
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [{ type: "thinking", thinking: "I should answer.", signature: "sig" }],
    },
  ]);
});

test("messages.create streaming ends once with partial output when the caller stops early", async () => {
  const spans = setupSpans();
  const counted = countedSseResponse(streamEvents());
  const fake = createFakeFetch(counted.response);

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  for await (const event of stream) {
    if (readString(asRecord(event)?.type) === "content_block_delta") break;
  }

  expect(counted.pulledChunks()).toBeLessThan(counted.chunkCount);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello " }] },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"],
  ).toBe(true);
});

test("messages.create streaming ends with error when aborted before iteration starts", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(namedSseResponse(streamEvents()));

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  stream.controller.abort();

  const span = await exportedSpan(spans);
  expect(fake.requests[0]?.body).toMatchObject({ stream: true });
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["gen_ai.request.model"]).toBe("claude-sonnet-4-6");
});

test("messages.stream helper routes through create and records one span", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(namedSseResponse(streamEvents()));
  const client = clientWith(fake.fetch);

  const stream = client.messages.stream({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
  });

  const events = await collectStream(stream);
  const finalMessage = await stream.finalMessage();

  expect(events.length).toBeGreaterThan(0);
  expect(finalMessage.id).toBe("msg_stream");
  expect(fake.requests[0]?.body).toMatchObject({ stream: true });
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
});

test("instrumentAnthropic and wrapAnthropic are idempotent and uninstrumentAnthropic restores create", async () => {
  const spans = setupSpans();
  instrumentAnthropic();
  instrumentAnthropic();
  const instrumented = createFakeFetch(jsonResponse(messagePayload({ id: "msg_global" })));

  await new Anthropic({ apiKey: "test", fetch: instrumented.fetch, maxRetries: 0 }).messages.create(
    {
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      messages: [{ role: "user", content: "Hi" }],
    },
  );
  await finishedSpans(spans, 1);

  uninstrumentAnthropic();
  const wrapped = createFakeFetch(jsonResponse(messagePayload({ id: "msg_wrapped" })));
  await wrapAnthropic(
    wrapAnthropic(new Anthropic({ apiKey: "test", fetch: wrapped.fetch, maxRetries: 0 })),
  ).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi wrapped" }],
  });
  await finishedSpans(spans, 2);

  const restored = createFakeFetch(jsonResponse(messagePayload({ id: "msg_restored" })));
  await new Anthropic({ apiKey: "test", fetch: restored.fetch, maxRetries: 0 }).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi again" }],
  });
  await flush();

  expect(instrumented.requests).toHaveLength(1);
  expect(wrapped.requests).toHaveLength(1);
  expect(restored.requests).toHaveLength(1);
  expect(spans.getFinishedSpans()).toHaveLength(2);
});

test("wrapAnthropic keeps a client instrumented after global instrumentation is removed", async () => {
  const spans = setupSpans();
  instrumentAnthropic();
  const fake = createFakeFetch(jsonResponse(messagePayload({ id: "msg_instance_wrapped" })));
  const client = wrapAnthropic(new Anthropic({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }));
  uninstrumentAnthropic();

  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi wrapped" }],
  });

  const span = await exportedSpan(spans);
  expect(fake.requests).toHaveLength(1);
  expect(span.attributes["gen_ai.request.model"]).toBe("claude-sonnet-4-6");
});

test("instrumentAnthropic and wrapAnthropic together record one span per call", async () => {
  const spans = setupSpans();
  instrumentAnthropic();

  const fake = createFakeFetch(
    jsonResponse(messagePayload({ id: "msg_global_wrapped" })),
    jsonResponse(messagePayload({ id: "msg_instance_wrapped_after_global" })),
  );

  const client = wrapAnthropic(new Anthropic({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }));

  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi wrapped" }],
  });

  const activeSpans = await finishedSpans(spans, 1);
  expect(fake.requests).toHaveLength(1);
  expect(activeSpans[0]!.attributes["gen_ai.request.model"]).toBe("claude-sonnet-4-6");

  uninstrumentAnthropic();
  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi after uninstrument" }],
  });

  const allSpans = await finishedSpans(spans, 2);
  expect(fake.requests).toHaveLength(2);
  expect(allSpans[1]!.attributes["gen_ai.request.model"]).toBe("claude-sonnet-4-6");
});

test("wrapAnthropic records Bedrock provider names from client constructor names", async () => {
  const spans = setupSpans();

  class AnthropicBedrock {
    messages = {
      create: async (_params: JsonValue) => messagePayload({ id: "msg_bedrock" }),
    };
  }

  await wrapAnthropic(new AnthropicBedrock()).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi" }],
  });

  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.provider.name"]).toBe("aws.bedrock");
});
