import {
  AggregationTemporality,
  InMemoryMetricExporter,
  type PushMetricExporter,
} from "@opentelemetry/sdk-metrics";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  type ChatMessage,
  flush,
  getPrompt,
  init,
  shutdown,
  type TelemetryOptions,
} from "@telemetry-dev/sdk";
import * as telemetrySdk from "@telemetry-dev/sdk";
import Anthropic from "@anthropic-ai/sdk";
import { createRequire } from "node:module";
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

function setupSpans(
  metricExporter?: PushMetricExporter,
  captureOutput = true,
  mask?: TelemetryOptions["mask"],
  options: Pick<TelemetryOptions, "captureInput" | "maxAttributeLength" | "onError"> = {},
): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "anthropic-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      captureOutput,
      mask,
      ...options,
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter, metricExporter },
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

test.each([
  { messages: "compiled", olderCore: false },
  { messages: "copied", olderCore: false },
  { messages: "compiled", olderCore: true },
])(
  "messages.create outside withPrompt records the prompt of $messages messages, older core $olderCore",
  async ({ messages: kind, olderCore }) => {
    if (olderCore) {
      vi.spyOn(
        telemetrySdk as { promptAttributesFor: unknown },
        "promptAttributesFor",
        "get",
      ).mockReturnValue(undefined);
    }

    const prompt = await getPrompt("support/reply", {
      apiKey: "td_live_a",
      baseUrl: "https://prompts.anthropic.test",
      fetch: async () =>
        Response.json({
          name: "support/reply",
          type: "chat",
          label: "production",
          version: 12,
          messages: [
            { role: "system", content: "You answer for the {{team}} desk." },
            { role: "user", content: "Where is order {{order}}?" },
          ],
          text: null,
          model: { provider: "anthropic", name: "claude-sonnet-4-6" },
          parameters: {},
          tools: [],
          responseFormat: { type: "text" },
          rollout: null,
        }),
    });
    const [system, user] = prompt.compile({ team: "billing", order: "#4" }) as ChatMessage[];
    const message = user as { role: "user"; content: string };
    const spans = setupSpans();
    const client = clientWith(createFakeFetch(jsonResponse(messagePayload())).fetch);

    await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      system: system!.content,
      messages: [kind === "compiled" ? message : { ...message }],
    });

    const { attributes } = await exportedSpan(spans);
    expect(attributes["gen_ai.input.messages"]).toBeDefined();

    if (olderCore || kind === "copied") {
      expect(attributes["td.prompt.name"]).toBeUndefined();

      return;
    }

    expect(attributes).toMatchObject({
      "td.prompt.name": "support/reply",
      "td.prompt.version": 12,
      "td.prompt.label": "production",
      "td.prompt.variables": '{"team":"billing","order":"#4"}',
    });
  },
);

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
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10 + 3 + 2);
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

test("messages.create returns the provider result when response mapping fails", async () => {
  const errors: Error[] = [];

  const spans = setupSpans(undefined, true, undefined, {
    onError: (error) => errors.push(error),
  });

  const mappingError = new Error("mapping failed");

  const source = {
    get role(): never {
      throw mappingError;
    },
  };

  const client = wrapAnthropic({
    messages: { create: async (_params: unknown) => source },
  });

  const result = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
  });

  expect(result).toBe(source);
  expect(errors).toEqual([mappingError]);
  expect((await exportedSpan(spans)).attributes["telemetry.dev.capture.truncated"]).toBe(true);
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

test("messages.parse sends the stable request and records its response", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(jsonResponse(messagePayload({ id: "msg_stable_parse" })));
  const client = clientWith(fake.fetch);

  const parsed = await client.messages.parse({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Parse this" }],
  });

  expect(parsed.id).toBe("msg_stable_parse");
  expect(fake.requests).toEqual([
    expect.objectContaining({
      path: "/v1/messages",
      body: expect.objectContaining({ messages: [{ role: "user", content: "Parse this" }] }),
    }),
  ]);
  expect((await exportedSpan(spans)).attributes["gen_ai.response.id"]).toBe("msg_stable_parse");
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

  const tools = [
    tool,
    { type: "custom", name: "issue_refund", input_schema: { type: "object" } },
    { type: "web_search_20250305", name: "web_search" },
  ] as const;

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
    tools: [...tools],
  });

  const span = await exportedSpan(spans);
  // The dashboard lists these names as the tools offered; schemas stay in the captured input.
  expect(jsonAttr(span, "gen_ai.tool.definitions")).toEqual([
    { type: "function", name: "get_weather" },
    { type: "function", name: "issue_refund" },
    { type: "web_search_20250305", name: "web_search" },
  ]);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [toolUse] },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_use"]);
  expect(jsonAttr(span, "gen_ai.input.messages")).toEqual({
    messages: [{ role: "user", content: "Weather in Paris?" }],
    tools,
  });
});

test("messages.create records offered tool names with capture off", async () => {
  const spans = setupSpans(undefined, false, undefined, { captureInput: false });
  const fake = createFakeFetch(jsonResponse(messagePayload({ id: "msg_private" })));

  await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Refund order 42" }],
    tools: [
      {
        name: "issue_refund",
        description: "Refund an order",
        input_schema: { type: "object", properties: { order: { type: "string" } } },
      },
    ],
  });

  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.tool.definitions"]).toBe(
    JSON.stringify([{ type: "function", name: "issue_refund" }]),
  );
  expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test.each<{ name: string; start: JsonRecord; delta: JsonRecord; input: number | undefined }>([
  {
    name: "cache counts only on message_start",
    start: { input_tokens: 40, cache_read_input_tokens: 1000, cache_creation_input_tokens: 200 },
    delta: { output_tokens: 12 },
    input: 40 + 1000 + 200,
  },
  {
    name: "message_delta repeats the cumulative counts",
    start: { input_tokens: 40, cache_read_input_tokens: 1000, cache_creation_input_tokens: 200 },
    delta: {
      input_tokens: 40,
      cache_read_input_tokens: 1000,
      cache_creation_input_tokens: 200,
      output_tokens: 12,
    },
    input: 40 + 1000 + 200,
  },
  {
    name: "message_delta repeats input without cache counts",
    start: { input_tokens: 40, cache_read_input_tokens: 1000, cache_creation_input_tokens: 200 },
    delta: { input_tokens: 40, output_tokens: 12 },
    input: 40 + 1000 + 200,
  },
  {
    name: "no input count reported",
    start: { cache_read_input_tokens: 1000 },
    delta: { output_tokens: 12 },
    input: undefined,
  },
])(
  "streamed input tokens include cache reads and writes: $name",
  async ({ start, delta, input }) => {
    const spans = setupSpans();

    const events = streamEvents().map((event) => {
      if (event.type === "message_start")
        return { ...event, message: { ...(event.message as JsonRecord), usage: start } };

      return event.type === "message_delta" ? { ...event, usage: delta } : event;
    });

    const fake = createFakeFetch(namedSseResponse(events));

    const stream = await clientWith(fake.fetch).messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      messages: [{ role: "user", content: "Say hello" }],
      stream: true,
    });

    await collectStream(stream);

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(input);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(12);
    expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(1000);
  },
);

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

test("stream capture retains no output when output capture is disabled", async () => {
  const spans = setupSpans(undefined, false);
  const capture = vi.spyOn(telemetrySdk, "boundedCaptureDetails");
  const startSpan = telemetrySdk.startSpan;
  const endedFields: unknown[] = [];
  vi.spyOn(telemetrySdk, "startSpan").mockImplementation((...args) => {
    const span = startSpan(...args);
    const end = span.end.bind(span);
    vi.spyOn(span, "end").mockImplementation((fields) => {
      endedFields.push(fields);

      return end(fields);
    });

    return span;
  });
  const fake = createFakeFetch(namedSseResponse(streamEvents()));

  const stream = await clientWith(fake.fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(streamEvents());
  expect(capture).not.toHaveBeenCalled();
  expect(endedFields).toHaveLength(1);
  expect(endedFields[0]).toHaveProperty("output", undefined);
  expect((await exportedSpan(spans)).attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("wrapAnthropic observes a CommonJS client subclass stream", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(namedSseResponse(streamEvents()));
  const require = createRequire(import.meta.url);
  const CommonJsAnthropic = require("@anthropic-ai/sdk").default as typeof Anthropic;

  class CustomAnthropic extends CommonJsAnthropic {}

  const client = wrapAnthropic(
    new CustomAnthropic({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }),
  );

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toHaveLength(streamEvents().length);
  expect(jsonAttr(await exportedSpan(spans), "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
  ]);
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
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"],
  ).toBe(true);
});

test("messages.create streaming omits early partial output when a mask is configured", async () => {
  const spans = setupSpans(undefined, true, (value) => value);
  const counted = countedSseResponse(streamEvents());

  const stream = await clientWith(createFakeFetch(counted.response).fetch).messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  for await (const event of stream) {
    if (readString(asRecord(event)?.type) === "content_block_delta") break;
  }

  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("messages.create streaming reports mapping failures without changing provider delivery", async () => {
  const errors: Error[] = [];

  const spans = setupSpans(undefined, true, (value) => value, {
    onError: (error) => errors.push(error),
  });

  const broken = {
    get type() {
      throw new Error("mapping failed");
    },
  };

  const events = [
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial" } },
    broken,
    broken,
    { type: "message_stop" },
  ];

  const source = {
    controller: new AbortController(),
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };

  const client = wrapAnthropic({ messages: { create: async (_params: unknown) => source } });

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(errors).toHaveLength(1);
  expect(errors[0]?.message).toBe("mapping failed");
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("messages.create streaming keeps telemetry on a core span without a capture policy", async () => {
  const spans = setupSpans();
  const startSpan = telemetrySdk.startSpan;
  vi.spyOn(telemetrySdk, "startSpan").mockImplementation((...args) => {
    const legacySpan = { ...startSpan(...args) };
    Reflect.deleteProperty(legacySpan, "capturePolicy");
    Reflect.deleteProperty(legacySpan, "recordOutputChunk");
    Reflect.deleteProperty(legacySpan, "reportError");

    return legacySpan;
  });

  const events = [
    {
      type: "message_start",
      message: messagePayload({
        id: "msg_legacy",
        model: "claude-sonnet-4-6",
        content: [],
        usage: { input_tokens: 5, output_tokens: 0 },
      }),
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "delivered" } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
    { type: "message_stop" },
  ];

  const source = {
    controller: new AbortController(),
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };

  const client = wrapAnthropic({ messages: { create: async (_params: unknown) => source } });

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.model"]).toBe("claude-sonnet-4-6");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "delivered" }] },
  ]);
});

test("messages.create streaming leaves the configured attribute limit to the core", async () => {
  const spans = setupSpans(undefined, true, undefined, { maxAttributeLength: 128 });

  const events = [
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "x".repeat(1_000) },
    },
    { type: "message_stop" },
  ];

  const source = {
    controller: new AbortController(),
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };

  const client = wrapAnthropic({ messages: { create: async (_params: unknown) => source } });

  const stream = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(output).toHaveLength(128);
  expect(output.endsWith("...[truncated]")).toBe(true);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
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

test("uninstrumentAnthropic preserves a method installed after its own wrapper", () => {
  const prototype = Object.getPrototypeOf(
    new Anthropic({ apiKey: "test", maxRetries: 0 }).messages,
  ) as { create: (...args: never[]) => unknown };

  const original = prototype.create;

  const laterOwner = function (this: unknown, ...args: never[]) {
    return original.apply(this, args);
  };

  try {
    instrumentAnthropic();
    prototype.create = laterOwner;
    uninstrumentAnthropic();

    expect(prototype.create).toBe(laterOwner);
  } finally {
    prototype.create = original;
  }
});

test("instrumentAnthropic and wrapAnthropic together record one span per call", async () => {
  const spans = setupSpans();
  instrumentAnthropic();

  const fake = createFakeFetch(
    jsonResponse(messagePayload({ id: "msg_global_wrapped" })),
    jsonResponse(messagePayload({ id: "msg_beta_both" })),
    jsonResponse(messagePayload({ id: "msg_instance_wrapped_after_global" })),
  );

  const client = wrapAnthropic(new Anthropic({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }));

  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi wrapped" }],
  });
  await client.beta.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi beta" }],
  });

  const activeSpans = await finishedSpans(spans, 2);
  expect(fake.requests).toHaveLength(2);
  expect(activeSpans.map((span) => span.attributes["gen_ai.response.id"])).toEqual([
    "msg_global_wrapped",
    "msg_beta_both",
  ]);

  uninstrumentAnthropic();
  await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi after uninstrument" }],
  });

  const allSpans = await finishedSpans(spans, 3);
  expect(fake.requests).toHaveLength(3);
  expect(allSpans[2]!.attributes["gen_ai.response.id"]).toBe("msg_instance_wrapped_after_global");
});

const providerClients = {
  AnthropicBedrock: class AnthropicBedrock {},
  AnthropicBedrockMantle: class AnthropicBedrockMantle {},
  AnthropicVertex: class AnthropicVertex {},
  AnthropicFoundry: class AnthropicFoundry {},
  AnthropicAws: class AnthropicAws {},
};

test.each<{ client: keyof typeof providerClients; subclass: boolean; expected: string }>([
  { client: "AnthropicBedrockMantle", subclass: false, expected: "aws.bedrock" },
  { client: "AnthropicBedrock", subclass: true, expected: "aws.bedrock" },
  { client: "AnthropicVertex", subclass: true, expected: "gcp.vertex_ai" },
  { client: "AnthropicFoundry", subclass: false, expected: "anthropic" },
  { client: "AnthropicAws", subclass: false, expected: "anthropic" },
])(
  "wrapAnthropic maps $client (subclass: $subclass) to $expected",
  async ({ client, subclass, expected }) => {
    const spans = setupSpans();
    const Base = providerClients[client];
    const Client = subclass ? class CustomClient extends Base {} : Base;

    const instance = Object.assign(new Client(), {
      messages: { create: async (_params: JsonValue) => messagePayload({ id: "msg_provider" }) },
    });

    await wrapAnthropic(instance).messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 64,
      messages: [{ role: "user", content: "Hi" }],
    });

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.provider.name"]).toBe(expected);
  },
);

test("beta.messages.create records a generation span for the beta endpoint", async () => {
  const spans = setupSpans();
  const messages = [{ role: "user" as const, content: "Say hello" }];
  const fake = createFakeFetch(jsonResponse(messagePayload({ id: "msg_beta" })));
  const client = clientWith(fake.fetch);

  await client.beta.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages,
    betas: ["context-management-2025-06-27"],
  });

  const span = await exportedSpan(spans);
  expect(fake.requests).toHaveLength(1);
  expect(fake.requests[0]?.path).toBe("/v1/messages");
  expect(fake.requests[0]?.body).not.toHaveProperty("betas");
  expect(span.name).toBe("chat claude-sonnet-4-6");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("anthropic");
  expect(span.attributes["gen_ai.response.id"]).toBe("msg_beta");
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(messages));
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10 + 3 + 2);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);
});

test("beta.messages.stream helper routes through create and records one span", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(namedSseResponse(streamEvents()));
  const client = clientWith(fake.fetch);

  const stream = client.beta.messages.stream({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Say hello" }],
  });

  const finalMessage = await stream.finalMessage();

  expect(finalMessage.id).toBe("msg_stream");
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: "Hello world" }] },
  ]);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
});

test("beta.messages.parse routes through create and records one span", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse(messagePayload({ content: [{ type: "text", text: '{"city":"Accra"}' }] })),
  );

  const client = clientWith(fake.fetch);

  const parsed = await client.beta.messages.parse({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Where?" }],
  });

  expect(parsed.id).toBe("msg_1");
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "text", text: '{"city":"Accra"}' }] },
  ]);
});

test("beta.messages.toolRunner records one generation span per model turn", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse(
      messagePayload({
        id: "msg_tool_turn",
        content: [{ type: "tool_use", id: "toolu_1", name: "weather", input: { city: "Accra" } }],
        stop_reason: "tool_use",
      }),
    ),
    jsonResponse(messagePayload({ id: "msg_final_turn" })),
  );

  const client = clientWith(fake.fetch);
  const run = vi.fn(() => "sunny");

  const finalMessage = await client.beta.messages.toolRunner({
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user", content: "Weather in Accra?" }],
    tools: [
      {
        type: "custom",
        name: "weather",
        description: "Look up the weather",
        input_schema: { type: "object", properties: { city: { type: "string" } } },
        run,
        parse: (input: unknown) => input,
      },
    ],
  });

  expect(finalMessage.id).toBe("msg_final_turn");
  expect(run).toHaveBeenCalledOnce();
  expect(fake.requests).toHaveLength(2);
  const turns = await finishedSpans(spans, 2);
  expect(turns.map((span) => span.attributes["gen_ai.response.id"])).toEqual([
    "msg_tool_turn",
    "msg_final_turn",
  ]);
  expect(turns.map((span) => span.attributes["gen_ai.response.finish_reasons"])).toEqual([
    ["tool_use"],
    ["end_turn"],
  ]);
});

test("instrumentAnthropic covers beta.messages and uninstrumentAnthropic restores it", async () => {
  const spans = setupSpans();
  instrumentAnthropic();

  const fake = createFakeFetch(
    jsonResponse(messagePayload({ id: "msg_beta_global" })),
    jsonResponse(messagePayload({ id: "msg_beta_restored" })),
  );

  const client = new Anthropic({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 });

  const params = {
    model: "claude-sonnet-4-6",
    max_tokens: 64,
    messages: [{ role: "user" as const, content: "Hi" }],
  };

  await client.beta.messages.create(params);
  await finishedSpans(spans, 1);

  uninstrumentAnthropic();
  await client.beta.messages.create(params);
  await flush();

  expect(fake.requests).toHaveLength(2);
  expect(spans.getFinishedSpans()).toHaveLength(1);
});

function betaStreamEvents(...blocks: [JsonRecord, JsonRecord[]][]): JsonRecord[] {
  const events: JsonRecord[] = [
    {
      type: "message_start",
      message: messagePayload({
        id: "msg_beta_stream",
        model: "claude-opus-5",
        content: [],
        usage: { input_tokens: 5, output_tokens: 0 },
      }),
    },
  ];

  blocks.forEach(([block, deltas], index) => {
    events.push({ type: "content_block_start", index, content_block: block });

    for (const delta of deltas) events.push({ type: "content_block_delta", index, delta });
    events.push({ type: "content_block_stop", index });
  });

  events.push(
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
    { type: "message_stop" },
  );

  return events;
}

async function streamedBetaSpan(
  events: JsonRecord[],
  spans: InMemorySpanExporter = setupSpans(),
): Promise<ReadableSpan> {
  const client = clientWith(createFakeFetch(namedSseResponse(events)).fetch);

  const stream = await client.beta.messages.create({
    model: "claude-opus-5",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi" }],
    stream: true,
  });

  await collectStream(stream);

  return exportedSpan(spans);
}

const fallbackBlock = {
  type: "fallback",
  from: { model: "claude-opus-5" },
  to: { model: "claude-opus-4-8" },
};

// A fallback block marks the switch point; only a fallback_message iteration in the terminal
// usage proves the fallback model served the response.
function fallbackStream(iterations: JsonRecord[]): JsonRecord[] {
  const events = betaStreamEvents(
    [fallbackBlock, []],
    [{ type: "text", text: "" }, [{ type: "text_delta", text: "Hi" }]],
  );

  const messageDelta = events.find((event) => event.type === "message_delta")!;
  messageDelta.usage = { output_tokens: 2, iterations };

  return events;
}

test("a stream delta over the encoded limit is dropped and earlier output is kept", async () => {
  const spans = setupSpans();

  const events = betaStreamEvents(
    [{ type: "text", text: "" }, [{ type: "text_delta", text: "first" }]],
    [{ type: "text", text: "" }, [{ type: "text_delta", text: '"'.repeat(30_000) }]],
  );

  const span = await streamedBetaSpan(events, spans);

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("[truncated]");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [
        { type: "text", text: "first" },
        { type: "text", text: "" },
      ],
    },
  ]);
});

test.each([
  [400, true],
  [6_000, false],
] as const)(
  "many small deltas do not exhaust the item budget (%s deltas)",
  async (count, complete) => {
    const spans = setupSpans(undefined, true, (value) => value);
    const deltas = Array.from({ length: count }, () => ({ type: "text_delta", text: "Hello wor" }));
    const events = betaStreamEvents([{ type: "text", text: "" }, deltas]);

    const span = await streamedBetaSpan(events, spans);

    if (complete) {
      expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
      expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
        { role: "assistant", content: [{ type: "text", text: "Hello wor".repeat(count) }] },
      ]);
    } else {
      expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
      expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
    }
  },
);

test("deltas that fill the stream limit keep the accepted text and deliver every event", async () => {
  const spans = setupSpans();

  const events = betaStreamEvents([
    { type: "text", text: "" },
    [
      { type: "text_delta", text: "x".repeat(49_000) },
      { type: "text_delta", text: "y".repeat(127) },
    ],
  ]);

  const stream = await clientWith(
    createFakeFetch(namedSseResponse(events)).fetch,
  ).beta.messages.create({
    model: "claude-opus-5",
    max_tokens: 64,
    messages: [{ role: "user", content: "Hi" }],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);

  const output = jsonAttr<Array<{ content: Array<{ text?: string }> }>>(
    span,
    "gen_ai.output.messages",
  );

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(output[0]?.content[0]?.text).toBe("x".repeat(49_000));
  expect(String(span.attributes["gen_ai.output.messages"]).length).toBeLessThanOrEqual(48 * 1024);
});

test.each([
  [200, true],
  [300, false],
] as const)(
  "escape-heavy deltas are charged at their serialized size (%s deltas)",
  async (count, complete) => {
    const spans = setupSpans();
    const chunk = '"\n'.repeat(50);
    const deltas = Array.from({ length: count }, () => ({ type: "text_delta", text: chunk }));

    const span = await streamedBetaSpan(
      betaStreamEvents([{ type: "text", text: "" }, deltas]),
      spans,
    );

    const text = jsonAttr<Array<{ content: Array<{ text: string }> }>>(
      span,
      "gen_ai.output.messages",
    )[0]?.content[0]?.text;

    if (complete) {
      expect(text).toBe(chunk.repeat(count));
      expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
    } else {
      expect(text?.length).toBeGreaterThan(0);
      expect(chunk.repeat(count).startsWith(text ?? "x")).toBe(true);
      expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
    }
  },
);

test.each([
  [1_000, false, "東".repeat(400)],
  [1_000, true, "[redacted]"],
  [0, false, undefined],
] as const)(
  "the configured cap applies after the mask, not to stream retention (cap %s, redact %s)",
  async (maxAttributeLength, redact, expected) => {
    const spans = setupSpans(
      undefined,
      true,
      (value) =>
        redact ? [{ role: "assistant", content: [{ type: "text", text: "[redacted]" }] }] : value,
      { maxAttributeLength },
    );

    const text = redact ? "x".repeat(5_000) : "東".repeat(400);

    const span = await streamedBetaSpan(
      betaStreamEvents([{ type: "text", text: "" }, [{ type: "text_delta", text }]]),
      spans,
    );

    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();

    if (expected === undefined) expect(span.attributes["gen_ai.output.messages"] ?? "").toBe("");
    else
      expect(
        jsonAttr<Array<{ content: Array<{ text: string }> }>>(span, "gen_ai.output.messages")[0]
          ?.content[0]?.text,
      ).toBe(expected);
  },
);

const declinedIteration = {
  type: "message",
  model: "claude-opus-5",
  input_tokens: 5,
  output_tokens: 0,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
};

const servedIteration = {
  ...declinedIteration,
  type: "fallback_message",
  model: "claude-opus-4-8",
  output_tokens: 2,
};

const advisorIteration = {
  ...declinedIteration,
  type: "advisor_message",
  model: "claude-haiku-4-5",
};

test.each([
  ["at the oldest tail index", 1, "claude-opus-4-8"],
  ["absent from the tail", undefined, "claude-opus-5"],
] as const)(
  "stream fallback inspection reads only the bounded tail (served %s)",
  async (_case, servedIndex, model) => {
    const errors: unknown[] = [];

    const spans = setupSpans(undefined, true, undefined, {
      onError: (error) => errors.push(error),
    });

    const iterations = Array<JsonRecord>(1_001).fill(declinedIteration);
    Object.defineProperty(iterations, 0, {
      get() {
        throw new Error("inspected an iteration outside the bounded tail");
      },
    });

    if (servedIndex !== undefined) iterations[servedIndex] = servedIteration;
    iterations[1_000] = advisorIteration;

    const events = [
      {
        type: "message_start",
        message: messagePayload({
          id: "msg_bounded_fallback",
          model: "claude-opus-5",
          content: [],
          usage: { input_tokens: 5, output_tokens: 0 },
        }),
      },
      {
        type: "message_delta",
        delta: { stop_reason: "end_turn" },
        usage: { output_tokens: 2, iterations },
      },
      { type: "message_stop" },
    ];

    const source = {
      controller: new AbortController(),
      async *[Symbol.asyncIterator]() {
        yield* events;
      },
    };

    const client = wrapAnthropic({ messages: { create: async (_params: unknown) => source } });

    const stream = await client.messages.create({
      model: "claude-opus-5",
      max_tokens: 64,
      messages: [],
      stream: true,
    });

    expect(await collectStream(stream)).toHaveLength(events.length);
    const span = await exportedSpan(spans);
    expect(errors).toEqual([]);
    expect(span.attributes["gen_ai.response.model"]).toBe(model);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test("fallback metadata overflow preserves complete masked output", async () => {
  const spans = setupSpans(undefined, true, (value) => value);
  const iterations = Array<JsonRecord>(1_001).fill(declinedIteration);
  iterations[999] = servedIteration;
  iterations[1_000] = advisorIteration;

  const span = await streamedBetaSpan(fallbackStream(iterations), spans);

  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [fallbackBlock, { type: "text", text: "Hi" }],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test.each<{ name: string; iterations: JsonRecord[]; expected: string }>([
  {
    name: "fallback served",
    iterations: [declinedIteration, servedIteration],
    expected: "claude-opus-4-8",
  },
  { name: "fallback failed", iterations: [declinedIteration], expected: "claude-opus-5" },
  {
    name: "advisor without fallback",
    iterations: [declinedIteration, advisorIteration],
    expected: "claude-opus-5",
  },
  {
    name: "advisor after fallback",
    iterations: [declinedIteration, servedIteration, advisorIteration],
    expected: "claude-opus-4-8",
  },
  {
    name: "last fallback wins",
    iterations: [
      servedIteration,
      { ...servedIteration, model: "claude-sonnet-5" },
      advisorIteration,
    ],
    expected: "claude-sonnet-5",
  },
])(
  "beta streams take the served model from the last fallback_message iteration: $name",
  async ({ iterations, expected }) => {
    const span = await streamedBetaSpan(fallbackStream(iterations));

    expect(span.attributes["gen_ai.response.model"]).toBe(expected);
  },
);

const compactionStream = (...deltas: JsonRecord[]) =>
  betaStreamEvents([
    { type: "compaction", content: null },
    deltas.map((delta) => ({ type: "compaction_delta", ...delta })),
  ]);

test.each<{ name: string; deltas: JsonRecord[]; expected: JsonRecord }>([
  {
    name: "repeated content replaces the earlier value",
    deltas: [
      { content: "Summary ", encrypted_content: "enc_1" },
      { content: "Summary so far.", encrypted_content: "enc_2" },
    ],
    expected: { content: "Summary so far.", encrypted_content: "enc_2" },
  },
  {
    name: "null content marks a failed compaction",
    deltas: [
      { content: "Summary", encrypted_content: "enc_1" },
      { content: null, encrypted_content: null },
    ],
    expected: { content: null, encrypted_content: null },
  },
  {
    name: "omitted content is left out like the SDK does",
    deltas: [{ content: "Summary", encrypted_content: "enc_1" }, { encrypted_content: "enc_2" }],
    expected: { encrypted_content: "enc_2" },
  },
  {
    name: "omitted encrypted_content keeps the earlier value",
    deltas: [{ content: "Summary", encrypted_content: "enc_1" }, { content: "Summary." }],
    expected: { content: "Summary.", encrypted_content: "enc_1" },
  },
])("beta streams match SDK compaction semantics: $name", async ({ deltas, expected }) => {
  const span = await streamedBetaSpan(compactionStream(...deltas));

  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "compaction", ...expected }] },
  ]);
});

test("beta streams record output chunk timing for compaction content", async () => {
  const metricExporter = new InMemoryMetricExporter(AggregationTemporality.DELTA);
  await streamedBetaSpan(
    compactionStream(
      { content: "Summary ", encrypted_content: "enc_1" },
      { encrypted_content: "enc_ignored" },
      { content: "Summary so far.", encrypted_content: "enc_2" },
    ),
    setupSpans(metricExporter),
  );

  const histogram = metricExporter
    .getMetrics()
    .flatMap((batch) => batch.scopeMetrics.flatMap((scope) => scope.metrics))
    .find((metric) => metric.descriptor.name === "gen_ai.client.operation.time_per_output_chunk");

  expect(histogram?.dataPoints).toEqual([
    expect.objectContaining({ value: expect.objectContaining({ count: 1 }) }),
  ]);
});

test("beta stream replacement recovery clears capture truncation", async () => {
  const span = await streamedBetaSpan(
    compactionStream(
      { content: "x".repeat(60 * 1024) },
      { content: "recovered", encrypted_content: "e" },
    ),
  );

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [{ type: "compaction", content: "recovered", encrypted_content: "e" }],
    },
  ]);
});

test("beta stream recovers from an oversized compaction start", async () => {
  const span = await streamedBetaSpan(
    betaStreamEvents([
      { type: "compaction", content: "x".repeat(60 * 1024) },
      [{ type: "compaction_delta", content: "recovered" }],
    ]),
  );

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "compaction", content: "recovered" }] },
  ]);
});

test("beta stream masks recovered compaction when its rejected start had encrypted content", async () => {
  const spans = setupSpans(undefined, true, (value) => value);

  const span = await streamedBetaSpan(
    betaStreamEvents([
      {
        type: "compaction",
        content: "x".repeat(60 * 1024),
        encrypted_content: "encrypted",
      },
      [{ type: "compaction_delta", content: "recovered" }],
    ]),
    spans,
  );

  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("beta stream recovers after repeated rejected compaction replacements", async () => {
  const span = await streamedBetaSpan(
    compactionStream(
      { content: "x".repeat(60 * 1024) },
      { content: "y".repeat(60 * 1024) },
      { content: "recovered" },
    ),
  );

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "compaction", content: "recovered" }] },
  ]);
});

test("beta stream retains a fitting compaction replacement after other output truncates", async () => {
  const events = betaStreamEvents(
    [{ type: "compaction" }, [{ type: "compaction_delta", content: "before" }]],
    [{ type: "text", text: "" }, [{ type: "text_delta", text: "x".repeat(60 * 1024) }]],
  );

  events.splice(-2, 0, {
    type: "content_block_delta",
    index: 0,
    delta: { type: "compaction_delta", content: "after" },
  });

  const span = await streamedBetaSpan(events);

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [
        { type: "compaction", content: "after" },
        { type: "text", text: "" },
      ],
    },
  ]);
});

test("beta stream recovers after rejecting an unseen compaction replacement", async () => {
  const events = betaStreamEvents();
  events.splice(
    -2,
    0,
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "compaction_delta", content: "x".repeat(60 * 1024) },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "compaction_delta", content: "recovered" },
    },
  );

  const span = await streamedBetaSpan(events);

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "compaction", content: "recovered" }] },
  ]);
});

test("beta stream keeps rejected inherited encrypted content unresolved until explicit replacement", async () => {
  const unresolved = await streamedBetaSpan(
    compactionStream(
      { content: "ok", encrypted_content: "enc" },
      { content: "x".repeat(60 * 1024) },
      { content: "recovered" },
    ),
  );

  expect(unresolved.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(unresolved, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "compaction", content: "recovered" }] },
  ]);

  const resolved = await streamedBetaSpan(
    compactionStream(
      { content: "ok", encrypted_content: "enc" },
      { content: "x".repeat(60 * 1024) },
      { content: "recovered" },
      { content: "resolved", encrypted_content: "enc-2" },
    ),
  );

  expect(resolved.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(resolved, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [{ type: "compaction", content: "resolved", encrypted_content: "enc-2" }],
    },
  ]);
});

test("beta stream rejected final replacement marks capture incomplete", async () => {
  const span = await streamedBetaSpan(compactionStream({ content: "x".repeat(60 * 1024) }));

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("beta stream rejected signature stays incomplete until a later signature fits", async () => {
  const events = betaStreamEvents([
    { type: "thinking", thinking: "" },
    [
      { type: "signature_delta", signature: "sig" },
      { type: "signature_delta", signature: "x".repeat(60 * 1024) },
    ],
  ]);

  const rejected = await streamedBetaSpan(events);

  expect(rejected.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(rejected, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ type: "thinking", thinking: "" }] },
  ]);

  events.splice(-2, 0, {
    type: "content_block_delta",
    index: 0,
    delta: { type: "signature_delta", signature: "recovered" },
  });
  const recovered = await streamedBetaSpan(events);

  expect(recovered.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(recovered, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [{ type: "thinking", thinking: "", signature: "recovered" }],
    },
  ]);
});

test("beta streams omit incomplete retained output when a mask is configured", async () => {
  const spans = setupSpans(undefined, true, (value) => value);

  const span = await streamedBetaSpan(
    betaStreamEvents([
      { type: "thinking", thinking: "" },
      [
        { type: "thinking_delta", thinking: "retained" },
        { type: "signature_delta", signature: "x".repeat(60 * 1024) },
      ],
    ]),
    spans,
  );

  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("beta stream releases a rejected signature before retaining a sibling", async () => {
  const events = betaStreamEvents(
    [
      { type: "thinking", thinking: "" },
      [
        { type: "signature_delta", signature: "s".repeat(40 * 1024) },
        { type: "signature_delta", signature: "x".repeat(60 * 1024) },
      ],
    ],
    [{ type: "text", text: "" }, [{ type: "text_delta", text: "t".repeat(10 * 1024) }]],
  );

  events.splice(-2, 0, {
    type: "content_block_delta",
    index: 0,
    delta: { type: "signature_delta", signature: "recovered" },
  });
  const span = await streamedBetaSpan(events);

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "", signature: "recovered" },
        { type: "text", text: "t".repeat(10 * 1024) },
      ],
    },
  ]);
});

test("beta stream does not retain rejected signatures for unseen indexes", async () => {
  const events = betaStreamEvents();
  const rejectedSignature = "x".repeat(60 * 1024);
  const firstIndex = 10_000;
  const count = 2_000;

  for (let offset = 0; offset < count; offset += 1) {
    events.splice(-2, 0, {
      type: "content_block_delta",
      index: firstIndex + offset,
      delta: { type: "signature_delta", signature: rejectedSignature },
    });
  }

  const additions = vi.spyOn(Set.prototype, "add");
  const span = await streamedBetaSpan(events);

  const retainedIndexes = additions.mock.calls.filter(
    ([value]) => typeof value === "number" && value >= firstIndex && value < firstIndex + count,
  );

  additions.mockRestore();

  expect(retainedIndexes).toEqual([]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("beta stream replacement budget retains raw tool JSON bytes", async () => {
  const signatures = Array.from({ length: 20 }, (_, index) => ({
    type: "signature_delta",
    signature: `sig-${index}`,
  }));

  const span = await streamedBetaSpan(
    betaStreamEvents([
      { type: "tool_use", id: "toolu_1", name: "lookup", input: {} },
      [
        { type: "input_json_delta", partial_json: `{}${" ".repeat(40 * 1024)}` },
        ...signatures,
        { type: "input_json_delta", partial_json: `${" ".repeat(10 * 1024)}x` },
      ],
    ]),
  );

  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_1",
          name: "lookup",
          input: {},
          signature: "sig-19",
        },
      ],
    },
  ]);
});

test("beta streams record mcp_tool_use input", async () => {
  const block = {
    type: "mcp_tool_use",
    id: "mcptoolu_1",
    name: "search",
    server_name: "docs",
    input: {},
  };

  const span = await streamedBetaSpan(
    betaStreamEvents([
      block,
      [
        { type: "input_json_delta", partial_json: '{"q": ' },
        { type: "input_json_delta", partial_json: '"otel"}' },
      ],
    ]),
  );

  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: [{ ...block, input: { q: "otel" } }] },
  ]);
});

test("beta streams keep a compaction delta without a start block as compaction", async () => {
  const events = betaStreamEvents();
  events.splice(1, 0, {
    type: "content_block_delta",
    index: 0,
    delta: { type: "compaction_delta", content: "Summary.", encrypted_content: "e" },
  });

  const span = await streamedBetaSpan(events);

  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: [{ type: "compaction", content: "Summary.", encrypted_content: "e" }],
    },
  ]);
});

const compactionIteration = {
  type: "compaction",
  input_tokens: 40,
  output_tokens: 7,
  cache_creation_input_tokens: 4,
  cache_read_input_tokens: 6,
  cache_creation: { ephemeral_5m_input_tokens: 1, ephemeral_1h_input_tokens: 3 },
};

const firstTurn = {
  type: "message",
  model: "claude-opus-5",
  input_tokens: 4,
  output_tokens: 2,
  cache_creation_input_tokens: 1,
  cache_read_input_tokens: 1,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 1 },
};

const secondTurn = {
  ...firstTurn,
  type: "fallback_message",
  model: "claude-opus-4-8",
  input_tokens: 6,
  output_tokens: 3,
  cache_creation_input_tokens: 2,
  cache_read_input_tokens: 2,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 2 },
};

// Advisor tokens are billed at the advisor's rates and left out of the top-level usage.
const advisorTurn = {
  ...firstTurn,
  type: "advisor_message",
  model: "claude-haiku-4-5",
  input_tokens: 100,
  output_tokens: 50,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 5 },
};

// The top-level usage sums both executor turns, while message_start reports only the first.
const turnsUsage = {
  input_tokens: 10,
  output_tokens: 5,
  cache_creation_input_tokens: 3,
  cache_read_input_tokens: 3,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 3 },
};

const firstTurnUsage = {
  input_tokens: 4,
  output_tokens: 0,
  cache_creation_input_tokens: 1,
  cache_read_input_tokens: 1,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 1 },
};

// A stream reports startUsage on message_start and finalUsage, minus the one-hour split it
// never carries there, on message_delta; a unary response reports finalUsage.
async function usageSpan(
  streaming: boolean,
  startUsage: JsonRecord,
  finalUsage: JsonRecord,
): Promise<ReadableSpan> {
  const errors: unknown[] = [];

  const spans = setupSpans(undefined, true, undefined, {
    onError: (error) => errors.push(error),
  });

  const { cache_creation: _split, ...deltaUsage } = finalUsage;
  const abortController = new AbortController();

  const events = [
    {
      type: "message_start",
      message: messagePayload({ id: "msg_usage", content: [], usage: startUsage }),
    },
    { type: "message_delta", delta: {}, usage: deltaUsage },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: deltaUsage },
    { type: "message_stop" },
  ];

  // A plain { abort, signal } controller: stream detection must not require an AbortController.
  const source = {
    controller: {
      abort: (reason?: unknown) => abortController.abort(reason),
      signal: abortController.signal,
    },
    async *[Symbol.asyncIterator]() {
      yield* events;
    },
  };

  const response = messagePayload({ usage: finalUsage });

  const client = wrapAnthropic({
    messages: { create: async (_params: unknown) => (streaming ? source : response) },
  });

  const result = await client.messages.create({
    model: "claude-opus-5",
    max_tokens: 64,
    messages: [],
    stream: streaming,
  });

  if (streaming) await collectStream(result as AsyncIterable<unknown>);
  const span = await exportedSpan(spans);
  expect(errors).toEqual([]);

  return span;
}

test.each([false, true])(
  "compaction iterations are added to usage and one-hour cache writes are recorded (streaming: %s)",
  async (streaming) => {
    const span = await usageSpan(streaming, firstTurnUsage, {
      ...turnsUsage,
      iterations: [
        firstTurn,
        advisorTurn,
        compactionIteration,
        secondTurn,
        { ...advisorTurn, input_tokens: 23, output_tokens: 7, cache_read_input_tokens: 9 },
      ],
    });

    expect(span.attributes["anthropic.advisor.model"]).toBe("claude-haiku-4-5");
    expect(span.attributes["anthropic.advisor.usage.input_tokens"]).toBe(123);
    expect(span.attributes["anthropic.advisor.usage.output_tokens"]).toBe(57);
    expect(span.attributes["anthropic.advisor.usage.cache_read_input_tokens"]).toBe(10);
    expect(span.attributes["anthropic.advisor.usage.cache_creation_input_tokens"]).toBe(2);
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10 + 3 + 3 + 40 + 6 + 4);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5 + 7);
    expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3 + 6);
    expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(3 + 4);
    expect(span.attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"]).toBe(
      1 + 2 + 3,
    );
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  },
);

test.each([false, true])(
  "one-hour cache writes are recorded without usage.iterations (streaming: %s)",
  async (streaming) => {
    const span = await usageSpan(streaming, firstTurnUsage, {
      ...firstTurnUsage,
      output_tokens: 2,
    });

    expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(1);
    expect(span.attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"]).toBe(1);
    expect(
      Object.keys(span.attributes).filter((key) => key.startsWith("anthropic.advisor.")),
    ).toEqual([]);
  },
);

test.each([
  [false, null],
  [true, null],
  [false, ""],
  [true, " "],
] as const)(
  "advisor usage preserves zero and omits absent fields (streaming: %s, model: %s)",
  async (streaming, model) => {
    const span = await usageSpan(streaming, firstTurnUsage, {
      ...turnsUsage,
      iterations: [{ type: "advisor_message", model, input_tokens: 0, output_tokens: 0 }],
    });
    expect(span.attributes["anthropic.advisor.usage.input_tokens"]).toBe(0);
    expect(span.attributes["anthropic.advisor.usage.output_tokens"]).toBe(0);
    expect(span.attributes["anthropic.advisor.model"]).toBeUndefined();
    expect(span.attributes["anthropic.advisor.usage.cache_read_input_tokens"]).toBeUndefined();
    expect(span.attributes["anthropic.advisor.usage.cache_creation_input_tokens"]).toBeUndefined();
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(16);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test.each([false, true])(
  "advisor totals with mixed models are not assigned one model's rates (streaming: %s)",
  async (streaming) => {
    const span = await usageSpan(streaming, firstTurnUsage, {
      ...turnsUsage,
      iterations: [advisorTurn, { ...advisorTurn, model: "claude-opus-4-8" }, advisorTurn],
    });
    expect(span.attributes["anthropic.advisor.usage.input_tokens"]).toBe(300);
    expect(span.attributes["anthropic.advisor.model"]).toBeUndefined();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test.each([false, true])(
  "compaction usage reads only the bounded tail of usage.iterations (streaming: %s)",
  async (streaming) => {
    const iterations = Array<JsonRecord>(1_001).fill(firstTurn);
    Object.defineProperty(iterations, 0, {
      get() {
        throw new Error("inspected an iteration outside the bounded tail");
      },
    });
    iterations[1] = compactionIteration;

    const span = await usageSpan(streaming, firstTurnUsage, { ...firstTurnUsage, iterations });

    // Executor iterations past the bound go unseen, so a stream keeps message_start's split.
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(7);
    expect(span.attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"]).toBe(1 + 3);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test.each([
  { name: "start only", deltas: [], advisorInput: 100 },
  {
    name: "delta without iterations",
    deltas: [{ type: "message_delta", delta: {}, usage: { output_tokens: 3 } }],
    advisorInput: 100,
  },
  {
    name: "delta with cumulative iterations",
    deltas: [
      {
        type: "message_delta",
        delta: {},
        usage: { ...turnsUsage, iterations: [firstTurn, advisorTurn, advisorTurn] },
      },
    ],
    advisorInput: 200,
  },
])(
  "interrupted streams keep advisor usage already received on message_start ($name)",
  async ({ deltas, advisorInput }) => {
    const spans = setupSpans();
    const events = [
      {
        type: "message_start",
        message: messagePayload({
          id: "msg_usage",
          content: [],
          usage: { ...firstTurnUsage, iterations: [firstTurn, advisorTurn] },
        }),
      },
      ...deltas,
    ];

    const source = {
      controller: new AbortController(),
      async *[Symbol.asyncIterator]() {
        yield* events;
        yield { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } };
        throw new Error("the caller stopped before this event");
      },
    };

    const client = wrapAnthropic({ messages: { create: async (_params: unknown) => source } });
    const stream = await client.messages.create({
      model: "claude-opus-5",
      max_tokens: 64,
      messages: [],
      stream: true,
    });

    let seen = 0;
    for await (const _event of stream as AsyncIterable<unknown>) {
      seen += 1;
      if (seen === events.length) break;
    }

    const span = await exportedSpan(spans);
    expect(span.attributes["anthropic.advisor.model"]).toBe("claude-haiku-4-5");
    expect(span.attributes["anthropic.advisor.usage.input_tokens"]).toBe(advisorInput);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test("streams keep message_start's one-hour split when an executor iteration leaves it out", async () => {
  const span = await usageSpan(
    true,
    {
      ...firstTurnUsage,
      cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 10 },
    },
    { ...turnsUsage, iterations: [{ ...firstTurn, cache_creation: null }, secondTurn] },
  );

  expect(span.attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"]).toBe(10);
});
