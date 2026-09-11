import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  AggregationTemporality,
  InMemoryMetricExporter,
  type PushMetricExporter,
} from "@opentelemetry/sdk-metrics";
import { flush, init, shutdown } from "@telemetry-dev/sdk";
import * as sdk from "@telemetry-dev/sdk";
import OpenAI, { AzureOpenAI } from "openai";
import { Stream } from "openai/core/streaming";
import { afterEach, expect, test, vi } from "vitest";

import {
  instrumentOpenAI,
  type InstrumentOpenAIOptions,
  uninstrumentOpenAI,
  wrapOpenAI,
} from "../src/index.ts";

const SPAN_STATUS_UNSET = 0;
const SPAN_STATUS_ERROR = 2;

interface CapturedRequest {
  method: string | undefined;
  path: string;
  body: unknown;
}

function jsonResponse<T>(body: T): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json", "x-request-id": "req_test" },
  });
}

function jsonErrorResponse(status: number, message: string): Response {
  return new Response(
    JSON.stringify({ error: { message, type: "invalid_request_error", code: "bad_request" } }),
    {
      status,
      headers: { "content-type": "application/json", "x-request-id": "req_error" },
    },
  );
}

function sseResponse(events: unknown[]): Response {
  const body = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream", "x-request-id": "req_stream" },
  });
}

function erroringSseResponse<TEvent, TError>(event: TEvent, error: TError): Response {
  const encoder = new TextEncoder();
  let sent = false;
  const body = new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        if (!sent) {
          sent = true;
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
          return;
        }
        controller.error(error);
      },
    },
    { highWaterMark: 0 },
  );
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream", "x-request-id": "req_stream_error" },
  });
}

interface CountedResponse {
  response: Response;
  pulledChunks: () => number;
  chunkCount: number;
}

function countedSseResponse(events: unknown[]): CountedResponse {
  const encoder = new TextEncoder();
  const parts = [
    ...events.map((event) => `data: ${JSON.stringify(event)}\n\n`),
    "data: [DONE]\n\n",
  ];
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
      headers: { "content-type": "text/event-stream", "x-request-id": "req_stream_counted" },
    }),
    pulledChunks: () => pulled,
    chunkCount: parts.length,
  };
}

interface FakeFetch {
  fetch: typeof fetch;
  requests: CapturedRequest[];
}

function createFakeFetch(...responses: Response[]): FakeFetch {
  const requests: CapturedRequest[] = [];
  const fetchImpl: typeof fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : String(input);
    const bodyText = String(init?.body) === init?.body ? init.body : undefined;
    requests.push({
      method: init?.method,
      path: new URL(url).pathname,
      body: bodyText ? JSON.parse(bodyText) : undefined,
    });
    const response = responses.shift();
    if (!response) throw new Error(`unexpected request to ${url}`);
    return response;
  };
  return { fetch: fetchImpl, requests };
}

function setupSpans(metricExporter?: PushMetricExporter): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "openai-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter, metricExporter },
  );
  return spanExporter;
}

function clientWith(fetchImpl: typeof fetch, options?: InstrumentOpenAIOptions): OpenAI {
  return wrapOpenAI(new OpenAI({ apiKey: "test", fetch: fetchImpl, maxRetries: 0 }), options);
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

function isObject<T>(value: T): value is T & object {
  return value !== null && Object(value) === value;
}

function hasSyntheticUsageChunk<T>(value: T): boolean {
  const raw: unknown = value;
  if (!isObject(raw) || Array.isArray(raw)) return false;
  if (!("usage" in raw) || !("choices" in raw)) return false;
  return raw.usage !== undefined && Array.isArray(raw.choices) && raw.choices.length === 0;
}

async function collectStream(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const chunks: unknown[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

afterEach(async () => {
  uninstrumentOpenAI();
  await shutdown();
  vi.restoreAllMocks();
});

test("chat completions map request, response, usage, finish reason, provider, and sampling attributes", async () => {
  const spans = setupSpans();
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: "Be terse" },
    { role: "user", content: "Say hello" },
  ];
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_1",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Hello." }, finish_reason: "stop" },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
        prompt_tokens_details: { cached_tokens: 3 },
        completion_tokens_details: { reasoning_tokens: 2 },
      },
    }),
  );
  const client = clientWith(fake.fetch);

  await client.chat.completions.create({
    model: "gpt-4o",
    messages,
    temperature: 0.7,
    top_p: 0.9,
    max_completion_tokens: 64,
    stop: "END",
    seed: 7,
    frequency_penalty: 0.1,
    presence_penalty: 0.2,
  });

  const span = await exportedSpan(spans);
  expect(fake.requests).toHaveLength(1);
  expect(fake.requests[0]?.path).toBe("/v1/chat/completions");
  expect(span.name).toBe("chat gpt-4o");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(span.attributes["gen_ai.request.model"]).toBe("gpt-4o");
  expect(span.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(span.attributes["gen_ai.response.id"]).toBe("chatcmpl_1");
  expect(span.attributes["gen_ai.input.messages"]).toBe(JSON.stringify(messages));
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Hello." },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(10);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(15);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(64);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["END"]);
  expect(span.attributes["gen_ai.request.seed"]).toBe(7);
  expect(span.attributes["gen_ai.request.frequency_penalty"]).toBe(0.1);
  expect(span.attributes["gen_ai.request.presence_penalty"]).toBe(0.2);
});

test("wrapped chat completion APIPromise keeps parse helper and promise identity", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_parse",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Parsed" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
    }),
  );

  const result = clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Parse this" }],
  });
  const rawResult: unknown = result;
  const resultWithPrivateParse = rawResult as {
    parse(): Promise<Awaited<typeof result>>;
  };

  expect(result).toBeInstanceOf(Promise);
  expect(resultWithPrivateParse.parse instanceof Function ? "function" : "other").toBe("function");
  const parsed = await resultWithPrivateParse.parse();
  const awaited = await result;

  expect(awaited).toBe(parsed);
  expect(parsed.id).toBe("chatcmpl_parse");
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.id"]).toBe("chatcmpl_parse");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Parsed" },
  ]);
});

test("chat completions record one error span when the OpenAI API returns 4xx", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(jsonErrorResponse(400, "bad model"));

  await expect(
    clientWith(fake.fetch).chat.completions.create({
      model: "gpt-bad",
      messages: [{ role: "user", content: "Fail" }],
    }),
  ).rejects.toMatchObject({ status: 400 });

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.request.model"]).toBe("gpt-bad");
  // The SDK falls back to the constructor name when err.name is the generic "Error"
  // (openai's APIError subclasses never override name) — Python parity: type(e).__name__.
  expect(span.attributes["error.type"]).toBe("BadRequestError");
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.type"]).toBe("BadRequestError");
  expect(exception?.attributes?.["exception.message"]).toContain("bad model");
  expect(exception?.attributes?.["log.severity_number"]).toBe(17);
});

test("chat completions record non-Error promise rejections", async () => {
  const spans = setupSpans();
  const rejection = "request rejected";
  const fetch: typeof globalThis.fetch = async () => {
    throw new Error("unexpected fetch");
  };
  const client = new OpenAI({ apiKey: "test", fetch, maxRetries: 0 });
  Object.assign(client.chat.completions, {
    create<T>(_body: T) {
      return Promise.reject(rejection);
    },
  });
  const wrapped = wrapOpenAI(client);

  await expect(
    wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Fail" }],
    }),
  ).rejects.toBe(rejection);

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.message"]).toBe(rejection);
});

test("synchronous create and retrieve failures end spans and preserve thrown values", async () => {
  const spans = setupSpans();
  const unusedFetch: typeof fetch = async () => {
    throw new Error("unexpected fetch");
  };
  const client = new OpenAI({ apiKey: "test", fetch: unusedFetch, maxRetries: 0 });
  const createFailure = "create failed";
  const retrieveFailure = { message: "retrieve failed" };
  Object.assign(client.chat.completions, {
    create<T>(_body: T) {
      throw createFailure;
    },
  });
  Object.assign(client.responses, {
    retrieve<T>(_responseId: T) {
      throw retrieveFailure;
    },
  });
  const wrapped = wrapOpenAI(client);

  let caughtCreate = false;
  try {
    void wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Fail" }],
    });
  } catch (error) {
    caughtCreate = true;
    expect(error).toBe(createFailure);
  }
  expect(caughtCreate).toBe(true);

  let caughtRetrieve = false;
  try {
    void wrapped.responses.retrieve("resp_sync", { stream: true });
  } catch (error) {
    caughtRetrieve = true;
    expect(error).toBe(retrieveFailure);
  }
  expect(caughtRetrieve).toBe(true);

  const finished = await finishedSpans(spans, 2);
  expect(finished.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
  expect(
    finished.map(
      (span) =>
        span.events.find((event) => event.name === "exception")?.attributes?.["exception.message"],
    ),
  ).toEqual(["create failed", "[object Object]"]);
});

test("chat completions preserve tool-call messages in output", async () => {
  const spans = setupSpans();
  const toolCall = {
    id: "call_1",
    type: "function",
    function: { name: "get_weather", arguments: '{"location":"Paris"}' },
  };
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_tool",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null, tool_calls: [toolCall] },
          finish_reason: "tool_calls",
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
    }),
  );

  await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Weather in Paris?" }],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          parameters: { type: "object", properties: { location: { type: "string" } } },
        },
      },
    ],
  });

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: null, tool_calls: [toolCall] },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);
});

test("chat completions capture every returned choice", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_two",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "First" }, finish_reason: "stop" },
        { index: 1, message: { role: "assistant", content: "Second" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 8, total_tokens: 18 },
    }),
  );

  await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Give two options" }],
    n: 2,
  });

  const span = await exportedSpan(spans);
  expect(jsonAttr<unknown[]>(span, "gen_ai.output.messages")).toHaveLength(2);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "First" },
    { role: "assistant", content: "Second" },
  ]);
});

test("chat completions record finish reasons for single and multi-choice responses", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_single_finish",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Single" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
    jsonResponse({
      id: "chatcmpl_multi_finish",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "First" }, finish_reason: "stop" },
        { index: 1, message: { role: "assistant", content: "Second" }, finish_reason: "length" },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 8, total_tokens: 18 },
    }),
  );
  const client = clientWith(fake.fetch);

  await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "One answer" }],
  });
  await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Two answers" }],
    n: 2,
  });

  const finished = await finishedSpans(spans, 2);
  const single = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_single_finish",
  );
  const multi = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_multi_finish",
  );
  expect(single?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(multi?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop", "length"]);
});

test.each([false, true])("chunk timing preserves delivery with old core=%s", async (oldCore) => {
  const spans = setupSpans();

  if (oldCore) {
    const start = sdk.startSpan;
    vi.spyOn(sdk, "startSpan").mockImplementation((...args) => {
      const handle = start(...args);
      Reflect.deleteProperty(handle, "recordOutputChunk");

      return handle;
    });
  }

  const deltas = [
    { role: "assistant" },
    { content: "A" },
    { content: null },
    { content: "" },
    { tool_calls: [] },
    { tool_calls: [{ index: 0, function: { arguments: "" } }] },
    { content: "B" },
    { function_call: { name: "weather" } },
    { function_call: { arguments: "" } },
    { function_call: { arguments: '{"city":' } },
    {
      content: "mixed",
      function_call: { arguments: '"Paris"}' },
      tool_calls: [{ index: 0, function: { arguments: "{}" } }],
    },
    {},
  ];

  const fake = createFakeFetch(
    sseResponse(
      deltas.map((delta) => ({
        id: "chatcmpl_timing",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o",
        choices: [{ index: 0, delta, finish_reason: null }],
      })),
    ),
  );

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hi" }],
    stream: true,
  });

  expect(await collectStream(stream)).toHaveLength(deltas.length);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);

  if (oldCore) expect(Symbol.for("telemetry.dev.outputChunkHistogram") in span).toBe(false);
  else {
    expect(span).toMatchObject({
      [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 3 },
    });
  }
});

test.each(["chat", "responses"])("%s timestamps precede telemetry mapping", async (operation) => {
  const spans = setupSpans();
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => now);

  const source = new Stream(async function* () {
    for (const [receivedAt, mappingMs] of [
      [100, 30],
      [240, 90],
    ] as const) {
      now = receivedAt;
      yield operation === "chat"
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
          };
    }
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create: async (_params: unknown) => source },
    embeddings: {},
  });

  const stream =
    operation === "chat"
      ? await client.chat.completions.create({ model: "gpt-4o", messages: [], stream: true })
      : await client.responses.create({ model: "gpt-4o", input: "Hi", stream: true });

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

test("chat streams preserve tool-call-only output with null content", async () => {
  const spans = setupSpans();
  const toolCall = {
    id: "call_stream",
    type: "function",
    function: { name: "get_weather", arguments: '{"location":"Paris"}' },
  };
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream_tool",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: 0,
                  id: "call_stream",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"loc' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: "chatcmpl_stream_tool",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [{ index: 0, function: { arguments: 'ation":"Paris"}' } }],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
    ]),
  );

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Weather in Paris?" }],
    stream: true,
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          parameters: { type: "object", properties: { location: { type: "string" } } },
        },
      },
    ],
  });
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: null, tool_calls: [toolCall] },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 1 },
  });
});

test("chat streams record finish reasons for every choice", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream_multi_finish",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "First" },
            finish_reason: "stop",
          },
          {
            index: 1,
            delta: { role: "assistant", content: "Second" },
            finish_reason: "length",
          },
        ],
      },
    ]),
  );

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Give two options" }],
    stream: true,
    n: 2,
  });
  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "First" },
    { role: "assistant", content: "Second" },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop", "length"]);
});

test("chat streams leave the request unchanged and capture no usage by default", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream_default",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: null }],
      },
      {
        id: "chatcmpl_stream_default",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { content: "lo" }, finish_reason: "stop" }],
      },
    ]),
  );

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });
  const visibleChunks = await collectStream(stream);
  const rawRequestBody: unknown = fake.requests[0]?.body;
  const requestBody = rawRequestBody as { stream?: boolean; stream_options?: object } | undefined;
  expect(requestBody).toMatchObject({ stream: true });
  expect(requestBody?.stream_options).toBeUndefined();
  expect(visibleChunks).toHaveLength(2);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Hello" },
  ]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBeUndefined();
});

test("chat stream helper routes through wrapped create and ends span", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream_helper",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hi" }, finish_reason: null }],
      },
      {
        id: "chatcmpl_stream_helper",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { content: "!" }, finish_reason: "stop" }],
      },
    ]),
  );

  const runner = clientWith(fake.fetch).chat.completions.stream({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Say hi" }],
  });
  const events = await collectStream(runner);

  expect(events.length).toBeGreaterThan(0);
  expect(fake.requests[0]?.body).toMatchObject({ stream: true });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.id"]).toBe("chatcmpl_stream_helper");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([{ role: "assistant", content: "Hi!" }]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
});

test("chat streams inject include_usage when opted in and hide only the synthetic usage chunk", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: null }],
      },
      {
        id: "chatcmpl_stream",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { content: "lo" }, finish_reason: "stop" }],
      },
      {
        id: "chatcmpl_stream",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [],
        usage: { prompt_tokens: 9, completion_tokens: 2, total_tokens: 11 },
      },
    ]),
  );

  const stream = await clientWith(fake.fetch, { injectStreamUsage: true }).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });
  const visibleChunks = await collectStream(stream);

  expect(fake.requests[0]?.body).toMatchObject({
    stream: true,
    stream_options: { include_usage: true },
  });
  expect(visibleChunks).toHaveLength(2);
  expect(visibleChunks.some(hasSyntheticUsageChunk)).toBe(false);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "Hello" },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(9);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(11);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.response.id"]).toBe("chatcmpl_stream");
  expect(span.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"]
      ? "number"
      : "other",
  ).toBe("number");
});

test("chat streams end once with partial output when the caller stops early", async () => {
  const spans = setupSpans();
  const counted = countedSseResponse([
    {
      id: "chatcmpl_abandoned",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: null }],
    },
    {
      id: "chatcmpl_abandoned",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { content: "lo" }, finish_reason: "stop" }],
    },
    {
      id: "chatcmpl_abandoned",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [],
      usage: { prompt_tokens: 9, completion_tokens: 2, total_tokens: 11 },
    },
  ]);
  const fake = createFakeFetch(counted.response);

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });

  let visibleChunks = 0;
  for await (const chunk of stream) {
    visibleChunks += 1;
    expect(chunk.choices[0]?.delta.content).toBe("Hel");
    break;
  }

  expect(visibleChunks).toBe(1);
  expect(counted.pulledChunks()).toBeLessThan(counted.chunkCount);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([{ role: "assistant", content: "Hel" }]);
  expect(span.attributes["gen_ai.response.id"]).toBe("chatcmpl_abandoned");
  expect(span.attributes["gen_ai.response.model"]).toBe("gpt-4o-2024-11-20");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBeUndefined();
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"]
      ? "number"
      : "other",
  ).toBe("number");
});

test("chat streams record one error span with partial output when the SSE body errors", async () => {
  const spans = setupSpans();
  const streamError = "stream exploded";
  const firstChunk = {
    id: "chatcmpl_stream_error",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4o-2024-11-20",
    choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: null }],
  };
  const fake = createFakeFetch(erroringSseResponse(firstChunk, streamError));

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Say hello" }],
    stream: true,
  });
  const visibleChunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = [];

  await expect(
    (async () => {
      for await (const chunk of stream) visibleChunks.push(chunk);
    })(),
  ).rejects.toBe(streamError);

  expect(visibleChunks).toHaveLength(1);
  expect(visibleChunks[0]?.choices[0]?.delta.content).toBe("Hel");
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([{ role: "assistant", content: "Hel" }]);
  expect(span.attributes["error.type"]).toBe("Error");
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.message"]).toBe("stream exploded");
});

test("chat streams preserve caller-requested usage chunks", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_stream_user_usage",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hi" }, finish_reason: "stop" }],
      },
      {
        id: "chatcmpl_stream_user_usage",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [],
        usage: { prompt_tokens: 6, completion_tokens: 1, total_tokens: 7 },
      },
    ]),
  );

  const stream = await clientWith(fake.fetch).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Greet" }],
    stream: true,
    stream_options: { include_usage: true },
  });
  const visibleChunks = await collectStream(stream);

  expect(fake.requests[0]?.body).toMatchObject({
    stream_options: { include_usage: true },
  });
  expect(visibleChunks).toHaveLength(2);
  expect(visibleChunks.some(hasSyntheticUsageChunk)).toBe(true);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(7);
});

test("responses create maps instructions to system instructions", async () => {
  const spans = setupSpans();
  const output = [
    { type: "message", role: "assistant", content: [{ type: "output_text", text: "No." }] },
  ];
  const fake = createFakeFetch(
    jsonResponse({
      id: "resp_1",
      object: "response",
      status: "completed",
      model: "gpt-4.1-2025-04-14",
      output,
      usage: {
        input_tokens: 11,
        output_tokens: 3,
        total_tokens: 14,
        input_tokens_details: { cached_tokens: 4 },
        output_tokens_details: { reasoning_tokens: 1 },
      },
    }),
  );

  await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    instructions: "You must never tell jokes",
    input: "Tell me a joke",
    temperature: 0.2,
    top_p: 0.8,
    max_output_tokens: 50,
  });

  const span = await exportedSpan(spans);
  expect(span.name).toBe("chat gpt-4.1");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.system_instructions"]).toBe("You must never tell jokes");
  expect(span.attributes["gen_ai.input.messages"]).toBe("Tell me a joke");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(output);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(1);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(50);
});

test("responses create failed body records error while completed body stays OK", async () => {
  const spans = setupSpans();
  const failedError = { code: "server_error", message: "model exploded" };
  const successOutput = [
    { type: "message", role: "assistant", content: [{ type: "output_text", text: "Recovered" }] },
  ];
  const fake = createFakeFetch(
    jsonResponse({
      id: "resp_failed",
      object: "response",
      status: "failed",
      model: "gpt-4.1-2025-04-14",
      output: [],
      error: failedError,
    }),
    jsonResponse({
      id: "resp_ok",
      object: "response",
      status: "completed",
      model: "gpt-4.1-2025-04-14",
      output: successOutput,
      usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
    }),
  );
  const client = clientWith(fake.fetch);

  await client.responses.create({ model: "gpt-4.1", input: "fail" });
  await client.responses.create({ model: "gpt-4.1", input: "succeed" });

  const finished = await finishedSpans(spans, 2);
  const failed = finished.find((span) => span.attributes["gen_ai.response.id"] === "resp_failed");
  const success = finished.find((span) => span.attributes["gen_ai.response.id"] === "resp_ok");
  expect(failed?.status.code).toBe(SPAN_STATUS_ERROR);
  expect(failed?.attributes["error.type"]).toBe("Error");
  const exception = failed?.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.type"]).toBe("Error");
  expect(exception?.attributes?.["exception.message"]).toBe(
    "response.failed: server_error: model exploded",
  );
  expect(exception?.attributes?.["log.severity_number"]).toBe(17);
  expect(success?.status.code).toBe(SPAN_STATUS_UNSET);
  expect(success?.events.some((event) => event.name === "exception")).toBe(false);
  expect(success?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(jsonAttr(success!, "gen_ai.output.messages")).toEqual(successOutput);
});

test.each([
  "response.custom_tool_call_input",
  "response.code_interpreter_call_code",
  "response.mcp_call_arguments",
  "response.audio.transcript",
])("%s deltas retain timing after completion and interruption", async (eventType) => {
  const metricExporter = new InMemoryMetricExporter(AggregationTemporality.DELTA);
  const spans = setupSpans(metricExporter);

  const outputEvents = [
    { type: `${eventType}.delta`, delta: "first", item_id: "item_1", output_index: 0 },
    { type: `${eventType}.delta`, delta: "", item_id: "item_1", output_index: 0 },
    { type: `${eventType}.delta`, delta: "second", item_id: "item_1", output_index: 0 },
    {
      type: `${eventType}.done`,
      input: "firstsecond",
      code: "firstsecond",
      arguments: "firstsecond",
      transcript: "firstsecond",
    },
  ];

  const completed = {
    type: "response.completed",
    response: { id: "resp_timing", model: "gpt-4.1", status: "completed", output: [] },
  };

  const fake = createFakeFetch(
    sseResponse([...outputEvents, completed]),
    countedSseResponse([...outputEvents, completed]).response,
  );

  const client = clientWith(fake.fetch);
  const stream = await client.responses.create({ model: "gpt-4.1", input: "Run", stream: true });
  expect(await collectStream(stream)).toEqual([...outputEvents, completed]);

  const interrupted = await client.responses.create({
    model: "gpt-4.1",
    input: "Run",
    stream: true,
  });

  const iterator = interrupted[Symbol.asyncIterator]();

  for (const event of outputEvents) expect((await iterator.next()).value).toEqual(event);
  await iterator.return?.();

  expect(await finishedSpans(spans, 2)).toHaveLength(2);

  const metrics = metricExporter
    .getMetrics()
    .flatMap((batch) => batch.scopeMetrics.flatMap((scope) => scope.metrics));

  const histogram = metrics.find(
    (metric) => metric.descriptor.name === "gen_ai.client.operation.time_per_output_chunk",
  );

  expect(histogram?.dataPoints).toHaveLength(2);
  expect(histogram?.dataPoints).toEqual([
    expect.objectContaining({ value: expect.objectContaining({ count: 1 }) }),
    expect.objectContaining({ value: expect.objectContaining({ count: 1 }) }),
  ]);
});

test("responses streams end from response.completed terminal event", async () => {
  const spans = setupSpans();
  const completed = {
    id: "resp_stream",
    object: "response",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Done" }] },
    ],
    usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 },
  };
  const fake = createFakeFetch(
    sseResponse([
      { type: "response.created", response: { id: "resp_stream", status: "in_progress" } },
      { type: "response.completed", response: completed },
    ]),
  );

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Finish",
    stream: true,
  });
  const events = await collectStream(stream);

  expect(events).toHaveLength(2);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_stream");
  expect(span.attributes["gen_ai.response.model"]).toBe("gpt-4.1-2025-04-14");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(completed.output);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(10);
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"]
      ? "number"
      : "other",
  ).toBe("number");
});

test("responses stream helper routes through wrapped create and ends span", async () => {
  const spans = setupSpans();
  const completed = {
    id: "resp_stream_helper",
    object: "response",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Done" }] },
    ],
    usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 },
  };
  const fake = createFakeFetch(
    sseResponse([
      {
        type: "response.created",
        response: { id: "resp_stream_helper", status: "in_progress", output: [] },
      },
      { type: "response.completed", response: completed },
    ]),
  );

  const runner = clientWith(fake.fetch).responses.stream({
    model: "gpt-4.1",
    input: "Finish",
  });
  const events = await collectStream(runner);

  expect(events.length).toBeGreaterThan(0);
  expect(fake.requests[0]?.body).toMatchObject({ stream: true });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_stream_helper");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(completed.output);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
});

test("responses stream helper traces response-id retrieval streams", async () => {
  const spans = setupSpans();
  const completed = {
    id: "resp_existing",
    object: "response",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Done" }] },
    ],
    usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 },
  };
  const fake = createFakeFetch(
    sseResponse([
      {
        type: "response.created",
        response: { id: "resp_existing", status: "in_progress", output: [] },
      },
      { type: "response.completed", response: completed },
    ]),
  );

  const runner = clientWith(fake.fetch).responses.stream({
    response_id: "resp_existing",
  });
  const events = await collectStream(runner);

  expect(events.length).toBeGreaterThan(0);
  expect(fake.requests[0]?.method).toBe("GET");
  expect(fake.requests[0]?.path).toBe("/v1/responses/resp_existing");
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_existing");
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(completed.output);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
});

test("responses streams end from terminal event before reading the stream sentinel", async () => {
  const spans = setupSpans();
  const completed = {
    id: "resp_stream_terminal",
    object: "response",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Done" }] },
    ],
    usage: { input_tokens: 8, output_tokens: 2, total_tokens: 10 },
  };
  const counted = countedSseResponse([
    { type: "response.created", response: { id: "resp_stream_terminal", status: "in_progress" } },
    { type: "response.completed", response: completed },
  ]);
  const fake = createFakeFetch(counted.response);

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Finish",
    stream: true,
  });
  const iterator = stream[Symbol.asyncIterator]();

  try {
    expect(await iterator.next()).toMatchObject({
      done: false,
      value: { type: "response.created" },
    });
    expect(await iterator.next()).toMatchObject({
      done: false,
      value: { type: "response.completed" },
    });
    expect(counted.pulledChunks()).toBeLessThan(counted.chunkCount);

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.response.id"]).toBe("resp_stream_terminal");
    expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(completed.output);
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
    expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(10);
  } finally {
    await iterator.return?.();
  }
});

test("responses streams record error status from bare error events", async () => {
  const spans = setupSpans();
  const errorEvent = { type: "error", code: "rate_limit_exceeded", message: "try later" };
  const fake = createFakeFetch(sseResponse([errorEvent]));

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Stream",
    stream: true,
  });
  const events = await collectStream(stream);

  expect(events).toEqual([errorEvent]);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("Error");
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.type"]).toBe("Error");
  expect(exception?.attributes?.["exception.message"]).toBe(
    "response.error: rate_limit_exceeded: try later",
  );
  expect(
    Number(span.attributes["gen_ai.response.time_to_first_chunk"]) ===
      span.attributes["gen_ai.response.time_to_first_chunk"]
      ? "number"
      : "other",
  ).toBe("number");
});

test("responses streams record error status from response.failed terminal events", async () => {
  const spans = setupSpans();
  const failedResponse = {
    id: "resp_stream_failed",
    object: "response",
    status: "failed",
    model: "gpt-4.1-2025-04-14",
    output: [],
    error: { code: "server_error", message: "stream failed" },
  };
  const failedEvent = { type: "response.failed", response: failedResponse };
  const fake = createFakeFetch(
    sseResponse([
      { type: "response.created", response: { id: "resp_stream_failed" } },
      failedEvent,
    ]),
  );

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Stream",
    stream: true,
  });
  const events = await collectStream(stream);

  expect(events).toEqual([
    { type: "response.created", response: { id: "resp_stream_failed" } },
    failedEvent,
  ]);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_stream_failed");
  expect(span.attributes["gen_ai.response.model"]).toBe("gpt-4.1-2025-04-14");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["failed"]);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([]);
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.type"]).toBe("Error");
  expect(exception?.attributes?.["exception.message"]).toBe(
    "response.failed: server_error: stream failed",
  );
});

test("responses streams record non-Error iterator failures", async () => {
  const spans = setupSpans();
  const streamError = "responses stream exploded";
  const firstEvent = {
    type: "response.created",
    response: { id: "resp_stream_error", status: "in_progress", output: [] },
  };
  const fake = createFakeFetch(erroringSseResponse(firstEvent, streamError));

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Stream",
    stream: true,
  });
  await expect(collectStream(stream)).rejects.toBe(streamError);

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("Error");
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.message"]).toBe(streamError);
});
test("embeddings map token usage without capturing embedding vectors as output", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      object: "list",
      model: "text-embedding-3-small",
      data: [],
      usage: { prompt_tokens: 6, total_tokens: 6 },
    }),
  );

  await clientWith(fake.fetch).embeddings.create({
    model: "text-embedding-3-small",
    input: "OpenTelemetry traces",
    encoding_format: "float",
  });

  const span = await exportedSpan(spans);
  expect(span.name).toBe("embeddings text-embedding-3-small");
  expect(span.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(span.attributes["gen_ai.request.model"]).toBe("text-embedding-3-small");
  expect(span.attributes["gen_ai.response.model"]).toBe("text-embedding-3-small");
  expect(span.attributes["gen_ai.input.messages"]).toBe("OpenTelemetry traces");
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("global instrumentation records responses and embeddings with wrapOpenAI parity attributes", async () => {
  const spans = setupSpans();
  instrumentOpenAI();
  const output = [
    { type: "message", role: "assistant", content: [{ type: "output_text", text: "No." }] },
  ];
  const fake = createFakeFetch(
    jsonResponse({
      id: "resp_global",
      object: "response",
      status: "completed",
      model: "gpt-4.1-2025-04-14",
      output,
      usage: {
        input_tokens: 11,
        output_tokens: 3,
        total_tokens: 14,
        input_tokens_details: { cached_tokens: 4 },
        output_tokens_details: { reasoning_tokens: 1 },
      },
    }),
    jsonResponse({
      object: "list",
      model: "text-embedding-3-small",
      data: [],
      usage: { prompt_tokens: 6, total_tokens: 6 },
    }),
  );
  const client = new OpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 });

  await client.responses.create({
    model: "gpt-4.1",
    instructions: "You must never tell jokes",
    input: "Tell me a joke",
    max_output_tokens: 50,
  });
  await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "OpenTelemetry traces",
    encoding_format: "float",
  });

  const finished = await finishedSpans(spans, 2);
  const responseSpan = finished.find((span) => span.name === "chat gpt-4.1");
  expect(responseSpan?.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(responseSpan?.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(responseSpan?.attributes["gen_ai.system_instructions"]).toBe("You must never tell jokes");
  expect(responseSpan?.attributes["gen_ai.input.messages"]).toBe("Tell me a joke");
  expect(jsonAttr(responseSpan!, "gen_ai.output.messages")).toEqual(output);
  expect(responseSpan?.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(responseSpan?.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(4);
  expect(responseSpan?.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(1);
  expect(responseSpan?.attributes["gen_ai.request.max_tokens"]).toBe(50);

  const embeddingSpan = finished.find((span) => span.name === "embeddings text-embedding-3-small");
  expect(embeddingSpan?.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(embeddingSpan?.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(embeddingSpan?.attributes["gen_ai.request.model"]).toBe("text-embedding-3-small");
  expect(embeddingSpan?.attributes["gen_ai.response.model"]).toBe("text-embedding-3-small");
  expect(embeddingSpan?.attributes["gen_ai.input.messages"]).toBe("OpenTelemetry traces");
  expect(embeddingSpan?.attributes["gen_ai.usage.input_tokens"]).toBe(6);
  expect(embeddingSpan?.attributes["gen_ai.usage.total_tokens"]).toBe(6);
  expect(embeddingSpan?.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("instrumentOpenAI threads injectStreamUsage into chat streams", async () => {
  const spans = setupSpans();
  instrumentOpenAI({ injectStreamUsage: true });
  const fake = createFakeFetch(
    sseResponse([
      {
        id: "chatcmpl_global_stream",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hi" }, finish_reason: "stop" }],
      },
      {
        id: "chatcmpl_global_stream",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o-2024-11-20",
        choices: [],
        usage: { prompt_tokens: 6, completion_tokens: 1, total_tokens: 7 },
      },
    ]),
  );
  const client = new OpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Greet" }],
    stream: true,
  });
  const visibleChunks = await collectStream(stream);

  expect(fake.requests[0]?.body).toMatchObject({
    stream_options: { include_usage: true },
  });
  expect(visibleChunks).toHaveLength(1);
  expect(visibleChunks.some(hasSyntheticUsageChunk)).toBe(false);
  expect((await exportedSpan(spans)).attributes["gen_ai.usage.total_tokens"]).toBe(7);
});

test("wrapOpenAI shadows global instrumentation and survives uninstrumentOpenAI", async () => {
  const spans = setupSpans();
  instrumentOpenAI();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_global_wrapped",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "First" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
    jsonResponse({
      id: "chatcmpl_after_uninstrument",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Second" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );
  const client = wrapOpenAI(new OpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }));

  await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Before" }],
  });
  const firstSpans = await finishedSpans(spans, 1);
  expect(firstSpans[0]?.attributes["gen_ai.response.id"]).toBe("chatcmpl_global_wrapped");

  uninstrumentOpenAI();
  await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "After" }],
  });

  const finished = await finishedSpans(spans, 2);
  expect(finished.map((span) => span.attributes["gen_ai.response.id"])).toEqual([
    "chatcmpl_global_wrapped",
    "chatcmpl_after_uninstrument",
  ]);
});

test("AzureOpenAI clients report Azure provider for wrapped and global instrumentation", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_azure_wrapped",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-azure",
      choices: [
        { index: 0, message: { role: "assistant", content: "Wrapped" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
    jsonResponse({
      id: "chatcmpl_azure_global",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-azure",
      choices: [
        { index: 0, message: { role: "assistant", content: "Global" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );
  const azureOptions = {
    apiKey: "test",
    apiVersion: "2024-10-21",
    endpoint: "https://example.azure.com",
    fetch: fake.fetch,
    maxRetries: 0,
  };

  await wrapOpenAI(new AzureOpenAI(azureOptions)).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Wrapped" }],
  });
  instrumentOpenAI();
  await new AzureOpenAI(azureOptions).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Global" }],
  });

  const finished = await finishedSpans(spans, 2);
  const wrapped = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_azure_wrapped",
  );
  const global = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_azure_global",
  );
  expect(wrapped?.attributes["gen_ai.provider.name"]).toBe("azure.ai.openai");
  expect(global?.attributes["gen_ai.provider.name"]).toBe("azure.ai.openai");
});

test("clients with an OpenRouter base URL report the openrouter provider", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_or_wrapped",
      object: "chat.completion",
      created: 1,
      model: "openai/gpt-4o-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: "Wrapped" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
    jsonResponse({
      id: "chatcmpl_or_global",
      object: "chat.completion",
      created: 1,
      model: "openai/gpt-4o-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: "Global" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );
  const openRouterOptions = {
    apiKey: "test",
    baseURL: "https://openrouter.ai/api/v1",
    fetch: fake.fetch,
    maxRetries: 0,
  };

  await wrapOpenAI(new OpenAI(openRouterOptions)).chat.completions.create({
    model: "openai/gpt-4o-mini",
    messages: [{ role: "user", content: "Wrapped" }],
  });
  instrumentOpenAI();
  await new OpenAI(openRouterOptions).chat.completions.create({
    model: "openai/gpt-4o-mini",
    messages: [{ role: "user", content: "Global" }],
  });

  const finished = await finishedSpans(spans, 2);
  const wrapped = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_or_wrapped",
  );
  const global = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_or_global",
  );
  expect(wrapped?.attributes["gen_ai.provider.name"]).toBe("openrouter");
  expect(global?.attributes["gen_ai.provider.name"]).toBe("openrouter");
});

test.each([
  ["subdomain", "https://api.openrouter.ai/api/v1", "openrouter"],
  ["fully qualified host", "https://openrouter.ai./api/v1", "openrouter"],
  ["fully qualified subdomain", "https://api.openrouter.ai./api/v1", "openrouter"],
  ["lookalike host", "https://openrouter.ai.example.com/api/v1", "openai"],
])(
  "clients with an OpenRouter %s report the expected provider",
  async (_case, baseURL, provider) => {
    const spans = setupSpans();
    const fake = createFakeFetch(
      jsonResponse({
        id: "chatcmpl_or_host",
        object: "chat.completion",
        created: 1,
        model: "openai/gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "Done" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
      }),
    );

    await wrapOpenAI(
      new OpenAI({ apiKey: "test", baseURL, fetch: fake.fetch, maxRetries: 0 }),
    ).chat.completions.create({
      model: "openai/gpt-4o-mini",
      messages: [{ role: "user", content: "Classify provider" }],
    });

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.provider.name"]).toBe(provider);
  },
);

test("wrapOpenAI resolves the provider from the current base URL for each operation", async () => {
  const spans = setupSpans();
  const completed = {
    id: "resp_dynamic_provider",
    object: "response",
    status: "completed",
    model: "openai/gpt-4o-mini",
    output: [],
    usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
  };
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_dynamic_provider",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-mini",
      choices: [
        { index: 0, message: { role: "assistant", content: "Done" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
    sseResponse([
      {
        type: "response.created",
        response: { id: "resp_dynamic_provider", status: "in_progress", output: [] },
      },
      { type: "response.completed", response: completed },
    ]),
  );
  const client = wrapOpenAI(
    new OpenAI({
      apiKey: "test",
      baseURL: "https://openrouter.ai/api/v1",
      fetch: fake.fetch,
      maxRetries: 0,
    }),
  );

  client.baseURL = "https://api.openai.com/v1";
  await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Classify at operation time" }],
  });
  client.baseURL = "https://openrouter.ai/api/v1";
  await collectStream(client.responses.stream({ response_id: "resp_dynamic_provider" }));

  const finished = await finishedSpans(spans, 2);
  const chat = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "chatcmpl_dynamic_provider",
  );
  const retrieve = finished.find(
    (span) => span.attributes["gen_ai.response.id"] === "resp_dynamic_provider",
  );
  expect(chat?.attributes["gen_ai.provider.name"]).toBe("openai");
  expect(retrieve?.attributes["gen_ai.provider.name"]).toBe("openrouter");
});

test("wrapped clients fail open when telemetry is not initialized", async () => {
  await shutdown();
  uninstrumentOpenAI();
  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_no_telemetry",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Still works" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );

  const response = await wrapOpenAI(
    new OpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }),
  ).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "No telemetry" }],
  });

  expect(fake.requests).toHaveLength(1);
  expect(response.choices[0]?.message.content).toBe("Still works");
});

test("instrumentOpenAI and wrapOpenAI are idempotent and uninstrumentOpenAI restores unpatched methods", async () => {
  const spans = setupSpans();
  instrumentOpenAI();
  instrumentOpenAI();
  const instrumented = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_global",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Instrumented" },
          finish_reason: "stop",
        },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );

  await new OpenAI({
    apiKey: "test",
    fetch: instrumented.fetch,
    maxRetries: 0,
  }).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hi" }],
  });
  await finishedSpans(spans, 1);

  uninstrumentOpenAI();
  const wrapped = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_wrapped",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Wrapped" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );
  await wrapOpenAI(
    wrapOpenAI(new OpenAI({ apiKey: "test", fetch: wrapped.fetch, maxRetries: 0 })),
  ).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hi wrapped" }],
  });
  await finishedSpans(spans, 2);

  const restored = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_restored",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [
        { index: 0, message: { role: "assistant", content: "Restored" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
    }),
  );

  await new OpenAI({
    apiKey: "test",
    fetch: restored.fetch,
    maxRetries: 0,
  }).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Hi again" }],
  });
  await flush();

  expect(instrumented.requests).toHaveLength(1);
  expect(wrapped.requests).toHaveLength(1);
  expect(restored.requests).toHaveLength(1);
  expect(spans.getFinishedSpans()).toHaveLength(2);
});
