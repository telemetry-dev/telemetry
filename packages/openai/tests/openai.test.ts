import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  AggregationTemporality,
  InMemoryMetricExporter,
  type PushMetricExporter,
} from "@opentelemetry/sdk-metrics";
import { flush, init, shutdown, type TelemetryOptions } from "@telemetry-dev/sdk";
import * as telemetrySdk from "@telemetry-dev/sdk";
import { EventEmitter } from "node:events";
import { createRequire } from "node:module";
import OpenAI, { AzureOpenAI } from "openai";
import { Stream } from "openai/core/streaming";
import { Completions } from "openai/resources/chat/completions/completions";
import type { ResponseCreateParamsStreaming } from "openai/resources/responses/responses";
import { VERSION } from "openai/version";
import { afterEach, expect, test, vi } from "vitest";

import packageJson from "../package.json" with { type: "json" };
import {
  instrumentOpenAI,
  type InstrumentOpenAIOptions,
  uninstrumentOpenAI,
  wrapOpenAI,
  wrapOpenAIRealtime,
} from "../src/index.ts";

const SPAN_STATUS_UNSET = 0;
const SPAN_STATUS_ERROR = 2;

test("runs the compatibility suite against the selected OpenAI major", () => {
  expect(VERSION).toBe(process.env.OPENAI_SDK_VERSION === "6" ? "6.45.0" : "7.18.0");
});

test("package exports resolve to built files inside the package", () => {
  expect(packageJson.exports["."].default).toBe("./dist/index.mjs");
  expect(packageJson.publishConfig.exports["."].types).toBe("./dist/index.d.mts");
  expect(packageJson.publishConfig.exports["."].import).toBe("./dist/index.mjs");
  expect(packageJson.publishConfig.exports["."].default).toBe("./dist/index.mjs");
});

test.each(["", "8"])("rejects unsupported OpenAI SDK selector %j", async (version) => {
  vi.stubEnv("OPENAI_SDK_VERSION", version);
  vi.resetModules();

  await expect(import("../vite.config.ts")).rejects.toThrow(
    `Unsupported OPENAI_SDK_VERSION: ${JSON.stringify(version)}`,
  );
});

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
    const bodyText = typeof init?.body === "string" ? init.body : undefined;
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

function setupSpans(
  metricExporter?: PushMetricExporter,
  options: Pick<TelemetryOptions, "captureOutput" | "mask" | "maxAttributeLength" | "onError"> = {},
): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "openai-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
      ...options,
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

interface ChatChoiceTestState {
  content: string;
  toolCalls: Map<number, { function?: { arguments?: string } }>;
}

function observeChatChoiceStates(): () => Map<number, ChatChoiceTestState> {
  const set = vi.spyOn(Map.prototype, "set");

  return () => {
    for (let index = set.mock.calls.length - 1; index >= 0; index -= 1) {
      const value = set.mock.calls[index]?.[1];

      if (
        isObject(value) &&
        typeof (value as { content?: unknown }).content === "string" &&
        (value as { toolCalls?: unknown }).toolCalls instanceof Map
      ) {
        return set.mock.contexts[index] as Map<number, ChatChoiceTestState>;
      }
    }

    throw new Error("chat choice state was not observed");
  };
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
  vi.unstubAllEnvs();
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
        prompt_tokens_details: {
          cached_tokens: 3,
          text_tokens: 6,
          image_tokens: 4,
          audio_tokens: 0,
          cached_tokens_details: { text_tokens: 2, image_tokens: 1 },
        },
        completion_tokens_details: { reasoning_tokens: 2, text_tokens: 3, audio_tokens: 2 },
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
  expect(span.attributes["gen_ai.usage.text.input_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.usage.image.input_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.audio.input_tokens"]).toBe(0);
  expect(span.attributes["gen_ai.usage.text.cache_read.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.image.cache_read.input_tokens"]).toBe(1);
  expect(span.attributes["gen_ai.usage.audio.cache_read.input_tokens"]).toBeUndefined();
  expect(span.attributes["gen_ai.usage.text.output_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.audio.output_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.image.output_tokens"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.7);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(64);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["END"]);
  expect(span.attributes["gen_ai.request.seed"]).toBe(7);
  expect(span.attributes["gen_ai.request.frequency_penalty"]).toBe(0.1);
  expect(span.attributes["gen_ai.request.presence_penalty"]).toBe(0.2);
});

test("chat completions return the provider result when response mapping fails", async () => {
  const errors: Error[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });
  const mappingError = new Error("mapping failed");

  const source = {
    get choices(): never {
      throw mappingError;
    },
  };

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: {},
    embeddings: {},
  });

  const result = await client.chat.completions.create({ model: "gpt-4o", messages: [] });

  expect(result).toBe(source);
  expect(errors).toEqual([mappingError]);
  expect((await exportedSpan(spans)).attributes["telemetry.dev.capture.truncated"]).toBe(true);
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

test("chunk timing preserves delivery", async () => {
  const spans = setupSpans();

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
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 3 },
  });
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

test("chat streams bound retained content without changing events or usage", async () => {
  const spans = setupSpans();
  const states = observeChatChoiceStates();

  const events = [
    {
      id: "chatcmpl_bounded_content",
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { role: "assistant", content: "kept" } }],
    },
    {
      id: "chatcmpl_bounded_content",
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { content: "x".repeat(70_000) }, finish_reason: "stop" }],
    },
    {
      id: "chatcmpl_bounded_content",
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { content: "not-retained" } }],
    },
    {
      choices: [],
      usage: { prompt_tokens: 9, completion_tokens: 2, total_tokens: 11 },
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
    stream_options: { include_usage: true },
  });

  const iterator = stream[Symbol.asyncIterator]();
  const delivered: unknown[] = [];

  for (let index = 0; index < events.length; index += 1) {
    delivered.push((await iterator.next()).value);

    if (index >= 1) expect(states().get(0)?.content).toBe("kept");
  }

  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  expect(delivered).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "kept" },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(11);
});

test.each(["chat", "responses", "transcription"] as const)(
  "%s streams keep telemetry on a core span without a capture policy",
  async (kind) => {
    const spans = setupSpans();
    const startSpan = telemetrySdk.startSpan;
    vi.spyOn(telemetrySdk, "startSpan").mockImplementation((...args) => {
      const legacySpan = { ...startSpan(...args) };
      Reflect.deleteProperty(legacySpan, "capturePolicy");
      Reflect.deleteProperty(legacySpan, "recordOutputChunk");
      Reflect.deleteProperty(legacySpan, "reportError");

      return legacySpan;
    });

    const output = [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "done" }] },
    ];

    const events = {
      chat: [
        {
          model: "gpt-4o-2024-11-20",
          choices: [{ index: 0, delta: { content: "done" }, finish_reason: "stop" }],
        },
      ],
      responses: [
        {
          type: "response.completed",
          response: { id: "resp_legacy", status: "completed", output },
        },
      ],
      transcription: [{ type: "transcript.text.done", text: "done" }],
    }[kind];

    const source = new Stream(async function* () {
      yield* events;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create: async (_params: unknown) => source },
      embeddings: { create() {} },
      audio: { transcriptions: { create: async (_params: unknown) => source } },
    });

    let stream: Stream<unknown>;

    if (kind === "chat")
      stream = await client.chat.completions.create({
        model: "gpt-4o",
        messages: [],
        stream: true,
      });
    else if (kind === "responses")
      stream = await client.responses.create({ model: "gpt-4.1", input: "x", stream: true });
    else
      stream = await client.audio!.transcriptions!.create({
        model: "gpt-4o-transcribe",
        stream: true,
      });

    expect(await collectStream(stream)).toEqual(events);
    const span = await exportedSpan(spans);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();

    if (kind === "chat") {
      expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
      expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
        { role: "assistant", content: "done" },
      ]);
    } else if (kind === "responses") {
      expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(output);
    } else {
      expect(span.attributes["gen_ai.output.messages"]).toBe("done");
    }
  },
);

test("chat streams retain output when the core span lacks chunk recording", async () => {
  const spans = setupSpans();
  const startSpan = telemetrySdk.startSpan;
  vi.spyOn(telemetrySdk, "startSpan").mockImplementation((...args) => {
    const legacySpan = { ...startSpan(...args) };
    Reflect.deleteProperty(legacySpan, "recordOutputChunk");

    return legacySpan;
  });

  const events = [
    { choices: [{ index: 0, delta: { content: "delivered" }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  expect(jsonAttr(await exportedSpan(spans), "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "delivered" },
  ]);
});

test("chat streams bound retained tool arguments", async () => {
  const spans = setupSpans();
  const states = observeChatChoiceStates();

  const events = [
    {
      choices: [
        {
          index: 0,
          delta: {
            role: "assistant",
            tool_calls: [
              {
                index: 0,
                id: "call_bounded",
                type: "function",
                function: { name: "lookup", arguments: '{"q":"' },
              },
            ],
          },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [{ index: 0, function: { arguments: "x".repeat(70_000) } }],
          },
          finish_reason: "tool_calls",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  const iterator = stream[Symbol.asyncIterator]();
  expect((await iterator.next()).value).toBe(events[0]);
  const retainedArguments = states().get(0)?.toolCalls.get(0)?.function?.arguments;
  expect(retainedArguments).toBe('{"q":"');

  expect((await iterator.next()).value).toBe(events[1]);
  expect(states().get(0)?.toolCalls.get(0)?.function?.arguments).toBe(retainedArguments);
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_bounded",
          type: "function",
          function: { name: "lookup", arguments: '{"q":"' },
        },
      ],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["tool_calls"]);
});

test("chat streams reconstruct legacy function calls and custom tools", async () => {
  const spans = setupSpans();

  const events = [
    {
      choices: [
        {
          index: 0,
          delta: {
            function_call: { name: "legacy_lookup", arguments: '{"city":' },
            tool_calls: [
              {
                index: 0,
                id: "custom_1",
                type: "custom",
                custom: { name: "code_exec", input: "print(" },
              },
            ],
          },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: {
            function_call: { arguments: '"Paris"}' },
            tool_calls: [{ index: 0, custom: { input: "42)" } }],
          },
          finish_reason: "stop",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: null,
      function_call: { name: "legacy_lookup", arguments: '{"city":"Paris"}' },
      tool_calls: [
        {
          id: "custom_1",
          type: "custom",
          custom: { name: "code_exec", input: "print(42)" },
        },
      ],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams retain the sanitized tool-call value they measured", async () => {
  const spans = setupSpans();
  let argumentReads = 0;

  const event = {
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: [
            {
              index: 0,
              id: 1n as never,
              type: "function",
              function: {
                name: "lookup",
                get arguments() {
                  argumentReads += 1;

                  return argumentReads <= 3 ? '{"q":"ok"}' : "x".repeat(70_000);
                },
              },
            },
          ],
        },
      },
    ],
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  const visible = await collectStream(stream);

  expect(visible).toHaveLength(1);
  expect(visible[0]).toBe(event);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          type: "function",
          function: { name: "lookup", arguments: '{"q":"ok"}' },
        },
      ],
    },
  ]);
  expect(argumentReads).toBe(1);
});

test("chat stream field getter failures fail open and mark capture truncated", async () => {
  const errors: Error[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });

  const toolCall = {
    index: 0,
    function: {
      get arguments() {
        throw new Error("cannot read arguments");
      },
    },
  };

  const event = {
    choices: [{ index: 0, delta: { tool_calls: [toolCall] } }],
  };

  const source = new Stream(async function* () {
    yield event;
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event, event]);
  expect(errors).toHaveLength(1);
  expect(errors[0]?.message).toBe("cannot read arguments");
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat stream top-level field failures fail open", async () => {
  const spans = setupSpans();

  const event = {
    get choices() {
      throw new Error("cannot read choices");
    },
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat stream synthetic-usage classification failures fail open", async () => {
  const spans = setupSpans();
  let usageReads = 0;

  const event = {
    choices: [],
    get usage() {
      usageReads += 1;

      if (usageReads === 1) return { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 };
      throw new Error("cannot classify usage chunk");
    },
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI(
    {
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create() {} },
      embeddings: { create() {} },
    },
    { injectStreamUsage: true },
  );

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams do not promote __proto__ data into tool-call fields", async () => {
  const spans = setupSpans();

  const inheritedToolCall = Object.create({
    index: 7,
    id: "inherited",
    type: "function",
    function: { name: "polluted", arguments: "{}" },
  });

  const event = {
    choices: [
      JSON.parse(
        '{"index":0,"delta":{"tool_calls":[{"__proto__":{"index":7,"id":"injected","type":"function","function":{"name":"polluted","arguments":"{}"}}}]}}',
      ),
      { index: 0, delta: { tool_calls: [inheritedToolCall] } },
    ],
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([{ role: "assistant" }]);
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("injected");
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("inherited");
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("polluted");
});

test("chat streams apply one budget across individually fitting content and refusal deltas", async () => {
  const spans = setupSpans();

  const events = [
    ...Array.from({ length: 80 }, () => ({
      choices: [
        {
          index: 0,
          delta: { content: "c".repeat(400), refusal: "r".repeat(400) },
        },
      ],
    })),
    {
      choices: [],
      usage: { prompt_tokens: 8, completion_tokens: 80, total_tokens: 88 },
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
    stream_options: { include_usage: true },
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const serializedOutput = String(span.attributes["gen_ai.output.messages"]);

  const output = jsonAttr<Array<{ content: string; refusal: string }>>(
    span,
    "gen_ai.output.messages",
  );

  expect(output[0]!.content.length + output[0]!.refusal.length).toBeLessThanOrEqual(48 * 1024);
  expect(serializedOutput.length).toBeLessThanOrEqual(48 * 1024);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(88);
});

test("chat streams replace the default role within an exact byte budget", async () => {
  const spans = setupSpans();
  const emptyOutput = JSON.stringify([{ role: "developer", content: "" }]);
  const retainedContent = "x".repeat(48 * 1024 - emptyOutput.length);

  const event = {
    choices: [
      {
        index: 0,
        delta: { role: "developer", content: retainedContent },
        finish_reason: "stop",
      },
    ],
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(String(span.attributes["gen_ai.output.messages"])).toHaveLength(48 * 1024);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "developer", content: retainedContent },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams refund the role separator before an exact-budget recovery", async () => {
  const expectedOutput = [{ role: "developer", content: "x" }];

  const spans = setupSpans(undefined, {
    maxAttributeLength: JSON.stringify(expectedOutput).length,
  });

  const events = [
    { choices: [{ index: 0, delta: { content: "x" } }] },
    { choices: [{ index: 0, delta: { role: "y".repeat(256) } }] },
    { choices: [{ index: 0, delta: { role: "developer" }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(expectedOutput);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams recover a sole rejected role in a later choice", async () => {
  const expectedOutput = [{ role: "assistant", content: "x".repeat(100) }, { role: "developer" }];

  const spans = setupSpans(undefined, {
    maxAttributeLength: JSON.stringify(expectedOutput).length,
  });

  const events = [
    {
      choices: [
        { index: 0, delta: { content: "x".repeat(100) }, finish_reason: "stop" },
        { index: 1, delta: { role: "y".repeat(256) } },
      ],
    },
    { choices: [{ index: 1, delta: { role: "developer" }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(expectedOutput);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams recover rejected role and tool-call scalar replacements", async () => {
  const spans = setupSpans(undefined, { maxAttributeLength: 160 });

  const events = [
    {
      choices: [
        {
          index: 0,
          delta: {
            role: "x".repeat(256),
            tool_calls: [{ index: 0, id: "y".repeat(256) }],
          },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: { role: "developer", tool_calls: [{ index: 0, id: "call_1" }] },
          finish_reason: "stop",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "developer", content: null, tool_calls: [{ id: "call_1" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams replace retained scalars after additive capture truncates", async () => {
  const spans = setupSpans();

  const events = [
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "old" }] } }] },
    { choices: [{ index: 0, delta: { content: "x".repeat(49_200) } }] },
    {
      choices: [
        {
          index: 0,
          delta: { tool_calls: [{ index: 0, id: "new" }] },
          finish_reason: "tool_calls",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: null, tool_calls: [{ id: "new" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams refund rejected scalar fields before later capture", async () => {
  const spans = setupSpans(undefined, { maxAttributeLength: 170 });

  const events = [
    {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "old",
                type: "function",
                function: { name: "old_name" },
              },
            ],
          },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "x".repeat(256),
                type: "y".repeat(256),
                function: { name: "z".repeat(256) },
              },
            ],
          },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "new",
                type: "function",
                function: { name: "new_name", arguments: "{}" },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "new",
          type: "function",
          function: { name: "new_name", arguments: "{}" },
        },
      ],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams reserve a rejected function shell only once", async () => {
  const expectedOutput = [
    {
      role: "assistant",
      content: null,
      tool_calls: [{ function: { name: "lookup", arguments: "{}" } }],
    },
  ];

  const spans = setupSpans(undefined, {
    maxAttributeLength: JSON.stringify(expectedOutput).length + 3,
  });

  const events = [
    {
      choices: [
        {
          index: 0,
          delta: { tool_calls: [{ index: 0, function: { name: "x".repeat(256) } }] },
        },
      ],
    },
    {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [{ index: 0, function: { name: "lookup", arguments: "{}" } }],
          },
          finish_reason: "tool_calls",
        },
      ],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(expectedOutput);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams replace null content after a tool call within an exact byte budget", async () => {
  const spans = setupSpans();

  const emptyOutput = JSON.stringify([
    { role: "assistant", content: "", tool_calls: [{ id: "call" }] },
  ]);

  const retainedContent = "x".repeat(48 * 1024 - emptyOutput.length - 1);

  const events = [
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call" }] } }] },
    { choices: [{ index: 0, delta: { content: retainedContent }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(String(span.attributes["gen_ai.output.messages"])).toHaveLength(48 * 1024 - 1);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: retainedContent, tool_calls: [{ id: "call" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams leave the configured attribute limit to the core", async () => {
  const spans = setupSpans(undefined, { maxAttributeLength: 128 });

  const event = {
    choices: [{ index: 0, delta: { content: "x".repeat(1_000) }, finish_reason: "stop" }],
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(output).toHaveLength(128);
  expect(output.endsWith("...[truncated]")).toBe(true);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test.each([
  [1_000, false, "東".repeat(400)],
  [1_000, true, "[redacted]"],
  [0, false, undefined],
] as const)(
  "chat stream cap applies after the mask, not to stream retention (cap %s, redact %s)",
  async (maxAttributeLength, redact, expected) => {
    const spans = setupSpans(undefined, {
      maxAttributeLength,
      mask: (value) => (redact ? [{ role: "assistant", content: "[redacted]" }] : value),
    });

    const content = redact ? "x".repeat(5_000) : "東".repeat(400);
    const event = { choices: [{ index: 0, delta: { content }, finish_reason: "stop" }] };

    const source = new Stream(async function* () {
      yield event;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create() {} },
      embeddings: { create() {} },
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [],
      stream: true,
    });

    expect(await collectStream(stream)).toEqual([event]);
    const span = await exportedSpan(spans);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();

    if (expected === undefined) expect(span.attributes["gen_ai.output.messages"] ?? "").toBe("");
    else
      expect(jsonAttr<Array<{ content: string }>>(span, "gen_ai.output.messages")[0]?.content).toBe(
        expected,
      );
  },
);

test.each([0, 64])(
  "chat stream finish reasons do not depend on a %s-character content limit",
  async (maxAttributeLength) => {
    const spans = setupSpans(undefined, { maxAttributeLength });
    const event = { choices: [{ index: 0, delta: { content: "Hi" }, finish_reason: "stop" }] };

    const source = new Stream(async function* () {
      yield event;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create() {} },
      embeddings: { create() {} },
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [],
      stream: true,
    });

    expect(await collectStream(stream)).toEqual([event]);
    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);

    if (maxAttributeLength === 0) {
      expect(span.attributes["gen_ai.output.messages"] ?? "").toBe("");
    } else {
      expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
        { role: "assistant", content: "Hi" },
      ]);
      expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
    }
  },
);

test("chat streams reclaim a replaced tool-call scalar at the byte limit", async () => {
  const spans = setupSpans();

  const emptyOutput = JSON.stringify([
    { role: "assistant", content: null, tool_calls: [{ id: "" }] },
  ]);

  const initialId = "x".repeat(48 * 1024 - emptyOutput.length - 1);
  const replacementId = "call_replaced";
  const reclaimedContent = "y".repeat(initialId.length - replacementId.length + 2);

  const events = [
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: initialId }] } }] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: replacementId }] } }] },
    { choices: [{ index: 0, delta: { content: reclaimedContent }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    {
      role: "assistant",
      content: reclaimedContent,
      tool_calls: [{ id: replacementId }],
    },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams include the assembled output wrapper in the byte budget", async () => {
  const spans = setupSpans();
  const emptyOutput = JSON.stringify([{ role: "assistant", content: "" }]);
  const retainedContent = "x".repeat(48 * 1024 - emptyOutput.length);

  const events = [
    { choices: [{ index: 0, delta: { content: retainedContent } }] },
    { choices: [{ index: 0, delta: { content: "y" } }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const serializedOutput = String(span.attributes["gen_ai.output.messages"]);
  expect(serializedOutput).toHaveLength(48 * 1024);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: retainedContent },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams account for separators while extending later choices", async () => {
  const spans = setupSpans();
  const emptyOutput = JSON.stringify([{ role: "assistant" }, { role: "assistant", content: "" }]);
  const retainedContent = "x".repeat(48 * 1024 - emptyOutput.length);

  const events = [
    {
      choices: [
        { index: 0, delta: {} },
        { index: 1, delta: { content: retainedContent } },
      ],
    },
    { choices: [{ index: 1, delta: { content: "y" } }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const serializedOutput = String(span.attributes["gen_ai.output.messages"]);
  expect(serializedOutput).toHaveLength(48 * 1024);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant" },
    { role: "assistant", content: retainedContent },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams incrementally account for many tiny deltas", async () => {
  const spans = setupSpans();
  const states = observeChatChoiceStates();
  const chunkCount = 60_000;
  const emptyOutput = JSON.stringify([{ role: "assistant", content: "" }]);
  const retainedContent = "x".repeat(48 * 1024 - emptyOutput.length);

  const source = new Stream(async function* () {
    for (let index = 0; index < chunkCount; index += 1) {
      yield { choices: [{ index: 0, delta: { content: "x" } }] };
    }
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  let visibleChunks = 0;
  let peakRetainedLength = 0;

  for await (const _chunk of stream) {
    visibleChunks += 1;
    const retainedLength = states().get(0)?.content.length ?? 0;
    peakRetainedLength = Math.max(peakRetainedLength, retainedLength);
  }

  expect(visibleChunks).toBe(chunkCount);
  expect(peakRetainedLength).toBe(retainedContent.length);
  const span = await exportedSpan(spans);
  const serializedOutput = String(span.attributes["gen_ai.output.messages"]);
  expect(serializedOutput).toBe(JSON.stringify([{ role: "assistant", content: retainedContent }]));
  expect(serializedOutput).toHaveLength(48 * 1024);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams omit a bounded prefix when a mask cannot inspect the complete output", async () => {
  const spans = setupSpans(undefined, {
    mask: (value) => (JSON.stringify(value).includes("SECRET") ? "[REDACTED]" : value),
  });

  const events = [
    { choices: [{ index: 0, delta: { content: "x".repeat(49_200) } }] },
    { choices: [{ index: 0, delta: { content: "SECRET" }, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams omit masked output without a terminal finish reason", async () => {
  const spans = setupSpans(undefined, { mask: (value) => value });
  const event = { choices: [{ index: 0, delta: { content: "partial" } }] };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams mark clean EOF without a finish reason incomplete when output capture is disabled", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });
  const event = { choices: [{ index: 0, delta: { content: "partial" } }] };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams bound finish reasons independently of output capture", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });

  const event = {
    choices: Array.from({ length: 250 }, (_, index) => ({
      index,
      delta: {},
      finish_reason: `reason-${index}-${'"🙂'.repeat(128)}`,
    })),
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  const visible = await collectStream(stream);

  expect(visible).toHaveLength(1);
  expect(visible[0]).toBe(event);
  const span = await exportedSpan(spans);
  const finishReasons = span.attributes["gen_ai.response.finish_reasons"];
  expect(Array.isArray(finishReasons)).toBe(true);
  expect((finishReasons as unknown[]).length).toBeGreaterThan(0);
  expect((finishReasons as string[]).length).toBeLessThanOrEqual(333);
  expect(Buffer.byteLength(JSON.stringify(finishReasons), "utf8")).toBeLessThanOrEqual(48 * 1024);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("finish-reason truncation does not suppress captured output", async () => {
  const spans = setupSpans();

  const event = {
    choices: [
      {
        index: 0,
        delta: { content: "kept" },
        finish_reason: "x".repeat(70_000),
      },
    ],
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual([event]);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "kept" },
  ]);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams replace a retained finish reason within the shared limit", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });

  const events = [
    {
      choices: [{ index: 0, delta: {}, finish_reason: "x".repeat(49_100) }],
    },
    {
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams invalidate a retained finish reason when its replacement is rejected", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });

  const events = [
    { choices: [{ index: 0, delta: {}, finish_reason: "x".repeat(60) }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "x".repeat(49_200) }] },
    { choices: [{ index: 1, delta: {}, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams reclaim finish-reason items when replacing a choice", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });

  const events = [
    {
      choices: Array.from({ length: 333 }, (_, index) => ({
        index,
        delta: {},
        finish_reason: `reason-${index}`,
      })),
    },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const finishReasons = span.attributes["gen_ai.response.finish_reasons"] as string[];
  expect(finishReasons).toHaveLength(333);
  expect(finishReasons[0]).toBe("stop");
  expect(finishReasons.at(-1)).toBe("reason-332");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("chat streams bound retained choice states", async () => {
  const spans = setupSpans();
  const states = observeChatChoiceStates();

  const event = {
    choices: Array.from({ length: 500 }, (_, index) => ({
      index,
      delta: {},
    })),
  };

  const source = new Stream(async function* () {
    yield event;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  const iterator = stream[Symbol.asyncIterator]();
  expect((await iterator.next()).value).toBe(event);
  expect(states().size).toBe(499);
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  const span = await exportedSpan(spans);
  const output = jsonAttr<unknown[]>(span, "gen_ai.output.messages");
  expect(output).toHaveLength(499);
  expect(String(span.attributes["gen_ai.output.messages"]).length).toBeLessThanOrEqual(48 * 1024);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("chat streams cap per-chunk choice and tool-call inspection", async () => {
  const errors: unknown[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });
  let choiceReads = 0;
  let toolCallReads = 0;

  const boundedArray = <T>(items: T[]) =>
    new Proxy(items, {
      get(target, property) {
        if (typeof property === "string" && /^\d+$/.test(property) && Number(property) >= 1_000)
          throw new Error("inspected beyond the per-chunk limit");

        return target[property as keyof T[]];
      },
    });

  const choiceEvent = {
    choices: boundedArray(
      Array.from({ length: 5_000 }, (_, index) => ({
        index,
        get delta() {
          choiceReads += 1;

          return index === 1_000 ? { content: "beyond-limit" } : {};
        },
      })),
    ),
  };

  const toolEvent = {
    choices: [
      {
        index: 0,
        delta: {
          tool_calls: boundedArray(
            Array.from({ length: 5_000 }, (_, index) => ({
              index,
              function: index === 1_000 ? { arguments: "beyond-limit" } : undefined,
              get id() {
                toolCallReads += 1;

                return `call-${index}`;
              },
            })),
          ),
        },
      },
    ],
  };

  const source = new Stream(async function* () {
    yield choiceEvent;
    yield toolEvent;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  const delivered = await collectStream(stream);
  expect(delivered[0]).toBe(choiceEvent);
  expect(delivered[1]).toBe(toolEvent);
  expect(errors).toEqual([]);
  expect(choiceReads).toBe(1_000);
  expect(toolCallReads).toBe(1_000);
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 1 },
  });
});

test("chat streams skip retained output work when capture is disabled", async () => {
  const spans = setupSpans(undefined, { captureOutput: false });
  let roleReads = 0;

  const events = [
    {
      choices: Array.from({ length: 1_500 }, (_, index) => ({
        index,
        delta: {
          get role() {
            roleReads += 1;

            return "assistant";
          },
          content: "x",
        },
      })),
    },
    {
      choices: [],
      usage: { prompt_tokens: 6, completion_tokens: 1, total_tokens: 7 },
    },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
    stream_options: { include_usage: true },
  });

  const visible = await collectStream(stream);
  expect(visible).toHaveLength(2);
  expect(visible[0]).toBe(events[0]);
  expect(visible[1]).toBe(events[1]);
  expect(roleReads).toBe(0);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(7);
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

test.each([true, false])(
  "nonterminal chat stream close marks capture incomplete with output capture=%s",
  async (captureOutput) => {
    const spans = setupSpans(undefined, { captureOutput });

    const event = {
      choices: [{ index: 0, delta: { role: "assistant", content: "partial" } }],
    };

    const source = new Stream(async function* () {
      yield event;
      yield { choices: [{ index: 0, delta: { content: "unread" }, finish_reason: "stop" }] };
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create() {} },
      embeddings: { create() {} },
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [],
      stream: true,
    });

    for await (const chunk of stream) {
      expect(chunk).toBe(event);
      break;
    }

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);

    if (captureOutput)
      expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
        { role: "assistant", content: "partial" },
      ]);
    else expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  },
);

test("nonterminal chat stream errors omit finish reasons and mark capture incomplete", async () => {
  const spans = setupSpans();
  const failure = new Error("stream failed before terminal chunk");

  const event = {
    choices: [{ index: 0, delta: { role: "assistant", content: "partial" } }],
  };

  const source = new Stream(async function* () {
    yield event;
    throw failure;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create: async (_params: unknown) => source } },
    responses: { create() {} },
    embeddings: { create() {} },
  });

  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [],
    stream: true,
  });

  await expect(collectStream(stream)).rejects.toBe(failure);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", content: "partial" },
  ]);
});

test.each([false, true])(
  "a post-finish content filter chunk keeps a complete chat stream complete (masked: %s)",
  async (masked) => {
    const spans = setupSpans(undefined, masked ? { mask: (value) => value } : {});

    const events = [
      { choices: [{ index: 0, delta: { role: "assistant", content: "Hi" }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
      {
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: null,
            content_filter_offsets: { check_offset: 0, start_offset: 0, end_offset: 2 },
            content_filter_results: { hate: { filtered: false, severity: "safe" } },
          },
        ],
      },
    ];

    const source = new Stream(async function* () {
      yield* events;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create: async (_params: unknown) => source } },
      responses: { create() {} },
      embeddings: { create() {} },
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [],
      stream: true,
    });

    expect(await collectStream(stream)).toEqual(events);
    const span = await exportedSpan(spans);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
    expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
      { role: "assistant", content: "Hi" },
    ]);
  },
);

test("chat streams end once with partial output when the caller stops early", async () => {
  const spans = setupSpans();

  const counted = countedSseResponse([
    {
      id: "chatcmpl_abandoned",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o-2024-11-20",
      choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: "stop" }],
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
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
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
    choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: "stop" }],
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
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
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
    {
      id: "msg_1",
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "No." }],
    },
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

test("responses recursively omit binary media from request and response capture", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      id: "resp_media",
      status: "completed",
      output: [
        { type: "output_audio", data: "OUTPUT_AUDIO", transcript: "hello" },
        { type: "image_generation_call", result: "IMAGE_RESULT", status: "completed" },
      ],
    }),
    sseResponse([
      {
        type: "response.completed",
        response: {
          id: "resp_media_stream",
          status: "completed",
          output: [{ type: "image_generation_call", result: "STREAM_IMAGE", status: "completed" }],
        },
      },
    ]),
  );

  const client = clientWith(fake.fetch);

  const input = [
    {
      role: "user",
      content: [
        { type: "input_audio", input_audio: { data: "INPUT_AUDIO", format: "wav" } },
        { type: "input_image", image_url: "data:image/png;base64,INPUT_IMAGE" },
        { type: "input_image", image_url: "https://example.com/image.png" },
        { type: "input_file", file_data: "INPUT_FILE", filename: "report.pdf" },
      ],
    },
  ];

  await client.responses.create({ model: "gpt-4.1", input } as never);

  const stream = await client.responses.create({
    model: "gpt-4.1",
    input,
    stream: true,
  } as ResponseCreateParamsStreaming);

  await collectStream(stream);

  for (const span of await finishedSpans(spans, 2)) {
    const captured = JSON.stringify(span.attributes);
    expect(captured).not.toMatch(
      /INPUT_AUDIO|INPUT_IMAGE|INPUT_FILE|OUTPUT_AUDIO|IMAGE_RESULT|STREAM_IMAGE/,
    );
    expect(captured).toContain("https://example.com/image.png");
  }

  expect(JSON.stringify((await finishedSpans(spans, 2))[0]!.attributes)).toContain("hello");
});

test("chat completions recursively omit binary media from request and response capture", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_media",
      model: "gpt-4o-audio-preview",
      choices: [
        {
          finish_reason: "stop",
          message: {
            role: "assistant",
            content: "spoken reply",
            audio: { id: "audio_1", data: "OUTPUT_AUDIO", transcript: "spoken reply" },
          },
        },
      ],
    }),
  );

  const client = clientWith(fake.fetch);

  await client.chat.completions.create({
    model: "gpt-4o-audio-preview",
    modalities: ["text", "audio"],
    audio: { format: "wav", voice: "alloy" },
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "Describe this" },
          { type: "image_url", image_url: { url: "data:image/png;base64,INPUT_IMAGE" } },
          { type: "image_url", image_url: { url: "https://example.com/image.png" } },
          { type: "input_audio", input_audio: { data: "INPUT_AUDIO", format: "wav" } },
        ],
      },
    ],
  } as never);

  const captured = JSON.stringify((await exportedSpan(spans)).attributes);
  expect(captured).not.toMatch(/INPUT_IMAGE|INPUT_AUDIO|OUTPUT_AUDIO/);
  expect(captured).toContain("https://example.com/image.png");
  expect(captured).toContain("spoken reply");
});

test("responses create failed body records error while completed body stays OK", async () => {
  const spans = setupSpans();
  const failedError = { code: "server_error", message: "model exploded" };

  const successOutput = [
    {
      id: "msg_recovered",
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "Recovered" }],
    },
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
  "response.shell_call_command",
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
      {
        id: "msg_stream",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Done" }],
      },
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

test("responses streams retain the last fitting output when the terminal snapshot truncates", async () => {
  const spans = setupSpans();

  const retainedOutput = [
    {
      id: "msg_retained",
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "Useful partial output" }],
    },
  ];

  const completed = {
    id: "resp_truncated_terminal",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      {
        id: "msg_terminal",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "x".repeat(70_000) }],
      },
    ],
    usage: { input_tokens: 8, output_tokens: 20, total_tokens: 28 },
  };

  const fake = createFakeFetch(
    sseResponse([
      {
        type: "response.in_progress",
        response: {
          id: "resp_truncated_terminal",
          status: "in_progress",
          model: "gpt-4.1-2025-04-14",
          output: retainedOutput,
        },
      },
      { type: "response.completed", response: completed },
    ]),
  );

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Finish",
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(retainedOutput);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(28);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("responses streams omit stale output when a masked terminal snapshot truncates", async () => {
  const spans = setupSpans(undefined, {
    mask: (value) => (JSON.stringify(value).includes("SECRET") ? "[REDACTED]" : value),
  });

  const retained = {
    id: "resp_masked_terminal",
    status: "in_progress",
    output: [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "public" }] },
    ],
  };

  const terminal = {
    id: "resp_masked_terminal",
    status: "completed",
    output: [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: `SECRET${"x".repeat(70_000)}` }],
      },
    ],
  };

  const source = new Stream(async function* () {
    yield { type: "response.in_progress", response: retained };
    yield { type: "response.completed", response: terminal };
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create: async (_params: unknown) => source },
    embeddings: { create() {} },
  });

  const stream = await client.responses.create({ model: "gpt-4.1", input: "Finish", stream: true });
  expect(await collectStream(stream)).toHaveLength(2);

  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("responses streams leave the configured attribute limit to the core", async () => {
  const spans = setupSpans(undefined, { maxAttributeLength: 128 });

  const terminal = {
    id: "resp_small_limit",
    status: "completed",
    output: [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "x".repeat(1_000) }],
      },
    ],
  };

  const source = new Stream(async function* () {
    yield { type: "response.completed", response: terminal };
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create: async (_params: unknown) => source },
    embeddings: { create() {} },
  });

  const stream = await client.responses.create({ model: "gpt-4.1", input: "Finish", stream: true });
  expect(await collectStream(stream)).toHaveLength(1);

  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(output).toHaveLength(128);
  expect(output.endsWith("...[truncated]")).toBe(true);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test.each(["eof", "early-return", "error-event", "throw"] as const)(
  "responses streams omit masked partial output after %s",
  async (mode) => {
    const spans = setupSpans(undefined, { mask: (value) => value });

    const retained = {
      type: "response.in_progress",
      response: {
        id: "resp_incomplete",
        status: "in_progress",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "partial" }],
          },
        ],
      },
    };

    const streamError = new Error("responses stream interrupted");

    const source = new Stream(async function* () {
      yield retained;

      if (mode === "error-event") {
        yield { type: "error", code: "stream_error", message: "interrupted" };
      } else if (mode === "throw") {
        throw streamError;
      } else if (mode === "early-return") {
        yield {
          type: "response.completed",
          response: { ...retained.response, status: "completed" },
        };
      }
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create: async (_params: unknown) => source },
      embeddings: { create() {} },
    });

    const stream = await client.responses.create({
      model: "gpt-4.1",
      input: "Finish",
      stream: true,
    });

    if (mode === "early-return") {
      const iterator = stream[Symbol.asyncIterator]();
      expect((await iterator.next()).value).toBe(retained);
      await iterator.return?.();
    } else if (mode === "throw") {
      await expect(collectStream(stream)).rejects.toBe(streamError);
    } else {
      await collectStream(stream);
    }

    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test("responses streams preserve provider events when response mapping fails", async () => {
  const errors: unknown[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });
  const mappingError = new Error("response mapping failed");

  const retained = {
    type: "response.in_progress",
    response: {
      id: "resp_mapping_failure",
      status: "in_progress",
      output: [
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "public" }] },
      ],
    },
  };

  const failed = {
    type: "response.completed",
    response: {
      id: "resp_mapping_failure",
      status: "completed",
      get output() {
        throw mappingError;
      },
    },
  };

  const source = new Stream(async function* () {
    yield retained;
    yield failed;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create: async (_params: unknown) => source },
    embeddings: { create() {} },
  });

  const stream = await client.responses.create({ model: "gpt-4.1", input: "Finish", stream: true });
  const delivered = await collectStream(stream);

  expect(delivered[0]).toBe(retained);
  expect(delivered[1]).toBe(failed);
  expect(errors).toEqual([mappingError]);
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(retained.response.output);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test.each([false, true])(
  "a complete responses stream is not flagged incomplete (masked: %s)",
  async (masked) => {
    const spans = setupSpans(undefined, masked ? { mask: (value) => value } : {});

    const output = [
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "done" }] },
    ];

    const events = [
      {
        type: "response.created",
        response: { id: "resp_done", status: "in_progress", output: [] },
      },
      { type: "response.output_text.delta", delta: "done" },
      { type: "response.completed", response: { id: "resp_done", status: "completed", output } },
    ];

    const source = new Stream(async function* () {
      yield* events;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create: async (_params: unknown) => source },
      embeddings: { create() {} },
    });

    const stream = await client.responses.create({ model: "gpt-4.1", input: "x", stream: true });

    expect(await collectStream(stream)).toEqual(events);
    const span = await exportedSpan(spans);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
    expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(output);
  },
);

test("responses streams keep the provider failure when mapping a failed event throws", async () => {
  const errors: unknown[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });
  const mappingError = new Error("response mapping failed");

  const failed = {
    type: "response.failed",
    response: {
      id: "resp_failed_mapping",
      status: "failed",
      error: { code: "server_error", message: "boom" },
      get output() {
        throw mappingError;
      },
    },
  };

  const source = new Stream(async function* () {
    yield failed;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create: async (_params: unknown) => source },
    embeddings: { create() {} },
  });

  const stream = await client.responses.create({ model: "gpt-4.1", input: "Fail", stream: true });

  expect(await collectStream(stream)).toEqual([failed]);
  expect(errors).toEqual([mappingError]);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.message"]).toBe("response.failed: server_error: boom");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("responses streams keep the bounded terminal output over an empty created snapshot", async () => {
  const spans = setupSpans();

  const toolCall = {
    id: "fc_1",
    type: "function_call",
    call_id: "call_1",
    name: "lookup",
    arguments: '{"q":"a"}',
    status: "completed",
  };

  const fake = createFakeFetch(
    sseResponse([
      {
        type: "response.created",
        response: {
          id: "resp_empty_created",
          status: "in_progress",
          model: "gpt-4.1-2025-04-14",
          output: [],
        },
      },
      {
        type: "response.completed",
        response: {
          id: "resp_empty_created",
          status: "completed",
          model: "gpt-4.1-2025-04-14",
          output: [
            toolCall,
            {
              id: "msg_terminal",
              type: "message",
              role: "assistant",
              content: [{ type: "output_text", text: "x".repeat(70_000) }],
            },
          ],
          usage: { input_tokens: 8, output_tokens: 20, total_tokens: 28 },
        },
      },
    ]),
  );

  const stream = await clientWith(fake.fetch).responses.create({
    model: "gpt-4.1",
    input: "Finish",
    stream: true,
  });

  await collectStream(stream);

  const span = await exportedSpan(spans);
  const output = jsonAttr(span, "gen_ai.output.messages") as unknown[];

  expect(output[0]).toMatchObject({ type: "function_call", name: "lookup" });
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("responses stream helper routes through wrapped create and ends span", async () => {
  const spans = setupSpans();

  const completed = {
    id: "resp_stream_helper",
    object: "response",
    status: "completed",
    model: "gpt-4.1-2025-04-14",
    output: [
      {
        id: "msg_stream_helper",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Done" }],
      },
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
      {
        id: "msg_existing",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Done" }],
      },
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
      {
        id: "msg_stream_terminal",
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "Done" }],
      },
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
    {
      id: "msg_global",
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "No." }],
    },
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

test("wrapOpenAI supports require-created streams and Azure clients", async () => {
  const spans = setupSpans();
  const require = createRequire(import.meta.url);
  const packageName = process.env.OPENAI_SDK_VERSION === "6" ? "openai-v6" : "openai";
  const required = require(packageName) as typeof import("openai");
  const RequiredOpenAI = required.default;
  const RequiredAzureOpenAI = required.AzureOpenAI;

  const streamEvents = [
    {
      id: "chatcmpl_cjs_stream",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o",
      choices: [{ index: 0, delta: { role: "assistant", content: "CJS" } }],
    },
    {
      id: "chatcmpl_cjs_stream",
      object: "chat.completion.chunk",
      created: 1,
      model: "gpt-4o",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    },
  ];

  const fake = createFakeFetch(
    sseResponse(streamEvents),
    jsonResponse({
      id: "chatcmpl_cjs_azure",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [{ index: 0, message: { role: "assistant", content: "Azure" } }],
    }),
  );

  const streamClient = wrapOpenAI(
    new RequiredOpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 }),
  );

  const stream = await streamClient.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "CJS" }],
    stream: true,
  });

  expect(stream).not.toBeInstanceOf(Stream);
  expect("tee" in stream).toBe(true);
  expect("toReadableStream" in stream).toBe(true);
  const [left, right] = stream.tee();
  expect(left.controller).toBe(stream.controller);
  const [leftEvents, rightEvents] = await Promise.all([collectStream(left), collectStream(right)]);
  expect(leftEvents).toEqual(streamEvents);
  expect(rightEvents).toEqual(streamEvents);
  stream.controller.abort();
  expect(stream.controller.signal.aborted).toBe(true);

  await wrapOpenAI(
    new RequiredAzureOpenAI({
      apiKey: "test",
      apiVersion: "2024-10-21",
      endpoint: "https://example.azure.com",
      fetch: fake.fetch,
      maxRetries: 0,
    }),
  ).chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "Azure" }],
  });

  const finished = await finishedSpans(spans, 2);
  expect(finished.map((span) => span.attributes["gen_ai.provider.name"])).toEqual([
    "openai",
    "azure.ai.openai",
  ]);
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

test.each([
  ["https://api.groq.com/openai/v1", "groq"],
  ["https://edge.api.x.ai/v1", "x_ai"],
  ["https://api.deepseek.com/v1", "deepseek"],
  ["https://api.together.xyz/v1", "together_ai"],
  ["https://api.fireworks.ai/inference/v1", "fireworks_ai"],
  ["https://api.groq.com.example.test/v1", "openai"],
])("attributes compatible provider host %s", async (baseURL, provider) => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      id: "chatcmpl_provider",
      model: "model",
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    }),
  );

  await wrapOpenAI(
    new OpenAI({ apiKey: "test", baseURL, fetch: fake.fetch }),
  ).chat.completions.create({
    model: "model",
    messages: [{ role: "user", content: "hi" }],
  });

  expect((await exportedSpan(spans)).attributes["gen_ai.provider.name"]).toBe(provider);
});

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

test("uninstrumentOpenAI preserves a later prototype owner", () => {
  const original = Object.getOwnPropertyDescriptor(Completions.prototype, "create");
  const laterOwner = vi.fn();

  if (!original) throw new Error("Completions.create descriptor is missing");

  instrumentOpenAI();
  Object.defineProperty(Completions.prototype, "create", { ...original, value: laterOwner });
  uninstrumentOpenAI();

  expect(Object.getOwnPropertyDescriptor(Completions.prototype, "create")?.value).toBe(laterOwner);
  Object.defineProperty(Completions.prototype, "create", original);
});

test("images generate maps modality usage without capturing image bytes", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      created: 1,
      data: [{ b64_json: "secret-image-bytes", revised_prompt: "A revised prompt" }],
      usage: {
        input_tokens: 8,
        output_tokens: 12,
        total_tokens: 20,
        input_tokens_details: {
          text_tokens: 5,
          image_tokens: 3,
          cached_tokens: 3,
          cached_tokens_details: { text_tokens: 2, image_tokens: 1, audio_tokens: 0 },
        },
        output_tokens_details: { image_tokens: 12 },
      },
    }),
  );

  await clientWith(fake.fetch).images.generate({ model: "gpt-image-1", prompt: "an otter" });

  const span = await exportedSpan(spans);
  expect(span.name).toBe("image gpt-image-1");
  expect(span.attributes["gen_ai.operation.name"]).toBe("generate_content");
  expect(span.attributes["gen_ai.output.type"]).toBe("image");
  expect(span.attributes["gen_ai.usage.text.input_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.image.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.text.cache_read.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.image.cache_read.input_tokens"]).toBe(1);
  expect(span.attributes["gen_ai.usage.audio.cache_read.input_tokens"]).toBe(0);
  expect(span.attributes["gen_ai.usage.image.output_tokens"]).toBe(12);
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("secret-image-bytes");
});

test.each([true, false])(
  "image streams are flagged only without a terminal event (completed: %s)",
  async (completed) => {
    const spans = setupSpans();

    const events = [
      { type: "image_generation.partial_image", b64_json: "QQ==", partial_image_index: 0 },
      ...(completed
        ? [{ type: "image_generation.completed", b64_json: "QQ==", usage: { output_tokens: 9 } }]
        : []),
    ];

    const source = new Stream(async function* () {
      yield* events;
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create() {} },
      embeddings: { create() {} },
      images: { generate: async (_params: unknown) => source },
    });

    const stream = await client.images!.generate({
      model: "gpt-image-1",
      prompt: "otter",
      stream: true,
    });

    expect(await collectStream(stream as Stream<unknown>)).toEqual(events);
    const span = await exportedSpan(spans);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(completed ? undefined : true);
  },
);

test("endpoint modality fills aggregate image and transcription output usage", async () => {
  const spans = setupSpans();

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    images: {
      generate: async (_params: unknown) => ({
        data: [],
        usage: { output_tokens: 9, output_tokens_details: { text_tokens: 2 } },
      }),
    },
    audio: {
      transcriptions: {
        create: async (_params: unknown) => ({
          text: "hello",
          usage: { output_tokens: 7, output_tokens_details: {} },
        }),
      },
    },
  });

  await client.images!.generate({ model: "gpt-image-1", prompt: "otter" });
  await client.audio!.transcriptions!.create({ model: "gpt-4o-transcribe" });

  const [image, transcription] = await finishedSpans(spans, 2);
  expect(image?.name).toBe("image gpt-image-1");
  expect(transcription?.name).toBe("transcription gpt-4o-transcribe");
  expect(image?.attributes["gen_ai.usage.image.output_tokens"]).toBe(9);
  expect(image?.attributes["gen_ai.usage.text.output_tokens"]).toBe(2);
  expect(transcription?.attributes["gen_ai.usage.text.output_tokens"]).toBe(7);
  expect(transcription?.attributes["gen_ai.usage.image.output_tokens"]).toBeUndefined();
});

test("streaming transcription bounds captured output while yielding every event", async () => {
  const spans = setupSpans();
  const delta = "\\".repeat(70_000);

  const events = [
    { type: "transcript.text.delta", delta },
    { type: "transcript.text.done", text: delta, usage: { output_tokens: 5 } },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(output.length).toBeLessThanOrEqual(65_536);
  expect(output).toBe("\\".repeat(32_767));
  expect(output).not.toContain("...[truncated]");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.usage.text.output_tokens"]).toBe(5);
});

test("streaming transcription retains no more than the transcript ceiling", async () => {
  const spans = setupSpans();
  const events = [{ type: "transcript.text.delta", delta: "x".repeat(70_000) }];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBe("x".repeat(65_534));
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming transcription omits a bounded prefix that a mask cannot evaluate safely", async () => {
  const spans = setupSpans(undefined, {
    mask: (value) => (String(value).includes("SECRET") ? "[REDACTED]" : value),
  });

  const text = `${"x".repeat(70_000)}SECRET`;

  const events = [
    { type: "transcript.text.delta", delta: text },
    { type: "transcript.text.done", text },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming media reports mapping failures once without changing provider delivery", async () => {
  const errors: Error[] = [];
  const spans = setupSpans(undefined, { onError: (error) => errors.push(error) });
  const mappingError = new Error("media mapping failed");

  const broken = {
    get type(): never {
      throw mappingError;
    },
  };

  const source = new Stream(async function* () {
    yield broken;
    yield broken;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  const delivered = await collectStream(stream);
  expect(delivered).toHaveLength(2);
  expect(delivered[0]).toBe(broken);
  expect(delivered[1]).toBe(broken);
  expect(errors).toEqual([mappingError]);
  expect((await exportedSpan(spans)).attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming transcription bounds escaped Unicode at a code point boundary", async () => {
  const spans = setupSpans();
  const prefix = "😀\n\ud800";
  const delta = `${prefix}${"\\".repeat(40_000)}`;
  const events = [{ type: "transcript.text.delta", delta }];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(output.startsWith(prefix)).toBe(true);
  expect(JSON.stringify(output).length).toBeLessThanOrEqual(65_536);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test.each([
  [false, false],
  [false, true],
  [true, false],
  [true, true],
] as const)(
  "a complete transcript above the chat stream budget is kept (streaming: %s, masked: %s)",
  async (streaming, masked) => {
    const spans = setupSpans(undefined, masked ? { mask: (value) => value } : {});
    const transcript = "x".repeat(55_000);

    const events = [
      { type: "transcript.text.delta", delta: transcript },
      { type: "transcript.text.done", text: transcript },
    ];

    const response = streaming
      ? new Stream(async function* () {
          yield* events;
        }, new AbortController())
      : { text: transcript };

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create() {} },
      embeddings: { create() {} },
      audio: { transcriptions: { create: async (_params: unknown) => response } },
    });

    const result = await client.audio!.transcriptions!.create({
      model: "gpt-4o-transcribe",
      stream: streaming,
    });

    if (streaming) await collectStream(result as Stream<unknown>);
    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.output.messages"]).toBe(transcript);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  },
);

test("a complete masked transcription stream keeps its output and is not flagged", async () => {
  const spans = setupSpans(undefined, { mask: (value) => value });

  const events = [
    { type: "transcript.text.delta", delta: "partial" },
    { type: "transcript.text.done", text: "complete transcript" },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBe("complete transcript");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("streaming transcription replaces truncated deltas with complete terminal text", async () => {
  const spans = setupSpans();

  const events = [
    { type: "transcript.text.delta", delta: "x".repeat(70_000) },
    { type: "transcript.text.done", text: "complete transcript" },
  ];

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toEqual(events);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBe("complete transcript");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("streaming transcription captures many small deltas in linear work", async () => {
  const spans = setupSpans();

  const events = Array.from({ length: 70_000 }, () => ({
    type: "transcript.text.delta",
    delta: "x",
  }));

  const source = new Stream(async function* () {
    yield* events;
  }, new AbortController());

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create() {} },
    embeddings: { create() {} },
    audio: { transcriptions: { create: async (_params: unknown) => source } },
  });

  const stream = await client.audio!.transcriptions!.create({
    model: "gpt-4o-transcribe",
    stream: true,
  });

  expect(await collectStream(stream)).toHaveLength(events.length);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.messages"]).toBe("x".repeat(65_534));
});

test.each([
  ["enabled", true, "hello"],
  ["disabled", false, undefined],
])(
  "streaming transcription segment records output before an error with capture %s",
  async (_, captureOutput, expectedOutput) => {
    const spans = setupSpans(undefined, { captureOutput });

    const source = new Stream(async function* () {
      yield { type: "transcript.text.segment", text: "hello" };
      throw new Error("stream failed");
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create() {} },
      embeddings: { create() {} },
      audio: { transcriptions: { create: async (_params: unknown) => source } },
    });

    const stream = await client.audio!.transcriptions!.create({
      model: "gpt-4o-transcribe",
      stream: true,
    });

    await expect(collectStream(stream)).rejects.toThrow("stream failed");
    const span = await exportedSpan(spans);
    expect(span.status.code).toBe(SPAN_STATUS_ERROR);
    expect(span.attributes["gen_ai.output.messages"]).toBe(expectedOutput);
    expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toEqual(expect.any(Number));
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test.each([
  ["error", false],
  ["error", true],
  ["return", false],
  ["return", true],
  ["eof", false],
  ["eof", true],
] as const)(
  "transcription ending with %s before its terminal event is incomplete (masked: %s)",
  async (ending, masked) => {
    const spans = setupSpans(undefined, masked ? { mask: (value) => value } : undefined);
    const event = { type: "transcript.text.delta", delta: "partial" };

    const source = new Stream(async function* () {
      yield event;

      if (ending === "error") throw new Error("stream failed");
    }, new AbortController());

    const client = wrapOpenAI({
      chat: { completions: { create() {} } },
      responses: { create() {} },
      embeddings: { create() {} },
      audio: { transcriptions: { create: async (_params: unknown) => source } },
    });

    const stream = await client.audio!.transcriptions!.create({
      model: "gpt-4o-transcribe",
      stream: true,
    });

    const iterator = stream[Symbol.asyncIterator]();

    expect(await iterator.next()).toEqual({ done: false, value: event });

    if (ending === "error") await expect(iterator.next()).rejects.toThrow("stream failed");
    else if (ending === "return") await iterator.return?.();
    else expect(await iterator.next()).toEqual({ done: true, value: undefined });
    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.output.messages"]).toBe(masked ? undefined : "partial");
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test.each([
  ["wrapped", false],
  ["prototype", true],
])("batch lifecycle maps create and %s request identities", async (_case, prototype) => {
  const spans = setupSpans();

  const batch = {
    id: "batch_123",
    object: "batch",
    endpoint: "/v1/responses",
    input_file_id: "file_123",
    completion_window: "24h",
    status: "in_progress",
    request_counts: { total: 2, completed: 0, failed: 0 },
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      total_tokens: 15,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens_details: { reasoning_tokens: 0 },
    },
  };

  const fake = createFakeFetch(jsonResponse(batch), jsonResponse(batch), jsonResponse(batch));

  if (prototype) instrumentOpenAI();
  const client = new OpenAI({ apiKey: "test", fetch: fake.fetch, maxRetries: 0 });
  const instrumented = prototype ? client : wrapOpenAI(client);

  await instrumented.batches.create({
    endpoint: "/v1/responses",
    input_file_id: "file_123",
    completion_window: "24h",
  });
  await instrumented.batches.retrieve("batch_123");
  await instrumented.batches.cancel("batch_123");

  const finished = await finishedSpans(spans, 3);

  const inputs = Object.fromEntries(
    finished.map((span) => [
      span.attributes["gen_ai.operation.name"],
      jsonAttr<{ batch_id?: string; input_file_id?: string }>(span, "gen_ai.input.messages"),
    ]),
  );

  expect(inputs["openai.batch.create"]).toMatchObject({ input_file_id: "file_123" });
  expect(inputs["openai.batch.create"]).not.toHaveProperty("batch_id");
  expect(inputs["openai.batch.retrieve"]).toMatchObject({ batch_id: "batch_123" });
  expect(inputs["openai.batch.retrieve"]).not.toHaveProperty("input_file_id");
  expect(inputs["openai.batch.cancel"]).toMatchObject({ batch_id: "batch_123" });
  expect(inputs["openai.batch.cancel"]).not.toHaveProperty("input_file_id");

  for (const span of finished) {
    expect(Object.keys(span.attributes).filter((key) => key.startsWith("gen_ai.usage."))).toEqual(
      [],
    );
    expect(span.attributes["openai.batch.id"]).toBe("batch_123");
    expect(span.attributes["openai.batch.status"]).toBe("in_progress");
  }
});

test("realtime wrapper traces response lifecycle, modality usage, and is idempotent", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();

  const connection = {
    send: vi.fn(),
    close: vi.fn(),
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(
        event,
        (listeners.get(event) ?? []).filter((candidate) => candidate !== listener),
      );
    },
  };

  const originalSend = connection.send;
  const wrapped = wrapOpenAIRealtime(wrapOpenAIRealtime(connection, { model: "gpt-realtime" }));

  const emit = (event: string, value: unknown) => {
    for (const listener of listeners.get(event) ?? []) listener(value);
  };

  wrapped.send({ type: "response.create", response: { instructions: "be brief" } });
  const [sentEvent] = originalSend.mock.calls[0]!;
  const sentResponse = (sentEvent as { response?: object }).response;
  emit("event", { type: "response.created", response: { id: "resp_rt", ...sentResponse } });
  emit("event", { type: "response.output_text.delta", response_id: "resp_rt", delta: "hi" });
  emit("event", {
    type: "response.done",
    response: {
      id: "resp_rt",
      model: "gpt-realtime-1",
      status: "completed",
      output: [
        { type: "message", content: [{ type: "output_text", text: "hi" }] },
        { type: "output_audio", data: "REALTIME_AUDIO", transcript: "spoken" },
      ],
      usage: {
        input_tokens: 7,
        output_tokens: 3,
        total_tokens: 10,
        input_token_details: { text_tokens: 2, audio_tokens: 5 },
        output_token_details: { text_tokens: 3 },
      },
    },
  });

  const span = await exportedSpan(spans);
  expect(originalSend).toHaveBeenCalledTimes(1);
  expect(listeners.get("event")).toHaveLength(1);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_rt");
  expect(span.attributes["gen_ai.usage.audio.input_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.text.output_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toEqual(expect.any(Number));
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("REALTIME_AUDIO");
  expect(String(span.attributes["gen_ai.output.messages"])).toContain("spoken");
});

test("realtime retains bounded completed snapshots when a connection closes before response.done", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection);
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create" });
  emit({ type: "response.created", response: { id: "partial", ...sent[0]?.response } });
  emit({
    type: "response.output_item.added",
    response_id: "partial",
    output_index: 0,
    item: { id: "message", type: "message", role: "assistant", content: [] },
  });
  emit({
    type: "response.content_part.added",
    response_id: "partial",
    output_index: 0,
    item_id: "message",
    content_index: 0,
    part: { type: "text", text: "draft" },
  });
  emit({
    type: "response.output_text.delta",
    response_id: "partial",
    output_index: 0,
    item_id: "message",
    content_index: 0,
    delta: " ignored",
  });
  emit({
    type: "response.output_text.done",
    response_id: "partial",
    output_index: 0,
    item_id: "message",
    content_index: 0,
    text: "hello",
  });
  emit({
    type: "response.content_part.done",
    response_id: "partial",
    output_index: 0,
    item_id: "message",
    content_index: 0,
    part: { type: "text", text: "hello from part" },
  });
  emit({
    type: "response.output_item.done",
    response_id: "partial",
    output_index: 0,
    item: {
      id: "message",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: "hello final" }],
    },
  });
  emit({
    type: "response.output_item.added",
    response_id: "partial",
    output_index: 1,
    item: { id: "audio", type: "message", role: "assistant", content: [] },
  });
  emit({
    type: "response.content_part.added",
    response_id: "partial",
    output_index: 1,
    item_id: "audio",
    content_index: 0,
    part: { type: "audio", audio: "SECRET_AUDIO", transcript: "draft" },
  });
  emit({
    type: "response.output_audio_transcript.delta",
    response_id: "partial",
    output_index: 1,
    item_id: "audio",
    content_index: 0,
    delta: "spoken",
  });
  emit({
    type: "response.output_audio_transcript.done",
    response_id: "partial",
    output_index: 1,
    item_id: "audio",
    content_index: 0,
    transcript: "spoken final",
  });
  emit({
    type: "response.output_item.added",
    response_id: "partial",
    output_index: 2,
    item: {
      id: "function",
      type: "function_call",
      call_id: "call_1",
      name: "lookup",
      arguments: "",
    },
  });
  emit({
    type: "response.function_call_arguments.delta",
    response_id: "partial",
    output_index: 2,
    item_id: "function",
    call_id: "call_1",
    delta: '{"q":',
  });
  emit({
    type: "response.function_call_arguments.done",
    response_id: "partial",
    output_index: 2,
    item_id: "function",
    call_id: "call_1",
    name: "lookup",
    arguments: '{"q":"weather"}',
  });
  emit({
    type: "response.output_item.added",
    response_id: "partial",
    output_index: 3,
    item: {
      id: "mcp",
      type: "mcp_call",
      name: "search",
      server_label: "docs",
      arguments: "",
    },
  });
  emit({
    type: "response.mcp_call_arguments.delta",
    response_id: "partial",
    output_index: 3,
    item_id: "mcp",
    delta: '{"term":',
  });
  emit({
    type: "response.mcp_call_arguments.done",
    response_id: "partial",
    output_index: 3,
    item_id: "mcp",
    arguments: '{"term":"telemetry"}',
  });
  wrapped.close();

  const span = await exportedSpan(spans);

  const output = jsonAttr<
    Array<{
      type: string;
      arguments?: string;
      content?: Array<{ audio?: string; text?: string; transcript?: string }>;
    }>
  >(span, "gen_ai.output.messages");

  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toEqual(expect.any(Number));
  expect(output[0]?.content?.[0]?.text).toBe("hello final");
  expect(output[1]?.content?.[0]).toMatchObject({ transcript: "spoken final" });
  expect(output[1]?.content?.[0]?.audio).toBeUndefined();
  expect(output[2]).toMatchObject({ type: "function_call", arguments: '{"q":"weather"}' });
  expect(output[3]).toMatchObject({ type: "mcp_call", arguments: '{"term":"telemetry"}' });
  expect(String(span.attributes["gen_ai.output.messages"])).not.toContain("SECRET_AUDIO");
});

test("realtime caps reconstructed output before a trace timeout", async () => {
  const spans = setupSpans();
  vi.useFakeTimers();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection, { traceTimeoutMs: 100 });
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create" });
  emit({ type: "response.created", response: { id: "bounded", ...sent[0]?.response } });
  emit({
    type: "response.output_item.added",
    response_id: "bounded",
    output_index: 0,
    item: { id: "first", type: "message", role: "assistant", content: [] },
  });
  emit({
    type: "response.output_text.delta",
    response_id: "bounded",
    output_index: 1,
    item_id: "second",
    content_index: 0,
    delta: "x".repeat(100_000),
  });
  emit({
    type: "response.content_part.added",
    response_id: "bounded",
    output_index: 0,
    item_id: "first",
    content_index: 999,
    part: { type: "output_text", text: "must not be retained" },
  });
  vi.advanceTimersByTime(100);
  vi.useRealTimers();

  const span = spans.getFinishedSpans()[0]!;
  const output = jsonAttr<Array<{ content?: unknown[] }>>(span, "gen_ai.output.messages");
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(String(span.attributes["gen_ai.output.messages"]).length).toBeLessThanOrEqual(65_536);
  expect(output[0]?.content).toEqual([]);
  wrapped.close();
});

test("realtime terminal output supersedes truncated deltas", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection);
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create" });
  emit({ type: "response.created", response: { id: "terminal", ...sent[0]?.response } });
  emit({
    type: "response.output_text.delta",
    response_id: "terminal",
    output_index: 0,
    item_id: "message",
    content_index: 0,
    delta: "x".repeat(100_000),
  });
  emit({
    type: "response.done",
    response: {
      id: "terminal",
      status: "completed",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "final" }],
        },
      ],
    },
  });

  const span = await exportedSpan(spans);

  const output = jsonAttr<Array<{ content?: Array<{ text?: string }> }>>(
    span,
    "gen_ai.output.messages",
  );

  expect(output[0]?.content?.[0]?.text).toBe("final");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
  wrapped.close();
});

test("realtime wrapper correlates reversed manual responses and ignores automatic responses", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();

  interface RealtimeCreateEvent {
    response?: { metadata?: Record<string, string> };
  }

  const sent: RealtimeCreateEvent[] = [];

  const connection = {
    send(event: unknown) {
      const typed = event as RealtimeCreateEvent;

      sent.push(typed);
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(
        event,
        (listeners.get(event) ?? []).filter((candidate) => candidate !== listener),
      );
    },
  };

  const emit = (value: unknown) => {
    for (const listener of listeners.get("event") ?? []) listener(value);
  };

  const wrapped = wrapOpenAIRealtime(connection);

  wrapped.send({
    type: "response.create",
    response: {
      model: "model-a",
      instructions: "first",
      metadata: { caller: "kept-a", __telemetry_dev_response_id: "caller-owned" },
    },
  });
  wrapped.send({
    type: "response.create",
    response: {
      model: "model-b",
      instructions: "second",
      metadata: { caller: "kept-b", __telemetry_dev_response_id: "caller-owned-b" },
    },
  });
  expect(sent[0]?.response?.metadata?.caller).toBe("kept-a");
  expect(sent[0]?.response?.metadata?.__telemetry_dev_response_id).toBe("caller-owned");
  expect(Object.keys(sent[0]?.response?.metadata ?? {})).toHaveLength(3);
  expect(sent[1]?.response?.metadata?.caller).toBe("kept-b");
  expect(Object.keys(sent[1]?.response?.metadata ?? {})).toHaveLength(3);
  const secondCorrelation = sent[1]?.response?.metadata?.__telemetry_dev_response_id_1;
  expect(secondCorrelation).toEqual(expect.any(String));
  emit({ type: "response.created", response: { id: "automatic" } });
  emit({
    type: "response.created",
    response: {
      id: "response-a",
      ...sent[0]?.response,
      metadata: {
        ...sent[0]?.response?.metadata,
        __telemetry_dev_response_id: secondCorrelation,
      },
    },
  });
  emit({ type: "response.created", response: { id: "response-b", ...sent[1]?.response } });
  emit({
    type: "response.done",
    response: {
      id: "automatic",
      model: "automatic-model",
      status: "completed",
      output: ["automatic"],
      usage: { output_tokens: 99 },
    },
  });
  emit({
    type: "response.done",
    response: {
      id: "response-b",
      model: "result-b",
      status: "completed",
      output: ["output-b"],
      usage: { output_tokens: 2, output_token_details: { text_tokens: 2 } },
    },
  });
  emit({
    type: "response.done",
    response: {
      id: "response-a",
      model: "result-a",
      status: "completed",
      output: ["output-a"],
      usage: { output_tokens: 1, output_token_details: { text_tokens: 1 } },
    },
  });

  const finished = await finishedSpans(spans, 2);

  const byInput = Object.fromEntries(
    finished.map((span) => [String(span.attributes["gen_ai.input.messages"]), span]),
  );

  expect(byInput.first?.attributes["gen_ai.response.model"]).toBe("result-a");
  expect(jsonAttr(byInput.first!, "gen_ai.output.messages")).toEqual(["output-a"]);
  expect(byInput.first?.attributes["gen_ai.usage.output_tokens"]).toBe(1);
  expect(byInput.second?.attributes["gen_ai.response.model"]).toBe("result-b");
  expect(jsonAttr(byInput.second!, "gen_ai.output.messages")).toEqual(["output-b"]);
  expect(byInput.second?.attributes["gen_ai.usage.output_tokens"]).toBe(2);

  const fullMetadata = Object.fromEntries(
    Array.from({ length: 16 }, (_, index) => [`caller_${index}`, `value_${index}`]),
  );

  wrapped.send({ type: "response.create", response: { metadata: fullMetadata } });
  expect(sent[2]?.response?.metadata).toEqual(fullMetadata);
  const withMetadataFailure = await finishedSpans(spans, 3);
  expect(withMetadataFailure).toHaveLength(3);
  expect(withMetadataFailure.find((span) => span.status.code === SPAN_STATUS_ERROR)).toBeDefined();
});

test("realtime wrapper ends active spans on transport errors and close", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();

  const connection = {
    send(_event?: unknown) {},
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(
        event,
        (listeners.get(event) ?? []).filter((candidate) => candidate !== listener),
      );
    },
  };

  const wrapped = wrapOpenAIRealtime(connection);

  const emit = (event: string, value: unknown) => {
    for (const listener of listeners.get(event) ?? []) listener(value);
  };

  wrapped.send({ type: "response.create" });
  emit("error", "socket failed");
  wrapped.send({ type: "response.create" });
  wrapped.close();
  wrapped.send({ type: "response.create" });

  const finished = await finishedSpans(spans, 2);
  expect(finished.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
  expect(spans.getFinishedSpans()).toHaveLength(2);
});

test("realtime wrapper preserves EventEmitter unhandled errors while finishing telemetry", async () => {
  const spans = setupSpans();

  const connection = Object.assign(new EventEmitter(), {
    socket: new EventEmitter(),
    send(_event?: unknown) {},
    close() {},
  });

  const wrapped = wrapOpenAIRealtime(connection);
  const transportError = new Error("socket failed");

  wrapped.send({ type: "response.create" });
  expect(() => connection.socket.emit("error", transportError)).toThrow(transportError);
  expect(() => connection.emit("error", transportError)).toThrow(transportError);

  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(connection.listenerCount("error")).toBe(0);

  wrapped.close();
  expect(connection.socket.listenerCount("error")).toBe(0);
});

test("realtime wrapper keeps recoverable API errors scoped and handles remote close", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const socketListeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: unknown[] = [];

  const add = (
    target: Map<string, Array<(...args: unknown[]) => void>>,
    event: string,
    listener: (...args: unknown[]) => void,
  ) => target.set(event, [...(target.get(event) ?? []), listener]);

  const remove = (
    target: Map<string, Array<(...args: unknown[]) => void>>,
    event: string,
    listener: (...args: unknown[]) => void,
  ) =>
    target.set(
      event,
      (target.get(event) ?? []).filter((candidate) => candidate !== listener),
    );

  const connection = {
    send(event?: unknown) {
      sent.push(event);
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      add(listeners, event, listener);
    },
    off(event: string, listener: (...args: unknown[]) => void) {
      remove(listeners, event, listener);
    },
    socket: {
      on(event: string, listener: (...args: unknown[]) => void) {
        add(socketListeners, event, listener);
      },
      off(event: string, listener: (...args: unknown[]) => void) {
        remove(socketListeners, event, listener);
      },
    },
  };

  const wrapped = wrapOpenAIRealtime(wrapOpenAIRealtime(connection));

  const emit = (event: string, value: unknown) => {
    for (const listener of listeners.get(event) ?? []) listener(value);
  };

  wrapped.send({ type: "response.create", event_id: "create_1" });
  emit("event", {
    type: "response.created",
    response: { id: "resp_1", ...(sent.at(-1) as { response?: object }).response },
  });
  emit("event", {
    type: "error",
    error: {
      type: "invalid_request_error",
      code: "bad_session_update",
      message: "unrelated update rejected",
      event_id: "session_update_1",
    },
  });
  emit("error", { error: { event_id: "session_update_1" } });
  emit("event", {
    type: "response.done",
    response: { id: "resp_1", status: "completed", output: [], usage: {} },
  });

  const [completed] = await finishedSpans(spans, 1);
  expect(completed?.status.code).toBe(SPAN_STATUS_UNSET);

  wrapped.send({ type: "response.create", event_id: "create_2" });
  emit("event", {
    type: "error",
    error: {
      type: "invalid_request_error",
      code: "invalid_response",
      message: "response rejected",
      event_id: "create_2",
    },
  });
  emit("error", { error: { event_id: "create_2" } });

  const failed = await finishedSpans(spans, 2);
  expect(failed[1]?.status.code).toBe(SPAN_STATUS_ERROR);

  wrapped.send({ type: "response.create" });
  const sentEvent = sent.at(-1) as { event_id?: unknown } | undefined;
  const generatedEventId = typeof sentEvent?.event_id === "string" ? sentEvent.event_id : "";
  expect(generatedEventId).toMatch(/^event_telemetry_/);
  emit("event", {
    type: "error",
    error: {
      type: "invalid_request_error",
      code: "invalid_response",
      message: "unkeyed response rejected",
      event_id: generatedEventId,
    },
  });
  emit("error", { error: { event_id: generatedEventId } });

  const unkeyedFailed = await finishedSpans(spans, 3);
  expect(unkeyedFailed[2]?.status.code).toBe(SPAN_STATUS_ERROR);

  wrapped.send({ type: "response.create", event_id: "create_3" });
  emit("event", {
    type: "response.created",
    response: { id: "resp_3", ...(sent.at(-1) as { response?: object }).response },
  });
  emit("event", {
    type: "response.done",
    response: { id: "resp_3", status: "completed", output: [], usage: {} },
  });

  const subsequent = await finishedSpans(spans, 4);
  expect(subsequent[3]?.status.code).toBe(SPAN_STATUS_UNSET);

  wrapped.send({ type: "response.create", event_id: "create_4" });
  emit("event", {
    type: "response.created",
    response: { id: "resp_4", ...(sent.at(-1) as { response?: object }).response },
  });
  wrapped.send({ type: "session.update", session: { instructions: 7 } });
  emit("event", {
    type: "error",
    error: {
      type: "invalid_request_error",
      code: "bad_session_update",
      message: "unkeyed unrelated update rejected",
    },
  });
  emit("event", {
    type: "response.done",
    response: { id: "resp_4", status: "completed", output: [], usage: {} },
  });

  const afterUncorrelated = await finishedSpans(spans, 5);
  expect(afterUncorrelated[4]?.status.code).toBe(SPAN_STATUS_UNSET);

  wrapped.send({ type: "response.create", event_id: "create_5" });
  wrapped.send({ type: "response.create", event_id: "create_6" });

  for (const listener of socketListeners.get("close") ?? []) listener();

  const finished = await finishedSpans(spans, 7);
  expect(finished[6]?.status.code).toBe(SPAN_STATUS_ERROR);
  expect(listeners.get("event")).toHaveLength(0);
  expect(socketListeners.get("close")).toHaveLength(0);
  expect(socketListeners.get("error")).toHaveLength(0);
});

test("responses bounds cyclic, deep, oversized, binary, and shared capture without changing results", async () => {
  const spans = setupSpans();
  const shared = { text: "shared" };

  interface CaptureFixture {
    items?: Array<{ text: string }>;
    binary?: { type: string; data: string };
    self?: CaptureFixture;
    next?: CaptureFixture;
  }

  const input: CaptureFixture = {
    items: Array.from({ length: 2_000 }, () => shared),
    binary: { type: "input_audio", data: "SECRET_BINARY" },
  };

  input.self = input;
  let deep = input;

  for (let index = 0; index < 100; index += 1) {
    const next: CaptureFixture = {};
    deep.next = next;
    deep = next;
  }

  const output = { id: "resp_bounded", status: "completed", output: input };
  const create = vi.fn((_params: unknown) => output);

  const client = wrapOpenAI({
    chat: { completions: { create() {} } },
    responses: { create },
    embeddings: { create() {} },
  });

  await expect(client.responses.create({ model: "model", input } as never)).resolves.toBe(output);
  expect(create).toHaveBeenCalledOnce();
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(String(span.attributes["gen_ai.input.messages"])).not.toContain("SECRET_BINARY");
  expect(String(span.attributes["gen_ai.input.messages"]).length).toBeLessThan(65_536);
});

test("realtime expires pending and active traces, enforces cap, and ignores late events", () => {
  const spans = setupSpans();
  vi.useFakeTimers();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection, { traceTimeoutMs: 1_000, maxInFlight: 2 });
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create", response: { instructions: "pending-expiry" } });
  vi.advanceTimersByTime(1_000);
  wrapped.send({ type: "response.create", response: { instructions: "active-expiry" } });
  emit({ type: "response.created", response: { id: "active", ...sent.at(-1)?.response } });
  vi.advanceTimersByTime(1_000);
  emit({ type: "response.done", response: { id: "active", status: "completed", output: [] } });

  wrapped.send({ type: "response.create", response: { instructions: "evicted" } });
  wrapped.send({ type: "response.create", response: { instructions: "kept-1" } });
  wrapped.send({ type: "response.create", response: { instructions: "kept-2" } });
  vi.useRealTimers();
  const finished = spans.getFinishedSpans();
  expect(finished).toHaveLength(3);
  expect(finished.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
  emit({ type: "response.created", response: { id: "late", ...sent[0]?.response } });
  emit({ type: "response.done", response: { id: "late", status: "completed", output: [] } });
  expect(spans.getFinishedSpans()).toHaveLength(3);
  wrapped.close();
});

test.each([0, -1, 1.5, Number.NaN])("realtime rejects invalid maxInFlight %s", (maxInFlight) => {
  const connection = {
    send() {},
    close() {},
    on() {},
    off() {},
  };

  expect(() => wrapOpenAIRealtime(connection, { maxInFlight })).toThrow(
    "maxInFlight must be a positive integer",
  );
});

test.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
  "realtime rejects invalid traceTimeoutMs %s",
  (traceTimeoutMs) => {
    const connection = {
      send() {},
      close() {},
      on() {},
      off() {},
    };

    expect(() => wrapOpenAIRealtime(connection, { traceTimeoutMs })).toThrow(
      "traceTimeoutMs must be a positive finite number",
    );
  },
);

test("realtime in-flight eviction selects the oldest trace across active and pending", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection, { maxInFlight: 2 });
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create", response: { instructions: "A" } });
  emit({ type: "response.created", response: { id: "response-a", ...sent[0]?.response } });
  emit({
    type: "response.output_text.delta",
    response_id: "response-a",
    output_index: 0,
    item_id: "message-a",
    content_index: 0,
    delta: "partial A",
  });
  wrapped.send({ type: "response.create", response: { instructions: "B" } });
  wrapped.send({ type: "response.create", response: { instructions: "C" } });
  emit({ type: "response.created", response: { id: "response-b", ...sent[1]?.response } });
  emit({ type: "response.done", response: { id: "response-b", status: "completed", output: [] } });

  const finished = await finishedSpans(spans, 2);

  expect(finished.map((span) => span.status.code)).toEqual([SPAN_STATUS_ERROR, SPAN_STATUS_UNSET]);
  expect(
    jsonAttr<Array<{ content: Array<{ text: string }> }>>(finished[0]!, "gen_ai.output.messages"),
  ).toEqual([
    {
      type: "message",
      role: "assistant",
      id: "message-a",
      content: [{ type: "output_text", text: "partial A" }],
    },
  ]);
  expect(finished[0]?.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  wrapped.close();
});

test("realtime assigns unique in-flight event ids before binding API errors", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ event_id?: unknown; response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { event_id?: unknown; response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection);
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create", event_id: "duplicate" });
  wrapped.send({ type: "response.create", event_id: "duplicate" });

  expect(sent[0]?.event_id).toBe("duplicate");
  expect(sent[1]?.event_id).not.toBe("duplicate");
  emit({ type: "error", error: { event_id: "duplicate", message: "first failed" } });

  const first = await exportedSpan(spans);
  expect(first.status.code).toBe(SPAN_STATUS_ERROR);

  emit({
    type: "response.created",
    response: { id: "second", ...(sent[1]?.response as object) },
  });
  emit({
    type: "response.done",
    response: { id: "second", status: "completed", output: [], usage: {} },
  });

  const finished = await finishedSpans(spans, 2);
  expect(finished[1]?.status.code).toBe(SPAN_STATUS_UNSET);
  wrapped.close();
});

test("realtime duplicate response ids finish the incoming trace without orphaning the active trace", async () => {
  const spans = setupSpans();
  const listeners = new Map<string, Array<(...args: unknown[]) => void>>();
  const sent: Array<{ response?: object }> = [];

  const connection = {
    send(event: unknown) {
      sent.push(event as { response?: object });
    },
    close() {},
    on(event: string, listener: (...args: unknown[]) => void) {
      listeners.set(event, [...(listeners.get(event) ?? []), listener]);
    },
    off() {},
  };

  const wrapped = wrapOpenAIRealtime(connection, { maxInFlight: 2 });
  const emit = (value: unknown) => listeners.get("event")?.forEach((listener) => listener(value));

  wrapped.send({ type: "response.create", response: { instructions: "A" } });
  wrapped.send({ type: "response.create", response: { instructions: "B" } });
  emit({ type: "response.created", response: { id: "same", ...sent[0]?.response } });
  emit({ type: "response.created", response: { id: "same", ...sent[1]?.response } });

  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(spans.getFinishedSpans()[0]?.status.code).toBe(SPAN_STATUS_ERROR);
  expect(
    spans
      .getFinishedSpans()[0]
      ?.events.some((event) =>
        String(event.attributes?.["exception.message"]).includes("duplicate response id same"),
      ),
  ).toBe(true);

  wrapped.send({ type: "response.create", response: { instructions: "C" } });
  expect(spans.getFinishedSpans()).toHaveLength(1);
  emit({ type: "response.done", response: { id: "same", status: "completed", output: [] } });

  const finished = await finishedSpans(spans, 2);
  expect(finished.map((span) => span.status.code)).toEqual([SPAN_STATUS_ERROR, SPAN_STATUS_UNSET]);
  wrapped.close();
  expect(spans.getFinishedSpans()).toHaveLength(3);
});

test.each([
  [new Error("bare socket error"), "bare socket error"],
  [{ error: new Error("nested socket error") }, "nested socket error"],
])("realtime socket errors finish all traces from %j", async (cause, message) => {
  const spans = setupSpans();
  const socketListeners = new Map<string, (...args: unknown[]) => void>();

  const connection = {
    send(_event: unknown) {},
    close() {},
    on() {},
    off() {},
    socket: {
      addEventListener(event: string, listener: (...args: unknown[]) => void) {
        socketListeners.set(event, listener);
      },
      removeEventListener() {},
    },
  };

  const wrapped = wrapOpenAIRealtime(connection);
  wrapped.send({ type: "response.create" });
  socketListeners.get("error")?.(cause);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.events.some((event) => event.attributes?.["exception.message"] === message)).toBe(
    true,
  );
});
