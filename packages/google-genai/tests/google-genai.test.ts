import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  AggregationTemporality,
  DataPointType,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import { flush, init, shutdown } from "@telemetry-dev/sdk";
import {
  FunctionCallingConfigMode,
  GoogleGenAI,
  Type,
  type FunctionCall,
  type Part,
  type Tool,
} from "@google/genai";
import { afterEach, expect, test, vi } from "vitest";

import { wrapGoogleGenAI } from "../src/index.ts";

const SPAN_STATUS_UNSET = 0;
const SPAN_STATUS_ERROR = 2;

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

interface CapturedRequest {
  method: string | undefined;
  path: string;
  body: JsonValue | undefined;
}

function jsonResponse(body: JsonValue): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function jsonErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { message, status, code: status } }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function sseResponse(chunks: JsonValue[]): Response {
  const body = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("");

  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function erroringSseResponse(chunks: JsonValue | JsonValue[], error: Error): Response {
  const encoder = new TextEncoder();
  const pending = Array.isArray(chunks) ? [...chunks] : [chunks];

  const body = new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        const chunk = pending.shift();

        if (chunk !== undefined) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));

          return;
        }

        controller.error(error);
      },
    },
    { highWaterMark: 0 },
  );

  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function createFakeFetch(...responses: Response[]) {
  const requests: CapturedRequest[] = [];

  const fetchImpl: typeof fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : String(input);

    const bodyText = typeof init?.body === "string" ? init.body : undefined;

    requests.push({
      method: init?.method,
      path: new URL(url).pathname + new URL(url).search,
      body: bodyText ? JSON.parse(bodyText) : undefined,
    });
    const response = responses.shift();

    if (!response) throw new Error(`unexpected request to ${url}`);

    return response;
  };

  return { fetch: fetchImpl, requests };
}

function setupSpans(
  options: {
    captureInput?: boolean;
    captureOutput?: boolean;
    mask?: (value: unknown, context: { key: string }) => unknown;
  } = {},
): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "google-genai-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
      ...options,
    },
    { spanExporter },
  );

  return spanExporter;
}

function setupSpansAndMetrics() {
  const spanExporter = new InMemorySpanExporter();
  const metricBatches: ResourceMetrics[] = [];

  const metricExporter: PushMetricExporter = {
    export: (resourceMetrics, resultCallback) => {
      metricBatches.push(resourceMetrics);
      resultCallback({ code: 0 });
    },
    selectAggregationTemporality: () => AggregationTemporality.DELTA,
    forceFlush: () => Promise.resolve(),
    shutdown: () => Promise.resolve(),
  };

  init(
    {
      apiKey: "td_live_test",
      serviceName: "google-genai-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter, metricExporter },
  );

  return { spanExporter, metricBatches };
}

function outputChunkIntervalCount(metricBatches: ResourceMetrics[]): number {
  let count = 0;

  for (const batch of metricBatches) {
    for (const scope of batch.scopeMetrics) {
      for (const metric of scope.metrics) {
        if (metric.descriptor.name !== "gen_ai.client.operation.time_per_output_chunk") continue;

        if (metric.dataPointType !== DataPointType.HISTOGRAM) throw new Error("expected histogram");

        for (const point of metric.dataPoints) count += point.value.count;
      }
    }
  }

  return count;
}

function clientWith(): GoogleGenAI {
  return wrapGoogleGenAI(new GoogleGenAI({ apiKey: "test" }));
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
  expect(Object.prototype.toString.call(value)).toBe("[object String]");

  return JSON.parse(String(value)) as T;
}

function messagesAttr(span: ReadableSpan, key: "gen_ai.input.messages" | "gen_ai.output.messages") {
  return jsonAttr<unknown[]>(span, key);
}

afterEach(async () => {
  vi.unstubAllGlobals();
  await shutdown();
  vi.restoreAllMocks();
});

test("generateContent maps request, response, usage, finish reason, provider, and sampling attributes", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [
        {
          content: { role: "model", parts: [{ text: "Hello there" }] },
          finishReason: "STOP",
        },
      ],
      modelVersion: "gemini-2.5-flash-001",
      responseId: "resp_123",
      usageMetadata: {
        promptTokenCount: 12,
        candidatesTokenCount: 4,
        totalTokenCount: 16,
        cachedContentTokenCount: 2,
        thoughtsTokenCount: 1,
        toolUsePromptTokenCount: 3,
      },
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [{ role: "user", parts: [{ text: "Say hello" }] }],
    config: {
      systemInstruction: { parts: [{ text: "Be terse" }] },
      temperature: 0.4,
      topP: 0.9,
      topK: 20,
      maxOutputTokens: 128,
      stopSequences: ["END"],
      seed: 42,
    },
  });

  expect(response.text).toBe("Hello there");
  const span = await exportedSpan(spans);
  expect(span.name).toBe("chat gemini-2.5-flash");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("gcp.gemini");
  expect(span.attributes["gen_ai.request.model"]).toBe("gemini-2.5-flash");
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.4);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.top_k"]).toBe(20);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(128);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["END"]);
  expect(span.attributes["gen_ai.request.seed"]).toBe(42);
  expect(jsonAttr(span, "gen_ai.system_instructions")).toEqual({ parts: [{ text: "Be terse" }] });
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual([
    { role: "user", parts: [{ text: "Say hello" }] },
  ]);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "Hello there" }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(12);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(16);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.reasoning.output_tokens"]).toBe(1);
  expect(span.attributes["google_genai.usage.tool_use_prompt_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.response.id"]).toBe("resp_123");
  expect(span.attributes["gen_ai.response.model"]).toBe("gemini-2.5-flash-001");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["STOP"]);
});

test("generateContent normalizes part-array input before recording", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [{ content: { role: "model", parts: [{ text: "An image." }] } }],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const contents = [
    { text: "Describe this image" },
    { inlineData: { mimeType: "image/png", data: "abc123" } },
  ];

  await client.models.generateContent({ model: "gemini-2.5-flash", contents });

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual([
    {
      role: "user",
      parts: [{ text: "Describe this image" }, { inlineData: { mimeType: "image/png" } }],
    },
  ]);
});

test("structured output records json output type", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [
        {
          content: { role: "model", parts: [{ text: '{"answer":"42"}' }] },
          finishReason: "STOP",
        },
      ],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Return JSON",
    config: {
      responseMimeType: "application/json",
      responseJsonSchema: {
        type: "object",
        properties: { answer: { type: "string" } },
      },
    },
  });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.output.type"]).toBe("json");
});

test("function tools map definitions, toolConfig, and functionCall output", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
          },
          finishReason: "STOP",
        },
      ],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Weather?",
    config: {
      tools: [
        {
          functionDeclarations: [
            {
              name: "get_weather",
              description: "Get weather",
              parameters: { type: Type.OBJECT, properties: { city: { type: Type.STRING } } },
            },
          ],
        },
      ],
      toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } },
    },
  });
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.tool.definitions")).toEqual([
    {
      type: "function",
      name: "get_weather",
      description: "Get weather",
      parameters: { type: Type.OBJECT, properties: { city: { type: Type.STRING } } },
    },
  ]);
  expect(jsonAttr(span, "google_genai.request.tool_config")).toEqual({
    functionCallingConfig: { mode: FunctionCallingConfigMode.ANY },
  });
  expect(messagesAttr(span, "gen_ai.output.messages")[0]).toEqual({
    role: "model",
    parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
  });
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["STOP"]);
});

test("generateContent automatic function calls sum usage from internal generateContent calls", async () => {
  const spans = setupSpans();

  const callableTool = {
    name: "get_weather",
    async tool(): Promise<Tool> {
      return {
        functionDeclarations: [
          {
            name: "get_weather",
            description: "Get weather",
            parameters: { type: Type.OBJECT, properties: { city: { type: Type.STRING } } },
          },
        ],
      };
    },
    async callTool(_functionCalls: FunctionCall[]): Promise<Part[]> {
      return [
        {
          functionResponse: {
            name: "get_weather",
            response: { temperature: 21 },
          },
        },
      ];
    },
  };

  const functionCallResponse = {
    responseId: "afc_1",
    candidates: [
      {
        content: {
          role: "model",
          parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
        },
        finishReason: "STOP",
      },
    ],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 2,
      totalTokenCount: 12,
      toolUsePromptTokenCount: 2,
    },
  };

  const finalResponse = {
    responseId: "afc_2",
    candidates: [
      {
        content: { role: "model", parts: [{ text: "Sunny." }] },
        finishReason: "STOP",
      },
    ],
    usageMetadata: {
      promptTokenCount: 4,
      candidatesTokenCount: 3,
      totalTokenCount: 7,
      toolUsePromptTokenCount: 3,
    },
  };

  const fake = createFakeFetch(jsonResponse(functionCallResponse), jsonResponse(finalResponse));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [{ role: "user", parts: [{ text: "Weather?" }] }],
    config: {
      tools: [callableTool],
      automaticFunctionCalling: { maximumRemoteCalls: 2 },
    },
  });

  expect(response.text).toBe("Sunny.");
  expect(fake.requests).toHaveLength(2);
  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual([
    { role: "user", parts: [{ text: "Weather?" }] },
    {
      role: "model",
      parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
    },
    {
      role: "user",
      parts: [{ functionResponse: { name: "get_weather", response: { temperature: 21 } } }],
    },
  ]);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "Sunny." }] },
  ]);
  expect(span.attributes["google_genai.automatic_function_calling"]).toBe(true);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(14);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(19);
  expect(span.attributes["google_genai.usage.tool_use_prompt_tokens"]).toBe(5);
});

test("built-in tools map every SDK marker to tool definitions", async () => {
  const spans = setupSpans();

  const client = wrapGoogleGenAI({
    models: {
      generateContent(_params: any) {
        return { text: "ok" };
      },
    },
  });

  client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Use tools",
    config: {
      tools: [
        {
          googleSearch: {},
          googleSearchRetrieval: {},
          codeExecution: {},
          urlContext: {},
          computerUse: {},
          fileSearch: {},
          retrieval: {},
          googleMaps: {},
          enterpriseWebSearch: {},
          parallelAiSearch: {},
          mcpServers: [],
        },
      ],
    },
  });
  const span = await exportedSpan(spans);
  expect(jsonAttr(span, "gen_ai.tool.definitions")).toEqual([
    { type: "googleSearch" },
    { type: "googleSearchRetrieval" },
    { type: "codeExecution" },
    { type: "urlContext" },
    { type: "computerUse" },
    { type: "fileSearch" },
    { type: "retrieval" },
    { type: "googleMaps" },
    { type: "enterpriseWebSearch" },
    { type: "parallelAiSearch" },
    { type: "mcpServers" },
  ]);
});

test("multi-candidate responses capture choice count and finish reasons", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [
        {
          index: 0,
          content: { role: "model", parts: [{ text: "A" }] },
          finishReason: "STOP",
        },
        {
          index: 1,
          content: { role: "model", parts: [{ text: "B" }] },
          finishReason: "MAX_TOKENS",
        },
      ],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Two answers",
    config: { candidateCount: 2 },
  });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.request.choice.count"]).toBe(2);
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["STOP", "MAX_TOKENS"]);
  expect(messagesAttr(span, "gen_ai.output.messages")).toHaveLength(2);
});

test("streaming aggregates chunks, preserves passthrough, and records time to first chunk", async () => {
  const spans = setupSpans();

  const chunk1 = {
    candidates: [{ content: { role: "model", parts: [{ text: "Hel" }] } }],
    responseId: "stream_1",
    modelVersion: "gemini-2.5-flash-stream",
  };

  const chunk2 = {
    candidates: [{ content: { role: "model", parts: [{ text: "lo" }] } }],
  };

  const chunk3 = {
    candidates: [{ content: { role: "model", parts: [{ text: "!" }] }, finishReason: "STOP" }],
    usageMetadata: {
      promptTokenCount: 3,
      candidatesTokenCount: 2,
      totalTokenCount: 5,
    },
  };

  const fake = createFakeFetch(sseResponse([chunk1, chunk2, chunk3]));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  });

  const stripTransport = (value: any) => {
    const copy = JSON.parse(JSON.stringify(value)) as { [key: string]: JsonValue };
    delete copy.sdkHttpResponse;

    return copy;
  };

  const seen: object[] = [];

  for await (const chunk of stream) seen.push(stripTransport(chunk));
  expect(seen).toEqual([chunk1, chunk2, chunk3]);
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toBeDefined();
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "Hello!" }] },
  ]);
});

test("streaming automatic function calls keep tool turns out of final output", async () => {
  const spans = setupSpans();
  const requestContents = [{ role: "user", parts: [{ text: "Weather?" }] }];

  const expectedInput = [
    { role: "user", parts: [{ text: "Weather?" }] },
    {
      role: "model",
      parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
    },
    {
      role: "user",
      parts: [{ functionResponse: { name: "get_weather", response: { temperature: 21 } } }],
    },
  ];

  const functionCallChunk = {
    responseId: "afc_1",
    candidates: [
      {
        content: {
          role: "model",
          parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
        },
      },
    ],
    usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12 },
  };

  const functionResponseChunk = {
    candidates: [
      {
        content: {
          role: "user",
          parts: [{ functionResponse: { name: "get_weather", response: { temperature: 21 } } }],
        },
      },
    ],
  };

  const finalChunk = {
    responseId: "afc_2",
    candidates: [
      {
        content: { role: "model", parts: [{ text: "Sunny." }] },
        finishReason: "STOP",
      },
    ],
    usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 3, totalTokenCount: 7 },
  };

  const fake = createFakeFetch(sseResponse([functionCallChunk, functionResponseChunk, finalChunk]));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: requestContents,
  });

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(requestContents).toEqual([{ role: "user", parts: [{ text: "Weather?" }] }]);
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual(expectedInput);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "Sunny." }] },
  ]);
  expect(span.attributes["google_genai.automatic_function_calling"]).toBe(true);
  expect(span.attributes["gen_ai.response.id"]).toBe("afc_2");
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(14);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(19);
});

test("streaming AFC input does not consume the resumed output budget", async () => {
  const spans = setupSpans();
  const functionArgument = "x".repeat(47_000);
  const finalText = "y".repeat(3_000);

  const chunks = [
    {
      responseId: "afc_large_1",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ functionCall: { name: "lookup", args: { value: functionArgument } } }],
          },
        },
      ],
    },
    {
      candidates: [
        {
          content: {
            role: "user",
            parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
          },
        },
      ],
    },
    {
      responseId: "afc_large_2",
      candidates: [{ content: { role: "model", parts: [{ text: finalText }] } }],
    },
  ];

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield* chunks;
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "go" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: finalText }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("streaming AFC applies one cumulative input budget", async () => {
  const spans = setupSpans();
  const requestText = "r".repeat(40_000);
  const functionArgument = "a".repeat(40_000);
  const finalText = "final";

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield {
          responseId: "afc_bounded_1",
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ functionCall: { name: "lookup", args: { value: functionArgument } } }],
              },
            },
          ],
        };
        yield {
          candidates: [
            {
              content: {
                role: "user",
                parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
              },
            },
          ],
        };
        yield {
          responseId: "afc_bounded_2",
          candidates: [{ content: { role: "model", parts: [{ text: finalText }] } }],
        };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: requestText }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  const input = String(span.attributes["gen_ai.input.messages"]);
  expect(new TextEncoder().encode(input).byteLength).toBeLessThanOrEqual(48 * 1024);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: finalText }] },
  ]);
});

test("authoritative AFC history clears truncation from superseded synthetic turns", async () => {
  const spans = setupSpans();

  const authoritativeHistory = [
    { role: "user", parts: [{ text: "run lookup" }] },
    { role: "model", parts: [{ functionCall: { name: "lookup", args: {} } }] },
    {
      role: "user",
      parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
    },
  ];

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        for (let index = 0; index < 1_001; index += 1) {
          yield { candidates: [{ content: { role: "model", parts: [{ text: "x" }] } }] };
        }

        yield {
          candidates: [
            {
              content: {
                role: "user",
                parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
              },
            },
          ],
        };
        yield {
          automaticFunctionCallingHistory: authoritativeHistory,
          candidates: [{ content: { role: "model", parts: [{ text: "done" }] } }],
        };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "run lookup" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual(authoritativeHistory);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "done" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("streaming output applies its cumulative limit in UTF-8 bytes", async () => {
  const spans = setupSpans();
  const text = "界".repeat(10_000);

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield { candidates: [{ content: { role: "model", parts: [{ text }] } }] };
        yield { candidates: [{ content: { role: "model", parts: [{ text }] } }] };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "go" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  const output = String(span.attributes["gen_ai.output.messages"]);
  expect(new TextEncoder().encode(output).byteLength).toBeLessThanOrEqual(48 * 1024);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming capture ignores unretained candidate metadata", async () => {
  const spans = setupSpans();

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield {
          candidates: [
            {
              content: { role: "model", parts: [{ text: "kept" }] },
              unusedMetadata: "x".repeat(70_000),
            },
          ],
        };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "go" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "kept" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();
});

test("streaming capture resumes after rejecting an oversized output delta", async () => {
  const spans = setupSpans();
  const retained = "a".repeat(40_000);

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield { candidates: [{ content: { role: "model", parts: [{ text: retained }] } }] };
        yield {
          candidates: [{ content: { role: "model", parts: [{ text: "b".repeat(10_000) }] } }],
        };
        yield {
          candidates: [
            { content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" },
          ],
        };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "go" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: `${retained}ok` }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming records terminal metadata when the terminal candidate is truncated", async () => {
  const spans = setupSpans();
  const retained = "a".repeat(40_000);

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield {
          candidates: [{ index: 1, content: { role: "model", parts: [{ text: retained }] } }],
        };
        yield {
          candidates: [
            {
              index: 1,
              content: { role: "model", parts: [{ text: "b".repeat(50_000) }] },
              finishReason: "MAX_TOKENS",
            },
          ],
        };
      },
    },
  });

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: "go" }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);

  const output = messagesAttr(span, "gen_ai.output.messages") as Array<{
    role: string;
    parts: Array<{ text?: string }>;
  }>;

  expect(output[0]?.parts[0]?.text).toBe(retained);
  expect(new TextEncoder().encode(JSON.stringify(output)).byteLength).toBeLessThanOrEqual(
    48 * 1024,
  );
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["MAX_TOKENS"]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming AFC capture accepts fitting entries after a rejected turn", async () => {
  const spans = setupSpans();

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: unknown) {
        yield {
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ functionCall: { name: "lookup", args: { value: "x".repeat(10_000) } } }],
              },
            },
          ],
        };
        yield {
          candidates: [
            {
              content: {
                role: "user",
                parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
              },
            },
          ],
        };
        yield { candidates: [{ content: { role: "model", parts: [{ text: "done" }] } }] };
      },
    },
  });

  const prompt = "p".repeat(40_000);

  const stream = await Promise.resolve(
    client.models.generateContentStream({ model: "gemini", contents: prompt }),
  );

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);

  const input = messagesAttr(span, "gen_ai.input.messages") as Array<{
    role: string;
    parts: unknown[];
  }>;

  expect(input).toHaveLength(2);
  expect(input[0]).toEqual({ role: "user", parts: [{ text: prompt }] });
  expect(input[1]).toEqual({
    role: "user",
    parts: [{ functionResponse: { name: "lookup", response: { ok: true } } }],
  });
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "done" }] },
  ]);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("unary AFC replacement reports terminal input truncation", async () => {
  const spans = setupSpans();

  const response = {
    candidates: [{ content: { role: "model", parts: [{ text: "done" }] } }],
    automaticFunctionCallingHistory: [{ role: "user", parts: [{ text: "x".repeat(60_000) }] }],
  };

  const client = wrapGoogleGenAI({
    models: { generateContent: (_params: unknown) => response },
  });

  expect(client.models.generateContent({ model: "gemini", contents: "go" })).toBe(response);

  const span = await exportedSpan(spans);
  const input = String(span.attributes["gen_ai.input.messages"]);
  expect(new TextEncoder().encode(input).byteLength).toBeLessThanOrEqual(48 * 1024);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("streaming automatic function history excludes prior synthetic output", async () => {
  const spans = setupSpans();

  const functionCallChunk = {
    responseId: "afc_1",
    candidates: [
      {
        content: {
          role: "model",
          parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
        },
      },
    ],
  };

  const finalChunk = {
    responseId: "afc_2",
    candidates: [
      {
        content: { role: "model", parts: [{ text: "Sunny." }] },
        finishReason: "STOP",
      },
    ],
    automaticFunctionCallingHistory: [
      { role: "user", parts: [{ text: "Weather?" }] },
      {
        role: "model",
        parts: [{ functionCall: { name: "get_weather", args: { city: "Paris" } } }],
      },
      {
        role: "user",
        parts: [{ functionResponse: { name: "get_weather", response: { temperature: 21 } } }],
      },
    ],
  };

  const client = wrapGoogleGenAI({
    models: {
      async *generateContentStream(_params: any) {
        yield functionCallChunk;
        yield finalChunk;
      },
    },
  });

  const stream = await (client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: [{ role: "user", parts: [{ text: "Weather?" }] }],
  }) as any);

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual(
    finalChunk.automaticFunctionCallingHistory,
  );
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "Sunny." }] },
  ]);
});

test("streaming keeps thought parts separate from plain text parts", async () => {
  const spans = setupSpans();

  const chunk1 = {
    candidates: [{ content: { role: "model", parts: [{ text: "thinking", thought: true }] } }],
  };

  const chunk2 = {
    candidates: [{ content: { role: "model", parts: [{ text: "answer" }] } }],
  };

  const chunk3 = {
    candidates: [{ finishReason: "STOP" }],
    usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
  };

  const fake = createFakeFetch(sseResponse([chunk1, chunk2, chunk3]));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Think",
  });

  for await (const _chunk of stream) {
  }

  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")[0]).toEqual({
    role: "model",
    parts: [{ text: "thinking", thought: true }, { text: "answer" }],
  });
});

test("streaming mid-stream error records error status with partial output", async () => {
  const spans = setupSpans();

  const chunk = {
    candidates: [{ content: { role: "model", parts: [{ text: "partial" }] } }],
  };

  const fake = createFakeFetch(erroringSseResponse(chunk, new Error("stream failed")));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  });

  await expect(async () => {
    for await (const _chunk of stream) {
    }
  }).rejects.toThrow("stream failed");
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "partial" }] },
  ]);
});

test.each<{ partialArgs: JsonValue[] }>([
  { partialArgs: [{ jsonPath: "$.enabled", boolValue: false }] },
  { partialArgs: [{ jsonPath: "$.retries", numberValue: 0 }] },
  {
    partialArgs: [
      { jsonPath: "$.enabled", boolValue: false },
      { jsonPath: "$.retries", numberValue: 0 },
    ],
  },
])("Vertex streaming counts partial values once but not shells: %j", async ({ partialArgs }) => {
  const { spanExporter, metricBatches } = setupSpansAndMetrics();

  const chunks: JsonValue[] = [
    {
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                functionCall: {
                  name: "set_flags",
                  partialArgs: [
                    { jsonPath: "$.enabled", willContinue: true },
                    { jsonPath: "$.label", stringValue: "" },
                  ],
                },
              },
            ],
          },
        },
      ],
    },
    {
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                functionCall: {
                  name: "set_flags",
                  partialArgs,
                },
              },
            ],
          },
        },
      ],
    },
    {
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                functionCall: {
                  name: "set_flags",
                  partialArgs: [{ jsonPath: "$.fallback", nullValue: "NULL_VALUE" }],
                },
              },
            ],
          },
          finishReason: "STOP",
        },
      ],
    },
  ];

  const fake = createFakeFetch(sseResponse(chunks));
  vi.stubGlobal("fetch", fake.fetch);
  const client = wrapGoogleGenAI(new GoogleGenAI({ vertexai: true, apiKey: "test" }));

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Set flags",
  });

  const received = [];

  for await (const chunk of stream) received.push(chunk);

  expect(received).toHaveLength(3);
  expect(received[1]!.candidates?.[0]?.content?.parts?.[0]?.functionCall?.partialArgs).toEqual(
    partialArgs,
  );
  await exportedSpan(spanExporter);
  await flush();
  expect(outputChunkIntervalCount(metricBatches)).toBe(1);
});

test("Vertex streaming retains partial argument chunk intervals when interrupted", async () => {
  const { spanExporter, metricBatches } = setupSpansAndMetrics();

  const first = {
    candidates: [
      {
        content: {
          role: "model",
          parts: [
            {
              functionCall: {
                name: "lookup",
                partialArgs: [{ jsonPath: "$.query", stringValue: "north" }],
              },
            },
          ],
        },
      },
    ],
  };

  const second = {
    candidates: [
      {
        content: {
          role: "model",
          parts: [
            {
              functionCall: {
                name: "lookup",
                partialArgs: [{ jsonPath: "$.limit", numberValue: 0 }],
              },
            },
          ],
        },
      },
    ],
  };

  const fake = createFakeFetch(
    erroringSseResponse([first, second], new Error("stream interrupted")),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = wrapGoogleGenAI(new GoogleGenAI({ vertexai: true, apiKey: "test" }));

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Lookup",
  });

  const iterator = stream[Symbol.asyncIterator]();
  expect(
    (await iterator.next()).value.candidates[0].content.parts[0].functionCall.partialArgs,
  ).toEqual([{ jsonPath: "$.query", stringValue: "north" }]);
  expect(
    (await iterator.next()).value.candidates[0].content.parts[0].functionCall.partialArgs,
  ).toEqual([{ jsonPath: "$.limit", numberValue: 0 }]);
  await expect(iterator.next()).rejects.toThrow("stream interrupted");

  const span = await exportedSpan(spanExporter);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  await flush();
  expect(outputChunkIntervalCount(metricBatches)).toBe(1);
});

test.each(["next", "return", "throw"] as const)(
  "streaming %s timestamps precede telemetry classification",
  async (method) => {
    const spans = setupSpans();
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);

    const timings = [
      [100, 30],
      [240, 90],
    ] as const;

    let index = 0;

    const pull = async () => {
      const timing = timings[index++];

      if (!timing) return { done: true as const, value: undefined };
      const [receivedAt, mappingMs] = timing;
      now = receivedAt;

      return {
        done: false as const,
        value: {
          get candidates() {
            now = receivedAt + mappingMs;

            return [{ content: { role: "model", parts: [{ text: "A" }] } }];
          },
        },
      };
    };

    const source = {
      [Symbol.asyncIterator]() {
        return this;
      },
      next: pull,
      return: pull,
      throw: pull,
    };

    const client = wrapGoogleGenAI({
      models: { generateContentStream: async (_params: unknown) => source },
    });

    const stream = await client.models.generateContentStream({
      model: "gemini-2.5-flash",
      contents: "Hi",
    });

    expect((await stream.next()).done).toBe(false);
    expect((await stream[method]()).done).toBe(false);
    expect((await stream.return()).done).toBe(true);
    expect(await exportedSpan(spans)).toMatchObject({
      [Symbol.for("telemetry.dev.outputChunkHistogram")]: {
        count: 1,
        sum: 0.14,
        min: 0.14,
        max: 0.14,
      },
    });
  },
);

test("streaming early break ends span with partial aggregate", async () => {
  const spans = setupSpans();

  const chunk1 = {
    candidates: [{ content: { role: "model", parts: [{ text: "first" }] } }],
  };

  const chunk2 = {
    candidates: [{ content: { role: "model", parts: [{ text: "second" }] } }],
  };

  const fake = createFakeFetch(sseResponse([chunk1, chunk2]));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  });

  for await (const _chunk of stream) break;
  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "first" }] },
  ]);
});

test("streaming return before first chunk ends span", async () => {
  const spans = setupSpans();
  let returned = false;

  const source = {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          return {
            done: false as const,
            value: { candidates: [{ content: { role: "model", parts: [{ text: "never" }] } }] },
          };
        },
        async return() {
          returned = true;

          return { done: true as const, value: undefined };
        },
      };
    },
  };

  const client = wrapGoogleGenAI({
    models: {
      generateContentStream(_params: any) {
        return source;
      },
    },
  });

  const stream = await (client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  }) as any);

  await stream.return(undefined);
  expect(returned).toBe(true);
  const span = await exportedSpan(spans);
  expect(span.name).toBe("chat gemini-2.5-flash");
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
});

test("streaming return forwards the source iterator result", async () => {
  const spans = setupSpans();

  const returnChunk = {
    candidates: [
      {
        content: { role: "model", parts: [{ text: "cleanup" }] },
        finishReason: "STOP",
      },
    ],
    usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 2, totalTokenCount: 3 },
  };

  const source = {
    [Symbol.asyncIterator]() {
      let done = false;

      return {
        async next() {
          if (done) return { done: true as const, value: undefined };
          done = true;

          return {
            done: false as const,
            value: {
              candidates: [{ content: { role: "model", parts: [{ text: "done" }] } }],
              usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 4, totalTokenCount: 7 },
            },
          };
        },
        async return(_value?: any) {
          return { done: false as const, value: returnChunk };
        },
      };
    },
  };

  const client = wrapGoogleGenAI({
    models: {
      generateContentStream(_params: any) {
        return source;
      },
    },
  });

  const stream = await (client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  }) as any);

  await expect(stream.return("stop")).resolves.toEqual({ done: false, value: returnChunk });
  await finishedSpans(spans, 0);
  await expect(stream.next()).resolves.toMatchObject({ done: false });
  await expect(stream.next()).resolves.toEqual({ done: true, value: undefined });
  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "cleanupdone" }] },
  ]);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(7);
});

test("streaming next forwards values to the source iterator", async () => {
  const spans = setupSpans();
  const seen: object[] = [];

  const source = {
    [Symbol.asyncIterator]() {
      return {
        async next(value?: any) {
          seen.push(value);

          return {
            done: false as const,
            value: {
              candidates: [{ content: { role: "model", parts: [{ text: String(value) }] } }],
            },
          };
        },
        async return() {
          return { done: true as const, value: undefined };
        },
      };
    },
  };

  const client = wrapGoogleGenAI({
    models: {
      generateContentStream(_params: any) {
        return source;
      },
    },
  });

  const stream = await (client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  }) as any);

  await expect(stream.next("resume")).resolves.toMatchObject({ done: false });
  await stream.return(undefined);
  expect(seen).toEqual(["resume"]);
  const span = await exportedSpan(spans);
  expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "model", parts: [{ text: "resume" }] },
  ]);
});

test("streaming return failure records a non-Error rejection", async () => {
  const spans = setupSpans();
  const cleanupFailure = "cleanup failed";

  const source = {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          return { done: false as const, value: { candidates: [] } };
        },
        async return() {
          throw cleanupFailure;
        },
      };
    },
  };

  const client = wrapGoogleGenAI({
    models: {
      generateContentStream<T>(_params: T) {
        return Promise.resolve(source);
      },
    },
  });

  const stream = await client.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Stream",
  });

  const iterator = stream[Symbol.asyncIterator]();

  if (!iterator.return) throw new Error("stream iterator does not support return");

  await expect(iterator.return()).rejects.toBe(cleanupFailure);
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("Error");
  const exception = span.events.find((event) => event.name === "exception");
  expect(exception?.attributes?.["exception.message"]).toBe(cleanupFailure);
});

test("API error 400 rethrows ApiError and records error span", async () => {
  const spans = setupSpans();
  const fake = createFakeFetch(jsonErrorResponse(400, "bad request"));
  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await expect(
    client.models.generateContent({ model: "gemini-2.5-flash", contents: "fail" }),
  ).rejects.toMatchObject({ name: "ApiError", status: 400 });
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_ERROR);
  expect(span.attributes["error.type"]).toBe("ApiError");
});

test("non-Error unary failures end spans and preserve the thrown values", async () => {
  const spans = setupSpans();
  const rejected = "unary rejected";

  const rejecting = wrapGoogleGenAI({
    models: {
      generateContent<T>(_params: T) {
        return Promise.reject(rejected);
      },
    },
  });

  await expect(
    rejecting.models.generateContent({ model: "gemini-2.5-flash", contents: "fail" }),
  ).rejects.toBe(rejected);

  const thrown = { message: "unary threw" };

  const throwing = wrapGoogleGenAI({
    models: {
      generateContent<T>(_params: T) {
        throw thrown;
      },
    },
  });

  let caught = false;

  try {
    throwing.models.generateContent({ model: "gemini-2.5-flash", contents: "fail" });
  } catch (error) {
    caught = true;
    expect(error).toBe(thrown);
  }

  expect(caught).toBe(true);

  const finished = await finishedSpans(spans, 2);
  expect(finished.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
  expect(
    finished.map(
      (span) =>
        span.events.find((event) => event.name === "exception")?.attributes?.["exception.message"],
    ),
  ).toEqual(["unary rejected", "[object Object]"]);
});

test("non-Error stream setup failures end spans and preserve the thrown values", async () => {
  const spans = setupSpans();
  const rejected = "stream setup rejected";

  const rejecting = wrapGoogleGenAI({
    models: {
      generateContentStream<T>(_params: T) {
        return Promise.reject(rejected);
      },
    },
  });

  await expect(
    rejecting.models.generateContentStream({ model: "gemini-2.5-flash", contents: "fail" }),
  ).rejects.toBe(rejected);

  const thrown = { message: "stream setup threw" };

  const throwing = wrapGoogleGenAI({
    models: {
      generateContentStream<T>(_params: T) {
        throw thrown;
      },
    },
  });

  let caught = false;

  try {
    throwing.models.generateContentStream({ model: "gemini-2.5-flash", contents: "fail" });
  } catch (error) {
    caught = true;
    expect(error).toBe(thrown);
  }

  expect(caught).toBe(true);

  const finished = await finishedSpans(spans, 2);
  expect(finished.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
  expect(
    finished.map(
      (span) =>
        span.events.find((event) => event.name === "exception")?.attributes?.["exception.message"],
    ),
  ).toEqual(["stream setup rejected", "[object Object]"]);
});

test("blocked prompt records block attrs without output", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      promptFeedback: {
        blockReason: "SAFETY",
        blockReasonMessage: "blocked for safety",
        safetyRatings: [{ category: "HARM_CATEGORY_DANGEROUS_CONTENT", probability: "HIGH" }],
      },
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await client.models.generateContent({ model: "gemini-2.5-flash", contents: "blocked" });
  const span = await exportedSpan(spans);
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
  expect(span.attributes["google_genai.response.block_reason"]).toBe("SAFETY");
  expect(span.attributes["google_genai.response.block_reason_message"]).toBe("blocked for safety");
  expect(jsonAttr(span, "google_genai.response.prompt_safety_ratings")).toEqual([
    { category: "HARM_CATEGORY_DANGEROUS_CONTENT", probability: "HIGH" },
  ]);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("embedContent maps embedding span fields without output", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      embeddings: [
        { values: [0.1, 0.2, 0.3], statistics: { tokenCount: 5 } },
        { values: [0.4, 0.5, 0.6], statistics: { tokenCount: 7 } },
      ],
      metadata: { billableCharacterCount: 42 },
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = clientWith();
  await client.models.embedContent({
    model: "gemini-embedding-001",
    contents: ["one", "two"],
    config: { taskType: "RETRIEVAL_DOCUMENT", outputDimensionality: 3 },
  });
  const span = await exportedSpan(spans);
  expect(span.name).toBe("embeddings gemini-embedding-001");
  expect(span.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(span.attributes["google_genai.response.embedding_count"]).toBe(2);
  expect(span.attributes["google_genai.response.embedding_dimensions"]).toBe(3);
  expect(span.attributes["google_genai.request.task_type"]).toBe("RETRIEVAL_DOCUMENT");
  expect(span.attributes["google_genai.request.output_dimensionality"]).toBe(3);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(12);
  expect(span.attributes["google_genai.usage.billable_characters"]).toBe(42);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
});

test("chats sendMessage emits span input with history for wrap-before-create and create-before-wrap", async () => {
  const responseBody = {
    candidates: [
      {
        content: { role: "model", parts: [{ text: "Hi back" }] },
        finishReason: "STOP",
      },
    ],
  };

  const history = [{ role: "user", parts: [{ text: "Hi" }] }];
  const spans = setupSpans();
  const fakeBefore = createFakeFetch(jsonResponse(responseBody), jsonResponse(responseBody));
  vi.stubGlobal("fetch", fakeBefore.fetch);
  const wrappedFirst = clientWith();

  const chatBefore = wrappedFirst.chats.create({
    model: "gemini-2.5-flash",
    history,
  });

  await chatBefore.sendMessage({ message: "Again" });
  const spanBefore = (await finishedSpans(spans, 1))[0]!;
  expect(messagesAttr(spanBefore, "gen_ai.input.messages")).toEqual([
    ...history,
    { role: "user", parts: [{ text: "Again" }] },
  ]);
  await shutdown();

  const spansAfter = setupSpans();
  const fakeAfter = createFakeFetch(jsonResponse(responseBody));
  vi.stubGlobal("fetch", fakeAfter.fetch);
  const raw = new GoogleGenAI({ apiKey: "test" });
  const chatAfter = raw.chats.create({ model: "gemini-2.5-flash", history });
  wrapGoogleGenAI(raw);
  await chatAfter.sendMessage({ message: "Later" });
  const spanAfter = (await finishedSpans(spansAfter, 1))[0]!;
  expect(messagesAttr(spanAfter, "gen_ai.input.messages")).toEqual([
    ...history,
    { role: "user", parts: [{ text: "Later" }] },
  ]);
});

test("vertex clients report gcp.vertex_ai provider", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = wrapGoogleGenAI(new GoogleGenAI({ vertexai: true, apiKey: "test" }));
  await client.models.generateContent({ model: "gemini-2.5-flash", contents: "hi" });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.provider.name"]).toBe("gcp.vertex_ai");
});

test("double wrapGoogleGenAI emits one span per call", async () => {
  const spans = setupSpans();

  const fake = createFakeFetch(
    jsonResponse({
      candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
    }),
    jsonResponse({
      candidates: [{ content: { role: "model", parts: [{ text: "ok2" }] }, finishReason: "STOP" }],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const once = wrapGoogleGenAI(new GoogleGenAI({ apiKey: "test" }));
  const twice = wrapGoogleGenAI(once);
  await twice.models.generateContent({ model: "gemini-2.5-flash", contents: "one" });
  await twice.models.generateContent({ model: "gemini-2.5-flash", contents: "two" });
  expect((await finishedSpans(spans, 2)).length).toBe(2);
});

test("wrapped calls fail open when telemetry mapping throws", async () => {
  const spans = setupSpans();

  const response = {
    text: "ok",
    get candidates() {
      throw new Error("response mapper failed");
    },
  };

  const client = wrapGoogleGenAI({
    models: {
      generateContent(_params: any) {
        return response;
      },
    },
  });

  const params = {
    model: "gemini-2.5-flash",
    contents: "hi",
    get config() {
      throw new Error("request mapper failed");
    },
  };

  const result = client.models.generateContent(params);
  expect(result).toBe(response);
  const span = await exportedSpan(spans);
  expect(span.name).toBe("chat gemini-2.5-flash");
  expect(span.status.code).toBe(SPAN_STATUS_UNSET);
});

test("wrapped calls fail open when telemetry is not initialized", async () => {
  const fake = createFakeFetch(
    jsonResponse({
      candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
    }),
  );

  vi.stubGlobal("fetch", fake.fetch);
  const client = wrapGoogleGenAI(new GoogleGenAI({ apiKey: "test" }));

  const response = await client.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "hi",
  });

  expect(response.text).toBe("ok");
});

test("usage modality details map text, image, and audio without video attributes", async () => {
  const spans = setupSpans();

  const client = wrapGoogleGenAI({
    models: {
      generateContent(_params: unknown) {
        return {
          usageMetadata: {
            promptTokensDetails: [
              { modality: "TEXT", tokenCount: 2 },
              { modality: "IMAGE", tokenCount: 3 },
              { modality: "VIDEO", tokenCount: 99 },
            ],
            candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 4 }],
            cacheTokensDetails: [
              { modality: "TEXT", tokenCount: 5 },
              { modality: "IMAGE", tokenCount: 6 },
              { modality: "AUDIO", tokenCount: 7 },
            ],
          },
        };
      },
    },
  });

  client.models.generateContent({ model: "gemini", contents: "hello" });
  const span = await exportedSpan(spans);
  expect(span.attributes["gen_ai.usage.text.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.image.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.audio.output_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.text.cache_read.input_tokens"]).toBe(5);
  expect(span.attributes["gen_ai.usage.image.cache_read.input_tokens"]).toBe(6);
  expect(span.attributes["gen_ai.usage.audio.cache_read.input_tokens"]).toBe(7);
  expect(
    Object.keys(span.attributes).some((key) => key.includes("video") && key.includes("usage")),
  ).toBe(false);
});

test.each(["generateImages", "editImage", "upscaleImage"] as const)(
  "%s creates image generation spans without binary payloads",
  async (method) => {
    const spans = setupSpans();
    const binary = "base64-secret-image";

    const client = wrapGoogleGenAI({
      vertexai: true,
      models: {
        [method](_params: unknown) {
          return {
            generatedImages: [{ image: { imageBytes: binary, gcsUri: "gs://bucket/output.png" } }],
          };
        },
      },
    });

    client.models[method]({
      model: "imagen-3",
      prompt: "a lighthouse",
      image: { imageBytes: binary, mimeType: "image/png" },
      referenceImages: [{ referenceImage: { imageBytes: binary } }],
      upscaleFactor: "x2",
      config: { numberOfImages: 1, outputGcsUri: "gs://bucket", httpOptions: { headers: {} } },
    });
    const span = await exportedSpan(spans);
    expect(span.attributes["gen_ai.operation.name"]).toBe("generate_content");
    expect(span.attributes["gen_ai.output.type"]).toBe("image");
    expect(span.attributes["gen_ai.provider.name"]).toBe("gcp.vertex_ai");
    expect(span.attributes["google_genai.response.image_count"]).toBe(1);
    expect(messagesAttr(span, "gen_ai.output.messages")).toEqual([
      { type: "image", uri: "gs://bucket/output.png" },
    ]);
    expect(span.attributes["google_genai.response.image_uris"]).toBeUndefined();
    expect(JSON.stringify(span.attributes)).not.toContain(binary);
  },
);

test.each([
  { captureInput: false, captureOutput: false, expected: undefined },
  {
    mask: (_value: unknown, context: { key: string }) => `redacted:${context.key}`,
    expected: "redacted",
  },
])("media content obeys capture privacy controls %#", async (options) => {
  const spans = setupSpans(options);

  const client = wrapGoogleGenAI({
    models: {
      generateImages(_params: unknown) {
        return { generatedImages: [{ image: { gcsUri: "gs://secret/output.png" } }] };
      },
    },
  });

  client.models.generateImages({
    model: "imagen-3",
    prompt: "secret prompt",
    image: { gcsUri: "gs://secret/input.png", mimeType: "image/png" },
    config: {
      negativePrompt: "secret negative prompt",
      outputGcsUri: "gs://secret/output-prefix",
      pubsubTopic: "projects/secret/topics/private",
    },
  });

  const span = await exportedSpan(spans);
  const serialized = JSON.stringify(span.attributes);

  expect(serialized).not.toContain("secret prompt");
  expect(serialized).not.toContain("gs://secret");
  expect(serialized).not.toContain("projects/secret");
  expect(span.attributes["google_genai.response.image_count"]).toBe(1);

  if (options.expected) {
    expect(span.attributes["gen_ai.input.messages"]).toContain(options.expected);
    expect(span.attributes["gen_ai.output.messages"]).toContain(options.expected);
  } else {
    expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  }
});

test("generateVideos traces submit and tracked polling idempotently", async () => {
  const spans = setupSpans();
  const operation = { name: "operations/video-1", done: false };

  const models = {
    generateVideos(_params: unknown) {
      return Promise.resolve(operation);
    },
  };

  const operations = {
    get(_params: unknown) {
      return Promise.resolve({
        name: operation.name,
        done: true,
        response: { generatedVideos: [{ video: { uri: "gs://bucket/video.mp4" } }] },
      });
    },
  };

  const client = { models, operations };

  expect(wrapGoogleGenAI(client)).toBe(client);
  expect(wrapGoogleGenAI(client)).toBe(client);
  const submitted = await client.models.generateVideos({ model: "veo", prompt: "ocean" });
  const completed = await client.operations.get({ operation: submitted });
  expect(completed.done).toBe(true);
  const exported = await finishedSpans(spans, 2);
  expect(exported[0]!.attributes["gen_ai.response.id"]).toBe(operation.name);
  expect(messagesAttr(exported[1]!, "gen_ai.output.messages")).toEqual([
    { type: "video", uri: "gs://bucket/video.mp4" },
  ]);
  expect(exported[1]!.attributes["google_genai.response.video_uris"]).toBeUndefined();
  expect(exported.every((span) => span.attributes["gen_ai.output.type"] === "video")).toBe(true);
});

test("video operation tracking is identity-only and client-local", async () => {
  const spans = setupSpans();
  const operationName = "operations/shared";

  const makeClient = () => ({
    models: {
      generateVideos(_params?: unknown) {
        return { name: operationName, done: false };
      },
    },
    operations: {
      get(_params: unknown) {
        return { name: operationName, done: true };
      },
    },
  });

  const first = wrapGoogleGenAI(makeClient());
  const second = wrapGoogleGenAI(makeClient());

  const submitted = first.models.generateVideos();
  second.operations.get({ operation: { name: operationName } });
  first.operations.get({ operation: submitted });
  first.operations.get({ operation: { name: operationName } });

  const exported = await finishedSpans(spans, 2);
  expect(exported.map((span) => span.name)).toEqual([
    "generate_content unknown",
    "generate_content video operation",
  ]);
});

test.each(["get", "getVideosOperation"] as const)(
  "operations.%s tracks each successful non-terminal response for the next poll",
  async (method) => {
    const spans = setupSpans();
    const submitted = { name: "operations/video-chain", done: false };
    const pending = { name: submitted.name, done: false };
    const completed = { name: submitted.name, done: true };

    const operationMethods = {
      [method]({ operation }: { operation: object }) {
        return operation === submitted ? pending : completed;
      },
    };

    const client = wrapGoogleGenAI({
      models: { generateVideos: () => submitted },
      operations: operationMethods,
    });

    client.models.generateVideos();
    expect(client.operations[method]({ operation: submitted })).toBe(pending);
    expect(client.operations[method]({ operation: pending })).toBe(completed);
    client.operations[method]({ operation: { name: submitted.name } });

    const exported = await finishedSpans(spans, 3);
    expect(exported.map((span) => span.name)).toEqual([
      "generate_content unknown",
      "generate_content video operation",
      "generate_content video operation",
    ]);
  },
);

test.each(["get", "getVideosOperation"] as const)(
  "operations.%s tracks a non-terminal response when telemetry mapping fails",
  async (method) => {
    const spans = setupSpans();
    const submitted = { name: "operations/video-mapping-failure", done: false };

    const pending = Object.defineProperty({ name: submitted.name, done: false }, "response", {
      get() {
        throw new Error("telemetry response getter");
      },
    });

    const completed = { name: submitted.name, done: true };

    const client = wrapGoogleGenAI({
      models: { generateVideos: () => submitted },
      operations: {
        [method]({ operation }: { operation: object }) {
          return operation === submitted ? pending : completed;
        },
      },
    });

    client.models.generateVideos();
    expect(client.operations[method]({ operation: submitted })).toBe(pending);
    expect(client.operations[method]({ operation: pending })).toBe(completed);

    const exported = await finishedSpans(spans, 3);
    expect(exported.map((span) => span.name)).toEqual([
      "generate_content unknown",
      "generate_content video operation",
      "generate_content video operation",
    ]);
  },
);

test("content telemetry strips binary media from unary, streaming, and AFC history", async () => {
  const spans = setupSpans();
  const binary = "binary-secret";

  const content = {
    role: "model",
    parts: [
      { text: "kept" },
      { inlineData: { mimeType: "image/png", data: binary, displayName: "preview" } },
      { inline_data: { mime_type: "image/jpeg", data: binary, display_name: "snake preview" } },
      { image: { imageBytes: binary, mimeType: "image/png" } },
      { video: { videoBytes: binary, uri: "gs://bucket/video.mp4" } },
      {
        functionResponse: {
          name: "lookup",
          response: {
            data: "useful result",
            mimeType: "application/json",
            inlineData: { data: "nested tool result", mimeType: "application/json" },
            image: { imageBytes: "nested image result" },
            video: { videoBytes: "nested video result" },
          },
        },
      },
    ],
  };

  const unary = wrapGoogleGenAI({
    models: {
      generateContent(_params: unknown) {
        return {
          candidates: [{ content }],
          automaticFunctionCallingHistory: [{ role: "user", parts: content.parts }],
        };
      },
    },
  });

  const streaming = wrapGoogleGenAI({
    models: {
      async generateContentStream(_params: unknown) {
        return (async function* () {
          yield {
            candidates: [{ content }],
            automaticFunctionCallingHistory: [{ role: "user", parts: content.parts }],
          };
        })();
      },
    },
  });

  unary.models.generateContent({
    contents: [{ role: "user", parts: content.parts }],
    config: {
      systemInstruction: {
        parts: [
          { text: "system kept" },
          { inlineData: { mimeType: "image/png", data: binary, displayName: "system preview" } },
        ],
      },
    },
  });

  const stream = await streaming.models.generateContentStream({
    contents: content,
  });

  for await (const _chunk of stream) {
  }

  const exported = await finishedSpans(spans, 2);

  for (const span of exported) {
    const serialized = JSON.stringify(span.attributes);
    expect(serialized).not.toContain(binary);
    expect(serialized).toContain("kept");
    expect(serialized).toContain("preview");
    expect(serialized).toContain("snake preview");
    expect(serialized).toContain("gs://bucket/video.mp4");
    expect(serialized).toContain("useful result");
    expect(serialized).toContain("application/json");
    expect(serialized).toContain("nested tool result");
    expect(serialized).toContain("nested image result");
    expect(serialized).toContain("nested video result");
  }

  const unarySerialized = JSON.stringify(exported[0]!.attributes);
  expect(unarySerialized).toContain("system kept");
  expect(unarySerialized).toContain("system preview");
});

test.each(["get", "getVideosOperation"] as const)(
  "operations.%s preserves throwing getters and successful responses",
  async (method) => {
    const spans = setupSpans();
    const operation = { name: "operations/getters", done: false };

    const response = Object.defineProperty({ name: operation.name }, "done", {
      enumerable: true,
      get() {
        throw new Error("response getter");
      },
    });

    let calls = 0;

    const client = wrapGoogleGenAI({
      models: { generateVideos: (_params: unknown) => operation },
      operations: {
        [method](_params: unknown) {
          calls += 1;

          return Promise.resolve(response);
        },
      },
    });

    client.models.generateVideos({ model: "veo" });

    const request = Object.defineProperty({}, "operation", {
      get() {
        throw new Error("request getter");
      },
    });

    await expect(client.operations[method](request)).resolves.toBe(response);
    await expect(client.operations[method]({ operation })).resolves.toBe(response);
    expect(calls).toBe(2);
    await finishedSpans(spans, 2);
  },
);

test("media methods preserve rejected and synchronously thrown failures", async () => {
  const spans = setupSpans();
  const rejected = new Error("image rejected");
  const thrown = new Error("video threw");

  const client = wrapGoogleGenAI({
    models: {
      generateImages(_params: unknown) {
        return Promise.reject(rejected);
      },
      generateVideos(_params: unknown) {
        throw thrown;
      },
    },
  });

  await expect(client.models.generateImages({ model: "imagen", prompt: "fail" })).rejects.toBe(
    rejected,
  );
  expect(() => client.models.generateVideos({ model: "veo", prompt: "fail" })).toThrow(thrown);
  const exported = await finishedSpans(spans, 2);
  expect(exported.every((span) => span.status.code === SPAN_STATUS_ERROR)).toBe(true);
});

test("capture traversal is bounded, cycle-aware, binary-safe, and preserves provider outcomes", async () => {
  const spans = setupSpans();
  const shared = { text: "shared" };

  interface CaptureFixture {
    role?: string;
    parts?: Array<{ text: string }>;
    inlineData?: { mimeType: string; data: string };
    self?: CaptureFixture;
    next?: CaptureFixture;
  }

  const contents: CaptureFixture = {
    role: "user",
    parts: Array.from({ length: 2_000 }, () => shared),
    inlineData: { mimeType: "image/png", data: "SECRET_BINARY" },
  };

  contents.self = contents;
  let deep = contents;

  for (let index = 0; index < 100; index += 1) {
    const next: CaptureFixture = {};
    deep.next = next;
    deep = next;
  }

  const response = {
    candidates: [{ content: { role: "model", parts: contents.parts } }],
  };

  const generateContent = vi.fn((_params: unknown) => response);
  const client = wrapGoogleGenAI({ models: { generateContent } });

  expect(client.models.generateContent({ model: "gemini", contents })).toBe(response);
  expect(generateContent).toHaveBeenCalledOnce();
  const span = await exportedSpan(spans);
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(String(span.attributes["gen_ai.input.messages"])).not.toContain("SECRET_BINARY");
  expect(String(span.attributes["gen_ai.input.messages"]).length).toBeLessThan(65_536);
  expect(String(span.attributes["gen_ai.output.messages"]).length).toBeLessThan(65_536);
});
