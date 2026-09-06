import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
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

function erroringSseResponse(chunk: JsonValue, error: Error): Response {
  const encoder = new TextEncoder();
  let sent = false;
  const body = new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        if (!sent) {
          sent = true;
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
    const bodyText =
      Object.prototype.toString.call(init?.body) === "[object String]"
        ? String(init?.body)
        : undefined;
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

function setupSpans(): InMemorySpanExporter {
  const spanExporter = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "google-genai-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter },
  );
  return spanExporter;
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
  expect(messagesAttr(span, "gen_ai.input.messages")).toEqual([{ role: "user", parts: contents }]);
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
