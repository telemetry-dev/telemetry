import {
  ApplyGuardrailCommand,
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
  type ConverseCommandInput,
} from "@aws-sdk/client-bedrock-runtime";
import { afterEach, expect, test, vi } from "vitest";

import { instrumentBedrock, uninstrumentBedrock, wrapBedrock } from "../src/index.ts";
import { FakeClient, bytes, collect, jsonAttr, setup, streamOf, teardown } from "./helpers.ts";

afterEach(async () => {
  uninstrumentBedrock();
  await teardown();
});

test("Converse captures normalized messages, usage, metadata, request id, and sampling", async () => {
  const spans = setup();
  const request: ConverseCommandInput = {
    modelId: "anthropic.claude-3-5-haiku-20241022-v1:0",
    messages: [
      {
        role: "user",
        content: [
          { text: "hello" },
          { image: { format: "png", source: { bytes: new Uint8Array([1, 2, 3]) } } },
        ],
      },
    ],
    system: [{ text: "be terse" }],
    inferenceConfig: { temperature: 0.2, topP: 0.8, maxTokens: 64, stopSequences: ["stop"] },
  };
  const requestSnapshot = structuredClone(request);
  const client = wrapBedrock(
    new FakeClient([
      {
        output: { message: { role: "assistant", content: [{ text: "hi" }] } },
        stopReason: "end_turn",
        usage: {
          inputTokens: 3,
          outputTokens: 4,
          totalTokens: 7,
          cacheReadInputTokens: 1,
          cacheWriteInputTokens: 2,
        },
        metrics: { latencyMs: 12 },
        trace: { promptRouter: { invokedModelId: "routed-model" } },
        $metadata: { requestId: "req-123", attempts: 2, httpStatusCode: 200, totalRetryDelay: 7 },
      },
    ]),
  );

  await client.send(new ConverseCommand(request));

  expect(request).toEqual(requestSnapshot);
  expect(client.calls[0]).toEqual(requestSnapshot);
  const [span] = spans.getFinishedSpans();
  expect(span.name).toBe("chat anthropic.claude-3-5-haiku-20241022-v1:0");
  expect(span.attributes["gen_ai.operation.name"]).toBe("chat");
  expect(span.attributes["gen_ai.provider.name"]).toBe("amazon-bedrock");
  expect(span.attributes["gen_ai.request.model"]).toBe(request.modelId);
  expect(span.attributes["gen_ai.response.id"]).toBe("req-123");
  expect(span.attributes["aws.request.attempts"]).toBe(2);
  expect(span.attributes["aws.http.status_code"]).toBe(200);
  expect(span.attributes["aws.request.total_retry_delay_ms"]).toBe(7);
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.response.model"]).toBe("routed-model");
  expect(span.attributes["td.metadata.server_latency_ms"]).toBe("12");
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.2);
  expect(jsonAttr(span, "gen_ai.input.messages")).toEqual([
    {
      role: "user",
      parts: [
        { type: "text", content: "hello" },
        { type: "blob", modality: "image", mime_type: "image/png" },
      ],
    },
  ]);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", parts: [{ type: "text", content: "hi" }], finish_reason: "end_turn" },
  ]);
});

test("Converse normalizes tool calls, tool results, and reasoning", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        output: {
          message: {
            role: "assistant",
            content: [
              { toolUse: { toolUseId: "tool-1", name: "weather", input: { city: "Paris" } } },
              { reasoningContent: { reasoningText: { text: "Need weather." } } },
            ],
          },
        },
        stopReason: "tool_use",
      },
    ]),
  );

  await client.send(
    new ConverseCommand({
      modelId: "anthropic.claude-3-sonnet",
      messages: [
        { role: "user", content: [{ text: "weather" }] },
        {
          role: "user",
          content: [
            {
              toolResult: {
                toolUseId: "tool-1",
                content: [
                  { text: "sunny" },
                  { image: { format: "png", source: { bytes: new Uint8Array([1, 2, 3]) } } },
                ],
              },
            },
          ],
        },
      ],
    }),
  );

  const span = spans.getFinishedSpans()[0]!;
  expect(jsonAttr(span, "gen_ai.input.messages")[1].parts[0]).toEqual({
    type: "tool_call_response",
    id: "tool-1",
    response: [
      { type: "text", content: "sunny" },
      { type: "blob", modality: "image", mime_type: "image/png" },
    ],
  });
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts).toEqual([
    { type: "tool_call", id: "tool-1", name: "weather", arguments: { city: "Paris" } },
    { type: "reasoning", content: "Need weather." },
  ]);
});
test("Converse normalizes citationsContent as cited text", async () => {
  const spans = setup();
  const citation = {
    title: "source",
    location: { s3Location: { uri: "s3://bucket/doc.txt" } },
  };
  const client = wrapBedrock(
    new FakeClient([
      {
        output: {
          message: {
            role: "assistant",
            content: [
              {
                citationsContent: {
                  content: [{ text: "grounded answer" }],
                  citations: [citation],
                },
              },
            ],
          },
        },
      },
    ]),
  );

  await client.send(new ConverseCommand({ modelId: "anthropic.claude", messages: [] }));

  const [span] = spans.getFinishedSpans();
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0]).toEqual({
    type: "text",
    content: "grounded answer",
    citations: [citation],
  });
});

test("ConverseStream accumulates output and ends partial on early break", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        stream: streamOf([
          { messageStart: { role: "assistant" } },
          { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "Hello " } } },
          { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "world" } } },
          { messageStop: { stopReason: "end_turn" } },
          { metadata: { usage: { inputTokens: 5, outputTokens: 2 } } },
        ]),
        $metadata: { requestId: "req-stream" },
      },
    ]),
  );
  const response = (await client.send(
    new ConverseStreamCommand({ modelId: "anthropic.claude", messages: [] }),
  )) as { stream: AsyncIterable<unknown> };
  await collect(response.stream, 3);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.response.id"]).toBe("req-stream");
  expect(span.attributes["gen_ai.response.finish_reasons"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.time_to_first_chunk"]).toBeGreaterThanOrEqual(0);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual([
    { role: "assistant", parts: [{ type: "text", content: "Hello world" }] },
  ]);
});

test("ConverseStream records modeled errors and non-text blocks", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        stream: streamOf([
          { messageStart: { role: "assistant" } },
          {
            contentBlockStart: {
              contentBlockIndex: 0,
              start: { image: { format: "png" } },
            },
          },
          {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { image: { source: { bytes: new Uint8Array([1, 2, 3]) } } },
            },
          },
          {
            contentBlockDelta: {
              contentBlockIndex: 0,
              delta: { citation: { title: "doc", source: "kb" } },
            },
          },
          {
            modelStreamErrorException: {
              name: "ModelStreamErrorException",
              message: "stream failed",
              $metadata: { requestId: "modeled-stream-error", httpStatusCode: 424 },
            },
          },
        ]),
      },
    ]),
  );
  const response = (await client.send(
    new ConverseStreamCommand({ modelId: "anthropic.claude", messages: [] }),
  )) as { stream: AsyncIterable<unknown> };
  await collect(response.stream);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["error.type"]).toBe("ModelStreamErrorException");
  expect(span.attributes["gen_ai.response.id"]).toBe("modeled-stream-error");
  expect(span.attributes["aws.http.status_code"]).toBe(424);
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0]).toEqual({
    type: "blob",
    modality: "image",
    mime_type: "image/png",
  });
});

test("ConverseStream records stream errors with partial output", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        stream: streamOf(
          [
            { messageStart: { role: "assistant" } },
            { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "partial" } } },
          ],
          2,
          Object.assign(new Error("stream boom"), {
            name: "ModelStreamError",
            $metadata: { requestId: "stream-error", attempts: 2, httpStatusCode: 424 },
          }),
        ),
      },
    ]),
  );
  const response = (await client.send(
    new ConverseStreamCommand({ modelId: "anthropic.claude", messages: [] }),
  )) as { stream: AsyncIterable<unknown> };
  await expect(collect(response.stream)).rejects.toThrow("stream boom");

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["error.type"]).toBe("ModelStreamError");
  expect(span.attributes["gen_ai.response.id"]).toBe("stream-error");
  expect(span.attributes["aws.http.status_code"]).toBe(424);
  expect(span.attributes["aws.request.attempts"]).toBe(2);
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe("partial");
});

test("InvokeModel captures provider-native bodies and embedding spans", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        body: bytes({
          content: [{ type: "text", text: "answer" }],
          usage: { input_tokens: 9, output_tokens: 4, cache_read_input_tokens: 1 },
          stop_reason: "end_turn",
        }),
        contentType: "application/json",
        $metadata: { requestId: "invoke-1" },
      },
      {
        body: bytes({ embedding: [0.1, 0.2], inputTextTokenCount: 6 }),
        contentType: "application/json",
      },
    ]),
  );

  await client.send(
    new InvokeModelCommand({
      modelId: "anthropic.claude-3-haiku",
      contentType: "application/json",
      body: bytes({ messages: [], max_tokens: 10, top_k: 3 }),
    }),
  );
  await client.send(
    new InvokeModelCommand({
      modelId: "amazon.titan-embed-text-v2:0",
      contentType: "application/json",
      body: bytes({ inputText: "hello" }),
    }),
  );

  const [chat, embedding] = spans.getFinishedSpans();
  expect(chat.attributes["gen_ai.request.top_k"]).toBe(3);
  expect(chat.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(1);
  expect(chat.attributes["gen_ai.response.id"]).toBe("invoke-1");
  expect(embedding.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(embedding.attributes["gen_ai.output.type"]).toBe("embedding");
  expect(embedding.attributes["gen_ai.usage.input_tokens"]).toBe(6);
});

test("InvokeModel captures Nova native sampling fields", async () => {
  const spans = setup();
  const client = wrapBedrock(new FakeClient([{ body: bytes({ output: { message: {} } }) }]));

  await client.send(
    new InvokeModelCommand({
      modelId: "amazon.nova-pro-v1:0",
      contentType: "application/json",
      body: bytes({
        messages: [],
        inferenceConfig: {
          temperature: 0.4,
          topP: 0.9,
          topK: 20,
          maxTokens: 500,
          stopSequences: ["stop"],
        },
      }),
    }),
  );

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.request.temperature"]).toBe(0.4);
  expect(span.attributes["gen_ai.request.top_p"]).toBe(0.9);
  expect(span.attributes["gen_ai.request.top_k"]).toBe(20);
  expect(span.attributes["gen_ai.request.max_tokens"]).toBe(500);
  expect(span.attributes["gen_ai.request.stop_sequences"]).toEqual(["stop"]);
});

test("InvokeModelWithResponseStream preserves optional invocation metrics", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        body: streamOf([
          { chunk: { bytes: bytes({ message: { usage: { input_tokens: 2 } } }) } },
          { chunk: { bytes: bytes({ delta: { text: "hi", stop_reason: "end_turn" } }) } },
          { chunk: { bytes: bytes({ delta: { thinking: "reasoning" } }) } },
          { chunk: { bytes: bytes({ delta: { partial_json: '{"city":' } }) } },
          { chunk: { bytes: bytes({ completion: " legacy" }) } },
          { chunk: { bytes: bytes({ outputs: [{ text: " mistral", stop_reason: "stop" }] }) } },
          {
            chunk: { bytes: bytes({ "amazon-bedrock-invocationMetrics": { inputTokenCount: 1 } }) },
          },
        ]),
      },
    ]),
  );
  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude",
      contentType: "application/json",
      body: bytes({ messages: [], max_tokens: 10 }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(response.body);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(1);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
  expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe("hi legacy mistral");
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 4 },
  });
});

test("InvokeModelWithResponseStream parses each payload once and delivers malformed payloads", async () => {
  const spans = setup();
  const parse = vi.spyOn(JSON, "parse");
  const malformed = new TextEncoder().encode("not-json");

  const events = [
    { chunk: { bytes: bytes({ delta: { text: "hi" } }) } },
    { chunk: { bytes: new Uint8Array() } },
    { chunk: { bytes: bytes({ delta: { text: "there" } }) } },
    { chunk: { bytes: malformed } },
  ];

  const client = wrapBedrock(new FakeClient([{ body: streamOf(events) }]));

  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude",
      contentType: "application/json",
      body: bytes({ messages: [], max_tokens: 10 }),
    }),
  )) as { body: AsyncIterable<unknown> };

  const before = parse.mock.calls.length;

  expect(await collect(response.body)).toEqual(events);
  expect(parse.mock.calls.length - before).toBe(4);
  parse.mockRestore();
  const [span] = spans.getFinishedSpans();
  expect(span).toMatchObject({
    [Symbol.for("telemetry.dev.outputChunkHistogram")]: { count: 2 },
  });
});

test("InvokeModelWithResponseStream captures Titan token counts and Cohere generations", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        body: streamOf([
          { chunk: { bytes: bytes({ outputText: "Titan ", inputTextTokenCount: 5 }) } },
          {
            chunk: {
              bytes: bytes({
                outputText: "done",
                totalOutputTextTokenCount: 2,
                completionReason: "FINISHED",
              }),
            },
          },
        ]),
      },
      {
        body: streamOf([
          { chunk: { bytes: bytes({ generations: [{ text: "Cohere " }] }) } },
          {
            chunk: {
              bytes: bytes({ generations: [{ text: "done", finish_reason: "COMPLETE" }] }),
            },
          },
        ]),
      },
    ]),
  );
  const titan = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "amazon.titan-text-express-v1",
      contentType: "application/json",
      body: bytes({ inputText: "hello" }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(titan.body);
  const cohere = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "cohere.command-text-v14",
      contentType: "application/json",
      body: bytes({ prompt: "hello" }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(cohere.body);

  const [titanSpan, cohereSpan] = spans.getFinishedSpans();
  expect(jsonAttr(titanSpan, "gen_ai.output.messages")[0].parts[0].content).toBe("Titan done");
  expect(titanSpan.attributes["gen_ai.usage.input_tokens"]).toBe(5);
  expect(titanSpan.attributes["gen_ai.usage.output_tokens"]).toBe(2);
  expect(titanSpan.attributes["gen_ai.response.finish_reasons"]).toEqual(["FINISHED"]);
  expect(jsonAttr(cohereSpan, "gen_ai.output.messages")[0].parts[0].content).toBe("Cohere done");
  expect(cohereSpan.attributes["gen_ai.response.finish_reasons"]).toEqual(["COMPLETE"]);
});

test("InvokeModelWithResponseStream normalizes Nova contentBlockDelta chunks", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        body: streamOf([
          { chunk: { bytes: bytes({ contentBlockDelta: { delta: { text: "Nova " } } }) } },
          { chunk: { bytes: bytes({ contentBlockDelta: { delta: { text: "done" } } }) } },
        ]),
      },
    ]),
  );
  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "amazon.nova-pro-v1:0",
      contentType: "application/json",
      body: bytes({ messages: [] }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(response.body);

  const [span] = spans.getFinishedSpans();
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe("Nova done");
});

test("InvokeModelWithResponseStream records modeled stream errors with partial output", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        body: streamOf([
          { chunk: { bytes: bytes({ completion: "partial" }) } },
          {
            modelStreamErrorException: {
              name: "ModelStreamErrorException",
              message: "modeled stream failed",
              $metadata: { requestId: "invoke-modeled-error", httpStatusCode: 424 },
            },
          },
        ]),
      },
    ]),
  );
  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude",
      contentType: "application/json",
      body: bytes({ messages: [], max_tokens: 10 }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(response.body);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["error.type"]).toBe("ModelStreamErrorException");
  expect(span.attributes["gen_ai.response.id"]).toBe("invoke-modeled-error");
  expect(span.attributes["aws.http.status_code"]).toBe(424);
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe("partial");
});

test("ApplyGuardrail captures action metadata", async () => {
  const spans = setup();
  const client = wrapBedrock(
    new FakeClient([
      {
        action: "GUARDRAIL_INTERVENED",
        actionReason: "blocked",
        outputs: [{ text: "blocked" }],
        $metadata: { requestId: "guardrail-1" },
      },
    ]),
  );

  await client.send(
    new ApplyGuardrailCommand({
      guardrailIdentifier: "gr-1",
      guardrailVersion: "1",
      source: "INPUT",
      content: [{ text: { text: "bad" } }],
    }),
  );

  const [span] = spans.getFinishedSpans();
  expect(span.name).toBe("apply_guardrail gr-1");
  expect(span.attributes["gen_ai.operation.name"]).toBe("function");
  expect(span.attributes["td.metadata.guardrail_action"]).toBe("GUARDRAIL_INTERVENED");
  expect(span.attributes["td.metadata.guardrail_action_reason"]).toBe("blocked");
});

test("errors, fail-open, idempotency, and prototype restore behave", async () => {
  const spans = setup();
  const error = Object.assign(new Error("throttled"), {
    name: "ThrottlingException",
    $metadata: { requestId: "err-1", attempts: 2, httpStatusCode: 429, totalRetryDelay: 25 },
  });
  const client = wrapBedrock(wrapBedrock(new FakeClient([error])));

  await expect(client.send(new ConverseCommand({ modelId: "m", messages: [] }))).rejects.toThrow(
    "throttled",
  );
  expect(spans.getFinishedSpans()).toHaveLength(1);
  expect(spans.getFinishedSpans()[0]!.attributes["error.type"]).toBe("ThrottlingException");
  expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.response.id"]).toBe("err-1");
  expect(spans.getFinishedSpans()[0]!.attributes["aws.http.status_code"]).toBe(429);
  expect(spans.getFinishedSpans()[0]!.attributes["aws.request.total_retry_delay_ms"]).toBe(25);

  await teardown();
  const disabled = wrapBedrock(new FakeClient([{ output: { message: { content: [] } } }]));
  await disabled.send(new ConverseCommand({ modelId: "m", messages: [] }));
  expect(disabled.calls).toHaveLength(1);

  instrumentBedrock();
  uninstrumentBedrock();
});

test("callback-style send creates a span while returning undefined", () => {
  const spans = setup();
  let callbackData: unknown;
  const client = wrapBedrock({
    calls: 0,
    send<TCommand>(_command: TCommand, ...rest: unknown[]): undefined {
      this.calls += 1;
      const callback = rest.find(
        (arg): arg is (cause: unknown, data?: { output: object; $metadata: object }) => void =>
          arg instanceof Function,
      );
      callback?.(undefined, {
        output: { message: { role: "assistant", content: [{ text: "ok" }] } },
        $metadata: { requestId: "callback-1" },
      });
      return undefined;
    },
  });

  const returned = client.send(
    new ConverseCommand({ modelId: "m", messages: [] }),
    <TError, TData>(_err: TError, data?: TData) => {
      callbackData = data;
    },
  );

  expect(returned).toBeUndefined();
  expect(client.calls).toBe(1);
  expect(callbackData).toEqual({
    output: { message: { role: "assistant", content: [{ text: "ok" }] } },
    $metadata: { requestId: "callback-1" },
  });
  const [span] = spans.getFinishedSpans();
  expect(span.name).toBe("chat m");
  expect(span.attributes["gen_ai.response.id"]).toBe("callback-1");
  expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe("ok");
});

test("global instrumentation is idempotent", async () => {
  const spans = setup();
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    BedrockRuntimeClient.prototype,
    "send",
  );
  const fakeSend = async () => ({
    output: { message: { content: [] } },
    $metadata: { requestId: "global-1" },
  });

  Object.defineProperty(BedrockRuntimeClient.prototype, "send", {
    configurable: true,
    writable: true,
    value: fakeSend,
  });

  try {
    instrumentBedrock();
    instrumentBedrock();
    const client = new BedrockRuntimeClient({
      region: "us-east-1",
      credentials: { accessKeyId: "test", secretAccessKey: "test" },
    });
    await client.send(new ConverseCommand({ modelId: "m", messages: [] }));
    expect(spans.getFinishedSpans()).toHaveLength(1);
    expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.response.id"]).toBe("global-1");
    const telemetryWrapper = Object.getOwnPropertyDescriptor(
      BedrockRuntimeClient.prototype,
      "send",
    )?.value;
    const laterWrapper = async () => ({ output: { message: { content: [] } } });
    Object.defineProperty(BedrockRuntimeClient.prototype, "send", {
      configurable: true,
      writable: true,
      value: laterWrapper,
    });
    uninstrumentBedrock();
    expect(Object.getOwnPropertyDescriptor(BedrockRuntimeClient.prototype, "send")?.value).toBe(
      laterWrapper,
    );
    Object.defineProperty(BedrockRuntimeClient.prototype, "send", {
      configurable: true,
      writable: true,
      value: telemetryWrapper,
    });
    uninstrumentBedrock();
    expect(Object.getOwnPropertyDescriptor(BedrockRuntimeClient.prototype, "send")?.value).toBe(
      fakeSend,
    );
  } finally {
    uninstrumentBedrock();
    if (originalDescriptor) {
      Object.defineProperty(BedrockRuntimeClient.prototype, "send", originalDescriptor);
    }
  }
});
