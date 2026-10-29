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

import { type ChatMessage, getPrompt, type MaskFn } from "@telemetry-dev/sdk";
import * as telemetrySdk from "@telemetry-dev/sdk";

import { instrumentBedrock, uninstrumentBedrock, wrapBedrock } from "../src/index.ts";
import {
  FakeClient,
  bytes,
  collect,
  jsonAttr,
  jsonRequestHandler,
  setup,
  streamOf,
  teardown,
} from "./helpers.ts";

afterEach(async () => {
  uninstrumentBedrock();
  await teardown();
  vi.restoreAllMocks();
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
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(3 + 1 + 2);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(3 + 1 + 2 + 4);
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
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

function textStream(text: string, repeat = 1) {
  return streamOf([
    { messageStart: { role: "assistant" } },
    ...Array.from({ length: repeat }, () => ({
      contentBlockDelta: { contentBlockIndex: 0, delta: { text } },
    })),
    { messageStop: { stopReason: "end_turn" } },
  ]);
}

async function streamedConverse(stream: AsyncIterable<unknown>) {
  const client = wrapBedrock(new FakeClient([{ stream }]));

  const response = (await client.send(
    new ConverseStreamCommand({ modelId: "anthropic.claude", messages: [] }),
  )) as { stream: AsyncIterable<unknown> };

  await collect(response.stream);
}

function recordingMask(seen: string[]): MaskFn {
  return (value) => {
    seen.push(JSON.stringify(value));

    return value;
  };
}

test("masked ConverseStream early break withholds partial output", async () => {
  const seen: string[] = [];
  const spans = setup({ mask: recordingMask(seen) });
  const client = wrapBedrock(new FakeClient([{ stream: textStream("partial secret", 3) }]));

  const response = (await client.send(
    new ConverseStreamCommand({ modelId: "anthropic.claude", messages: [] }),
  )) as { stream: AsyncIterable<unknown> };

  await collect(response.stream, 2);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(seen.some((value) => value.includes("partial secret"))).toBe(false);
});

test.each([false, true])(
  "ConverseStream past the fixed bound is flagged even when the cap is larger (masked %s)",
  async (masked) => {
    const seen: string[] = [];

    const spans = setup({
      maxAttributeLength: 1024 * 1024,
      ...(masked ? { mask: recordingMask(seen) } : {}),
    });

    await streamedConverse(textStream("x".repeat(100), 1_100));

    const [span] = spans.getFinishedSpans();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
    expect(span.attributes["gen_ai.response.finish_reasons"]).toEqual(["end_turn"]);

    if (masked) {
      expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
      expect(seen.some((value) => value.includes("x".repeat(100)))).toBe(false);
    } else {
      const retained = jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content as string;
      expect(retained.length).toBeGreaterThan(0);
      expect(retained.length).toBeLessThanOrEqual(48 * 1024);
      expect(retained).toBe("x".repeat(retained.length));
    }
  },
);

test.each([
  [1_000, false, "東".repeat(400)],
  [1_000, true, "[redacted]"],
  [0, false, undefined],
] as const)(
  "maxAttributeLength %i cuts the exported ConverseStream output after the mask (redact %s)",
  async (maxAttributeLength, redact, expected) => {
    const spans = setup({
      maxAttributeLength,
      mask: (value, context) =>
        redact && context.key === "gen_ai.output.messages"
          ? [{ role: "assistant", parts: [{ type: "text", content: "[redacted]" }] }]
          : value,
    });

    await streamedConverse(textStream(redact ? "x".repeat(5_000) : "東".repeat(400)));

    const [span] = spans.getFinishedSpans();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBeUndefined();

    if (expected === undefined) {
      expect(span.attributes["gen_ai.output.messages"] ?? "").toBe("");
    } else {
      expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe(expected);
    }
  },
);

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
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test.each([
  { messages: "changed in place", olderCore: false },
  { messages: "new", olderCore: false },
  { messages: "changed in place", olderCore: true },
])(
  "Converse outside withPrompt records the prompt of $messages messages, older core $olderCore",
  async ({ messages: kind, olderCore }) => {
    const spans = setup();

    if (olderCore) {
      vi.spyOn(
        telemetrySdk as { promptAttributesFor: unknown },
        "promptAttributesFor",
        "get",
      ).mockReturnValue(undefined);
    }

    const prompt = await getPrompt("support/reply", {
      apiKey: "td_live_a",
      baseUrl: "https://prompts.bedrock.test",
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
          model: { provider: "amazon-bedrock", name: "anthropic.claude-3-haiku" },
          parameters: {},
          tools: [],
          responseFormat: { type: "text" },
          rollout: null,
        }),
    });
    const [system, user] = prompt.compile({ team: "billing", order: "#4" }) as ChatMessage[];
    const blocks = [{ text: user!.content }];
    // Converse takes content blocks: a message changed in place keeps its tag, a new one has none.
    const message =
      kind === "new"
        ? { role: "user", content: blocks }
        : Object.assign(user!, { content: blocks });

    const client = wrapBedrock(
      new BedrockRuntimeClient({
        region: "us-east-1",
        credentials: { accessKeyId: "test", secretAccessKey: "test" },
        maxAttempts: 1,
        requestHandler: jsonRequestHandler("prompt-1", {
          output: { message: { role: "assistant", content: [{ text: "Soon." }] } },
          stopReason: "end_turn",
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        }),
      }),
    );

    await client.send(
      new ConverseCommand({
        modelId: "anthropic.claude-3-haiku",
        system: [{ text: system!.content }],
        messages: [message] as ConverseCommandInput["messages"],
      }),
    );

    const [span] = spans.getFinishedSpans();
    expect(span!.attributes["gen_ai.response.id"]).toBe("prompt-1");
    expect(
      Object.entries(span!.attributes).filter(([key]) => key.startsWith("td.prompt.")),
    ).toEqual(
      olderCore || kind === "new"
        ? []
        : [
            ["td.prompt.name", "support/reply"],
            ["td.prompt.version", 12],
            ["td.prompt.label", "production"],
            ["td.prompt.variables", '{"team":"billing","order":"#4"}'],
          ],
    );
  },
);

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
  expect(jsonAttr(chat, "gen_ai.input.messages")).toEqual({
    messages: [],
    max_tokens: 10,
    top_k: 3,
  });
  expect(jsonAttr(chat, "gen_ai.output.messages")).toEqual({
    content: [{ type: "text", text: "answer" }],
    usage: { input_tokens: 9, output_tokens: 4, cache_read_input_tokens: 1 },
    stop_reason: "end_turn",
  });
  expect(chat.attributes["gen_ai.usage.input_tokens"]).toBe(9 + 1);
  expect(chat.attributes["gen_ai.request.top_k"]).toBe(3);
  expect(chat.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(1);
  expect(chat.attributes["gen_ai.response.id"]).toBe("invoke-1");
  expect(embedding.attributes["gen_ai.operation.name"]).toBe("embeddings");
  expect(embedding.attributes["gen_ai.output.type"]).toBe("embedding");
  expect(embedding.attributes["gen_ai.usage.input_tokens"]).toBe(6);
});

test.each(["absent", "compaction", "at-bound", "bounded"])(
  "Claude InvokeModel usage: %s",
  async (kind) => {
    const spans = setup();
    const compaction = {
      type: "compaction",
      input_tokens: 11,
      output_tokens: 7,
      cache_read_input_tokens: 3,
      cache_creation_input_tokens: 5,
    };
    const usage = {
      input_tokens: 20,
      output_tokens: 9,
      cache_read_input_tokens: 2,
      cache_creation_input_tokens: 4,
      ...(kind === "absent"
        ? {}
        : {
            iterations:
              kind === "bounded" || kind === "at-bound"
                ? [
                    {
                      ...compaction,
                      input_tokens: 17,
                      output_tokens: 13,
                      cache_read_input_tokens: 6,
                      cache_creation_input_tokens: 8,
                    },
                    ...Array.from({ length: kind === "bounded" ? 999 : 998 }, () => ({
                      type: "message",
                      input_tokens: 20,
                    })),
                    compaction,
                  ]
                : [compaction, { type: "message", input_tokens: 20, output_tokens: 9 }],
          }),
    };
    const events = [
      {
        type: "message_start",
        message: {
          usage: { input_tokens: 1, cache_read_input_tokens: 1, cache_creation_input_tokens: 1 },
        },
      },
      { type: "message_delta", usage, delta: { stop_reason: "end_turn" } },
      {
        type: "message_stop",
        ...(kind === "compaction"
          ? {}
          : { "amazon-bedrock-invocationMetrics": { inputTokenCount: 31, outputTokenCount: 16 } }),
      },
    ].map((event) => ({ chunk: { bytes: bytes(event) } }));
    const client = wrapBedrock(
      new FakeClient([
        { body: bytes({ usage }), contentType: "application/json" },
        { body: streamOf(events) },
      ]),
    );
    const input = { modelId: "anthropic.claude", body: bytes({ messages: [] }) };
    await client.send(new InvokeModelCommand(input));
    const response = (await client.send(new InvokeModelWithResponseStreamCommand(input))) as {
      body: AsyncIterable<unknown>;
    };
    await collect(response.body);
    const finished = spans.getFinishedSpans();
    expect(finished).toHaveLength(2);
    for (const span of finished) {
      expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(
        kind === "absent" ? 26 : kind === "at-bound" ? 76 : 45,
      );
      expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(
        kind === "absent" ? 9 : kind === "at-bound" ? 29 : 16,
      );
      expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(
        kind === "absent" ? 2 : kind === "at-bound" ? 11 : 5,
      );
      expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(
        kind === "absent" ? 4 : kind === "at-bound" ? 17 : 9,
      );
      if (kind === "bounded") expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
    }
  },
);

const interruptedCompaction = {
  type: "compaction",
  input_tokens: 11,
  output_tokens: 7,
  cache_read_input_tokens: 3,
  cache_creation_input_tokens: 5,
};
const interruptedBase = {
  input_tokens: 20,
  cache_read_input_tokens: 2,
  cache_creation_input_tokens: 4,
};

test.each([
  { name: "start only", deltas: [], input: 45, output: 9 },
  {
    name: "delta without iterations",
    deltas: [{ type: "message_delta", delta: {}, usage: { output_tokens: 3 } }],
    input: 45,
    output: 10,
  },
  {
    name: "delta with cumulative iterations",
    deltas: [
      {
        type: "message_delta",
        delta: {},
        usage: {
          ...interruptedBase,
          output_tokens: 3,
          iterations: [interruptedCompaction, { type: "message" }, interruptedCompaction],
        },
      },
    ],
    input: 26 + 2 * 19,
    output: 3 + 2 * 7,
  },
])(
  "interrupted Claude InvokeModel stream keeps message_start compaction usage ($name)",
  async ({ deltas, input, output }) => {
    const spans = setup();
    const events = [
      {
        type: "message_start",
        message: {
          usage: {
            ...interruptedBase,
            output_tokens: 2,
            iterations: [interruptedCompaction, { type: "message", input_tokens: 20 }],
          },
        },
      },
      ...deltas,
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "unread" } },
      { type: "message_stop" },
    ].map((event) => ({ chunk: { bytes: bytes(event) } }));
    const client = wrapBedrock(new FakeClient([{ body: streamOf(events) }]));
    const response = (await client.send(
      new InvokeModelWithResponseStreamCommand({
        modelId: "anthropic.claude",
        body: bytes({ messages: [] }),
      }),
    )) as { body: AsyncIterable<unknown> };
    await collect(response.body, 1 + deltas.length);

    const [span] = spans.getFinishedSpans();
    expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(input);
    expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(output);
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  },
);

test("message_start compaction usage is never added to invocation metrics", async () => {
  const spans = setup();
  const events = [
    {
      type: "message_start",
      message: { usage: { ...interruptedBase, iterations: [interruptedCompaction] } },
    },
    {
      type: "message_stop",
      "amazon-bedrock-invocationMetrics": { inputTokenCount: 31, outputTokenCount: 16 },
    },
  ].map((event) => ({ chunk: { bytes: bytes(event) } }));
  const client = wrapBedrock(new FakeClient([{ body: streamOf(events) }]));
  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude",
      body: bytes({ messages: [] }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(response.body);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(37);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(16);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(2);
  expect(span.attributes["gen_ai.usage.cache_creation.input_tokens"]).toBe(4);
});

test.each([0, undefined])("Claude final usage takes precedence over metrics: %s", async (count) => {
  const spans = setup();
  const events = [
    {
      type: "message_delta",
      usage: {
        input_tokens: count,
        output_tokens: count,
        iterations: [
          {
            type: "compaction",
            input_tokens: 11,
            output_tokens: 7,
            cache_read_input_tokens: 3,
            cache_creation_input_tokens: 5,
          },
        ],
      },
    },
    {
      type: "message_stop",
      "amazon-bedrock-invocationMetrics": { inputTokenCount: 31, outputTokenCount: 16 },
    },
  ].map((event) => ({ chunk: { bytes: bytes(event) } }));
  const client = wrapBedrock(new FakeClient([{ body: streamOf(events) }]));
  const response = (await client.send(
    new InvokeModelWithResponseStreamCommand({
      modelId: "anthropic.claude",
      body: bytes({ messages: [] }),
    }),
  )) as { body: AsyncIterable<unknown> };
  await collect(response.body);
  const [span] = spans.getFinishedSpans();
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(count === 0 ? 19 : 31);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(count === 0 ? 7 : 16);
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

test("InvokeModelWithResponseStream input tokens include cache reads and writes", async () => {
  const spans = setup();

  const start = {
    type: "message_start",
    message: {
      usage: { input_tokens: 40, cache_read_input_tokens: 1000, cache_creation_input_tokens: 200 },
    },
  };

  const client = wrapBedrock(
    new FakeClient([
      {
        body: streamOf([
          { chunk: { bytes: bytes(start) } },
          { chunk: { bytes: bytes({ delta: { text: "hi", stop_reason: "end_turn" } }) } },
          { chunk: { bytes: bytes({ type: "message_delta", usage: { output_tokens: 12 } }) } },
          {
            chunk: {
              bytes: bytes({
                "amazon-bedrock-invocationMetrics": { inputTokenCount: 40, outputTokenCount: 12 },
              }),
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
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(40 + 1000 + 200);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(12);
  expect(span.attributes["gen_ai.usage.cache_read.input_tokens"]).toBe(1000);
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

test("send errors end one error span, double wrapping traces once, and shutdown passes through", async () => {
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
  await expect(disabled.send(new ConverseCommand({ modelId: "m", messages: [] }))).resolves.toEqual(
    { output: { message: { content: [] } } },
  );
  expect(disabled.calls).toHaveLength(1);
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

test("global instrumentation patches the real client once and restores its inherited send", async () => {
  const spans = setup();
  const proto = BedrockRuntimeClient.prototype;
  const command = new ConverseCommand({ modelId: "m", messages: [] });

  const client = new BedrockRuntimeClient({
    region: "us-east-1",
    credentials: { accessKeyId: "test", secretAccessKey: "test" },
    maxAttempts: 1,
    requestHandler: jsonRequestHandler("global-1", {
      output: { message: { role: "assistant", content: [{ text: "ok" }] } },
      stopReason: "end_turn",
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      metrics: { latencyMs: 1 },
    }),
  });

  // AWS clients inherit send from the Smithy base class.
  expect(Object.hasOwn(proto, "send")).toBe(false);

  try {
    instrumentBedrock();
    instrumentBedrock();

    await client.send(command);
    expect(spans.getFinishedSpans()).toHaveLength(1);
    expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.response.id"]).toBe("global-1");

    const telemetryWrapper = Object.getOwnPropertyDescriptor(proto, "send")?.value;
    const laterWrapper = async () => ({ output: { message: { content: [] } } });
    Object.defineProperty(proto, "send", {
      configurable: true,
      writable: true,
      value: laterWrapper,
    });
    uninstrumentBedrock();
    expect(Object.getOwnPropertyDescriptor(proto, "send")?.value).toBe(laterWrapper);
    Object.defineProperty(proto, "send", {
      configurable: true,
      writable: true,
      value: telemetryWrapper,
    });
    uninstrumentBedrock();
    expect(Object.hasOwn(proto, "send")).toBe(false);

    await client.send(command);
    expect(spans.getFinishedSpans()).toHaveLength(1);

    const priorOwner = async () => ({ output: { message: { content: [] } } });
    Object.defineProperty(proto, "send", {
      configurable: true,
      writable: true,
      value: priorOwner,
    });
    instrumentBedrock();
    expect(Object.getOwnPropertyDescriptor(proto, "send")?.value).not.toBe(priorOwner);
    uninstrumentBedrock();
    expect(Object.getOwnPropertyDescriptor(proto, "send")?.value).toBe(priorOwner);
  } finally {
    uninstrumentBedrock();
    Reflect.deleteProperty(proto, "send");
  }
});
