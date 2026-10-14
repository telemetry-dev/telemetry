import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
  InvokeFlowCommand,
  RerankCommand,
  RetrieveAndGenerateCommand,
  RetrieveAndGenerateStreamCommand,
  RetrieveCommand,
  type RerankCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import type { Attributes } from "@opentelemetry/api";
import { SamplingDecision } from "@opentelemetry/sdk-trace-base";
import { afterEach, expect, test } from "vitest";

import {
  instrumentBedrockAgents,
  uninstrumentBedrockAgents,
  wrapBedrockAgents,
} from "../src/agents.ts";
import { FakeClient, bytes, collect, jsonAttr, setup, streamOf, teardown } from "./helpers.ts";

afterEach(async () => {
  await teardown();
  uninstrumentBedrockAgents();
});

test("InvokeAgent aggregates trace usage and return control", async () => {
  const spans = setup();

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        completion: streamOf([
          {
            chunk: {
              bytes: bytes("hello"),
              attribution: { citations: [{ generatedResponsePart: { textResponsePart: {} } }] },
            },
          },
          {
            trace: {
              trace: {
                orchestrationTrace: {
                  modelInvocationOutput: {
                    metadata: { usage: { inputTokens: 3, outputTokens: 4 } },
                  },
                },
              },
            },
          },
          { returnControl: { invocationId: "inv-1", invocationInputs: [{ function: "lookup" }] } },
          { files: { files: [{ name: "chart.png", type: "image/png", bytes: bytes("raw") }] } },
        ]),
        sessionId: "sess-1",
        memoryId: "mem-1",
        $metadata: { requestId: "agent-req" },
      },
    ]),
    { captureAgentTrace: true },
  );

  const response = (await client.send(
    new InvokeAgentCommand({
      agentId: "agent-1",
      agentAliasId: "alias-1",
      sessionId: "sess-1",
      inputText: "hello",
      enableTrace: true,
    }),
  )) as { completion: AsyncIterable<unknown> };

  await collect(response.completion);

  const [span] = spans.getFinishedSpans();
  expect(span.name).toBe("invoke_agent agent-1");
  expect(span.attributes["gen_ai.operation.name"]).toBe("invoke_agent");
  expect(span.attributes["gen_ai.agent.id"]).toBe("agent-1");
  expect(span.attributes["gen_ai.provider.name"]).toBe("amazon-bedrock");
  expect(span.attributes["gen_ai.usage.input_tokens"]).toBe(3);
  expect(span.attributes["gen_ai.usage.output_tokens"]).toBe(4);
  expect(span.attributes["gen_ai.usage.total_tokens"]).toBe(7);
  expect(span.attributes["td.metadata.bedrock_session_id"]).toBe("sess-1");
  expect(span.attributes["gen_ai.response.id"]).toBe("agent-req");
  expect(span.attributes["td.metadata.return_control"]).toBe("true");
  expect(span.attributes["td.metadata.citation_count"]).toBe("1");
  expect(jsonAttr(span, "td.metadata.output_files")).toEqual([
    { name: "chart.png", type: "image/png" },
  ]);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual({
    returnControl: { invocationId: "inv-1", invocationInputs: [{ function: "lookup" }] },
  });
});

test.each([false, true])(
  "InvokeAgent text past the fixed bound is flagged and withheld from a mask (masked %s)",
  async (masked) => {
    const seen: string[] = [];

    const spans = setup(
      masked
        ? {
            mask: (value) => {
              seen.push(JSON.stringify(value));

              return value;
            },
          }
        : {},
    );

    const client = wrapBedrockAgents(
      new FakeClient([
        {
          completion: streamOf(
            Array.from({ length: 1_100 }, () => ({ chunk: { bytes: bytes("x".repeat(100)) } })),
          ),
        },
      ]),
    );

    const response = (await client.send(
      new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "a", sessionId: "s" }),
    )) as { completion: AsyncIterable<unknown> };

    await collect(response.completion);

    const [span] = spans.getFinishedSpans();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);

    if (masked) {
      expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
      expect(seen.some((value) => value.includes("x".repeat(100)))).toBe(false);
    } else {
      const text = jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content as string;
      expect(text.length).toBeGreaterThan(0);
      expect(text.length).toBeLessThan(1_100 * 102);
    }
  },
);

test.each([
  ["trace", { trace: { trace: { orchestrationTrace: { rawResponse: "t".repeat(50_000) } } } }],
  ["output file", { files: { files: [{ name: "f".repeat(50_000), type: "text/plain" }] } }],
])(
  "masked InvokeAgent keeps complete text when only %s metadata passed the bound",
  async (_kind, metadataEvent) => {
    const seen: string[] = [];

    const spans = setup({
      mask: (value) => {
        seen.push(JSON.stringify(value));

        return value;
      },
    });

    const client = wrapBedrockAgents(
      new FakeClient([
        {
          completion: streamOf([{ chunk: { bytes: bytes("Complete answer") } }, metadataEvent]),
        },
      ]),
      { captureAgentTrace: true },
    );

    const response = (await client.send(
      new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "a", sessionId: "s" }),
    )) as { completion: AsyncIterable<unknown> };

    await collect(response.completion);

    const [span] = spans.getFinishedSpans();
    expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
    expect(jsonAttr(span, "gen_ai.output.messages")[0].parts[0].content).toBe(
      JSON.stringify("Complete answer"),
    );
    expect(seen.some((value) => value.includes("Complete answer"))).toBe(true);
  },
);

test("masked InvokeAgent withholds text when a chunk after the metadata bound is rejected", async () => {
  const seen: string[] = [];

  const spans = setup({
    mask: (value) => {
      seen.push(JSON.stringify(value));

      return value;
    },
  });

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        completion: streamOf([
          { chunk: { bytes: bytes("Complete answer") } },
          { trace: { trace: { orchestrationTrace: { rawResponse: "t".repeat(50_000) } } } },
          { chunk: { bytes: bytes(" and more") } },
        ]),
      },
    ]),
    { captureAgentTrace: true },
  );

  const response = (await client.send(
    new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "a", sessionId: "s" }),
  )) as { completion: AsyncIterable<unknown> };

  await collect(response.completion);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(seen.some((value) => value.includes("Complete answer"))).toBe(false);
});

test("masked InvokeAgent keeps complete return control after the text bound", async () => {
  const returnControl = { invocationId: "inv-after-bound" };
  const spans = setup({ mask: (value) => value });

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        completion: streamOf([
          ...Array.from({ length: 1_100 }, () => ({ chunk: { bytes: bytes("x".repeat(100)) } })),
          { returnControl },
        ]),
      },
    ]),
  );

  const response = (await client.send(
    new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "a", sessionId: "s" }),
  )) as { completion: AsyncIterable<unknown> };

  await collect(response.completion);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual({ returnControl });
});

test.each([
  ["bytes", [{ apiInvocationInput: { actionGroup: "x".repeat(60_000) } }]],
  ["items", Array.from({ length: 1_500 }, () => ({ function: "f" }))],
])("masked InvokeAgent drops return control beyond its own %s bound", async (_, inputs) => {
  const seen: string[] = [];

  const spans = setup({
    mask: (value) => {
      seen.push(JSON.stringify(value));

      return value;
    },
  });

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        completion: streamOf([
          { chunk: { bytes: bytes("Short answer") } },
          { returnControl: { invocationId: "inv-oversized", invocationInputs: inputs } },
        ]),
      },
    ]),
  );

  const response = (await client.send(
    new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "a", sessionId: "s" }),
  )) as { completion: AsyncIterable<unknown> };

  await collect(response.completion);

  const [span] = spans.getFinishedSpans();
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
  expect(span.attributes["td.metadata.return_control"]).toBe("true");
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(
    seen.some((value) => value.includes("inv-oversized") || value.includes("Short answer")),
  ).toBe(false);
});

test("Rerank captures queries, sources, model, ranking, and pagination metadata", async () => {
  const spans = setup();
  const modelArn = "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0";

  const request: RerankCommandInput = {
    queries: [{ type: "TEXT", textQuery: { text: "best observability platform" } }],
    sources: [
      {
        type: "INLINE",
        inlineDocumentSource: { type: "TEXT", textDocument: { text: "Telemetry data" } },
      },
      {
        type: "INLINE",
        inlineDocumentSource: { type: "JSON", jsonDocument: { title: "Other result" } },
      },
    ],
    rerankingConfiguration: {
      type: "BEDROCK_RERANKING_MODEL",
      bedrockRerankingConfiguration: {
        numberOfResults: 2,
        modelConfiguration: {
          modelArn,
          additionalModelRequestFields: { max_tokens_per_doc: 256 },
        },
      },
    },
  };

  const requestSnapshot = structuredClone(request);

  const results = [
    {
      index: 0,
      relevanceScore: 0.91,
      document: { type: "TEXT", textDocument: { text: "Telemetry data" } },
    },
    { index: 1, relevanceScore: 0.12 },
  ];

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        results,
        nextToken: "page-2",
        $metadata: { requestId: "rerank-1", httpStatusCode: 200 },
      },
    ]),
  );

  await client.send(new RerankCommand(request));

  expect(request).toEqual(requestSnapshot);
  expect(client.calls[0]).toEqual(requestSnapshot);
  const [span] = spans.getFinishedSpans();
  expect(span.name).toBe("rerank cohere.rerank-v3-5:0");
  expect(span.attributes["gen_ai.operation.name"]).toBe("rerank");
  expect(span.attributes["gen_ai.provider.name"]).toBe("amazon-bedrock");
  expect(span.attributes["gen_ai.request.model"]).toBe(modelArn);
  expect(span.attributes["gen_ai.response.id"]).toBe("rerank-1");
  expect(span.attributes["td.metadata.requested_result_count"]).toBe("2");
  expect(span.attributes["td.metadata.result_count"]).toBe("2");
  expect(span.attributes["td.metadata.has_next_token"]).toBe("true");
  expect(jsonAttr(span, "gen_ai.input.messages")).toEqual({
    queries: request.queries,
    sources: request.sources,
    additionalModelRequestFields: { max_tokens_per_doc: 256 },
  });
  expect(jsonAttr(span, "gen_ai.output.messages")).toEqual(results);
});

test("Rerank bounds large source and result captures while preserving counts", async () => {
  const spans = setup();

  const sources = Array.from({ length: 2_000 }, (_, index) => ({
    type: "INLINE" as const,
    inlineDocumentSource: {
      type: "TEXT" as const,
      textDocument: { text: `source-${index}:${"x".repeat(1_000)}` },
    },
  }));

  const results = Array.from({ length: 2_000 }, (_, index) => ({
    index,
    relevanceScore: 1 / (index + 1),
    document: { type: "TEXT", textDocument: { text: `result-${index}:${"y".repeat(1_000)}` } },
  }));

  const client = wrapBedrockAgents(new FakeClient([{ results }]));

  await client.send(
    new RerankCommand({
      queries: [{ type: "TEXT", textQuery: { text: "observability" } }],
      sources,
      rerankingConfiguration: {
        type: "BEDROCK_RERANKING_MODEL",
        bedrockRerankingConfiguration: {
          modelConfiguration: {
            modelArn: "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
          },
        },
      },
    }),
  );

  const [span] = spans.getFinishedSpans();
  const input = jsonAttr(span, "gen_ai.input.messages");
  const output = jsonAttr(span, "gen_ai.output.messages");

  expect(input.sources.length).toBeLessThan(sources.length);
  expect(output.length).toBeLessThan(results.length);
  expect(JSON.stringify(input).length).toBeLessThanOrEqual(48 * 1_024);
  expect(JSON.stringify(output).length).toBeLessThanOrEqual(48 * 1_024);
  expect(span.attributes["td.metadata.result_count"]).toBe("2000");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("Rerank withholds captures cut to the retention bound from a mask", async () => {
  const keys: unknown[] = [];

  const spans = setup({
    mask: (value, context) => {
      keys.push(context.key);

      return value;
    },
  });

  const sources = Array.from({ length: 2_000 }, (_, index) => ({
    type: "INLINE" as const,
    inlineDocumentSource: {
      type: "TEXT" as const,
      textDocument: { text: `source-${index}:${"x".repeat(1_000)}` },
    },
  }));

  const results = Array.from({ length: 2_000 }, (_, index) => ({
    index,
    relevanceScore: 1 / (index + 1),
    document: { type: "TEXT", textDocument: { text: `result-${index}:${"y".repeat(1_000)}` } },
  }));

  const client = wrapBedrockAgents(new FakeClient([{ results }]));

  await client.send(
    new RerankCommand({
      queries: [{ type: "TEXT", textQuery: { text: "observability" } }],
      sources,
      rerankingConfiguration: {
        type: "BEDROCK_RERANKING_MODEL",
        bedrockRerankingConfiguration: {
          modelConfiguration: {
            modelArn: "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
          },
        },
      },
    }),
  );

  const [span] = spans.getFinishedSpans();

  expect(keys).not.toContain("gen_ai.input.messages");
  expect(keys).not.toContain("gen_ai.output.messages");
  expect(span.attributes["gen_ai.input.messages"]).toBeUndefined();
  expect(span.attributes["gen_ai.output.messages"]).toBeUndefined();
  expect(span.attributes["td.metadata.result_count"]).toBe("2000");
  expect(span.attributes["telemetry.dev.capture.truncated"]).toBe(true);
});

test("Rerank without a mask passes cut input to the sampler", async () => {
  const sampled: Attributes[] = [];

  setup({
    sampler: {
      shouldSample(_context, _traceId, _name, _kind, attributes) {
        sampled.push({ ...attributes });

        return { decision: SamplingDecision.RECORD_AND_SAMPLED };
      },
      toString: () => "recording",
    },
  });

  const sources = Array.from({ length: 2_000 }, (_, index) => ({
    type: "INLINE" as const,
    inlineDocumentSource: {
      type: "TEXT" as const,
      textDocument: { text: `source-${index}:${"x".repeat(1_000)}` },
    },
  }));

  const client = wrapBedrockAgents(new FakeClient([{ results: [] }]));

  await client.send(
    new RerankCommand({
      queries: [{ type: "TEXT", textQuery: { text: "observability" } }],
      sources,
      rerankingConfiguration: {
        type: "BEDROCK_RERANKING_MODEL",
        bedrockRerankingConfiguration: {
          modelConfiguration: {
            modelArn: "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
          },
        },
      },
    }),
  );

  expect(sampled).toHaveLength(1);
  expect(String(sampled[0]?.["gen_ai.input.messages"])).toContain("source-0");
  expect(sampled[0]?.["telemetry.dev.capture.truncated"]).toBe(true);
});

test("Retrieve and RetrieveAndGenerate capture outputs and citations", async () => {
  const spans = setup();

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        retrievalResults: [{ content: { text: "doc" }, score: 0.9 }],
        guardrailAction: "NONE",
      },
      {
        output: { text: "answer" },
        citations: [{ generatedResponsePart: {} }],
        sessionId: "sess-2",
      },
    ]),
  );

  await client.send(
    new RetrieveCommand({ knowledgeBaseId: "kb-1", retrievalQuery: { text: "question" } }),
  );
  await client.send(
    new RetrieveAndGenerateCommand({
      input: { text: "question" },
      retrieveAndGenerateConfiguration: {
        type: "KNOWLEDGE_BASE",
        knowledgeBaseConfiguration: {
          knowledgeBaseId: "kb-1",
          modelArn: "arn:aws:bedrock:us::model/foo",
        },
      },
    }),
  );

  const [retrieve, rag] = spans.getFinishedSpans();
  expect(retrieve.name).toBe("retrieve kb-1");
  expect(retrieve.attributes["gen_ai.provider.name"]).toBe("amazon-bedrock");
  expect(retrieve.attributes["td.metadata.citation_count"]).toBe("1");
  expect(rag.name).toBe("retrieve_and_generate foo");
  expect(rag.attributes["td.metadata.citation_count"]).toBe("1");
  expect(jsonAttr(rag, "gen_ai.output.messages")[0].parts[0].content).toBe("answer");
});

test("RetrieveAndGenerateStream and InvokeFlow finish from streams", async () => {
  const spans = setup();

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        stream: streamOf([
          { output: { text: "rag" } },
          { citation: {} },
          { guardrail: { action: "NONE" } },
        ]),
        sessionId: "sess-stream",
      },
      {
        responseStream: streamOf([
          {
            flowOutputEvent: { content: { document: { value: 1 } }, nodeName: "n", nodeType: "x" },
          },
          {
            flowMultiTurnInputRequestEvent: { content: { document: { prompt: "more" } } },
          },
          { flowCompletionEvent: { completionReason: "SUCCESS" } },
        ]),
      },
    ]),
  );

  const rag = (await client.send(
    new RetrieveAndGenerateStreamCommand({ input: { text: "q" } }),
  )) as { stream: AsyncIterable<unknown> };

  await collect(rag.stream);

  const flow = (await client.send(
    new InvokeFlowCommand({ flowIdentifier: "flow-1", flowAliasIdentifier: "alias", inputs: [] }),
  )) as { responseStream: AsyncIterable<unknown> };

  await collect(flow.responseStream);

  const [ragSpan, flowSpan] = spans.getFinishedSpans();
  expect(jsonAttr(ragSpan, "gen_ai.output.messages")[0].parts[0].content).toBe("rag");
  expect(ragSpan.attributes["td.metadata.citation_count"]).toBe("1");
  expect(ragSpan.attributes["td.metadata.bedrock_session_id"]).toBe("sess-stream");
  expect(flowSpan.name).toBe("invoke_flow flow-1");
  expect(flowSpan.attributes["gen_ai.provider.name"]).toBe("amazon-bedrock");
  expect(jsonAttr(flowSpan, "gen_ai.output.messages")).toEqual([
    { document: { value: 1 } },
    { document: { prompt: "more" } },
  ]);
  expect(flowSpan.attributes["gen_ai.response.finish_reasons"]).toEqual(["SUCCESS"]);
});

test("modeled stream errors from Agent, RAG, and Flow mark spans as errors", async () => {
  const spans = setup();

  const client = wrapBedrockAgents(
    new FakeClient([
      {
        completion: streamOf([
          { chunk: { bytes: "agent partial" } },
          {
            internalServerException: {
              name: "InternalServerException",
              message: "agent stream failed",
              $metadata: { requestId: "agent-stream-error", httpStatusCode: 500 },
            },
          },
        ]),
      },
      {
        stream: streamOf([
          { output: { text: "rag partial" } },
          {
            throttlingException: {
              name: "ThrottlingException",
              message: "rag stream failed",
              $metadata: { requestId: "rag-stream-error", httpStatusCode: 429 },
            },
          },
        ]),
      },
      {
        responseStream: streamOf([
          { flowOutputEvent: { content: { document: { value: 1 } } } },
          {
            validationException: {
              name: "ValidationException",
              message: "flow stream failed",
              $metadata: { requestId: "flow-stream-error", httpStatusCode: 400 },
            },
          },
        ]),
      },
    ]),
  );

  const agent = (await client.send(
    new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "alias", sessionId: "sess" }),
  )) as { completion: AsyncIterable<unknown> };

  await collect(agent.completion);

  const rag = (await client.send(
    new RetrieveAndGenerateStreamCommand({ input: { text: "q" } }),
  )) as { stream: AsyncIterable<unknown> };

  await collect(rag.stream);

  const flow = (await client.send(
    new InvokeFlowCommand({ flowIdentifier: "flow-1", flowAliasIdentifier: "alias", inputs: [] }),
  )) as { responseStream: AsyncIterable<unknown> };

  await collect(flow.responseStream);

  const [agentSpan, ragSpan, flowSpan] = spans.getFinishedSpans();
  expect(agentSpan.attributes["error.type"]).toBe("InternalServerException");
  expect(agentSpan.attributes["gen_ai.response.id"]).toBe("agent-stream-error");
  expect(agentSpan.attributes["aws.http.status_code"]).toBe(500);
  expect(jsonAttr(agentSpan, "gen_ai.output.messages")[0].parts[0].content).toBe("agent partial");
  expect(ragSpan.attributes["error.type"]).toBe("ThrottlingException");
  expect(ragSpan.attributes["gen_ai.response.id"]).toBe("rag-stream-error");
  expect(ragSpan.attributes["aws.http.status_code"]).toBe(429);
  expect(jsonAttr(ragSpan, "gen_ai.output.messages")[0].parts[0].content).toBe("rag partial");
  expect(flowSpan.attributes["error.type"]).toBe("ValidationException");
  expect(flowSpan.attributes["gen_ai.response.id"]).toBe("flow-stream-error");
  expect(flowSpan.attributes["aws.http.status_code"]).toBe(400);
  expect(jsonAttr(flowSpan, "gen_ai.output.messages")).toEqual([{ document: { value: 1 } }]);
});

test("global agents instrumentation is idempotent", async () => {
  const spans = setup();

  const originalDescriptor = Object.getOwnPropertyDescriptor(
    BedrockAgentRuntimeClient.prototype,
    "send",
  );

  const fakeSend = async () => ({
    completion: streamOf([{ chunk: { bytes: bytes("global") } }]),
    $metadata: { requestId: "agent-global-1" },
  });

  Object.defineProperty(BedrockAgentRuntimeClient.prototype, "send", {
    configurable: true,
    writable: true,
    value: fakeSend,
  });

  try {
    instrumentBedrockAgents();
    instrumentBedrockAgents();

    const client = new BedrockAgentRuntimeClient({
      region: "us-east-1",
      credentials: { accessKeyId: "test", secretAccessKey: "test" },
    });

    const response = (await client.send(
      new InvokeAgentCommand({ agentId: "agent-1", agentAliasId: "alias", sessionId: "sess" }),
    )) as { completion: AsyncIterable<unknown> };

    await collect(response.completion);
    expect(spans.getFinishedSpans()).toHaveLength(1);
    expect(spans.getFinishedSpans()[0]!.attributes["gen_ai.response.id"]).toBe("agent-global-1");
    uninstrumentBedrockAgents();
    expect(
      Object.getOwnPropertyDescriptor(BedrockAgentRuntimeClient.prototype, "send")?.value,
    ).toBe(fakeSend);
  } finally {
    uninstrumentBedrockAgents();

    if (originalDescriptor) {
      Object.defineProperty(BedrockAgentRuntimeClient.prototype, "send", originalDescriptor);
    }
  }
});
