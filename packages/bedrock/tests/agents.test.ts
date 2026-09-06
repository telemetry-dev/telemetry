import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
  InvokeFlowCommand,
  RetrieveAndGenerateCommand,
  RetrieveAndGenerateStreamCommand,
  RetrieveCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
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
