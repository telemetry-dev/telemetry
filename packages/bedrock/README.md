# @telemetry-dev/bedrock

First-class AWS Bedrock instrumentation for the telemetry.dev TypeScript SDK.

This package instruments AWS SDK for JavaScript v3 clients for:

- `@aws-sdk/client-bedrock-runtime`
  - `Converse`
  - `ConverseStream`
  - `InvokeModel`
  - `InvokeModelWithResponseStream`
  - `ApplyGuardrail`
- `@aws-sdk/client-bedrock-agent-runtime`
  - `InvokeAgent`
  - `InvokeInlineAgent`
  - `Retrieve`
  - `RetrieveAndGenerate`
  - `RetrieveAndGenerateStream`
  - `InvokeFlow`

Telemetry is emitted through the public `@telemetry-dev/sdk` span API. No AWS request payloads are mutated.

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/bedrock @aws-sdk/client-bedrock-runtime
# Agent Runtime support is optional:
pnpm add @aws-sdk/client-bedrock-agent-runtime
```

## Quickstart

```ts
import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { init, shutdown } from "@telemetry-dev/sdk";
import { wrapBedrock } from "@telemetry-dev/bedrock";

init({
  apiKey: process.env.TELEMETRY_DEV_API_KEY,
  serviceName: "bedrock-app",
  environment: "production",
});

const bedrock = wrapBedrock(new BedrockRuntimeClient({ region: "us-east-1" }));

await bedrock.send(
  new ConverseCommand({
    modelId: "anthropic.claude-3-5-haiku-20241022-v1:0",
    messages: [{ role: "user", content: [{ text: "Hello" }] }],
  }),
);

await shutdown();
```

`wrapBedrock` is per-client and idempotent. It accepts any structural client with a `send(command, options?)` method, so it works with `BedrockRuntimeClient`, `BedrockRuntime`, and tests/fakes using real command objects.

## Global runtime instrumentation

```ts
import { instrumentBedrock, uninstrumentBedrock } from "@telemetry-dev/bedrock";

instrumentBedrock();
// BedrockRuntimeClient instances created before or after this point are instrumented.
uninstrumentBedrock();
```

Global instrumentation shadows `BedrockRuntimeClient.prototype.send`. It does not patch the shared Smithy base client, so other AWS SDK clients are untouched.

## Agent Runtime entrypoint

Agent Runtime support lives in a subpath so `@aws-sdk/client-bedrock-agent-runtime` can stay optional for runtime-only users.

```ts
import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { wrapBedrockAgents, instrumentBedrockAgents } from "@telemetry-dev/bedrock/agents";

const agent = wrapBedrockAgents(new BedrockAgentRuntimeClient({}), {
  captureAgentTrace: true,
});

await agent.send(
  new InvokeAgentCommand({
    agentId: process.env.BEDROCK_AGENT_ID!,
    agentAliasId: process.env.BEDROCK_AGENT_ALIAS_ID!,
    sessionId: crypto.randomUUID(),
    inputText: "Hello",
    enableTrace: true,
  }),
);

instrumentBedrockAgents({ captureAgentTrace: false });
```

## Options

| Option              | Default | Applies to            | Notes                                                                                                                                                 |
| ------------------- | ------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `captureAgentTrace` | `false` | Agent Runtime streams | Aggregates trace usage and counts by default. When enabled, also attaches raw trace events as `td.metadata.agent_trace` after SDK masking/truncation. |

## Signal coverage

All instrumented AWS responses/errors include `gen_ai.response.id` from the AWS request id, plus `aws.http.status_code`, `aws.request.attempts` when retries occurred, and `aws.request.total_retry_delay_ms` when the SDK exposes them.

| Operation                           | Span type                   | Span name                                             | Captured fields                                                                                                                                                                                        |
| ----------------------------------- | --------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Converse`                          | `generation`                | `chat {modelId}`                                      | normalized input/output messages, system instructions, sampling params, usage/cache usage, finish reason, request id, retry attempts, prompt-router response model, server latency, guardrail metadata |
| `ConverseStream`                    | `generation`                | `chat {modelId}`                                      | pull-through stream output, time-to-first-chunk, usage/latency from `metadata`, partial output on early break/error                                                                                    |
| `InvokeModel`                       | `generation` or `embedding` | `chat {modelId}` / `embeddings {modelId}`             | native JSON request/response bodies, provider-native sampling, provider-native usage, embedding output type                                                                                            |
| `InvokeModelWithResponseStream`     | `generation` or `embedding` | `chat {modelId}` / `embeddings {modelId}`             | provider-native chunk text where known, optional final `amazon-bedrock-invocationMetrics` usage                                                                                                        |
| `ApplyGuardrail`                    | `span`                      | `apply_guardrail {guardrailIdentifier}`               | guardrail input/output, action, action reason, request id                                                                                                                                              |
| `InvokeAgent` / `InvokeInlineAgent` | `agent`                     | `invoke_agent {agentId}` / `invoke_agent {agentName}` | user input, streamed answer, session/memory ids, alias id, trace usage aggregation, trace event count, return-control output                                                                           |
| `Retrieve`                          | `span`                      | `retrieve {knowledgeBaseId}`                          | query, retrieval results, guardrail action, result count metadata                                                                                                                                      |
| `RetrieveAndGenerate` / stream      | `generation`                | `retrieve_and_generate {modelArn basename}`           | input text, generated text, citations count, session id, guardrail action                                                                                                                              |
| `InvokeFlow`                        | `agent`                     | `invoke_flow {flowIdentifier}`                        | inputs, flow output events, completion reason                                                                                                                                                          |

## Message normalization

Converse messages follow the OpenTelemetry GenAI non-normative LLM-call examples:

- text blocks become `{ type: "text", content }`
- `toolUse` becomes `{ type: "tool_call", id, name, arguments }`
- `toolResult` becomes `{ type: "tool_call_response", id, response }`
- reasoning text becomes `{ type: "reasoning", content }`
- image/document/video/audio bytes become blob parts without byte content
- S3/URI sources become `{ type: "uri", uri, modality }`

InvokeModel captures provider-native JSON bodies verbatim. Non-JSON bodies are not captured.

## Semantics and guarantees

- Provider is emitted as `amazon-bedrock`. This intentionally differs from the OpenTelemetry registry value `aws.bedrock` so telemetry.dev pricing keys match Bedrock model IDs exactly. The raw caller `modelId`, `modelArn`, or `foundationModel` is emitted as the model.
- Instrumentation is fail-open. Normalization, span updates, and span endings are guarded so telemetry failures do not break caller code.
- Requests are never modified. This is especially important for AWS SigV4 signing and for application-level request object reuse.
- Streams are pull-through. The wrapper does not pre-buffer chunks and therefore preserves backpressure.
- Streams end spans exactly once. Exhaustion records full output and finish reason; early `break`, `return()`, `close()`, abort, or stream errors record partial output.
- `AbortSignal` passed to `send(command, { abortSignal })` is forwarded untouched. Pre-response aborts end the span with the thrown abort error. Mid-stream aborts end the span with partial output and the stream error.
- Callback-style `send(command, callback)` and `send(command, options, callback)` calls are instrumented; callbacks receive the original data/error while the span is ended before callback invocation.

## Coverage gaps

The following calls pass through unless a supported operation above is used:

- `CountTokens`
- `InvokeGuardrailChecks`
- `InvokeModelWithBidirectionalStream`
- `StartAsyncInvoke`, `GetAsyncInvoke`, `ListAsyncInvokes`
- Agent Runtime session CRUD
- `Rerank`
- `GenerateQuery`
- `OptimizePrompt`
- `AgenticRetrieveStream`

Provider-native `InvokeModel` usage is best-effort. Anthropic Claude, Amazon Titan/Nova, Meta Llama, and Titan embeddings expose known token fields. Cohere and Mistral text models do not consistently include usage in the JSON body; spans still capture request/response bodies and finish reasons where present.
