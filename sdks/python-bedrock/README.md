# telemetry-dev-bedrock

AWS Bedrock instrumentation for the telemetry.dev Python SDK.

This package instruments boto3/botocore clients for:

- `bedrock-runtime`
  - `Converse`
  - `ConverseStream`
  - `InvokeModel`
  - `InvokeModelWithResponseStream`
  - `ApplyGuardrail`
- `bedrock-agent-runtime`
  - `InvokeAgent`
  - `InvokeInlineAgent`
  - `Rerank`
  - `Retrieve`
  - `RetrieveAndGenerate`
  - `RetrieveAndGenerateStream`
  - `InvokeFlow`

Telemetry is emitted through `telemetry_dev.start_span`. No AWS request parameters are mutated.

## Install

```sh
uv add telemetry-dev telemetry-dev-bedrock boto3
```

or with pip:

```sh
pip install telemetry-dev telemetry-dev-bedrock boto3
```

## Quickstart

```py
import os
import boto3
import telemetry_dev
from telemetry_dev_bedrock import wrap_bedrock

telemetry_dev.init(
    api_key=os.getenv("TELEMETRY_DEV_API_KEY"),
    service_name="bedrock-app",
    environment="production",
)

bedrock = wrap_bedrock(boto3.client("bedrock-runtime", region_name="us-east-1"))

bedrock.converse(
    modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)

telemetry_dev.shutdown()
```

`wrap_bedrock` accepts either a `bedrock-runtime` or `bedrock-agent-runtime` boto3 client. It patches the operation methods on that instance and is idempotent.

## Global instrumentation

```py
from telemetry_dev_bedrock import instrument_bedrock, uninstrument_bedrock

instrument_bedrock()
# bedrock-runtime and bedrock-agent-runtime clients created before or after this point are covered.
uninstrument_bedrock()
```

Global instrumentation wraps `botocore.client.BaseClient._make_api_call` and filters by service name, so unrelated boto3 clients pass through untouched. Per-client `wrap_bedrock()` clients are skipped by the global wrapper to avoid double spans.

## Options

| Option | Default | Applies to | Notes |
| --- | --- | --- | --- |
| `capture_agent_trace` | `False` | Agent Runtime streams | Aggregates trace usage and counts by default. When enabled, also attaches trace events as `td.metadata.agent_trace`, filtered as described below. |

`td.metadata.agent_trace` is truncated but never passed to your `mask`, so each trace event is rebuilt from an explicit allowlist instead:

- Envelope fields: `agentId`, `agentAliasId`, `agentVersion`, `sessionId`, `collaboratorName`, `eventTime`, and the alias ARNs in `callerChain`.
- Content the agent sends to a model, tool, or guardrail (`modelInvocationInput`, `invocationInput`, `inputAssessments`) is kept as received when `capture_input` is on.
- Content that comes back (`modelInvocationOutput`, `rationale`, `observation`, `outputAssessments`, custom orchestration `event`) is kept as received when `capture_output` is on.
- A failure reason (`failureTrace.failureReason`, and `td.metadata.failure_reason`) is free-form text, so it is kept only when both are on. `td.metadata.failure` always records that the invocation failed, with `td.metadata.failure_code` when AWS sends one.
- When a `mask` is set, all of this content is withheld.
- Withheld content and the trace structure around it keep only trace ids, step types, the foundation model, guardrail action, failure code, and the usage, timings, and `clientRequestId` from AWS's trace `metadata`.

Any other field, including unknown trace types and the AWS SDK's `$unknown` union member, is omitted even when both capture options are on; a content part that is kept is exported whole, including any fields inside it that are not listed here. Errors are not covered by this filtering: AWS exceptions, including errors raised inside the stream, still record the provider's error message on the span.

```py
agent = wrap_bedrock(
    boto3.client("bedrock-agent-runtime"),
    capture_agent_trace=True,
)
```

## Signal coverage

All instrumented AWS responses/errors include `gen_ai.response.id` from the AWS request id, plus `aws.http.status_code`, `aws.request.attempts` when retries occurred, and `aws.request.total_retry_delay_ms` when the SDK exposes them.

| Operation | Span type | Span name | Captured fields |
| --- | --- | --- | --- |
| `Converse` | `generation` | `chat {modelId}` | normalized input/output messages, system instructions, sampling params, usage/cache usage, finish reason, request id, retry attempts, prompt-router response model, server latency, guardrail metadata |
| `ConverseStream` | `generation` | `chat {modelId}` | pull-through stream output, time-to-first-chunk, usage/latency from `metadata`, partial output on early break/error |
| `InvokeModel` | `generation` or `embedding` | `chat {modelId}` / `embeddings {modelId}` | native JSON request/response bodies, provider-native sampling, provider-native usage, embedding output type, HTTP-header token fallback |
| `InvokeModelWithResponseStream` | `generation` or `embedding` | `chat {modelId}` / `embeddings {modelId}` | provider-native chunk text where known, optional final `amazon-bedrock-invocationMetrics` usage |
| `ApplyGuardrail` | `span` | `apply_guardrail {guardrailIdentifier}` | guardrail input/output, action, action reason, request id |
| `InvokeAgent` / `InvokeInlineAgent` | `agent` | `invoke_agent {agentId}` / `invoke_agent {agentName}` | user input, streamed answer, session/memory ids, alias id, trace usage aggregation, trace event count, return-control output |
| `Rerank` | `span` (`rerank` operation) | `rerank {modelArn basename}` | queries, inline sources, model request fields, ranked results and scores, result counts, pagination metadata, request id |
| `Retrieve` | `span` | `retrieve {knowledgeBaseId}` | query, retrieval results, guardrail action, result count metadata |
| `RetrieveAndGenerate` / stream | `generation` | `retrieve_and_generate {modelArn basename}` | input text, generated text, citations count, session id, guardrail action |
| `InvokeFlow` | `agent` | `invoke_flow {flowIdentifier}` | inputs, flow output events, completion reason |

## StreamingBody behavior

`InvokeModel` returns a botocore `StreamingBody`. The integration reads it once to capture output and token usage, then replaces it with a new `StreamingBody` over the same bytes. Caller code can still call `response["body"].read()` normally.

Event streams are wrapped lazily. The wrapper does not pre-read events, preserves backpressure, forwards unknown attributes to the inner stream, and calls `close()` on the inner stream when closed.

## Message normalization

Converse messages follow the OpenTelemetry GenAI non-normative LLM-call examples:

- text blocks become `{ "type": "text", "content": ... }`
- `toolUse` becomes `{ "type": "tool_call", "id", "name", "arguments" }`
- `toolResult` becomes `{ "type": "tool_call_response", "id", "response" }`
- reasoning text becomes `{ "type": "reasoning", "content" }`
- image/document/video/audio bytes become blob parts without byte content
- S3/URI sources become `{ "type": "uri", "uri", "modality" }`

InvokeModel captures provider-native JSON bodies verbatim. Non-JSON bodies are not captured.

## Semantics and guarantees

- Provider is emitted as `amazon-bedrock`. This intentionally differs from the OpenTelemetry registry value `aws.bedrock` so telemetry.dev pricing keys match Bedrock model IDs exactly. The raw caller `modelId`, `modelArn`, or `foundationModel` is emitted as the model.
- Instrumentation is fail-open. Normalization, span updates, and span endings are guarded so telemetry failures do not break caller code.
- Requests are never modified.
- Streams end spans exactly once. Exhaustion records full output and finish reason; early `break`, `close()`, `GeneratorExit`, or stream errors record partial output and set `telemetry.dev.capture.truncated`.
- Event-stream reconstruction retains up to 48 KiB and 1,000 items. `InvokeModel` bodies and `Rerank` inputs and results are retained up to 64 KiB and 1,024 items; a body over that bound records no output and sets `telemetry.dev.capture.truncated`. The configured `max_attribute_length` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs.
- A stream cut at its retention bound, closed early, or failed sets `telemetry.dev.capture.truncated`. An `InvokeModel` body that is closed before it is fully read, or whose read fails, records no output and sets the attribute too. With a mask configured, incomplete reconstructed output is omitted because the mask cannot inspect the complete value. Agent trace events share the bound, but dropping them only sets the attribute; Agent text is omitted only when a text chunk was dropped or the stream was closed early or failed. A normally exhausted Agent stream that returned control keeps its complete `returnControl` output even if earlier text or trace metadata passed the bound; an early close or failure still omits the output when a mask is configured.
- boto3 has no async client. `aiobotocore` is a separate package and is out of scope.

## Coverage gaps

The following calls pass through unless a supported operation above is used:

- `CountTokens`
- `InvokeGuardrailChecks`
- `InvokeModelWithBidirectionalStream`
- `StartAsyncInvoke`, `GetAsyncInvoke`, `ListAsyncInvokes`
- Agent Runtime session CRUD
- `GenerateQuery`
- `OptimizePrompt`
- `AgenticRetrieveStream`

Provider-native `InvokeModel` usage is best-effort. Anthropic Claude, Amazon Titan/Nova, Meta Llama, and Titan embeddings expose known token fields. Cohere and Mistral text models do not consistently include usage in the JSON body; spans still capture request/response bodies and finish reasons where present.
