# @telemetry-dev/anthropic

Anthropic SDK instrumentation for telemetry.dev. It wraps the Anthropic Messages API and emits OpenTelemetry GenAI spans through `@telemetry-dev/sdk`.

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/anthropic @anthropic-ai/sdk
```

Set `TELEMETRY_DEV_API_KEY` for telemetry export and `ANTHROPIC_API_KEY` for Anthropic.

## Per-client wrapping

```ts
import Anthropic from "@anthropic-ai/sdk";
import { init, shutdown } from "@telemetry-dev/sdk";
import { wrapAnthropic } from "@telemetry-dev/anthropic";

init({ serviceName: "anthropic-worker" });

const anthropic = wrapAnthropic(new Anthropic());
const response = await anthropic.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "Say hello." }],
});

await shutdown();
```

`wrapAnthropic(client)` mutates and returns that client. It is idempotent.

## Global instrumentation

```ts
import { instrumentAnthropic, uninstrumentAnthropic } from "@telemetry-dev/anthropic";

instrumentAnthropic();
// new Anthropic().messages.create(...) is now traced.
uninstrumentAnthropic();
```

Global instrumentation patches `Messages.prototype.create`; call it once during process startup.

## Instrumented surfaces

- `client.messages.create(...)`, including `stream: true` responses.
- `client.messages.stream(...)` when the SDK helper is used.
- Anthropic, Bedrock, and Vertex clients. Provider attributes are emitted as `anthropic`, `aws.bedrock`, or `gcp.vertex_ai`.

Captured request fields include model, max tokens, temperature, top-p, stop sequences, system instructions, tools, and messages. Captured response fields include model, finish reason, content blocks, message id, and token usage including cache and thinking token details when Anthropic returns them.

## Streaming behavior

Streaming spans start when the request is made and end when the stream is consumed, closes early, or errors. The integration aggregates text, tool input JSON fragments, thinking deltas, signatures, usage, and the latest known stop reason before ending the span.

## Limitations

- `beta.messages`, `messages.countTokens`, and other non-Messages surfaces are not instrumented.
- `with_raw_response` and `with_streaming_response` helper namespaces are not patched directly.
- If application code never consumes or closes a stream, the span cannot finish until the stream is finalized by the runtime.
