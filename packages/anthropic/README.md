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

Global instrumentation patches both the stable and beta `Messages.prototype.create`, so it covers `messages` and `beta.messages` on every client that loads the same copy of `@anthropic-ai/sdk` through `import` (see Limitations). Call it once during process startup. `uninstrumentAnthropic()` restores both methods.

## Instrumented surfaces

- `client.messages.create(...)`, including `stream: true` responses.
- `client.messages.stream(...)` when the SDK helper is used.
- `client.messages.parse(...)`, which routes through `create()`.
- `client.beta.messages.create(...)`, `stream(...)`, and `parse(...)`.
- `client.beta.messages.toolRunner(...)`: each model request in the tool loop records its own generation span.
- Anthropic, Bedrock, Bedrock Mantle, and Vertex clients, including subclasses. Provider attributes are emitted as `anthropic`, `aws.bedrock` (Bedrock and Bedrock Mantle), or `gcp.vertex_ai`. `AnthropicAws` (Claude Platform on AWS) and `AnthropicFoundry` (Microsoft Foundry) serve the Anthropic API, so they are recorded as `anthropic`.

Captured request fields include model, max tokens, temperature, top-p, stop sequences, system instructions, tools, and messages. Captured response fields include model, finish reason, content blocks, message id, and token usage including cache and thinking token details when Anthropic returns them.

## Streaming behavior

Streaming spans start when the request is made and end when the stream is consumed, closes early, or errors. The integration aggregates text, tool input JSON fragments, thinking deltas, signatures, usage, and the latest known stop reason before ending the span.

Reconstructed output is bounded to 48 KiB and 1,000 items. The configured `maxAttributeLength` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs, with its `...[truncated]` marker, without setting `telemetry.dev.capture.truncated`. A stream that ends without `message_stop`, encounters a mapping failure, or
exceeds a capture bound sets `telemetry.dev.capture.truncated`. With a mask configured, incomplete
output is omitted because the mask cannot inspect the complete value. When output capture is
disabled, response IDs and models, finish reasons, usage, timing, and errors may still be recorded.
Capture flags do not gate stop sequences, caller-supplied metadata or raw attributes, or exception
messages and stack traces; redact those separately when needed. Request tools and `tool_choice` are part of the captured input, so `captureInput: false` removes them.

## Limitations

- `messages.countTokens`, `beta.messages.countTokens`, batches, and other non-Messages surfaces are not instrumented.
- `with_raw_response` and `with_streaming_response` helper namespaces are not patched directly.
- If application code never consumes or closes a stream, the span cannot finish until the stream is finalized by the runtime.
- Global instrumentation patches the ES module build of the `@anthropic-ai/sdk` copy this package resolves. Clients loaded through `require()` use the package's CommonJS build, which has separate classes. The Bedrock, Vertex, AWS, and Foundry client packages depend on `@anthropic-ai/sdk` 0.115.1 or newer, so a package manager can also install them a separate copy. Wrap those clients with `wrapAnthropic()`.
