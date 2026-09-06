# @telemetry-dev/openrouter

OpenRouter SDK instrumentation for telemetry.dev. It wraps the official `@openrouter/sdk` client and emits generation and embedding spans through `@telemetry-dev/sdk`.

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/openrouter @openrouter/sdk
```

Initialize the core SDK before making OpenRouter calls:

```ts
import { init } from "@telemetry-dev/sdk";

init({
  apiKey: process.env.TELEMETRY_DEV_API_KEY,
  baseUrl: process.env.TELEMETRY_DEV_BASE_URL,
  serviceName: "my-service",
});
```

Calls continue to work when telemetry.dev has not been initialized; instrumentation becomes a no-op.

## Per-client wrapping

```ts
import { OpenRouter } from "@openrouter/sdk";
import { wrapOpenRouter } from "@telemetry-dev/openrouter";

const openRouter = wrapOpenRouter(new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY }));

await openRouter.chat.send({
  chatRequest: {
    model: "openai/gpt-4o-mini",
    messages: [{ role: "user", content: "Tell me a joke about OpenTelemetry" }],
  },
});
```

`wrapOpenRouter()` returns the same client and is idempotent. Use it when only selected clients should be instrumented.

## Global instrumentation

```ts
import { OpenRouter } from "@openrouter/sdk";
import { instrumentOpenRouter, uninstrumentOpenRouter } from "@telemetry-dev/openrouter";

instrumentOpenRouter();
const openRouter = new OpenRouter({ apiKey: process.env.OPENROUTER_API_KEY });

try {
  await openRouter.responses.send({
    responsesRequest: {
      model: "openai/gpt-4o-mini",
      input: "Tell me a joke about OpenTelemetry",
    },
  });
} finally {
  uninstrumentOpenRouter();
}
```

Global instrumentation patches the OpenRouter `Chat`, `Responses`, and `Embeddings` resource prototypes. Repeated installation is safe, and `uninstrumentOpenRouter()` restores the original prototype methods. Clients explicitly passed to `wrapOpenRouter()` remain wrapped after global instrumentation is removed.

## Instrumented surfaces

- `client.chat.send(...)`
- `client.responses.send(...)`
- `client.embeddings.generate(...)`

Every span records `gen_ai.provider.name = "openrouter"`. Chat and Responses calls use the span name `chat <model>` and operation `chat`; embedding calls use `embeddings <model>` and operation `embeddings`.

## Captured fields

Chat spans capture:

- request model and messages
- temperature, top-p, top-k, maximum tokens, stop sequences, seed, frequency penalty, and presence penalty
- response model, response ID, assistant messages, and finish reason or reasons
- prompt/input, completion/output, total, cached-input, and reasoning-output token counts

Responses spans capture:

- request model, input, instructions, temperature, top-p, top-k, maximum output tokens, frequency penalty, and presence penalty
- response model, response ID, output items, status-derived finish reason, and response errors
- input, output, total, cached-input, and reasoning-output token counts

Embedding spans capture:

- request model and input
- response model and response ID
- prompt/input and total token counts

Embedding vectors are never captured as span output.

## Streaming and cost

Chat and Responses streams remain usable as `ReadableStream` instances and async iterables. Instrumentation observes the stream only as the application consumes it. It records time to first chunk, the latest response ID and model, final or last reported usage and cost, terminal errors, and partial data when a stream is cancelled early.

Chat output is reconstructed from consumed deltas when possible, including content, reasoning, refusal text, finish reasons, and fragmented tool-call arguments. Responses streams recognize `response.completed`, `response.failed`, `response.incomplete`, and `error` events. A stream span ends once on a terminal event, normal completion, failure, or cancellation.

Responses stream output can include reconstructed output items followed by bounded `{ type: "telemetry.dev.response_stream_event", event_type, payload }` entries for consumed image-generation lifecycle and partial-image events, apply-patch diff events, fusion lifecycle, analysis, and panel events, web-search lifecycle events, and debug timing events. OpenRouter debug upstream request-body echoes are not retained. These provider-event entries share the stream capture budget and set `telemetry.dev.capture.truncated = true` when only a prefix can be retained.

Requests are passed through unchanged. The integration does not inject `stream_options`, so streamed usage is available only when OpenRouter includes it in the stream.

For every supported response, cost uses the first available value in this order:

1. `usage.cost`
2. `usage.costDetails.upstreamInferenceCost`

The selected value is recorded as `gen_ai.usage.cost`.

## Limitations

- Streaming fields include only events consumed by the application. A returned stream that is never read or cancelled cannot produce a terminal span.
- Streamed output capture is limited to 64 KiB and 1,024 retained items. When a span contains only a captured prefix, `telemetry.dev.capture.truncated` is `true`; terminal metadata and usage can still be recorded.
- Stream reconstruction covers OpenRouter chat content, reasoning, refusal, tool-call deltas, usage, and finish reasons; it does not synthesize fields that OpenRouter did not emit.
- Global instrumentation affects the official `@openrouter/sdk` 1.x resource classes. Standalone generated functions and other OpenRouter endpoints are not instrumented.

## Development

From the repository root:

```sh
vp install
cd packages/openrouter
vp test
vp check
vp run build
```
