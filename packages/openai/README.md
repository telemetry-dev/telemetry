# @telemetry-dev/openai

OpenAI SDK instrumentation for telemetry.dev. It wraps the official `openai` Node SDK and emits telemetry.dev generation and embedding spans through `@telemetry-dev/sdk`.

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/openai openai
```

Initialize the core SDK first:

```ts
import { init, flush, shutdown } from "@telemetry-dev/sdk";

init({
  apiKey: process.env.TELEMETRY_DEV_API_KEY,
  baseUrl: process.env.TELEMETRY_DEV_BASE_URL,
  serviceName: "my-service",
});

// app code...
await flush();
await shutdown();
```

## Per-client wrapping

```ts
import OpenAI from "openai";
import { wrapOpenAI } from "@telemetry-dev/openai";

const openai = wrapOpenAI(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));

await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Tell me a joke about OpenTelemetry" }],
});
```

Use this when you want explicit control over which clients are instrumented.

## Global instrumentation

```ts
import OpenAI from "openai";
import { instrumentOpenAI, uninstrumentOpenAI } from "@telemetry-dev/openai";

instrumentOpenAI();
const openai = new OpenAI();

try {
  await openai.responses.create({
    model: "gpt-4o-mini",
    input: "Tell me a joke about OpenTelemetry",
  });
} finally {
  uninstrumentOpenAI();
}
```

Use this as the app-wide one-liner at startup when all OpenAI clients should be instrumented.

## Instrumented surfaces

- `client.chat.completions.create(...)`
- `client.chat.completions.parse(...)` is covered by the OpenAI SDK because it routes through `create()`
- `client.chat.completions.stream(...)` is covered for the same reason
- `client.responses.create(...)`
- `client.responses.parse(...)` is covered by the OpenAI SDK because it routes through `create()`
- `client.responses.stream({ input, model, ... })` is covered for new responses because it routes through `create({ stream: true })`
- `client.embeddings.create(...)`

The integration maps native OpenAI request/response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

## Streaming

Chat completion streams are traced. Requests are sent unchanged by default, so token usage is only captured when the caller sets `stream_options.include_usage` themselves. Pass `{ injectStreamUsage: true }` to `wrapOpenAI` or `instrumentOpenAI` to inject `stream_options.include_usage` automatically; the synthetic usage-only chunk is then hidden from the caller. Injection is opt-in because some providers reject `stream_options` — for example Azure OpenAI "on your data" (`data_sources`) returns 400 for it while plain `stream: true` works.

```ts
const openai = wrapOpenAI(new OpenAI(), { injectStreamUsage: true });
```

Responses API streams are traced through `responses.create({ stream: true })`; terminal `response.completed`, `response.failed`, and `response.incomplete` events close the span.

## Embeddings

Embedding calls emit `gen_ai.operation.name = "embeddings"`, request model/input, response model, and token usage. Embedding vectors are intentionally not captured as output.

## Provider detection

`wrapOpenAI(new AzureOpenAI(...))` records provider `azure.ai.openai`. Global prototype instrumentation defaults to provider detection from the resource's client when available.

OpenAI clients configured with `https://openrouter.ai/api/v1` as their base URL record provider `openrouter`. The same detection applies to OpenRouter subdomains; unrelated hosts containing `openrouter.ai` are not matched.

## Limitations

- `responses.stream({ response_id: ... })` resumes an existing response through `retrieve()`, which is not instrumented in this version.
- Wrapped calls preserve `withResponse()` and `asResponse()`, but the returned promise is not guaranteed to be `instanceof` the OpenAI SDK's internal `APIPromise` class.
- Unawaited OpenAI calls keep the SDK's lazy behavior: no request is made and no span is finished until the returned promise/stream is consumed.
