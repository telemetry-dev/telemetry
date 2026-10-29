# @telemetry-dev/openai

OpenAI SDK instrumentation for telemetry.dev. It wraps the official `openai` Node SDK and emits telemetry.dev generation and embedding spans through `@telemetry-dev/sdk`.

Docs: https://telemetry.dev/docs/integrations/openai

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/openai openai
```

Requires `openai` 6 or 7, and `@telemetry-dev/sdk`, which needs Node.js 20.19.0 or later.

Initialize the core SDK first. `init()` reads your project API key from the `TELEMETRY_DEV_API_KEY` environment variable, so set it before you start the app:

```ts
import { init, flush, shutdown } from "@telemetry-dev/sdk";

init({ serviceName: "my-service" });

// app code...
await flush();
await shutdown();
```

Versions of `@telemetry-dev/sdk` before 0.2.0 sent data to `https://ingest.telemetry.dev` by
default. That hostname reaches the same service and stays supported: if a firewall allows only it,
set `TELEMETRY_DEV_BASE_URL` to it.

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

Use this when you want explicit control over which clients are instrumented. It also works when
your application creates the OpenAI client with CommonJS `require()`.

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

Use this as the app-wide one-liner at startup when all OpenAI clients should be instrumented. Global
instrumentation patches the ESM build imported by this package, so it does not instrument clients
created from the OpenAI SDK's CommonJS build. Use `wrapOpenAI()` for CommonJS clients.

## Instrumented surfaces

- `client.chat.completions.create(...)`
- `client.chat.completions.parse(...)` is covered by the OpenAI SDK because it routes through `create()`
- `client.chat.completions.stream(...)` is covered for the same reason
- `client.responses.create(...)`
- `client.responses.parse(...)` is covered by the OpenAI SDK because it routes through `create()`
- `client.responses.stream({ input, model, ... })` is covered for new responses because it routes through `create({ stream: true })`
- `client.embeddings.create(...)`
- `client.images.generate(...)`, `edit(...)`, and `createVariation(...)`, including image streams
- `client.audio.speech.create(...)`, `transcriptions.create(...)` (including streams), and `translations.create(...)`
- `client.videos.create(...)` and `retrieve(...)` when exposed by the installed OpenAI SDK (legacy: OpenAI shut down the Videos API on 2026-09-24; kept for compatible endpoints)
- `client.batches.create(...)`, `retrieve(...)`, and `cancel(...)`

The integration maps native OpenAI request/response shapes directly into telemetry.dev fields. It does not normalize messages into another schema.

Non-streamed chat and Responses input and output, and Realtime instructions and output, are bounded to 48 KiB and 1,000 items. A value over the bound sets `telemetry.dev.capture.truncated`. Without a mask, the bounded part is kept. With a mask configured, the value is omitted because the mask cannot inspect the complete value. Input is omitted before the span starts, so a custom `sampler` sees the bounded input when no mask is configured; with an older `@telemetry-dev/sdk` that cannot report whether a mask is configured, cut input is omitted as if one were.

## Streaming

Chat completion streams are traced. Requests are sent unchanged by default, so token usage is only captured when the caller sets `stream_options.include_usage` themselves. Pass `{ injectStreamUsage: true }` to `wrapOpenAI` or `instrumentOpenAI` to inject `stream_options.include_usage` automatically; the synthetic usage-only chunk is then hidden from the caller. Injection is opt-in because some providers reject `stream_options` — for example Azure OpenAI "on your data" (`data_sources`) returns 400 for it while plain `stream: true` works.

Streamed chat output and finish reasons use separate budgets of up to 48 KiB and 1,000 items; the configured `maxAttributeLength` does not shrink these retention bounds; the core applies it to the exported attribute after the mask runs, with its `...[truncated]` marker, without setting `telemetry.dev.capture.truncated`. When present, `gen_ai.output.messages` contains bounded partial output. If output and/or finish-reason capture is incomplete, including when a choice ends without `finish_reason`, `telemetry.dev.capture.truncated` is `true`. With a mask configured, incomplete output is omitted because the mask cannot inspect the complete value. When output capture is disabled or omitted, response IDs and models, finish reasons, usage, timing, and errors may still be recorded.

Capture flags do not gate stop sequences or caller-supplied metadata or raw attributes; redact those separately when needed. Exception messages and stack traces are recorded only when both `captureInput` and `captureOutput` are on, and pass through `mask`. Older `@telemetry-dev/sdk` versions record them regardless of capture flags.

```ts
const openai = wrapOpenAI(new OpenAI(), { injectStreamUsage: true });
```

Responses API streams are traced through `responses.create({ stream: true })`; terminal `response.completed`, `response.failed`, and `response.incomplete` events close the span.

Responses and transcription streams follow the same rule. A Responses stream that ends without a terminal snapshot, ends with an `error` event, fails to map an event, or has a terminal snapshot over the limit sets `telemetry.dev.capture.truncated`; it keeps the last snapshot that fit, unless a mask is configured, in which case output is omitted. Transcription text has its own retention limit of up to 65,536 characters of serialized JSON (UTF-16 code units) with no item limit. A complete non-streamed transcript is passed to the core whole. A transcription stream that ends before `transcript.text.done` (an error, early close, or end of stream) or whose text exceeds the limit is flagged the same way and keeps its bounded partial text only when no mask is configured.

## Embeddings

Embedding calls emit `gen_ai.operation.name = "embeddings"`, request model/input, response model, and token usage. Embedding vectors are intentionally not captured as output.

## Provider detection

`wrapOpenAI(new AzureOpenAI(...))` records provider `azure.ai.openai`, including clients created with
CommonJS `require()`. Global prototype instrumentation detects Azure clients from the ESM resource's
client.

OpenAI clients configured with `https://openrouter.ai/api/v1` as their base URL record provider `openrouter`. The same detection applies to OpenRouter subdomains; unrelated hosts containing `openrouter.ai` are not matched.

OpenAI-compatible URLs on Groq, xAI, DeepSeek, Together, and Fireworks domains (including their subdomains) record the corresponding provider. Other compatible endpoints retain the `openai` default.

## Realtime WebSocket

Realtime WebSocket emitters require explicit wrapping; global prototype instrumentation does not cover them:

```ts
import { OpenAIRealtimeWS } from "openai/realtime/ws";
import { wrapOpenAIRealtime } from "@telemetry-dev/openai";

const realtime = wrapOpenAIRealtime(new OpenAIRealtimeWS({ model: "gpt-realtime" }), {
  model: "gpt-realtime",
});
```

The wrapper creates one generation span for each `response.create`, finishes it on `response.done`, transport error, or early close, and records output timing and modality usage. To match responses that can complete out of order, it adds a private correlation field to `response.metadata` without replacing caller fields. OpenAI permits at most 16 metadata fields; when the caller already supplies 16, the request is sent unchanged and the wrapper emits an immediately failed span because response events cannot be correlated safely. By default, an unfinished span fails after five minutes, and the wrapper retains at most 100 in-flight spans per connection. A positive finite `traceTimeoutMs` and a positive-integer `maxInFlight` override those limits; exceeding the in-flight limit fails and removes the oldest span. Wrapping the same emitter more than once is safe.

Image, audio, and video request/response bytes are never captured, including binary fields nested in Responses API input and output. Non-streaming image spans retain the prompt, image count, revised prompts, and token usage. Streamed image spans retain supported timing and usage only. Speech spans retain text input but not returned audio; transcription and translation spans retain returned text and token usage. Batch spans use the custom `openai.batch.create`, `openai.batch.retrieve`, and `openai.batch.cancel` operations because OpenTelemetry defines no standard GenAI batch lifecycle operation.

## Limitations

- Wrapped calls preserve `withResponse()` and `asResponse()`, but the returned promise is not guaranteed to be `instanceof` the OpenAI SDK's internal `APIPromise` class.
- Unawaited OpenAI calls keep the SDK's lazy behavior: no request is made and no span is finished until the returned promise/stream is consumed.
- Realtime instrumentation is explicit through `wrapOpenAIRealtime`; `instrumentOpenAI()` does not automatically patch Realtime WebSocket prototypes.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/openai/LICENSE).
