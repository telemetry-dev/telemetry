# @telemetry-dev/sdk

Pure TypeScript SDK for [telemetry.dev](https://telemetry.dev): add LLM traces, logs, and GenAI
metrics to any app. Built on OpenTelemetry — spans use the `gen_ai.*` semantic conventions and
ship as OTLP protobuf to `https://ingest.telemetry.dev`. Works on Node >= 20.19 and edge runtimes
(Vercel Edge, Cloudflare Workers with `nodejs_compat`). ESM-only; CJS consumers can `require()` it
on Node >= 20.19.

```sh
npm install @telemetry-dev/sdk
```

## Quickstart

```ts
import { init, observe, startSpan, propagateAttributes, log, flush } from "@telemetry-dev/sdk";

init({ apiKey: "td_live_..." }); // or TELEMETRY_DEV_API_KEY

const answer = observe(
  async function answer(question: string) {
    const generation = startSpan("chat-completion", {
      type: "generation",
      model: "gpt-4o",
      provider: "openai",
      input: [{ role: "user", content: question }],
    });
    const res = await llm.chat(question);
    generation.end({
      output: res.message,
      usage: { inputTokens: res.usage.input, outputTokens: res.usage.output },
    });
    log("generation finished", { attributes: { cached: false } });
    return res.message;
  },
  { type: "agent" },
);

await propagateAttributes({ userId: "user_123", sessionId: "conv_42" }, () =>
  answer("What is OTLP?"),
);
await flush();
```

Without an API key every call is a silent no-op — the SDK never throws into your code.

## Configuration

| Option        | Env var                     | Default                        |
| ------------- | --------------------------- | ------------------------------ |
| `apiKey`      | `TELEMETRY_DEV_API_KEY`     | — (absent ⇒ no-op)             |
| `baseUrl`     | `TELEMETRY_DEV_BASE_URL`    | `https://ingest.telemetry.dev` |
| `environment` | `TELEMETRY_DEV_ENVIRONMENT` | `production`                   |
| `serviceName` | `OTEL_SERVICE_NAME`         | `unknown_service`              |

Other `init()` options: `enabled` (kill switch), `registerGlobal`, `exportMode: "batched" |
"immediate"`, `captureInput` / `captureOutput`, `mask(value, { key })` redaction hook,
`maxAttributeLength` (default 65536, truncated values end with `...[truncated]`), `batch`
(`maxExportBatchSize` 64, `scheduledDelayMillis` 1000, `maxQueueSize` 2048,
`exportTimeoutMillis` 30000), `spanFilter`, `resourceAttributes`, `logLevel` (SDK diagnostics,
default `"warn"`), `fetch`, `waitUntil`, `onError`.

Root spans remain independent by default. `init({ sessionMode: "process" })` groups otherwise
uncorrelated spans and logs using a fresh opaque UUID for each enabled SDK initialization.
Explicit and propagated session IDs take precedence, and valid parent trace context is preserved.
The generated session ends when the client is shut down or replaced; a later `init()` generates a
new ID. It is not persisted across processes.

`sampler` accepts an OpenTelemetry `Sampler` and overrides `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`.
The default is parent-based always-on. Session roots keep the configured root policy.
Nested spans and explicitly parented spans keep the real parent's policy.
The SDK installs the [session sampling wrapper](../otel/README.md#correlation-attributes) on its provider. This includes the global provider that Eve uses.

## API

- `init(options?)` — create the client (isolated OTel TracerProvider by default).
- `observe(fn, options?)` — wrap a sync/async function: args → input, return → output, errors
  captured and rethrown; activates context for nesting.
- `startSpan(name, options?)` — manual handle; does **not** activate context. `handle.update()`,
  `handle.end()`, `handle.recordOutputChunk(timestampMs?)`, `handle.traceparent`, `handle.span`
  (raw OTel span).
- `startActiveSpan(name, options?, fn)` — callback-scoped; activates context, auto-ends.
- `updateActiveSpan(fields)` — update the innermost active span.
- `propagateAttributes({ userId, sessionId, metadata }, fn)` — stamps `user.id`,
  `gen_ai.conversation.id`, and `td.metadata.*` on every span and log record in scope.
- `log(message, { level, eventName, attributes }?)` — log record correlated to the active trace.
- `getTraceparent()` — W3C traceparent of the active span (pass to other services; accept it via
  `startSpan(name, { parent })`).
- `activeContext()` / `withContext(context, fn)` — read or temporarily activate an OpenTelemetry
  context.
- `extractW3cContext(carrier, options?)` / `injectW3cContext(context, carrier, options?)` — extract
  or inject W3C trace context. Baggage is omitted unless `includeBaggage` is true.
- `flush()` / `shutdown()` — export pending data; required before serverless freeze/exit.

Span types: `span` (default) | `generation` | `tool` | `agent` | `embedding` — mapped to
`gen_ai.operation.name` `function` / `chat` / `execute_tool` / `invoke_agent` / `embeddings`.
Generation fields (`model`, `provider`, `usage`, `costUsd`, sampling params, …) map to the OTel
GenAI semantic conventions; cost is computed server-side from usage unless `costUsd` is set.

Duration and token-usage histograms (`gen_ai.client.operation.duration`,
`gen_ai.client.token.usage`) are recorded automatically for generation/agent/embedding/tool spans.

Generation spans also emit `gen_ai.client.operation.time_to_first_chunk` when a valid
`timeToFirstChunkMs` is supplied. `handle.recordOutputChunk()` records each nonempty output chunk's
arrival; after the first chunk, intervals contribute to
`gen_ai.client.operation.time_per_output_chunk`. Both histograms use seconds and are emitted only
after the ended span passes `spanFilter`, including interrupted streams. Chunk timing uses bounded
histogram state and does not retain individual intervals. Pull-based timing includes consumer delay
between reads. An explicit timestamp uses the same monotonic millisecond clock for every chunk.

Provider integrations feature-detect chunk recording. With an older core SDK that lacks
`recordOutputChunk`, streams and existing tracing continue normally, but chunk-interval metrics are
unavailable. Upgrade the core SDK with provider integrations to enable the new metric.

## Serverless

Use `exportMode: "immediate"` and either `await flush()` before returning or pass the platform
extender: `init({ waitUntil: (p) => ctx.waitUntil(p) })`. In batched mode spans wait up to
`scheduledDelayMillis` for a timer that frozen isolates may never fire.

## Bring your own OpenTelemetry

If your app already runs an OTel setup (NodeSDK, `@vercel/otel`), don't use `init()` — attach the
span processor to your provider; it exports every span it sees (narrow with `spanFilter`). The BYO
surface ships as its own package and the SDK is not required for it:

```sh
npm i @telemetry-dev/otel
```

```ts
import { NodeSDK } from "@opentelemetry/sdk-node";
import { TelemetrySpanProcessor } from "@telemetry-dev/otel";

const sdk = new NodeSDK({ spanProcessors: [new TelemetrySpanProcessor()] });
sdk.start();
```

Alternatively `init({ registerGlobal: true })` makes the SDK's provider the app's global one; in
that mode only this SDK's spans are exported by default.

Any vanilla OTel app can also skip the SDK entirely: point a stock OTLP/HTTP protobuf exporter at
`https://ingest.telemetry.dev/v1/traces` with an `Authorization: Bearer td_live_...` header.

## Notes

- AsyncLocalStorage powers context propagation. On runtimes without it (Workers without
  `nodejs_compat`) automatic nesting is unavailable entirely unless the host app registered its
  own global OTel context manager — pass `parent` explicitly there.
- `@opentelemetry/api` is a peer dependency (installed automatically by npm/pnpm) so the SDK
  shares the host app's OTel API singleton.
