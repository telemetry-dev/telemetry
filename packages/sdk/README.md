# @telemetry-dev/sdk

Pure TypeScript SDK for [telemetry.dev](https://telemetry.dev): add LLM traces, logs, and GenAI
metrics to any app. Built on OpenTelemetry — spans use the `gen_ai.*` semantic conventions and
ship as OTLP protobuf to `https://api.telemetry.dev`. Works on Node >= 20.19 and edge runtimes
(Vercel Edge, and Cloudflare Workers with Node.js compatibility, which is on by default for
compatibility dates of 2026-08-04 or later; earlier dates need the `nodejs_compat` flag). ESM-only;
CJS consumers can `require()` it on Node >= 20.19.

Docs: https://telemetry.dev/docs/sdk/typescript

```sh
npm install @telemetry-dev/sdk
```

## Quickstart

Set `TELEMETRY_DEV_API_KEY` to your project API key. `init()` reads it from the environment.

```ts
import { init, observe, startSpan, propagateAttributes, log, flush } from "@telemetry-dev/sdk";

init();

// Stand-in for your model client, so the example runs on its own. Replace it with a real call.
async function callModel(question: string) {
  return {
    message: { role: "assistant", content: `You asked: ${question}` },
    usage: { input: 12, output: 9 },
  };
}

const answer = observe(
  async function answer(question: string) {
    const generation = startSpan("chat-completion", {
      type: "generation",
      model: "gpt-4o",
      provider: "openai",
      input: [{ role: "user", content: question }],
    });
    const res = await callModel(question);
    generation.end({
      output: [res.message],
      usage: { inputTokens: res.usage.input, outputTokens: res.usage.output },
    });
    log("generation finished", { attributes: { cached: false } });
    return res.message.content;
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

| Option        | Env var                     | Default                     |
| ------------- | --------------------------- | --------------------------- |
| `apiKey`      | `TELEMETRY_DEV_API_KEY`     | — (absent ⇒ no-op)          |
| `baseUrl`     | `TELEMETRY_DEV_BASE_URL`    | `https://api.telemetry.dev` |
| `environment` | `TELEMETRY_DEV_ENVIRONMENT` | `production`                |
| `serviceName` | `OTEL_SERVICE_NAME`         | `unknown_service`           |

Earlier versions sent data to `https://ingest.telemetry.dev` by default. That hostname reaches the
same service and stays supported: if a firewall allows only it, set `TELEMETRY_DEV_BASE_URL` or
`baseUrl` to it.

Other `init()` options: `enabled` (kill switch), `registerGlobal`, `exportMode: "batched" |
"immediate"`, `captureInput` / `captureOutput`, `mask(value, { key })` redaction hook,
`maxAttributeLength` (a non-negative safe integer or `Infinity`; default 65536; `null` and
`undefined` select the default, while other invalid values disable telemetry and are reported
through `onError`; truncated values use `...[truncated]`, or its prefix when the cap is shorter than
the marker; the cap also applies, with the marker, to string metadata and log attributes, and any
other string attribute is cut at the cap without a marker, except that a cap of `0` leaves those at
65536), `batch`
(`maxExportBatchSize` 64, `scheduledDelayMillis` 1000, `maxQueueSize` 2048,
`exportTimeoutMillis` 30000), `spanFilter`, `resourceAttributes`, `logLevel` (SDK diagnostics,
default `"warn"`), `fetch`, `waitUntil`, `onError`.

Core SDK error recording keeps failure status, error type, and an exception event. It includes
`exception.message` and `exception.stacktrace` only when both effective `captureInput` and
`captureOutput` are true, including per-span overrides: exceptions can echo either inputs or
outputs. The policy is per span: when an error propagates to an enclosing span whose capture is on,
that span records its message and stack. Included fields pass through `mask` with their exception
attribute keys, then serialization and truncation. A throwing mask drops the affected field and reports a generic error to `onError`
and SDK diagnostics without the hook's original message or cause. The original application error
is rethrown unchanged. These controls do not scrub error type names, metadata, identifiers, resource
attributes, or events written directly through raw OpenTelemetry APIs.

`captureInput: false` and `captureOutput: false` omit captured content. Provider integrations may
still record non-content metadata such as response IDs and models, finish reasons, usage, timing,
and error types. Capture flags do not gate stop sequences, tool descriptions and definitions
(`gen_ai.tool.description`, `gen_ai.tool.definitions`), caller-supplied metadata or raw attributes;
redact those separately when needed.

`mask` receives input and output content, system instructions, log messages, log attributes
other than numbers and booleans, and content that integrations pass to
`handle.serializeContent`, including captured exception messages and stack traces. `ctx.key` names
the attribute being written. It does not receive span or propagated `metadata` or raw `attributes`;
redact those separately. Log masking is independent of span capture flags.

Root spans remain independent by default. `init({ sessionMode: "process" })` groups otherwise
uncorrelated spans and logs using a fresh opaque UUID for each enabled SDK initialization.
Explicit and propagated session IDs take precedence, and valid parent trace context is preserved.
The generated session ends when the client is shut down or replaced; a later `init()` generates a
new ID. It is not persisted across processes.

`sampler` accepts an OpenTelemetry `Sampler` and overrides `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`.
The default is parent-based always-on. Session roots keep the configured root policy.
Nested spans and explicitly parented spans keep the real parent's policy.
The SDK installs the [session sampling wrapper](https://github.com/telemetry-dev/telemetry/blob/main/packages/otel/README.md#correlation-attributes) on its provider. This includes the global provider that Eve uses.

## Prompts

`getPrompt` fetches a prompt that you manage in telemetry.dev. Fetch it by its label, so that you
deploy a version when you move the label, without a change to the code:

```ts
import OpenAI from "openai";
import { getPrompt, init, withPrompt } from "@telemetry-dev/sdk";
import { wrapOpenAI } from "@telemetry-dev/openai";

init();
const openai = wrapOpenAI(new OpenAI());

const prompt = await getPrompt("support/reply", { label: "production" });
const messages = prompt.compile({ customer_message, tone });

const response = await withPrompt(prompt, () =>
  openai.chat.completions.create({ model: prompt.model?.name ?? "gpt-4o", messages }),
);

// Pin one version, as in a test:
const pinned = await getPrompt("support/reply", { version: 13 });
```

- Without `label` and `version`, getPrompt fetches the `production` label. A text prompt needs
  `type: "text"`; getPrompt rejects a prompt of the other type.
- `compile(variables)` puts the values in place of `{{name}}`. A missing value becomes an empty
  text, and a value that is not a string becomes JSON. In a chat prompt, a placeholder message
  becomes the list of messages in its variable. A text prompt compiles to a string.
- The prompt also has `model`, `parameters`, `tools`, and `responseFormat` of its version.
- getPrompt keeps each prompt. When the copy is older than `cacheTtlSeconds` (default 60), a call
  returns it at once and fetches a new copy in the background. Only a call with no copy waits for a
  fetch, for 10 seconds at most. Calls for the same prompt share one fetch.
- If a fetch fails, getPrompt returns the last copy that it fetched and reports the error through
  `onError`. With no copy, it returns the `fallback` option (`version` is `null` and `isFallback`
  is `true`). With no fallback, it rejects with a `PromptError` whose `code` tells why, for
  example `prompt_not_found`, `label_not_found`, `insufficient_scope`, `missing_api_key`, or
  `timeout`. A response with a different status and no API error gives `http_<status>`, for
  example `http_302` for a redirect. getPrompt does not send the request to the location of a
  redirect, thus your key does not go to a different host. With no copy, the calls of the next 10
  seconds (or the TTL, if it is shorter) get the same result without a fetch. After a 429 or 503
  response with a `Retry-After` header, getPrompt does not fetch that prompt again before that time
  (one hour at most). This is the one SDK function that can reject: your app cannot run without
  its prompt.
- getPrompt uses the API key and base URL of an enabled `init()`, or `TELEMETRY_DEV_API_KEY` and
  `TELEMETRY_DEV_BASE_URL`, or its own `apiKey`, `baseUrl`, and `fetch` options. It reads the same
  host that telemetry goes to, `https://api.telemetry.dev` by default. The key needs the read
  scope for prompts.
- When a label sends part of the traffic to a second version, each session keeps one version: the
  choice uses the session id of `propagateAttributes` (or of `sessionMode: "process"`). Without a
  session id, each call chooses at random.

`withPrompt(prompt, fn)` records the prompt on each model call that starts inside `fn`:
`td.prompt.name`, `td.prompt.version`, and `td.prompt.label` (only for a prompt fetched by label).
When input capture is on, `td.prompt.variables` holds the JSON of the variables of the last
`compile()` of that prompt object before `withPrompt` starts, or of the last `compile()` inside
`fn`, after `mask` and truncation. If tasks that run at the same time share one prompt object, call
`compile()` in `withPrompt`, or call getPrompt in each task. getPrompt returns a new object from
the cache. If you do not, a task can record the variables of a different task. Tasks that run at the
same time inside one `withPrompt` share its variables, so call `withPrompt` in each task that
compiles its own. Model calls are the spans of type `generation` (operations `chat`,
`text_completion`, and `generate_content`), including the spans of the provider integrations that
use this SDK: OpenAI, Anthropic, Amazon Bedrock, Google GenAI, and OpenRouter. Spans from the AI SDK
and TanStack AI integrations, and from other OpenTelemetry tracers, do not get the attributes.

Outside `withPrompt`, those provider integrations record the prompt of the message objects that
`compile()` returned. `compile()` of a chat prompt tags each message that it makes with its prompt
and a copy of its variables and of each list in them, in a non-enumerable symbol property: the keys
and the JSON of the message do not change. The objects in a list are not copied, so a change to one
of them after `compile()` changes the variables that a call records. A chat call that sends tagged
messages gets `td.prompt.name`, `td.prompt.version`, and `td.prompt.label` of the newest `compile()`
among them, also when input capture is off, and `td.prompt.variables` when it is on. The SDK looks
at the first 1000 and the last 1000 messages of the call. A message from a placeholder variable
keeps the tag of its own, older `compile()`, so it does not win over the `compile()` that put it in
the list. A copy of a message (a spread, `structuredClone`, or JSON) and a new message with the same
text have no tag, so the SDK does not record them. Bedrock Converse and Google GenAI take another
message shape: change the compiled messages in place, or use `withPrompt`. A text prompt compiles to
a string, which cannot carry a tag: use `withPrompt`. `withPrompt` wins over the tag. The
integrations do it for OpenAI chat completions and responses, Anthropic messages, Bedrock Converse,
Google GenAI `generateContent`, and OpenRouter chat and responses, but not for Bedrock InvokeModel
(its body is bytes), embeddings, images, audio, video, or OpenAI Realtime.

With input capture on in the client when `compile()` runs, a tagged message keeps the variables of
its `compile()` alive while the message lives. With input capture off, the tag keeps no variables.
When `compile()` gets a message in a variable, or in a list in a variable (for example a chat
history in a placeholder), and the message is from a `compile()` at least 100 compiles older, that
older `compile()` drops its variables. Thus a chat history of compiled messages keeps the variables
of about the last 100 compiles, not of each turn. A call that sends only messages of such an older
`compile()` records its prompt without `td.prompt.variables`. The SDK does not look deeper in a
variable for messages.

For your own spans, `promptAttributesFor(messages)` gives the attributes to pass as `attributes`.
It gives nothing inside `withPrompt`, and a span with `captureInput: false` drops
`td.prompt.variables`. A model call whose parent is in `withPrompt` gets the prompt of the
`withPrompt` and none of the `td.prompt.*` attributes that you pass, at start or in `update`,
`end`, or `updateActiveSpan`, also when its `parent` option leads into a `withPrompt` that the call
is not in.

## API

- `init(options?)` — create the client (isolated OTel TracerProvider by default).
- `observe(fn, options?)` — wrap a sync/async function: args → input, return → output, errors
  captured and rethrown; activates context for nesting.
- `startSpan(name, options?)` — manual handle; does **not** activate context. `handle.update()`,
  `handle.end()`, `handle.recordOutputChunk(timestampMs?)`, `handle.traceparent`, `handle.span`
  (raw OTel span).
- `handle.serializeContent(value, key)` — prepare content that an integration writes outside
  `input` and `output`, such as an error message or a raw attribute, the way `input` and `output`
  are prepared: the configured `mask` with `key`, JSON stringify (strings pass through), then
  truncation to `maxAttributeLength`. Returns `undefined` when the mask throws (reported through
  `onError`) or drops the value, or the value cannot be serialized; write nothing then. It does not
  check `captureInput` / `captureOutput`. Integrations that support older cores should treat a
  missing method as a configured mask.
- `startActiveSpan(name, options?, fn)` — callback-scoped; activates context, auto-ends.
- `updateActiveSpan(fields)` — update the innermost active span.
- `propagateAttributes({ userId, sessionId, metadata }, fn)` — stamps `user.id`,
  `gen_ai.conversation.id`, and `td.metadata.*` on every span and log record in scope.
- `getPrompt(name, options?)` — fetch a prompt by label or version; see [Prompts](#prompts).
- `withPrompt(prompt, fn)` — record the prompt on each model call in `fn`.
- `log(message, { level, eventName, attributes }?)` — log record correlated to the active trace.
- `getTraceparent()` — W3C traceparent of the active span (pass to other services; accept it via
  `startSpan(name, { parent })`).
- `activeContext()` / `withContext(context, fn)` — read or temporarily activate an OpenTelemetry
  context.
- `extractW3cContext(carrier, options?)` / `injectW3cContext(context, carrier, options?)` — extract
  or inject W3C trace context. Baggage is omitted unless `includeBaggage` is true.
- `captureEnabled("input" | "output")` — report whether the active initialized client permits that
  capture direction. Returns `false` when no client is initialized.
- `captureMasked()` — report whether the active initialized client has a `mask` configured.
  Returns `false` when no client is initialized.
- `boundedCapture(value, options?)` — copy a JSON-compatible value within bounded work and retained
  size. Defaults to 48 KiB, depth 32, and 1,000 traversal items. Returns `{ value, truncated }`;
  `value` can be partial or `undefined` when a limit, cycle, getter, or skip rule prevents capture.
- `flush()` / `shutdown()` — export pending data; required before serverless freeze/exit.

Span types: `span` (default) | `generation` | `tool` | `agent` | `embedding` — mapped to
`gen_ai.operation.name` `function` / `chat` / `execute_tool` / `invoke_agent` / `embeddings`.
Generation fields (`model`, `provider`, `usage`, `costUsd`, sampling params, …) map to the OTel
GenAI semantic conventions. An explicit `costUsd` takes precedence; otherwise, the server estimates
standard token cost when reported usage and matching model pricing are available. Missing pricing leaves cost
unavailable, not zero, and does not prevent tracing or token accounting.

Duration and token-usage histograms (`gen_ai.client.operation.duration`,
`gen_ai.client.token.usage`) are recorded automatically for generation/agent/embedding/tool spans.

Generation spans also emit `gen_ai.client.operation.time_to_first_chunk` when a valid
`timeToFirstChunkMs` is supplied. `handle.recordOutputChunk()` records each nonempty output chunk's
arrival; after the first chunk, intervals contribute to
`gen_ai.client.operation.time_per_output_chunk`. Both histograms use seconds and are emitted only
after the ended span passes `spanFilter`, including interrupted streams. Chunk timing uses bounded
histogram state and does not retain individual intervals. Pull-based timing includes consumer delay
between reads. An explicit timestamp uses the same monotonic millisecond clock for every chunk.

Provider integrations feature-detect stream APIs. With an older core SDK that lacks the stream
capture policy, streams are still traced with usage, model, finish reason, and timing. Output is
retained within the same fixed bounds as with a current core, but a mask is assumed to be
configured, so incomplete output is withheld. Instrumentation failures are not reported to `onError` with such a core. If only
`recordOutputChunk` is unavailable, tracing and reconstruction continue but chunk-interval metrics
are unavailable. Upgrade the core SDK with provider integrations to enable the full stream
telemetry.

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
`https://api.telemetry.dev/v1/traces` with an `Authorization: Bearer td_live_...` header.

### Native Claude SDK spans

`@anthropic-ai/sdk` 0.134.0 and later emits OpenTelemetry spans natively through the global tracer provider.
To send these spans to telemetry.dev, refer to [Claude SDK](https://telemetry.dev/docs/integrations/claude-sdk).
If you use `init({ registerGlobal: true })`, also set `spanFilter: () => true`. The default filter of that mode exports only the spans of this SDK.
Do not use the native spans and `@telemetry-dev/anthropic` together. If you use the two, each request has two spans.

## Notes

- AsyncLocalStorage powers context propagation. On runtimes without it (Cloudflare Workers without
  Node.js compatibility) automatic nesting is unavailable entirely unless the host app registered its
  own global OTel context manager — pass `parent` explicitly there.
- `@opentelemetry/api` is a peer dependency (installed automatically by npm/pnpm) so the SDK
  shares the host app's OTel API singleton.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/sdk/LICENSE).
