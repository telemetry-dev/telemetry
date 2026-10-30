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
A model call outside `withPrompt` does not get the attributes, even when it sends messages from
`compile()`; with input capture on, the call still records those messages as its input.

## Experiments

`runExperiment` runs your code on each item of a dataset that you manage in telemetry.dev, and
records the outputs and the scores as an experiment. The experiment page compares it with the
baseline of the dataset.

```ts
import { init, runExperiment } from "@telemetry-dev/sdk";

init({ environment: "experiment" });

const result = await runExperiment({
  name: "reply-v13-gpt-4o",
  dataset: "support-golden",
  task: (item) => replyToCustomer(item.input),
  evaluators: ["correctness", "helpfulness"],
  scorers: { short: (output) => String(output).length < 600 },
  minScores: { correctness: 0.9 },
});
```

- runExperiment reads the newest version of the dataset, or `datasetVersion`, 100 items at a time.
  With `split`, it runs only the items of that split. Then it makes the experiment, with a name
  that telemetry.dev picks when you give no `name`.
- `task(item)` gets the item: `id`, `number`, `input`, `expectedOutput`, `metadata`, `tags`,
  `split`, and `runIndex`. Its return value, or the value of its promise, is the output. A string
  stays as it is, and a different value becomes JSON, cut to 100,000 characters. A value that JSON
  cannot hold, such as an object that holds itself, gives no output. A task that throws fails only
  its item, and the item records the name and the message of the error.
- `concurrency` (1 to 100, default 4) items run at the same time. With `runsPerItem` (default 1,
  at most 5), each item runs more than one time. Each run gets its own copy of the item, so a task
  that changes its input does not change the next run. The pass rate and the mean of each score
  use all runs.
- Each item run is a trace of its own, with the span `experiment <name>`. The SDK puts
  `td.experiment.id` and `td.dataset.item_id` on each span of the run, also on the spans of the
  provider integrations. Each run also gets its own session id,
  `<experiment id>:<item id>:<run index>`, as `gen_ai.conversation.id`. Spans from the AI SDK and
  TanStack AI integrations do not get these attributes.
- Call `init({ environment: "experiment" })` before the run. Pages, metrics, and alerts leave out
  the `experiment` environment by default. In a different environment, runExperiment logs a
  warning. Without a client, the items record no traces and no cost.
- `evaluators` names evaluators of the project: LLM judges, code checks, and rules. telemetry.dev
  runs them on the outputs. `scorers` run on your machine, after the span of the item ends. A
  scorer returns `true` or `false` for a pass or a fail, a number, a label, or an object with
  `value`, `label`, and `reason`. A scorer that throws, or returns a score that is not valid,
  gives no score for that item and reports a `scorer_failed` error through `onError`. A scorer
  cannot have the name of an evaluator of the run, or the name `user_feedback`.
- runExperiment sends the outputs in batches of 100 items at most, and then waits up to 30
  minutes for the server evaluators. Then it prints the scores and returns the result: `status`,
  `checkStatus`, item counts, `scores` with the value, the baseline value, the change, and the
  pass mark of each score, and the `url` of the experiment page.
- `minScores` sets a pass mark for each score name: the lowest pass rate (0 to 1) of a pass/fail
  score, or the lowest mean of a number score. A score under its mark fails the check.
- A failed request rejects with an `ExperimentError` whose `code` tells why, for example
  `dataset_not_found`, `dataset_empty`, `evaluator_not_found`, `insufficient_scope`,
  `missing_api_key`, or `timeout`. A read of the dataset or the result, a post of outputs, and the
  completion go again up to 2 times after a timeout, a network error, a 429, or a 5xx. The server
  skips an item run that it has. The request that makes the experiment goes again only after a
  429, thus a lost response or a 503 cannot make a second experiment. When the run stops after it
  made the experiment, runExperiment sends the completion before it rejects: telemetry.dev then
  ends the experiment with the item runs that it got, or fails it when it got none.
- The key needs the `datasets:read` and `experiments:write` scopes. runExperiment uses the same
  API key, base URL, and `fetch` as getPrompt, or its own `apiKey`, `baseUrl`, and `fetch`
  options. With `environment: "experiment"`, the key of the telemetry client must be for all
  environments or for `experiment`: a key for another environment cannot send the traces of the
  items, and the items then have no trace and no cost.

The outputs, the errors of failed tasks, the scorer reasons, and `metadata` go to telemetry.dev
and stay with the experiment, because they are its result. `captureOutput`, `mask`, and the
capture setting of the environment in the project do not apply to them. They apply only to the
spans of the run.

### CI mode

When the `CI` variable is `true` or `1`, or with `ci: true`, runExperiment rejects when the check
fails (`check_failed`) or when the experiment does not complete (`experiment_failed`), so the
process exits with an error. The error has the `result`. In GitHub Actions, the experiment also
records the repository, the commit, the run URL, and the number, the title, and the URL of the pull
request. The commit of a pull request is its head commit.

The [experiment action](../../actions/experiment) runs your eval script in a workflow, and posts
the result on the pull request. It gives runExperiment the `dataset` input in
`TELEMETRY_DEV_EXPERIMENT_DATASET`, which must agree with the `dataset` of the script, and the
`min-scores` input in `TELEMETRY_DEV_EXPERIMENT_MIN_SCORES`, which wins over `minScores` for the
same name.

## Scores

`score` records a score on a trace, a span, or a session, for example the thumbs up or down of a
user on a reply:

```ts
import { score, startActiveSpan } from "@telemetry-dev/sdk";

// When the app replies, keep the trace id of the reply with the message:
const reply = await startActiveSpan("reply", async (span) => ({
  text: await replyToCustomer(message),
  traceId: span.traceId,
}));

// Later, when the user clicks thumbs down:
await score({ name: "user_feedback", traceId: reply.traceId, value: 0, comment: "Wrong amount" });
```

- Give `traceId`, `sessionId`, or the two, and a `spanId` of the trace for a score on one span.
  Give `value`, `label`, or the two. `user_feedback` takes the value 1 for up or 0 for down, and
  no label. A pass/fail score has the label `pass` with the value 1, or `fail` with 0.
- Like the telemetry functions, score never rejects. It resolves with the id of the score, or with
  `undefined` after it reports a `ScoreError` through `onError`. It checks the score before it
  sends it (`invalid_score`). The trace id of a span from a client that is not initialized is all
  zeros, and score reports it as `invalid_score`.
- score makes one request, with no retry, and waits 10 seconds at most. The key needs the
  `scores:write` scope. A browser must not hold the key, so call score from your server.

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
- `runExperiment(options)` — run your code on a dataset as an experiment; see
  [Experiments](#experiments).
- `score(options)` — record a score on a trace, a span, or a session; see [Scores](#scores).
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
