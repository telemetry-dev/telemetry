# @telemetry-dev/tanstack-ai

TanStack AI telemetry integration for [telemetry.dev](https://telemetry.dev). A `chat()` middleware
that streams every run to the telemetry.dev ingest API. Each `chat()` call produces a root span
(operation `chat`, or `invoke_agent` once tools are used), a `chat` span per agent-loop iteration,
and an `execute_tool` span per tool call — spans are typed by `gen_ai.operation.name`. All calls of
one conversation (`metadata.sessionId`, falling back to the chat's `threadId`) share one trace, with
the root of each call as a sibling in start order; the id is also stamped as
`gen_ai.conversation.id` on every span.

## Install

```sh
npm install @telemetry-dev/tanstack-ai @tanstack/ai
```

Requires `@tanstack/ai >= 0.28.0 < 1`.

## Environment

| Variable                    | Required | Default                        | Notes                                                                  |
| --------------------------- | -------- | ------------------------------ | ---------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                              | Ingest key (`td_live_…`). No key ⇒ the middleware is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://ingest.telemetry.dev` | Trailing slashes are stripped.                                         |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                   | Environment label on every trace.                                      |
| `OTEL_SERVICE_NAME`         | no       | `unknown_service`              | Service name on every trace.                                           |

All four are also settable via `telemetryDev({ apiKey, baseUrl, environment, serviceName })`, which
takes precedence over the environment.

Trace sampling uses `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`, which also control session roots.
The default is parent-based always-on.
Pass an OTel `Sampler` as `telemetryDev({ sampler })` to override the environment.
Real active parents keep their sampling decisions. The integration does not send dropped or record-only spans.
The [session sampling contract](../otel/README.md#correlation-attributes) lists environment modes and argument defaults.

## Usage

```ts
import { chat } from "@tanstack/ai";
import { openaiText } from "@tanstack/ai-openai";
import { telemetryDev } from "@telemetry-dev/tanstack-ai";

const stream = chat({
  adapter: openaiText("gpt-4o"),
  messages,
  metadata: { userId: "u_123", sessionId: "s_456", tenant: "acme" },
  middleware: [telemetryDev()],
});
```

`metadata.userId` is recorded as the `user.id` span attribute and `metadata.sessionId` as
`gen_ai.conversation.id` (when absent, the chat's `threadId` is used); any remaining metadata keys
ride along as `td.metadata.<key>` attributes. Calls that share a conversation id share one trace.

A single `telemetryDev()` instance is **concurrency-safe**: per-run state is keyed by the chat's
middleware context, so you can create one at module scope and share it across overlapping `chat()`
calls:

```ts
const telemetry = telemetryDev();

// reuse in every handler
chat({ adapter, messages, middleware: [telemetry] });
```

### Serverless (Cloudflare Workers etc.)

By default the terminal hook awaits the ingest POST so the runtime doesn't tear down before it
flushes. Supply `waitUntil` to hand the POST off to the platform instead:

```ts
telemetryDev({ waitUntil: (p) => ctx.waitUntil(p) });
```

## What gets captured

- **Tokens:** input/output per iteration, plus cache-read, cache-write, and reasoning token
  breakdowns and provider-reported cost when the adapter supplies them
  (`gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`,
  `gen_ai.usage.reasoning.output_tokens`, `gen_ai.usage.cost`). Absent fields are omitted, never
  zeroed; when no provider cost is reported, cost is computed server-side from pricing tables.
- **Content:** the per-iteration request messages (`gen_ai.input.messages`, exactly what the
  adapter sends) and the assistant text (`gen_ai.output.messages`); tool calls carry
  `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`.
- **Sampling:** `gen_ai.request.temperature` / `top_p` / `max_tokens`, read across provider-native
  spellings (including Ollama's nested `options`).
- **Errors:** a run that errors or aborts is still flushed — open spans are closed with
  `status: "error"` and an `error.type` attribute (`"cancelled"` for aborts); failed tool calls get
  an `exception` event.
- **Structured output:** when `chat({ outputSchema })` finalizes through a separate
  structured-output model call, that call gets its own span (`gen_ai.output.type: "json"`, output
  set to the raw JSON) and its usage is counted alongside — never instead of — the agent loop's.
- **Metrics:** `gen_ai.client.operation.duration` per iteration and tool call, and
  `gen_ai.client.token.usage` per iteration, both following the OTel GenAI semantic conventions.

## Limitations

- **TTFT:** time-to-first-chunk is not currently reported.
- Middleware hook exceptions are wrapped in `try/catch` and routed to the optional `onError`
  callback so instrumentation can never break your chat.
