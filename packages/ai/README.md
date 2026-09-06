# @telemetry-dev/ai-sdk

Vercel AI SDK telemetry integration for [telemetry.dev](https://telemetry.dev). Streams AI SDK
runs to the telemetry.dev ingest API. The package ships one entry per supported `ai` major:

- `@telemetry-dev/ai-sdk` — for `ai@7`. Covers `generateText` / `streamText` / `Agent` /
  `generateObject` / `streamObject` / `embed` / `embedMany` / `rerank`.
- `@telemetry-dev/ai-sdk/v6` — for `ai@6`. Covers `generateText` / `streamText` / `Agent`.

Each call produces a root span (operation `chat`, or `invoke_agent` once tools are used;
`embeddings` / `rerank` for those ops), a `chat` span per model step, and an `execute_tool` span
per tool call — spans are typed by `gen_ai.operation.name`. Calls that carry a `sessionId` share
one trace per session (the root of each call is a sibling under a session parent that is never
emitted, so the trace shows the calls in start order); a call without a session id is its own
trace. The session is also stamped as `gen_ai.conversation.id` on every span.

On `ai@7`, provider requests and a tool's `execute` run inside the step/tool span context, so
auto-instrumented provider spans and nested AI SDK calls made from within a tool parent into the
outer call's trace instead of starting their own.

## Install

```sh
npm install @telemetry-dev/ai-sdk ai
```

Requires `ai >= 6.0.111 < 8` (import from the entry matching your major).

## Environment

| Variable                    | Required | Default                        | Notes                                                                   |
| --------------------------- | -------- | ------------------------------ | ----------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                              | Ingest key (`td_live_…`). No key ⇒ the integration is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://ingest.telemetry.dev` | Trailing slashes are stripped.                                          |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                   | Environment label on every trace.                                       |
| `OTEL_SERVICE_NAME`         | no       | `unknown_service`              | Service name on every trace.                                            |

All four are also settable via `telemetryDev({ apiKey, baseUrl, environment, serviceName })`, which
takes precedence over the environment.

Trace sampling uses `OTEL_TRACES_SAMPLER` / `OTEL_TRACES_SAMPLER_ARG`, which also control session roots.
The default is parent-based always-on.
Pass an OTel `Sampler` as `telemetryDev({ sampler })` to override the environment for version 6 or version 7.
Real active parents keep their sampling decisions. The integration does not send dropped or record-only spans.
The [session sampling contract](../otel/README.md#correlation-attributes) lists environment modes and argument defaults.

## Usage (ai@7)

```ts
import { generateText } from "ai";
import { telemetryDev } from "@telemetry-dev/ai-sdk";

const { text } = await generateText({
  model,
  prompt: "Summarize the incident report.",
  runtimeContext: { userId: "u_123", sessionId: "s_456", tenant: "acme" },
  telemetry: {
    functionId: "summarize-incident",
    includeRuntimeContext: { userId: true, sessionId: true, tenant: true },
    integrations: [telemetryDev()],
  },
});
```

`functionId` becomes the root-span name (it defaults to `chat`, or `embeddings` / `rerank` for
those operations). User context flows through the call-level `runtimeContext` option — **by
default none of it reaches telemetry integrations**; opt keys in per call via
`telemetry.includeRuntimeContext: { <key>: true }`. Of the included keys, `userId` is recorded as
the `user.id` span attribute and `sessionId` as `gen_ai.conversation.id`; any remaining included
keys ride along as `td.metadata.<key>` attributes. Calls that share a `sessionId` share one trace;
a call without one is its own trace.

`ai@7` still accepts `experimental_telemetry` as a deprecated alias for `telemetry`.

### Global registration

```ts
import { registerTelemetry } from "ai";
import { telemetryDev } from "@telemetry-dev/ai-sdk";

registerTelemetry(telemetryDev());
```

A single global instance is concurrency-safe on `ai@7`: state is keyed by the SDK's per-call
`callId`, so overlapping concurrent generations produce disjoint traces.

### Serverless (Cloudflare Workers etc.)

By default the end-of-call flush awaits the ingest POST so the runtime doesn't tear down before it
completes. Supply `waitUntil` to hand the POST off to the platform instead:

```ts
telemetryDev({ waitUntil: (p) => ctx.waitUntil(p) });
```

## Usage (ai@6)

```ts
import { generateText } from "ai";
import { telemetryDev } from "@telemetry-dev/ai-sdk/v6";

const { text } = await generateText({
  model,
  prompt: "Summarize the incident report.",
  experimental_telemetry: {
    functionId: "summarize-incident",
    metadata: { userId: "u_123", sessionId: "s_456", tenant: "acme" },
    integrations: [telemetryDev()],
  },
});
```

`metadata.userId` / `metadata.sessionId` / remaining keys map to the same attributes as the
`runtimeContext` keys above (no opt-in filter exists on `ai@6`).

> **Caveat (ai@6 only):** a single globally-registered instance holds mutable per-generation state
> and is **not** safe for overlapping concurrent generations. Prefer the per-call form above —
> pass a fresh `telemetryDev()` per call so overlapping generations never interleave.

## Logs

Alongside spans, each run emits structured logs (shown on the trace and the Logging page):

- One **info** log per text-generation turn — `Generation completed (<finish_reason>)`, with
  `: <in> in / <out> out tokens` appended when the model reported usage (an **error** log instead
  when `finishReason === "error"`, and a `Generation aborted` **info** log when a streaming call
  is aborted on `ai@7`).
- An **error** `exception` event on each failed tool call; successful tool calls emit no extra
  log.
- A **warn** log for each model warning (unsupported settings, etc.).

Logs are emitted as OTLP span events on the owning span (sent on the same `/v1/traces` request), so
they correlate to the turn/tool they describe.

## Limitations

On `ai@7` (root entry):

- **TTFT** is captured for streaming text steps (from the SDK's
  `performance.timeToFirstOutputMs`) and emitted as `gen_ai.client.operation.time_to_first_chunk`.
- **Thrown errors** are captured via the SDK's `onError` telemetry hook: the trace is flushed with
  `status: "error"` and an `exception` event.
- `generateObject` / `streamObject` / `embed` / `embedMany` / `rerank` are covered.

On `ai@6` (`/v6` entry) all three remain limitations:

- **Supported entry points:** integration hooks fire only for `generateText`, `streamText`, and
  `Agent`. `generateObject` / `streamObject` do **not** fire them and emit no telemetry.
- **TTFT:** there is no first-chunk/TTFT signal on any `ai@6` integration hook, so it is never
  sent by this client on `ai@6`.
- **Thrown errors:** a generation that throws never reaches `onFinish`, so no trace is emitted for
  it. A non-throwing failure (`finishReason === "error"`) and failed tool calls **are** captured
  and marked `status: "error"`.

On both majors:

- **Cost:** cost is computed server-side and is never sent by this client.
- Integration hook exceptions are swallowed by the AI SDK; this client additionally wraps every
  hook in `try/catch` and routes errors to the optional `onError` callback so instrumentation can
  never break your generation.
