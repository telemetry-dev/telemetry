# @telemetry-dev/eve

Eve telemetry integration for [telemetry.dev](https://telemetry.dev). It wires Vercel's
[Eve](https://eve.dev) agent framework to telemetry.dev through Eve's native OpenTelemetry,
lifecycle-hook, and HTTP-client surfaces.

## Install

```sh
npm install @telemetry-dev/eve eve
```

Requires `eve >=0.47.0 <1` and Node.js 24 or newer.

## Environment

| Variable                    | Required | Default                        | Notes                                                                   |
| --------------------------- | -------- | ------------------------------ | ----------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                              | Ingest key (`td_live_…`). No key ⇒ the integration is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://ingest.telemetry.dev` | Trailing slashes are stripped.                                          |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                   | Environment label on every trace/log.                                   |
| `OTEL_SERVICE_NAME`         | no       | SDK default                    | Server instrumentation falls back to the Eve agent name after this env. |

All four are also settable through SDK options. Pass SDK options to exactly one entry point in a
process; whichever initializes first wins. If you only use `telemetryDevHook()` or
`wrapEveClient()`, pass `serviceName` explicitly or set `OTEL_SERVICE_NAME`.

## Server instrumentation

Create `agent/instrumentation.ts`:

```ts
import { telemetryDevInstrumentation } from "@telemetry-dev/eve";

export default telemetryDevInstrumentation();
```

This registers telemetry.dev as the global OpenTelemetry provider during Eve startup, so Eve's tracer
scope (`"eve"`) and the AI SDK GenAI tracer scope (`"gen_ai"`) export to telemetry.dev.

Useful options:

```ts
export default telemetryDevInstrumentation({
  functionId: "support-agent",
  recordInputs: false,
  recordOutputs: true,
  runtimeContext: { "team.id": "support" },
  stepStarted(input) {
    return { runtimeContext: { "channel.kind": input.channel.kind } };
  },
});
```

`recordInputs` defaults to `captureInput ?? true`; `recordOutputs` defaults to
`captureOutput ?? true`. Runtime context is attached by Eve to model-call spans under the
`ai.settings.context.` prefix. The integration also adds `user.id` when Eve auth exposes a principal.

## Lifecycle logs

Create `agent/hooks/telemetry-dev.ts`:

```ts
import { telemetryDevHook } from "@telemetry-dev/eve";

export default telemetryDevHook();
```

The hook emits bounded lifecycle logs for:

- session start/completion/failure;
- turn start/completion/failure;
- user message receipt without message content;
- step completion/failure with usage where available;
- non-completed tool results;
- HITL input requests and authorization outcomes;
- subagent calls/completions;
- compaction requests/completions.

Every log includes `gen_ai.conversation.id`, `gen_ai.agent.name`, Eve turn/step ids when present, and
`eventName` equal to the Eve stream-event type. Hook handlers are fail-open; instrumentation errors are
routed to `onError` and never thrown back into Eve.

## Client wrapper

For apps that call an Eve agent over HTTP:

```ts
import { wrapEveClient } from "@telemetry-dev/eve";
import { Client } from "eve/client";

const client = wrapEveClient(new Client({ host: "http://127.0.0.1:3000" }), {
  agentName: "support-agent",
});

const { response, session } = await client.sessions.create({
  message: "Summarize this incident.",
});
const result = await response.result();
const followUp = await session.send("What do we do next?");
```

`wrapEveClient()` makes one caller-side `invoke_agent` span for each turn: `sessions.create()`,
`ClientSession.send()`, and `ClientSession.respond()`. It records the outgoing message or the HITL
input responses as span input, sets `gen_ai.conversation.id` from
the session state or Eve's response session id (so all turns of one session share one trace), streams
TTFT from the first message/reasoning delta, accumulates usage from `step.completed`, records final
message/result output, and marks failures/cancellations as errors.

`ClientSession.stream()`, `sessions.attach()`, `Client.info()`, and `Client.health()` are not spanned.
Attach-streams are unbounded and the probe routes are not agent turns.

## Trace shape

A typical turn contains:

- Eve server root span `ai.eve.turn`, normalized by telemetry.dev as `invoke_agent`.
- Nested AI SDK GenAI spans: `chat {modelId}`, `step N`, provider `chat {modelId}` client spans, and
  `execute_tool {toolName}` spans.
- Usage attributes such as `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, cache token
  counts, finish reasons, response ids, TTFT, and time-per-output-chunk when Eve's AI SDK integration
  emits them.
- Runtime context attributes under `ai.settings.context.*`, including Eve session/turn/step/channel
  context and any custom runtime context you return from `step.started`.
- Lifecycle logs joined to the same session via `gen_ai.conversation.id`.

By default, only AI spans are exported: tracer scopes `eve`, `eve.agent`, `gen_ai`, and this SDK. Spans from
other libraries on the same global tracer provider (for example better-auth, Nitro, or eve's
`workflow` engine) are dropped. Pass `spanFilter: () => true` to export every span.

## Content capture

Server-side model content depends on Eve's trace policy and the destination capture settings.
`telemetryDevInstrumentation()` maps `captureInput` / `captureOutput` to `recordInputs` /
`recordOutputs`. The defaults are `true`, but these flags cannot override Eve's trace policy.
In Eve 0.47.0 and 0.50.0, the default policy records content only for a `public` audience,
even if `EVE_DEV=1`. HTTP clients, web chat, and schedules have an `unknown` audience.

For approved local content capture, use Eve's provider layout.
Remove `agent/instrumentation.ts`.
Set `experimental.instrumentationProviders` to `true` in the Eve configuration.
Add these two files:

```ts
// agent/instrumentation/otel.ts
import { otel } from "eve/instrumentation/otel";

export default otel({ tracePolicy: () => true });
```

```ts
// agent/instrumentation/telemetry-dev.ts
import { telemetryDevOtelIntegration } from "@telemetry-dev/eve";

export default telemetryDevOtelIntegration();
```

This policy selects Eve's audience-aware capture.
For local `unknown`-audience channels, run `eve dev` or set `EVE_DEV=1` in the Eve server process.
Without `EVE_DEV=1`, local unknown-audience channels export metadata only.
Local private channels always export metadata only.
Destination settings and forwarded parent policies can decrease content capture.

The legacy single-file `telemetryDevInstrumentation()` API has no `tracePolicy` option.
Thus, it cannot record model content for new unknown-audience sessions on these Eve versions.
The single-file layout remains suitable for metadata-only traces.
Do not combine the two layouts.
This package cannot restore content that Eve removed.

Caller-side wrapper content follows the telemetry.dev SDK `captureInput` / `captureOutput`
settings. The SDK mask and truncation options apply to captured client-wrapper inputs, outputs,
and logs.

## Limitations

- Pass options to exactly one entry point in a process. Initialization is one-shot so duplicate entry
  points do not replace the first configuration.
- `ClientSession.stream()` is not spanned.
- The client wrapper exports a total cost only after an error-free terminal event. Each completed step must report a finite, non-negative `usage.costUsd` value, including zero. Otherwise, the wrapper omits the total but still records token usage. Model information is also necessary for server-side pricing.
- When caller and server spans share a trace and matching Eve turn metadata, the caller cost replaces the server's main model-call costs. Tool and subagent costs stay separate. Without that match, the two reported totals stay unchanged.
- The client wrapper ends its `invoke_agent` span after stream use or an abort signal. Without these actions, the span stays open.
- Eve 0.50 gives `telemetryDevOtelIntegration()` no tracer-provider or parent-context hook. Thus, Eve provider spans cannot join the deterministic session trace.
- Inline subagent child streams (`subagent.event`) are not fanned out in full; the hook logs
  `subagent.started`, `subagent.called`, `subagent.completed`, and child failure events.
- Eve workflow `$eve.*` run tags are a Vercel-dashboard surface and are not visible to OpenTelemetry,
  so this package does not capture them.
