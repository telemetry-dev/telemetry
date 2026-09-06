# @telemetry-dev/pi

Telemetry integration for the [pi coding agent](https://github.com/earendil-works/pi) and
[telemetry.dev](https://telemetry.dev). It records pi agent loops, model calls, tool executions, and
lifecycle events without changing pi's behavior.

## Install

Add the package to `~/.pi/agent/settings.json`:

```json
{
  "packages": ["npm:@telemetry-dev/pi"]
}
```

Pi reads the package's `pi.extensions` manifest and loads `@telemetry-dev/pi/register`
automatically.

Alternatively, install `@telemetry-dev/pi` where pi can resolve it and create
`~/.pi/agent/extensions/telemetry-dev.ts`:

```ts
import { telemetryDevExtension } from "@telemetry-dev/pi";
export default telemetryDevExtension();
```

The integration targets `@earendil-works/pi-coding-agent` 0.82.1. The pi package is an optional peer
dependency because pi bundles its extension API and provides it to installed extensions.

## Environment

| Variable                    | Required | Default                        | Notes                                                                   |
| --------------------------- | -------- | ------------------------------ | ----------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                              | Ingest key (`td_live_…`). No key ⇒ the integration is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://ingest.telemetry.dev` | Trailing slashes are stripped.                                          |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                   | Environment label on every trace/log.                                   |
| `OTEL_SERVICE_NAME`         | no       | `pi`                           | OpenTelemetry service name.                                             |

All four are also settable through SDK options when loading the extension directly:

```ts
import { telemetryDevExtension } from "@telemetry-dev/pi";

export default telemetryDevExtension({
  agentName: "pair-programmer",
  captureInput: false,
  captureOutput: true,
});
```

Initialization is one-shot; the first extension factory initialized in a process supplies the SDK
options.

## Trace shape

A typical session produces this hierarchy:

```text
session
└── invoke_agent
    ├── chat {model}                  (assistant message)
    │   ├── execute_tool {toolName}   (tool calls from that message)
    │   └── execute_tool {toolName}
    └── chat {model}                  (final assistant message)
```

- one `session` wrapper span for the full pi session;
- one `invoke_agent` span for each prompt, with the prompt and the final assistant content;
- one nested `chat {model}` span for each completed assistant message, with the provider, models,
  response id, finish reason, usage, pi cost, provider request, and all returned content blocks;
- one `execute_tool {toolName}` span for each tool execution, under the `chat` span that issued the
  call. If no message matches the tool call id, the span is under the `invoke_agent` span. The span
  includes the tool call id, arguments, result, and error state;
- lifecycle logs for sessions, turns, compaction, and model changes.

Each span and log reads `ctx.sessionManager.getSessionId()` when its event occurs. The span or log
records the value as `gen_ai.conversation.id`. This keeps `/new`, `/resume`, `/fork`, and `/reload`
events with the new session. Logs also have `gen_ai.agent.name` and the pi event type as `eventName`.

`agent_end` records the last run result and starts an asynchronous flush. `agent_settled` closes
unfinished tool spans and the prompt span. `session_shutdown` closes an unfinished prompt and the
session span, then waits for the final flush. It does not shut down the SDK during a pi session change.

## Content capture

The prompt includes its image blocks. Chat input records the provider-formatted request from
`before_provider_request`, after all `context` handlers and pi's model-message conversion.
This conversion removes `!!` shell output and custom-message `details` from the model input.
System instructions come from the effective `ctx.getSystemPrompt()` at the request boundary, after all
`before_agent_start` handlers. Chat output contains all returned content blocks. The integration also
records tool arguments and tool results.

These values obey the telemetry.dev SDK `captureInput` and `captureOutput` settings, which default to
`true`. Use `mask` to remove values before capture. Use `maxAttributeLength` to limit serialized
attributes.

## Limitations

- Chat input uses each provider's request shape, not pi's raw session messages. Pi's public hooks do
  not expose the final wire payload. A later `before_provider_request` handler can replace the request
  after capture. Custom providers must call pi's `onPayload` callback for request and system
  instruction capture. Without that callback, those fields are absent. Prompt, output, and tool
  capture still work.
- Pi does not expose per-request TTFT or duration fields. Chat duration is measured from
  `message_start` to `message_end` using local wall-clock time, and TTFT is not recorded.
- `tool_call` and `tool_result` are intentionally not intercepted. In pi, an uncaught `tool_call`
  handler error blocks the user's tool, so this integration uses the fail-open
  `tool_execution_start` / `tool_execution_end` events instead.
- An assistant message with `stopReason: "aborted"` is recorded with finish reason `aborted`, not as an
  error. Only `stopReason: "error"` marks chat and agent spans as errors.
- `gen_ai.usage.cost` is pi's client-side USD estimate from its bundled model pricing, not a charge or
  server-calculated invoice value.
- Initialization is one-shot. Loading the package more than once does not replace the first
  configuration.
