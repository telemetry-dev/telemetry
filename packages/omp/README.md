# @telemetry-dev/omp

Telemetry integration for the [Oh My Pi (omp)](https://www.npmjs.com/package/@oh-my-pi/pi-coding-agent)
coding agent and [telemetry.dev](https://telemetry.dev). It records omp agent loops, model calls,
tool executions, and lifecycle events without changing omp's behavior.

Docs: https://telemetry.dev/docs/integrations/omp

## Install

```sh
omp install @telemetry-dev/omp
```

omp installs the package and loads the extension listed in its `omp.extensions` manifest
(`@telemetry-dev/omp/register`), which reads its configuration from the environment variables
below.

To set options in code, load the extension from a file instead. Create
`~/.omp/agent/extensions/telemetry-dev.ts` (global) or `.omp/extensions/telemetry-dev.ts`
(project):

```ts
import { telemetryDevExtension } from "@telemetry-dev/omp";
export default telemetryDevExtension();
```

Install `@telemetry-dev/omp` somewhere that file can resolve it, for example with
`npm install @telemetry-dev/omp` in `~/.omp/agent` for the global file. Use one route, not both: if
the package is installed with `omp install` and also loaded from an extension file, omp loads the
extension twice and every span is recorded twice.

Requires Oh My Pi (`@oh-my-pi/pi-coding-agent`) 17 or later; tested with 17.1.8. The omp package
is an optional peer dependency because omp bundles its extension API and provides it to loaded
extensions.

## Environment

| Variable                    | Required | Default                     | Notes                                                                   |
| --------------------------- | -------- | --------------------------- | ----------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                           | Ingest key (`td_live_…`). No key ⇒ the integration is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://api.telemetry.dev` | Trailing slashes are stripped.                                          |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                | Environment label on every trace/log.                                   |
| `OTEL_SERVICE_NAME`         | no       | `omp`                       | OpenTelemetry service name.                                             |

All four are also settable through SDK options when loading the extension directly:

Earlier versions sent data to `https://ingest.telemetry.dev` by default. That hostname reaches the
same service and stays supported: if a firewall allows only it, set `TELEMETRY_DEV_BASE_URL` to it.

```ts
import { telemetryDevExtension } from "@telemetry-dev/omp";

export default telemetryDevExtension({
  agentName: "pair-programmer",
  captureInput: false,
  captureOutput: true,
});
```

Initialization is one-shot; the first extension factory initialized in a process supplies the SDK
options.

## Errors

The integration does not write to the terminal, because omp draws its TUI there. It sends SDK and
extension errors to the omp log through `pi.logger`, with the message `telemetry.dev error`. The
default location of the omp log is `~/.omp/logs`.

A series of failed exports, for example during an ingest outage, is one entry. The next failure
after an export reaches the ingest is a new entry.

To receive the errors in your code, pass `onError` to `telemetryDevExtension()`. The callback
replaces the omp log. To also write SDK diagnostics to the console, pass `logLevel`, for example
`"warn"`.

## Trace shape

A typical prompt produces this hierarchy:

```text
invoke_agent
├── chat {model}                  (assistant message)
│   ├── execute_tool {toolName}   (tool calls issued by that message)
│   └── execute_tool {toolName}
└── chat {model}                  (final assistant message)
```

- one `invoke_agent` span for the full agent loop, including the prompt and final assistant text;
- one nested `chat {model}` span per completed assistant message, with provider, model, response id,
  finish reason, usage (including cache and reasoning tokens), TTFT, and provider-reported duration;
- one `execute_tool {toolName}` span per tool execution, nested under the `chat` span whose tool
  call issued it (tool call ids are matched against the assistant message's tool-call blocks;
  executions with no matching message fall back to the `invoke_agent` span), including the tool
  call id, arguments, result, and error state;
- lifecycle logs for sessions, turns, compaction, and auto-retry.

Every span and log reads `ctx.sessionManager.getSessionId()` when its event fires and records it as
`gen_ai.conversation.id`, so session switches, branches, and tree navigation attach telemetry to the
correct session. All prompt spans with the same nonempty session id and API key share one trace.
Each `invoke_agent` span is a sibling in start order. Logs also have `gen_ai.agent.name` and the omp
event type as `eventName`. Without a session id or API key, the SDK does not join the prompt spans.

`agent_end` closes any unfinished tool spans and starts an asynchronous flush. `session_shutdown`
closes an unfinished agent loop and awaits a final flush before the process exits.

Chat spans use omp's own message timestamps: start = `message.timestamp`, end = start +
`message.duration`, so span timing matches provider-reported request duration rather than local
event-dispatch time.

## Content capture

Prompt input, assistant text output, tool arguments, and tool results follow the telemetry.dev SDK
`captureInput` and `captureOutput` settings. Both default to `true`. Use `mask` to redact values
before capture and `maxAttributeLength` to bound serialized attributes.

A failed tool's result text is also used as the exception message, but only when `captureOutput` is
on and no `mask` is set. Otherwise the exception message is `tool <name> failed`, and the result text
stays in the tool output, which follows `captureOutput` and `mask`. A failed assistant message's
`errorMessage` is a provider diagnostic, not tool output: chat and agent spans use it as the exception
message regardless of `captureOutput` and `mask`.

Thinking and tool-call content blocks are not copied into assistant output; only text blocks are
joined. Usage metadata still includes reasoning-token counts when omp reports them.

## Limitations

- `tool_call` and `tool_result` are intentionally not intercepted. In omp, an uncaught `tool_call`
  handler error blocks the tool (fail-closed), so this integration uses the fail-open
  `tool_execution_start` / `tool_execution_end` events instead.
- An assistant message with `stopReason: "aborted"` is recorded with finish reason `aborted`, not as
  an error. Only `stopReason: "error"` marks chat and agent spans as errors.
- omp does not report client-side cost; telemetry.dev computes cost server-side from usage and
  pricing data.
- Subagents (task tool) run in separate processes with their own extension instances, so their
  telemetry appears as separate sessions.
- Initialization is one-shot. Loading the extension more than once does not replace the first
  configuration.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/omp/LICENSE).
