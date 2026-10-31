# @telemetry-dev/opencode

telemetry.dev integration for [opencode](https://opencode.ai). It records agent turns, model generations, tool executions, and session lifecycle events with OpenTelemetry GenAI semantic conventions.

Docs: https://telemetry.dev/docs/integrations/opencode

## Install

Requires opencode 1.15.11 or later.

### npm plugin config

Register the package in `opencode.json`, either the project's file or `~/.config/opencode/opencode.json` for all projects:

```json
{
  "plugin": ["@telemetry-dev/opencode"]
}
```

opencode installs npm plugins itself when it starts, so you do not need to install the package. To pass options, use a `[name, options]` tuple:

```json
{
  "plugin": [["@telemetry-dev/opencode", { "environment": "production" }]]
}
```

The tuple's options are merged over options passed to `telemetryDevPlugin()`. Keep `apiKey` out of configuration files; provide it through `TELEMETRY_DEV_API_KEY`.

### Project plugin file

Use a plugin file when you need an option that is a function, such as `mask`. The plugin file and the package dependency go in the same opencode config directory: `.opencode/` for one project, or `~/.config/opencode/` for all projects. Add the package to `package.json` in that directory:

```json
{
  "dependencies": {
    "@telemetry-dev/opencode": "^0.2.2"
  }
}
```

Then create `plugins/telemetry-dev.ts` in the same directory, for example `.opencode/plugins/telemetry-dev.ts`:

```ts
import { telemetryDevPlugin } from "@telemetry-dev/opencode";

export const TelemetryDev = telemetryDevPlugin();
```

opencode loads every file in the plugins directory automatically.

Use one route, not both. If the package is in `opencode.json` and also loaded from a plugin file, opencode loads it twice and every span is recorded twice.

## Environment

| Variable                    | Required | Default                     | Description                                               |
| --------------------------- | -------- | --------------------------- | --------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | Yes      | —                           | telemetry.dev API key. Without it, telemetry is disabled. |
| `TELEMETRY_DEV_BASE_URL`    | No       | `https://api.telemetry.dev` | OTLP ingest base URL.                                     |
| `TELEMETRY_DEV_ENVIRONMENT` | No       | `production`                | Deployment environment attached to telemetry.             |
| `OTEL_SERVICE_NAME`         | No       | `opencode`                  | OpenTelemetry service name.                               |

Earlier versions sent data to `https://ingest.telemetry.dev` by default. That hostname reaches the
same service and stays supported: if a firewall allows only it, set `TELEMETRY_DEV_BASE_URL` to it.

All telemetry SDK options can also be passed to `telemetryDevPlugin`. The `agentName` integration option controls `gen_ai.agent.name` and defaults to `opencode`.

## Errors

The integration does not write to the console. opencode 1.x runs plugins in the process that draws its TUI, and console output from a plugin shows over the TUI. The integration sends SDK and plugin errors to the opencode log through `client.app.log`, with the service name `telemetry-dev`. The default location of the opencode log is `~/.local/share/opencode/log`.

A series of failed exports, for example during an ingest outage, is one entry. The next failure after an export reaches the ingest is a new entry.

To receive the errors in your code, pass `onError` to `telemetryDevPlugin()` in a plugin file. The callback replaces the opencode log. To also write SDK diagnostics to the console, set `logLevel`, for example `"warn"`.

## Trace shape

Each opencode turn produces this hierarchy, correlated by `gen_ai.conversation.id`:

```text
invoke_agent                        (session turn)
├── chat {model}                    (assistant message)
│   ├── execute_tool {toolName}
│   └── execute_tool task
│       └── invoke_agent            (subagent session, via session.created parentID)
│           └── chat {model}
└── chat {model}
```

- `invoke_agent` starts at `chat.message` and ends when the session goes idle (a `session.status` event with type `idle`, or the older `session.idle` event) or when opencode disposes the plugin. A `session.error` does not end it, because opencode can compact the context and continue the same prompt: if a compaction follows, the error is dropped; otherwise it is applied when the session goes idle.
- Assistant `message.updated` events open a `chat` span at first sight (start = `time.created`) and close it at `time.completed` with provider, model, finish reason, cost, and token usage.
- Tool callbacks become `execute_tool` spans nested under the session's open `chat` span (falling back to the `invoke_agent` span). Tool failures are reconstructed from `message.part.updated` events.
- Subagent (task) sessions run in-process; their `invoke_agent` spans nest under the parent session's open span using `session.created`'s `parentID`.
- Session creation, compaction, and errors are emitted as logs with their raw opencode event type as `eventName`.

## Content capture

Prompt text and tool arguments/results are captured by default. Configure `captureInput`, `captureOutput`, or `mask` through `telemetryDevPlugin` to disable or redact content before export:

```ts
export const TelemetryDev = telemetryDevPlugin({
  captureInput: false,
  captureOutput: false,
});
```

The SDK records an exception message only when both `captureInput` and `captureOutput` are on, and passes it through `mask`. A failed tool's error text is what opencode returns to the model, so it counts as tool output: it is the exception message only when no `mask` is set. With a `mask`, the exception message is `tool <name> failed`. Failed tool spans have no output attribute, so with a `mask` set the error text is not exported at all.

`session.error` text follows the same rule, because opencode publishes a failed prompt attachment's Read error there. Its exception message on the agent, chat, and pending tool spans becomes `session error` when a `mask` is set; the span keeps the host's error name as its type. The `session.error` log omits `opencode.error.message` when `captureOutput` is off and passes it through `mask` when one is set. An assistant message's `error` is a provider diagnostic: the chat span uses it as the exception message also when a `mask` is set, and the `mask` receives it.

Assistant output text is not captured because opencode's completed assistant message event exposes accumulated metadata and usage, not a final text payload. Prompt input remains available on the parent `invoke_agent` span when input capture is enabled.

## Limitations

- Assistant output text is not captured; only prompt input is recorded for the agent turn.
- `dispose` is not reached when `opencode serve` receives `SIGTERM`, so the last telemetry batch may be lost in that shutdown path. `session.idle` still triggers a non-blocking flush after each turn.
- Per-step token detail is available only as accumulated totals on completed assistant messages.
- Tool errors do not reach `tool.execute.after`; error spans are reconstructed from `message.part.updated`.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/opencode/LICENSE).
