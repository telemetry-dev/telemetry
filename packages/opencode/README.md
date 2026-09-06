# @telemetry-dev/opencode

Telemetry.dev integration for [opencode](https://opencode.ai). It records agent turns, model generations, tool executions, and session lifecycle events with OpenTelemetry GenAI semantic conventions.

## Install

```sh
npm install @telemetry-dev/opencode
```

### Project plugin file

Create `.opencode/plugin/telemetry-dev.ts`:

```ts
import { telemetryDevPlugin } from "@telemetry-dev/opencode";

export const TelemetryDev = telemetryDevPlugin();
```

opencode discovers project plugin files automatically.

### npm plugin config

Alternatively, register the package in `opencode.json`:

```json
{
  "plugin": [["@telemetry-dev/opencode", { "environment": "production" }]]
}
```

The tuple's options are merged over options passed to `telemetryDevPlugin()`. Keep `apiKey` out of configuration files; provide it through `TELEMETRY_DEV_API_KEY`.

## Environment

| Variable                    | Required | Default                        | Description                                               |
| --------------------------- | -------- | ------------------------------ | --------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | Yes      | —                              | telemetry.dev API key. Without it, telemetry is disabled. |
| `TELEMETRY_DEV_BASE_URL`    | No       | `https://ingest.telemetry.dev` | OTLP ingest base URL.                                     |
| `TELEMETRY_DEV_ENVIRONMENT` | No       | `production`                   | Deployment environment attached to telemetry.             |
| `OTEL_SERVICE_NAME`         | No       | `opencode`                     | OpenTelemetry service name.                               |

All telemetry SDK options can also be passed to `telemetryDevPlugin`. The `agentName` integration option controls `gen_ai.agent.name` and defaults to `opencode`.

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

- `invoke_agent` starts at `chat.message` and ends when opencode emits `session.idle` or `session.error`.
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

Assistant output text is not captured because opencode's completed assistant message event exposes accumulated metadata and usage, not a final text payload. Prompt input remains available on the parent `invoke_agent` span when input capture is enabled.

## Limitations

- Assistant output text is not captured; only prompt input is recorded for the agent turn.
- `dispose` is not reached when `opencode serve` receives `SIGTERM`, so the last telemetry batch may be lost in that shutdown path. `session.idle` still triggers a non-blocking flush after each turn.
- Per-step token detail is available only as accumulated totals on completed assistant messages.
- Tool errors do not reach `tool.execute.after`; error spans are reconstructed from `message.part.updated`.
