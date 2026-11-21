# @telemetry-dev/openai-agents

OpenAI Agents SDK integration for [telemetry.dev](https://telemetry.dev). It sends each agent run
to telemetry.dev: the agent and its turns, each model call with its token usage, each tool call, and
each command that a sandbox agent ran, with its exit code.

How runs group into traces:

- A run is one trace by default.
- A run that starts inside an active OpenTelemetry span, such as an HTTP request span, joins that
  span's trace.
- Runs that share a `groupId` share a session: the processor sends the `groupId` as the session ID
  (`gen_ai.conversation.id`). Runs inside the same
  [`propagateAttributes({ sessionId })`](https://telemetry.dev/docs/sdk/typescript#add-user-session-and-metadata)
  session (from `@telemetry-dev/sdk`) do too; a `groupId` takes precedence. Outside an active span,
  the runs of a session land in
  [one trace per session](https://telemetry.dev/docs/sdk/typescript#one-trace-per-session), each run
  as a root span.
- With `telemetryDevProcessor({ sessionMode: "process" })`, runs without a session of their own
  share the process's
  [generated session](https://telemetry.dev/docs/sdk/typescript#process-sessions) in the same way.
  Use this in a CLI or script that holds one conversation per process, not in a long-running server.

## Install

```sh
npm install @telemetry-dev/openai-agents @openai/agents
```

Requires `@openai/agents >=0.22.0 <1` and Node.js 22 or newer, the Agents SDK's minimum.

## Environment

| Variable                    | Required | Default                     | Notes                                                                   |
| --------------------------- | -------- | --------------------------- | ----------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY`     | yes      | —                           | Ingest key (`td_live_…`). No key ⇒ the integration is a complete no-op. |
| `TELEMETRY_DEV_BASE_URL`    | no       | `https://api.telemetry.dev` | Trailing slashes are stripped.                                          |
| `TELEMETRY_DEV_ENVIRONMENT` | no       | `production`                | Environment label on every trace.                                       |
| `OTEL_SERVICE_NAME`         | no       | SDK default                 | Service name on every trace.                                            |

All four are also settable through the processor's options.

## Usage

Register the processor once, before the first run:

```ts
import { addTraceProcessor, Agent, run } from "@openai/agents";
import { telemetryDevProcessor } from "@telemetry-dev/openai-agents";

addTraceProcessor(telemetryDevProcessor({ serviceName: "support-agent" }));

const agent = new Agent({ name: "Assistant", instructions: "Answer briefly." });
await run(agent, "What is OpenTelemetry?");
```

`addTraceProcessor` keeps the SDK's default exporter, so traces also reach the OpenAI Traces
dashboard. To send them only to telemetry.dev, replace the default exporter:

```ts
import { setTraceProcessors } from "@openai/agents";

setTraceProcessors([telemetryDevProcessor({ serviceName: "support-agent" })]);
```

The processor takes the options of `init` from `@telemetry-dev/sdk`, such as `captureInput`,
`captureOutput`, and `mask`. Pass SDK options to the processor and don't also call `init()`
yourself: the first processor created in a process initializes the SDK and replaces a client that
`init()` made earlier. The options of later processors are ignored.

The root span of a trace ends when the Agents trace ends. A run that throws before the SDK ends its
trace leaves it open, so the root also ends when the run's task or agent span ends with the error.
`forceFlush` and `shutdown` of the trace provider end the root of every trace that has no open span,
at the end of its last span. This covers traces that threw outside any span, such as a run that
failed before its first span. The flush covers every trace in the process, so it can also end the
root of another request's trace between two of its runs; that trace's later spans still nest under
the root. Spans that end after their root, such as those of a slower processor registered first,
nest under it while the processor holds it, which is for the 10,000 most recently active traces.
Call `getGlobalTraceProvider().forceFlush()` at the end of a request in short-lived runtimes.

The Agents SDK shuts the trace provider down on `SIGTERM`, `SIGINT`, and unhandled rejections. The
processor then ends every open span so that nothing is lost if the process exits. A server that keeps
running to finish its requests still gets their later spans in the same traces, nested under the
trace's root, but a model call that was in progress at that moment is sent without its token usage.

## Sandbox agents

Sandbox runs need no extra setup. Each `sandbox.*` span of the Agents SDK, such as
`sandbox.create_session` and `sandbox.exec`, becomes a span with `sandbox.operation`. The spans that
create or resume a session also get `sandbox.backend` (for example `e2b` or `modal`). The
`exec_command` and `write_stdin` tool spans get `process.exit.code` once the command has exited. A
command that exits with a non-zero code is not marked as an error, because many commands, such as
`grep`, exit non-zero without failing.

## Tool call ids

The Agents SDK puts no call id on its tool spans, so the processor takes it from the model call
that asked for the tool: each tool span gets `gen_ai.tool.call.id` from the matching call in the
output of the model call before it, matched by tool name and arguments. This needs the model's
output, which the Agents SDK records only with `traceIncludeSensitiveData` on.

## Content and privacy

- Model inputs and outputs, tool arguments and results, and the command and working directory of
  `sandbox.exec` (`process.command_line`, `process.working_directory`) are content. They are sent
  only when `captureInput` or `captureOutput` is on, and go through `mask`.
- Keep the Agents SDK's `traceIncludeSensitiveData` on, which is its default. With it off, the SDK
  leaves model, token usage, and tool output off its spans, so telemetry.dev cannot price model
  calls or read exit codes. Use `captureInput` and `captureOutput` to keep content out of
  telemetry.dev instead. With `addTraceProcessor`, the OpenAI exporter still receives content; use
  `setTraceProcessors` to stop that.
- With capture off, telemetry.dev still receives span names and timing, model names, token usage,
  agent, tool, handoff, and guardrail names, tool call ids, the MCP server of a tool call, exit codes, sandbox
  backends, error types, the session id (such as the trace's `groupId`, a
  `propagateAttributes()` session, or the process session), the user id and metadata of
  `propagateAttributes()`, and the string, number, and boolean values of trace metadata and of the
  data of custom spans. Metadata does not go through `mask`.

## Double counting

Each model call already gets a `chat` span from this processor. Do not also instrument the OpenAI
client that the Agents SDK uses with `@telemetry-dev/openai`: neither pass a client wrapped with
`wrapOpenAI` to `setDefaultOpenAIClient`, nor call `instrumentOpenAI()`, which instruments every
client in the process. Each model call would then be counted twice.
