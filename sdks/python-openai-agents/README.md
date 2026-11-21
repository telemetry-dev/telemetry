# telemetry-dev-openai-agents

OpenAI Agents SDK integration for [telemetry.dev](https://telemetry.dev). It sends each agent run
to telemetry.dev: the agent and its turns, each model call with its token usage, each tool call
with its call id, and each command that a sandbox agent ran, with its exit code.

Each run is one trace. Runs that share a `group_id` share a session: the processor sends the
`group_id` as the session ID (`gen_ai.conversation.id`).

## Install

```sh
pip install telemetry-dev-openai-agents
```

Requires Python 3.10 or later and `openai-agents` 0.25 or later.

## Usage

Initialize the core SDK first. `init()` reads your project API key from the
`TELEMETRY_DEV_API_KEY` environment variable, so set it before you start the app. Then register the
processor once, before the first run:

```py
import telemetry_dev
from agents import Agent, Runner, add_trace_processor
from telemetry_dev_openai_agents import TelemetryDevProcessor

telemetry_dev.init(service_name="support-agent")
add_trace_processor(TelemetryDevProcessor())

agent = Agent(name="Assistant", instructions="Answer briefly.")
Runner.run_sync(agent, "What is OpenTelemetry?")
```

`add_trace_processor` keeps the SDK's default exporter, so traces also reach the OpenAI Traces
dashboard. To send them only to telemetry.dev, replace the default exporter:

```py
from agents import set_trace_processors

set_trace_processors([TelemetryDevProcessor()])
```

The root span of a trace ends when the Agents trace ends, which the Agents SDK does even when a run
raises. `force_flush` and `shutdown` of the trace provider also end the root of every trace that has
no open span, at the end of its last span, such as a trace that your code started and never
finished. Spans that end after their root nest under it while the processor holds it, which is for
the 10,000 most recently active traces.

## Sandbox agents

Sandbox runs need no extra setup. Each `sandbox.*` span of the Agents SDK, such as
`sandbox.create_session` and `sandbox.exec`, becomes a span with `sandbox.operation` and, when the
Agents SDK records it, `sandbox.backend` (for example `unix_local` or `docker`). `sandbox.exec`
spans and the `exec_command` and `write_stdin` tool spans get `process.exit.code` once the command
has exited. A command that exits with a non-zero code is not marked as an error, because many
commands, such as `grep`, exit non-zero without failing.

## Tool call ids

The Agents SDK puts no call id on its tool spans, so the processor takes it from the model call
that asked for the tool: each tool span gets `gen_ai.tool.call.id` from the matching call in the
output of the model call before it, matched by tool name and arguments. This needs the model's
output, which the Agents SDK records only with `trace_include_sensitive_data` on.

## Content and privacy

- Model inputs and outputs and tool arguments and results are content. They are sent only when
  `capture_input` or `capture_output` of `telemetry_dev.init()` is on, and go through `mask`.
- Keep the Agents SDK's `trace_include_sensitive_data` on, which is its default. With it off, the
  SDK leaves tool output and model outputs off its spans, and the Responses API model leaves off
  the model name, so telemetry.dev cannot price Responses API calls or read exit codes and tool
  call ids. Use `capture_input` and `capture_output` to keep content out of
  telemetry.dev instead. With `add_trace_processor`, the OpenAI exporter still receives content;
  use `set_trace_processors` to stop that.
- With capture off, telemetry.dev still receives span names and timing, model names, token usage,
  agent, tool, handoff, and guardrail names, tool call ids, the MCP server of a tool call, exit
  codes, sandbox backends, error types, the session id (the trace's `group_id`), and the string,
  number, and boolean values of trace metadata and of the data of custom spans other than
  `sandbox.*` spans. Metadata does not go through `mask`.

## Double counting

Each model call already gets a `chat` span from this processor. Do not also instrument the OpenAI
client that the Agents SDK uses with `telemetry-dev-openai`: neither pass a client wrapped with
`wrap_openai` to `set_default_openai_client`, nor call `instrument_openai()`, which instruments
every client in the process. Each model call would then be counted twice.
