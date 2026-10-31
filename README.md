<p align="center">
  <a href="https://telemetry.dev">
    <img src="https://telemetry.dev/icon.svg" alt="telemetry.dev logo" width="64" height="64">
  </a>
</p>

<h1 align="center">telemetry.dev</h1>

<h3 align="center">LLM observability for AI applications and agents</h3>

---

<div align="center">

[npm](https://www.npmjs.com/package/@telemetry-dev/sdk) · [PyPI](https://pypi.org/project/telemetry-dev/) · [MIT License](LICENSE)

Trace model calls, tool steps, and agent runs. Find slow requests, debug errors, and monitor token usage and cost.

[Start tracing](https://telemetry.dev/signup) · [Documentation](https://telemetry.dev/docs) · [Integrations](https://telemetry.dev/docs/integrations)

</div>

[telemetry.dev](https://telemetry.dev) connects your AI application to a shared view of traces, logs, and metrics.
This repository contains the official TypeScript and Python SDKs and integrations, built on OpenTelemetry.
The ingest service and the dashboard are hosted by telemetry.dev and are not in this repository.

## Quick start

Make a project API key at [telemetry.dev](https://telemetry.dev/signup).
Set the API keys for telemetry.dev and your model provider:

```sh
export TELEMETRY_DEV_API_KEY="your-telemetry-dev-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### TypeScript

Use Node.js 22.18 or later, which runs TypeScript files directly. Install the SDK and the OpenAI integration:

```sh
npm install @telemetry-dev/sdk @telemetry-dev/openai openai
```

Save this as `trace.mts`:

```ts
import OpenAI from "openai";
import { init, shutdown } from "@telemetry-dev/sdk";
import { wrapOpenAI } from "@telemetry-dev/openai";

init({ serviceName: "my-ai-app" });
const openai = wrapOpenAI(new OpenAI());

try {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "What is OpenTelemetry?" }],
  });
  console.log(response.choices[0]?.message.content);
} finally {
  await shutdown();
}
```

Run the request:

```sh
node trace.mts
```

Open your project in [telemetry.dev](https://telemetry.dev/login) to inspect the trace.
The integration records the model call, reported token usage, duration, and errors.

<details>
<summary><strong>Python quick start</strong></summary>

Use Python 3.10 or newer. Install the OpenAI integration:

```sh
pip install telemetry-dev-openai
```

Use the same API keys as the TypeScript example. Save this as `trace.py`:

```py
import telemetry_dev
from openai import OpenAI
from telemetry_dev_openai import wrap_openai

telemetry_dev.init(service_name="my-ai-app")
client = wrap_openai(OpenAI())

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is OpenTelemetry?"}],
    )
    print(response.choices[0].message.content)
finally:
    telemetry_dev.shutdown()
```

Run the request:

```sh
python trace.py
```

</details>

The core SDKs and most integrations capture inputs and outputs by default.
To disable input and output capture, set `captureInput: false` and `captureOutput: false` in TypeScript `init()`.
For Python, use `capture_input=False` and `capture_output=False`.
Some integrations work differently: MCP captures no tool arguments or results unless you pass
`capturePayloads: true`, the AI SDK integration follows AI SDK's `recordInputs` and `recordOutputs`,
Cursor reads `captureInput` and `captureOutput` from its own config file, and TanStack AI has no
capture setting of its own and records chat messages, tool payloads, and error messages.
Disabling output capture omits message content; response IDs and models, finish reasons, usage,
timing, and errors may still be recorded when providers supply them. For streamed OpenAI chat,
usage requires provider support or the documented stream-usage option in the OpenAI integration.
Capture flags do not gate stop sequences, tool descriptions and definitions (`gen_ai.tool.description`, `gen_ai.tool.definitions`), or caller-supplied metadata or raw attributes; redact those separately when needed.
The core SDKs record exception messages and stack traces only when both input and output capture are on, and pass them through the mask.
Streamed output is kept within fixed size limits that each provider package's README lists, and
[Privacy](https://telemetry.dev/docs/platform/privacy) describes what is sent and how to redact it.

## Features

- **Agent traces.** Connect model calls, tool steps, and custom spans in one trace.
- **Usage and cost.** Inspect reported token usage, cost, latency, and errors in telemetry.dev.
- **Provider integrations.** Add tracing to your existing model client without a separate request API.
- **Session context.** Group activity by user and session, and pass trace context between services.
- **Capture controls.** Disable SDK input and output capture, or mask captured content before export.
- **Prompts.** Fetch the prompts that you manage in telemetry.dev, and record which version each model call used.
- **Experiments and scores.** Run your code on a dataset, check its scores in CI, and record scores, such as user feedback, on a trace, a span, or a session.
- **OpenTelemetry.** Send traces, logs, and metrics through standard OTLP/HTTP.

## Integrations

Each integration links to its installation instructions and API coverage. The link text is the npm or PyPI package name.

| Integration                  | TypeScript                                             | Python                                                   |
| ---------------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| Core SDK                     | [`@telemetry-dev/sdk`](packages/sdk)                   | [`telemetry-dev`](sdks/python)                           |
| Existing OpenTelemetry setup | [`@telemetry-dev/otel`](packages/otel)                 | [`telemetry-dev`](sdks/python) (`telemetry_dev.otel`)    |
| OpenAI                       | [`@telemetry-dev/openai`](packages/openai)             | [`telemetry-dev-openai`](sdks/python-openai)             |
| Anthropic                    | [`@telemetry-dev/anthropic`](packages/anthropic)       | [`telemetry-dev-anthropic`](sdks/python-anthropic)       |
| Amazon Bedrock               | [`@telemetry-dev/bedrock`](packages/bedrock)           | [`telemetry-dev-bedrock`](sdks/python-bedrock)           |
| Google GenAI                 | [`@telemetry-dev/google-genai`](packages/google-genai) | [`telemetry-dev-google-genai`](sdks/python-google-genai) |
| OpenRouter                   | [`@telemetry-dev/openrouter`](packages/openrouter)     | [`telemetry-dev-openrouter`](sdks/python-openrouter)     |
| Vercel AI SDK                | [`@telemetry-dev/ai-sdk`](packages/ai)                 |                                                          |
| TanStack AI                  | [`@telemetry-dev/tanstack-ai`](packages/tanstack-ai)   |                                                          |
| LiteLLM                      |                                                        | [`telemetry-dev-litellm`](sdks/python-litellm)           |
| MCP                          | [`@telemetry-dev/mcp`](packages/mcp)                   |                                                          |
| Eve                          | [`@telemetry-dev/eve`](packages/eve)                   |                                                          |
| Cursor                       | [`@telemetry-dev/cursor`](packages/cursor)             |                                                          |
| OpenCode                     | [`@telemetry-dev/opencode`](packages/opencode)         |                                                          |
| Oh My Pi                     | [`@telemetry-dev/omp`](packages/omp)                   |                                                          |
| Pi                           | [`@telemetry-dev/pi`](packages/pi)                     |                                                          |

Some integrations need no package from this repository, because the tool exports OpenTelemetry
itself or through a third-party library: Claude Code, Claude Platform, Claude SDK native spans,
Hermes Agent, OpenClaw, CrewAI, LangChain and LangGraph, and Pydantic AI. Their guides are in
[Integrations](https://telemetry.dev/docs/integrations).

Other languages can use an existing OTLP/HTTP exporter.
The [documentation](https://telemetry.dev/docs) describes endpoint configuration and authentication.

The Claude SDKs (`@anthropic-ai/sdk` 0.134.0 and later, Python `anthropic` 1.14.0 and later) emit OpenTelemetry spans natively.
To send these spans to telemetry.dev, refer to Native Claude SDK spans for [TypeScript](packages/otel/README.md#native-claude-sdk-spans) and [Python](sdks/python/README.md#native-claude-sdk-spans).
Do not use these spans together with the Anthropic integration, because each request then has two spans.
The Claude API can also send server-side spans through a Console destination. Refer to [Claude Platform](https://telemetry.dev/docs/integrations/claude-platform).

## Command line

[`@telemetry-dev/cli`](packages/cli) reads traces, sessions, logs, and usage from a terminal, a script, or an agent's shell. It uses the same read-only access as the telemetry.dev MCP server, so it works where an MCP client isn't available.

```sh
npx @telemetry-dev/cli login
```

## Development

Local development uses Node.js 24, pnpm 11.25.0, Vite+, uv, and Python 3.10 or newer.

```sh
pnpm install --frozen-lockfile
pnpm run check
pnpm run test
pnpm run build

pnpm run py:sync
pnpm run py:check
pnpm run py:test
pnpm run py:test:bounds
pnpm run py:build
```

`py:test:bounds` runs each Python package's tests against the lowest and the highest third-party dependency versions its declared ranges allow. It always uses the local `telemetry-dev` checkout, not the lowest published `telemetry-dev` a provider package accepts.

CI runs dependency, code, test, build, and release-safety checks for TypeScript and Python.
The TypeScript job also packs each npm package that has a release version and fails when a target of its `exports` or `bin` is not in the tarball.
The [SDK conformance guide](docs/sdk-conformance.md) defines the shared telemetry format and behavior.

Maintainers: [RELEASING.md](RELEASING.md) describes versioning, publishing, and release retries.

## License

This project uses the [MIT License](LICENSE).

## Links

- [Website](https://telemetry.dev)
- [Dashboard](https://telemetry.dev/login)
- [Documentation](https://telemetry.dev/docs)
- [Integrations](https://telemetry.dev/docs/integrations)
- [Pricing](https://telemetry.dev/pricing)
