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

[Start tracing](https://telemetry.dev/signup) · [Documentation](https://docs.telemetry.dev) · [Integrations](https://telemetry.dev/integrations)

</div>

[telemetry.dev](https://telemetry.dev) connects your AI application to a shared view of traces, logs, and metrics.
This repository contains the official TypeScript and Python SDKs and integrations, built on OpenTelemetry.

## Quick start

Make a project API key at [telemetry.dev](https://telemetry.dev/signup).
Set the API keys for telemetry.dev and your model provider:

```sh
export TELEMETRY_DEV_API_KEY="your-telemetry-dev-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### TypeScript

Use Node.js 24 or newer. Install the SDK and the OpenAI integration:

```sh
npm install @telemetry-dev/sdk @telemetry-dev/openai openai
```

Save this as `trace.ts`:

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
node trace.ts
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

The SDKs capture inputs and outputs by default.
To disable input and output capture, set `captureInput: false` and `captureOutput: false` in TypeScript `init()`.
For Python, use `capture_input=False` and `capture_output=False`.

## Features

- **Agent traces.** Connect model calls, tool steps, and custom spans in one trace.
- **Usage and cost.** Inspect reported token usage, cost, latency, and errors in telemetry.dev.
- **Provider integrations.** Add tracing to your existing model client without a separate request API.
- **Session context.** Group activity by user and session, and pass trace context between services.
- **Capture controls.** Disable SDK input and output capture, or mask captured content before export.
- **OpenTelemetry.** Send traces, logs, and metrics through standard OTLP/HTTP.

## Integrations

Each integration links to its installation instructions and API coverage.

| Integration | TypeScript | Python |
| --- | --- | --- |
| Core SDK | [SDK](packages/sdk) | [SDK](sdks/python) |
| Existing OpenTelemetry setup | [Span processor](packages/otel) | [Span processor](sdks/python) |
| OpenAI | [OpenAI](packages/openai) | [OpenAI](sdks/python-openai) |
| Anthropic | [Anthropic](packages/anthropic) | [Anthropic](sdks/python-anthropic) |
| Amazon Bedrock | [Bedrock](packages/bedrock) | [Bedrock](sdks/python-bedrock) |
| Google GenAI | [Google GenAI](packages/google-genai) | [Google GenAI](sdks/python-google-genai) |
| OpenRouter | [OpenRouter](packages/openrouter) | [OpenRouter](sdks/python-openrouter) |
| Vercel AI SDK | [AI SDK](packages/ai) | |
| TanStack AI | [TanStack AI](packages/tanstack-ai) | |
| LiteLLM | | [LiteLLM](sdks/python-litellm) |
| MCP | [MCP](packages/mcp) | |
| Eve | [Eve](packages/eve) | |
| Cursor | [Cursor](packages/cursor) | |
| OpenCode | [OpenCode](packages/opencode) | |
| Oh My Pi | [Oh My Pi](packages/omp) | |
| Pi | [Pi](packages/pi) | |

Other languages can use an existing OTLP/HTTP exporter.
The [documentation](https://docs.telemetry.dev) describes endpoint configuration and authentication.

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
pnpm run py:build
```

CI runs dependency, code, test, build, and release-safety checks for TypeScript and Python.
The [SDK conformance guide](docs/sdk-conformance.md) defines the shared telemetry format and behavior.

<details>
<summary><strong>Release maintenance</strong></summary>

## Releases

The [`Release` workflow](.github/workflows/release.yml) uses Release Please to propose version bumps on `main`. Merging a release PR lets Release Please create component releases such as `sdk-v0.1.3` or `python-v0.2.3`. Each published release builds and publishes only its matching package. Build jobs have no publishing credentials; separate publish jobs use registry OIDC trusted publishing, not long-lived npm or PyPI tokens.

The manifest retains versions already published from the original repository. `bootstrap-sha` points to the initial SDK import so that import's `feat` commit does not trigger new versions. Do not create releases to republish these baseline versions: wait for subsequent releasable changes. No new version is required merely to transfer publishing to this repository.

### Required setup

The workflow is restored, but publishing is not fully configured until the following handoff is complete:

- Provide the repository Actions secret `RELEASE_PLEASE_TOKEN`, authorized for `telemetry-dev/telemetry` with contents, issues, and pull-request write permissions. The original repository's secret cannot be read or copied from GitHub; its owner must supply or replace it. The workflow deliberately fails when this secret is missing. Do not substitute `GITHUB_TOKEN`: releases created with it do not trigger the downstream release workflow.
- Configure every npm package's GitHub Actions trusted publisher for owner `telemetry-dev`, repository `telemetry`, and workflow filename `release.yml`, without an environment name. npm publishing runs on GitHub-hosted runners and includes provenance for this public repository.
- Configure every PyPI project's GitHub Actions trusted publisher for owner `telemetry-dev`, repository `telemetry`, and workflow filename `release.yml`, and environment `pypi`. The repository's `pypi` environment permits branch `main` for retries and tags matching `python*-v*`; it has no required reviewers. These branch/tag restrictions are not a manual approval gate.
- Require successful CI before merging release PRs. Verify all trusted-publisher settings and the secret before expecting an end-to-end publish to succeed.

### Validation and retries

Run the release safety suite locally with Bash, Git, jq, Python 3, tar, and OpenSSL; the registry commands in these tests are mocked and publish nothing:

```sh
bash .github/scripts/release_test.sh
shellcheck .github/scripts/*.sh
actionlint
```

To retry an existing published release, run the `Release` workflow from `main` and set `release_tag` to that release's full tag. This is a real publication attempt, not a dry run. The tag must be an ancestor of the selected `main` commit. Every configured SDK directory, shared build scripts, lockfiles, release configuration, and root build configuration must be unchanged from the tag; only `packageManager` may differ within root `package.json`. Workflow-only recovery changes are allowed. Source-changing fixes need a new release, not a retry of an old version.

Artifact names and versions are checked before upload. npm retries skip an existing version only when its SHA-512 integrity matches the local tarball, and publication waits for published `@telemetry-dev/*` dependencies and peer dependencies to satisfy their declared ranges. PyPI retries use `uv publish --check-url` to check existing distributions. An integrity mismatch must be investigated, not bypassed or overwritten.

</details>

## License

This project uses the [MIT License](LICENSE).

## Links

- [Website](https://telemetry.dev)
- [Dashboard](https://telemetry.dev/login)
- [Documentation](https://docs.telemetry.dev)
- [Integrations](https://telemetry.dev/integrations)
- [Pricing](https://telemetry.dev/pricing)
