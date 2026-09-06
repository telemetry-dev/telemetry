# telemetry.dev SDKs

Open-source SDKs and integrations for sending OpenTelemetry-native GenAI traces, logs, and metrics to [telemetry.dev](https://telemetry.dev).

## Packages

The `packages/` workspace contains 15 TypeScript packages published in the `@telemetry-dev` npm scope: the core `sdk` and `otel` packages plus integrations for AI SDK, Anthropic, Amazon Bedrock, Cursor, Eve, Google GenAI, MCP, Oh My Pi, OpenAI, opencode, OpenRouter, Pi, and TanStack AI.

The `sdks/` directory contains seven Python distributions published on PyPI: `telemetry-dev`, `telemetry-dev-anthropic`, `telemetry-dev-bedrock`, `telemetry-dev-google-genai`, `telemetry-dev-litellm`, `telemetry-dev-openai`, and `telemetry-dev-openrouter`.

Package-specific installation and usage instructions are in each package directory. The shared cross-language telemetry emission contract is [`docs/sdk-conformance.md`](docs/sdk-conformance.md).

## Development

Node.js 24, pnpm 11.25.0, Vite+, uv, and Python 3.10 or newer are required.

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

CI verifies frozen dependency installs, checks, tests, and builds for both ecosystems. Publishing is intentionally disabled until a separate trusted-publisher handoff is completed; this repository contains no publish or release workflow.

## License

Each package is licensed under the MIT License included in its package directory.
