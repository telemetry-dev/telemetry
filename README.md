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

CI verifies frozen dependency installs, checks, tests, and builds for both ecosystems, plus the release safety tests.

## Releases

The [`Release` workflow](.github/workflows/release.yml) uses Release Please to propose version bumps on `main`. Merging a release PR lets Release Please create component releases such as `sdk-v0.1.3` or `python-v0.2.3`. Each published release builds and publishes only its matching package. Build jobs have no publishing credentials; separate publish jobs use registry OIDC trusted publishing, not long-lived npm or PyPI tokens.

The manifest retains versions already published from the original repository. `bootstrap-sha` points to the initial SDK import so that import's `feat` commit does not trigger new versions. Do not create releases to republish these baseline versions: wait for subsequent releasable changes. No new version is required merely to transfer publishing to this repository.

### Required setup

The workflow is restored, but publishing is not fully configured until the following handoff is complete:

- Provide the repository Actions secret `RELEASE_PLEASE_TOKEN`, authorized for `telemetry-dev/sdks` with contents, issues, and pull-request write permissions. The original repository's secret cannot be read or copied from GitHub; its owner must supply or replace it. The workflow deliberately fails when this secret is missing. Do not substitute `GITHUB_TOKEN`: releases created with it do not trigger the downstream release workflow.
- Configure every npm package's GitHub Actions trusted publisher for owner `telemetry-dev`, repository `sdks`, and workflow filename `release.yml`, without an environment name. npm publishing runs on GitHub-hosted runners and includes provenance for this public repository.
- Configure every PyPI project's GitHub Actions trusted publisher for owner `telemetry-dev`, repository `sdks`, workflow filename `release.yml`, and environment `pypi`. The repository's `pypi` environment permits branch `main` for retries and tags matching `python*-v*`; it has no required reviewers. These branch/tag restrictions are not a manual approval gate.
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

## License

Each package is licensed under the MIT License included in its package directory.
