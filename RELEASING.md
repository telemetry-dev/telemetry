# Releasing

This file is for maintainers. It describes how the SDK packages in this repository are versioned and published.

## Releases

The [`Release` workflow](.github/workflows/release.yml) uses Release Please to propose version bumps on `main`. Merging a release PR lets Release Please create component releases such as `sdk-v0.1.3` or `python-v0.2.3`. Each published release builds and publishes only its matching package. Build jobs have no publishing credentials; separate publish jobs use registry OIDC trusted publishing, not long-lived npm or PyPI tokens.

The manifest retains versions already published from the original repository. `bootstrap-sha` in `release-please-config.json` points to the initial SDK import so that import's `feat` commit does not trigger new versions. Do not create releases to republish these baseline versions.

## Publishing setup

Publishing from this repository is configured. These settings must stay in place, and a new package needs the same settings before its first release:

- The repository Actions secret `RELEASE_PLEASE_TOKEN`, authorized for `telemetry-dev/telemetry` with contents, issues, and pull-request write permissions. The workflow deliberately fails when this secret is missing. Do not substitute `GITHUB_TOKEN`: releases created with it do not trigger the downstream release workflow. GitHub does not let anyone read a secret's value back, so a replacement must come from the token's owner.
- Each npm package's GitHub Actions trusted publisher: owner `telemetry-dev`, repository `telemetry`, workflow filename `release.yml`, no environment name. npm publishing runs on GitHub-hosted runners and includes provenance for this public repository.
- Each PyPI project's GitHub Actions trusted publisher: owner `telemetry-dev`, repository `telemetry`, workflow filename `release.yml`, environment `pypi`. The repository's `pypi` environment permits branch `main` for retries and tags matching `python*-v*`; it has no required reviewers. These branch and tag restrictions are not a manual approval gate.
- Merge release PRs only after CI passes. No branch protection or ruleset on `main` enforces this.

## Validation and retries

Run the release safety suite locally with Bash, Git, jq, Python 3, tar, and OpenSSL; the registry commands in these tests are mocked and publish nothing:

```sh
bash .github/scripts/release_test.sh
shellcheck .github/scripts/*.sh
actionlint
```

To retry an existing published release, run the `Release` workflow from `main` and set `release_tag` to that release's full tag. This is a real publication attempt, not a dry run. The tag must be an ancestor of the selected `main` commit. Every configured SDK directory, shared build scripts, lockfiles, release configuration, and root build configuration must be unchanged from the tag; only `packageManager` may differ within root `package.json`. Workflow-only recovery changes are allowed. Source-changing fixes need a new release, not a retry of an old version.

Artifact names and versions are checked before upload. npm retries skip an existing version only when its SHA-512 integrity matches the local tarball, and publication waits for published `@telemetry-dev/*` dependencies and peer dependencies to satisfy their declared ranges. PyPI retries use `uv publish --check-url` to check existing distributions. An integrity mismatch must be investigated, not bypassed or overwritten.
