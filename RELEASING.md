# Releasing

This file is for maintainers. It describes how the SDK packages in this repository are versioned and published.

## Releases

The [`Release` workflow](.github/workflows/release.yml) uses Release Please to propose version bumps on `main`. Merging a release PR lets Release Please create component releases such as `sdk-v0.1.3` or `python-v0.2.3`. The same workflow run then builds and publishes every package released by that merge, in the order listed in `release-please-config.json`, so core packages go out before the integrations that depend on them. List a new package after every package it depends on; the release tests check this order. Running them in one job per registry also means integrations waiting for a core package do not hold the runners that core publication needs. Build jobs have no publishing credentials; separate publish jobs use registry OIDC trusted publishing, not long-lived npm or PyPI tokens.

The manifest retains versions already published from the original repository. `bootstrap-sha` in `release-please-config.json` points to the initial SDK import so that import's `feat` commit does not trigger new versions. Do not create releases to republish these baseline versions.

## Publishing setup

Publishing from this repository is configured. These settings must stay in place, and a new package needs the same settings before its first release:

- The repository Actions secret `RELEASE_PLEASE_TOKEN`, authorized for `telemetry-dev/telemetry` with contents, issues, and pull-request write permissions. The workflow deliberately fails when this secret is missing. Do not substitute `GITHUB_TOKEN`: release PRs opened with it do not trigger CI. GitHub does not let anyone read a secret's value back, so a replacement must come from the token's owner.
- Each npm package's GitHub Actions trusted publisher: owner `telemetry-dev`, repository `telemetry`, workflow filename `release.yml`, no environment name. npm publishing runs on GitHub-hosted runners and includes provenance for this public repository.
- Each PyPI project's GitHub Actions trusted publisher: owner `telemetry-dev`, repository `telemetry`, workflow filename `release.yml`, environment `pypi`. The repository's `pypi` environment permits branch `main`, which both release runs and retries use, and has no required reviewers. Its older rule for tags matching `python*-v*` is no longer used. These restrictions are not a manual approval gate.
- Merge release PRs only after CI passes. No branch protection or ruleset on `main` enforces this.

## Validation and retries

Run the release safety suite locally with Bash, Git, jq, Python 3, tar, and OpenSSL; the registry commands in these tests are mocked and publish nothing:

```sh
bash .github/scripts/release_test.sh
shellcheck .github/scripts/*.sh
actionlint
```

Each registry's publish job stops at the first package that fails, so the packages listed after it are not published. If a release did not fully publish, recover from the run for the release PR's merge commit, even if a later run reported the error. If that run's jobs failed, use **Re-run failed jobs**, which reuses build artifacts the first attempt already uploaded and skips versions already published. If that run succeeded without publishing the release or was cancelled, use **Re-run all jobs**, which rebuilds from the same commit. Either rerun also picks up releases that Release Please created before it failed; if a rerun fails with a duplicate release error, run it once more. Use a per-tag retry only when the workflow itself needs a fix, start it only after the release run has finished, and retry core packages such as `otel`, `sdk`, and `python` before the integrations that depend on them.

To retry an existing published release, run the `Release` workflow from `main` and set `release_tag` to that release's full tag. This is a real publication attempt, not a dry run. The tag must be an ancestor of the selected `main` commit. Every configured SDK directory, shared build scripts, lockfiles, release configuration, and root build configuration must be unchanged from the tag; only `packageManager` may differ within root `package.json`. Workflow-only recovery changes are allowed. Source-changing fixes need a new release, not a retry of an old version.

Artifact names and versions are checked before upload. npm retries skip an existing version only when its SHA-512 integrity matches the local tarball, and publication waits about 10 minutes for published `@telemetry-dev/*` dependencies and peer dependencies to satisfy their declared ranges. Python publication likewise waits about 10 minutes for each wheel's dependencies to resolve on PyPI. PyPI retries use `uv publish --check-url` to check existing distributions. An integrity mismatch must be investigated, not bypassed or overwritten.

## Experiment action

The experiment action in `actions/experiment` has no package. Release Please gives it releases such as `experiment-action-v1.0.0`. The `tag-experiment-action` job of the `Release` workflow then moves the major tag, `experiment-action-v1`, to the release commit. The README of the action and the experiment page of telemetry.dev tell users to use the major tag. The job moves a major tag only forward on `main`, so a rerun of an earlier release does not move it back.

A change that can break a workflow, for example a removed input or a changed output, must have a `feat!:` or `fix!:` commit. Release Please then gives the action a new major version, and `experiment-action-v1` stays at the last 1.x release. A `release_tag` retry does not apply to the action. If the job fails, re-run the failed job of the release run.

The job uses `GITHUB_TOKEN`, which cannot have the `workflows` permission. Thus GitHub rejects the tag when no branch head has the same `.github/workflows` files as the release commit. If a merge changed a workflow before the job ran, move the tag with your own credentials:

```sh
git fetch origin --tags
git push --force origin 'experiment-action-v1.0.1^{commit}:refs/tags/experiment-action-v1'
```
