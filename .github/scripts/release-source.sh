#!/usr/bin/env bash
set -euo pipefail

: "${RELEASE_TAG:?RELEASE_TAG is required}"
: "${GITHUB_EVENT_NAME:?GITHUB_EVENT_NAME is required}"
: "${GITHUB_REF:?GITHUB_REF is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"

if ! git check-ref-format "refs/tags/$RELEASE_TAG"; then
  echo "Invalid release tag" >&2
  exit 1
fi
if [[ "$GITHUB_EVENT_NAME" != "release" && "$GITHUB_EVENT_NAME" != "workflow_dispatch" ]]; then
  echo "Unsupported release event" >&2
  exit 1
fi

tag_sha=$(git rev-parse --verify "refs/tags/$RELEASE_TAG^{commit}")
head_sha=$(git rev-parse HEAD)

if [[ "$GITHUB_EVENT_NAME" == "workflow_dispatch" ]]; then
  if [[ "$GITHUB_REF" != "refs/heads/main" ]]; then
    echo "Release retries must use the workflow from main" >&2
    exit 1
  fi
  if [[ "$head_sha" != "$GITHUB_SHA" ]]; then
    echo "Retry checkout and event commits do not match" >&2
    exit 1
  fi
  if ! git merge-base --is-ancestor "$tag_sha" "$head_sha"; then
    echo "Release tag is not an ancestor of the retry commit" >&2
    exit 1
  fi
  release_paths=()
  while IFS= read -r path; do
    release_paths+=(":(literal)$path")
  done < <(jq -er '.packages | keys[]' release-please-config.json)
  release_inputs=(
    .release-please-manifest.json
    pnpm-lock.yaml
    pnpm-workspace.yaml
    release-please-config.json
    scripts
    tsconfig.json
    vite.config.ts
  )
  if ! git diff --quiet "$tag_sha" "$head_sha" -- "${release_paths[@]}" "${release_inputs[@]}"; then
    echo "Retry commit changes release source files" >&2
    exit 1
  fi
  tag_package=$(git show "$tag_sha:package.json" | jq --sort-keys 'del(.packageManager)')
  head_package=$(jq --sort-keys 'del(.packageManager)' package.json)
  if [[ "$tag_package" != "$head_package" ]]; then
    echo "Retry commit changes package.json outside packageManager" >&2
    exit 1
  fi
  printf '%s\n' "$head_sha"
  exit 0
fi

if [[ "$GITHUB_REF" != "refs/tags/$RELEASE_TAG" ]]; then
  echo "Package publication must run from a release tag" >&2
  exit 1
fi
if [[ "$tag_sha" != "$GITHUB_SHA" || "$head_sha" != "$GITHUB_SHA" ]]; then
  echo "Release tag, event, and checkout commits do not match" >&2
  exit 1
fi
printf '%s\n' "$tag_sha"
