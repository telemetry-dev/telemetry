#!/usr/bin/env bash
set -euo pipefail

artifact=${1:?wheel path is required}
attempts=${RELEASE_DEPENDENCY_ATTEMPTS-12}
delay=${RELEASE_DEPENDENCY_DELAY_SECONDS-10}
dependency_error=

validate_retry_setting() {
  local name=$1
  local value=$2
  local minimum=$3
  local maximum=$4

  if [[ ! "$value" =~ ^(0|[1-9][0-9]{0,2})$ ]] ||
    ((value < minimum || value > maximum)); then
    echo "$name must be an integer from $minimum to $maximum" >&2
    exit 2
  fi
}

validate_retry_setting RELEASE_DEPENDENCY_ATTEMPTS "$attempts" 1 60
validate_retry_setting RELEASE_DEPENDENCY_DELAY_SECONDS "$delay" 0 60

for ((attempt = 1; attempt <= attempts; attempt += 1)); do
  if dependency_error=$(uv pip install --dry-run --system --no-sources --no-build "$artifact" 2>&1); then
    exit 0
  fi
  if [[ "$dependency_error" != *"was not found in the package registry"* &&
    "$dependency_error" != *"there is no version of "* ]]; then
    printf '%s\n' "$dependency_error" >&2
    exit 1
  fi
  [[ "$attempt" == "$attempts" ]] || sleep "$delay"
done

echo "Cannot release $artifact: published dependencies did not become available" >&2
printf '%s\n' "$dependency_error" >&2
exit 1
