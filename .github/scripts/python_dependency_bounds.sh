#!/usr/bin/env bash
# Runs each Python package's tests against the lowest and the highest third-party dependency
# versions its declared ranges allow, so a range that admits a version the code cannot run on
# fails here instead of on a user's install. The local telemetry-dev checkout is always used.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ "$#" -eq 0 ]; then
  set -- sdks/python sdks/python-anthropic sdks/python-bedrock sdks/python-google-genai \
    sdks/python-litellm sdks/python-openai sdks/python-openrouter
fi

for package in "$@"; do
  for bound in lowest highest; do
    if [ "$bound" = lowest ]; then
      resolution=lowest-direct
      python=3.10
    else
      resolution=highest
      python=3.13
    fi
    venv="$tmp/$(basename "$package")-$bound"
    echo "::group::$package ($bound, Python $python)"
    uv venv --quiet --python "$python" "$venv"
    requirements=(-e "$root/sdks/python")
    if [ "$package" != sdks/python ]; then
      requirements+=(-e "$root/$package")
    fi
    uv pip install --quiet --python "$venv" --resolution "$resolution" \
      "${requirements[@]}" --group "$root/$package/pyproject.toml:dev"
    uv pip list --python "$venv"
    (cd "$root/$package" && "$venv/bin/python" -m pytest -q -p no:cacheprovider)
    echo "::endgroup::"
  done
done
