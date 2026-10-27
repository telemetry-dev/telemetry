#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
source_script="$root/.github/scripts/release-source.sh"
artifact_script="$root/.github/scripts/release-artifact.sh"
dependency_script="$root/.github/scripts/verify-python-release-dependencies.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
repo="$tmp/repo"

git init --quiet --initial-branch=main "$repo"
git -C "$repo" config user.name "Release Test"
git -C "$repo" config user.email "release-test@example.invalid"
mkdir -p "$repo/.github/scripts" "$repo/.github/workflows" "$repo/scripts" "$repo/packages/mcp" "$repo/sdks/python"
printf '{"name":"repo","packageManager":"pnpm@11.12.0"}\n' > "$repo/package.json"
printf '{"packages/mcp":"0.1.1","sdks/python":"0.2.2"}\n' > "$repo/.release-please-manifest.json"
printf '{"packages":{"packages/mcp":{"component":"mcp","release-type":"node"},"sdks/python":{"component":"python","release-type":"python","package-name":"telemetry-dev"}}}\n' > "$repo/release-please-config.json"
printf '{"name":"@telemetry-dev/mcp","version":"0.1.1"}\n' > "$repo/packages/mcp/package.json"
printf 'export const value = 1;\n' > "$repo/packages/mcp/index.ts"
printf 'name: Release\n' > "$repo/.github/workflows/release.yml"
git -C "$repo" add .
git -C "$repo" commit --quiet -m release
git -C "$repo" tag mcp-v0.1.1
release_sha=$(git -C "$repo" rev-parse HEAD)

run_source() {
  local event=$1
  local ref=$2
  local sha=$3
  (
    cd "$repo"
    RELEASE_TAG="${4:-mcp-v0.1.1}" \
      GITHUB_EVENT_NAME="$event" \
      GITHUB_REF="$ref" \
      GITHUB_SHA="$sha" \
      "$source_script"
  )
}

expect_failure() {
  local expected=$1
  shift
  local output
  if output=$("$@" 2>&1); then
    echo "Expected failure containing: $expected" >&2
    exit 1
  fi
  if [[ "$output" != *"$expected"* ]]; then
    echo "Expected '$expected', received '$output'" >&2
    exit 1
  fi
}

extract_run() {
  awk -v job="$1" '
    $0 == "  " job ":" { selected = 1; next }
    selected && /^  [a-z]/ { exit }
    selected && /^        run: \|/ { emit = 1; next }
    emit && /^          / { print substr($0, 11); next }
    emit && /^$/ { print; next }
    emit { exit }
  ' "$root/.github/workflows/release.yml"
}

[[ "$(run_source push refs/heads/main "$release_sha")" == "$release_sha" ]]
expect_failure "Package publication must run from main" \
  run_source push refs/tags/mcp-v0.1.1 "$release_sha"
expect_failure "Release tag, event, and checkout commits do not match" \
  run_source push refs/heads/main deadbeef
expect_failure "Unsupported release event" \
  run_source release refs/tags/mcp-v0.1.1 "$release_sha"
expect_failure "Invalid release tag" \
  run_source push refs/heads/main "$release_sha" 'mcp-v0.1.1^{}'
expect_failure "Invalid release tag" \
  run_source push refs/heads/main "$release_sha" '../main'
git -C "$repo" tag --annotate annotated-v0.1.1 --message release
[[ "$(run_source push refs/heads/main "$release_sha" annotated-v0.1.1)" == "$release_sha" ]]

printf '{"name":"repo","packageManager":"pnpm@11.25.0"}\n' > "$repo/package.json"
printf 'name: CI\n' > "$repo/.github/workflows/ci.yml"
printf 'name: Release fixed\n' > "$repo/.github/workflows/release.yml"
cp "$source_script" "$artifact_script" "$repo/.github/scripts/"
cp "$root/.github/scripts/release_test.sh" "$repo/.github/scripts/"
printf 'SDK release recovery\n' > "$repo/README.md"
git -C "$repo" add .
git -C "$repo" commit --quiet -m recovery
recovery_sha=$(git -C "$repo" rev-parse HEAD)

[[ "$(run_source workflow_dispatch refs/heads/main "$recovery_sha")" == "$recovery_sha" ]]
expect_failure "Retry checkout and event commits do not match" \
  run_source workflow_dispatch refs/heads/main "$release_sha"
expect_failure "Release retries must use the workflow from main" \
  run_source workflow_dispatch refs/heads/recovery "$recovery_sha"

mkdir -p "$tmp/bin"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$%s"\n' DRAFT > "$tmp/bin/gh"
chmod +x "$tmp/bin/gh"
extract_run discover-release > "$tmp/discover.sh"
run_discovery() {
  (
    cd "$repo"
    PATH="$tmp/bin:$PATH" DRAFT="$1" RELEASE_TAG="${2:-mcp-v0.1.1}" RELEASED_PATHS='[]' \
      GITHUB_REPOSITORY=telemetry-dev/sdks GITHUB_EVENT_NAME=workflow_dispatch \
      GITHUB_REF=refs/heads/main GITHUB_SHA="${3:-$recovery_sha}" GITHUB_OUTPUT="$tmp/output" \
      bash -eo pipefail "$tmp/discover.sh"
  )
}
run_discovery false
grep -Fxq 'npm=["packages/mcp"]' "$tmp/output"
grep -Fxq 'python=[]' "$tmp/output"
grep -Fxq "sha=$recovery_sha" "$tmp/output"
expect_failure "Release mcp-v0.1.1 is still a draft" run_discovery true
git -C "$repo" tag unknown-v0.1.1
expect_failure "Expected one package for unknown-v0.1.1, found 0" run_discovery false unknown-v0.1.1
git -C "$repo" tag python-v0.2.2 "$release_sha"
run_discovery false python-v0.2.2
grep -Fxq 'python=["sdks/python"]' "$tmp/output"

git -C "$repo" switch --quiet --detach "$recovery_sha"
printf '{"packages/otel":"0.1.0","packages/mcp":"0.1.2","sdks/python":"0.2.3"}\n' > "$repo/.release-please-manifest.json"
jq '.packages = ({"packages/otel": {"component": "otel", "release-type": "node"}} + .packages)' \
  "$repo/release-please-config.json" > "$repo/release-please-config.json.next"
mv "$repo/release-please-config.json.next" "$repo/release-please-config.json"
git -C "$repo" commit --quiet --all -m release-batch
batch_sha=$(git -C "$repo" rev-parse HEAD)
for tag in otel-v0.1.0 mcp-v0.1.2 python-v0.2.3; do
  git -C "$repo" tag "$tag"
done
run_push_discovery() {
  (
    cd "$repo"
    rm -f "$tmp/output"
    PATH="$tmp/bin:$PATH" DRAFT="$1" RELEASED_PATHS="$2" RELEASE_TAG='' \
      GITHUB_REPOSITORY=telemetry-dev/sdks GITHUB_EVENT_NAME=push \
      GITHUB_REF=refs/heads/main GITHUB_SHA="${3:-$batch_sha}" GITHUB_OUTPUT="$tmp/output" \
      bash -eo pipefail "$tmp/discover.sh"
  )
}
for released in '["sdks/python","packages/mcp","packages/otel"]' '["packages/mcp"]' '[]'; do
  run_push_discovery false "$released"
  grep -Fxq 'npm=["packages/otel","packages/mcp"]' "$tmp/output"
  grep -Fxq 'python=["sdks/python"]' "$tmp/output"
  grep -Fxq "sha=$batch_sha" "$tmp/output"
done
run_push_discovery false '[]' "$recovery_sha"
grep -Fxq 'npm=[]' "$tmp/output"
grep -Fxq 'python=[]' "$tmp/output"
expect_failure "Expected 2 configured packages for released paths, found 1" \
  run_push_discovery false '["packages/mcp","packages/unknown"]'
expect_failure "Release otel-v0.1.0 is still a draft" run_push_discovery true '["packages/otel"]'
expect_failure "Release tag, event, and checkout commits do not match" \
  run_push_discovery false '["packages/otel"]' "$release_sha"
git -C "$repo" commit --quiet --allow-empty -m after-batch
expect_failure "Release tag, event, and checkout commits do not match" \
  run_push_discovery false '["packages/otel"]' "$(git -C "$repo" rev-parse HEAD)"

release_tree=$(git -C "$repo" show --no-patch --format=%T "$release_sha")
unrelated_sha=$(printf 'unrelated\n' | git -C "$repo" commit-tree "$release_tree")
git -C "$repo" switch --quiet --detach "$unrelated_sha"
expect_failure "Release tag is not an ancestor of the retry commit" \
  run_source workflow_dispatch refs/heads/main "$unrelated_sha"

git -C "$repo" switch --quiet --detach "$recovery_sha"
printf 'export const value = 2;\n' > "$repo/packages/mcp/index.ts"
git -C "$repo" add packages/mcp/index.ts
git -C "$repo" commit --quiet -m source-change
source_change_sha=$(git -C "$repo" rev-parse HEAD)
expect_failure "Retry commit changes release source files" \
  run_source workflow_dispatch refs/heads/main "$source_change_sha"
expect_failure "Retry commit changes release source files" \
  run_discovery false mcp-v0.1.1 "$source_change_sha"

for path in scripts/package-exports.mjs sdks/python/pyproject.toml pnpm-lock.yaml; do
  git -C "$repo" switch --quiet --detach "$recovery_sha"
  mkdir -p "$(dirname "$repo/$path")"
  printf 'changed\n' > "$repo/$path"
  git -C "$repo" add "$path"
  git -C "$repo" commit --quiet -m input-change
  expect_failure "Retry commit changes release source files" \
    run_source workflow_dispatch refs/heads/main "$(git -C "$repo" rev-parse HEAD)"
done

git -C "$repo" switch --quiet --detach "$recovery_sha"
jq '.packages[":(exclude)release-please-config.json"] = .packages["packages/mcp"] | .packages[":(exclude)packages/mcp"] = .packages["packages/mcp"]' \
  "$repo/release-please-config.json" > "$repo/release-please-config.json.next"
mv "$repo/release-please-config.json.next" "$repo/release-please-config.json"
printf 'export const value = 3;\n' > "$repo/packages/mcp/index.ts"
git -C "$repo" add release-please-config.json packages/mcp/index.ts
git -C "$repo" commit --quiet -m pathspec-change
pathspec_change_sha=$(git -C "$repo" rev-parse HEAD)
expect_failure "Retry commit changes release source files" \
  run_source workflow_dispatch refs/heads/main "$pathspec_change_sha"

git -C "$repo" switch --quiet --detach "$recovery_sha"
jq '.name = "changed"' "$repo/package.json" > "$repo/package.json.next"
mv "$repo/package.json.next" "$repo/package.json"
git -C "$repo" add package.json
git -C "$repo" commit --quiet -m package-change
package_change_sha=$(git -C "$repo" rev-parse HEAD)
expect_failure "Retry commit changes package.json outside packageManager" \
  run_source workflow_dispatch refs/heads/main "$package_change_sha"

mkdir -p "$tmp/archive/package"
printf '{"name":"@telemetry-dev/mcp","version":"0.1.1"}\n' > "$tmp/archive/package/package.json"
tar -czf "$tmp/mcp.tgz" -C "$tmp/archive" package
printf '{"packages/mcp":"0.1.1"}\n' > "$repo/.release-please-manifest.json"
(
  cd "$repo"
  "$artifact_script" packages/mcp "$tmp/mcp.tgz"
)
printf '{"packages/mcp":"9.9.9"}\n' > "$repo/.release-please-manifest.json"
expect_failure "Packed version 0.1.1 does not match release version 9.9.9" \
  bash -c "cd '$repo' && '$artifact_script' packages/mcp '$tmp/mcp.tgz'"
printf '{"name":"@telemetry-dev/other","version":"0.1.1"}\n' > "$tmp/archive/package/package.json"
tar -czf "$tmp/wrong-name.tgz" -C "$tmp/archive" package
expect_failure "Packed name @telemetry-dev/other does not match release name @telemetry-dev/mcp" \
  bash -c "cd '$repo' && '$artifact_script' packages/mcp '$tmp/wrong-name.tgz'"

mkdir -p "$tmp/publish/npm"
printf 'mcp.tgz\n' > "$tmp/publish/npm/publish-order.txt"
extract_run publish-npm > "$tmp/publish.sh"
cat > "$tmp/bin/npm" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$QUERIES"
case "$*" in
  "view @telemetry-dev/mcp@0.1.1 dist.integrity") printf '%s\n' "$REMOTE_INTEGRITY" ;;
  "view @telemetry-dev/sdk@^0.1.1 version --json")
    echo attempt >> "$ATTEMPTS"
    [[ "$DEPENDENCY_AVAILABLE" == true ]] || { echo 'dependency unavailable' >&2; exit 1; }
    printf '"0.1.1"\n'
    ;;
  "publish ./npm/mcp.tgz --access public") echo published >> "$PUBLISHED" ;;
  *) echo "Unexpected npm arguments: $*" >&2; exit 1 ;;
esac
SH
printf '#!/usr/bin/env bash\nexit 0\n' > "$tmp/bin/sleep"
chmod +x "$tmp/bin/npm" "$tmp/bin/sleep"
run_publish() {
  (
    cd "$tmp/publish"
    PATH="$tmp/bin:$PATH" REMOTE_INTEGRITY="$1" DEPENDENCY_AVAILABLE="$2" \
      ATTEMPTS="$tmp/attempts" PUBLISHED="$tmp/published" QUERIES="$tmp/queries" \
      bash -euo pipefail "$tmp/publish.sh"
  )
}
for field in dependencies peerDependencies; do
  jq -n --arg field "$field" '{
    name: "@telemetry-dev/mcp", version: "0.1.1",
    ($field): {"@telemetry-dev/sdk": "^0.1.1", "@telemetry-dev-extra/helper": "^0.1.1"}
  }' > "$tmp/archive/package/package.json"
  tar -czf "$tmp/publish/npm/mcp.tgz" -C "$tmp/archive" package
  rm -f "$tmp/attempts" "$tmp/published" "$tmp/queries"
  integrity="sha512-$(openssl dgst -sha512 -binary "$tmp/publish/npm/mcp.tgz" | openssl base64 -A)"
  run_publish "$integrity" false
  [[ ! -e "$tmp/published" && ! -e "$tmp/attempts" ]]
  expect_failure 'Integrity mismatch' run_publish sha512-wrong true
  expect_failure 'unable to verify that published @telemetry-dev/sdk satisfies ^0.1.1' run_publish '' false
  [[ ! -e "$tmp/published" && $(wc -l < "$tmp/attempts") -eq 60 ]]
  run_publish '' true
  [[ $(wc -l < "$tmp/published") -eq 1 && $(wc -l < "$tmp/attempts") -eq 61 ]]
  if grep -Fq '@telemetry-dev-extra/helper' "$tmp/queries"; then
    echo "Unexpected registry query for namespace near-miss in $field" >&2
    exit 1
  fi
done

mkdir -p "$tmp/python-publish/.github/scripts" "$tmp/python-publish/python/python" \
  "$tmp/python-publish/python/python-anthropic" "$tmp/python-bin"
touch "$tmp/python-publish/python/python/telemetry_dev-0.2.3-py3-none-any.whl" \
  "$tmp/python-publish/python/python-anthropic/telemetry_dev_anthropic-0.1.4-py3-none-any.whl"
printf 'python\npython-anthropic\n' > "$tmp/python-publish/python/publish-order.txt"
cat > "$tmp/python-publish/.github/scripts/verify-python-release-dependencies.sh" <<'SH'
#!/usr/bin/env bash
printf 'verify %s\n' "$1" >> "$PYTHON_LOG"
[[ "$1" != *anthropic* || "$ANTHROPIC_DEPENDENCIES_AVAILABLE" == true ]] || {
  echo 'anthropic dependencies unavailable' >&2
  exit 1
}
SH
cat > "$tmp/python-bin/uv" <<'SH'
#!/usr/bin/env bash
printf 'uv %s\n' "$*" >> "$PYTHON_LOG"
SH
chmod +x "$tmp/python-publish/.github/scripts/verify-python-release-dependencies.sh" "$tmp/python-bin/uv"
extract_run publish-python > "$tmp/publish-python.sh"
run_python_publish() {
  (
    cd "$tmp/python-publish"
    rm -f "$tmp/python-log"
    PATH="$tmp/python-bin:$PATH" PYTHON_LOG="$tmp/python-log" ANTHROPIC_DEPENDENCIES_AVAILABLE="$1" \
      bash -eo pipefail "$tmp/publish-python.sh"
  )
}
uv_publish='uv publish --trusted-publishing always --check-url https://pypi.org/simple'
core_log="verify python/python/telemetry_dev-0.2.3-py3-none-any.whl
$uv_publish python/python/telemetry_dev-0.2.3-py3-none-any.whl
verify python/python-anthropic/telemetry_dev_anthropic-0.1.4-py3-none-any.whl"
run_python_publish true
[[ "$(cat "$tmp/python-log")" == "$core_log
$uv_publish python/python-anthropic/telemetry_dev_anthropic-0.1.4-py3-none-any.whl" ]]
expect_failure 'anthropic dependencies unavailable' run_python_publish false
[[ "$(cat "$tmp/python-log")" == "$core_log" ]]

touch "$tmp/provider.whl"
cat > "$tmp/bin/uv" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == venv ]]; then
  [[ "$#" -eq 3 && "$2" == --quiet && -d "$3" ]]
  printf '%s\n' "$3" > "$ATTEMPTS.venv"
  exit 0
fi
[[ "$*" == "pip install --dry-run --no-cache --python $(cat "$ATTEMPTS.venv") --no-sources --no-build $EXPECTED_ARTIFACT" ]]
printf 'attempt\n' >> "$ATTEMPTS"
attempt=$(wc -l < "$ATTEMPTS")
if [[ "${PERMANENT_ERROR:-false}" == true ]]; then
  echo 'No solution found: package requires Python >=4' >&2
  exit 1
fi
if ((attempt >= DEPENDENCY_AVAILABLE_AFTER)); then
  exit 0
fi
if [[ "${ONLY_OLDER_AVAILABLE:-false}" == true ]]; then
  echo '  ╰─▶ Because only telemetry-dev<=0.2.5 is available and' >&2
else
  echo 'Because there is no version of telemetry-dev==0.2.6' >&2
fi
exit 1
SH
chmod +x "$tmp/bin/uv"
run_dependency_check() {
  (
    PATH="$tmp/bin:$PATH" ATTEMPTS="$tmp/dependency-attempts" \
      EXPECTED_ARTIFACT="$tmp/provider.whl" \
      DEPENDENCY_AVAILABLE_AFTER="$1" RELEASE_DEPENDENCY_ATTEMPTS="$2" \
      PERMANENT_ERROR="${3:-false}" \
      RELEASE_DEPENDENCY_DELAY_SECONDS=0 \
      "$dependency_script" "$tmp/provider.whl"
  )
}
run_dependency_check_with_settings() {
  (
    PATH="$tmp/bin:$PATH" ATTEMPTS="$tmp/dependency-attempts" \
      EXPECTED_ARTIFACT="$tmp/provider.whl" DEPENDENCY_AVAILABLE_AFTER=1 \
      RELEASE_DEPENDENCY_ATTEMPTS="$1" RELEASE_DEPENDENCY_DELAY_SECONDS="$2" \
      "$dependency_script" "$tmp/provider.whl"
  )
}
rm -f "$tmp/dependency-attempts"
run_dependency_check 3 5
[[ $(wc -l < "$tmp/dependency-attempts") -eq 3 ]]
rm -f "$tmp/dependency-attempts"
ONLY_OLDER_AVAILABLE=true run_dependency_check 3 5
[[ $(wc -l < "$tmp/dependency-attempts") -eq 3 ]]
rm -f "$tmp/dependency-attempts"
expect_failure "published dependencies did not become available" run_dependency_check 5 4
[[ $(wc -l < "$tmp/dependency-attempts") -eq 4 ]]
rm -f "$tmp/dependency-attempts"
expect_failure "package requires Python >=4" run_dependency_check 5 4 true
[[ $(wc -l < "$tmp/dependency-attempts") -eq 1 ]]
expect_failure "RELEASE_DEPENDENCY_ATTEMPTS must be an integer from 1 to 60" \
  run_dependency_check_with_settings nope 0
expect_failure "RELEASE_DEPENDENCY_ATTEMPTS must be an integer from 1 to 60" \
  run_dependency_check_with_settings 0 0
expect_failure "RELEASE_DEPENDENCY_ATTEMPTS must be an integer from 1 to 60" \
  run_dependency_check_with_settings 61 0
expect_failure "RELEASE_DEPENDENCY_DELAY_SECONDS must be an integer from 0 to 60" \
  run_dependency_check_with_settings 1 1.5
expect_failure "RELEASE_DEPENDENCY_DELAY_SECONDS must be an integer from 0 to 60" \
  run_dependency_check_with_settings 1 -1
expect_failure "RELEASE_DEPENDENCY_DELAY_SECONDS must be an integer from 0 to 60" \
  run_dependency_check_with_settings 1 61

printf '{"sdks/python":"0.2.2"}\n' > "$repo/.release-please-manifest.json"
printf '{"packages":{"sdks/python":{"component":"python","release-type":"python","package-name":"telemetry-dev"}}}\n' > "$repo/release-please-config.json"
python3 - "$tmp" <<'PY'
import io
import pathlib
import sys
import tarfile
import zipfile

root = pathlib.Path(sys.argv[1])
for label, name, version in [("valid", "telemetry-dev", "0.2.2"), ("version", "telemetry-dev", "9.9.9"), ("name", "other", "0.2.2")]:
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n".encode()
    with zipfile.ZipFile(root / f"{label}.whl", "w") as archive:
        archive.writestr("telemetry_dev-0.2.2.dist-info/METADATA", metadata)
    with tarfile.open(root / f"{label}.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("telemetry_dev-0.2.2/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
PY
for suffix in whl tar.gz; do
  (cd "$repo" && "$artifact_script" sdks/python "$tmp/valid.$suffix")
  expect_failure "Packed version 9.9.9 does not match release version 0.2.2" \
    bash -c "cd '$repo' && '$artifact_script' sdks/python '$tmp/version.$suffix'"
  expect_failure "Packed name other does not match release name telemetry-dev" \
    bash -c "cd '$repo' && '$artifact_script' sdks/python '$tmp/name.$suffix'"
done

package_pnpm=$(jq -er '.packageManager | capture("^pnpm@(?<version>.+)$").version' "$root/package.json")
workflow_pnpm=$(
  awk '
    /uses: pnpm\/action-setup@/ { setup = 1; next }
    setup && /version:/ { print $2; exit }
    setup && /uses:|run:/ { exit 1 }
  ' "$root/.github/workflows/release.yml"
)
[[ "$workflow_pnpm" == "$package_pnpm" ]]
[[ $(grep -Ec '^[[:space:]]*concurrency:' "$root/.github/workflows/release.yml") -eq 2 ]]
grep -Fxq "  group: release-\${{ inputs.release_tag || github.sha }}" \
  "$root/.github/workflows/release.yml"
grep -Fxq "      group: release-please-\${{ github.ref }}" \
  "$root/.github/workflows/release.yml"
grep -Fxq "      !cancelled() && (github.event_name == 'workflow_dispatch' ||" \
  "$root/.github/workflows/release.yml"
grep -Fxq "      needs.release-please.result == 'success')" \
  "$root/.github/workflows/release.yml"
grep -Fxq "      queue: max" "$root/.github/workflows/release.yml"
[[ $(grep -Fxc "      !cancelled() && needs.discover-release.result == 'success' &&" \
  "$root/.github/workflows/release.yml") -eq 2 ]]
for job in npm python; do
  grep -Fxq "    if: \"!cancelled() && needs.build-$job.result == 'success'\"" \
    "$root/.github/workflows/release.yml"
done
grep -Fxq "          RELEASE_DEPENDENCY_ATTEMPTS: 60" "$root/.github/workflows/release.yml"
python3 - "$root" <<'PY'
import json
import pathlib
import re
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
packages = json.loads((root / "release-please-config.json").read_text())["packages"]
order = list(packages)


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


paths = {}
dependencies = {}
for path, package in packages.items():
    if package["release-type"] == "node":
        manifest = json.loads((root / path / "package.json").read_text())
        paths[manifest["name"]] = path
        dependencies[path] = [*manifest.get("dependencies", {}), *manifest.get("peerDependencies", {})]
    else:
        project = tomllib.loads((root / path / "pyproject.toml").read_text())["project"]
        paths[normalize(project["name"])] = path
        dependencies[path] = [
            normalize(re.match(r"[A-Za-z0-9._-]+", requirement)[0]) for requirement in project.get("dependencies", [])
        ]
for path in order:
    for dependency in dependencies[path]:
        dependency_path = paths.get(dependency)
        if dependency_path and order.index(dependency_path) > order.index(path):
            sys.exit(f"release-please-config.json lists {path} before its dependency {dependency_path}")
PY
grep -Fxq "          NPM_CONFIG_PROVENANCE: \${{ github.event.repository.private && 'false' || 'true' }}" \
  "$root/.github/workflows/release.yml"
grep -Fxq "            npm publish \"./npm/\$tarball\" --access public" \
  "$root/.github/workflows/release.yml"
