#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
source_script="$root/.github/scripts/release-source.sh"
artifact_script="$root/.github/scripts/release-artifact.sh"
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

[[ "$(run_source release refs/tags/mcp-v0.1.1 "$release_sha")" == "$release_sha" ]]
expect_failure "Package publication must run from a release tag" \
  run_source release refs/heads/main "$release_sha"
expect_failure "Release tag, event, and checkout commits do not match" \
  run_source release refs/tags/mcp-v0.1.1 deadbeef
expect_failure "Unsupported release event" \
  run_source push refs/tags/mcp-v0.1.1 "$release_sha"
expect_failure "Invalid release tag" \
  run_source release 'refs/tags/mcp-v0.1.1^{}' "$release_sha" 'mcp-v0.1.1^{}'
expect_failure "Invalid release tag" \
  run_source release 'refs/tags/../main' "$release_sha" '../main'
git -C "$repo" tag --annotate annotated-v0.1.1 --message release
[[ "$(run_source release refs/tags/annotated-v0.1.1 "$release_sha" annotated-v0.1.1)" == "$release_sha" ]]

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
    PATH="$tmp/bin:$PATH" DRAFT="$1" RELEASE_TAG="${2:-mcp-v0.1.1}" \
      GITHUB_REPOSITORY=telemetry-dev/sdks GITHUB_EVENT_NAME=workflow_dispatch \
      GITHUB_REF=refs/heads/main GITHUB_SHA="$recovery_sha" GITHUB_OUTPUT="$tmp/output" \
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
  [[ ! -e "$tmp/published" && $(wc -l < "$tmp/attempts") -eq 12 ]]
  run_publish '' true
  [[ $(wc -l < "$tmp/published") -eq 1 && $(wc -l < "$tmp/attempts") -eq 13 ]]
  if grep -Fq '@telemetry-dev-extra/helper' "$tmp/queries"; then
    echo "Unexpected registry query for namespace near-miss in $field" >&2
    exit 1
  fi
done

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
[[ $(grep -Ec '^[[:space:]]*concurrency:' "$root/.github/workflows/release.yml") -eq 1 ]]
grep -Fxq "  group: release-\${{ github.event.release.tag_name || inputs.release_tag || github.ref }}" \
  "$root/.github/workflows/release.yml"
grep -Fxq "          NPM_CONFIG_PROVENANCE: \${{ github.event.repository.private && 'false' || 'true' }}" \
  "$root/.github/workflows/release.yml"
grep -Fxq "            npm publish \"./npm/\$tarball\" --access public" \
  "$root/.github/workflows/release.yml"
