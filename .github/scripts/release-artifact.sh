#!/usr/bin/env bash
set -euo pipefail

package=${1:?package path is required}
artifact=${2:?artifact path is required}
expected_version=$(jq -er --arg path "$package" '.[$path]' .release-please-manifest.json)
release_type=$(jq -er --arg path "$package" '.packages[$path]["release-type"]' release-please-config.json)

case "$release_type" in
  node)
    expected_name=$(jq -er '.name' "$package/package.json")
    metadata=$(tar -xOf "$artifact" package/package.json)
    packed_name=$(jq -er '.name' <<< "$metadata")
    packed_version=$(jq -er '.version' <<< "$metadata")
    ;;
  python)
    expected_name=$(jq -er --arg path "$package" '.packages[$path]["package-name"]' release-please-config.json)
    metadata=$(python3 - "$artifact" <<'PY'
import email.parser
import json
import sys
import tarfile
import zipfile

path = sys.argv[1]
if path.endswith(".whl"):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(members) != 1:
            sys.exit("Expected one wheel metadata file")
        content = archive.read(members[0])
elif path.endswith(".tar.gz"):
    with tarfile.open(path) as archive:
        members = [member for member in archive.getmembers() if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO") and member.isfile()]
        if len(members) != 1:
            sys.exit("Expected one sdist metadata file")
        content = archive.extractfile(members[0]).read()
else:
    sys.exit("Unsupported Python artifact")

metadata = email.parser.BytesParser().parsebytes(content)
if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
    sys.exit("Expected one artifact name and version")
print(json.dumps({"name": metadata["Name"], "version": metadata["Version"]}))
PY
    )
    packed_name=$(jq -er '.name' <<< "$metadata")
    packed_version=$(jq -er '.version' <<< "$metadata")
    ;;
  *)
    echo "Unsupported release type: $release_type" >&2
    exit 1
    ;;
esac

if [[ "$packed_name" != "$expected_name" ]]; then
  echo "Packed name $packed_name does not match release name $expected_name" >&2
  exit 1
fi
if [[ "$packed_version" != "$expected_version" ]]; then
  echo "Packed version $packed_version does not match release version $expected_version" >&2
  exit 1
fi

if [[ "$release_type" == node ]]; then
  # pnpm pack writes publishConfig.exports into exports. A missing target fails only on import.
  packed_files=$(tar -tf "$artifact")
  targets=$(jq -r '.exports, .bin | .. | strings' <<< "$metadata")
  while IFS= read -r target; do
    [[ -n "$target" ]] || continue
    if ! grep -Fxq "package/${target#./}" <<< "$packed_files"; then
      echo "Packed $packed_name has no $target, a target of its exports or bin" >&2
      exit 1
    fi
  done <<< "$targets"
fi
