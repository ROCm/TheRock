# Experimental LLVM artifact cache

This prototype finds reusable LLVM artifacts by comparing a build's fingerprints
with fingerprints from previous CI runs. It writes a reuse plan; downloading
archives, restoring stages, and skipping compilation are separate steps.

[artifact_cache_index.json](artifact_cache_index.json) is an experimental,
manually maintained artifact cache index. It contains metadata (artifact names,
fingerprints, platforms, CI run information, archive locations,
and checksums), not artifact archives. JSON makes this data easy to inspect,
edit, and test. A planned dynamic cache database will replace this file and
allow CI to register and query artifacts automatically.

The `amd-llvm` bundle includes LLVM, COMGR, and HIPCC. Reuse requires all five
components: `dbg`, `dev`, `doc`, `lib`, and `run`. Keep one complete fingerprint
per platform. Recording both Linux and Windows gives ten entries: five per
platform. Builds requesting a removed fingerprint get a cache miss and must
build normally.

**Run commands from the TheRock repository root.** The default artifact cache
index is `build_tools/artifact_cache/artifact_cache_index.json`. Use `--index`
to select another file.

## Update the artifact cache index with the `record` subcommand

Run the `record` subcommand manually with a CI run ID. It downloads the run's
fingerprint and checksum files, then saves their values and artifact locations
in the artifact cache index. A different fingerprint replaces the selected
platform's previous entries; other platforms remain unchanged. Matching entries
keep their original run ID, artifact location, and checksum. The run you select
determines the retained fingerprint; the script does not compare run dates.

Choose a run whose relevant artifact-producing jobs succeeded and uploaded the
complete bundle; the script does not check job status.

Update both platforms directly in
`build_tools/artifact_cache/artifact_cache_index.json`:

```bash
cache_run_id="43936062480"

for cache_platform in linux windows; do
  python build_tools/artifact_cache/artifact_cache.py record \
    --index build_tools/artifact_cache/artifact_cache_index.json \
    --run-id "$cache_run_id" \
    --platform "$cache_platform" || break
done
```

For a future update, change `cache_run_id` and rerun the block. The loop stops
if a recording command fails. Each successful command updates one platform.

After both succeed, the artifact cache index contains ten entries. Different
fingerprints replace the previous entries; matching fingerprints print
`Already indexed` and retain their original run information. Use the regeneration
procedure below when every entry must reference the selected run ID.

By default, metadata is downloaded from:

```text
https://therock-ci-artifacts.s3.amazonaws.com/<run-id>-<platform>/
```

The files are `amd-llvm_<component>_generic.fprint` and
`amd-llvm_<component>_generic.tar.zst.sha256sum` for each component.

| Option                               | Purpose                                                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `--source-root`                      | Use a different artifact location.                                                                                                     |
| `--metadata-dir ./producer-metadata` | Read those files from `./producer-metadata/` instead of downloading them. Use `--source-root` to specify the actual artifact location. |
| `--archive-suffix .tar.xz`           | Read checksums for `.tar.xz` archives instead of `.tar.zst`.                                                                           |

Review changes to `build_tools/artifact_cache/artifact_cache_index.json`:

```bash
git diff -- build_tools/artifact_cache/artifact_cache_index.json
```

## Regenerate the artifact cache index with the `record` subcommand

Start with an empty artifact cache index so every entry is recorded from the
selected run, including entries whose fingerprints already matched. This example
creates a temporary file containing Linux and Windows entries from run
`43936062480`. The commands use your same local checkout and Python environment;
only the output JSON is temporary:

```bash
artifact_cache_index="$(mktemp /tmp/llvm-cache-index-XXXXXX.json)"
printf 'Temporary artifact cache index: %s\n' "$artifact_cache_index"

(
  set -e
  printf '%s\n' '{"schema_version": 1, "entries": []}' > "$artifact_cache_index"

  for cache_platform in linux windows; do
    python build_tools/artifact_cache/artifact_cache.py record \
      --index "$artifact_cache_index" \
      --run-id 43936062480 \
      --platform "$cache_platform"
  done
)
```

Expected: five components added per platform, ten entries in total.
Inspect the temporary artifact cache index at the printed `$artifact_cache_index`
path and compare it with `build_tools/artifact_cache/artifact_cache_index.json`:

```bash
python -m json.tool "$artifact_cache_index"
```

```bash
diff -u build_tools/artifact_cache/artifact_cache_index.json "$artifact_cache_index"
```

The comparison does not modify either file; exit code `1` means they differ.
After both recording commands succeed and you review the temporary artifact
cache index, copy it to the repository's file:

```bash
cp "$artifact_cache_index" build_tools/artifact_cache/artifact_cache_index.json
git diff -- build_tools/artifact_cache/artifact_cache_index.json
```

For routine updates that preserve already-matching entries, use the direct
update example above.

## Find matching artifacts with the `lookup` subcommand

After configuring TheRock, use the `lookup` subcommand to compare the local
`build/artifacts/amd-llvm_<component>_generic.fprint` files with the artifact
cache index. This subcommand makes no network requests and leaves the artifact
cache index unchanged.

```bash
python build_tools/artifact_cache/artifact_cache.py lookup \
  --fingerprints-dir build/artifacts \
  --platform linux \
  --output build/llvm-cache-plan.json
```

Matching uses the artifact name, fingerprint, and platform. Build configuration
differences must be represented by the configure-time fingerprint.
Inspect `build/llvm-cache-plan.json`:

```bash
python -m json.tool build/llvm-cache-plan.json
```

- **HIT:** all five components match; `action` is `populate`.
- **MISS:** a fingerprint is missing, invalid, inconsistent, or unmatched;
  `action` is `build`, with no copy requests.

Both results return exit code `0`; input or I/O errors return `2`. A hit identifies
candidate artifacts. The future population step must verify their availability
and archive checksums before reuse.

## Local test procedure

Use Bash, Python, and `curl` in the **same terminal**, with your Python environment
activated. Stop if a command fails. Only steps 3, 4, and 6 need network access.
These tests use temporary files and do not modify the checked-in artifact cache
index or `build/artifacts/`.

### 1. Run unit tests

Check the script's behavior using temporary local data and simulated downloads.

```bash
python build_tools/artifact_cache/artifact_cache_test.py -v
```

Expected: `OK`. These tests include replacement of older fingerprints,
preservation of other platforms, and failure handling. They use local fixtures
and mocked downloads.

### 2. Prepare temporary files

Create an isolated test workspace with an empty artifact cache index and empty fingerprint directories.

```bash
cache_test_dir="$(mktemp -d /tmp/llvm-cache-test-XXXXXX)"
cache_run_id="43936062480"
cache_source_root="https://therock-ci-artifacts.s3.amazonaws.com/${cache_run_id}-linux"

mkdir "$cache_test_dir/empty-fingerprints" \
      "$cache_test_dir/matching-fingerprints" \
      "$cache_test_dir/changed-fingerprints" \
      "$cache_test_dir/incomplete-fingerprints"

printf '%s\n' '{"schema_version": 1, "entries": []}' \
  > "$cache_test_dir/artifact_cache_index.json"
printf 'Test directory: %s\n' "$cache_test_dir"
```

Expected: `$cache_test_dir/artifact_cache_index.json` has no entries, and all
four fingerprint directories are empty. The printed path identifies the test
files used below.

### 3. Add a CI run with the `record` subcommand

Verify that the `record` subcommand downloads the CI run's metadata and adds five LLVM components to the artifact cache index.

```bash
python build_tools/artifact_cache/artifact_cache.py record \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --run-id "$cache_run_id" \
  --platform linux

python -m json.tool "$cache_test_dir/artifact_cache_index.json"
```

Expected: `Recorded 5 LLVM components from run 43936062480`.

The `record` subcommand downloads metadata from `$cache_source_root` and writes
five entries to `$cache_test_dir/artifact_cache_index.json`. It uses its own
temporary download directory, so the four fingerprint directories remain empty.

### 4. Repeat the `record` subcommand

Verify that running the `record` subcommand again leaves the artifact cache index unchanged.

```bash
cp "$cache_test_dir/artifact_cache_index.json" \
   "$cache_test_dir/artifact_cache_index-before-repeat.json"

python build_tools/artifact_cache/artifact_cache.py record \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --run-id "$cache_run_id" \
  --platform linux

cmp "$cache_test_dir/artifact_cache_index-before-repeat.json" \
    "$cache_test_dir/artifact_cache_index.json"
```

Expected: `Already indexed`; `cmp` produces no output and returns `0`.
Metadata is downloaded again, but `$cache_test_dir/artifact_cache_index.json`
is unchanged.

### 5. Test missing fingerprints with the `lookup` subcommand

Verify that the `lookup` subcommand returns a miss when the requested fingerprint files are missing.

```bash
python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/empty-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/missing-plan.json"

python -m json.tool "$cache_test_dir/missing-plan.json"
```

Expected: `MISS (0/5 components)` with `missing_expected_fingerprint`.
The input directory `$cache_test_dir/empty-fingerprints` is intentionally empty.

### 6. Test matching fingerprints with the `lookup` subcommand

Verify that the `lookup` subcommand returns a hit when all five requested fingerprints match the artifact cache index.

Download the same CI run's fingerprints into a separate input directory:

```bash
(
  set -e
  for component in dbg dev doc lib run; do
    filename="amd-llvm_${component}_generic.fprint"
    curl --fail --silent --show-error \
      "$cache_source_root/$filename" \
      --output "$cache_test_dir/matching-fingerprints/$filename"
  done
)

python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/matching-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/hit-plan.json"

python -m json.tool "$cache_test_dir/hit-plan.json"
```

Expected: `HIT (5/5 components)` with `matching_index_entry`.

Only `curl` accesses the network; the `lookup` subcommand reads the downloaded
files locally. This controlled hit tests matching against saved metadata. It
does not prove that your local TheRock configuration produces the same
fingerprints; use the earlier example with `build/artifacts/` for that check.

### 7. Test changed fingerprints with the `lookup` subcommand

Verify that the `lookup` subcommand returns a miss when the requested fingerprints do not match any saved entries.

Create a valid fingerprint absent from the artifact cache index and use it for
all five components:

```bash
python - "$cache_test_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
artifact_cache_index = json.loads((root / "artifact_cache_index.json").read_text())
known_fingerprints = {entry["fingerprint"] for entry in artifact_cache_index["entries"]}
fingerprint = hashlib.sha256(b"README deliberately unmatched fingerprint").hexdigest()
while fingerprint in known_fingerprints:
    fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()
for component in ("dbg", "dev", "doc", "lib", "run"):
    path = root / "changed-fingerprints" / f"amd-llvm_{component}_generic.fprint"
    path.write_text(fingerprint + "\n", encoding="utf-8")
PY

python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/changed-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/changed-plan.json"

python -m json.tool "$cache_test_dir/changed-plan.json"
```

Expected: `MISS (0/5 components)` with `no_matching_index_entry`.

### 8. Test an incomplete bundle with the `lookup` subcommand

```bash
cp "$cache_test_dir/matching-fingerprints/"*.fprint \
   "$cache_test_dir/incomplete-fingerprints/"
rm "$cache_test_dir/incomplete-fingerprints/amd-llvm_dbg_generic.fprint"

python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/incomplete-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/incomplete-plan.json"

python -m json.tool "$cache_test_dir/incomplete-plan.json"
```

Expected: `MISS (4/5 components)`. The debug fingerprint is missing. The four
individual hits must not allow partial bundle reuse.

### 9. Verify the saved plans

Check each plan's status, action, and copy requests:

```bash
python - "$cache_test_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "missing-plan.json": ("miss", "build", 0),
    "hit-plan.json": ("hit", "populate", 5),
    "changed-plan.json": ("miss", "build", 0),
    "incomplete-plan.json": ("miss", "build", 0),
}
for filename, wanted in expected.items():
    path = root / filename
    plan = json.loads(path.read_text())
    actual = (plan["status"], plan["action"], len(plan["copy_requests"]))
    assert actual == wanted, f"{path}: expected {wanted}, got {actual}"
    print(f"PASS {path}: status={actual[0]}, action={actual[1]}, copy_requests={actual[2]}")
PY
```

Expected: four `PASS` lines.

| Plan file                              | Status | Action     | Copy requests |
| -------------------------------------- | ------ | ---------- | ------------- |
| `$cache_test_dir/missing-plan.json`    | `miss` | `build`    | 0             |
| `$cache_test_dir/hit-plan.json`        | `hit`  | `populate` | 5             |
| `$cache_test_dir/changed-plan.json`    | `miss` | `build`    | 0             |
| `$cache_test_dir/incomplete-plan.json` | `miss` | `build`    | 0             |

Test files remain under `$cache_test_dir`. No LLVM compilation or artifact
archive downloads are performed.
