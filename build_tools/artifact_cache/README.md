# Artifact cache

Tools for reusing prebuilt TheRock artifacts instead of rebuilding them, by
matching a build's artifact fingerprints (`THEROCK_FPRINT`) with fingerprints
from earlier CI runs. See [#8651](https://github.com/ROCm/TheRock/issues/8651)
and RFC0016, [Fingerprint-Based CI Artifact Cache](https://github.com/ROCm/TheRock/pull/9500).

This directory is expected to grow. Planned contents include populating a local
cache, using the cache in a build, testing the cache, and talking to a
server-side cache. Today it contains only the first step: a prototype that finds
matching artifacts and writes a reuse plan.

| File                                  | Purpose                                        |
| ------------------------------------- | ---------------------------------------------- |
| `artifact_cache.py`                   | The `record` and `lookup` subcommands          |
| `prototype_artifact_cache_index.json` | Prototype index of known CI artifacts          |
| `artifact_cache_test.py`              | Unit tests (no network or AWS access required) |

## Experimental LLVM artifact cache

This prototype finds reusable artifacts by comparing a build's fingerprints with
fingerprints from previous CI runs. It writes a reuse plan; downloading
archives, restoring stages, and skipping compilation are separate steps.

[prototype_artifact_cache_index.json](prototype_artifact_cache_index.json) is a
prototype, manually maintained artifact cache index. It contains metadata
(artifact names, fingerprints, platforms, CI run information, archive
locations, and checksums), not artifact archives. JSON makes this data easy to
inspect, edit, and test. It is only meant to let us test locally and on CI, and
it is not expected to be maintained over time. A planned dynamic cache database
will replace this file and allow CI to register and query artifacts
automatically.

By default the subcommands use only the `amd-llvm` artifacts. The `amd-llvm`
bundle includes LLVM, COMGR, and HIPCC and has five components: `dbg`, `dev`,
`doc`, `lib`, and `run`. All components of an artifact share one fingerprint and
are reused together. Recording both Linux and Windows gives ten entries: five
per platform. Builds requesting a fingerprint that is not in the index get a
cache miss and must build normally. Other artifacts work the same way, see
[Select artifacts](#select-artifacts).

**Run commands from the TheRock repository root.** The default artifact cache
index is `build_tools/artifact_cache/prototype_artifact_cache_index.json`. Use
`--index` to select another file.

### Entry format

Each entry describes one artifact component on one platform:

```json
{
  "platform": "linux",
  "name": "amd-llvm",
  "component": "lib",
  "target": "generic",
  "fingerprint": "2d156bc1…",
  "provenance": { "repository": "ROCm/TheRock", "run_id": "43936062480" },
  "outputs": [
    {
      "key": "amd-llvm_lib_generic.tar.zst",
      "uri": "https://therock-ci-artifacts.s3.amazonaws.com/43936062480-linux/amd-llvm_lib_generic.tar.zst",
      "digest": "ad988f39…"
    }
  ]
}
```

`digest` is the SHA-256 of the archive. An optional `epoch` records the
`THEROCK_FPRINT_EPOCH` the producer used. The epoch is already part of the
fingerprint; recording it makes a bump visible.

### Update the prototype artifact cache index with the `record` subcommand

Run the `record` subcommand manually with a CI run ID. It reads the run's
fingerprint and checksum files through the artifact backend in
`build_tools/_therock_utils` (the code `fetch_artifacts.py` uses), then saves
their values and artifact locations in the artifact cache index. It never
downloads the archives, and it needs `boto3` for S3 access (`pip install boto3`).
As in `artifact_manager.py`, it asks GitHub for the workflow run to pick the
right bucket; set `GITHUB_TOKEN` or log in with the `gh` CLI to avoid API rate limits.

- A different fingerprint replaces the entry with the same platform, name,
  component, and target. Entries for other platforms and other artifacts remain
  unchanged.
- Matching entries keep their original run ID, artifact location, and checksum.
  The run you select determines the retained fingerprint; the script does not
  compare run dates.
- All components of one artifact must share a fingerprint, so record them
  together. `record` stops with an error otherwise.
- An artifact without a `.fprint` file is skipped and reported, because some
  artifacts have none. An artifact with a `.fprint` but no `.sha256sum` file is
  an error.
- Choose a run whose relevant artifact-producing jobs succeeded and uploaded the
  complete bundle; the script does not check job status.

Update both platforms directly in
`build_tools/artifact_cache/prototype_artifact_cache_index.json`:

```bash
cache_run_id="43936062480"

for cache_platform in linux windows; do
  python build_tools/artifact_cache/artifact_cache.py record \
    --index build_tools/artifact_cache/prototype_artifact_cache_index.json \
    --run-id "$cache_run_id" \
    --platform "$cache_platform" || break
done
```

For a future update, change `cache_run_id` and rerun the block. The loop stops
if a recording command fails. Each successful command updates one platform.

After both succeed, the artifact cache index contains ten entries. Different
fingerprints replace the previous entries; matching fingerprints are reported
as `already indexed` and retain their original run information. Use the
regeneration procedure below when every entry must reference the selected run
ID.

By default, metadata is read from:

```text
https://therock-ci-artifacts.s3.amazonaws.com/<run-id>-<platform>/
```

The files are `<name>_<component>_<target>.fprint` and
`<name>_<component>_<target>.tar.zst.sha256sum` (or `.tar.xz.sha256sum`) for
each selected artifact, for example `amd-llvm_lib_generic.fprint`.

| Option                    | Purpose                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--run-id`                | CI workflow run to read (required).                                                                  |
| `--platform`              | `linux` or `windows` (default: `THEROCK_PLATFORM` or this platform).                                 |
| `--run-github-repo`       | Repository that ran `--run-id`, as `owner/repo` (default: `GITHUB_REPOSITORY`, else `ROCm/TheRock`). |
| `--epoch`                 | Record this `THEROCK_FPRINT_EPOCH` in each new entry.                                                |
| `INCLUDE...`, `--exclude` | Select artifacts, see [Select artifacts](#select-artifacts).                                         |
| `--index`                 | Index file to update (default: the prototype index). A missing file is created.                      |

Set `THEROCK_LOCAL_STAGING_DIR` to read a local directory laid out like the S3
run (`<staging>/<run-id>-<platform>/`) instead of S3. Review changes to the
index:

```bash
git diff -- build_tools/artifact_cache/prototype_artifact_cache_index.json
```

### Regenerate the prototype artifact cache index with the `record` subcommand

Start with an empty artifact cache index so every entry is recorded from the
selected run, including entries whose fingerprints already matched. This example
creates a temporary file containing Linux and Windows entries from run
`43936062480`. The commands use your same local checkout and Python environment;
only the output JSON is temporary:

```bash
artifact_cache_index="$(mktemp /tmp/artifact-cache-index-XXXXXX.json)"
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

Expected: `5 added` for each platform, ten entries in total. Inspect the
temporary artifact cache index at the printed `$artifact_cache_index` path and
compare it with `build_tools/artifact_cache/prototype_artifact_cache_index.json`:

```bash
python -m json.tool "$artifact_cache_index"
```

```bash
diff -u build_tools/artifact_cache/prototype_artifact_cache_index.json "$artifact_cache_index"
```

The comparison does not modify either file; exit code `1` means they differ.
After both recording commands succeed and you review the temporary artifact
cache index, copy it to the repository's file:

```bash
cp "$artifact_cache_index" build_tools/artifact_cache/prototype_artifact_cache_index.json
git diff -- build_tools/artifact_cache/prototype_artifact_cache_index.json
```

For routine updates that preserve already-matching entries, use the direct
update example above.

### Select artifacts

Both subcommands accept include and exclude regular expressions, as in
`fetch_artifacts.py`. They are searched in `<name>_<component>_<target>`, for
example `amd-llvm_lib_generic`. An artifact is used if any include pattern
matches and no exclude pattern does. With no include pattern, only `amd-llvm`
is used.

```bash
# Record base and amd-llvm without debug components.
python build_tools/artifact_cache/artifact_cache.py record \
  --run-id 43936062480 --platform linux amd-llvm base --exclude _dbg_

# Record every artifact in the run.
python build_tools/artifact_cache/artifact_cache.py record \
  --run-id 43936062480 --platform linux .
```

Artifacts that have no fingerprint, such as the split per-GPU-target artifacts,
are reported as skipped.

### Find matching artifacts with the `lookup` subcommand

After configuring TheRock, use the `lookup` subcommand to compare the local
`build/artifacts/*.fprint` files with the artifact cache index. This subcommand
makes no network requests and leaves the artifact cache index unchanged.

```bash
python build_tools/artifact_cache/artifact_cache.py lookup \
  --fingerprints-dir build/artifacts \
  --platform linux \
  --output build/artifact-cache-plan.json
```

Matching uses the artifact name, component, target, fingerprint, and platform.
Build configuration differences must be represented by the configure-time
fingerprint. Only artifacts your build produces are checked: the selected local
`.fprint` files, plus the selected index entries for the same artifacts. An
indexed component with no local fingerprint is a miss, but an artifact with no
local `.fprint` file at all is not part of the plan, even with a broad pattern
such as `.`. Inspect `build/artifact-cache-plan.json`:

```bash
python -m json.tool build/artifact-cache-plan.json
```

- **HIT:** an artifact matches, and its `provenance` and `outputs` are in the plan.
  The plan's `action` is `populate` only when every artifact in the plan hits.
- **MISS:** a fingerprint is missing, invalid, inconsistent, or unmatched, and each
  artifact has a `reason`. The plan's `action` is `build`.
- All components of one artifact (same name and target) are reused together. If
  one component misses, every component of that artifact is a miss
  (`incomplete_group` or `inconsistent_group_fingerprints`), so nothing is reused
  partially. Different artifacts are judged independently.
- Artifacts that have no `.fprint` file, such as split per-target artifacts, never
  appear in the plan, so a plan alone does not say whether a build stage can be
  skipped. Combine it with the list of artifacts the stage produces.
- With no local fingerprints at all, the plan is a miss with the reason
  `no_local_fingerprints`; with fingerprints but no selected ones, the reason is
  `no_artifacts_selected`.
- `--epoch` requires a matching recorded epoch. It is skipped for entries that
  have none.

Both results return exit code `0`; input or I/O errors return `2`. A hit
identifies candidate artifacts. The future population step must verify their
availability and archive checksums before reuse.

### Local test procedure

Use Bash, Python, and `curl` in the **same terminal**, with your Python
environment activated and `boto3` installed. Stop if a command fails. Only steps
3, 4, 6, and 9 need network access. These tests use temporary files and do not
modify the checked-in artifact cache index or `build/artifacts/`. They read only
fingerprint and checksum files, never artifact archives.

#### 1. Run unit tests

Check the script's behavior using temporary local data and simulated CI runs.

```bash
python build_tools/artifact_cache/tests/artifact_cache_test.py -v
```

Expected: `OK`. These tests include replacement of older fingerprints,
preservation of other platforms and artifacts, artifact selection, and failure
handling. They use local fixtures and no network.

#### 2. Prepare temporary files

Create an isolated test workspace with an empty artifact cache index and empty
fingerprint directories.

```bash
cache_test_dir="$(mktemp -d /tmp/artifact-cache-test-XXXXXX)"
cache_run_id="43936062480"
cache_source_root="https://therock-ci-artifacts.s3.amazonaws.com/${cache_run_id}-linux"

mkdir "$cache_test_dir/empty-fingerprints" \
      "$cache_test_dir/matching-fingerprints" \
      "$cache_test_dir/changed-fingerprints" \
      "$cache_test_dir/incomplete-fingerprints" \
      "$cache_test_dir/base-fingerprints"

printf '%s\n' '{"schema_version": 1, "entries": []}' \
  > "$cache_test_dir/artifact_cache_index.json"
printf 'Test directory: %s\n' "$cache_test_dir"
```

Expected: `$cache_test_dir/artifact_cache_index.json` has no entries, and all five
fingerprint directories are empty. The printed path identifies the test files
used below.

#### 3. Add a CI run with the `record` subcommand

Verify that the `record` subcommand reads the CI run's metadata and adds the five
`amd-llvm` components to the artifact cache index.

```bash
python build_tools/artifact_cache/artifact_cache.py record \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --run-id "$cache_run_id" \
  --platform linux

python -m json.tool "$cache_test_dir/artifact_cache_index.json"
```

Expected: `Run 43936062480 (linux): 5 added, 0 replaced, 0 already indexed`.

The `record` subcommand reads metadata from `$cache_source_root` and writes five
entries to `$cache_test_dir/artifact_cache_index.json`. It uses its own
temporary download directory, so the fingerprint directories remain empty.

#### 4. Repeat the `record` subcommand

Verify that running the `record` subcommand again leaves the artifact cache
index unchanged.

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

Expected: `0 added, 0 replaced, 5 already indexed; index unchanged`; `cmp`
produces no output and returns `0`. Metadata is read again, but
`$cache_test_dir/artifact_cache_index.json` is unchanged.

#### 5. Test an empty fingerprints directory with the `lookup` subcommand

Verify that the `lookup` subcommand returns a miss when there are no local
fingerprint files.

```bash
python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/empty-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/missing-plan.json"

python -m json.tool "$cache_test_dir/missing-plan.json"
```

Expected: `MISS (0/0 artifacts)` and a line naming the empty directory with the
reason `no_local_fingerprints`. The input directory
`$cache_test_dir/empty-fingerprints` is intentionally empty, and the build
produces nothing to check, so no artifacts are listed.

#### 6. Test matching fingerprints with the `lookup` subcommand

Verify that the `lookup` subcommand returns a hit when all five requested
fingerprints match the artifact cache index.

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

Expected: `HIT (5/5 artifacts)` with `matching_index_entry` and `Action: populate`.

Only `curl` accesses the network; the `lookup` subcommand reads the downloaded
files locally. This controlled hit tests matching against saved metadata. It
does not prove that your local TheRock configuration produces the same
fingerprints; use the earlier example with `build/artifacts/` for that check.

#### 7. Test changed fingerprints with the `lookup` subcommand

Verify that the `lookup` subcommand returns a miss when the requested
fingerprints do not match any saved entries.

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

Expected: `MISS (0/5 artifacts)` with `no_matching_index_entry`.

#### 8. Test an incomplete artifact with the `lookup` subcommand

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

Expected: `MISS (0/5 artifacts)`. The debug fingerprint is missing
(`missing_expected_fingerprint`). The other four fingerprints match, but they
are reported as misses with `incomplete_group`: partial reuse of one artifact is
not allowed.

#### 9. Test another artifact with include and exclude patterns

Verify that the subcommands work for artifacts other than `amd-llvm`. This
records the `base` artifact from the same run, excluding its debug component,
and looks it up with the same patterns.

```bash
python build_tools/artifact_cache/artifact_cache.py record \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --run-id "$cache_run_id" \
  --platform linux \
  base --exclude _dbg_

(
  set -e
  for component in dev doc lib run test; do
    filename="base_${component}_generic.fprint"
    curl --fail --silent --show-error \
      "$cache_source_root/$filename" \
      --output "$cache_test_dir/base-fingerprints/$filename"
  done
)

python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/base-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/base-plan.json" \
  base --exclude _dbg_

python -m json.tool "$cache_test_dir/base-plan.json"
```

Expected: `5 added` for the `record` command, then `HIT (5/5 artifacts)` for
`base_dev`, `base_doc`, `base_lib`, `base_run`, and `base_test`. The `amd-llvm`
entries stay in the index.

The index now holds both `amd-llvm` and `base`. Look up the `amd-llvm`
fingerprints again with the pattern `.`, which selects every artifact:

```bash
python build_tools/artifact_cache/artifact_cache.py lookup \
  --index "$cache_test_dir/artifact_cache_index.json" \
  --fingerprints-dir "$cache_test_dir/matching-fingerprints" \
  --platform linux \
  --output "$cache_test_dir/broad-plan.json" \
  .
```

Expected: `HIT (5/5 artifacts)` and `Action: populate`, with only `amd-llvm`
artifacts listed. The index's `base` entries are ignored, because this build has
no `base` fingerprints.

#### 10. Verify the saved plans

Check each plan's status, action, and artifact count:

```bash
python - "$cache_test_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "missing-plan.json": ("miss", "build", 0),
    "hit-plan.json": ("hit", "populate", 5),
    "changed-plan.json": ("miss", "build", 5),
    "incomplete-plan.json": ("miss", "build", 5),
    "base-plan.json": ("hit", "populate", 5),
    "broad-plan.json": ("hit", "populate", 5),
}
for filename, wanted in expected.items():
    path = root / filename
    plan = json.loads(path.read_text())
    actual = (plan["status"], plan["action"], len(plan["artifacts"]))
    assert actual == wanted, f"{path}: expected {wanted}, got {actual}"
    print(f"PASS {path}: status={actual[0]}, action={actual[1]}, artifacts={actual[2]}")
PY
```

Expected: six `PASS` lines.

| Plan file                              | Status | Action     | Artifacts |
| -------------------------------------- | ------ | ---------- | --------- |
| `$cache_test_dir/missing-plan.json`    | `miss` | `build`    | 0         |
| `$cache_test_dir/hit-plan.json`        | `hit`  | `populate` | 5         |
| `$cache_test_dir/changed-plan.json`    | `miss` | `build`    | 5         |
| `$cache_test_dir/incomplete-plan.json` | `miss` | `build`    | 5         |
| `$cache_test_dir/base-plan.json`       | `hit`  | `populate` | 5         |
| `$cache_test_dir/broad-plan.json`      | `hit`  | `populate` | 5         |

Test files remain under `$cache_test_dir`. No compilation or artifact archive
downloads are performed.
