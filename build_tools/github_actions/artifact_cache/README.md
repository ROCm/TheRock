# Relational fingerprint artifact cache

This adapts the local workflow, planner, history ingestion, replay, policies and
CLI to the revision-6 SQLite schema. It does not create the original
`manifests` table. All commands below run from the TheRock repository root.

## Data contract

The default CLI reuse unit is **one artifact/target bundle**, containing its
requested components. A bundle lookup uses **epoch + bundle identity + raw CMake
fingerprint + the sorted component list**, for example:

```python
ArtifactKey(
    artifact_identity="rpp/bundle/generic",
    fingerprint="8fd6c8a0bf4a27b027c9b60dc03a2c4c3dd1f5a97eb800341f408a95e23bfb93",
    epoch=1,
    components=("rpp_dbg_generic", "rpp_dev_generic", "rpp_doc_generic",
                "rpp_lib_generic", "rpp_test_generic"),
)
```

The raw fingerprint is unchanged. Each member must supply that same valid
64-hex value. Missing, invalid or differing sidecars prevent reuse/publication
of the whole requested bundle. Different artifact names or targets never merge,
even if their fingerprints match. Split outputs without sidecars remain skipped.

There is no namespace. The database uses `(epoch, artifact_id, fingerprint,
components_json)` as its unique key. `components_json` is the canonical sorted
JSON array of full component directory identities. A request for `lib` alone
cannot hit a production for `lib + dev`, or vice versa. Component selection is
part of the contract even if the raw CMake fingerprint remains unchanged.

Existing component keys have `components_json='[]'` and keep their original
identity, key ID and productions. `--granularity component` selects the previous
behavior. Package `cache`, `keys`, `plan`, `publish` and `restore` CLIs default to
bundle mode. The lower-level Python APIs and `build_tools/github_actions/artifact_cache/collector.py`
entry points keep their component default for compatibility; Python callers
select bundles with `granularity="bundle"`. Commits/run labels are provenance.

`CACHE_EPOCH` in `cache_key.py` sets the default epoch. Increase it deliberately
when existing entries must no longer satisfy new requests, never per build or
experiment. Stored keys keep their original epochs. Fingerprints and payload
integrity digests do not change when the epoch changes. Legacy `e1:release:...`
strings remain historical simulation keys, not relational lookup keys.

Use separate databases to isolate experiments or incompatible build environments.
The fingerprint must cover all input differences that affect substitutability;
removing namespace does not add missing platform/toolchain inputs to CMake.

| Existing table/view | Role |
| --- | --- |
| `artifacts` | Full output identities |
| `artifact_keys` | Epoch + raw CMake fingerprint + exact component membership per identity |
| `prototype_runs` | Collection/build provenance; imported history in typed JSON |
| `artifact_observations` | Fingerprints and metadata lookup decisions per collection |
| `cache_entries` | Pending, ready or invalid productions and their object locations |
| `artifact_files` | Expected immutable objects, SHA-256 digests and sizes |
| `reusable_cache_entries` | Eligible metadata candidates; physical verification remains required |

Opening a database validates it and never creates or migrates tables. A Python
`LocalManifest` represents joined rows; no `manifests` table is needed.

### One-time bundle membership migration

Stop cache commands before migrating. After installing the revision-6 code:

```bash
python3 -m build_tools.github_actions.artifact_cache.migrate_schema \
  --db /home/master/erman/cache.db \
  --backup /home/master/erman/cache-before-bundles.db
```

The backup filename must be new. SQLite's backup API captures the old database
before mutation. The migration supports revisions 3, 4 and 5 and installs
revision 6. Existing component rows gain `components_json='[]'`; their cache
entry IDs, file inventories, payload directories, timestamps and producer
records remain intact. They are not relabeled as verified bundles. The first
bundle run builds incrementally and publishes separate complete bundle entries.

For older revisions 3/4, the migration also removes `key_namespace`, merging
equal keys onto the smallest key ID and remapping entry/observation references.
Version-3 keys gain fixed epoch 1; existing epochs remain unchanged. The backup
retains the old schema and values. No namespace is reintroduced.

The migration verifies foreign keys and integrity before committing. Failures
roll back changes and retain any backup already made. Unknown extensions that
refer to the removed column are rejected, never silently rewritten. Repeating
the command against revision 6 validates it and makes no new backup.

For a **new** database use `build_tools/therock_fingerprint_cache.sql`. Never run
this fresh-schema SQL against an existing database. Tests use temporary databases:

```bash
THEROCK_CACHE_SCHEMA="$PWD/build_tools/therock_fingerprint_cache.sql" \
  python3 -m unittest discover -s build_tools/github_actions/tests -p 'artifact_cache_*_test.py' -v
```

Update scripts by removing `--namespace` and `namespace=`. Old arguments are
rejected rather than silently ignored. `ArtifactKey` takes identity and raw
fingerprint, with epoch as a keyword argument. Collector JSON is now
`report_version=3` because `key_namespace` was removed; bundle-aware cache/publication/restore JSON is `report_version=2`, with
`granularity` and per-unit `components`. Summary field names are unchanged but
count bundles in bundle mode. Component observation collection stays version 3.

## Local one-command workflow

Use the same command for cold and warm runs:

```bash
python3 -m build_tools.github_actions.artifact_cache.local_cache cache \
  --build build \
  --db /home/master/erman/cache.db \
  --cache /home/master/erman/cache-test/managed-objects \
  --variant release
```

The command:

1. Locks the build directory, checks clean tracked source/submodules and
   reconfigures to refresh fingerprints.
2. Reads an explicit `--expected` component list, or queries configured Ninja
   artifact-manifest outputs and combines them with observed directories and
   sidecars. Groups by artifact name and target, validates member fingerprints,
   and constructs keys carrying the exact component set.
3. Makes a batched metadata lookup, then verifies cached archives and their
   complete unpacked inventories in temporary storage.
4. If keys are missing, runs the normal incremental CMake build through the
   managed publisher. It checks inputs again, packages all requested components of each missing bundle
   into one immutable generation, and records the inventory and ready entries last.
5. Reports `verified_cached`, `stored`, and `uncacheable` independently.

By default, the cache and publication CLIs print **one artifact report at the end** on
stderr, after the operation and its final input checks succeed. Each artifact
appears once. The `local_cache cache` CLI shows the first configuration's normal
CMake output, followed by short operation messages for cache verification,
build start/completion, publication, and final checks. Build output streams
normally. Per-artifact progress messages such as `MISS` and `STORING` are not
printed; their final outcomes appear at the end.

Later explicit configure passes save their combined stdout/stderr to separate
`logs/artifact-cache/configure-*.log` files under the selected build directory.
Successful output from those repeated passes, including warnings, stays in the
logs. A failed logged configure prints its log path and complete diagnostics,
then stops with a failure status. A failure in the first configure or build is
already visible in the streamed output. The configure passes and input checks
still run; this changes terminal verbosity, not configure time. CMake output
triggered internally by the build tool also remains visible.

The CLI reports `BUILD NOT RUN` when there are no cacheable misses. Artifacts
without usable fingerprints still appear in the final report; this message does
not claim their outputs were built or verified. A cold cache uses the existing
build directory incrementally: an up-to-date build may only merge
`compile_commands.json` rather than compile sources again.

The lines remain visible in terminal and CI job logs when stdout is redirected
to a JSON report. A mixed run could end with:

```text
[artifact-cache] REUSED amd-llvm/bundle/generic (components=dbg,dev,doc,lib,run; matching cached files verified)
[artifact-cache] NO_FINGERPRINT blas/bundle/gfx1100 (components=lib,test; blas_lib_gfx1100: expected fingerprint file is missing; blas_test_gfx1100: expected fingerprint file is missing; cache restore and storage skipped)
[artifact-cache] MISS -> STORED core-amdsmi/bundle/generic (components=dev,lib; no usable cache entry; saved after successful build)
[artifact-cache] SUMMARY stored=1 reused=1 cache_skipped=1 units=bundles
```

- `MISS -> STORED`: no eligible verified cached output was available; the managed
  build and consistency checks succeeded, and publication committed.
- `REUSED`: an existing cache entry's archive and inventory were verified in
  temporary storage. This is the display name for the JSON `verified_cached`
  outcome, not a claim that outputs were installed into the consumer build or
  that a CI stage was skipped.
- `NO_FINGERPRINT`: the expected fingerprint file is missing, so cache lookup,
  restoration and storage are skipped for this artifact. Its normal build is
  still possible.
- `CACHE_SKIPPED`: another fingerprint problem, such as invalid contents or an
  unreadable file, prevents cache use. The original diagnostic is included.
  Neither label claims that the artifact was built, restored or stored.

Direct publication reports `STORED` without a preceding `MISS`, because it does
not perform the cache command's lookup. Explicit restore reports `RESTORED` or
`NEEDS_BUILD` after its final checks. `RESTORED` means files were extracted into
the requested destination. If an operation fails, its error is printed instead
of a successful artifact report; failed configure and build diagnostics remain
visible.

Local cache outcome names and exit statuses are unchanged. The readable summary uses
`reused` and `cache_skipped`, while JSON retains `verified_cached` and
`uncacheable`. Epoch, entry IDs, verification flags and whether a build ran
remain in JSON. Omit `--expected` to include declared Ninja artifact outputs as well as observed
directories/sidecars, including those lacking fingerprints. Declared components
remain expected even if both their directory and sidecar are absent. Use `> report.json` to save
JSON while keeping logs visible; do not merge stderr with stdout using `2>&1`
when the output file must contain JSON only. Workflows must invoke these CLIs
for the lines to appear; plain `cmake --build` does not call this cache wrapper.
This change does not wire the cache into CI workflows or skip individual build
targets. Python callers can still opt into internal events using the optional
`progress` callback. The cache CLI attaches a filter that prints operation
phases only. Python callers can opt into the first configure's live output with
`cache_build(..., show_configure=True)`. Other entry points keep their previous
configure-output behavior.

This fixes the configure-only publication problem: old export directories
cannot be stored under newly configured keys without a successful managed build.
A failed build does not create ready entries. A warm run does not launch a build.
**This command is not connected to CMake target skipping.** Verified hits do not
suppress targets when the incremental build runs for other misses.

For automatic bundle mode, configured single-config Ninja outputs establish
expected component membership. An observed group without a complete declared
member set is reported as `CACHE_SKIPPED` with a request to supply `--expected`.
Other generators require an explicit expected list for cacheable bundles.
Component mode retains observed-only discovery. These inventories do not prove
that a CI stage is ready to skip. Missing/invalid sidecars are reported without
assuming all are split artifacts. `--variant` checks `CMAKE_BUILD_TYPE`; it is
not a second arbitrary key. Multi-config builds need a separate build contract.

For an explicit selection, use an expected-output JSON file outside the source:

```json
{
  "schema_version": 1,
  "artifacts": [
    {
      "artifact_identity": "core-amdsmi_lib_generic",
      "fingerprint_path": "core-amdsmi_lib_generic.fprint"
    },
    {
      "artifact_identity": "core-amdsmi_dev_generic",
      "fingerprint_path": "core-amdsmi_dev_generic.fprint"
    }
  ]
}
```

Pass `--expected /home/master/erman/amdsmi-expected.json` to restrict lookup and
publication. In this example the requested bundle is exactly `dev + lib`;
other components are outside this explicit selection. The key records that
selection so it cannot be mistaken for a larger bundle. The managed build still runs the default incremental build target;
it does not guess a safe partial-build dependency closure.

Inspect keys without publication:

```bash
python3 -m build_tools.github_actions.artifact_cache.local_cache keys --build build
python3 -m build_tools.github_actions.artifact_cache stats --db /home/master/erman/cache.db
```

`keys` reconfigures, so it writes build metadata. It does not write the database
or cache objects. The cache directory must be separate from source/build trees.
Tracked source changes, including changes to this package, must be committed
locally before managed build/restore commands. They need not be pushed.

## Cold and warm tests without namespaces

A cold cache starts with a new database and a new object directory. Its normal
CMake build can still be incremental. The warm run uses that same database,
object directory, build and inputs. A new log filename alone does not make a
cache cold. The update package includes `cold-warm.sh` to initialize both paths
and run the commands without relying on shell variables from an earlier session.
It selects configured/observed bundles, checks cold/warm member lists, and
writes separate detailed logs and JSON reports.

Several immutable productions can share one key, including productions migrated
from old experiment namespaces. Local cache verification and restore select
candidates under the supplied `--cache` / `--cache-root` directory **before**
ranking them by age. A newer candidate stored elsewhere does not hide an older
local one, and is not retired. The path comparison uses a directory boundary;
physical verification still rejects symlinks and invalid contents. A store with
no matching candidates returns a miss, even when another store has the key.
Storage root is a candidate-location filter, not another part of `ArtifactKey`.
Metadata-only lookup/collection/planning without a store root can inspect
candidates throughout the database and still make no physical-reuse claim.

## Explicit publication and restoration

```bash
python3 -m build_tools.github_actions.artifact_cache publish \
  --db /home/master/erman/cache.db --build build \
  --expected /home/master/erman/amdsmi-expected.json \
  --cache /home/master/erman/cache-test/managed-objects

python3 -m build_tools.github_actions.artifact_cache restore \
  --db /home/master/erman/cache.db --build build \
  --expected /home/master/erman/amdsmi-expected.json \
  --cache /home/master/erman/cache-test/managed-objects \
  --destination /home/master/erman/cache-test/restored-amdsmi
```

Restore requires a new destination. It verifies database digests and sizes,
archive members, file contents, modes and safe symlink targets. Missing or corrupt
productions are retired, and an older eligible production is tried. The result
is a real component export tree. No `.prebuilt` markers are written and no CI job
is skipped. Exit status is 1 for unresolved restore misses, 2 for an error.
The collector provides `collect`, `build-publish`, and `restore` commands.
These default to component granularity.

### Stored bundle format

One ready bundle entry points to one new generation containing `payload.tar`
and `inventory.json`. The archive contains each member under its original
component directory name. Inventory schema 2 records the exact component list
and every nested file/directory/symlink with sizes, SHA-256 and modes. Each
component must include `artifact_manifest.txt`, even if otherwise empty.
No component can supply another component's files or escaping symlinks.

Publication verifies the complete archive before committing one ready entry.
The producing run has component fingerprint observations plus a bundle
observation with no synthetic sidecar path. A missing member prevents that
bundle becoming ready. Restoring a bundle verifies all members before exposing
it in the new destination, under the original component names. A corrupt member
rejects the entire production; an older complete production may be tried.

Old inventory schema-1 component entries remain readable in component mode.
There is no automatic conversion of old component productions into a bundle,
which could otherwise mix unrelated producing runs or incomplete selections.

## Optional detailed diagnostics

Add `--verbose` to see cache events while an operation runs. Add `--log-file`
to retain those events in a **new** UTF-8 text file. The file receives detailed
events even without `--verbose`; the terminal then retains its normal compact
phases and final artifact report. The flags are supported by `local_cache`, the
package CLI, and `build_tools/github_actions/artifact_cache/collector.py` subcommands.

Keep the database, object-store directory and expected list from your
existing experiment. Changing the log location does not make a cache cold.
For example, when the earlier `THEROCK_CACHE_RUN` experiment variables are set:

```bash
cd /home/master/erman/TheRock
mkdir -p /home/master/erman/cache-test
THEROCK_DIAGNOSTIC_RUN="$(mktemp -d /home/master/erman/cache-test/logging-XXXXXX)"

python3 -m build_tools.github_actions.artifact_cache.local_cache cache \
  --build build \
  --db /home/master/erman/cache.db \
  --cache "${THEROCK_CACHE_RUN:?Use the existing experiment directory}/objects" \
  --variant release \
  --verbose \
  --log-file "$THEROCK_DIAGNOSTIC_RUN/cache.log" \
  > "$THEROCK_DIAGNOSTIC_RUN/report.json"
```

Omitting `--expected` combines declared Ninja outputs and observed directories/sidecars.
Retain `--expected /path/to/expected.json` if the experiment is intentionally
restricted. Use a different log filename for the next invocation; an existing
path is rejected before cache work, so diagnostics cannot overwrite a database,
archive or previous log.

Each diagnostic has a UTC timestamp, seconds since the command started, an
invocation-local sequence number, an event name and named fields. Long package,
copy, inventory-verification and hashing passes emit byte progress at most once
per five seconds while processing data. Start/end events include durations.
This adds no archive reads or extra hashing passes. Diagnostics are inert when
neither flag is supplied.

| Events | What they explain |
| --- | --- |
| `RUN_START`, `RUN_END`, `OPERATION_FAILED` | Selected paths/operation and final exit status |
| `BUILD_LOCK_*`, `SOURCE_CHECK_*`, `SOURCE_STATE` | Managed lock, clean tracked source checks, repository count and root commit |
| `COMMAND*`, `CONFIGURE_LOG` | Executed argument list, elapsed time/failure and the retained configure-log path |
| `DISCOVERY`, `FINGERPRINT`, `KEY`, `BUNDLE_KEY`, `BUNDLE_SKIPPED`, `BUNDLE_OUTPUTS_DECLARED` | Explicit vs observed inventory, sidecar path, validity/reason, raw hash and key scope |
| `DB_LOOKUP`, `DB_LOOKUP_RESULT`, `CACHE_DECISION` | Metadata lookup size, effective read policy, selected candidate or miss; no physical reuse claim |
| `CANDIDATE_CHECK`, `OBJECT_CHECK`, `OBJECT_VERIFIED`, `INVENTORY_VERIFIED` | Entry ID, actual object paths, expected and verified SHA-256/sizes, inventory member counts |
| `CANDIDATE_REJECTED`, `ENTRY_INVALIDATED`, `CANDIDATE_CHANGED` | Corruption or changed eligibility and retry of another candidate |
| `BUILD_DECISION`, `BUILD_INPUTS_VERIFIED`, `FINAL_KEYS_VERIFIED` | Why a managed build was invoked and when stable inputs were established |
| `EXPORT_INVENTORY`, `ARCHIVE_WRITE`, `INVENTORY_WRITTEN`, `OBJECT_SAVED` | Source export, new generation path, archive/inventory counts and saved object hashes/sizes |
| `PRODUCER_RECORD_COMMITTED`, `DB_TRANSACTION_*`, `ENTRY_COMMITTED` | Producer observation commit and separate ready-entry transaction; rollback is explicit |
| `PLAN_SCOPE`, `PLAN_ARTIFACT` | Topology/changed-path scope, stage mapping and metadata-only decisions |
| `RESTORE_DESTINATION_READY` | Explicit restore has installed verified exports at the requested new destination |

`OBJECT_SAVED` does not mean that a ready database entry exists.
`ENTRY_COMMITTED` reports the actual database commit; a later final input check
can still fail the overall cache command. In that case committed entries remain,
the failure is logged and no final success report is printed. This distinction
is deliberate. A cache warm check uses temporary extraction and logs that it is
not installing the artifact into a consumer build.

The usual `REUSED`, `MISS -> STORED`, `NO_FINGERPRINT` / `CACHE_SKIPPED` and summary
lines still appear once, at the end after successful final checks. Verbose traces
can mention an artifact repeatedly as it passes through different checks; they
are diagnostic events, not repeated final outcomes. JSON stdout stays unchanged.

The text log contains **cache/controller events and final report lines**. It does
not duplicate raw CMake/Ninja output: first-configuration and build output still
stream to stderr, while later explicit configure output stays in the existing
`build/logs/artifact-cache/configure-*.log` files. The detailed log identifies
those files. To retain the entire console transcript too, redirect stderr to a
separate file or use the CI job log. No environment or artifact contents are
dumped; logs do contain source/cache paths, fingerprints, digests and entry IDs.

If a log becomes unwritable after work starts, a warning is printed and cache
work continues. Starting with an unusable/existing log path fails before cache
work. The logging options add no logger thread, network service or trust decision. The existing clean tracked-tree requirement
still applies after these Python edits; commit the installation on the local
experiment branch before a real managed-build run.

## Planning and read policy

```bash
python3 -m build_tools.github_actions.artifact_cache plan \
  --db /home/master/erman/cache.db --build build \
  --topology BUILD_TOPOLOGY.toml

python3 -m build_tools.github_actions.artifact_cache lookup \
  --db /home/master/erman/cache.db \
  --artifact core-amdsmi/bundle/generic --fingerprint <raw-64-hex-fingerprint> \
  --components core-amdsmi_dev_generic,core-amdsmi_lib_generic
```

The planner consumes current CMake fingerprints; dependency fingerprint
propagation stays in CMake. `TopologyAdapter` uses TheRock's existing parser to
map component or bundle identities to stages. Unknown identities remain visible under
`unmapped`. Optional `--changed path1,path2` conservatively forces affected
units to rebuild. Optional `--targets` filters emitted family suffixes and
rejects requested families with no configured outputs.

Plans report **candidate**, **miss**, or **uncacheable**, not verified hits.
`all_requested_candidates` is useful for planning, but `can_skip` remains false:
complete stage inventories, restoration and scheduler wiring are needed before
skipping jobs. Test selection is independent of cache availability.

CLI policy defaults are suitable for a local cache: untrusted-or-better,
build-passed, ready entries. `--min-trust trusted`,
`--min-validation tests_passed` and `--created-before <zoned ISO time>` can
restrict reads. SQL applies policy before choosing the newest candidate, so a
newer ineligible production cannot hide an older eligible one.

Absent trust provenance is untrusted. Managed publication defaults to untrusted;
`publish --producer-trust trusted` explicitly records a **local policy label**.
It is not authentication or a GitHub attestation. SQLite write access remains
the trust boundary. No local command invents a tests-passed result. Pending,
failed and incomplete productions remain ineligible even with a relaxed policy.

## History and replay

History trust is restricted to `push` and `schedule` events on exactly `main`.
Release branches, feature branches, pull requests and manual (`workflow_dispatch`)
runs are untrusted. A supplied head repository that differs from the expected
repository also makes the run untrusted. `--require-trusted` skips those runs
before listing their artifacts. Without that option they may still be recorded
as untrusted evidence. This metadata classification does not authenticate the
producer or verify GitHub branch protections, and it does not change local
publication's untrusted default. Existing history records are not automatically
relabeled when the rule changes.

```bash
# Offline import: no GitHub or S3 access.
python3 -m build_tools.github_actions.artifact_cache ingest \
  --db /home/master/erman/cache.db --runs-json /path/to/runs.json \
  --repo ROCm/TheRock --platform linux --variant release

# Optional live history import; uses existing TheRock GitHub/S3 helpers and auth.
python3 -m build_tools.github_actions.artifact_cache ingest \
  --db /home/master/erman/cache.db --max-runs 20 --require-trusted

python3 -m build_tools.github_actions.artifact_cache replay \
  --db /home/master/erman/cache.db --repo ROCm/TheRock
```

Offline JSON is an array of GitHub-shaped runs: `id`, `head_sha`, `head_branch`,
`event`, `status`, `conclusion`, `created_at`/`run_started_at`, and
`completed_at`/`updated_at`. Each run has `artifacts`, an array of objects with
`key` (archive filename), `uri`, and optional `digest` and `size_bytes`.
An optional `source_fingerprints` object maps full component identities to real
64-hex fingerprints; otherwise a marked commit-derived fallback is used only
for the estimate. Platform/variant arguments describe the imported run set;
import different environments separately.

History is stored in `prototype_runs.parameters_json.cache_history`, with
`configure_state=not_run`. Identical imports are idempotent. **History imports
never create artifact keys, ready cache entries or file inventories.** An S3
listing is not proof of completeness, integrity or successful local production.
Historical URLs become usable payloads only through a future verified remote
importer, not by relabeling the history evidence as ready.

Replay sorts by run start and only considers producers whose outputs became
available before that start. It respects repository/platform/variant and policy,
counts only recorded outputs, and compares against an equally scoped same-commit
baseline. Reports describe potential reuse, not measured build time saved or
verified complete stages. Commit fallback cannot demonstrate cross-commit reuse;
real per-output fingerprints can. Tests exercise both modes and overlapping runs.

## Maintenance and fixtures

```bash
# Retire metadata; neither command deletes payload files or observation history.
python3 -m build_tools.github_actions.artifact_cache evict \
  --db /home/master/erman/cache.db --before 2026-01-01T00:00:00Z
python3 -m build_tools.github_actions.artifact_cache evict \
  --db /home/master/erman/cache.db --missing
```

`synth --out /new/fixture-build` creates an isolated miniature Git/CMake project.
It requires Git, CMake and Ninja, but no compiler or GPU. The fixture's five
component payloads have content-derived configure-time keys, plus one deliberately
fingerprint-less component. `synth --out /same/fixture-build --change blas`
changes and commits only that private fixture's BLAS input, then reconfigures.
Run `local_cache cache --build /same/fixture-build --db <existing-test-db>
--cache <separate-storage>` to exercise actual managed publication. `synth`
refuses ordinary existing directories. This fixture does not test TheRock's
source fingerprint implementation or GPU builds.

## Files and integration boundaries

| File | Role after adaptation |
| --- | --- |
| `cache_db.py` | Schema validation, batched full-key reads, policy, provenance, publication metadata, retirement and history evidence |
| `manifest.py` | `ArtifactKey`, `ArtifactFile`, `LocalManifest`; legacy model types retained for isolated simulations |
| `cache_key.py` | Marked fallback/history key helpers; not the real production key contract |
| `inventory.py` | Current output inventory, component parsing and `TopologyAdapter` |
| `diagnostics.py`, `reporting.py` | Optional timestamped diagnostics and the final artifact report; stdout remains JSON |
| `planner.py` | Component candidates and stage mapping; no unverified stage skipping |
| `local_store.py` | Immutable archives/inventories and safe extraction; compatibility loose-object storage |
| `publish.py` | Managed configure/build/input checks before ready publication |
| `restore.py` | Physical verification, fallback from bad candidates and extraction |
| `local_cache.py` | One-command cold/warm workflow, key inspection and isolated fixture |
| `history_ingest.py` | Offline/live history evidence, with no fabricated ready outputs |
| `replay.py` | Completion-aware reuse estimates and same-commit comparison |
| `cli.py`, `__main__.py`, `__init__.py` | Root/module commands and lazy Python API |
| `build_tools/github_actions/artifact_cache/collector.py` | Existing collection and explicit build/restore entry points |

The original `topology_view.py` is not required by this path; the new adapter
uses the real topology parser directly. Legacy APIs that accepted derived
string keys or arbitrary output lists are intentionally not interchangeable
with safe production operations. `publish_build_outputs` now forwards to the
managed build API. Tests for the original standalone `manifests` database must
be ported to this contract; they are not this integration's validation suite.

This is local publication/restoration plus CI planning/history tooling. It does
not implement GitHub Actions job skipping, remote object publication/restoration,
ClickHouse, split-artifact fingerprints, cross-machine compatibility or test
selection. Those were not completed in the original package either.

## Tests

```bash
THEROCK_CACHE_SCHEMA="$PWD/build_tools/therock_fingerprint_cache.sql" \
  python3 -m unittest discover -s build_tools/github_actions/tests -p 'artifact_cache_*_test.py' -v
```

The suite uses the supplied schema in temporary databases. Managed-workflow tests
use miniature Git/CMake projects; they need Git, CMake and Ninja and otherwise
skip explicitly. No tests touch your real DB, build or object store.

## Cache tests

All 16 cache test files live in `build_tools/github_actions/tests/` and use the
`artifact_cache_*_test.py` naming pattern. `_artifact_cache_fixtures.py` supplies
shared test data and is not itself a test suite. The collector fixture is in
`artifact_cache_collector_test.py`; the SQL schema remains at
`build_tools/therock_fingerprint_cache.sql`.

The DB contract and relational-schema tests are consolidated in
`artifact_cache_db_test.py`. Redundant mixed hit/miss and duplicate-key tests
are covered there once. The incomplete-publication check lives in
`artifact_cache_publish_restore_test.py`, including assertions that no entry
or file rows remain. Managed build, command-line workflow and diagnostic tests
retain their separate coverage because they exercise different boundaries.

Run all cache tests from the repository root:

```bash
python3 -m unittest discover -s build_tools/github_actions/tests -p 'artifact_cache_*_test.py' -v
```

The tests create isolated temporary databases. No real cache database is used.
The miniature build tests require CMake, Ninja and Git; they do not build ROCm
or require a GPU. Old cache-test copies should not remain in `build_tools/tests`
or under `fingerprint_cache_*_test.py` names, to avoid duplicate discovery.
Unrelated test files stay in their original locations.
