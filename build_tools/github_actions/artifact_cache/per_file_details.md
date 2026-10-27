# Files explained — artifact cache

This guide describes the current package in
`build_tools/github_actions/artifact_cache/`, its relational SQLite schema,
and its tests. Commands run from TheRock's repository root.

The implemented system supports local cache inspection, managed build and
publication, verified restoration, dependency-impact reporting, and historical
reuse estimates. It does not yet integrate cached outputs with CMake's prebuilt
markers or a CI scheduler to skip individual targets or stages.

## The lookup contract

The Python lookup type is **`ArtifactKey`**. The package CLIs default to bundle
mode: one artifact and target, containing an exact requested set of components.
For example, the `dbg`, `dev`, `doc`, `lib`, and `run` outputs of AMD SMI can form
`core-amdsmi/bundle/generic`.

An `ArtifactKey` contains:

| Field | Meaning |
| --- | --- |
| `artifact_identity` | Full component identity, or an `artifact/bundle/target` identity. |
| `fingerprint` | The actual lowercase, 64-hex value emitted by CMake in `.fprint` files. |
| `epoch` | Cache compatibility version, normally supplied by `CACHE_EPOCH`. |
| `components` | Canonical sorted tuple of full component identities for a bundle; empty for a component key. |

The database stores identities in `artifacts` and versions in `artifact_keys`.
Its uniqueness constraint is:

```sql
UNIQUE (epoch, artifact_id, fingerprint, components_json)
```

There is **no namespace**. Use separate databases to isolate experiments.
Different artifact names, targets, epochs, fingerprints, or component sets
produce different lookup identities. In bundle mode, every requested component
must have the same valid raw fingerprint. Grouping does not create a replacement
fingerprint or a separate Python key type.

Repository commits are recorded as producer provenance. CMake may also use a
repository commit when computing the raw fingerprint; the Python lookup layer
does not remove that dependency. Reducing broad invalidation from a monorepo's
`HEAD` is deferred work.

## How the active flow fits together

1. **Configure and discover:** `local_cache.py` refreshes CMake metadata.
   `inventory.py`, `collector.py`, and `bundles.py` identify requested outputs,
   read their fingerprints, and form `ArtifactKey` objects.
2. **Look up and verify:** `cache_db.py` returns eligible metadata candidates.
   `restore.py` and `local_store.py` verify the saved bytes and archive contents.
   A database match alone is not verified reuse.
3. **Build and publish misses:** when a cacheable request has no verified
   candidate, `publish.py` runs the default incremental build and rechecks its
   inputs. `local_store.py` saves artifact copies first; `collector.py` records
   producer evidence; `cache_db.py` commits complete ready entries last.
4. **Report:** `reporting.py` emits one final artifact report. `diagnostics.py`
   optionally adds detailed events without mixing them into JSON output.

The `cache` command verifies existing objects in temporary storage. It does not
install them into the current build. If there are no cacheable misses, it does
not invoke the build. Outputs without usable fingerprints remain visible as
cache-skipped outputs; they do not independently trigger a build.

## Package files

### `cache_key.py`

**High level:** defines the cache compatibility version and retains helpers for
the earlier derived-key design and historical simulations.

**Technical:** `CACHE_EPOCH`, `current_cache_epoch()`, and
`validate_cache_epoch()` supply and validate the epoch used by `ArtifactKey`.
Bumping the epoch makes older epochs ineligible for new requests; it does not
delete rows, erase files, or change raw fingerprints or output digests.

`ArtifactIdentity`, `CacheKey`, `compute_cache_key()`, and
`derive_fallback_fingerprint()` remain available. The derived string form
`e{epoch}:{variant}:{hash}` is **not** the relational production lookup key.
The `srcfp1:` fallback identifies commit-derived historical evidence, which
must not be passed to `CacheDB.lookup()` as a real artifact key. The local
`--variant` option checks the configured build type; it does not add a separate
variant column to the relational key.

### `manifest.py`

**High level:** defines the Python objects that describe keys, stored files,
and database candidates.

**Technical:** `ArtifactKey` validates output identity, raw fingerprint, epoch,
and bundle membership. `ArtifactFile` describes one stored object's safe
relative path, SHA-256 digest, and size. `LocalManifest` reconstructs an eligible
production from joined database rows, including its key, producing run,
location, expected objects, validation state, timestamps, and provenance.

`LocalManifest.payloads_verified` is false: constructing this object does not
read or verify payload bytes. The class also checks that the expected object
inventory and file records agree.

`TrustLevel`, `ValidationState`, and the earlier `Manifest`, `OutputObject`,
`Provenance`, and `PublicationState` types remain for supporting or legacy APIs.
The legacy manifest's `pending/published/revoked` enum must not be confused with
the relational entry states `pending/ready/invalid`. There is no `manifests`
database table and no serialized legacy manifest column replacing the tables.

### `cache_db.py`

**High level:** connects current keys to saved productions and keeps their
provenance, observations, and object metadata consistent.

**Technical:** `CacheDB` opens an existing revision-6 SQLite database, validates
its schema, and enables foreign keys. Opening does not initialize or migrate
the database. Explicit transactions commit on success and roll back on failure;
nested transactions are rejected.

`lookup_batch()` accepts `ArtifactKey` objects, deduplicates requests, selects
candidates with a batched JSON-backed SQL query, and loads their joined file
metadata in another batched query. This is not the old string-key `IN` query
chunked at 500. Lookup performs no payload hashing or extraction. Multiple
productions may satisfy one key, allowing an older eligible production to be
selected when a newer one is unsuitable.

Read eligibility requires ready, validated, unexpired entries with complete
available object metadata. `ReadPolicy` additionally filters trust, validation,
and an optional time cutoff. A cache-root filter restricts candidate storage
locations without changing key identity. Relaxing policy never makes pending
or incomplete outputs ready.

Defaults depend on the entry point:

| Entry point | Default trust requirement |
| --- | --- |
| An explicitly constructed `ReadPolicy()` | `trusted` |
| `CacheDB` lookup with no policy | Accepts `untrusted` metadata |
| Local/package read-policy CLI options | `--min-trust untrusted` |

All still require at least build-passed eligible metadata. Local publication
also defaults to `untrusted`; a supplied trust label is not authentication.

`record_observations()` records collection results. `register_local_outputs()`
requires managed-build evidence and observations for the exact keys, inserts
object metadata, and commits the complete set of ready transitions together.
Builds and filesystem packaging occur outside this transaction. The legacy
`publish(Manifest)` API is explicitly rejected.

`invalidate_local_output()`, `evict_before()`, and `evict_where_missing()` retire
metadata. Eviction does not delete payload files or observation history.
`record_history()` and `history()` store and retrieve historical evidence in
typed producer metadata. `stats()` reports table and eligible-entry counts.

### `collector.py`

**High level:** records which artifacts a configured run expected, which
fingerprints were present, and what the database said about them.

**Technical:** `load_expected()` reads the explicit expected-output JSON
contract. `read_fingerprints()` validates sidecar paths and contents, preserves
missing/invalid output records, and normalizes accepted hexadecimal values.
`collect()` performs metadata lookup and persists the producing/collection run,
artifact identities, valid keys, and observations in a short transaction.

Collection reports distinguish a previously observed fingerprint from an
eligible stored production. A collector `hit` is a **metadata** result, with
`lookup_scope="database_metadata"` and `payloads_verified=false`. Collection
alone does not build, package, restore, or mark payloads ready. Unmeasured
source-preparation, configuration, and restoration timings remain null.

This file is the relocated collector, formerly `build_tools/artifact_cache.py`.
Its CLI provides `collect`, `build-publish`, and `restore`; the last two delegate
to the managed implementations. Component collection remains the lower-level
observation contract used by bundle publication.

### `inventory.py`

**High level:** determines which outputs to inspect and connects their names
to the static build topology.

**Technical:** an explicit `--expected` JSON list is authoritative.
Otherwise, `expected_outputs()` collects identities from both immediate
subdirectories and `.fprint` filenames under `build/artifacts/`. This preserves
an artifact with a missing sidecar instead of silently losing it from reports.

`scan_build()` validates emitted component identities and sidecars. In bundle
mode without an explicit list, it additionally incorporates configured Ninja
export declarations and asks `bundles.py` to validate complete membership.
`configured_build()` reads CMake's source directory, executable, and build type
from `CMakeCache.txt`; a supplied variant must match that build type.

`TopologyAdapter` uses `_therock_utils.build_topology` and
`BUILD_TOPOLOGY.toml` to map outputs to stages and changed paths to affected
artifact names. It is the adapter used by the current planner. It is separate
from the configured CMake consumer graph used by `dependency_impact.py`.

### `bundles.py`

**High level:** groups the requested components of one artifact and target
into a single cache unit.

**Technical:** `validate_bundle()` checks identity and canonical component
membership. `declared_ninja_outputs()` obtains configured artifact-export
membership. `group_records()` groups records by artifact and target and
requires a known requested member set with matching valid fingerprints.
Missing, invalid, or differing fingerprints prevent an `ArtifactKey` for the
whole requested bundle. Different artifacts or targets never merge merely
because their fingerprints match.

The resulting `ArtifactKey` keeps the raw CMake fingerprint and adds exact
component membership. Requests for `lib` and `lib + dev` therefore cannot match
the same key. `component_requests()` expands a bundle request back into the
component sidecar contract used by managed publication.

Diagnostic event names describe these decisions; they are not Python types or
database identifiers. Renaming a key-creation event to `ARTIFACT_KEY` does not
change any stored key or require a migration.

### `local_cache.py`

**High level:** runs the local cold/warm cache experiment and coordinates the
managed build when cacheable outputs need publication.

**Technical:** `prepared_scan()` checks source revisions, acquires the managed
build lock, reconfigures, scans current outputs, and rechecks source,
configuration, expected-list, and key stability. `cache_build()` performs a
batched metadata lookup and verifies candidates through `restore_local_key()`
using temporary destinations.

Verified candidates become `verified_cached` results. Cacheable misses are
passed to `build_and_publish()`. The publisher invokes the **default incremental
build**, not a set of targets selected by the cache planner. Only the requested
missing cache units are published, and final keys are rechecked afterward.

If there are no cacheable misses, the command does not invoke a build. This
does not establish that outputs lacking fingerprints were built or restored.

Subcommands are `cache`, `keys`, `stats`, and `synth`. `keys` reconfigures and
therefore writes build metadata, but does not publish payloads or write the
cache database. `synth` creates a small isolated CMake/Git fixture for testing;
it is not a real ROCm build. CLI `cache` and `keys` default to bundle granularity;
lower-level Python defaults retain component compatibility.

### `local_store.py`

**High level:** saves stable artifact copies and checks the contents that will
be extracted later.

**Technical:** `store_component()` and `store_bundle()` package component
export trees into a new generation under:

```text
<cache>/objects/artifact-<generation>/files/
    payload.tar
    inventory.json
```

`payload.tar` contains the exported files, directories, and supported symlinks.
`inventory.json` records member paths, types, modes, file sizes and SHA-256
digests, symlink targets, and bundle component membership where applicable.
Bundle archives preserve the original component directory names.

The writer detects changes during packaging, validates the archive against its
inventory, computes digests and sizes for both stored objects, and exposes the
finished generation. New publications use new locations; existing generations
are not overwritten. `unpack_component()` checks paths, member completeness,
contents, modes, and safe symlink targets during verification/extraction.

The database's `artifact_files` rows describe the two stored objects. The
inventory inside `inventory.json` describes their unpacked members. Neither
kind of output digest is the source/configuration fingerprint.

`LocalObjectStore` and `sha256_file()` remain available for the generic
object-store helpers and their tests. This local implementation does not make
historical S3 archive references reusable automatically.

### `publish.py`

**High level:** ensures a successful managed build precedes publication, so
new fingerprints cannot simply be paired with old files after configuration.

**Technical:** `build_and_publish()` owns configuration, build execution,
post-build configuration/input checks, packaging, producer recording, and ready
entry registration. It records actual source revisions and rejects tracked
source edits, unstable inputs, concurrent managed use of the build directory,
and `.prebuilt` imports in this publication path.

The sequence is:

1. Validate the database, requested outputs, source/build locations, and keys.
2. Run the default `cmake --build` command and require success.
3. Recheck source revisions, configuration, fingerprints, and expected inputs.
4. Save and validate new artifact generations with `local_store.py`.
5. Record producer evidence and component observations with `collector.py`.
6. Register exact output keys and commit ready entries through `cache_db.py`.

The producer record and ready-entry registration are separate database steps;
an observation alone does not imply a ready payload. Readiness is committed
only after all requested saved copies have been validated. A build or input
check failure does not publish those outputs.

`publish_build_outputs()` delegates to managed publication. It no longer
accepts an arbitrary list of existing archive URLs as proof of a successful
build. Correctness still depends on CMake tracking build inputs and the raw
fingerprint covering the environment needed for compatible reuse.

### `restore.py`

**High level:** proves saved files match their records and, when explicitly
requested, extracts them into a new destination.

**Technical:** `restore_local_key()` looks up eligible productions, verifies
object containment, digests and sizes, and checks the archive against its
inventory. Physically bad candidates are invalidated and older candidates
are tried. Successful verification yields a `restored` result with
`payloads_verified=true`; exhausted candidates yield `needs_build`.

`restore_local_outputs()` configures current inputs and rechecks their
stability around restoration. `restore_units()` materializes candidate plan
entries, leaving current-input checks to its caller. Bundle extraction
preserves component names and exposes a bundle's components only after its
verification succeeds. Destinations must be new.

The `cache` workflow discards its temporary verification output. Explicit
`restore` keeps the resulting export tree. Neither operation writes `.prebuilt`
markers or tells CI to skip a stage.

The legacy `ObjectStore`, `InMemoryObjectStore`, `RestoreResult`, and
`restore_manifest()` helpers remain. Their object-store check is distinct from
the relational extraction path; a successful legacy check does not itself
materialize files into a consumer build.

### `planner.py`

**High level:** reports current cache candidates and misses, organized by
stage, without declaring stages safe to skip.

**Technical:** `plan_cache()` consumes current scan records and `ArtifactKey`
objects. It optionally uses `inventory.TopologyAdapter` for stage ownership
and conservative changed-path misses, and performs a batched database lookup.
Every request remains visible, including unmapped outputs and those without
usable fingerprints.

`RequiredUnit`, `PlanEntry`, `StagePlan`, and `CachePlan` carry the report.
Statuses are `candidate`, `miss`, and `uncacheable`. `PlanEntry.is_hit` and
`StagePlan.can_skip` remain false; serialized plans have
`payloads_verified=false` and no skippable stages. Metadata candidates still
need physical verification and eventual build/scheduler integration.

The optional `dependency_impact` section is explanatory metadata. Passing it
does not change cache keys, select build targets, or turn a candidate into a
verified hit.

### `dependency_impact.py`

**High level:** explains which configured subprojects and artifact groups
depend on a changed CMake subproject.

**Technical:** `configured_impact()` reads the freshly configured
`build/therock_consumer_graph.json` and `build/artifact_subprojects.json`.
It validates the metadata, resolves target spelling with case-folding, rejects
ambiguous or missing graph nodes, walks consumer edges, and maps affected
subprojects to artifact groups.

For each changed root and affected artifact, it reports one shortest dependency
path. An artifact that directly contains the changed subproject may have a
one-node path. Results reflect the enabled CMake configuration, including
optional dependencies that may not be represented in the static topology.

The package `plan` command accepts repeatable `--changed-subproject` options
and adds the intersection with requested outputs to the report. This feature
has `build_selection_changed=false`. It does not narrow source fingerprints or
fix broad invalidation caused by a shared repository commit.

### `topology_view.py`

**High level:** retains the topology adapters from the earlier planner design.

**Technical:** the earlier `BuildTopologyView` and `DictTopologyView` describe
stage/artifact and path/artifact relationships. They remain separate from the
current production planner path, which uses `inventory.TopologyAdapter` and
the configured CMake metadata described above. Do not infer current stage-skip
behavior from the earlier adapter API or its original tests.

### `history_ingest.py`

**High level:** records what previous CI runs reported producing so reuse can
be studied later.

**Technical:** `ingest_runs()` parses archive names into component identities,
records object references and run/attempt provenance, and distinguishes real
supplied CMake fingerprints from commit-derived `srcfp1:` fallbacks. Offline
JSON input is supported; optional live helpers use existing GitHub/S3 listing
utilities. Network helpers are imported lazily.

`trust_for_run()` labels only `push` or `schedule` events on `main` as trusted,
subject to the expected-repository check on supplied metadata. Other
branches/events are untrusted. This is metadata classification, not producer
authentication. Successful completed runs are classified separately as
build-passed; trust and build success are different properties.

Evidence is stored through `CacheDB.record_history()` in
`prototype_runs.parameters_json`, with `configure_state='not_run'`.
Identical imports are idempotent; corrections can form later snapshots.
Ingestion creates **no ready cache entries, no artifact keys, and no local
payload files**. A verified remote import path is still required before those
historical outputs can participate in physical reuse.

### `replay.py`

**High level:** estimates potential reuse from historical evidence and compares
it with reuse restricted to the same commit.

**Technical:** `replay()` evaluates recorded outputs at each run's start.
Eligible producers must have become available strictly earlier, match the
repository/platform/variant scope, and satisfy validation and trust filters.
This prevents an overlapping run from seeing outputs that had not finished.
The latest imported snapshot for a run attempt supplies its evidence.

Real fingerprint matches and commit-fallback matches are kept distinct.
`ReplayReport` contains `potential_hits`, `same_commit_hits`, rates, and
per-run decisions with `payloads_verified=false`. These are model estimates
over recorded outputs, not measured build-time savings or proof that a whole
stage can be skipped.

### `reporting.py`

**High level:** shows concise operation progress and a final result for every
requested cache unit.

**Technical:** `log_build_progress()` shows operation-level phases while
keeping per-artifact outcomes for the end. `log_report()` formats completed
operations on stderr; stdout stays parseable JSON.

| Display in a cache run | Meaning |
| --- | --- |
| `MISS -> STORED` | No verified candidate was available; a successful managed build and publication saved a new production. |
| `REUSED` | JSON outcome `verified_cached`: saved files were verified in temporary storage, not installed into the build. |
| `NO_FINGERPRINT` | An expected sidecar is missing; cache restoration and storage are skipped for that unit. |
| `CACHE_SKIPPED` | Another condition prevents caching, such as invalid/differing fingerprints or unknown complete membership. |

Explicit publication reports `STORED` without inventing a preceding miss.
Explicit restoration reports `RESTORED` or `NEEDS_BUILD`. Intermediate
per-artifact `STORING` events are suppressed in the normal compact report.

### `diagnostics.py`

**High level:** supplies optional detailed logs for understanding cache
decisions and failures.

**Technical:** `trace()`, `span()`, `traced()`, and `ByteProgress` record events,
UTC timestamps, elapsed time, sequence numbers, operation durations, and
rate-limited byte progress. Logging sessions are scoped and restore their
previous context when nested. `run_logged()` wraps commands with start/end
events and exit status.

`--verbose` streams detailed events to stderr. `--log-file` writes details to
a new UTF-8 file even without `--verbose`; it does not overwrite an existing
log. Normal JSON stdout and final-report semantics are preserved. An event
name such as `ARTIFACT_KEY` is a diagnostic label, not a database column.

### `migrate_schema.py`

**High level:** upgrades a recognized older database while preserving its
saved productions and history.

**Technical:** the explicit migration validates recognized revision-3/4/5
layouts, creates a new backup with SQLite's backup API, and installs revision 6.
It adds the legacy epoch where needed, removes namespace from older keys,
merges equal keys with reference remapping, and adds `components_json`.
Existing component entries retain empty membership; they are not relabeled as
verified bundles.

Foreign-key and integrity checks run before commit. Failures roll back the
migration; unrecognized layouts or incompatible custom schema objects are
rejected. An already-current database is validated without another migration.
Normal `CacheDB` opening never performs these changes implicitly.

### `cli.py`

**High level:** exposes database inspection, planning, publication, restoration,
history, and maintenance commands.

**Technical:** argparse subcommands are `stats`, `lookup`, `plan`, `ingest`,
`replay`, `evict`, `publish`, and `restore`. They delegate to the modules above
and keep machine-readable results on stdout. Build-based planning/publication/
restoration default to bundle granularity. Read-policy, expected-list, target,
dependency-impact, and diagnostic options are applied at the relevant entry
points. Publication runs a managed build; it does not merely index existing
build output directories.

### `__main__.py`

**High level:** makes the package runnable with Python's `-m` option.

**Technical:** delegates to `cli.main()` and returns its exit status. Use the
current module path from the repository root:

```bash
python3 -m build_tools.github_actions.artifact_cache --help
```

The local cold/warm driver has its own entry point:

```bash
python3 -m build_tools.github_actions.artifact_cache.local_cache --help
```

### `__init__.py`

**High level:** provides the package's importable public API without eagerly
loading every subsystem.

**Technical:** lazy exports load the defining module when an attribute is
requested. This keeps basic collection independent of planner and optional
network helper imports. Public names include the relational types and managed
operations, along with explicitly retained legacy helpers. Exporting a legacy
type does not make it valid for the relational database API.

### `README.md`

**High level:** provides setup instructions, usage examples, and the current
behavioral contract.

**Technical:** documents schema initialization/migration, bundle membership,
CLI options, managed publication, verification, logging, policies, and current
limitations. Examples must use `build_tools.github_actions.artifact_cache` and
must not reintroduce the removed namespace option.

### `configured-impact.md`

**High level:** explains how to request and interpret configured dependency
impact reports.

**Technical:** documents the graph and artifact-map inputs, repeatable
`--changed-subproject`, reported dependency paths, and the separation between
impact reporting, fingerprint invalidation, and actual build selection.

### `fingerprint_cache_file_map.svg`

**High level:** provides a visual architecture overview.

**Technical:** this is documentation, not executable input to the planner or
database. Keep the diagram synchronized with `ArtifactKey`, bundles, the
relational schema, verification before reuse, objects-before-readiness
publication, and the reporting-only dependency-impact feature. Its historical
filename does not determine the Python package name.

### `per_file_details.md`

**High level:** explains the purpose of every package file and its connection
to the overall flow.

**Technical:** this guide distinguishes the active relational APIs from retained
legacy helpers and documents the corresponding schema and test locations.
It does not alter runtime behavior or migrate data.

## SQL schema outside the package

### `build_tools/therock_fingerprint_cache.sql`

**High level:** creates a new relational database for the artifact cache.

**Technical:** defines revision-6 tables, foreign keys, uniqueness constraints,
indexes, validation/immutability triggers, and the eligible-production view.
It is a fresh-schema script, not a migration script for an existing database.

| Table or view | What it records |
| --- | --- |
| `artifacts` | One full component or bundle identity per `artifact_id`. |
| `artifact_keys` | One version of that identity: epoch, fingerprint, exact component membership, and `artifact_key_id`. |
| `prototype_runs` | Collection/build provenance, source revisions, parameters, timings, and typed imported history evidence. |
| `artifact_observations` | What each run observed, including missing/invalid sidecars and metadata lookup results. |
| `cache_entries` | A particular saved production, linked to its key and producing run; location, state, validation, and expiry. |
| `artifact_files` | Expected stored-object paths, SHA-256 digests, sizes, and availability for a production. |
| `reusable_cache_entries` | A view of eligible metadata; it does not inspect physical files. |

`artifact_id` identifies the artifact. `artifact_key_id` identifies a particular
input state and component set. `cache_entry_id` identifies one saved production
of that key. Multiple productions can share an `artifact_key_id`.

The current local writer records `payload.tar` and `inventory.json` as the
stored objects. The files inside the archive are listed in the JSON inventory.
The database links to bytes through `cache_entries.local_directory`; the
payload bytes themselves are not stored in SQLite.

## Tests and fixtures

Tests live under `build_tools/github_actions/tests/` and use the naming pattern
`artifact_cache_*_test.py`. The following files cover both the relational
implementation and adapted regressions from the original implementation:

| File | Coverage |
| --- | --- |
| `artifact_cache_adaptation_test.py` | Policy/ranking behavior, planner boundaries, history evidence, replay, and adapted CLI/API behavior. |
| `artifact_cache_bundle_test.py` | Grouping, exact membership, equal fingerprints, missing-member rejection, discovery completeness, and bundle publication/restoration. |
| `artifact_cache_collector_test.py` | Expected lists, fingerprint validation, missing outputs, observation persistence, repeated keys, and metadata-only reports. |
| `artifact_cache_db_test.py` | Schema validation, foreign keys, transactions, batched lookup, candidate selection, and joined manifests. |
| `artifact_cache_dependency_impact_test.py` | Configured consumer traversal, artifact mapping, dependency paths, invalid metadata, and reporting without build selection. |
| `artifact_cache_diagnostics_test.py` | Detailed events, scoped logging, JSON preservation, progress, and failure diagnostics. |
| `artifact_cache_epoch_test.py` | Epoch validation, separation/coexistence of keys, stored-epoch preservation, and migration behavior. |
| `artifact_cache_history_ingest_test.py` | Archive parsing, main push/schedule trust classification, run validation, and evidence-only ingestion. |
| `artifact_cache_key_test.py` | Retained derived-key helpers and raw relational-key regression coverage. |
| `artifact_cache_local_test.py` | Immutable generations, inventories, safe extraction, real miniature builds, stale-output rejection, and failures. |
| `artifact_cache_logging_test.py` | Compact final status wording, configure/build visibility, JSON stdout, and absence of false success/reuse claims. |
| `artifact_cache_manifest_test.py` | Metadata types, trust/validation ordering, serialization, and relational manifest invariants. |
| `artifact_cache_namespace_test.py` | Removal of legacy namespaces, equal-key merging, reference preservation, backups, and rollback. The filename describes migration coverage, not a current namespace feature. |
| `artifact_cache_planner_test.py` | Candidate/miss classification, changed-path handling, read policy, and no unverified stage skips. |
| `artifact_cache_publish_restore_test.py` | Publication evidence requirements, complete metadata, validation/trust behavior, and restore integrity checks. |
| `artifact_cache_replay_test.py` | Historical potential reuse, same-commit comparison, policy, and serialization of estimates. |
| `artifact_cache_workflow_test.py` | Local object-store helpers and cold/warm/source-change workflows using isolated fixtures. |

### `_artifact_cache_fixtures.py`

**High level:** supplies reusable test data and database setup helpers.

**Technical:** creates temporary databases from the real SQL schema and supplies
controlled keys, runs, observations, and productions for tests. Fixture setup
is not a production publication API. The leading underscore keeps the helper
outside the `artifact_cache_*_test.py` discovery pattern.

Run the suite from the repository root:

```bash
THEROCK_CACHE_SCHEMA="$PWD/build_tools/therock_fingerprint_cache.sql" \
  python3 -m unittest discover \
    -s build_tools/github_actions/tests \
    -p 'artifact_cache_*_test.py' -v
```

The last recorded validation of the configured-impact checkpoint was 266
passing tests. That is a recorded result, not a fixed requirement for future
test counts. Isolated fixture tests complement real-build cold/warm and
source-change experiments; they do not prove full CI-stage reuse integration.
