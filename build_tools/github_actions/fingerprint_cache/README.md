# Fingerprint-based CI artifact cache

Implements [ROCm/TheRock#8651](https://github.com/ROCm/TheRock/issues/8651):
replace the "baseline run" stage-reuse mechanism (GitHub-API run-history
traversal + commit ancestry + S3 archive-filename matching) with a direct,
batched **fingerprint → manifest** cache lookup.

> **Status: early implementation.** The cache key is currently a *derived fallback*
> (see the [key seam](#the-key-seam) below) because real artifact fingerprints
> are not yet uploaded with artifacts. The pipeline, data model, planning,
> restore, publish ordering, trust model, and replay harness are the production
> shape; only the key derivation changes when real fingerprints land.

## Design contract

> **equal keys imply safe substitution of build outputs**

A cache key folds in everything that can change an artifact's bytes in a way
consumers care about: source identity, build variant, toolchain/recipe inputs,
and a global `CACHE_EPOCH`. The value is an **output manifest** (not just an S3
URL): output objects + digests, dependency identities, producer provenance, and
trust / validation / publication state.

## The three flowcharts

| Flowchart | Module | What it does |
|-----------|--------|--------------|
| 1. Per-run planning | `planner.py` | changed paths → required artifact units & stages → compute keys → one **batched** hit/miss classification → decide which whole stages can be skipped |
| 2. Restore a unit | `restore.py` | lookup → compatibility/completeness/read-policy → download objects → verify digests & required outputs → *reusable* vs *needs building* |
| 3. Publish a unit | `publish.py` | outputs complete? producer & validation policy satisfied? → upload immutable objects **first** → publish manifest + fingerprint entry **last** |

Supporting modules: `cache_key.py` (the key contract + seam), `manifest.py`
(the cache value), `cache_db.py` (SQLite store: batched lookup, ordered publish,
read policy, eviction), `topology_view.py` (adapts the real `BUILD_TOPOLOGY.toml`
and provides a test double), `history_ingest.py` (populate from real run
history), `replay.py` (point-in-time reuse measurement), `cli.py` (entry point).

## The key seam

`cache_key.compute_cache_key(source_fingerprint=..., build_variant=..., ...)` is
the single place that changes when real fingerprints are available. Today
`source_fingerprint` comes from `derive_poc_fingerprint(...)`, a **placeholder**
keyed on the run's head commit + artifact identity. It is deliberately coarser
than the real `THEROCK_FPRINT` (which hashes per-subproject source, deps,
toolchain, and build-flag state), so it *under*-estimates achievable reuse —
the safe direction for a feasibility estimate. Derived keys carry a `pocfp1:`
marker so they can never collide with future real-fingerprint keys.

The planner also accepts a `source_fingerprints` override keyed by
`(artifact_name, target_family)`; this is the exact hook where an uploaded
`.fprint` plugs in, and is what the replay harness uses to model per-subproject
(cross-commit) reuse.

## Usage

```bash
# From build_tools/github_actions/ (so `fingerprint_cache` is importable):

# Plan cache hits/misses for a run against the real topology + a local cache.
python -m fingerprint_cache plan --db cache.sqlite \
    --commit <sha> --targets gfx94X-dcgpu \
    --changed projects/rocblas/library/foo.cpp \
    --topology ../../BUILD_TOPOLOGY.toml

# Ingest real run history into the cache (needs GITHUB_TOKEN / gh auth, and AWS
# access to the CI artifact buckets for the S3 artifact listing).
python -m fingerprint_cache ingest --db cache.sqlite --max-runs 50 --require-trusted

python -m fingerprint_cache stats  --db cache.sqlite
python -m fingerprint_cache lookup --db cache.sqlite --key <cache-key>
```

### Worked example (real `BUILD_TOPOLOGY.toml`, 75 artifact units)

* Same commit, no changes → **100% hits, all 12 stages skippable**.
* Same commit, a `projects/rocblas/...` change → **74/75 units reused (98.7%),
  11/12 stages skipped**; within `math-libs` only `blas` rebuilds, the other 19
  units are reused.
* An unmapped path (e.g. `build_tools/...`) conservatively forces a full
  rebuild — same fail-safe stance as today's `stage_impact` full-CI fallback.

## Local build cache (Phase 0) — real local build, no `.yml`, no GitHub Actions

`local_cache.py` drives cold/warm artifact reuse from a **real local TheRock build**
against a **local cache** (SQLite + a directory object store, `local_store.py`),
on one machine with no CI and no network. It consumes exactly what a real build
emits under `build/artifacts/`:

```
build/artifacts/
    <slice>_<component>_<suffix>/        # component OUTPUT DIRECTORY
    <slice>_<component>_<suffix>.fprint  # real THEROCK_FPRINT sidecar (configure-time)
```

`<suffix>` is `generic` or a GPU family (e.g. `gfx94X-dcgpu`). The cached object
for a unit is the component **directory** (tarred locally); in CI the same unit
maps to the uploaded **archive** — i.e. the local cache caches local build dirs, the CI
implementation caches S3 archives.

You run **one command, the same way every time**, after each local build. The
first run stores every artifact; the next run reuses whatever is unchanged and
only stores what's new:

```bash
# from build_tools/github_actions/, after a local TheRock build into <build>/
python -m fingerprint_cache.local_cache cache --build <build> --cache ~/.therock-cache
# ...edit a subproject (e.g. rocBLAS), rebuild, then run the same command again:
python -m fingerprint_cache.local_cache cache --build <build> --cache ~/.therock-cache
# -> reuses everything except the artifact you changed, which it stores anew.
```

No real build handy? `synth` fabricates a build tree in the **real layout** (a
test fixture, not the ) so the loop runs anywhere:

```bash
python -m fingerprint_cache.local_cache synth --out /tmp/build                 # real-layout fixture
python -m fingerprint_cache.local_cache cache --build /tmp/build --cache /tmp/cache   # first: stores all
python -m fingerprint_cache.local_cache cache --build /tmp/build --cache /tmp/cache   # again: reuses all
python -m fingerprint_cache.local_cache synth --out /tmp/build --change blas    # edit rocBLAS
python -m fingerprint_cache.local_cache cache --build /tmp/build --cache /tmp/cache   # reuses all but blas
```

**Design issue already surfaced:** split / per-arch artifacts emit no `.fprint`
(`cmake/therock_artifacts.cmake` TODO), so they are **uncacheable** today — the
driver reports them (the synth `miopen` slice reproduces this). These contracts
are deliberately **provisional**; the local cache exists to flush issues like this
before anything is frozen.

## What is and isn't modeled

* **Is:** the full plan → restore → publish loop; batched O(1)-ish lookups;
  trusted (push/main) vs untrusted (PR) entries and read policy; publish write
  ordering; point-in-time replay; eviction tolerant of S3 retention; runs
  against the real topology.
* **Isn't yet:** real uploaded fingerprints (the key seam); real S3/ClickHouse
  backends (SQLite here, query layer abstracted for repointing); split /
  per-arch artifact identities beyond the bundle unit; dependency-fingerprint
  propagation in the key.

## Phase 2 (when real fingerprints are uploaded)

1. Upload `.fprint` / manifest metadata as part of CI runs.
2. Replace `derive_poc_fingerprint` with the uploaded fingerprint at the
   `compute_cache_key` seam (and feed `source_fingerprints` in the planner).
3. Repoint `CacheDB` from SQLite to the team's ClickHouse instance (same API).
4. Wire the planner into `configure_multi_arch_ci` / the "Copy Prebuilt Stages"
   path so stage jobs are skipped from cache decisions instead of baseline runs.

## Tests

```bash
# From the repo root:
python -m pytest build_tools/github_actions/tests/fingerprint_cache_*_test.py -q
```

58 unit tests cover key composition, the manifest model, DB publish ordering /
batched lookup / read policy / eviction, the planner's hit/miss & stage-skip
logic, publish gates, restore verification, history-ingest parsing/grouping/
trust classification, and the replay harness (including cross-commit reuse
beating the naive same-commit baseline).

----------------------------------------------------------------------------------
# Technical overview — fingerprint cache implementation

How the local build cache is actually built. Package lives at
`build_tools/github_actions/fingerprint_cache/` (~2,000 lines, 63 unit tests, Black-formatted).

## Module map

| Module | Responsibility |
|---|---|
| `cache_key.py` | Turn a source fingerprint into a cache key (the **seam**) |
| `manifest.py` | The cache **value**: what a key points to + metadata |
| `cache_db.py` | SQLite index: batched lookup, ordered publish, read policy, eviction |
| `local_store.py` | Directory-backed object store (digests) — local stand-in for S3 |
| `restore.py` | Resolve a key → download + verify → reusable / needs-build |
| `publish.py` | Write a built unit back to the cache (ordering + policy gates) |
| `local_cache.py` | The CLI: scan a build tree → decide → store/reuse (`keys`/`cache`/`stats`) |
| `planner.py` | CI-side planning: changed paths → stage skip decisions (batched) |
| `topology_view.py` | Adapter over `BUILD_TOPOLOGY.toml` (+ a test double) |

## End-to-end data flow (the `cache` command)

```
build/artifacts/<slice>_<comp>_<suffix>.fprint   (Stella's fingerprint, configure-time)
        │  scan_build(): read sidecar, group components by (slice, target) into a "unit"
        ▼
source_fingerprint  ──compute_cache_key()──►  CacheKey.value = "e{EPOCH}:{variant}:{sha256}"
        │                                             │
        │                                    CacheDB.lookup_batch()  (one SQL WHERE IN)
        ▼                                             ▼
   HIT? ── yes ──► restore_manifest(): object exists? digest matches? ──► REUSE
        └─ no  ──► tar component dir(s) ► LocalObjectStore.upload_file() ► publish_build_outputs() ► STORE
```

## 1. The cache key — `cache_key.py` (the seam)

```python
compute_cache_key(*, source_fingerprint, build_variant,
                  cmake_files_hash=None, extra_inputs=None, epoch=CACHE_EPOCH) -> CacheKey
```

- `source_fingerprint` is **Stella's `.fprint` value** (read verbatim). This is the *only*
  input that changes when real uploaded fingerprints replace the fallback — everything else
  is layered on here, so the rest of the system never changes.
- The key is `SHA256` over a length-prefixed list of parts (`fp=…`, `variant=…`,
  optional `cmake=…`, `extra=…`), wrapped as:
  `CacheKey.value = f"e{epoch}:{build_variant}:{fingerprint}"`.
  - `epoch` (`CACHE_EPOCH`) = global version; bump it to invalidate the whole cache.
  - `build_variant` = release / asan / debug (different bytes → different key).
- `ArtifactIdentity(name, component, target_family)` is the *output* identity (used in the
  manifest). **Known gap:** `target_family` is not yet folded into the key, so once split
  artifacts emit fingerprints, per-arch components would collide — the next key change.
- `derive_fallback_fingerprint()` exists only for the future history-ingest/replay path
  (no real `.fprint`); it is **not** used by the local build (that reads the real file).
  Fallback keys carry a `srcfp1:` marker so they can never collide with real ones.

## 2. The cache value — `manifest.py`

A key points to a **manifest**, not just a URL:

```
Manifest(cache_key, artifact: ArtifactIdentity,
         outputs: [OutputObject(key, uri, digest, size)],
         provenance: Provenance(repo, run_id, branch, commit, workflow, created_at),
         build_variant, dep_identities,
         trust       ∈ {trusted, untrusted},
         validation  ∈ {unvalidated, build_passed, tests_passed},
         publication ∈ {pending, published, revoked})
```

- `outputs` are the actual objects (one per component archive/dir) with SHA-256 digests —
  this is what enables integrity verification on reuse.
- `trust` / `validation` / `publication` drive the read policy (below). `is_servable()` =
  `publication == published and every output has a URI`.
- JSON-serializable (`to_json` / `from_json`) — the whole manifest is stored as a column.

## 3. The index — `cache_db.py` (SQLite, swappable for ClickHouse)

Single table, `cache_key` as primary key, plus the manifest JSON and queryable columns:

```sql
CREATE TABLE manifests (
  cache_key TEXT PRIMARY KEY, artifact_name, component, target_family, build_variant,
  trust, validation, publication, repository, run_id, branch, commit_sha, workflow,
  created_at, manifest_json TEXT);
-- indexes on run_id, (artifact_name,target_family), created_at
```

- **`lookup_batch(keys, policy)`** resolves a whole batch in one `WHERE cache_key IN (…)`
  query (chunked at 500 to stay under SQLite's variable limit). This is the O(1)-style
  lookup that replaces the old GitHub-history crawl.
- **`ReadPolicy(min_trust, min_validation, require_published, created_before)`** filters at
  read time — default: trusted + build-passed + published. `created_before` gives
  point-in-time correctness for replay.
- **`publish(manifest)`** refuses anything not `PUBLISHED` or not complete (enforces the
  write-ordering invariant). Idempotent via `ON CONFLICT(cache_key) DO UPDATE`.
- **Eviction:** `evict_before(ts)` (retention) and `evict_where_missing(exists_fn)` (drop
  entries whose backing objects disappeared — the cache points into retention-governed S3).
- The whole class is a thin wrapper, so swapping SQLite → ClickHouse is a drop-in later
  (same method surface).

## 4. The object store — `local_store.py`

`LocalObjectStore(root)` is the local stand-in for S3 immutable storage:
- `upload_file(src, key)` copies the file in, computes its real SHA-256 + size, returns an
  `OutputObject(key, uri="local://key", digest, size)`.
- `exists(uri)` / `fetch_digest(uri)` satisfy the `ObjectStore` protocol used by restore.
- In the local cache, the cached object for a unit is the component **directory**, tarred;
  in CI the same unit maps to the uploaded S3 **archive** (same manifest shape).

## 5. Restore + verify — `restore.py`

`restore_manifest(manifest, store, verify_digests=True)`:
1. `manifest.is_servable()`? else needs-build.
2. For each output: `store.exists(uri)`? else needs-build (object gone from storage).
3. If a digest is recorded, `store.fetch_digest(uri)` must match, else needs-build.
4. All good → `reusable`.

This is why a corrupted/truncated artifact is never reused — it's re-hashed on the way out
and downgraded to a rebuild. Defends against DB↔storage drift.

## 6. Publish ordering — `publish.py`

`publish_build_outputs(...)` enforces flowchart-3 ordering:
1. **Gate A:** all outputs complete (have URIs)? else return `None` (don't publish).
2. **Gate B:** producer policy satisfied? (default: `validation >= build_passed`) else `None`.
3. Upload immutable objects **first**.
4. `db.publish(manifest)` — writes the fingerprint entry **last**, as `PUBLISHED`.

So a reader can never resolve a key to a manifest whose objects aren't fully written.
Untrusted (PR) entries *are* published but tagged `UNTRUSTED`; the read policy decides reuse.

## 7. The local driver — `local_cache.py`

- `scan_build(build_dir)` walks `build/artifacts/`: indexes `*.fprint` sidecars and the
  sibling component directories, groups components into `(slice, target_family)` **units**.
  A directory with no `.fprint` (split artifacts) goes to `ScanResult.uncacheable`.
- A unit's `source_fingerprint` is the shared `.fprint` value for that slice.
- `cmd_keys` (your half): compute each key, report cacheable vs uncacheable — no store, no
  disk, no network.
- `cmd_cache`: for each unit, `lookup` → if servable+verifiable → **REUSED**; else tar the
  component dir(s), `upload_file`, `publish_build_outputs` → **STORED**.
- `cmd_stats`: `db.stats()`.

## 8. The CI planner — `planner.py` (not exercised in the local demo)

`plan_cache(topology, changed_paths, target_families, source_commit, db, …)` is the CI-side
entry: it maps changed paths → required artifact units per build stage (via
`BuildTopologyView` over the real `BUILD_TOPOLOGY.toml`), computes all keys, does **one
batched lookup**, and marks a stage `can_skip` iff *every* required unit is a hit. Units
whose source changed are forced misses. A `source_fingerprints` override is the exact hook
where a real uploaded `.fprint` plugs in. This stays separate from test selection (RFC0013).

## Design invariants to remember

- **One seam.** Only `source_fingerprint` changes when real fingerprints arrive;
  `compute_cache_key` and everything downstream are untouched.
- **Equal keys ⇒ safe substitution.** The correctness contract; the key must fold in every
  input that changes the bytes (source, deps, toolchain, flags, variant, epoch — and
  pending: container image, compiler id, target family).
- **Verify on read.** Digests re-checked on restore; bad bytes → rebuild.
- **Fail safe.** Missing/incompatible/unavailable cache always falls back to building.
- **Ordered writes.** Objects first, entry last; incomplete entries are never published.
- **Swappable store.** SQLite now, ClickHouse later, same interface; objects point into
  existing S3, not duplicated.

## Known gaps (surfaced by the real build — talking points)

1. **Split/per-arch artifacts emit no `.fprint`** (cmake `if(NOT _should_split)`), so the 15
   target-specific math/ML libs are uncacheable today. Top fix (cmake change, review w/ Stella).
2. **Key lacks output identity** — fold `target_family` into `compute_cache_key` so per-arch
   components don't collide once (1) lands.
3. **Fingerprint composition** — decide whether container image / compiler id / env vars go
   in the key (correctness audit).

What "trusted" means

It's about where the artifact was built, i.e. how much you can rely on it:

trusted — built by a push/scheduled run on a protected branch (main or release/*). These went through the normal gated pipeline, so they're safe for anyone to reuse.
untrusted — built by a pull-request run. A PR can contain arbitrary code from anyone, so an artifact it produced shouldn't automatically be trusted as a dependency for other people's builds.

In the code that's the TrustLevel enum in manifest.py, set at publish time (history_ingest.py decides it from the run's event/branch).