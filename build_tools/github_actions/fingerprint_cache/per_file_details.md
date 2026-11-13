# Files explained — fingerprint cache

Every file in `build_tools/github_actions/fingerprint_cache/`, each with a plain-English
("high level") and a precise ("technical") description. The system is three flows:
**Planning** (compute each artifact's key), **Restore** (look up a key, verify, reuse or
rebuild), **Publish** (save a freshly built unit — objects first, entry last).

---

## cache_key.py
**High level:** turns a fingerprint into the cache key — the ID everything is looked up by.
**Technical:** `compute_cache_key(source_fingerprint, build_variant, cmake_files_hash?,
extra_inputs?, epoch)` SHA-256s a length-prefixed parts list and wraps it as
`CacheKey.value = "e{epoch}:{variant}:{hash}"`. `source_fingerprint` is TheRock's `.fprint`
value — the single input that changes when real uploaded fingerprints arrive, so everything
downstream is insulated ("the seam"). Also defines `ArtifactIdentity(name, component,
target_family)` (the output identity) and `derive_fallback_fingerprint()` (a commit-derived
stand-in used only by the history path, marked `srcfp1:` so it can never collide with a real
key). `CACHE_EPOCH` is the global invalidation lever (bump it to wipe the whole cache).

## manifest.py
**High level:** describes what a key points to — the files plus metadata.
**Technical:** `Manifest` = `outputs: [OutputObject(key, uri, digest, size)]` +
`Provenance(repo, run_id, branch, commit, workflow, created_at)` + three enums:
`TrustLevel` (trusted/untrusted), `ValidationState` (unvalidated/build_passed/tests_passed),
`PublicationState` (pending/published/revoked). `is_servable()` = published + every output has
a URI. JSON-serializable, stored as a column. The per-object digests are what make reuse
verifiable.

## cache_db.py
**High level:** the index — the phone book from key → entry. SQLite now, ClickHouse later.
**Technical:** one `manifests` table, `cache_key` primary key, manifest stored as JSON plus
queryable columns and indexes. `lookup_batch(keys, policy)` resolves many keys in one
`WHERE cache_key IN (…)` query (chunked at 500). `ReadPolicy(min_trust, min_validation,
require_published, created_before)` filters at read time (default: trusted + build_passed +
published). `publish()` refuses incomplete or non-published entries (the write-ordering
invariant); idempotent via `ON CONFLICT(cache_key) DO UPDATE`. Eviction: `evict_before()`
(retention) and `evict_where_missing()` (objects vanished from storage). Thin wrapper → the
storage backend is swappable without touching callers.

## local_store.py
**High level:** where the bytes live locally — the warehouse shelf; stands in for S3.
**Technical:** `LocalObjectStore(root)`. `upload_file(src, key)` copies a file in, computes its
real SHA-256 + size, returns `OutputObject(key, uri="local://key", digest, size)`.
`exists(uri)` / `fetch_digest(uri)` implement the `ObjectStore` protocol used by `restore`.
In production this role is played by the existing `ci-artifacts` S3 buckets.

## restore.py
**High level:** decides whether a cache hit is actually safe to use, and proves it.
**Technical:** `restore_manifest(manifest, store, verify_digests=True)`: is the manifest
servable? does each output still exist in the store? does the recorded digest match the
re-hashed bytes? all yes → `reusable`; any no → `needs_build`. Defines the `ObjectStore`
protocol (`exists`/`fetch_digest`) and `InMemoryObjectStore` for tests. This is the integrity
safety net — corrupt or missing bytes get rebuilt, never reused.

## publish.py
**High level:** safely saves a freshly built unit into the cache, in the right order.
**Technical:** `publish_build_outputs(...)`: Gate A (outputs complete?) → Gate B
(`default_producer_policy`: `validation >= build_passed`?) → upload objects **first** →
`db.publish(manifest)` **last**, marked `PUBLISHED`. Untrusted (PR) entries are still stored,
just tagged. `ProducerPolicy` is pluggable. Composes the key via the single `cache_key` seam.

## planner.py
**High level:** the CI brain — given what changed, decides which whole build stages can skip.
**Technical:** `plan_cache(topology, changed_paths, target_families, source_commit, db, …)`
maps changed paths → required artifact units per build stage (via `BuildTopologyView`),
computes all keys, does one batched lookup, and marks a stage `can_skip` iff *every* required
unit is a hit; changed units are forced misses. A `source_fingerprints` override is the hook
where a real uploaded `.fprint` plugs in. Deliberately separate from test selection (RFC0013).

## topology_view.py
**High level:** reads TheRock's build map so the planner knows what produces what.
**Technical:** `BuildTopologyView` adapts `_therock_utils.build_topology` over the real
`BUILD_TOPOLOGY.toml` (stage → produced artifacts, path → artifacts). `DictTopologyView` is a
hand-built double so the planner is unit-testable without a full checkout.

## local_cache.py
**High level:** the thing you run after a build; ties the whole loop together locally.
**Technical:** `scan_build()` reads every `.fprint` under `build/artifacts/`, groups component
directories into `(slice, target_family)` units, and flags fingerprint-less dirs as
uncacheable (`ScanResult`). Subcommands: `keys` (compute keys, report cacheable vs not — no
store, no disk), `cache` (lookup → verify → REUSED, else tar + upload + publish → STORED),
`stats` (`db.stats()`).

## history_ingest.py
**High level:** fills the cache from past CI runs so it isn't empty on day one.
**Technical:** enumerates workflow runs (reuses `github_actions_api`), parses artifact
filenames into `(name, component, target_family)`, classifies trust from event/branch, and
publishes entries. Uses the fallback fingerprint since real ones aren't uploaded yet.

## replay.py
**High level:** measures how much the cache would actually save.
**Technical:** replays runs in time order with point-in-time visibility (a run only sees
entries produced by earlier runs), comparing the fingerprint cache's hit rate against a naive
"same-commit only" baseline. Produces a serializable report.

## cli.py · __main__.py · __init__.py
**High level:** lets you run `python -m fingerprint_cache …` and import the package.
**Technical:** `cli.py` holds the argparse subcommands for the non-local paths
(ingest/plan/stats/lookup); `__main__.py` is the `-m` entry point; `__init__.py` exports the
public API.

## tests/fingerprint_cache_*_test.py
**High level:** proves each piece works (63 tests).
**Technical:** one `*_test.py` per module — key composition, DB publish/lookup/policy/eviction,
planner hit/miss + stage-skip, restore verification, publish gating, ingest parsing, and the
replay comparison.
