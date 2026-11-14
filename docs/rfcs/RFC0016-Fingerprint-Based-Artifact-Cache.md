---
author: Ameya Mallya (ammallya), Erman Gurses (erman-gurses)
created: 2026-10-26
modified: 2026-10-26
status: draft
discussion: https://github.com/ROCm/TheRock/issues/8651
---

# Fingerprint-Based CI Artifact Cache

Replace the "baseline run" stage-reuse mechanism with a content-addressed artifact
cache keyed on `THEROCK_FPRINT`. Each run computes a cache key per artifact,
resolves hits in one batched database lookup, verifies object integrity, and
reuses prebuilt artifacts instead of rebuilding. No GitHub-API history traversal,
no git-history heuristics. Tracks [#8651](https://github.com/ROCm/TheRock/issues/8651)
(parent [#3399](https://github.com/ROCm/TheRock/issues/3399)).

## Problem

Reuse today selects a baseline workflow run by **commit ancestry + S3
archive-filename presence + job health** (`build_tools/github_actions/baseline_runs.py`,
`stage_reuse_decision.py`); fingerprints are never used. This is O(N) over workflow
runs, costs many API calls, cannot express rocm-libraries → rocm-libraries
dependencies or arbitrary build-graph subsets, and is fragile on release branches
([#8454](https://github.com/ROCm/TheRock/issues/8454)).

`THEROCK_FPRINT` (configure-time, per artifact) already hashes source identity,
dependency fingerprints, toolchain, and build flags. We use it as the cache key.

## Goals

- O(1)-style, batched key lookup, independent of workflow-history length.
- Reuse unchanged artifacts across pin updates and component commits; invalidate on
  recipe/environment changes.
- Integrity-verified reuse; unavailable/ineligible cache always falls back to build.
- Works on non-default/release branches; composes with RFC0013 test selection.

## Non-goals

- Test selection — stays [RFC0013](./RFC0013-Consumer-Based-Test-Selection.md);
  a cache hit never suppresses a test.
- Replacing `ccache` (compiler-call cache; complementary, stacks with this).
- New storage — reuse existing `ci-artifacts` S3 buckets as object storage; store
  pointers, not copies.

## Design

### Cache key

```
cache_key = e{EPOCH}:{build_variant}:{SHA256(source_fingerprint, build_variant,
                                             recipe_inputs, output_identity)}
```

| Input | In `THEROCK_FPRINT` | Action |
| --- | --- | --- |
| Subproject source identity (committed rev/HEAD; invalid if dirty) | yes | keep |
| Dependency fingerprints | yes | keep |
| Toolchain file | yes | keep |
| Build flags / variant | partial | keep + explicit `build_variant` in key |
| Output identity (`name`, `component`, `target_family`) | no | **add** — per-arch split artifacts share one source fingerprint and must not collide |
| Container/image digest | no | **add** |
| Compiler id + version | indirect | add |
| `CACHE_EPOCH` (global) | n/a | keep — bump to invalidate the whole cache |

Key composition is isolated in one function so `source_fingerprint` is the only term
that changes when real uploaded fingerprints replace the PoC stand-in.

### Cache value (manifest)

```
cache_key -> Manifest {
  artifact: (name, component, target_family)
  outputs:  [ {key, uri->S3 pointer, digest sha256, size} ]
  provenance: {repository, run_id, branch, commit, workflow, created_at}
  build_variant, dep_identities
  trust:       trusted | untrusted
  validation:  unvalidated | build_passed | tests_passed
  publication: pending | published | revoked
}
```

A **read policy** filters at lookup time (default: `trusted` + `build_passed` +
`published`). `trust` = push/schedule on a protected branch vs. pull_request (PR
entries are stored but tagged, so PRs cannot poison shared reuse). `publication`
enforces write ordering (readers trust only `published`).

### Flows

![Cache flow and the file behind each step](./assets/RFC0016-cache-flow.svg)

1. **Plan** (per run): changed paths → required artifact units per build stage →
   compute keys → one batched `WHERE cache_key IN (...)` lookup → a stage is
   skippable iff every required unit is a hit; otherwise only missing units build.
   Source-changed units are forced misses.
2. **Restore** (per unit): lookup under read policy → check completeness → download
   objects → re-verify digests → reusable / needs-build. A hit with missing or
   corrupt objects is downgraded to a build (defends against DB/storage drift).
3. **Publish** (per built unit): gate on outputs-complete + producer policy →
   upload objects first → publish manifest + fingerprint entry last.

### Storage

Objects stay in the existing S3 buckets (manifest holds pointers; entries evictable
when objects are retention-deleted). Index is a `fingerprint -> manifest` store
behind one interface: SQLite for local builds, the team's **ClickHouse** instance
behind a batched lookup/publish API in production.

Reference implementation: `build_tools/github_actions/fingerprint_cache/` (plan →
`planner.py`/`local_cache.py`, key → `cache_key.py`, value → `manifest.py`, index →
`cache_db.py`, objects → `local_store.py`, restore → `restore.py`, publish →
`publish.py`); 63 unit tests.

## Proof of concept

Local `gfx110X-all` build, no workflow edits:

- Cold build populated the cache; second build reused **50 artifacts** with per-object
  SHA-256 verification; entries tagged `trusted`/`build_passed`/`published`.
- Coverage: **62 `target-neutral` artifacts cacheable, 15 `target-specific` (split)
  uncacheable** — split artifacts emit no `.fprint` (`if(NOT _should_split)` in
  `cmake/therock_artifacts.cmake`). The uncacheable set is the expensive math/ML libs.
- Committed single-subproject edit → reuse all but that artifact. Uncommitted (dirty)
  edit → fingerprint invalid → reported uncacheable, never reused.

## Required changes

1. Emit `.fprint` for split (Kpack, [RFC0008](./RFC0008-Multi-Arch-Packaging.md))
   artifacts — core fingerprinting change; largest reuse unlock.
2. Fold `target_family` (output identity) into the key (prevents per-arch collision).
3. Finalize + freeze key composition (container digest, env allowlist) under `CACHE_EPOCH`.
4. Upload fingerprints + manifests as part of CI runs.

## Disabling the cache

| Mechanism | Scope | Effect |
| --- | --- | --- |
| Env flag (e.g. `THEROCK_CACHE=off`) | any | skip cache, build from source |
| Dev-build default | local | proposed off-by-default locally |
| Dry-run | any/CI | report hits, always build |
| Read-bypass | any | ignore hits, repopulate (recover from poisoned entry) |
| `CACHE_EPOCH` bump | global | invalidate entire cache |
| Revoke manifest | targeted | never serve a known-bad entry |
| `stage_reuse_mode = off` | workflow | disable per-workflow / release branches |

## Test scenarios

| Scenario | Expected |
| --- | --- |
| Rebuild, no changes | 100% reuse |
| Committed edit of one subproject | only that artifact + consumers rebuild |
| Uncommitted (dirty) edit | uncacheable, reported, never reused |
| Compiler/toolchain change | broad invalidation |
| release → ASan | no cross-variant reuse |
| Container/image change | invalidation (once image digest in key) |
| Split/per-arch artifact | uncacheable today — closed by this RFC |
| Same source, different dir | same key (path-independent) |
| Cache unavailable / object missing / digest mismatch | build, no failure |
| Untrusted (PR) entry, default policy | ignored |

Benchmarks (repeatable harness, same commit/flags/runner): cold vs. warm wall-time
per arch, hit rate (no-op and single-subproject edit), batched lookup latency,
runner-hours saved/week, storage growth.

## Implementation plan

1. Fingerprint audit + split-artifact `.fprint` emission + output identity in key.
2. Upload fingerprints/manifests in CI; ingest run history into a dev cache.
3. ClickHouse `fingerprint -> manifest` schema + lookup/publish API; eviction;
   monitoring.
4. Wire planner into `configure_multi_arch_ci.py` in dry-run; validate predicted
   vs. actual.
5. Replace `baseline_runs.py` checks + the "Copy Prebuilt Stages" path in
   `multi_arch_ci_linux.yml` behind a flag; enable trusted reuse for `main` +
   rocm-libraries; retire the legacy search after parity.

## Alternatives

- Extend the baseline-run search — rejected: O(N), API-heavy, cannot express
  arbitrary subsets, fragile on release branches.
- `key -> S3 URL` map instead of a manifest — rejected: loses integrity,
  provenance, trust/validation state, multi-run assembly.
- Dedicated DB service from day one — deferred: store abstracted (ClickHouse),
  backend chosen with real data.

