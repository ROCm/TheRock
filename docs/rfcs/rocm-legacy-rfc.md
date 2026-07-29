---
author: Saad Rahim (@saadrahim)
created: 2026-07-28
modified: 2026-07-28
status: draft
---
 
# RFC00XX: Consolidate the ROCm Repositories — Rename TheRock to ROCm, Retire Current ROCm as rocm-legacy
 
## Overview
 
The org maintains two confusingly similar top-level repos:
 
- **`ROCm/ROCm`** — docs/manifest/governance landing page (`README`, `CHANGELOG.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, `default.xml`). No source, little active development.
- **`ROCm/TheRock`** — the actively developed unified build platform: CMake super-project, multi-arch/multi-OS builds, nightly releases, CI, RFC process, active issues/discussions.
 
Users can't tell which is authoritative and file issues, read release notes, and contribute in the wrong place. The org's most discoverable name (`ROCm/ROCm`) points at the *less* active repo.
 
**Proposal:** swap the names so the canonical name follows the active repo.
 
1. Rename `ROCm/TheRock` → `ROCm/ROCm`.
2. Rename current `ROCm/ROCm` → `ROCm/rocm-legacy`.
3. Migrate still-relevant docs/governance from `rocm-legacy` into the new `ROCm/ROCm`, **preserving git history**.
 
## Redirect behavior (important)
 
A name *swap* does not preserve both redirects. Renaming current `ROCm/ROCm` → `rocm-legacy` creates a `ROCm/ROCm` → `rocm-legacy` redirect, but then renaming TheRock *into* `ROCm/ROCm` puts a live repo on that path, which overrides (destroys) that redirect. End state:
 
- `ROCm/ROCm` — live repo (formerly TheRock).
- `ROCm/TheRock` → redirects to new `ROCm/ROCm` (**persists**).
- `ROCm/ROCm` → `rocm-legacy` redirect — **does not survive**.
- `rocm-legacy` — reachable only by explicit name; nothing redirects to it.
 
This is acceptable: old `ROCm/ROCm` links land on the new active repo. The one casualty is deep-links to legacy *file paths* (e.g. old `CHANGELOG.md`), which will 404 against the new tree — hence migrating legacy content into the new repo and linking `rocm-legacy` explicitly.
 
## Scope
 
**In scope:** the two renames; history-preserving migration of README content, `GOVERNANCE.md`, `CONTRIBUTING.md`, and relevant `docs/` into the new `ROCm/ROCm`; reconciling `CHANGELOG.md` into `RELEASES.md`; updating cross-links in other ROCm-org repos; issue-tracker cutover; a "moved" notice on `rocm-legacy`.
 
**Out of scope:** changes to `rocm-libraries`/`rocm-systems`; changes to TheRock's build/CI/RFC process beyond adding migrated docs; the "ROCm" product name/branding; any eventual full archive of `rocm-legacy` (retained as an explicitly-linked repo, no inbound redirect).
 
## Motivation
 
- **Removes the confusion directly** — the active repo *becomes* `ROCm/ROCm`, resolving ambiguity at the most visible layer, the name.
- **Canonical name → active repo** — the best-known, best-SEO'd URL now fronts the repo users actually need.
- **Formalizes the de facto direction** — current `ROCm/ROCm`'s README already points to TheRock.
- **Low risk for common traffic** — top-level links to either repo resolve to the new active repo on day one.
 
## Proposed Approach
 
1. **Audit** current `ROCm/ROCm` files; map each to a destination (merge / new / drop).
2. **Stage migration in TheRock first, with history** — use `git filter-repo` (or `subtree`) to import the selected docs paths with full commit history so `git log`/`blame` stay intact; reconcile `CHANGELOG.md` into `RELEASES.md`.
3. **Rename** current `ROCm/ROCm` → `rocm-legacy`; add a "moved" notice.
4. **Rename** `ROCm/TheRock` → `ROCm/ROCm` (must follow step 3, back-to-back, by an org admin).
5. **Update cross-links** in `rocm-libraries`, `rocm-systems`, etc. to the new canonical repo; link `rocm-legacy` explicitly.
6. **Issue cutover** — announce a date; triage/migrate open `rocm-legacy` issues; pin a notice.
7. **Communicate** — post plan/timeline in Discussions; 30-day notice before executing.
 
## Risks & Open Questions
 
- **`rocm-legacy` has no inbound redirect** — must be linked directly from the new README, pinned issues, and release notes.
- **Legacy deep-links 404** — inventory legacy-only file paths so the new repo can point at their `rocm-legacy` equivalents.
- **Rename ordering** — the `ROCm/ROCm` name must be freed before TheRock claims it; execute both renames back-to-back.
- **Never recreate a repo** at the old paths (would break the persistent `TheRock` → `ROCm/ROCm` redirect).
- **History import mechanics** — validate on a scratch clone; don't pull in unrelated legacy history or bloat the repo; verify any path renames.
- **`default.xml` dependents** — validate `repo`-tool/CI consumers before moving it; decide whether it moves, stays, or is superseded by `BUILD_TOPOLOGY.toml`.
- **`GOVERNANCE.md` placement** — new `ROCm/ROCm`, an org `.github` profile repo, or retained `rocm-legacy`?
 
## Alternatives Considered
 
- **Archive current `ROCm/ROCm`, keep TheRock's name** — leaves the best-known name dead and users still hunting for "TheRock." Rejected.
- **Deprecation banner only, no rename** — preserves the dual-source and naming ambiguity. Rejected.
- **Fold TheRock into current `ROCm/ROCm`** — rewrites the active repo's identity and disrupts its issues/discussions/RFC history. Rejected.
- **True git-history merge into a new repo** — needlessly disruptive; sacrifices the existing `ROCm/ROCm` name value. Rejected.
