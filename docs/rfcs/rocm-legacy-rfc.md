---
author: Saad Rahim (@saadrahim)
created: 2026-07-28
modified: 2026-07-28
status: draft
---
 
# RFC00XX: Consolidate the ROCm Repositories — Rename TheRock to ROCm, Retire the Current ROCm as rocm-legacy
 
## Overview
 
The ROCm organization currently maintains two top-level repositories with overlapping and confusingly similar purposes:
 
- **`ROCm/ROCm`** — a documentation, manifest, and governance landing page. It holds the umbrella `README`, the aggregate `CHANGELOG.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, and a `default.xml` repo manifest, but contains no build system and no source code.
- **`ROCm/TheRock`** — the actively developed unified build platform for HIP and ROCm: a CMake super-project with multi-arch, multi-OS (Linux and native Windows) builds, nightly releases, CI/CD, an active RFC process, and its own issue tracker and discussion forum.
 
Users are getting confused about which of these two repositories is the "real" ROCm. They file issues in the wrong place, read release notes from the wrong source, and are unsure where to contribute. `ROCm/ROCm`'s own README already acknowledges TheRock as the forward path ("A new open-source build platform for ROCm is under development at ROCm/TheRock..."), but the two repos coexist with no formal relationship, and the most prominent, best-known name in the org — `ROCm/ROCm` — points at the repo with the *least* active development.
 
This RFC proposes to resolve the confusion by swapping the names so that the canonical `ROCm/ROCm` name belongs to the actively developed build platform:
 
1. **Rename `ROCm/TheRock` → `ROCm/ROCm`.** TheRock becomes the canonical, discoverable home of the ROCm build platform.
2. **Rename the current `ROCm/ROCm` → `ROCm/rocm-legacy`.** The existing umbrella/docs/manifest repo steps aside under a name that clearly signals its legacy status.
3. **Move the still-relevant documentation and governance content from `rocm-legacy` into the new `ROCm/ROCm`** (formerly TheRock).
 
GitHub automatically creates a redirect from a renamed repository's old path to its new one — **but only while no live repository occupies the old path.** Because this is a name *swap*, the redirects do not both survive: renaming the current `ROCm/ROCm` to `rocm-legacy` creates a `ROCm/ROCm` → `rocm-legacy` redirect, but the subsequent rename of `TheRock` into `ROCm/ROCm` places a live repo at that path, which overrides (destroys) the redirect. The resulting end state is:
 
- `ROCm/ROCm` — the live, active repository (formerly TheRock).
- `ROCm/TheRock` → redirects to the new `ROCm/ROCm`. This redirect persists.
- `ROCm/ROCm` → `rocm-legacy` redirect — **does not survive**; it is clobbered by the live repo taking the name.
- `ROCm/rocm-legacy` — reachable **only by its explicit name**; nothing redirects to it.
 
This behavior is acceptable for the common case: anyone following an old `ROCm/ROCm` link lands on the new active repository, which is the desired outcome. The only casualty is deep-links to specific *legacy file paths* (e.g. the old `CHANGELOG.md` URL), which will now resolve against the new repo's tree and 404 — which is precisely why the legacy content must be migrated into the new repo (with history) rather than left behind, and why `rocm-legacy` must be linked explicitly from the new `ROCm/ROCm`.
 
## Scope
 
### In scope
 
- Renaming `ROCm/TheRock` to `ROCm/ROCm`.
- Renaming the current `ROCm/ROCm` to `ROCm/rocm-legacy`.
- Migrating the umbrella README content (stack overview, component directory, links to docs portals) from `rocm-legacy` into the new `ROCm/ROCm`, **preserving git commit history** for the migrated documentation.
- Migrating `GOVERNANCE.md` and `CONTRIBUTING.md` into the new `ROCm/ROCm` **with their commit history intact**, and updating cross-links from other ROCm-org repos (`rocm-libraries`, `rocm-systems`, etc.) that currently point at the old `ROCm/ROCm` copies.
- Reconciling `rocm-legacy`'s `CHANGELOG.md` with the new `ROCm/ROCm`'s existing `RELEASES.md` into a single canonical release record.
- Redirecting new issue filing to the new `ROCm/ROCm` tracker; triaging/migrating open issues out of `rocm-legacy`.
- Updating `rocm-legacy`'s README to a short notice pointing at the new `ROCm/ROCm`, and freezing it from further active development.
 
### Out of scope
 
- Any change to `ROCm/rocm-libraries` or `ROCm/rocm-systems`, which remain independent component super-repos.
- Changes to TheRock's build architecture, CI pipelines, or existing RFC process (RFC0002, RFC0009, RFC0012, etc.) beyond adding the migrated governance/docs content.
- Changes to the "ROCm" product name or any marketing/branding decision — this is a repository-naming change only.
- A decision on whether to eventually fully archive `rocm-legacy`; this RFC keeps it as a retained, explicitly-linked repo (note: it will have **no inbound redirect** — see Overview) rather than archiving it outright (see Open Questions).
 
## Motivation
 
- **Direct user confusion.** The presenting problem is that users cannot tell which repository is authoritative. Two repos named `ROCm/ROCm` and `ROCm/TheRock`, both surfacing ROCm content, split issue reports, release-note traffic, and contributions across the wrong destinations. Making the actively developed repo *be* `ROCm/ROCm` removes the ambiguity at the most visible layer: the repo name itself.
- **The canonical name should point at the active repo.** `ROCm/ROCm` is the most discoverable, best-SEO'd, most-linked name in the org. Today it points at a repo with no source and little active development. The name swap puts the org's best-known URL in front of the repo people actually need.
- **TheRock is already the de facto direction.** The current `ROCm/ROCm` README already points to TheRock. This formalizes what is implicitly true rather than maintaining two repos with an unclear relationship indefinitely.
- **Redirects keep the highest-traffic paths working.** After the swap, both `github.com/ROCm/TheRock` and `github.com/ROCm/ROCm` resolve to the new active repository (the former via a persistent redirect, the latter because it *is* the new repo). The main ecosystem of links and bookmarks pointing at either top-level repo continues to work on day one. The one exception — deep-links to legacy file paths, which do not survive because `rocm-legacy` has no inbound redirect — is handled by migrating that content into the new repo (see Overview and Risks).
 
## Proposed Approach
 
1. **Pre-swap content audit.** Enumerate every file in the current `ROCm/ROCm` (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `GOVERNANCE.md`, `default.xml`, `docs/`, `tools/`) and map each to its destination in the new `ROCm/ROCm` (formerly TheRock): merge-into-existing-file, new file, or drop.
2. **Stage the documentation migration in TheRock first, preserving history.** Before any rename, bring the migrated documentation (umbrella README content, `GOVERNANCE.md`, `CONTRIBUTING.md`, and relevant `docs/`) into `ROCm/TheRock` using a history-preserving import rather than a copy-paste. Use `git filter-repo` (or `git subtree`) to extract the selected paths from the current `ROCm/ROCm` with their full commit history, then merge that filtered history into TheRock so `git log` and `git blame` continue to attribute the original authorship and change history of each migrated file. Reconcile `CHANGELOG.md` history into `RELEASES.md` as part of the same effort. This ensures the destination is ready — and its provenance intact — the moment the rename happens.
3. **Rename the current `ROCm/ROCm` → `ROCm/rocm-legacy`.** Immediately update its README to a short "this repository has moved" notice pointing at the new location.
4. **Rename `ROCm/TheRock` → `ROCm/ROCm`.** GitHub redirects `ROCm/TheRock` links to the new path automatically. Note the ordering: the old `ROCm/ROCm` name must be freed (step 3) before TheRock can take it.
5. **Update cross-links.** Update references in `rocm-libraries`, `rocm-systems`, and any other ROCm-org repo that links to the old `ROCm/ROCm` `CONTRIBUTING.md`/`GOVERNANCE.md` so they point at the new canonical repo rather than relying on redirects.
6. **Issue tracker transition.** Announce a cutover date; triage `rocm-legacy`'s open issues (migrate still-relevant ones with a reference comment, close stale/duplicates), and pin a notice explaining the move.
7. **Communication.** Post the swap plan and timeline in TheRock's Discussions, link it from both repos' READMEs ahead of the rename, and give a defined notice window (proposed: 30 days) before executing.
 
## Risks and Open Questions
 
- **Rename ordering and timing.** The swap requires freeing the `ROCm/ROCm` name (rename to `rocm-legacy`) *before* TheRock can claim it. There is a brief window between the two renames; the steps should be executed back-to-back by an org admin to minimize it.
- **Redirect collision.** Once `ROCm/TheRock` is renamed to `ROCm/ROCm`, GitHub will redirect `ROCm/TheRock` → `ROCm/ROCm`. Confirm there is no conflict with the freshly-vacated `ROCm/ROCm` → `ROCm/rocm-legacy` redirect, and that a future repo is never created at the old `ROCm/ROCm` path (which would break the legacy redirect).
- **History-preserving import mechanics.** The `git filter-repo`/`subtree` import must be validated so that migrated files retain their commit history and `git blame` attribution, without pulling in unrelated `rocm-legacy` history or bloating TheRock's repo. Path filtering and any path renames (to fit TheRock's directory layout) should be verified on a scratch clone before landing.
- **Manifest tooling dependents.** If `default.xml` is consumed by any `repo`-tool or CI automation, its move to `rocm-legacy` must be validated before dependents break.
- **`default.xml` / manifest ownership.** Decide whether the manifest moves into the new `ROCm/ROCm`, stays in `rocm-legacy`, or is superseded by TheRock's `BUILD_TOPOLOGY.toml`.
- **Governance file placement.** `GOVERNANCE.md` describes the whole org, not just the build platform; open question whether it belongs in the new `ROCm/ROCm`, an org-level `.github` profile repo, or a retained `rocm-legacy`.
- **Archive vs. retain `rocm-legacy`.** This RFC proposes retaining `rocm-legacy` as a redirect-only repo rather than archiving it, to keep its historical issues and changelog reachable. Confirm whether the org prefers a full archive later.
 
## Alternatives Considered
 
- **Archive the current `ROCm/ROCm`, keep TheRock's name.** Deprecate and archive `ROCm/ROCm`, leaving the actively developed repo as `ROCm/TheRock`. Rejected because it leaves the org's most discoverable name (`ROCm/ROCm`) dead/archived while the repo users actually need keeps a non-obvious name ("TheRock"), only partially resolving the confusion.
- **Keep both repos, formalize the pointer only.** Add an explicit deprecation banner to `ROCm/ROCm` without renaming anything. Rejected as a half-measure that preserves the dual-source-of-truth problem and the naming ambiguity.
- **Merge TheRock's build system into the current `ROCm/ROCm`.** Keep `ROCm/ROCm` as the primary repo and fold TheRock into it. Rejected because it would rewrite the actively developed repo's identity, disrupt its issue/discussion/RFC history, and is far more invasive than a rename swap.
- **True git-history merge into a new repo.** Combine both histories into a brand-new repository. Rejected as unnecessarily disruptive and as sacrificing the SEO/link value of the existing `ROCm/ROCm` name that the rename swap preserves.
