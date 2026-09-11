---
author: Saad Rahim (@saadrahim)
created: 2026-07-28
modified: 2026-07-28
status: draft
---
 
# RFC00XX: Consolidate the ROCm Repositories — Rename TheRock to ROCm, Retire Current ROCm as legacy-rocm-build
 
> [!NOTE]
> **ROCm/ROCm is dead! Long live ROCm/ROCm!**
 
## Problem
 
Two confusingly similar top-level repos:
 
- **`ROCm/ROCm`** — docs/manifest/governance landing page. Little active development.
- **`ROCm/TheRock`** — the active build platform (CMake super-project, CI, RFCs, active issues). Nightlies run from `ROCm/rockrel`, reusing TheRock workflows.
 
Users can't tell which is authoritative and file issues / read release notes / contribute in the wrong place. The most discoverable name (`ROCm/ROCm`) points at the *less* active repo.
 
## Proposal
 
Deliver in **two phases** to decouple the risky rename from the contentious doc/changelog migration.
 
### Phase 1 — Repository rename + build system
 
Swap the names so the build system becomes canonical:
 
- `ROCm/TheRock` → `ROCm/ROCm`
- current `ROCm/ROCm` → `ROCm/legacy-rocm-build`
 
**No git history touched, no docs moved, no changelog changes.** Documentation and changelog keep publishing from `legacy-rocm-build`.
 
Steps:
 
1. **Pre-plan with DevOps/owners** — inventory everything keyed to the exact repo name (Quartz cross-repo hooks, cloud-project resource auth, registered resources, CI incl. `rockrel`). Owners schedule the change.
2. Rename current `ROCm/ROCm` → `legacy-rocm-build` (add a disambiguation notice).
3. Rename `ROCm/TheRock` → `ROCm/ROCm` (back-to-back with step 2, by an org admin).
4. Repoint infra (cloud-resource auth, Quartz) and validate CI/build end-to-end.
5. Update cross-links in other ROCm-org repos (`rocm-libraries`, `rocm-systems`, `.github` PR template); link `legacy-rocm-build` explicitly.
6. **Keep "TheRock" as the build/CI system name** where embedded (`therock-ci.yml`, `-DHIPBLASLT_ENABLE_THEROCK`, `THEROCK_INSTALL_RPATH_ORIGIN`, `docs/development`). Don't strip these; the README disambiguates "ROCm the repo" from "TheRock the build/CI system."
7. **Timing** — target a quiet period (proposed: after the last ROCm release before the holidays). Reversible if issues arise.
 
### Phase 2 — Documentation & changelog consolidation
 
After Phase 1 is stable, on its own timeline, migrate still-relevant docs/changelog into the new `ROCm/ROCm`. Constraints (from reviewer feedback):
 
- **No history merge/rewrite** — ~293 forks make this too disruptive.
- **New `CHANGELOG.md`, no history import** — not folded into `RELEASES.md` (different purposes); historical changelog stays in `legacy-rocm-build`.
- **Retire all legacy `ROCm/ROCm` labels** — the new repo keeps its own set; re-create any label still needed rather than bulk-migrating.
- **Retire `default.xml`** — superseded by `BUILD_TOPOLOGY.toml`; confirm no `repo`-tool/CI consumer breaks first.
- CONTRIBUTING/GOVERNANCE placement TBD (org-wide default likely in `ROCm/.github`, TheRock-specific guidance layered on).
 
## Redirect behavior
 
A name *swap* keeps only one redirect:
 
- `ROCm/ROCm` — live repo (formerly TheRock).
- `ROCm/TheRock` → new `ROCm/ROCm` (**persists**).
- `ROCm/ROCm` → `legacy-rocm-build` — **does not survive** (live repo overrides it).
- `legacy-rocm-build` — reachable only by explicit name.
 
Acceptable: old `ROCm/ROCm` links land on the new active repo. Legacy file deep-links stay reachable at `legacy-rocm-build` paths (Phase 1 keeps docs/changelog there).
 
> [!WARNING]
> The redirect does **not** cover cross-repo Quartz interaction, cloud-resource auth keyed to the repo name, or registered resources. These need real work and careful validation by DevOps/owners — not a settings-only rename.
 
## Motivation
 
- Removes the confusion at the most visible layer — the name.
- One canonical top-level entry point; triage routes mis-filed issues to the correct super-repos.
- Best-known URL fronts the repo users actually need.
- Phased = the reversible rename ships without waiting on the harder doc/governance decisions.
 
## Open Questions
 
- **Single entry point vs. build-focused repo** — some reviewers prefer keeping `ROCm/ROCm` (or `ROCm/ROCm-community`) as a docs/triage entry point, arguing a rename floods the build repo with mis-routed issues. This RFC's position: one entry point with triage routing. Central debate to resolve.
- **"TheRock" name retention** — confirm the README disambiguation so references like "Integration with TheRock" stay discoverable.
- **RFC process location** — TheRock's `docs/rfcs` is build/release-focused; broader ecosystem RFCs may belong elsewhere (`ROCm/community` / a forum).
- **Rename mechanics** — names freed/claimed back-to-back; never recreate a repo at the old paths (breaks the `TheRock` → `ROCm/ROCm` redirect); `legacy-rocm-build` needs explicit linking.
 
## Alternatives Considered
 
- **Do it all at once** (rename + doc/changelog/history merge) — couples a reversible change to contentious migration. Rejected; the two-phase split is the response.
- **Archive current `ROCm/ROCm`, keep TheRock's name** — leaves the best-known name dead. Rejected.
- **Deprecation banner only** — preserves the ambiguity. Rejected.
- **Keep `ROCm/ROCm` as docs/community entry point, TheRock stays the build repo** — main reviewer counter; captured as the central Open Question.
- **Fold TheRock into current `ROCm/ROCm`** — disrupts the active repo's issues/discussions/RFC history. Rejected
