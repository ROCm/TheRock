---
name: therock-pr-quality
description: TheRock build-repo supplements to the ROCm PR quality base skill. Use for TheRock PR author, review, or pre-merge gating where the change touches the superbuild, submodules/patches, artifact descriptors, or reusable CI workflows. Adds and tightens base rules; never relaxes a base MUST.
argument-hint: '[author | review | pre-merge] [PR URL | branch:<name> | local]'
extends: rocm-pr-quality
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# TheRock PR Quality (overlay)

## Dependency (mandatory — do this first)

Read and apply the `rocm-pr-quality` base skill before anything below. It lives in this same repo at
`skills/rocm-pr-quality/` (`SKILL.md` + `reference.md`). The supplements here only **ADD** rules or
**TIGHTEN** thresholds; they never relax a base MUST-rule. On any conflict, the base MUST-rule wins.

This overlay is for the *build-repo* concerns that the library overlays (e.g. `hipblaslt-pr-quality`)
do not cover: the superbuild itself, submodules and patches, artifact descriptors, and the reusable
CI workflows that every downstream build depends on.

______________________________________________________________________

## Scope

Apply this overlay (in addition to the base) when a PR touches any of:

- The superbuild / build system: root `CMakeLists.txt`, `cmake/`, `THEROCK_ENABLE_*` options,
  component `CMakeLists.txt` under `math-libs/`, `ml-libs/`, `comm-libs/`, `compiler/`, etc.
- Submodules and patches: `.gitmodules` (root and nested), `third-party/`, patch sets applied to
  subprojects.
- Artifact descriptors: `artifact-*.toml`, `BUILD_TOPOLOGY.toml`, and the artifact
  fetch/install tooling (`build_tools/**`, `install_rocm_from_artifacts.py`).
- Reusable CI: `.github/workflows/**` (especially reusable `workflow_call` workflows and their
  callers) and `build_tools/github_actions/**`.

Changes outside these areas follow the base bar only.

______________________________________________________________________

## Canonical TheRock references

Consult and cite these when reviewing build-repo PRs:

- Build system overview: `docs/development/build_system.md`.
- Style guides: `docs/development/style_guides/` —
  [python](../../docs/development/style_guides/python_style_guide.md),
  [cmake](../../docs/development/style_guides/cmake_style_guide.md),
  [bash](../../docs/development/style_guides/bash_style_guide.md),
  [github_actions](../../docs/development/style_guides/github_actions_style_guide.md).
- Formatting: `.pre-commit-config.yaml` (run `pre-commit run --all-files`).

Cite the specific guide section for a style finding.

______________________________________________________________________

## PR-policy gate

The source of truth for TheRock's contributing policies is the repo's
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) — follow it first. In practice those policies are enforced
by an automated gate (`therock_pr_bot`), which is the hoop every PR must actually clear before it can
be reviewed. Treat that gate as **authoritative**: conform the PR to it before author or pre-merge
sign-off, it overrides this skill's waivers and self-evident exemptions, and the skill never works
around it. This overlay only points at the gate; it does not restate the gate's rules, so if the gate
is later corrected to match `CONTRIBUTING.md`, the overlay needs no change.

When you advise the author to do something solely to clear the gate that is **not** stated in
`CONTRIBUTING.md`, say so explicitly — name it as a gate requirement, not a guide requirement — so the
author knows where it came from (for example: "the bot requires a resolving `ISSUE ID` / `JIRA ID`
line even though the contributing guide doesn't, so add one"). `skills/therock_pr_bot/FAQ.md` explains
how to clear a specific failure.

______________________________________________________________________

## Supplements

### Adds — change classes (bind to TheRock paths)

On top of the base classes, tag TheRock PRs with:

- `submodule-bump` — advancing a submodule pointer (and/or its patch set).
- `superbuild-cmake` — changes to superbuild options / component wiring.
- `artifact-descriptor` — changes to `artifact-*.toml` / `BUILD_TOPOLOGY.toml`.
- `reusable-ci` — changes to a reusable workflow or its callers.
- `dependency-add` — adding a new third-party dependency or subproject.

### Adds — submodules & patches review checks

- A submodule pointer bump is a real change: the PR must say **what** moved and **why** (the target
  commit/range and its purpose), and link the upstream PR/commit. An unexplained pointer move is an
  M5 gap.
- Pointer bumps and patch-set edits must be **intentional and isolated** — flag an incidental
  pointer move bundled into an unrelated PR (the "undeclared submodule drift" smell).
- Patches under a subproject must still apply cleanly against the new pointer; a patch that no longer
  applies (or is now upstreamed and redundant) is `BLOCKING`.
- New/changed patches need a one-line rationale and, where it exists, a link to the upstreaming
  effort so the patch can later be dropped.

### Adds — superbuild vs sub-project CMake

- Distinguish superbuild-level changes (root `CMakeLists.txt`, `cmake/`, `THEROCK_ENABLE_*`) from
  sub-project CMake; a change in the wrong layer is a maintainability finding.
- A new `THEROCK_ENABLE_*` (or similar) option needs a sane default, a help string, and a note on
  how it interacts with existing components. See `docs/development/build_system.md` and the CMake
  style guide for conventions.
- Enabling a component by default (or changing a default) is a behavior change → treat as the base
  `heuristic/default-selection` class (safe default + flag/tracker).

### Adds — artifact descriptor checks

- No **duplicate component ownership**: a given component/file should be claimed by exactly one
  `artifact-*.toml`. Flag two descriptors claiming the same thing.
- TOML components must match what the build actually provides — the components listed line up with
  the `therock_provide_artifact()` (or equivalent) calls that produce them.
- **Stale-descriptor-after-split**: when a component is split/moved, the old descriptor must be
  updated, not left pointing at the pre-split layout.
- `BUILD_TOPOLOGY.toml` stays consistent with the descriptors and component graph (no orphaned or
  dangling entries).
- Changes to artifact fetch/install flags (`install_rocm_from_artifacts.py` and friends) keep the
  documented flags and defaults in sync.

### Adds — reusable CI workflow wiring

- When a reusable (`workflow_call`) workflow's interface changes, **all callers** are updated in the
  same PR; a caller left on the old interface is `BLOCKING`.
- Reusable-workflow inputs are read via `inputs.*`, not `github.event.inputs.*` (that only exists for
  `workflow_dispatch`); flag the mismatch.
- `runs-on` for self-hosted/specialized runners is pinned to the intended label set, not a generic
  default that will mis-route the job.
- Multi-checkout wiring (TheRock + a submodule/sibling repo) is correct: paths, fetch depth, and
  submodule flags are set for what the job actually needs.
- No complex inline `bash` embedded in YAML — non-trivial logic belongs in a script under
  `build_tools/` (per the GitHub Actions style guide), where it is testable and lintable.
- Any script a workflow calls has its runtime dependencies declared/available on the runner.

### Adds — dependency / subproject additions

A `dependency-add` PR must state: the **build-time and binary-size** impact (with a before/after
metric where feasible), the **license** and its compatibility, and the **maintenance owner**. A new
dependency with none of these is `IMPORTANT` at minimum, `BLOCKING` if it lands on a default/shipping
path.

### Tightens — stale-base on high-coupling build files

Make the base pre-merge stale-base check concrete for TheRock. High-coupling files:
root `CMakeLists.txt`, `cmake/**`, `.gitmodules`, `artifact-*.toml`, `BUILD_TOPOLOGY.toml`, and
reusable workflows under `.github/workflows/`. Overlap with the base branch on any of these since the
PR diverged → **mandatory** rebase + re-run (base default is strong-recommend), because these break
combinations that neither PR's own CI can see.

______________________________________________________________________

## What the overlay cannot do

Drop the regression/test obligations (M1/M2), allow disabling tests to green CI (M3), or skip work
tracking/linking (M4/M5). Those are base MUSTs; this overlay can only make them stricter or bind them
to TheRock build paths.
