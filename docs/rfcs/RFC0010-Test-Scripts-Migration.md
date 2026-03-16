---
author: Joseph Macaranas (jayhawk-commits)
created: 2026-03-11
modified: 2026-03-11
status: draft
discussion: https://github.com/ROCm/TheRock/discussions/3964
---

# Migrate Test Executable Scripts to Owning Project Repositories

This RFC proposes moving the test runner scripts in
`build_tools/github_actions/test_executable_scripts/` out of `TheRock` and
into the `rocm-systems` and `rocm-libraries` repositories where the components
they test actually live.

## Motivation

`TheRock` currently hosts ~35 pytest runner scripts that exercise binaries from
components whose source code resides in `rocm-systems` and `rocm-libraries`. This
creates a structural mismatch:

- **Discoverability**: Developers working in `rocm-libraries` or `rocm-systems`
  must know to look in `TheRock` to find or modify their component's test runner.
- **Ownership**: Test scripts should be owned and reviewed by the same
  team/codeowners as the source they test. As-is, component teams must submit
  PRs to a different repo for test changes.
- **Co-evolution**: When a project's binary interface or test binary changes,
  the test script needs to change too. Keeping them in the same repo enables
  this as a single atomic commit.
- **Scalability**: As more components are added, `test_executable_scripts/`
  grows without clear ownership boundaries.

## Placement Options

Before migration begins, we need to agree on a canonical location for scripts
within the owning repos. Three options are considered:

### Option A — Co-located in each project subdirectory

Scripts live alongside the project source under a standard subdirectory:

```
rocm-systems/projects/amdsmi/tests/therock/test_amdsmi.py
rocm-libraries/projects/hipblas/tests/therock/test_hipblas.py
```

**Pros:** Tightest coupling to source; easy to find from within a project.
**Cons:** No single location to discover all TheRock-level integration test
runners across a repo; some projects already have a `tests/` directory with
different conventions.

### Option B — Centralized `tests/therock/` per repo

All TheRock integration test runners for a given repo live under a single
top-level directory:

```
rocm-systems/tests/therock/test_amdsmi.py
rocm-systems/tests/therock/test_rccl.py
rocm-libraries/tests/therock/test_hipblas.py
```

**Pros:** Single location to discover and run all TheRock integration tests
within a given repo. Consistent, predictable path for CI to reference.
**Cons:** Slight separation from per-project source.

### Option C — Reorganize within `TheRock`

Keep all scripts in `TheRock` but move them to a more intentional home such as
`tests/integration/`, making the intent explicit without distributing them.

**Pros:** No cross-repo coordination required.
**Cons:** Does not resolve the ownership or discoverability problems; component
teams still submit test PRs to the wrong repo.

## Migration Plan

### Step 1: Inventory and map all scripts

Produce a complete mapping of every script to its owning repo and target path.

### Step 2: Create a temporary sync workflow in `TheRock`

Add a GitHub Actions workflow that copies scripts **from
`TheRock`'s existing `test_executable_scripts/`** into the agreed canonical
paths in `rocm-systems` and `rocm-libraries` (e.g., opening a PR in each repo).
This seeds the owning repos from the current source of truth so that Step 3
work (cleanup, CODEOWNERS, path fixes) can happen there without starting from
scratch. This workflow is **temporary** and will be removed in the final step.

### Step 3: Finalize scripts in owning repos

For each script, refine the seeded copy in the owning repo:

- Remove `TheRock`-specific path assumptions (e.g., `SCRIPT_DIR.parent.parent.parent`)
  and replace with environment-variable-driven paths already used by many scripts
  (e.g., `THEROCK_BIN_DIR`).
- Resolve the `github_actions_utils` import dependency — either vendor the
  needed helpers, expose them as a small installable package, or accept an
  import from a checked-out `TheRock` path.
- Add or update `CODEOWNERS` entries for the new path.
- Add a brief docstring or `README` note explaining how to run the script.

### Step 4: Update CI workflow references

Update all `pytest` invocations that reference
`build_tools/github_actions/test_executable_scripts/test_X.py` to point to the
new canonical locations. Known callers:

- `TheRock/.github/workflows/unit_tests.yml`
- `rocm-systems/.github/workflows/therock-rccl-test-packages-single-node.yml`
- Any additional callers discovered during the Step 1 inventory.

### Step 5: Validate end-to-end

Run a full CI pass with updated workflow references to confirm all tests execute
correctly from their new locations before removing the old copies.

### Step 6: Remove old scripts and sync workflow

Once CI is green with all workflows pointing to canonical locations:

- Delete `TheRock/build_tools/github_actions/test_executable_scripts/` (or
  the subset of migrated files if any are intentionally retained).
- Delete the temporary sync workflow from `TheRock`.

## Open Questions

- **`github_actions_utils` dependency**: Several scripts import shared helpers
  from a relative path into `TheRock`. The resolution strategy (vendor, package,
  or reference) should be decided before Step 3 begins.
- **`hipdnn_install_tests/` subdirectory**: This directory lives alongside the
  `.py` scripts and must be migrated together with the three `test_hipdnn_*.py`
  scripts.
