---
author: Joseph Macaranas (jayhawk-commits)
created: 2026-03-11
modified: 2026-03-18
status: in-progress
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
rocm-systems/projects/amdsmi/test/therock/test_amdsmi.py
rocm-libraries/projects/hipblas/test/therock/test_hipblas.py
```

**Pros:** Tightest coupling to source; easy to find from within a project.
**Cons:** No single location to discover all TheRock-level integration test
runners across a repo; some projects already have a `tests/` directory with
different conventions.

### Option B — Centralized `test/therock/` per repo

All TheRock integration test runners for a given repo live under a single
top-level directory:

```
rocm-systems/test/therock/test_amdsmi.py
rocm-systems/test/therock/test_rccl.py
rocm-libraries/test/therock/test_hipblas.py
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

## Chosen Approach

**Initial migration: Option B** — all scripts land in `test/therock/` at the
repo root of `rocm-systems` and `rocm-libraries`. This minimises the number of
codeowner approvals required (one `CODEOWNERS` entry per repo rather than one
per project subdirectory) and provides a single predictable location for CI to
reference.

**Long-term goal: Option A** — once scripts are stable and owned by the right
teams inside each super-repo, they will be moved to sit alongside
`test_categories.yaml` in each project's own test subdirectory. That second move
is a follow-on RFC.

## Migration Plan

### Step 1: Inventory and map all scripts ✅

Produce a complete mapping of every script to its owning repo and target path.
See the [Step 1 Inventory](#step-1-inventory) section below.

### Step 2: Copy scripts into owning repos ✅

Manually copy scripts from `TheRock` at commit `fc47892f1` into `test/therock/`
in each owning repo, opening a PR per repo to seed the canonical location.

- rocm-systems: [PR #4190](https://github.com/ROCm/rocm-systems/pull/4190)
- rocm-libraries: [PR #5563](https://github.com/ROCm/rocm-libraries/pull/5563)

Note: the scripts are not usable from their new locations until Step 3 is
complete and the updated submodule refs have been picked up by TheRock (Step 4).

### Step 3: Finalize scripts in owning repos ✅

For each script, refine the seeded copy in the owning repo:

- **Path resolution**: Replace the hardcoded `SCRIPT_DIR.parent.parent.parent`
  fallback with `Path(os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent).resolve()`
  so scripts work both when the super-repo is a TheRock submodule and when run
  standalone with `THEROCK_DIR` pointing to a checked-out TheRock installation.
  Done in rocm-systems [PR #4197](https://github.com/ROCm/rocm-systems/pull/4197)
  and rocm-libraries [PR #5572](https://github.com/ROCm/rocm-libraries/pull/5572).
- **`github_actions_api` import**: Scripts that import helpers via `sys.path`
  automatically benefit from the corrected `THEROCK_DIR`. `test_rccl.py` required
  an explicit header reorder so `THEROCK_DIR` is defined before the `sys.path`
  insertion that depends on it.
- **`CODEOWNERS`**: Add entries for `test/therock/` in each repo.
- **README**: Add a note explaining how to run the scripts locally.

### Step 4: Update CI workflow references

> **Blocked**: This step requires the submodule refs for `rocm-systems` and
> `rocm-libraries` to be bumped in TheRock, incorporating the changes from
> Steps 2 and 3. Only once those updated refs are present in TheRock can the
> CI callers be safely switched to the new script locations. This will be done
> in a separate PR.

Update all `pytest` invocations that reference
`build_tools/github_actions/test_executable_scripts/test_X.py` to point to the
new canonical locations. Known callers:

- `TheRock/build_tools/github_actions/fetch_test_configurations.py`
- `rocm-systems/.github/workflows/therock-rccl-test-packages-single-node.yml`
- Any additional callers discovered during the Step 1 inventory.

### Step 5: Validate end-to-end

Run a full CI pass with updated workflow references to confirm all tests execute
correctly from their new locations before removing the old copies.

### Step 6: Remove old scripts from `TheRock`

Once CI is green with all workflows pointing to canonical locations, delete
`TheRock/build_tools/github_actions/test_executable_scripts/` (or the subset
of migrated files if any are intentionally retained).

## Step 1 Inventory

The tables below map every file in `test_executable_scripts/` to its owning
repository and target path. Placement follows **Option B** — all scripts for a
given repo land in a single `test/therock/` directory at the repo root.

### Naming convention

Scripts keep their existing `test_<component>.py` names. In the flat
`test/therock/` layout every script shares the same directory, so the component
name in the filename is necessary for disambiguation — there is no redundancy.

Scripts are flagged with the `github_actions_api` helpers they import; see
[TheRock#3968](https://github.com/ROCm/TheRock/issues/3968) for the planned
resolution of that dependency once scripts have landed in their owning repos.

**`fetch_test_configurations.py` callers** — indicates whether and how a script
is invoked from CI (`python`, `pytest`, or `commented-out`). Scripts not listed
there are currently manual-run only.

### rocm-systems (8 scripts)

| Script                        | Project                        | CI caller   | `github_actions_utils` import | Target path                                |
| ----------------------------- | ------------------------------ | ----------- | ----------------------------- | ------------------------------------------ |
| `test_amdsmi.py`              | `projects/amdsmi`              | manual only | —                             | `test/therock/test_amdsmi.py`              |
| `test_aqlprofile.py`          | `projects/aqlprofile`          | `python`    | —                             | `test/therock/test_aqlprofile.py`          |
| `test_hiptests.py`            | `projects/hip-tests`           | `python`    | `is_asan`                     | `test/therock/test_hiptests.py`            |
| `test_rccl.py`                | `projects/rccl`                | `pytest`    | `get_visible_gpu_count`       | `test/therock/test_rccl.py`                |
| `test_rocprofiler_compute.py` | `projects/rocprofiler-compute` | `python`    | —                             | `test/therock/test_rocprofiler_compute.py` |
| `test_rocprofiler_systems.py` | `projects/rocprofiler-systems` | `python`    | —                             | `test/therock/test_rocprofiler_systems.py` |
| `test_rocr-debug-agent.py`    | `projects/rocr-debug-agent`    | `python`    | —                             | `test/therock/test_rocr-debug-agent.py`    |
| `test_rocrtst.py`             | `projects/rocr-runtime`        | `python`    | —                             | `test/therock/test_rocrtst.py`             |

### rocm-libraries (22 scripts + `hipdnn_install_tests/`)

| Script                   | Project                | CI caller                          | `github_actions_utils` import | Target path                           |
| ------------------------ | ---------------------- | ---------------------------------- | ----------------------------- | ------------------------------------- |
| `test_hipblas.py`        | `projects/hipblas`     | `python`                           | `is_asan`                     | `test/therock/test_hipblas.py`        |
| `test_hipblaslt.py`      | `projects/hipblaslt`   | `python`                           | `is_asan`                     | `test/therock/test_hipblaslt.py`      |
| `test_hipcub.py`         | `projects/hipcub`      | `python`                           | —                             | `test/therock/test_hipcub.py`         |
| `test_hipdnn.py`         | `projects/hipdnn`      | `python`                           | —                             | `test/therock/test_hipdnn.py`         |
| `test_hipdnn_install.py` | `projects/hipdnn`      | `python`                           | —                             | `test/therock/test_hipdnn_install.py` |
| `test_hipdnn_samples.py` | `projects/hipdnn`      | `python`                           | —                             | `test/therock/test_hipdnn_samples.py` |
| `test_hipfft.py`         | `projects/hipfft`      | `python`                           | —                             | `test/therock/test_hipfft.py`         |
| `test_hiprand.py`        | `projects/hiprand`     | `python`                           | —                             | `test/therock/test_hiprand.py`        |
| `test_hipsolver.py`      | `projects/hipsolver`   | `python`                           | —                             | `test/therock/test_hipsolver.py`      |
| `test_hipsparse.py`      | `projects/hipsparse`   | `python`                           | —                             | `test/therock/test_hipsparse.py`      |
| `test_hipsparselt.py`    | `projects/hipsparselt` | `python`                           | —                             | `test/therock/test_hipsparselt.py`    |
| `test_miopen.py`         | `projects/miopen`      | manual only¹                       | —                             | `test/therock/test_miopen.py`         |
| `test_rocblas.py`        | `projects/rocblas`     | `python`                           | `is_asan`                     | `test/therock/test_rocblas.py`        |
| `test_rocfft.py`         | `projects/rocfft`      | `python`                           | —                             | `test/therock/test_rocfft.py`         |
| `test_rocprim.py`        | `projects/rocprim`     | `python`                           | —                             | `test/therock/test_rocprim.py`        |
| `test_rocrand.py`        | `projects/rocrand`     | `python`                           | —                             | `test/therock/test_rocrand.py`        |
| `test_rocroller.py`      | `shared/rocroller`     | `python`                           | —                             | `test/therock/test_rocroller.py`      |
| `test_rocsolver.py`      | `projects/rocsolver`   | `python`                           | —                             | `test/therock/test_rocsolver.py`      |
| `test_rocsparse.py`      | `projects/rocsparse`   | `python`                           | —                             | `test/therock/test_rocsparse.py`      |
| `test_rocthrust.py`      | `projects/rocthrust`   | `python`                           | —                             | `test/therock/test_rocthrust.py`      |
| `test_rocwmma.py`        | `projects/rocwmma`     | `python`                           | —                             | `test/therock/test_rocwmma.py`        |
| `hipdnn_install_tests/`  | `projects/hipdnn`      | (used by `test_hipdnn_install.py`) | —                             | `test/therock/hipdnn_install_tests/`  |

¹ The CI `miopen` job uses `test_runner.py` (a generic multi-component runner
that stays in `TheRock`), not `test_miopen.py` directly. `test_miopen.py` is
currently manual-run only.

### Standalone submodules — out of scope for initial migration

These scripts test components that live outside both `rocm-systems` and
`rocm-libraries`. Their destination repos and migration timelines need separate
discussion.

| Script                     | Source set    | Owning repo (tentative) | CI caller | `github_actions_utils` import |
| -------------------------- | ------------- | ----------------------- | --------- | ----------------------------- |
| `test_rocgdb.py`           | `debug-tools` | `ROCm/ROCgdb`           | `python`  | —                             |
| `test_libhipcxx_hipcc.py`  | `math-libs`   | `ROCm/libhipcxx`        | `python`  | —                             |
| `test_libhipcxx_hiprtc.py` | `math-libs`   | `ROCm/libhipcxx`        | `python`  | —                             |

### iree-libs / fusilli plugins — out of scope for initial migration

These scripts test Fusilli provider plugins whose build artifacts come from the
`iree-libs` source set. Their natural home is a Fusilli or IREE repo.

| Script                      | Artifact                    | CI caller     | `github_actions_utils` import |
| --------------------------- | --------------------------- | ------------- | ----------------------------- |
| `test_fusilliprovider.py`   | `fusilli_plugin_test_infra` | commented out | —                             |
| `test_hipblasltprovider.py` | `hipblaslt_plugin`          | `python`      | —                             |
| `test_miopenprovider.py`    | `miopen_plugin`             | `python`      | —                             |

### Stays in TheRock

| Script           | Reason                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `test_sanity.py` | Tests TheRock-level ROCm installation sanity, not a specific component.                                                       |
| `test_runner.py` | Generic multi-component runner driven by `TEST_COMPONENT` env var; currently used as the CI entry point for the `miopen` job. |

### CI caller summary

The primary CI entry point for all scripts is
`build_tools/github_actions/fetch_test_configurations.py`, which constructs
`test_script` strings that reference paths relative to the repo root. That file
and `test_component.yml` will be the main callers to update in Step 4.

The `rocm-systems` workflow
`therock-rccl-test-packages-single-node.yml` also directly invokes
`test_rccl.py` and will need a separate update.

### `github_actions_api` dependency summary

Five scripts have a hard import dependency on helpers from
`TheRock/build_tools/github_actions/github_actions_api.py` (renamed from
`github_actions_utils.py` in TheRock @ `76ad42a63`):

| Helper                  | Used by                                                                       |
| ----------------------- | ----------------------------------------------------------------------------- |
| `is_asan`               | `test_hipblas.py`, `test_hipblaslt.py`, `test_hiptests.py`, `test_rocblas.py` |
| `get_visible_gpu_count` | `test_rccl.py`                                                                |

These scripts resolve the import via `sys.path` constructed from `THEROCK_DIR`,
so the dependency is satisfied as long as `THEROCK_DIR` points to a valid
TheRock checkout. See [TheRock#3968](https://github.com/ROCm/TheRock/issues/3968)
for longer-term cleanup (vendoring, packaging, or local runnability improvements).

## Open Questions

- **`github_actions_api` dependency**: Resolved for Steps 2–3 — scripts use
  `THEROCK_DIR` to locate and import from TheRock's `build_tools/github_actions/`.
  Longer-term resolution (vendoring, packaging) tracked in
  [TheRock#3968](https://github.com/ROCm/TheRock/issues/3968).
- **`hipdnn_install_tests/` subdirectory**: Resolved — migrated alongside the
  three `test_hipdnn_*.py` scripts in rocm-libraries
  [PR #5563](https://github.com/ROCm/rocm-libraries/pull/5563).
- **Step 4 dependency**: CI caller updates in TheRock are blocked until the
  rocm-systems and rocm-libraries submodule refs (carrying Steps 2 & 3) are
  bumped in TheRock. That update and the CI caller changes will be a single PR.
