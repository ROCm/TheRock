# Test Runner

`test_runner.py` is a **generic test runner** used by GitHub Actions to run component tests (e.g. MIOpen). It relies on a standardized CTest naming and labeling scheme so that only tests valid for the current scenario are run.

This document describes how the mechanism works, what the component must provide, and how it ties into CI.

______________________________________________________________________

## Overview

1. The script is invoked by the **Test component** workflow with env vars set (component name, GPU arch, test type, sharding).
1. The script expects the component under test to have a ctest based interface, with labels corresponding to test categories like quick, standard, comprehensive and full(which can be run for scenarios like quick test, pre-commit etc)
1. The component can opt to always exclude some tests or based on condition like OS and GPU where the test is getting run
1. The script discovers which **GPU-specific test suites** exist by running `ctest --print-labels` and parsing labels of the form `ex_gpu_{gpu_arch}`.
1. It builds and runs a **ctest** command with the right labels and options (category, parallelism, sharding, etc.).

No component-specific logic is required beyond using the expected test names and CTest labels.

______________________________________________________________________

## Test Naming and Labels (Component Contract)

For this runner to work, the component’s CTest configuration must follow this convention.

### CTest labels (discovery)

The script runs `ctest --print-labels --test-dir <component>` and collects every label that starts with **`ex_gpu_`**. The suffix is treated as the GPU architecture (e.g. `ex_gpu_gfx110X` → `gfx110X`, `ex_gpu_gfx950` → `gfx950`). Only labels whose suffix starts with `gfx` are used.

### CTest labels (selection)

- **Category:** Tests are selected by **label** for the run (e.g. “run all tests with label `quick`” or `standard`). The script uses `-L <category>`.
- **GPU-specific suites:** For a given `gpu_arch`, the script looks for a label `ex_gpu_{gpu_arch}` (e.g. `ex_gpu_gfx1150`, `ex_gpu_gfx11X`). It uses `-L ex_gpu_<arch>` when it finds a matching label, or `-LE ex_gpu` to exclude all GPU-specific tests when no match is found or when no GPU is specified.

So the component must assign GPU-specific tests the label `ex_gpu_{gpu_arch}` (and typically a category label like `quick` or `standard`).

______________________________________________________________________

## Execution Flow

1. **Resolve component directory**
   Map `TEST_COMPONENT` (e.g. `miopen`) to the test directory name (e.g. `MIOpen`) via `COMPONENT_DIR_MAPPING`. Fail if the test directory does not exist under `THEROCK_BIN_DIR`.
1. **Discover GPU suites**
   Run `ctest --print-labels --test-dir {THEROCK_BIN_DIR}/{TEST_COMPONENT}`. Collect every label that starts with `ex_gpu_` and whose suffix starts with `gfx` (e.g. `ex_gpu_gfx110X` → `gfx110X`). This yields the set of **available GPU architectures** for which a label exists.
   These are the set of GPU's for which some test exclusions apply - i.e there are some tests which should not be run on these GPU models. The exp_gpu\_ labels with these gpu models points to the ctest entries where the tests are excluded.
1. **Choose category**
   From `TEST_TYPE`: `quick` → `full`, else → `standard`.
1. **Resolve GPU arch**
   Parse `AMDGPU_FAMILIES` for the first `gfx...` token (e.g. `gfx1151`). If missing or generic, the script will exclude all GPU-specific tests (`-LE ex_gpu`).
1. **Match GPU to suite**
   Using `find_matching_gpu_arch()` (defined in `test_runner.py`):

- Prefer **exact** match in the discovered set (e.g. `gfx1151`).
- Else try **wildcard** patterns from most to least specific (e.g. for `gfx1151`: `gfx115X`, then `gfx11X`).
- If a match is found, add `-L ex_gpu_{matching_arch}`; otherwise add `-LE ex_gpu`.

6. **Build ctest command**

- `ctest -L <category>` (and optionally `-L ex_gpu_<arch>` or `-LE ex_gpu`).
- Common options: `--output-on-failure`, `--parallel <N>`, `--test-dir`, `-V`, `--tests-information <SHARD_INDEX>,<TOTAL_SHARDS>`.
- Parallelism: default 8; can be adjusted according to `AMDGPU_FAMILIES`

7. **Run ctest**
   Execute the command in `THEROCK_DIR` with the environment that includes `ROCM_PATH`, `GTEST_SHARD_INDEX`, and `GTEST_TOTAL_SHARDS`.

______________________________________________________________________

## CI Integration

- **Workflow:** `.github/workflows/test_component.yml` runs the test step with env vars such as `TEST_COMPONENT`, `TEST_TYPE`, `AMDGPU_FAMILIES`, `SHARD_INDEX`, `TOTAL_SHARDS`.
- **Test script:** For components that use this mechanism, the test script is set to `python .../test_runner.py` in `build_tools/github_actions/fetch_test_configurations.py` (e.g. MIOpen).
- **Sharding:** The workflow matrix uses `shard_arr` from the same config; the script passes sharding to ctest via `--tests-information` and to GTest via `GTEST_SHARD_INDEX` / `GTEST_TOTAL_SHARDS`.

______________________________________________________________________

## Adding a New Component to This Runner

1. **In the component (e.g. CMake/CTest):**

- Name GPU-specific tests `{target}_{category}_{gpu_arch}_suite`.
- Assign labels `ex_gpu_{gpu_arch}` and the category label (`quick` / `standard` or equivalent).

2. **In TheRock:**

- In `test_filters.py`, add the job name → directory mapping in `COMPONENT_DIR_MAPPING`.
- In `fetch_test_configurations.py`, set the component’s `test_script` to `python .../test_filters.py` and set `job_name` (and shards, timeout, etc.) as needed.

After that, the generic flow (discovery → match GPU → run ctest with the right labels) applies without further changes to `test_filters.py` for that component.
