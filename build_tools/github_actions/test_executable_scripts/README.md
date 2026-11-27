# Test Runner

`test_runner.py` is a **generic test runner** used by GitHub Actions to run component tests (e.g. MIOpen). It relies on a standardized CTest labeling scheme so that only tests valid for the current scenario are run.

Around 30 components in `fetch_test_configurations.py` currently use it, replacing what used to be a hand-written `test_<component>.py` per component.

This document describes how the mechanism works, what the component must provide, and how it ties into CI.

______________________________________________________________________

## Overview

1. The script is invoked by the **Test component** workflow with env vars set (component name, GPU arch, test type, sharding).
1. The script expects the component under test to have a ctest based interface, with labels corresponding to test categories like `quick`, `standard`, `comprehensive` and `full` (which can be run for scenarios like quick test, pre-commit etc).
1. The component can opt to always exclude some tests, or to exclude them conditionally based on the OS and GPU where the test is running.
1. The script discovers which **GPU-specific test suites** and **exclude labels** exist by running `ctest --print-labels`.
1. It builds and runs a **ctest** command with the right labels and options (category, exclusions, parallelism, sharding, etc.).

For most components no TheRock-side logic is required beyond the CTest labels plus a `fetch_test_configurations.py` entry. Components with unusual install layouts or environment needs can opt into [Component overrides](#component-overrides).

______________________________________________________________________

## Component contract

For this runner to work, the component's CTest configuration must expose the labels described below. Note that the runner only ever looks at **labels** — test names are never inspected, so components are free to name their tests however they like.

### Category labels

Tests are selected by category label. The valid values of `TEST_TYPE` are:

- `quick`, `standard`, `comprehensive`, `full`
- `ffm-quick`, `ffm-standard`, `ffm-comprehensive`, `ffm-full`

`TEST_TYPE` is lower-cased and validated at startup. An unrecognised value falls back to `quick` with an error printed to stderr. The category is used **as-is**: `TEST_TYPE=quick` selects the `quick` label, `TEST_TYPE=comprehensive` selects `comprehensive`, and so on.

Include labels are matched **exactly**, using an anchored regex:

```
ctest -L "^quick$"
```

The anchoring matters when choosing label names. `ctest -L` performs partial regex matching, so an unanchored `-L full` would also select `multigpu_full` and `ffm-full`. Because the runner anchors, a component can safely define labels that share a prefix or suffix with a tier name.

### GPU exclusion labels

The script runs `ctest --print-labels --test-dir <component>` and collects every label starting with **`ex_gpu_`** whose suffix begins with `gfx` (e.g. `ex_gpu_gfx110X` → `gfx110X`, `ex_gpu_gfx950` → `gfx950`). This yields the set of GPU architectures for which the component defines exclusions.

For a given GPU, the runner adds `-L "^ex_gpu_<arch>$"` when a matching label exists, or `-LE ex_gpu` to exclude all GPU-specific tests when there is no match or no GPU was specified.

These labels mark the ctest entries in which the GPU-specific exclusions have **already been applied**. Selecting `ex_gpu_gfx950` therefore means "run the variant of this category that omits the tests known to fail on gfx950".

### Category exclude labels

In addition to the category label itself, the runner looks for two optional exclude labels and adds them to `-LE`:

| Label                           | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `{category}_exclude`            | Tests the component always excludes from this category                |
| `{category}_therock_ci_exclude` | Tests excluded from this category **only when running in TheRock CI** |

The second one lets a component keep a test enabled in its own CI while skipping it in TheRock's environment, without having to disable it outright.

On the component side these come from `shared/ctest/parse_ctest_categories.py`, which turns **any** category key ending in `exclude` into a label named `{category}_{key}`:

```yaml
test_categories:
  quick:
    test_patterns:
      - test_mi_gpu_spec
    exclude: # -> quick_exclude
      - test_flaky_thing
    therock_ci_exclude: # -> quick_therock_ci_exclude
      - test_profile_live_attach_detach
    rocprofiler_compute_ci_exclude: # -> quick_rocprofiler_compute_ci_exclude
      - test_profile_live_attach_detach
```

Only the first two are consumed by this runner; the third is available for the component's own CI to use and is ignored here.

This applies to the **pre-registered-CTest** integration style (components whose tests are already registered with `add_test()`, e.g. rocwmma, hipdnn, rocprofiler-compute). For components built around a single GTest or Catch2 binary, the shared parsers fold exclusions into the `--gtest_filter` string or the Catch2 tag expression instead of emitting labels, so no `_exclude` label exists and the `-LE` is simply never added.

Multiple exclude patterns are OR-joined into a **single** `-LE` argument:

```
ctest -L "^quick$" -LE "quick_exclude|quick_therock_ci_exclude"
```

Passing several `-LE` flags would AND them, which would only exclude tests matching *every* pattern.

______________________________________________________________________

## Environment variables

| Variable          | Required | Default | Description                                                                         |
| ----------------- | -------- | ------- | ----------------------------------------------------------------------------------- |
| `TEST_COMPONENT`  | yes      | —       | Job name of the component (e.g. `miopen`). The script exits with an error if unset. |
| `THEROCK_BIN_DIR` | yes      | —       | Path to the extracted artifact `bin/` directory.                                    |
| `TEST_TYPE`       | no       | `quick` | Test category. Invalid values fall back to `quick`.                                 |
| `AMDGPU_FAMILIES` | no       | unset   | Parsed for the first `gfx...` token (e.g. `gfx1151`).                               |
| `SHARD_INDEX`     | no       | `1`     | 1-based shard number.                                                               |
| `TOTAL_SHARDS`    | no       | `1`     | Total number of shards.                                                             |

The script derives and exports the following into the ctest environment:

- `ROCM_PATH` — the parent of `THEROCK_BIN_DIR` (i.e. the install prefix).
- `GTEST_SHARD_INDEX` — `SHARD_INDEX - 1`, since GTest's shard index is 0-based.
- `GTEST_TOTAL_SHARDS` — `TOTAL_SHARDS`.

______________________________________________________________________

## Execution flow

1. **Resolve component directory**
   Map `TEST_COMPONENT` (e.g. `miopen`) to the test directory name (e.g. `MIOpen`) via `COMPONENT_DIR_MAPPING`. Job names absent from the map are used verbatim, so only components whose install directory differs from their job name need an entry. The default test directory is `{THEROCK_BIN_DIR}/{TEST_COMPONENT}`, which [component overrides](#component-overrides) may redirect.

1. **Verify tests exist**
   Run `ctest -N --test-dir <test_dir>` and fail if the directory is missing or if zero tests are registered. A missing directory usually means the artifact `.toml` does not bundle the component's install-tree `CTestTestfile.cmake`.

1. **Discover labels**
   Run `ctest --print-labels --test-dir <test_dir>` and collect both the `ex_gpu_*` architectures and the `*_exclude` labels described in the [component contract](#component-contract).

1. **Choose category**
   The category is `TEST_TYPE` as normalized at startup.

1. **Resolve GPU arch**
   Parse `AMDGPU_FAMILIES` for the first `gfx...` token (e.g. `gfx1151`). If it is missing, or is `generic` / `none` / empty, the script excludes all GPU-specific tests with `-LE ex_gpu`.

1. **Match GPU to suite**
   Using `find_matching_gpu_arch()`:

   - Prefer an **exact** match in the discovered set (e.g. `gfx1151`).
   - Else try **wildcard** patterns from most to least specific (e.g. for `gfx1151`: `gfx115X`, then `gfx11X`).
   - If a match is found, add `-L "^ex_gpu_<arch>$"`; otherwise add `-LE ex_gpu`.

1. **Generate a resource spec (optional)**
   If the test directory contains a `generate_resource_spec` executable (currently hipcub, rocthrust and rocprim), run it to produce `resources.json` and pass `--resource-spec-file resources.json`. Without a spec, CTest ignores each test's `RESOURCE_GROUPS` property and GPU tests would run unconstrained under `--parallel`.

1. **Build ctest command**

   - `-L "^<category>$"`, plus `-L "^ex_gpu_<arch>$"` or `-LE ex_gpu`.
   - `-LE` with any discovered exclude labels, OR-joined.
   - `--output-on-failure`, `--timeout 7200`, `--test-dir <test_dir>`.
   - `--parallel <N>` when the effective parallel count is greater than 0. The default is 1 (serial).
   - `-V` unless the component opts out.
   - `--tests-information <SHARD_INDEX>,,<TOTAL_SHARDS>` (see [Sharding](#sharding)).

1. **Run ctest**
   Execute the command with the working directory set to the repository root.

______________________________________________________________________

## Component overrides

`COMPONENT_OVERRIDES` in `test_runner.py` adapts the generic flow to components with unusual layouts. Each key is a job name; the value is a dict of any of:

| Key                        | Type      | Effect                                                                                                                                 |
| -------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `test_dir`                 | list[str] | Test directory path components, relative to `ROCM_PATH`. Overrides the default `bin/<component>`.                                      |
| `test_dir_by_type`         | dict      | Maps `TEST_TYPE` to path components relative to `ROCM_PATH`. Takes precedence over `test_dir` when the current tier matches.           |
| `additional_env_paths`     | dict      | Path components relative to `ROCM_PATH`, prepended to the named env vars (`PATH`, `LD_LIBRARY_PATH`, ...).                             |
| `env_prepend_from_therock` | dict      | Same, but resolved relative to the repository root instead of the install prefix. For tests that load libraries out of the build tree. |
| `env`                      | dict      | Literal env vars to set. Values are run through `str.format()` and support the `{rocm_path}` placeholder.                              |
| `ctest_parallel_count`     | int       | Per-component `--parallel` value. `0` drops the flag entirely (serial).                                                                |
| `ctest_verbose`            | bool      | Defaults to `True`. Set `False` to drop `-V` for components with very large per-test output. `--output-on-failure` still applies.      |

Current users:

- **rocprofiler-compute** — ctest fragments live under `libexec/`, not `bin/`; pinned to serial because its tests are pytest runs that already parallelize internally.
- **rocprofiler-systems** — pytest-driven CTests under `share/`; needs `PATH`/`LD_LIBRARY_PATH` additions plus MPI prefix vars, runs serially, and drops `-V`.
- **rocwmma** — installs three independent CTestTestfile fragments; `test_dir_by_type` routes `quick` to `bin/rocwmma/regression` so the per-target emulation regression runs are exercised.
- **rocroller** — prepends build-tree library directories to `LD_LIBRARY_PATH`.
- **rocshmem** — prepends install and bundled-sysdeps lib directories so its backend-detection probe resolves.

______________________________________________________________________

## Sharding

Sharding is applied along two axes:

- **CTest entry stride** via `--tests-information <SHARD_INDEX>,,<TOTAL_SHARDS>`. The empty middle field means "no end bound", so this selects every `TOTAL_SHARDS`-th ctest entry starting at `SHARD_INDEX`.
- **GTest case level** via `GTEST_SHARD_INDEX` / `GTEST_TOTAL_SHARDS`, which splits cases within each binary.

For components whose category label matches **multiple** ctest entries, combining both axes compounds them and silently drops roughly `(1 - 1/N)` of the suite, because only one (entry × sub-shard) pair runs per shard. Such components are listed in `GTEST_ONLY_SHARDING_COMPONENTS` (currently `rocsparse` and `hipsparse`), for which the `--tests-information` stride is omitted and sharding happens purely at the GTest case level. Single-entry components are unaffected either way.

______________________________________________________________________

## CI Integration

- **Workflow:** `.github/workflows/test_component.yml` runs the test step with env vars such as `TEST_COMPONENT`, `TEST_TYPE`, `AMDGPU_FAMILIES`, `SHARD_INDEX`, `TOTAL_SHARDS`.
- **Test script:** For components that use this mechanism, the test script is set to `python .../test_runner.py` in `build_tools/github_actions/fetch_test_configurations.py` (e.g. MIOpen).
- **Sharding:** The workflow matrix uses `shard_arr` from the same config.

______________________________________________________________________

## Adding a new component to this runner

1. **In the component (CMake/CTest):**

   - Assign each test a category label (`quick` / `standard` / `comprehensive` / `full`).
   - Assign GPU-specific variants the label `ex_gpu_{gpu_arch}` alongside their category labels.
   - Optionally add `exclude` / `therock_ci_exclude` lists so the corresponding `{category}_exclude` labels are generated.
   - Install a `CTestTestfile.cmake` into the component's `bin/` subdirectory so ctest can run from the install tree.

1. **In TheRock:**

   - In `fetch_test_configurations.py`, set the component's `test_script` to `python .../test_runner.py` and set `job_name` (and shards, timeout, etc.) as needed.
   - In `test_runner.py`, add a `COMPONENT_DIR_MAPPING` entry **only if** the install directory name differs from the job name (e.g. `miopen` → `MIOpen`).
   - Add the install directory glob (`bin/<component>/**`) to the component's `artifact-*.toml` test include, so the install-tree `CTestTestfile.cmake` is bundled.

After that, the generic flow (discovery → match GPU → run ctest with the right labels) applies without further changes to `test_runner.py` for that component.

______________________________________________________________________

## Verifying locally

```bash
# List what the installed tree exposes
ctest -N              --test-dir $THEROCK_BIN_DIR/<component>
ctest --print-labels  --test-dir $THEROCK_BIN_DIR/<component>

# Reproduce a CI invocation
THEROCK_BIN_DIR=<path>/bin \
TEST_COMPONENT=<component> \
TEST_TYPE=quick \
AMDGPU_FAMILIES=gfx1151 \
  python build_tools/github_actions/test_executable_scripts/test_runner.py
```

If `ctest --print-labels` reports `No Labels Exist` against the install tree while the build tree looks fine, the component's install-time `CTestTestfile.cmake` is emitting labels in a form that ctest's script interpreter does not expose. Labels must be set with `set_tests_properties(...)`; those set via `set_property(TEST ...)` parse without error but are invisible to `--print-labels` and `-L`.
