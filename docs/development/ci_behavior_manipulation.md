# CI Behavior Manipulation

**Multi-Arch CI** ([`multi_arch_ci.yml`](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml)) is configured by [`configure_multi_arch_ci.py`](../../build_tools/github_actions/configure_multi_arch_ci.py) and reads GPU family definitions from [`amdgpu_family_matrix.py`](../../build_tools/github_actions/amdgpu_family_matrix.py).

## Trigger behavior

The CI pipelines test a growing set of GPU targets depending on trigger type/frequency:

| Trigger type   | Included family groups                                                                                                                             | Notes                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `pull_request` | <ul><li>`amdgpu_family_info_matrix_presubmit`</li></ul>                                                                                            | Common targets with the most test runners       |
| `push`         | <ul><li>`amdgpu_family_info_matrix_presubmit`</li><li>`amdgpu_family_info_matrix_postsubmit`</li></ul>                                             | High priority targets with limited test runners |
| `schedule`     | <ul><li>`amdgpu_family_info_matrix_presubmit`</li><li>`amdgpu_family_info_matrix_postsubmit`</li><li>`amdgpu_family_info_matrix_nightly`</li></ul> | All targets, even those that fail to build      |

### Pull request

CI runs on pull requests if modified files pass the filters in
[`configure_ci_path_filters.py`](../../build_tools/github_actions/configure_ci_path_filters.py).

The following labels may be added to a pull request to modify CI behavior:

| Label or group     | Description                                                                                                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ci:skip`          | Skip all builds and tests                                                                                                                                                                         |
| `ci:run-all-archs` | Build and test all possible architectures                                                                                                                                                         |
| `gfx...`           | Opt-in to building and testing the specified gfx family (e.g. `gfx120X`, `gfx950`)                                                                                                                |
| `test:...`         | Run tests only for the specified projects (e.g. `test:rocthrust`, `test:hipblaslt`). Sets test level to `full` unless overridden by `test_filter:`. Multiple `test:` labels can be combined.      |
| `test_runner:...`  | Run tests on only custom test machines (e.g. `test_runner:oem`). Single-arch CI only.                                                                                                             |
| `test_filter:...`  | Override the test level (e.g. `test_filter:comprehensive`, `test_filter:quick`). Takes priority over all other test level logic. See [test_filtering.md](./test_filtering.md) for allowed values. |

### Push

CI runs on pushes to `main` if modified files pass the filters in
[`configure_ci_path_filters.py`](../../build_tools/github_actions/configure_ci_path_filters.py).

### Schedule

The
[`multi_arch_release.yml`](https://github.com/ROCm/rockrel/blob/main/.github/workflows/multi_arch_release.yml)
workflow in https://github.com/ROCm/rockrel runs once a day. It selects _all_
families, builds release artifacts, and runs comprehensive tests.

### Workflow dispatch

The Multi-Arch CI pipeline can be triggered manually from its GitHub
Actions workflow page:
\[\[ [Multi-Arch CI workflow dispatch](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml) \]\]
Inputs allow per-platform family selection, test label filtering, and prebuilt
stage configuration.

## Prebuilt stages

> [!NOTE]
> This feature is under active development and will evolve as
> automatic stage selection and baseline run lookup are added.
>
> See https://github.com/ROCm/TheRock/issues/3399 for details and
> [`stage_reuse.md`](stage_reuse.md) for the automatic stage-reuse layer
> that builds on the manual inputs described below.

The [Multi-Arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml)
workflow supports skipping individual build stages by copying their artifacts
from a previous workflow run. This will be used in a few scenarios. For example:

- Changes to the rocm-libraries project will use prebuilt artifacts for
  `compiler-runtime`
- Changes to just test scripts or python packages will use prebuilt artifacts for
  all stages

Two workflow inputs control this:

- **`prebuilt_stages`**: Comma-separated list of stage names to skip
  (e.g. `compiler-runtime,runtime-tests,math-libs`). Artifacts for these stages are copied
  from the baseline run instead of being built. Applied to both Linux and
  Windows; stages not present on a platform are ignored.
- **`baseline_run_id`**: The workflow run ID to copy prebuilt artifacts from.
  Required when `prebuilt_stages` is set. Find this in the URL of a previous
  successful Multi-Arch CI run
  (e.g. https://github.com/ROCm/TheRock/actions/runs/22777631940).

> [!IMPORTANT]
> The baseline run must have built the GPU families you want for the current
> run, otherwise the copy will find no matching artifacts.

### Stage names

Stage names come from [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml).

Currently, stage names must be explicitly specified. In the future these may
be computed based on dependencies and a special "all" option may be available.

<!-- TODO: The workflows currently use `contains(prebuilt_stages, 'name')` for
     substring matching, which would break if a stage name is a prefix of
     another. When configure_multi_arch_ci.py generates the stage list
     automatically, switch to a JSON array and use `fromJSON()` + `contains()`
     for exact matching. -->

For now, these are the common configurations used for testing:

```
compiler-runtime
compiler-runtime,runtime-tests,math-libs,comm-libs,debug-tools,dctools-core,profiler-apps,media-libs
```
