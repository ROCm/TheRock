# CI Behavior Manipulation

TheRock CI is controlled by [`configure_ci.py`](../../build_tools/github_actions/configure_ci.py), where it controls push, pull request, workflow dispatch and schedule CI behavior.

## CI (non-multi-arch)

<!-- TODO: restructure this once multi-arch CI is further along

* Selection of which GPUs to build for on each platform
* Selection of which GPUs to test for on each platform
* Selection of which tests to run
* Variants (ASan, etc.)
* Cache behavior, opt-in builds/tests like PyTorch
-->

### Push behavior

For `push`, TheRock CI only runs builds and tests when pushed to the `main` branch. From [`amdgpu_family_matrix.py`](../../build_tools/github_actions/amdgpu_family_matrix.py), TheRock CI collects the AMD GPU families from `amdgpu_family_info_matrix_presubmit` and `amdgpu_family_info_matrix_postsubmit` dictionaries, then runs builds and tests.

### Pull request behavior

For `pull_request`, TheRock CI collects the `amdgpu_family_info_matrix_presubmit` dictionary from [`amdgpu_family_matrix.py`](../../build_tools/github_actions/amdgpu_family_matrix.py) and runs build/tests.

However, if additional options are wanted, you can add a label to manipulate the behavior. The labels we provide are:

- `skip-ci`: The CI will skip all builds and tests
- `run-all-archs-ci`: The CI will build all possible architectures
- `gfx...`: A build and test (if a test machine is available) is added to the CI matrix for the specified gfx family. (ex: `gfx120X`, `gfx950`)
- `test:...`: The full test will run only for the specified label and other labeled projects (ex: `test:rocthrust`, `test:hipblaslt`)
- `test_runner:...`: The CI will run tests on only custom test machines (ex: `test_runner:oem`)
- `test_filter:...`: The CI will run tests based on the specified filter (ex: `test_filter:comprehensive`). [test_filtering.md](./test_filtering.md) has additional information on allowed test filters.

### Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI in [GitHub's ci.yml workflow page](https://github.com/ROCm/TheRock/actions/workflows/ci.yml). To trigger a workflow dispatch, click "Run workflow" and fill in the fields accordingly:

<img src="./assets/ci_workflow_dispatch.png" />

### Schedule behavior

For `schedule` runs, the `CI Nightly` runs everyday at 2AM UTC. This collects all families from [`amdgpu_family_matrix.py`](../../build_tools/github_actions/amdgpu_family_matrix.py), running all builds and tests.

## Prebuilt stages (Multi-Arch CI)

> [!NOTE]
> This feature is under active development and will evolve as
> automatic stage selection and baseline run lookup are added.
>
> See https://github.com/ROCm/TheRock/issues/3399 for details.

The [Multi-Arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml)
workflow supports skipping individual build stages by copying their artifacts
from a previous workflow run. This will be used in a few scenarios. For example:

- Changes to the rocm-libraries project will use prebuilt artifacts for
  `foundation,compiler-runtime`
- Changes to just test scripts or python packages will use prebuilt artifacts for
  all stages

Two workflow inputs control this:

- **`prebuilt_stages`**: Comma-separated list of stage names to skip
  (e.g. `foundation,compiler-runtime`). Artifacts for these stages are copied
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
     another. When configure_ci.py generates the stage list automatically,
     switch to a JSON array and use `fromJSON()` + `contains()` for exact
     matching. -->

For now, these are the common configurations used for testing:

```
foundation,compiler-runtime
foundation,compiler-runtime,math-libs,comm-libs,debug-tools,dctools-core,profiler-apps,media-libs
```
