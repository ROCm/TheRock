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

| Label or group             | Description                                                                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ci:skip`                  | Skip all builds and tests                                                                                                                                                                         |
| `ci:run-all-archs`         | Build and test all possible architectures                                                                                                                                                         |
| `ci:asan`                  | Enable ASAN CI builds and tests. ASAN CI is skipped by default on PRs unless this label is present.                                                                                               |
| `ci:host-asan`             | Alias for `ci:asan`. Enable ASAN CI builds and tests.                                                                                                                                             |
| `ci:run-multi-gpu`         | Opt-in to running multi-GPU tests.                                                                                                                                                                |
| `ci:cpu-test-only`         | Run only CPU tests (skip GPU tests). Useful for changes that don't require GPU testing.                                                                                                           |
| `ci:platform:linux`        | Run CI only on Linux (skip Windows builds/tests)                                                                                                                                                  |
| `ci:platform:windows`      | Run CI only on Windows (skip Linux builds/tests)                                                                                                                                                  |
| `ci:gfx...`                | Opt-in to building and testing the specified gfx family (e.g. `ci:gfx120X-all`, `ci:gfx950-dcgpu`)                                                                                                |
| `ci:build:gfx...`          | Opt-in to building only (no tests) for the specified gfx family (e.g. `ci:build:gfx94x`, `ci:build:gfx950`). See [Build/Test-specific labels](#buildtest-specific-architecture-labels).           |
| `ci:test:gfx...`           | Opt-in to testing only for the specified gfx family (e.g. `ci:test:gfx94x`). Requires a corresponding build label. See [Build/Test-specific labels](#buildtest-specific-architecture-labels).     |
| `test:...`                 | Run tests only for the specified projects (e.g. `test:rocthrust`, `test:hipblaslt`). Multiple `test:` labels can be combined.                                                                     |
| `test_runner:...`          | Run tests on only custom test machines (e.g. `test_runner:oem`). Single-arch CI only.                                                                                                             |
| `test_filter:...`          | Override the test level (e.g. `test_filter:comprehensive`, `test_filter:quick`). Takes priority over all other test level logic. See [test_filtering.md](./test_filtering.md) for allowed values. |
| `ci:build-native-linux`    | Opt-in to building native Linux packages (deb/rpm).                                                                                                                                               |
| `ci:skip-native-linux`     | Skip native Linux package builds.                                                                                                                                                                 |
| `ci:build-python-packages` | Opt-in to building Python packages.                                                                                                                                                               |
| `ci:skip-python-packages`  | Skip Python package builds.                                                                                                                                                                       |
| `ci:build-pytorch`         | Opt-in to building PyTorch packages.                                                                                                                                                              |
| `ci:skip-pytorch`          | Skip PyTorch package builds.                                                                                                                                                                      |
| `ci:build-jax`             | Opt-in to building JAX packages.                                                                                                                                                                  |
| `ci:skip-jax`              | Skip JAX package builds.                                                                                                                                                                          |

#### Build/Test-specific architecture labels

For fine-grained control over which architectures build vs. test, you can use
`ci:build:gfx...` and `ci:test:gfx...` labels instead of (or in addition to) the
standard `ci:gfx...` labels.

**Use cases:**

- **Build-only (`ci:build:gfx...`)**: Verify compilation for an architecture without
  running tests. Useful when you only need to validate that code compiles for a
  specific GPU target, or when test hardware isn't available/needed.

- **Test-only (`ci:test:gfx...`)**: Force-enable tests for an architecture that
  wouldn't normally run tests on PRs (e.g., due to trigger-based gating). This
  requires a corresponding build label (`ci:gfx...` or `ci:build:gfx...`) since
  tests depend on build artifacts.

**Examples:**

| Labels                               | Result                                             |
| ------------------------------------ | -------------------------------------------------- |
| `ci:build:gfx950`                    | Build for gfx950, skip tests                       |
| `ci:gfx94x`                          | Build and test for gfx94x (standard behavior)      |
| `ci:gfx950` + `ci:test:gfx950`       | Build for gfx950 and force-enable tests            |
| `ci:build:gfx90a` + `ci:test:gfx90a` | Build for gfx90a and force-enable tests            |
| `ci:test:gfx950` (alone)             | **Error**: test label requires corresponding build |

> [!NOTE]
> The `ci:test:gfx...` label will only enable tests if test hardware is available
> for that architecture. If no test runner exists for the family, the label is
> ignored with a warning.

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

#### Running tests against existing artifacts (`test_artifacts.yml`)

The [`test_artifacts.yml`](https://github.com/ROCm/TheRock/actions/workflows/test_artifacts.yml)
workflow runs tests against pre-built artifacts. It can be triggered manually
via workflow dispatch.

**Quick-start example — test a single component:**

| Input | Value |
|---|---|
| `artifact_group` | `multi-arch-release` |
| `amdgpu_families` | `gfx94X-dcgpu` |
| `test_type` | `standard` |
| `test_labels` | `test:rocblas` |

**Quick-start example — test everything downstream of a project change:**

| Input | Value |
|---|---|
| `artifact_group` | `multi-arch-release` |
| `amdgpu_families` | `gfx94X-dcgpu` |
| `test_type` | `standard` |
| `changed_projects` | `rocblas` |

##### `test_labels` vs `changed_projects`

These two inputs serve different purposes and can be used independently or
together:

- **`test_labels`** — directly selects which test components to run by name.
  Use the `test:<component>` format (e.g. `test:rocblas`, `test:hipblas`).
  Only the named components will run (plus sanity, which always runs).

- **`changed_projects`** — specifies which *source projects* changed, then
  walks the consumer dependency graph to determine which downstream projects
  need testing.  For example, `changed_projects=rocblas` may also select
  `hipblas` and other consumers of rocblas.  Leave empty to run all tests.

When both are set, a test component must match **both** filters to run.

##### `test_labels` format

`test_labels` accepts two formats:

1. **Comma-separated string** (recommended for manual runs):
   `test:rocblas,test:hipblas`

2. **Python list literal**:
   `["test:rocblas", "test:hipblas"]`

Label groups expand automatically:

| Label | Expands to |
|---|---|
| `test:rocgdb` | `rocgdb-cpu`, `rocgdb-gpu` |
| `test:tensilelite` | `tensilelite`, `tensilelite-common` |

##### `amdgpu_families` vs `amdgpu_targets`

- **`amdgpu_families`** — the GPU family group string that selects which
  hardware family the tests target. Common values:

  | Value | GPU targets |
  |---|---|
  | `gfx94X-dcgpu` | gfx942 |
  | `gfx110X-all` | gfx1100, gfx1101, gfx1102, gfx1103 |
  | `gfx120X-all` | gfx1200, gfx1201 |
  | `gfx125X-dcgpu` | gfx1250 |
  | `gfx950-dcgpu` | gfx950 |

- **`amdgpu_targets`** — optional, specific chip variants within the family
  (e.g. `gfx942`). Leave empty to use the default targets for the family.

##### `test_type`

Controls the depth and duration of testing. See
[test_filtering.md](test_filtering.md) for details.

| Type | Duration | Use case |
|---|---|---|
| `quick` | < 5 min | Sanity checks, build system changes |
| `standard` | < 30 min | Component changes, submodule bumps |
| `comprehensive` | < 2 hours | Nightly, extended coverage |
| `full` | 2+ hours | Pre-release, maximum confidence |

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
compiler-runtime,runtime-tests,math-libs,comm-libs,debug-tools,dctools-core,profiler-apps,cv-libs,media-libs
```
