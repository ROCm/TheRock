# Running the test_artifacts workflow manually

The [`test_artifacts.yml`](/.github/workflows/test_artifacts.yml) workflow can
be triggered manually via `workflow_dispatch` to run tests against pre-built
artifacts.  This guide explains the key inputs and how to use them.

## Quick start examples

### Run tests for a specific component

To test only `rocblas` on `gfx942`:

| Input | Value |
|---|---|
| `artifact_group` | `multi-arch-release` |
| `amdgpu_families` | `gfx94X-dcgpu` |
| `test_type` | `standard` |
| `test_labels` | `test:rocblas` |

### Run tests for multiple components

To test `tensilelite` and `tensilelite-common` (note: the `test:tensilelite`
label automatically expands to both):

| Input | Value |
|---|---|
| `artifact_group` | `multi-arch-release` |
| `amdgpu_families` | `gfx94X-dcgpu` |
| `test_type` | `standard` |
| `test_labels` | `test:tensilelite` |

### Run all tests affected by a project change

To test everything downstream of `rocblas` changes (uses the consumer graph):

| Input | Value |
|---|---|
| `artifact_group` | `multi-arch-release` |
| `amdgpu_families` | `gfx94X-dcgpu` |
| `test_type` | `standard` |
| `changed_projects` | `rocblas` |

## Input reference

### `test_labels` vs `changed_projects`

These two inputs serve different purposes and can be used independently or
together:

- **`test_labels`** — directly selects which test components to run by name.
  Use the `test:<component>` format (e.g. `test:rocblas`, `test:hipblas`).
  Only the named components will run.

- **`changed_projects`** — specifies which *source projects* changed, then
  walks the consumer dependency graph to determine which downstream projects
  need testing.  For example, `changed_projects=rocblas` may also select
  `hipblas` and other consumers of rocblas.  Leave empty to run all tests.

When both are set, a test component must match **both** filters to run.

### `test_labels` format

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

CI control labels (prefixed with `ci:`) can also be included:
- `ci:run-multi-gpu` — force multi-GPU tests to run

### `amdgpu_families` vs `amdgpu_targets`

- **`amdgpu_families`** — the GPU family group string that selects which
  hardware family the tests target.  This is the primary selection mechanism.

  Common values:

  | Value | GPU targets |
  |---|---|
  | `gfx94X-dcgpu` | gfx942 |
  | `gfx110X-all` | gfx1100, gfx1101, gfx1102, gfx1103 |
  | `gfx120X-all` | gfx1200, gfx1201 |
  | `gfx125X-dcgpu` | gfx1250 |
  | `gfx950-dcgpu` | gfx950 |
  | `gfx90a` | gfx90a |

- **`amdgpu_targets`** — optional, specific chip variants within the family
  (e.g. `gfx942`, `gfx1100`).  Leave empty to use the default targets for
  the selected family.

### `test_type`

Controls the depth and duration of testing:

| Type | Duration | Use case |
|---|---|---|
| `quick` | < 5 min | Sanity checks, build system changes |
| `standard` | < 30 min | Component changes, submodule bumps |
| `comprehensive` | < 2 hours | Nightly, extended coverage |
| `full` | 2+ hours | Pre-release, maximum confidence |

See [test_filtering.md](test_filtering.md) for details.

### Other inputs

| Input | Description |
|---|---|
| `artifact_group` | Artifact group name for fetching build artifacts (e.g. `multi-arch-release`) |
| `artifact_run_id` | Previous workflow run ID to retrieve artifacts from; leave empty for latest |
| `test_runs_on` | GitHub Actions runner label for GPU tests; leave empty to skip GPU tests |
| `test_runs_on_cpu` | Runner label for CPU-only tests |
| `build_variant` | Build type: `release`, `asan`, `asan-debug`, `host-asan`, `host-asan-debug` |
| `release_type` | `ci` (default) or `dev` |
| `skip_stages` | Comma-separated build stages to exclude from artifact reuse |
| `sanity_check_only_for_family` | When true, only run sanity checks |
| `repository` | Repository to checkout (defaults to current) |
| `ref` | Branch, tag, or SHA to checkout |

## How it works

1. **Determine test dependencies** — `test_tools/determine_rocm_test_dependencies.py`
   uses the `changed_projects` input to walk the consumer dependency graph and
   produce a list of projects that need testing.

2. **Configure test matrix** — `build_tools/github_actions/fetch_test_configurations.py`
   combines the projects list with `test_labels`, `amdgpu_families`, and
   `test_type` to build the final matrix of test jobs.

3. **Run tests** — Sanity checks run first, then component tests run in
   parallel (with optional sharding).

See also:
- [ci_overview.md](ci_overview.md) — overall CI architecture
- [test_filtering.md](test_filtering.md) — test type definitions
- [ci_behavior_manipulation.md](ci_behavior_manipulation.md) — CI trigger
  behavior and PR labels
