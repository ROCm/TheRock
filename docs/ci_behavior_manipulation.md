# CI Behavior Manipulation

TheRock CI is controlled by the workflows in [`.github/workflows`](../.github/workflows).
rocm-libraries uses
[`therock-multi-arch-ci.yml`](../.github/workflows/therock-multi-arch-ci.yml)
as the default TheRock CI path. The classic single-arch workflow,
[`therock-ci.yml`](../.github/workflows/therock-ci.yml), remains available as
an explicit fallback.

## Default behavior for push and pull request

TheRock Multi-Arch CI runs by default for pull requests and pushes to
`develop` and `release/therock-*`.

The multi-arch workflow uses
[`configure_external_repo_ci.py`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/configure_external_repo_ci.py)
with [`repos-config.json`](../.github/repos-config.json) to determine file
changes, changed projects, and tests to run.

Example: a change made to `projects/rocfft` will run the tests selected for
that project.

For CI changes or changes that otherwise require broader coverage, TheRock
Multi-Arch CI runs all tests.

The classic single-arch TheRock CI workflow is skipped by default for pull
requests and pushes.

## Pull request behavior

Here are additional labels that manipulate CI behavior. The labels we provide
are:

- `ci:skip`: skip TheRock Multi-Arch CI builds and tests.
- `ci:run-all-archs`: run TheRock Multi-Arch CI for all known GPU families.
- `ci:exactly`: treat the attached labels as the complete specification. See
  [Running exactly what you label](#running-exactly-what-you-label) below.
- `ci:single-arch`: run the classic single-arch TheRock CI fallback workflow.
- `ci:asan` or `ci:host-asan`: enable the TheRock Multi-Arch CI ASAN
  workflow for the pull request.
- `gfx...`: add a GPU family to the TheRock Multi-Arch CI run, such as
  `gfx950`.
- `test:<project>`: request tests for a specific project, such as
  `test:rocprim` or `test:hipblaslt`.
- `test_filter:<level>`: override the multi-arch test level. Valid values are
  `quick`, `standard`, `comprehensive`, and `full`.
- `test_runner:<kernel>`: use a custom test runner where the selected GPU
  family supports one.

- `skip-therockci`: skip the classic single-arch TheRock CI fallback workflow.
- `test_type:<level>`: override the classic single-arch test level. Valid
  values are `quick`, `standard`, `comprehensive`, and `full`.

The classic single-arch labels affect only
[`therock-ci.yml`](../.github/workflows/therock-ci.yml) when that fallback
workflow is enabled.

### Running exactly what you label

By default the `ci:gfx*` labels *add* to the presubmit GPU families, so a pull
request labeled `ci:gfx110X-all` builds the presubmit set **and** gfx110X.

Add `ci:exactly` to turn the attached labels into the complete specification
instead. The trigger defaults are dropped, so `ci:exactly` plus
`ci:gfx110X-all` builds and tests gfx110X and nothing else. This also applies
to package builds: Python package, PyTorch, JAX, and native package jobs are
skipped unless the matching `ci:build-*` label is attached.

`test:<project>`, `test_filter:<level>`, and `ci:platform:*` keep their usual
meaning, so `ci:exactly` composes with them to pin a run down to a single
architecture, platform, and component.

Two combinations are rejected to avoid a run that silently covers nothing:

- `ci:exactly` without any `ci:gfx*` or `ci:build:gfx*` label, which would
  build no architectures at all.
- `ci:exactly` together with `ci:run-all-archs`, which ask for opposite things.

`ci:exactly` applies to pull requests only. Pushes, schedules, and workflow
dispatch runs already state their coverage explicitly.

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI from GitHub Actions:

- [TheRock Multi-Arch CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci.yml)
- [TheRock CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-ci.yml)

Use workflow dispatch for one-off runs that need custom GPU families, test
labels, package build options, or the classic single-arch fallback without
adding a pull request label.

Python package, PyTorch, JAX, and native Linux package builds are not enabled
by default in the pull request and push multi-arch workflow. They can be
enabled through workflow dispatch inputs when needed.

## Schedule behavior

The scheduled nightly TheRock workflow is
[TheRock Multi-Arch Nightly CI](https://github.com/ROCm/rocm-libraries/actions/workflows/therock-multi-arch-ci-nightly.yml).
The classic single-arch nightly workflow is available for workflow dispatch
only.
