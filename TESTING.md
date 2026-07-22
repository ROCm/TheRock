# Testing in TheRock

**TheRock is the integration point for build, test, packaging, and release
infrastructure for ROCm Core.** The code here is used by developers building
ROCm from source, CI systems validating pull request contributions
in repositories like [rocm-systems](https://github.com/ROCm/rocm-systems) and
[rocm-libraries](https://github.com/ROCm/rocm-libraries), and release workflows
publishing nightly and stable releases in
[rockrel](https://github.com/ROCm/rockrel) which are trusted by users and
downstream projects.

**TheRock aims to keep ROCm "ready to release" at any time.** Achieving this at
scale requires robust automated tests that detect issues as close as possible
to their source. Early detection limits the impact of regressions and makes
them easier to diagnose and fix, while continuous validation provides the
confidence needed to release frequently.

**Testing must scale across a broad support surface.** ROCm includes
40+ subprojects, supports 25+ GPU targets across multiple hardware generations,
runs on multiple operating systems, is distributed via multiple packaging
formats, and is used by many downstream frameworks. Testing every combination
for every change is not practical, so TheRock layers automated tests according
to their cost and the confidence they provide. Presubmit testing prioritizes
fast, high-signal test suites and configurations with enough capacity to run for
every change. Longer test suites and hardware with limited runner capacity are
exercised through nightly, scheduled, and on-demand testing.

**Testing should be accessible to all contributors.** Wherever possible,
code and automation should be structured so that important behavior can be
tested quickly on commonly available development machines. Local testing usually
provides the fastest feedback, while continuous integration (CI) workflows
provide consistent environments for validating changes across representative
project-wide configurations and component boundaries.

**ROCm Core is built and released as a single product.** Individual subprojects
may validate their own behavior in isolation, then TheRock assembles and tests
those projects together as often as practical throughout development to provide
confidence in cross-component behavior and product-wide properties that
component-level testing cannot evaluate.

_This page describes how these testing layers work together to validate changes
to TheRock and the ROCm subprojects it integrates. It also explains how code and
automation are structured for testing, which validation methods apply to each
feature area, and the known limitations of the current coverage._

______________________________________________________________________

## Table of contents

- [Testing changes to TheRock](#testing-changes-to-therock)
  - [Test categories in TheRock](#test-categories-in-therock)
  - [TheRock feature area: CMake and super-project build logic](#therock-feature-area-cmake-and-super-project-build-logic)
  - [TheRock feature area: GitHub Actions workflows](#therock-feature-area-github-actions-workflows)
  - [TheRock feature area: Python scripts and tools](#therock-feature-area-python-scripts-and-tools)
  - [TheRock feature area: Packaging](#therock-feature-area-packaging)
- [Testing changes to ROCm subprojects with TheRock](#testing-changes-to-rocm-subprojects-with-therock)
  - [Building subprojects through TheRock](#building-subprojects-through-therock)
  - [Testing subprojects through TheRock](#testing-subprojects-through-therock)

______________________________________________________________________

## Testing changes to TheRock

### Test categories in TheRock

Tests for the code in TheRock itself are split into a few broad categories:

- pre-commit and static checks
  - These are fast checks for formatting, linting, repository policies, and more
  - Example checks (see [`.pre-commit-config.yaml`](/.pre-commit-config.yaml)):
    - `actionlint` for GitHub Actions workflow files
    - `black` formatting for Python scripts
- unit tests
  - These are tests for script behavior, runnable on generic hardware
  - Example tests:
    - [`build_tools/tests/build_topology_test.py`](/build_tools/tests/build_topology_test.py)
    - [`build_tools/tests/py_packaging_test.py`](/build_tools/tests/py_packaging_test.py)
    - [`build_tools/github_actions/tests/workflow_dispatch_inputs_test.py`](/build_tools/github_actions/tests/workflow_dispatch_inputs_test.py)
- integration tests
  - These provide validation for build outputs and packages and may run on real hardware
  - Example tests:
    - [`tests/test_artifact_structure.py`](/tests/test_artifact_structure.py)
    - [`tests/test_rocm_sanity.py`](/tests/test_rocm_sanity.py)
    - [`build_tools/packaging/linux/native_linux_package_install_test.py`](/build_tools/packaging/linux/native_linux_package_install_test.py)
    - [`build_tools/packaging/python/templates/rocm/src/rocm_sdk/tests/core_test.py`](/build_tools/packaging/python/templates/rocm/src/rocm_sdk/tests/core_test.py)

Tests in each category should be runnable as part of local development and
also run as part of our CI workflows:

| Test type         | Target test runtime               | CI workflows                                                                                                                                                                                                                                                                                                                                      |
| ----------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pre-commit        | 10 seconds                        | <ul><li>[`.github/workflows/pre-commit.yml`](/.github/workflows/pre-commit.yml)</li></ul>                                                                                                                                                                                                                                                         |
| unit tests        | 5 minutes (independent of builds) | <ul><li>[`.github/workflows/unit_tests.yml`](/.github/workflows/unit_tests.yml)</li></ul>                                                                                                                                                                                                                                                         |
| integration tests | 30 minutes (after builds)         | <ul><li>[`.github/workflows/test_artifacts_structure.yml`](/.github/workflows/test_artifacts_structure.yml)</li><li>[`.github/workflows/test_native_linux_packages_install.yml`](/.github/workflows/test_native_linux_packages_install.yml)</li><li>[`.github/workflows/test_rocm_wheels.yml`](/.github/workflows/test_rocm_wheels.yml)</li></ul> |

Project features should be tested using a combination of these test types
that balance time to signal and representative coverage. For example, Python
packages should have both unit tests for package building and integration tests
for package installation and runtime behavior.

______________________________________________________________________

### TheRock feature area: CMake and super-project build logic

#### Super-project CMake build - Scope

As the centralized build system for ROCm Core, TheRock includes a CMake
super-project using code in:

- CMake files like [`CMakeLists.txt`](/CMakeLists.txt),
  [`cmake/therock_amdgpu_targets.cmake`](/cmake/therock_amdgpu_targets.cmake),
  and [`FLAGS.cmake`](/FLAGS.cmake)
- Topology metadata in [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml)
- Sub-project declarations like [`math-libs/CMakeLists.txt`](/math-libs/CMakeLists.txt)
- Sub-project artifact descriptors like [`math-libs/BLAS/artifact-blas.toml`](/math-libs/BLAS/artifact-blas.toml)
- Scripts used by the build system like [`build_tools/fileset_tool.py`](/build_tools/fileset_tool.py)

The build system supports a broad matrix of configurations:

| Matrix dimension          | Available configurations                                                                        | Typical CI coverage                             |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Operating system          | Linux (multiple distros), WSL, Windows                                                          | Linux (manylinux), WSL, Windows                 |
| Build variant             | Release, Debug, Address Sanitizer (ASan), etc.                                                  | Release                                         |
| AMDGPU build/test targets | `gfx942`, `gfx950`, `gfx1100`, `gfx1200`, etc.                                                  | 1-5 targets (based on test runner availability) |
| Enabled subprojects       | `THEROCK_ENABLE_ALL`, `THEROCK_ENABLE_PROFILER`, etc.                                           | All enabled, subsets as an optimization         |
| Enabled feature flags     | See [`FLAGS.cmake`](/FLAGS.cmake) and [`docs/development/flags.md`](/docs/development/flags.md) | Default values                                  |
| Other CMake options       | `THEROCK_BUILD_TESTING`, `THEROCK_BUNDLE_SYSDEPS`, etc.                                         | Default values                                  |

#### Super-project CMake build - Design for testing

The CMake build system is designed to be reproducible, configurable, and
debuggable:

- We use the same build system for Linux and Windows with minimal branching.
- Subproject builds can be run in isolation and their configured options can be
  viewed via `_init.cmake` and `_toolchain.cmake` files (see
  [`build_system.md`](/docs/development/build_system.md)).
- Build commands are routed through [`teatime.py`](/build_tools/teatime.py)
  so all logs are written to `${build}/logs/`. CI/CD workflow runs upload
  logs to S3 buckets (see [`s3_buckets.md`](/docs/development/s3_buckets.md))
  following the schema in
  [`workflow_outputs.md`](/docs/development/workflow_outputs.md).
- Build performance logs are collected by [ninja](https://ninja-build.org/)
  and uploaded together with other logs using
  [`post_stage_upload.py`](/build_tools/github_actions/post_stage_upload.py).
- Common base CMake option combinations are managed through
  [`CMakePresets.json`](/CMakePresets.json).

Source code:

- First-party dependencies are loaded through pinned git submodules.
- Third-party dependencies are vendored with source mirrors in
  [`third-party/`](/third-party/) (see also
  [`dependencies.md`](/docs/development/dependencies.md) and
  [`git_chores.md#updating-a-third-party-mirror`](/docs/development/git_chores.md#updating-a-third-party-mirror)).
- Workflows generate commit manifest files and diff reports
  ([`manifest_diff.md`](/docs/development/manifest_diff.md)).

Build environments:

- On Linux we recommend building inside a
  [manylinux](https://github.com/pypa/manylinux) Docker container that includes a
  known-working, *minimal* set of dependencies. See
  [`dockerfiles/README.md`](/dockerfiles/README.md).
- Other environments are also supported, though they aren't tested as regularly:
  [`environment_setup_guide.md`](/docs/environment_setup_guide.md).

#### Super-project CMake build - Validation methods

We are evaluating adding unit tests for certain features of the CMake
build system itself, see https://github.com/ROCm/TheRock/pull/6984 for
example.

The CI systems in [TheRock](https://github.com/ROCm/TheRock) and component
repositories like [rocm-systems](https://github.com/ROCm/rocm-systems)
continuously build a few slices through our support matrix. For changes
to build system files, we generally look for

- The build and test jobs in
  [`.github/workflows/multi_arch_ci.yml`](/.github/workflows/multi_arch_ci.yml)
  should not have new failures.
- The build jobs should not significantly regress in duration.
  - _We currently only monitor for this after merge, we'd like to watch these metrics more proactively in the future_
- The build artifacts should not unexpectedly grow in size.
  - _We currently only monitor for this after merge, we'd like to watch these metrics more proactively in the future_

> [!IMPORTANT]
> Certain types of changes benefit from additional validation, such as:
>
> - Adding new subprojects
> - Adjusting support for specific AMDGPU targets
> - Updates to the compiler (llvm-project)
>
> Pull requests that modify key git submodules in TheRock automatically run
> extra CI jobs. These extra CI jobs can be enabled for other PRs through
> the mechanisms documented in
> [ci_behavior_manipulation.md](/docs/development/ci_behavior_manipulation.md).

#### Super-project CMake build - Limitations and known gaps

> [!WARNING]
> The full matrix of all build settings and feature combinations is too expensive
> to test as part of every change, so we rely on a progressively expanding list of
> jobs as part of our CI/CD systems. Some non-default build variants like
> Debug and Address Sanitizer (ASan) also stress the build system and CI servers
> in unique ways so they are particularly costly to test regularly.

> [!TIP]
> As a general reference, here are some metrics for different CI jobs as of
> July 2026:
>
> | Job description                                                                                                    | Wall time | Build runner usage | Test runner usage |
> | ------------------------------------------------------------------------------------------------------------------ | --------- | ------------------ | ----------------- |
> | rocm-systems per-commit CI<br><ul><li>Linux, Windows</li><li>2 GPU families</li><li>"standard" test type</li></ul> | 3 hours   | 4 hours            | 2 hours           |
> | TheRock per-commit CI<br><ul><li>Linux, Windows</li><li>5 GPU families</li><li>"quick" test type</li></ul>         | 4 hours   | 12 hours           | 10 hours          |
> | Nightly releases<br><ul><li>Linux, Windows</li><li>15+ GPU families</li><li>"comprehensive" test type</li></ul>    | 6 hours   | 40+ hours          | 100+ hours        |
>
> Our target is 30 minutes "time to signal" wall time including builds and tests.

______________________________________________________________________

### TheRock feature area: GitHub Actions workflows

#### GitHub Actions workflows - Scope

We use [GitHub Actions](https://github.com/features/actions) in the
[`.github/workflows`](/.github/workflows/) directory for a variety of workflows:

- Lightweight checks: codeql.yml, gitleaks.yml, pre-commit.yml, unit_tests.yml, therock-pr-bot.yml, etc.
- CI/CD workflows: multi_arch_ci.yml, multi_arch_release.yml, etc.
- Other automation: bump_submodules.yml, copy_release.yml, publish_build_manylinux_x86_64.yml

Many of these workflows are central to day-to-day project development and
official releases, so care must be taken to test them thoroughly.

#### GitHub Actions workflows - Design for testing

Workflows can take hours to run and can be difficult to debug, so we follow
these practices to make testing manageable:

- Keep workflows as simple as possible, e.g. by putting logic in
  Python scripts rather than inline Bash and then writing unit tests for those
  scripts (see [this section in `github_actions_style_guide.md`](/docs/development/style_guides/github_actions_style_guide.md#prefer-python-scripts-over-inline-bash)).
- Support running using prebuilt artifacts/packages and minimal matrices for
  efficient testing. For example,
  [`test_rocm_wheels.yml`](/.github/workflows/test_rocm_wheels.yml)
  is used as part of the
  [`multi_arch_ci.yml`](/.github/workflows/multi_arch_ci.yml) workflow which
  builds ROCm fully from source, but it can be run directly against prebuilt
  ROCm Python packages for any specific Python version, runner type, and test
  container image.
- Where possible, support testing workflows in repository forks (see
  ["Working effectively from forks" in `github_actions_debugging.md`](/docs/development/github_actions_debugging.md#working-effectively-from-forks)).
- When workflows and scripts are used across repositories, pin to specific
  commits so workflow runs are reproducible and updates can be tested prior
  to rollout.
  - In https://github.com/ROCm/rocm-libraries and
    https://github.com/ROCm/rocm-systems, "TheRock CI" uses commit pins that
    receive regular update pull requests via
    [`build_tools/github_actions/bump_automation.py`](/build_tools/github_actions/bump_automation.py).
    These pull requests can be reviewed and fixed when there are breaking
    changes to the build system, workflows, or scripts.

#### GitHub Actions workflows - Validation methods

We test our GitHub Actions workflows using a combination of these practices:

- Run workflows through [actionlint](https://github.com/rhysd/actionlint)
  static analysis (this is required via a pre-commit hook).
- Add unit tests like
  [`build_tools/github_actions/tests/workflow_dispatch_inputs_test.py`](/build_tools/github_actions/tests/workflow_dispatch_inputs_test.py)
  where actionlint falls short.
- Test changes to workflows using "CI" and "dev" environments isolated from
  production (see
  ["Testing release workflows" in `github_actions_debugging.md`](/docs/development/github_actions_debugging.md#testing-release-workflows)
  and [`s3_buckets.md`](/docs/development/s3_buckets.md)).
- If a workflow does not run on `pull_request` events, test manually with
  `workflow_dispatch`
  (https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
  and link the test runs in the pull request description.

<!-- TODO: inline code/log example showing what each check catches? -->

#### GitHub Actions workflows - Limitations and known gaps

Cross-repository workflow design and testing is difficult, so we try to limit
such usage and review changes carefully.

> [!WARNING]
> In https://github.com/ROCm/rockrel (our dedicated releases repository with
> tighter access controls) we use unpinned references so nightly releases
> always use the latest code:
>
> ```yml
> uses: ROCm/TheRock/.github/workflows/multi_arch_release.yml@main
> ```
>
> This has been a frequent source of breaks where workflow inputs differ
> across repositories if parity commits are not merged together. See
> https://github.com/ROCm/rockrel/issues/49 for ideas to improve that.

______________________________________________________________________

### TheRock feature area: Python scripts and tools

#### Python scripts and tools - Scope

Most build system and utility scripts are written in Python, not Bash or other
languages. Our GitHub Actions workflows also use Python scripts for the bulk of
their logic (see the
[GitHub Actions workflows](#therock-feature-area-github-actions-workflows)
section above).

#### Python scripts and tools - Design for testing

We test our Python scripts using [pytest](https://docs.pytest.org/), aiming to
follow the style guidelines in
[`python_style_guide.md`](/docs/development/style_guides/python_style_guide.md)
and particularly the
["testing standards" section](/docs/development/style_guides/python_style_guide.md#testing-standards).

> [!TIP]
> These patterns can make testing scripts easier:
>
> - Add `--dry-run` modes to scripts with dangerous or expensive side effects.
> - Design for running locally, iterate via local usage and unit tests, and
>   _then_ integrate into GitHub Actions workflows or other cloud pipelines as
>   needed.

#### Python scripts and tools - Validation methods

All Python unit tests should be run as part of
[`.github/workflows/unit_tests.yml`](/.github/workflows/unit_tests.yml), with
the help of files like
[`build_tools/pyproject.toml`](/build_tools/pyproject.toml).

Note that simple unit tests do not fully replace integration testing using real
build tools, packages, or remote APIs.

#### Python scripts and tools - Limitations and known gaps

> [!WARNING]
> Some tests have been added without including them on CI, which is getting
> fixed via https://github.com/ROCm/TheRock/issues/6927.

> [!WARNING]
> We measure Python code coverage as part of
> [`.github/workflows/unit_tests.yml`](/.github/workflows/unit_tests.yml), but
> we do not yet track it continuously, surface the coverage diff on PRs, or set
> any project-wide or area-specific target percentages.

> [!WARNING]
> A few tests require authenticated API access to use real services instead of
> mocks. These tests are skipped automatically when credentials are missing.

______________________________________________________________________

### TheRock feature area: Packaging

#### Packaging - Scope

The artifacts produced by the build system are assembled into tarballs/archives,
Python packages, and native operating system packages for distribution.

#### Packaging - Design for testing

- Packages should be buildable from ROCm
  [artifacts](/docs/development/artifacts.md) using scripts in
  [`build_tools/packaging/`](/build_tools/packaging/) with documentation in
  [`docs/packaging/`](/docs/packaging/).
- Packages should use dev/nightly/stable versions following
  [`docs/packaging/versioning.md`](/docs/packaging/versioning.md) for
  version/channel sorting and auditability.

#### Packaging - Validation methods

Packages are tested using a combination of these practices:

- Unit tests for package construction scripts
  - Test structural metadata for inputs and outputs, file inclusion/exclusion
    filters, script portability across environments
- Installation tests which check that packages can be installed and used:
  - Package self-tests (example: `rocm-sdk test`, see
    [Python Packaging - Testing](/docs/packaging/python_packaging.md#testing)).
  - Install tests may run on multiple operating systems / distros since we build
    packages to be portably distributed.
  - Integration and regression tests for interactions between multiple packages,
    ensuring that ROCm packages are self-sufficient, don't conflict with system
    packages, and can be used together with other ecosystem packages

> [!TIP]
> Package installation tests should be modeled closely after user-facing install
> instructions. If the installation instructions are complicated or include
> workarounds, aim to improve that at the source rather than apply workarounds
> local to CI tests.

<!-- TODO: inline code samples for each test type? 10 lines max -->

#### Packaging - Limitations and known gaps

> [!WARNING]
> We currently only run packaging-focused tests on packages. This can miss when
> subprojects pass their tests for one package type but fail for another
> package type, such as when
>
> - Python packages are missing multi-arch / kpack split .kpack files
> - Native Linux packages are missing xnack+ files for ASan
>
> See https://github.com/ROCm/TheRock/issues/5384.

______________________________________________________________________

<!-- ### TheRock feature area: CI infrastructure

TODO: document how we test changes to:
* Build containers/dockerfiles
* Self-hosted CPU build runners
* Self-hosted GPU test runners
* Cloud storage buckets
* Cloud lambda functions
* Cloud cache servers
 -->

______________________________________________________________________

## Testing changes to ROCm subprojects with TheRock

This section focuses on how TheRock is used to test changes to ROCm subprojects
such as amd-llvm, hip-clr, RCCL, MIOpen, etc.

Subprojects which get built and released by TheRock as "ROCm core" store their
own source code in separate repositories such as:

- [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
- [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
- [ROCm/llvm-project](https://github.com/ROCm/llvm-project/)

During development in those repositories, changes may be tested with a
combination of

- "Component CI" that can be specialized per subproject and generally exercises
  individual subprojects directly
- "TheRock CI" that leverages the unified build system, builds and tests
  multiple subprojects together, and closely matches the release environment

Attempts are made to "bump" each submodule reference in TheRock regularly (see
[`.github/workflows/bump_submodules.yml`](/.github/workflows/bump_submodules.yml)),
and these "bump PRs" run more exhaustive builds and tests than "TheRock CI" runs
in the subproject source repositories.

Workflows running in https://github.com/ROCm/rockrel build release packages
using code from TheRock and then trigger even more comprehensive tests across
available GPU test runners.

<!-- TODO: diagrams for superrepo -> TheRock -> rockrel -->

### Building subprojects through TheRock

Once a subproject's source code is included in TheRock, it can be integrated
into the build system to produce artifacts by following
[build_system.md - Adding subprojects](/docs/development/build_system.md#adding-sub-projects).
Artifacts produced for subprojects are included in packages using the
configuration code in [`build_tools/packaging/`](/build_tools/packaging/).

> [!TIP]
> See
> [git_chores.md - Adding a new submodule](/docs/development/git_chores.md#adding-a-new-submodule)
> for instructions on how to add new submodules.

> [!WARNING]
> Adding or removing subprojects, or making deep changes to existing
> subprojects, can affect configurations outside of the default CI matrix. Such
> changes may benefit from targeted validation on additional GPU targets,
> platforms, or build variants. See the
> [TheRock feature area: CMake and super-project build logic](#therock-feature-area-cmake-and-super-project-build-logic)
> section above as well as the
> [CI Behavior Manipulation](/docs/development/ci_behavior_manipulation.md)
> docs.

### Testing subprojects through TheRock

#### Subproject build tests

Subprojects can run CPU-only tests during the build by using the
`therock_cmake_subproject_build_test()` function defined in
[`cmake/therock_subproject.cmake`](/cmake/therock_subproject.cmake), like so:

```cmake
# Using ctest
therock_cmake_subproject_build_test(rocjitsu
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
)

# Using a python test runner
therock_cmake_subproject_build_test(amd-comgr
  COMMAND
    "${Python3_EXECUTABLE}" "${_llvm_lit_script}"
    "${CMAKE_BINARY_DIR}/compiler/amd-comgr/build/test-lit" -v
)
```

These build tests can be run from the superproject CMake build via

```bash
cmake --build "${BUILD_DIR}" --target therock-build-tests -- -k 0
```

> [!TIP]
> Logs for build tests can be found next to the other build outputs:
>
> ```diff
>  amd-comgr_build.log
> +amd-comgr_build_test_1.log
> +amd-comgr_build_test_2.log
>  amd-comgr_configure.log
>  amd-comgr_install.log
> ```
>
> See also the
> [Super-project CMake build - Design for testing](#super-project-cmake-build---design-for-testing) section above.

> [!WARNING]
> Build tests currently run _after_ other build steps rather than in parallel
> with them. This can introduce bottlenecks and could be revisited once all
> build tests pass and can be marked blocking, see notes in
> [`.github/workflows/multi_arch_build_portable_linux_artifacts.yml`](/.github/workflows/multi_arch_build_portable_linux_artifacts.yml)
> and
> [`.github/workflows/multi_arch_build_windows_artifacts.yml`](/.github/workflows/multi_arch_build_windows_artifacts.yml)
> as well as https://github.com/ROCm/TheRock/issues/4789.

#### Subproject test runner

Subprojects can run installed component tests, including GPU tests, after the
build by adding a test script to
[`/build_tools/github_actions/test_executable_scripts/`](/build_tools/github_actions/test_executable_scripts/).

Tests should

- Use test filter levels per
  [`docs/development/test_filtering.md`](/docs/development/test_filtering.md)
- Follow the guidelines in
  [`test_executable_scripts/README.md`](/build_tools/github_actions/test_executable_scripts/README.md)
- Use the common
  [`test_executable_scripts/test_runner.py`](/build_tools/github_actions/test_executable_scripts/test_runner.py)

<!-- TODO: more content here and/or in the linked docs

- patterns
    - env vars: SHARD_INDEX, TEST_TYPE, etc.
    - ignore/skip tests filtered by GPU target?
    - tests should run with minimal configuration from a rocm install and not
      be dependent on the test runner script
- anti-patterns
    - copying DLLs as part of the test runner
    - gotchas with using ROCM_PATH, LD_LIBRARY_PATH, LD_PRELOAD
-->

> [!TIP]
> See
> [adding_tests.md](/docs/development/adding_tests.md)
> for instructions on how to add tests for a subproject to TheRock's CI.
