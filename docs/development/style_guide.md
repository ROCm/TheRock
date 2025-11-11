# TheRock Style Guide

Table of contents:

- [Introduction](#introduction)
- [CMake guidelines](#cmake-guidelines)
- [Python guidelines](#python-guidelines)
- [Bash guidelines](#bash-guidelines)
- [GitHub Actions guidelines](#github-actions-guidelines)

## Introduction

TheRock is the central build/test/release repository for dozens of ROCm
subprojects and external builds. Tooling in this repository is shared across
multiple repositories.

These are some of our guiding principles:

- Optimize for readability and debuggability
- Explicit is better than implicit
- [Don't repeat yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
- [You aren't gonna need it (YAGNI)](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)
- [Keep it simple, silly (KISS)](https://en.wikipedia.org/wiki/KISS_principle)
- Write portable code where possible, across...
  - Operating systems (Linux distributions, Windows)
  - Devices (dcgpu, dgpu, igpu)
  - Software versions (e.g. Python)

### Formatting using pre-commit hooks

We enforce formatting for certain languages using
[_pre-commit_](https://pre-commit.com/) with hooks defined in
[`.pre-commit-config.yaml`](/.pre-commit-config.yaml). See the pre-commit checks
[section in `CONTRIBUTING.md`](/CONTRIBUTING.md#pre-commit-checks) for
instructions on setting up pre-commit.

## CMake guidelines

> [!TIP]
> The "Mastering CMake" book hosted at
> https://cmake.org/cmake/help/book/mastering-cmake/index.html is a good
> resource.

### CMake dependencies

See [dependencies.md](./dependencies.md) for guidance on how to add dependencies
between subprojects and third party sources.

Note that within each superrepo
([rocm-systems](https://github.com/ROCm/rocm-systems),
[rocm-libraries](https://github.com/ROCm/rocm-libraries)), subprojects **must**
be compatible with one another at the same git commit, and TheRock enforces
this.

## Python guidelines

We generally follow the [PEP 8 style guide](https://peps.python.org/pep-0008/)
using the [_Black_ formatter](https://github.com/psf/black). The guidelines here
extend PEP 8 for our projects.

## Bash guidelines

> [!WARNING]
> Bash is **strongly discouraged** for nontrivial usage in .yml GitHub Actions
> workflow files and script files. Prefer to use Python scripts in most cases.

## GitHub Actions guidelines

### Pin action versions to commit SHAs

Pin actions to specific commit SHAs for security and reproducibility.

❌ **Bad:** Using unpinned or branch references

```yaml
- uses: actions/checkout@main  # Branches are regularly updated
- uses: actions/setup-python@v6.0.0  # Tags can be moved (even for releases)
```

✅ **Good:** Pin to specific commit SHA with the semantic version tag in a comment

```yaml
- uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
- uses: docker/setup-buildx-action@c47758b77c9736f4b2ef4073d4d51994fabfe349  # v3.7.1
```

> [!TIP]
> We use
> [Dependabot](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/keeping-your-actions-up-to-date-with-dependabot)
> to automatically update pinned actions while maintaining security.

### Prefer Python scripts over inline Bash

Where possible, put workflow logic in Python scripts.

Python scripts have several benefits:

- **Testable:** Can be tested locally and with unit tests
- **Debuggable:** Easier to debug with standard Python tools
- **Portable:** Works consistently across platforms (Linux/Windows/macOS)
- **Maintainable:** Better error handling and logging support
- **Modular:** Functions can be shared across multiple scripts

> [!TIP]
> Use your judgement for what logic is trivial enough to stay in bash.
>
> Some signs of complicated bash are _conditionals_, _loops_, _regex_,
> _piping command output_, and _string manipulation_.

❌ **Bad:** Complex inline bash logic

```yaml
- name: Process artifacts
  shell: bash
  run: |
    for family in $(echo "${{ inputs.amdgpu_families }}" | tr ',' ' '); do
      if [[ -f "artifacts/${family}/rocm.tar.gz" ]]; then
        tar -xzf "artifacts/${family}/rocm.tar.gz" -C "install/${family}"
        echo "Extracted ${family}"
      else
        echo "::error::Missing artifact for ${family}"
        exit 1
      fi
    done
```

✅ **Good:** Dedicated Python script

```yaml
- name: Process artifacts
  run: |
    python build_tools/process_artifacts.py \
      --families "${{ inputs.amdgpu_families }}" \
      --artifact-dir artifacts \
      --install-dir install
```

### Use safe defaults for inputs

Workflow inputs should have safe default values that work in common scenarios.

Good defaults should:

- **Work without configuration:** Safe defaults that won't trigger production changes
- **Be well-documented:** Clear descriptions explaining when to override
- **Fail safely:** Prefer dev/test behavior over production releases

❌ **Bad:** Unsafe defaults that could trigger unintended releases

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        description: "Type of release to create"
        options:
          - dev
          - nightly
          - stable
        default: nightly  # Unsafe: publishes to production
```

✅ **Good:** Safe defaults that require explicit intent

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        description: "Type of release to create"
        options:
          - dev
          - nightly
          - stable
        default: dev  # Safe: development releases don't affect production

      amdgpu_families:
        type: string
        description: "GPU families to build (comma-separated). Leave empty for default set."
        default: ""  # Empty string handled gracefully in workflow logic
```

### Minimize permissions

Use minimal permissions to limit security exposure.

(read access, no secrets for runs from forks)

### Separate build and test stages

Use CPU runners to build from source and pass artifacts to test runners.

Benefits of separation:

- **Cost optimization:** GPU runners are expensive; use them only when needed
- **Parallelization:** Multiple test jobs can share build artifacts
- **Packaging enforcement:** Testing in this way enforces that build artifacts
  are installable and usable on other machines

❌ **Bad:** Building and testing on expensive GPU runners

```yaml
jobs:
  build_and_test:
    name: Build and Test
    runs-on: linux-mi325-1gpu-ossci-rocm  # Expensive GPU runner
    steps:
      # ...

      - name: Build ROCm artifacts
        run: |
          cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
          cmake --build build

      - name: Run tests on GPU
        run: build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

✅ **Good:** Separate build on CPU runners and test on GPU runners

```yaml
jobs:
  build_artifacts:
    name: Build Artifacts
    runs-on: azure-linux-scale-rocm  # Dedicated CPU runner pool for builds
    steps:
      # ...

      - name: Build ROCm artifacts
        run: |
          cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
          cmake --build build

      # ... Upload artifacts, logs, etc.

  test_artifacts:
    name: Test Artifacts
    needs: build_artifacts
    runs-on: linux-mi325-1gpu-ossci-rocm  # Expensive GPU runner only for tests
    steps:
      # ... Download artifacts, setup test environment, etc.

      - name: Run tests on GPU
        run: build_tools/github_actions/test_executable_scripts/test_hipblas.py
```

## Appendix (move these into sections)

✅❌

- Prefer to work with upstream more than downstream (e.g. PyTorch build scripts)
- Versions: consistency, compatibility with the ecosystem (can CUDA/CPU packages be installed concurrently with ROCm packages?)
- Path handling with pathlib, relative to script path in repository
