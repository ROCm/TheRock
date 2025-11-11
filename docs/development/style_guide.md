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

______________________________________________________________________

______________________________________________________________________

______________________________________________________________________

## Appendix (move these into sections)

✅❌

- Pin actions
- Use python scripts instead of inlined bash or bash scripts
- Safe defaults for inputs
- Minimal permissions (read access, no secrets for runs from forks)
- Use cpu runners for builds, separate build and test actions
- Add logging where it helps
- Keep windows and Linux in sync
- Build dependencies from source and bundle them instead of installing -dev packages in build dockerfiles (keep minimal)
- Variable naming: qualify context/usage "rocm_package_version" instead of "version"
- Explicit is better than implicit, no magic
- Prefer to work with upstream more than downstream (e.g. PyTorch build scripts)
- Versions: consistency, compatibility with the ecosystem (can CUDA/CPU packages be installed concurrently with ROCm packages?)
- Path handling with pathlib, relative to script path in repository
- Write for local/developer use first, then have CI/CD follow documented steps
