# TheRock Style Guide

> [!IMPORTANT]
> This is a living document meant to steer developers towards agreed upon best
> practices.
>
> 📝 Feel free to propose new sections and amend (or remove) existing sections.

Table of contents:

- [Introduction](#introduction)
- [Language and Tool-Specific Guides](#language-and-tool-specific-guides)
  - [Python Style Guide](python_style_guide.md)
  - [CMake Style Guide](cmake_style_guide.md)
  - [Bash Style Guide](bash_style_guide.md)
  - [GitHub Actions Style Guide](github_actions_style_guide.md)

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
- Collaborate with upstream projects

### Formatting using pre-commit hooks

We enforce formatting for certain languages using
[_pre-commit_](https://pre-commit.com/) with hooks defined in
[`.pre-commit-config.yaml`](/.pre-commit-config.yaml).

To get started with pre-commit:

```bash
# Download.
pip install pre-commit

# Run locally on staged files.
pre-commit run

# Run locally on all files.
pre-commit run --all-files

# (Optional but recommended)
# Install git hook.
pre-commit install
```

## Language and Tool-Specific Guides

For detailed guidelines on specific languages and tools, see the dedicated style guides:

### [Python Style Guide](python_style_guide.md)

Comprehensive Python coding standards including:

- **Core Principles:** Fail-fast behavior, dataclasses vs tuples, type hints, error handling
- **Style Guidelines:** Using pathlib, argparse, type hints, `__main__` guards
- **Code Review Checklist:** Pre-submission verification steps
- **Testing Standards:** How to write effective tests
- **Common Patterns:** Reusable code examples for common tasks

We follow [PEP 8](https://peps.python.org/pep-0008/) and use the
[_Black_ formatter](https://github.com/psf/black) (run automatically as a
[pre-commit hook](#formatting-using-pre-commit-hooks)).

### [CMake Style Guide](cmake_style_guide.md)

CMake build system standards and best practices, including:

- **CMake dependencies:** How to add dependencies between subprojects
- **Build system patterns:** Common CMake patterns and conventions

> [!TIP]
> The "Mastering CMake" book at
> https://cmake.org/cmake/help/book/mastering-cmake/index.html is a good
> resource.

### [Bash Style Guide](bash_style_guide.md)

Bash scripting guidelines (use sparingly - prefer Python):

- **Safe bash modes:** Using `set -euo pipefail`
- **When to use Bash:** Appropriate use cases
- **Best practices:** Following Google's shell style guide

> [!WARNING]
> Bash is **strongly discouraged** for nontrivial usage. **Use Python scripts in
> most cases instead**.

### [GitHub Actions Style Guide](github_actions_style_guide.md)

GitHub Actions workflow standards including:

- **Action pinning:** Pinning to commit SHAs for security
- **Runner versioning:** Using specific runner versions
- **Python over Bash:** Preferring Python scripts in workflows
- **Safe defaults:** Using safe default values for inputs
- **Build/test separation:** Optimizing runner usage
