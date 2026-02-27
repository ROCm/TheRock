# TheRock style guides

> [!IMPORTANT]
> These style guides are living documents meant to steer developers towards
> agreed upon best practices.
>
> 📝 Feel free to propose edits 📝

## Language and tool-specific guides

- [Bash Style Guide](bash_style_guide.md)
- [CMake Style Guide](cmake_style_guide.md)
- [GitHub Actions Style Guide](github_actions_style_guide.md)
- [Python Style Guide](python_style_guide.md)

## Core principles

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
  - Devices (dcgpu, dgpu, igpu, apu, etc.)
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
# Install git hook. Now pre-commit runs on every git commit.
pre-commit install
```
