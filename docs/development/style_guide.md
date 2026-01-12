# TheRock Style Guide

> [!IMPORTANT]
> This is a living document meant to steer developers towards agreed upon best
> practices.
>
> 📝 Feel free to propose new sections and amend (or remove) existing sections.

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
using the [_Black_ formatter](https://github.com/psf/black) (run automatically
as a [pre-commit hook](#formatting-using-pre-commit-hooks)). The guidelines here
extend PEP 8 for our projects.

### Use `pathlib` for filesystem paths

Use [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html) for
path and filesystem operations. Avoid string manipulation and
[`os.path`](https://docs.python.org/3/library/os.path.html).

Benefits:

- **Platform-independent:** Handles Windows vs Unix path separators, symlinks,
  and other features automatically
- **Readable:** Operators like `/` and `.suffix` are easier to understand
- **Type-safe:** Dedicated types help catch errors at development time
- **Feature-rich:** Built-in methods like `.exists()`, `.mkdir()`, `.glob()`

> [!TIP]
> See the official
> ["Corresponding tools" documentation](https://docs.python.org/3/library/pathlib.html#corresponding-tools)
> for a table mapping from various `os` functions to `Path` equivalents.

✅ **Preferred:** Using `pathlib.Path`

```python
from pathlib import Path

# Clear, readable, platform-independent
artifact_path = Path(output_dir) / artifact_group / "rocm.tar.gz"

# Concise and type-safe
artifacts_dir = Path(base_dir) / "build" / "artifacts"
if artifacts_dir.exists():
    files = list(artifacts_dir.iterdir())
```

❌ **Avoid:** String-based path manipulation

```python
import os

# Hard to read, platform-specific separators (Windows uses `\`)
artifact_path = output_dir + "/" + artifact_group + "/" + "rocm.tar.gz"

# Portable but verbose and may repeat separators if arguments include them already
artifact_path = output_dir + os.path.sep + artifact_group + os.path.sep + "rocm.tar.gz"

# Verbose and error-prone
if os.path.exists(os.path.join(base_dir, "build", "artifacts")):
    files = os.listdir(os.path.join(base_dir, "build", "artifacts"))
```

### Don't make assumptions about the current working directory

Scripts should be runnable from the repository root, their script subdirectory,
and other locations. They should not assume any particular current working
directory.

Benefits:

- **Location-independent:** Script works from any directory
- **Explicit:** Clear where files are relative to the script
- **CI-friendly:** Works in CI environments with varying working directories,
  especially when scripts and workflows are used in other repositories

✅ **Preferred:** Paths relative to the script location

```python
from pathlib import Path

# Establish script's location as reference point
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent

# Build paths relative to script location
config_file = THIS_SCRIPT_DIR / "config.json"
# Build paths relative to repository root
version_file = THEROCK_DIR / "version.json"
```

❌ **Avoid:** Assuming script runs from a specific directory

```python
from pathlib import Path

# Assumes script is run from repository root
config_file = Path("build_tools/config.json")

# Assumes script is run from its own directory
data_file = Path("../data/artifacts.tar.gz")
```

### Use `argparse` for CLI flags

Use [`argparse`](https://docs.python.org/3/library/argparse.html) for
command-line argument parsing with clear help text and type conversion.

Benefits:

- **Automatic help:** Users get `-h/--help` for free
- **Type conversion:** Arguments are converted to correct types
- **Validation:** Required arguments are enforced

✅ **Preferred:** Using `argparse` with types and defaults

```python
import argparse
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="Fetches artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/artifacts"),
        help="Output path for fetched artifacts (default: build/artifacts)",
    )
    parser.add_argument(
        "--include-tests",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include test artifacts",
    )
    parser.add_argument(
        "--test-filter",
        type=str,
        help="Regular expression filter to apply when fetching test artifacts",
    )

    args = parser.parse_args(argv)
    if args.test_filter and not args.include_tests:
        parser.error("Cannot set --test-filter if --include-tests is not enabled")

    # ... then call functions using the parsed arguments


if __name__ == "__main__":
    main(sys.argv[1:])
```

❌ **Avoid:** Manual argument parsing

```python
import sys

# Fragile, no help text, no type checking
if len(sys.argv) < 3:
    print("Usage: script.py <run-id> <output-dir>")
    sys.exit(1)

run_id = sys.argv[1]  # String, not validated
output_dir = sys.argv[2]
```

### Add type hints liberally

Add type hints (see [`typing`](https://docs.python.org/3/library/typing.html))
to function signatures to improve code clarity and enable static analysis.

Benefits:

- **Self-documenting:** Function signatures clearly show expected types
- **Editor support:** IDEs provide better autocomplete and error detection
- **Static analysis:** Tools like `mypy` can catch type errors before runtime
- **Refactoring safety:** Easier to refactor with confidence

✅ **Preferred:** Clear type hints

```python
from pathlib import Path
from typing import Optional


def fetch_artifacts(
    run_id: int,
    output_dir: Path,
    include_patterns: list[str],
    exclude_patterns: Optional[list[str]] = None,
) -> list[Path]:
    """Fetch artifacts matching the given patterns.

    Args:
        run_id: GitHub Actions run ID
        output_dir: Directory to save artifacts
        include_patterns: Regex patterns to include
        exclude_patterns: Regex patterns to exclude

    Returns:
        List of paths to downloaded artifacts
    """
    if exclude_patterns is None:
        exclude_patterns = []

    artifacts: list[Path] = []
    for pattern in include_patterns:
        # ... fetch logic
        artifacts.append(result)
    return artifacts
```

❌ **Avoid:** No type hints

```python
def fetch_artifacts(run_id, output_dir, include_patterns, exclude_patterns=None):
    # What types are these? What does this return?
    if exclude_patterns is None:
        exclude_patterns = []

    artifacts = []
    for pattern in include_patterns:
        # ... fetch logic
        artifacts.append(result)
    return artifacts
```

### Use `__main__` guard

Use [`__main__`](https://docs.python.org/3/library/__main__.html) to limit what
code runs when a file is imported. Typically, Python files should define
functions in the top level scope and only call those functions themselves if
executed as the top-level code environment (`if __name__ == "__main__"`).

Benefits:

- **Importable:** Other scripts can import and reuse functions
- **Testable:** Unit tests can call functions with controlled arguments
- **Composable:** Functions can be imported for use in other scripts

✅ **Preferred:** Separate definition from execution

```python
import sys
import argparse
from pathlib import Path


# This function can be used from other scripts by importing this file,
# without side effects like running the argparse code below.
def fetch_artifacts(run_id: int, output_dir: Path) -> list[Path]:
    """Fetch artifacts from the given run ID."""
    # ... implementation here
    return artifacts


# This function can called from unit tests (or other scripts).
def main(argv: list[str]) -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fetch artifacts from GitHub Actions")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))

    args = parser.parse_args(argv)

    artifacts = fetch_artifacts(args.run_id, args.output_dir)
    print(f"Downloaded {len(artifacts)} artifacts")
    return 0


if __name__ == "__main__":
    # This code runs only if the script is executed directly.
    sys.exit(main(sys.argv[1:]))
```

❌ **Avoid:** Code runs on import

```python
import sys
import argparse

# This runs immediately when imported, making testing difficult
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=int, required=True)
args = parser.parse_args()

# Global side effects on import
print(f"Fetching artifacts for run {args.run_id}")
result = fetch_artifacts(args.run_id)
print(f"Downloaded {len(result)} artifacts")
```

### Use named arguments for complicated function signatures

Using positional arguments for functions that accept many arguments is error
prone. Use keyword arguments to make function calls explicit and
self-documenting.

Benefits:

- **Readability:** Clear what each argument represents at the call site
- **Safety:** Prevents accidentally swapping arguments of the same type
- **Maintainability:** Function signature can evolve without breaking calls

> [!TIP]
> Consider using named arguments when:
>
> - Function has more than 2-3 parameters
> - Multiple parameters have the same type (especially booleans)
> - The meaning of arguments isn't obvious from context

✅ **Preferred:** Named arguments are clear and safe

```python
# Intent is immediately clear
result = build_artifacts(
    amdgpu_family="gfx942",
    enable_testing=True,
    use_ccache=False,
    build_dir="/tmp/build",
    components=["rocblas", "hipblas"],
)

# Flags are self-documenting
process_files(
    input_dir=input_dir,
    output_dir=output_dir,
    overwrite=True,
    validate=False,
    compress=True,
)
```

❌ **Avoid:** Positional arguments are error prone

```python
# What do these values mean? Easy to mix up the order
result = build_artifacts(
    "gfx942",
    True,
    False,
    "/tmp/build",
    ["rocblas", "hipblas"],
)

# Even worse: easy to swap boolean flags
process_files(input_dir, output_dir, True, False, True)
```

## Bash guidelines

> [!WARNING]
> Bash is **strongly discouraged** for nontrivial usage in .yml GitHub Actions
> workflow files and script files.
>
> **Use Python scripts in most cases instead**.

Writing and maintaining safe and portable scripts in Bash is significantly
harder than it is in Python. When appropriate, we write Bash scripts following
some of the guidelines at https://google.github.io/styleguide/shellguide.html.

Those sections are particularly noteworthy:

- https://google.github.io/styleguide/shellguide.html#variable-expansion
- https://google.github.io/styleguide/shellguide.html#quoting

### Setting bash modes

Scripts should generally set modes like

```bash
set -euo pipefail
```

- `set -e` Exits if any command has a non-zero exit status
- `set -u` Treats undefined variables as errors
- `set -o pipefail` Uses the return code of a failing command as the pipeline
  return code
- `set -x` Prints commands to the terminal (useful for debugging and CI logging)

See https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425 for
an explanation of what each of these options does.

## GitHub Actions guidelines

### Pin action `uses:` versions to commit SHAs

Pin actions in
[`jobs.<job_id>.steps[*].uses`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsuses)
to specific commit SHAs for security and reproducibility.

✅ **Preferred:** Pin to specific commit SHA with the semantic version tag in a comment

```yaml
- uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
- uses: docker/setup-buildx-action@c47758b77c9736f4b2ef4073d4d51994fabfe349  # v3.7.1
```

> [!TIP]
> We use
> [Dependabot](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/keeping-your-actions-up-to-date-with-dependabot)
> to automatically update pinned actions while maintaining security.
>
> Dependabot matches our "commit hash with the tag in a comment" style.

❌ **Avoid:** Using unpinned or branch references

```yaml
- uses: actions/checkout@main  # Branches are regularly updated
- uses: actions/setup-python@v6.0.0  # Tags can be moved (even for releases)
```

### Pin action `runs-on:` labels to specific versions

Pin GitHub-hosted runner labels in
[`jobs.<job_id>.runs-on`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idruns-on)
to specific versions from the
[available images list](https://github.com/actions/runner-images?tab=readme-ov-file#available-images)
for security and reproducibility.

✅ **Preferred:** Use specific versions

```yaml
jobs:
  build:
    runs-on: ubuntu-24.04  # We can change this across our projects when we want
```

❌ **Avoid:** Using unpinned (e.g. latest) versions

```yaml
jobs:
  build:
    runs-on: ubuntu-latest  # This could change outside of our control
```

### Prefer Python scripts over inline Bash

Where possible, put workflow logic in Python scripts.

Python scripts have several benefits:

- **Testable:** Can be tested locally and with unit tests
- **Debuggable:** Easier to debug with standard Python tools
- **Portable:** Works consistently across platforms (Linux/Windows)
- **Maintainable:** Better error handling and logging support
- **Modular:** Functions can be shared across multiple scripts

> [!TIP]
> Use your judgement for what logic is trivial enough to stay in bash.
>
> Some signs of complicated bash are _conditionals_, _loops_, _regex_,
> _piping command output_, and _string manipulation_.

✅ **Preferred:** Dedicated Python script

```yaml
- name: Process artifacts
  run: |
    python build_tools/process_artifacts.py \
      --families "${{ inputs.amdgpu_families }}" \
      --artifact-dir artifacts \
      --install-dir install
```

❌ **Avoid:** Complex inline bash logic

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

### Use safe defaults for inputs

Workflow inputs must have safe default values that work in common scenarios.

Good defaults should:

- **Work without configuration:** Safe defaults that won't trigger production changes
- **Be well-documented:** Clear descriptions explaining when to override
- **Fail safely:** Default to dev/test behavior and not affecting production resources

> [!NOTE]
> Some workflows may be configured to have stricter security boundaries, such
> as only accepting "nightly" release types from certain branches or from
> certain repositories.

✅ **Preferred:** Safe defaults that require explicit intent

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        description: Type of release to create. All developer-triggered jobs should use "dev"!
        options:
          - dev
          - nightly
          - prerelease
        default: dev  # Safe: development releases don't affect production

      amdgpu_families:
        type: string
        description: "GPU families to build (comma-separated). Leave empty for default set."
        default: ""  # Empty string handled gracefully in workflow logic
```

❌ **Avoid:** Unsafe defaults that could trigger unintended releases

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

### Separate build and test stages

Use CPU runners to build from source and pass artifacts to test runners.

Benefits of separation:

- **Cost optimization:** GPU runners are expensive; use them only when needed
- **Parallelization:** Multiple test jobs can share build artifacts
- **Packaging enforcement:** Testing in this way enforces that build artifacts
  are installable and usable on other machines

✅ **Preferred:** Separate build on CPU runners and test on GPU runners

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

❌ **Avoid:** Building and testing on expensive GPU runners

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
