# Python Style Guide

This guide documents Python coding standards and best practices for TheRock and related ROCm build infrastructure projects.

We generally follow the [PEP 8 style guide](https://peps.python.org/pep-0008/)
using the [_Black_ formatter](https://github.com/psf/black) (run automatically
as a [pre-commit hook](README.md#formatting-using-pre-commit-hooks)).

The guidelines here extend PEP 8 for our projects.

## Use `pathlib` for filesystem paths

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

✅ **Preferred:**

```python
from pathlib import Path

# Clear, readable, platform-independent
artifact_path = Path(output_dir) / artifact_group / "rocm.tar.gz"

# Concise and type-safe
artifacts_dir = Path(base_dir) / "build" / "artifacts"
if artifacts_dir.exists():
    files = list(artifacts_dir.iterdir())
```

❌ **Avoid:**

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

## Don't make assumptions about the current working directory

Scripts should be runnable from the repository root, their script subdirectory,
and other locations. They should not assume any particular current working
directory.

Benefits:

- **Location-independent:** Script works from any directory
- **Explicit:** Clear where files are relative to the script
- **CI-friendly:** Works in CI environments with varying working directories,
  especially when scripts and workflows are used in other repositories

✅ **Preferred:**

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

❌ **Avoid:**

```python
from pathlib import Path

# Assumes script is run from repository root
config_file = Path("build_tools/config.json")

# Assumes script is run from its own directory
data_file = Path("../data/artifacts.tar.gz")
```

## Use `argparse` for CLI flags

Use [`argparse`](https://docs.python.org/3/library/argparse.html) for
command-line argument parsing with clear help text and type conversion.

Benefits:

- **Automatic help:** Users get `-h/--help` for free
- **Type conversion:** Arguments are converted to correct types
- **Validation:** Required arguments are enforced

✅ **Preferred:**

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

❌ **Avoid:**

```python
import sys

# Fragile, no help text, no type checking
if len(sys.argv) < 3:
    print("Usage: script.py <run-id> <output-dir>")
    sys.exit(1)

run_id = sys.argv[1]  # String, not validated
output_dir = sys.argv[2]
```

## Add type hints liberally

Add type hints (see [`typing`](https://docs.python.org/3/library/typing.html))
to function signatures to improve code clarity and enable static analysis.

Benefits:

- **Self-documenting:** Function signatures clearly show expected types
- **Editor support:** IDEs provide better autocomplete and error detection
- **Static analysis:** Tools like `mypy` can catch type errors before runtime
- **Refactoring safety:** Easier to refactor with confidence

✅ **Preferred:**

```python
from pathlib import Path


def fetch_artifacts(
    run_id: int,
    output_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
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

❌ **Avoid:**

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

## Use `__main__` guard

Use [`__main__`](https://docs.python.org/3/library/__main__.html) to limit what
code runs when a file is imported. Typically, Python files should define
functions in the top level scope and only call those functions themselves if
executed as the top-level code environment (`if __name__ == "__main__"`).

Benefits:

- **Importable:** Other scripts can import and reuse functions
- **Testable:** Unit tests can call functions with controlled arguments
- **Composable:** Functions can be imported for use in other scripts

✅ **Preferred:**

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

❌ **Avoid:**

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

## Use named arguments for complicated function signatures

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

✅ **Preferred:**

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

❌ **Avoid:**

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
