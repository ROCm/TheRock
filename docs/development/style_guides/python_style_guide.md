# Python style guide

Table of Contents

- [General recommendations](#general-recommendations)
- [Style guidelines](#style-guidelines)
  - [Code quality and readability](#code-quality-and-readability)
    - [Add specific type hints liberally](#add-specific-type-hints-liberally)
    - [Extract complex type signatures](#extract-complex-type-signatures)
    - [Use dataclasses, not tuples](#use-dataclasses-not-tuples)
    - [Use named arguments for complicated function signatures](#use-named-arguments-for-complicated-function-signatures)
    - [No magic numbers](#no-magic-numbers)
  - [Script structure and organization](#script-structure-and-organization)
    - [Use `__main__` guard](#use-__main__-guard)
    - [Use `argparse` for CLI flags](#use-argparse-for-cli-flags)
    - [Import organization](#import-organization)
    - [Code organization](#code-organization)
    - [No duplicate code](#no-duplicate-code)
  - [Error handling and reliability](#error-handling-and-reliability)
    - [Distinguish between different error conditions](#distinguish-between-different-error-conditions)
    - [Validate that operations actually succeeded](#validate-that-operations-actually-succeeded)
    - [Fail-fast behavior](#fail-fast-behavior)
    - [No timeouts on basic binutils](#no-timeouts-on-basic-binutils)
  - [Filesystem and path operations](#filesystem-and-path-operations)
    - [Use `pathlib` for filesystem paths](#use-pathlib-for-filesystem-paths)
    - [Don't make assumptions about the current working directory](#dont-make-assumptions-about-the-current-working-directory)
    - [No hard-coded project paths](#no-hard-coded-project-paths)
  - [Performance best practices](#performance-best-practices)
  - [Testing standards](#testing-standards)
- [Reference material](#reference-material)
  - [Code review checklist](#code-review-checklist)
  - [Common patterns](#common-patterns)

## Core principles

We generally follow the [PEP 8 style guide](https://peps.python.org/pep-0008/)
using the [_Black_ formatter](https://github.com/psf/black) (run automatically
as a [pre-commit hook](README.md#formatting-using-pre-commit-hooks)).

The guidelines here extend PEP 8 for our projects.

- This guide reflects lessons learned from multiple production incidents
- When in doubt, fail fast and loud
- False positives (spurious errors) are better than false negatives (silent corruption)

## Style guidelines

### Code quality and readability

#### Add specific type hints liberally

Add type hints (see [`typing`](https://docs.python.org/3/library/typing.html))
to function signatures to improve code clarity and enable static analysis.

- Use modern type hint syntax (Python 3.10+). We're on Python 3.13+.
- Use specific type hints. Never use `Any` except in rare generic code.

Benefits:

- **Self-documenting:** Function signatures clearly show expected types
- **Editor support:** IDEs provide better autocomplete and error detection
- **Static analysis:** Tools like `mypy` can catch type errors before runtime
- **Refactoring safety:** Easier to refactor with confidence

Type hint best practices:

- Use built-in generics: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]`
- Use `T | None` instead of `Optional[T]`
- Use `X | Y` instead of `Union[X, Y]`
- Import the actual types you need (not from `typing` for basic containers)
- Use specific return types (not `tuple`, use `tuple[Path, int]`)
- For dict values with structure, define a dataclass

✅ **Preferred:**

```python
def fetch_artifacts(
    run_id: int,
    output_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    pass


def process(handlers: list[DatabaseHandler]) -> dict[str, KpackInfo]:
    pass
```

❌ **Avoid:**

```python
# What types are these? What does this return?
def fetch_artifacts(run_id, output_dir, include_patterns, exclude_patterns=None):
    pass


# Avoid overly generic type units like 'Any'
def process(handlers: List[Any]) -> Dict[str, Any]:
    pass
```

#### Extract complex type signatures

**If a type signature is complex or repeated, extract it into a named type.**

When to extract:

- Type appears in multiple signatures → Use NamedTuple or TypeAlias
- Type signature is hard to read at a glance → Extract it
- Dict used to pass data within a file has 3+ fields → Use NamedTuple or dataclass
- Tuple has 3+ fields → Use NamedTuple or dataclass
- You find yourself documenting what tuple fields mean → Use NamedTuple

What to use:

- **NamedTuple**: Immutable, lightweight, for simple data containers
- **dataclass**: When you need methods, mutability, or inheritance
- **TypeAlias**: For complex generic types that are reused

✅ **Preferred:**

```python
class KernelInput(NamedTuple):
    """Input data for preparing a kernel for packing.

    Attributes:
        relative_path: Path relative to archive root (e.g., "kernels/my_kernel")
        gfx_arch: GPU architecture (e.g., "gfx1100")
        hsaco_data: Raw HSACO binary data
        metadata: Optional metadata dict to store in TOC
    """

    relative_path: str
    gfx_arch: str
    hsaco_data: bytes
    metadata: dict[str, object] | None


def parallel_prepare_kernels(
    archive: PackedKernelArchive,
    kernels: list[KernelInput],  # Self-documenting!
    executor: Executor | None = None,
) -> list[PreparedKernel]:
    """Prepare multiple kernels in parallel..."""
    for k in kernels:
        # k.relative_path, k.gfx_arch, etc. - clear and IDE-friendly
        ...
```

❌ **Avoid:**

```python
def parallel_prepare_kernels(
    archive: PackedKernelArchive,
    kernels: list[tuple[str, str, bytes, dict[str, object] | None]],
    executor: Executor | None = None,
) -> list[PreparedKernel]:
    """What is this tuple again? Have to read the docstring..."""
    for relative_path, gfx_arch, hsaco_data, metadata in kernels:
        ...
```

#### Use dataclasses, not tuples

**For non-trivial data with multiple fields, use dataclasses instead of tuples.**

Benefits:

- Self-documenting: Field names make code clearer
- IDE-friendly: Autocomplete and type checking work
- Refactoring-safe: Adding fields doesn't break positional unpacking
- Less error-prone: Can't accidentally swap fields of the same type

✅ **Preferred:**

```python
@dataclass
class KpackInfo:
    """Information about a created kpack file."""
    kpack_path: Path
    size: int
    kernel_count: int

def create_kpack_files(...) -> dict[str, KpackInfo]:
    """Returns: Dict mapping arch to KpackInfo"""
    return {"gfx1100": KpackInfo(kpack_path=path, size=12345, kernel_count=42)}
```

❌ **Avoid:**

```python
def create_kpack_files(...) -> dict[str, tuple[Path, int, int]]:
    """Returns: Dict mapping arch to (kpack_path, size, kernel_count)"""
    return {"gfx1100": (path, 12345, 42)}  # What's what?
```

When tuples are OK:

- Simple pairs where meaning is obvious: `(x, y)`, `(min, max)`
- Unpacking from standard library functions: `os.path.split()`
- Single-use internal return values that are immediately unpacked

#### Use named arguments for complicated function signatures

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

#### No magic numbers

**Don't use unexplained magic numbers, especially for estimates.**

Benefits:

- Code is self-documenting
- No false precision from made-up values
- Prevents misleading information

✅ **Preferred:**

```python
# Either track the real size or don't log it
new_size = binary_path.stat().st_size
print(f"Device code stripped, new size: {new_size} bytes")
```

❌ **Avoid:**

```python
original_size = binary_path.stat().st_size + 8000000  # Estimate original size
print(f"Stripped {original_size - new_size} bytes")
```

### Script structure and organization

#### Use `__main__` guard

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


# This function can be used from other scripts by importing this file,
# without side effects like running the argparse code below.
def count_artifacts(run_id: int) -> int:
    # ... implementation here
    return count


# This function can called from unit tests (or other scripts).
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch artifacts from GitHub Actions")
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)

    count = count_artifacts(args.run_id)
    print(f"Counted {count} artifacts")
    return 0


if __name__ == "__main__":
    # This code runs only if the script is executed directly.
    sys.exit(main(sys.argv[1:]))
```

❌ **Avoid:**

```python
import sys
import argparse


def count_artifacts(run_id: int) -> int:
    # ... implementation here
    return count


# This runs immediately when imported, making testing difficult
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=int, required=True)
args = parser.parse_args()

# Global side effects on import
count = count_artifacts(args.run_id)
print(f"Counted {count} artifacts")
```

#### Use `argparse` for CLI flags

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

#### Import organization

**Put all imports at the top of the file. Avoid inline imports except for rare special cases.**

**Do NOT use `from __future__ import annotations`.** It will be many years
before we can rely on this as a default and we'd rather write code in a
compatible by default way.

Benefits:

- Clear view of all dependencies at the top
- Easier to spot circular dependencies
- Standard Python convention
- Better for static analysis tools

✅ **Preferred:**

```python
import shutil
from pathlib import Path


def process_binary(input_path: Path, output_path: Path) -> None:
    """Process a binary file."""
    # ... some code ...

    if needs_special_processing:
        shutil.copy2(input_path, temp_file)
```

❌ **Avoid:**

```python
from __future__ import annotations


def process_binary(input_path: Path, output_path: Path) -> None:
    """Process a binary file."""
    # ... some code ...

    if needs_special_processing:
        import shutil  # Inline import

        shutil.copy2(input_path, temp_file)
```

When inline imports ARE acceptable:

- **Circular dependency workaround**: If module A imports module B and B imports A, one can use an inline import
- **Optional heavy dependency**: Importing a very heavy module that's rarely used (but document why)

Example of acceptable inline import for circular dependency:

```python
def create_host_only(self, output_path: Path) -> None:
    """Create host-only binary."""
    # Import here to avoid circular dependency:
    # binutils.py → elf_offload_kpacker.py → binutils.py
    from rocm_kpack.elf_offload_kpacker import kpack_offload_binary

    kpack_offload_kpacker(self.file_path, output_path, toolchain=self.toolchain)
```

Key points:

- Inline imports should be the exception, not the rule
- Always add a comment explaining WHY the import is inline
- Consider refactoring to eliminate circular dependencies instead

#### Code organization

**Keep functions focused and modules cohesive.**

Benefits:

- Easier to understand and test
- Promotes reusability
- Reduces cognitive load
- Makes code reviews more effective

Guidelines:

- Classes should be < 200 lines (ideally)
- Methods should be < 30 lines (ideally)
- If a class has 7+ responsibilities, split it

When to split:

- God objects doing everything → multiple focused classes
- 100+ line methods → extract helper methods
- Duplicate code → extract to shared function

#### No duplicate code

**Extract common code to shared functions.**

Benefits:

- Single source of truth
- Bug fixes apply everywhere
- Easier to maintain and test
- Reduces codebase size

✅ **Preferred:**

```python
def compute_manifest_relative_path(self, binary_path: Path, prefix_root: Path) -> str:
    """Compute the relative path from a binary to its kpack manifest."""
    rel_path = binary_path.relative_to(prefix_root)
    depth = len(rel_path.parts) - 1
    if depth == 0:
        return f".kpack/{self.component_name}.kpm"
    else:
        up_path = "/".join([".."] * depth)
        return f"{up_path}/.kpack/{self.component_name}.kpm"


# Use in both places
manifest_relpath = self.compute_manifest_relative_path(binary_path, prefix_dir)
```

❌ **Avoid:**

```python
# In method 1:
depth = len(binary_relpath.parts) - 1
if depth == 0:
    manifest_relpath = f".kpack/{self.component_name}.kpm"
else:
    up_path = "../" * depth
    manifest_relpath = f"{up_path}.kpack/{self.component_name}.kpm"

# In method 2:
# Same code repeated
```

### Error handling and reliability

#### Distinguish between different error conditions

**Don't treat all errors the same.**

Benefits:

- Debugging is easier when errors are specific
- Callers can handle different errors appropriately
- Preserves exception chain with `from e`
- Avoids hiding bugs behind generic exception handlers

✅ **Preferred:**

```python
# Fast check: Is this even an ELF file?
try:
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"\x7fELF":
            return False  # Not ELF, definitely not fat
except FileNotFoundError:
    raise  # Propagate - caller should know file is missing
except OSError as e:
    raise RuntimeError(f"Cannot read file {file_path}: {e}") from e

# Now check for .hip_fatbin section
try:
    output = subprocess.check_output([readelf, "-S", str(file_path)])
    return ".hip_fatbin" in output
except subprocess.CalledProcessError as e:
    if e.returncode == 1:
        return False  # readelf returns 1 for valid ELF without target section
    raise RuntimeError(f"readelf failed on {file_path}: {e.output}") from e
except FileNotFoundError as e:
    raise RuntimeError(f"readelf not found: {readelf}") from e
```

❌ **Avoid:**

```python
try:
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != b"\x7fELF":
            return False
    output = subprocess.check_output([readelf, "-S", str(file_path)])
    return ".hip_fatbin" in output
except Exception:
    return False  # File not found? Not ELF? readelf crashed? Who knows!
```

Key points:

- Catch specific exceptions, not broad `Exception`
- Re-raise exceptions that indicate bugs or missing tools
- Return False only for legitimate "not found" cases
- Use `from e` to preserve exception chain

#### Validate that operations actually succeeded

**Don't just assume that an operation succeeded, check that it did.**

Benefits:

- Catches failures early before they propagate
- Makes debugging easier with clear error messages
- Prevents downstream consumers from receiving bad data
- Documents expected invariants in the code

✅ **Preferred:**

```python
archive.write(kpack_file)

# Validate kpack file was created successfully
if not kpack_file.exists():
    raise RuntimeError(f"Failed to create kpack file: {kpack_file}")

kpack_size = kpack_file.stat().st_size
if kpack_size == 0:
    raise RuntimeError(f"Kpack file is empty: {kpack_file}")
```

❌ **Avoid:**

```python
archive.write(kpack_file)
# Assume it worked
kpack_size = kpack_file.stat().st_size
```

What to validate:

- Files exist after creation
- Files are non-empty when they should have content
- Processed files are smaller after stripping
- Critical operations completed successfully

#### Fail-fast behavior

**Always fail immediately on errors. Never silently continue or produce incomplete results.**

Benefits:

- Catches problems early before they cascade
- Makes debugging easier by failing at the source of the problem
- Prevents incomplete or corrupted artifacts
- Makes build failures explicit and actionable

✅ **Preferred:**

```python
if not path.exists():
    raise FileNotFoundError(
        f"Path does not exist: {path}\n"
        f"This indicates a corrupted or incomplete artifact"
    )

# Let exceptions propagate - if we can't process, we must fail
process_file(path)
```

❌ **Avoid:**

```python
if not path.exists():
    print(f"Warning: Path does not exist: {path}")
    continue  # Silently produces incomplete output

try:
    process_file(path)
except Exception as e:
    print(f"Warning: {e}")
    # Continues with incomplete data
```

Key points:

- If data is missing, corrupted, or unreadable → raise an exception
- Don't catch exceptions unless you can meaningfully handle them
- "Warnings" that indicate data problems should be errors
- Incomplete artifacts are worse than failed builds

#### No timeouts on basic binutils

**NEVER add timeouts to basic binutils operations (readelf, objcopy, etc.).**

Benefits:

- Prevents spurious failures on loaded systems
- If a tool hangs, that's a bug to fix, not mask with timeouts
- Build systems handle global timeouts more appropriately
- Simpler code without arbitrary timeout values

✅ **Preferred:**

```python
subprocess.check_output([readelf, "-S", file])
```

❌ **Avoid:**

```python
subprocess.check_output([readelf, "-S", file], timeout=10)
```

### Filesystem and path operations

#### Use `pathlib` for filesystem paths

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

#### Don't make assumptions about the current working directory

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

#### No hard-coded project paths

**Never hard-code project-specific paths. Code should be portable.**

Benefits:

- Works across different development environments
- CI/CD friendly
- Easier for new contributors
- No accidental dependencies on specific machine setups

✅ **Preferred:**

```python
# Use system defaults or user-configurable paths
with tempfile.TemporaryDirectory() as tmpdir:
    process(tmpdir)

# Use environment variables or relative paths
CONFIG_PATH = Path(os.environ.get("ROCM_CONFIG", "config.json"))

# Or derive from module location
CONFIG_PATH = Path(__file__).parent / "config.json"
```

❌ **Avoid:**

```python
# Hard-coded developer-specific paths
with tempfile.TemporaryDirectory(dir="/develop/tmp") as tmpdir:
    process(tmpdir)

CONFIG_PATH = Path("/home/stella/rocm-workspace/config.json")
```

Key points:

- Use `tempfile.TemporaryDirectory()` without `dir=` argument (uses system default)
- Use environment variables for configurable paths
- Use relative paths or derive from `__file__` when appropriate
- If a specific temp location is needed, make it configurable via environment variable

### Performance best practices

**Optimize hot paths, but keep code readable.**

Benefits:

- Faster builds and tests
- Reduced resource consumption
- Better user experience
- Still maintainable code

✅ **Preferred:**

```python
# Compile once at module level
_GFX_ARCH_PATTERN = re.compile(r"gfx(\d+[a-z]*)")


def detect(self, path: Path) -> str | None:
    match = _GFX_ARCH_PATTERN.search(path.name)
```

❌ **Avoid:**

```python
# Compiles regex on every call
def detect(self, path: Path) -> str | None:
    match = re.search(r"gfx(\d+[a-z]*)", path.name)
```

Other optimizations:

- Check cheap conditions before expensive ones (e.g., magic bytes before subprocess)
- Cache expensive computations when called repeatedly
- Use generators for large datasets

### Testing standards

**Tests should verify fail-fast behavior:**

```python
def test_fails_on_missing_file(self, tmp_path):
    """Test that processing fails fast on missing files."""
    splitter = ArtifactSplitter(...)

    non_existent = tmp_path / "non_existent"

    # Should raise, not continue with incomplete data
    with pytest.raises(FileNotFoundError, match="does not exist"):
        splitter.split(non_existent, output_dir)
```

**Use real files in tests when possible:**

- Prefer real temporary files over mocks for filesystem operations
- Mock only external dependencies (network, expensive tools)
- Integration tests should exercise the full path

## Reference material

### Code review checklist

Before submitting code, verify:

- [ ] No silent error handling (fail-fast on all errors)
- [ ] No `Any` type hints (use specific types)
- [ ] Modern type syntax (`list[T]`, `T | None`, not `List[T]`, `Optional[T]`)
- [ ] No `from __future__ import annotations`
- [ ] Complex type signatures extracted to NamedTuple/dataclass
- [ ] No magic numbers or fake estimates
- [ ] Tuples only for simple pairs, dataclasses for structured data
- [ ] All imports at top of file (except documented circular dependencies)
- [ ] No timeouts on binutils operations
- [ ] No hard-coded project paths (use system defaults or env vars)
- [ ] Output validation after critical operations
- [ ] No duplicate code
- [ ] Specific exception handling (not broad `except Exception`)
- [ ] Methods < 30 lines (or have a good reason)
- [ ] Classes < 200 lines (or split into focused components)
- [ ] Using `pathlib.Path` for filesystem operations
- [ ] Scripts work from any directory (no CWD assumptions)
- [ ] CLI scripts use `argparse`
- [ ] All functions have type hints
- [ ] Scripts have `__main__` guard
- [ ] Complex function calls use named arguments

### Common patterns

#### Pattern: reading and validating a file

```python
def read_config(path: Path) -> ConfigData:
    """Read and validate configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Cannot read {path}: {e}") from e

    # Validate required fields
    if "version" not in data:
        raise ValueError(f"Missing 'version' field in {path}")

    return ConfigData(**data)
```

#### Pattern: running subprocess with proper error handling

```python
def run_binutil(tool: Path, args: list[str], input_file: Path) -> str:
    """Run a binutil tool with proper error handling."""
    try:
        result = subprocess.check_output(
            [str(tool)] + args + [str(input_file)], stderr=subprocess.STDOUT, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        # Distinguish between different exit codes
        if e.returncode == 1:
            # Tool-specific handling for returncode 1
            return ""
        raise RuntimeError(
            f"{tool.name} failed on {input_file} with code {e.returncode}: {e.output}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(f"Tool not found: {tool}") from e
```
