# Coding Conventions

*Patterns observed in TheRock codebase*

## File Organization

### Directory Structure Pattern
- **By category**: Components organized into `base/`, `compiler/`, `core/`, `math-libs/`, `ml-libs/`, etc.
- **Submodules**: Each major component is a git submodule (30+ submodules)
- **Build artifacts separated**: Source in component dir, build outputs in `build/component/`
- **Documentation co-located**: `docs/` for project-wide, READMEs in component dirs

### Build Configuration Layout
```
component/
├── CMakeLists.txt          # Build orchestration (required)
├── artifact-name.toml      # Packaging metadata (optional)
├── pre_hook_*.cmake        # Pre-configuration hooks (optional)
├── post_hook_*.cmake       # Post-build hooks (optional)
├── README.md               # Component documentation
└── submodule/              # Git submodule (actual source)
```

### Python Module Structure
```
build_tools/
├── script_name.py          # Executable scripts (shebang, main)
├── _therock_utils/         # Internal libraries (no shebang)
│   └── module.py
└── github_actions/         # CI-specific scripts
    └── action_script.py
```

## Naming Conventions

### Files

**CMake:**
- **Module files**: `snake_case.cmake` (e.g., `therock_subproject.cmake`)
- **Component files**: `CMakeLists.txt` (standard)
- **Hook files**: `{pre|post}_hook_component-name.cmake` (hyphens in component name)

**Python:**
- **Scripts**: `snake_case.py` (e.g., `fetch_sources.py`)
- **Modules**: `snake_case.py` (e.g., `build_topology.py`)
- **Packages**: `_internal_prefix/` for utilities

**Configuration:**
- **TOML files**: `SCREAMING_SNAKE_CASE.toml` or `artifact-name.toml`
  - `BUILD_TOPOLOGY.toml` (primary)
  - `artifact-rocblas.toml` (component-specific)

**Documentation:**
- **Markdown**: `snake_case.md` or `kebab-case.md`
  - `build_system.md`, `development-guide.md`
  - `README.md` (standard)

### Variables

**CMake:**
```cmake
# Project-scoped: THEROCK_ prefix, SCREAMING_SNAKE_CASE
set(THEROCK_ENABLE_ROCBLAS ON)
set(THEROCK_AMDGPU_FAMILIES "gfx1100")
set(THEROCK_BUILD_TESTING ON)

# Local/internal: lowercase or Mixed_Case
set(component_source_dir "/path")
set(_internal_var "value")

# Cache variables: SCREAMING_SNAKE_CASE
option(BUILD_TESTING "Enable testing" ON)
```

**Python:**
```python
# Constants: SCREAMING_SNAKE_CASE
DEFAULT_TIMEOUT = 300
MAX_RETRIES = 3

# Functions/methods: snake_case
def fetch_artifacts(commit_hash: str) -> List[str]:
    pass

# Classes: PascalCase
class ArtifactBuilder:
    pass

# Variables: snake_case
artifact_name = "rocblas"
build_dir = Path("/path/to/build")
```

**TOML:**
```toml
# Sections: snake_case or kebab-case
[source_sets.base]
[artifact_groups.math-libs]

# Keys: snake_case
description = "Math libraries"
artifact_group_deps = ["core-runtime"]

# Values: depend on type
type = "target-specific"  # lowercase enum
feature_name = "ROCBLAS"  # UPPERCASE
```

### Functions and Targets

**CMake Functions:**
```cmake
# Project functions: therock_ prefix
therock_add_subproject(...)
therock_add_feature(...)

# Component functions: component_ prefix
rocblas_pre_hook()
```

**CMake Targets:**
```cmake
# Component targets: component-name (hyphens)
ninja rocblas
ninja hip-clr

# Phased targets: component+phase
ninja rocblas+build
ninja rocblas+dist

# Aggregate targets: descriptive names
ninja dist
ninja artifacts
ninja build-tests
```

**Python Functions:**
```python
# Public API: descriptive snake_case
def parse_topology(toml_path: Path) -> BuildTopology:
    pass

# Internal: leading underscore
def _validate_artifact_deps(artifact: Artifact) -> bool:
    pass
```

### Classes

**Python:**
```python
# Public classes: PascalCase
class BuildTopology:
    pass

class ArtifactGroup:
    pass

# Internal/helper: _PascalCase
class _InternalHelper:
    pass
```

## Code Structure

### CMake Patterns

**Module Structure:**
```cmake
# Header comment
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Include guards (for functions, not include_guard())
if(COMMAND therock_add_subproject)
  return()
endif()

# Dependencies
include(ExternalProject)

# Main function definition
function(therock_add_subproject)
  # Parse arguments
  cmake_parse_arguments(ARG "OPT1" "SINGLE" "MULTI" ${ARGN})
  
  # Implementation
  # ...
endfunction()
```

**Component CMakeLists.txt:**
```cmake
# Copyright header
# SPDX-License-Identifier: MIT

# Add pre-hook if exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pre_hook_component.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/pre_hook_component.cmake")
  component_pre_hook()
endif()

# Add subproject
therock_add_subproject(
  NAME component-name
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodule"
  CMAKE_ARGS
    "-DOPTION=VALUE"
  ARTIFACT_GROUP group-name
)

# Add post-hook if exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/post_hook_component.cmake")
  include("${CMAKE_CURRENT_SOURCE_DIR}/post_hook_component.cmake")
  component_post_hook()
endif()
```

### Python Patterns

**Script Structure:**
```python
#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Module docstring explaining purpose.

Detailed description of what this script does.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add build_tools to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from _therock_utils.module import Class


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", required=True, help="Option help")
    args = parser.parse_args()
    
    # Implementation
    # ...
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Type Hints:**
```python
# Always use type hints for public APIs
def fetch_artifacts(
    artifact_name: str,
    commit: str,
    output_dir: Path
) -> List[Path]:
    """Fetch artifacts from storage."""
    pass

# Return types
def parse_config(path: Path) -> Optional[Dict[str, Any]]:
    pass
```

**Path Handling:**
```python
# Always use pathlib.Path
from pathlib import Path

# Script-relative paths (don't assume cwd)
script_dir = Path(__file__).parent
config_file = script_dir / "config.toml"

# Absolute paths in arguments
def process_file(file_path: Path) -> None:
    file_path = file_path.resolve()  # Make absolute
```

## Testing Patterns

### CMake Test Organization

```cmake
# In component CMakeLists.txt
if(THEROCK_BUILD_TESTING)
  add_subdirectory(tests)
endif()

# In tests/CMakeLists.txt
add_executable(component-test test_main.cpp)
target_link_libraries(component-test component_lib gtest)

add_test(NAME component-test COMMAND component-test)
```

### Python Test Organization

```python
# Use pytest for Python tests
# File: test_module.py

import pytest
from module import function_to_test


def test_basic_functionality():
    result = function_to_test("input")
    assert result == "expected"


def test_error_handling():
    with pytest.raises(ValueError):
        function_to_test(None)
```

### Test Naming

- **C++ tests**: `test_component` or `component-test`
- **Python tests**: `test_*.py` (pytest convention)
- **CTest names**: Descriptive names, e.g., `rocblas-level3-gemm`

## Common Idioms

### CMake Error Handling

```cmake
# Check required variables
if(NOT DEFINED REQUIRED_VAR)
  message(FATAL_ERROR "REQUIRED_VAR must be set")
endif()

# Conditional execution
if(THEROCK_ENABLE_COMPONENT)
  # Only runs if enabled
endif()

# Status messages
message(STATUS "Building component: ${COMPONENT_NAME}")
message(WARNING "Feature X is deprecated")
message(FATAL_ERROR "Build failed: ${REASON}")
```

### Python Error Handling

```python
# Raise meaningful exceptions
if not artifact_exists(name):
    raise ValueError(f"Artifact '{name}' not found")

# Use context managers
with open(file_path) as f:
    data = f.read()

# Logging instead of print
import logging
logging.info("Processing artifact %s", artifact_name)
logging.error("Failed to fetch: %s", error_msg)
```

### TOML Schema Pattern

```toml
# Entity definitions follow consistent structure
[artifacts.component-name]
artifact_group = "group-name"         # Required: parent group
type = "target-neutral"               # Required: artifact type
artifact_deps = ["dep1", "dep2"]      # Optional: dependencies
feature_name = "COMPONENT_NAME"       # Optional: CMake variable override
platform = "linux"                    # Optional: platform restriction
disable_platforms = ["windows"]       # Optional: exclude platforms
python_requires = ["-r requirements.txt"]  # Optional: Python deps
```

## Comments and Documentation

### CMake Comments

```cmake
# Single line comments for brief explanations
set(VAR "value")  # Inline comment

# Block comments for complex logic
# This function does X by:
# 1. Parsing arguments
# 2. Validating inputs
# 3. Calling ExternalProject_Add
function(complex_function)
  # ...
endfunction()

# No multi-line comment syntax - use # on each line
```

### Python Docstrings

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Short one-line summary.
    
    Longer description if needed, explaining what the function does,
    its parameters, return value, and any side effects.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If arg1 is empty
    """
    pass

class MyClass:
    """
    Class summary.
    
    Detailed description of the class purpose and usage.
    """
    
    def method(self) -> None:
        """Method docstring."""
        pass
```

### Code Comments (General)

**When to comment:**
- Non-obvious "why" (not "what")
- Workarounds for specific bugs/limitations
- Performance-critical sections
- Complex algorithms

**When NOT to comment:**
- Obvious code (let names speak)
- Repeating what code does
- Outdated information

**Examples:**

```python
# Good: Explains WHY
# Use msgpack format for 10x faster deserialization than JSON
kernel_db = load_msgpack(db_path)

# Bad: Explains WHAT (obvious from code)
# Load the kernel database
kernel_db = load_msgpack(db_path)

# Good: Workaround explanation
# hipBLAS can't detect GPU at configure time, defer to runtime
# See: https://github.com/ROCm/hipBLAS/issues/123
use_runtime_detection = True

# Good: Performance note
# Batch size 64 measured as optimal for gfx90a in benchmarks
BATCH_SIZE = 64
```

## Build System Conventions

### ExternalProject Pattern

```cmake
ExternalProject_Add(
  component-name
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/source"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build"
  CMAKE_ARGS
    "-DCMAKE_INSTALL_PREFIX=${STAGE_DIR}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  BUILD_ALWAYS TRUE
  INSTALL_DIR "${STAGE_DIR}"
)
```

### Artifact Hook Pattern

```cmake
# pre_hook runs before configure
function(component_pre_hook)
  # Set options that affect CMake configuration
  set(COMPONENT_OPTION "value" PARENT_SCOPE)
endfunction()

# post_hook runs after dist
function(component_post_hook)
  # Custom post-install operations
  # File manipulation, wrapper generation, etc.
endfunction()
```

### Feature Flag Pattern

```cmake
# topology_to_cmake.py generates
therock_add_feature(
  FEATURE COMPONENT_NAME      # Creates THEROCK_ENABLE_COMPONENT_NAME
  GROUP GROUP_NAME            # Creates THEROCK_ENABLE_GROUP_NAME
  DEFAULT ON
  DESCRIPTION "Component description"
)

# Users configure with
cmake -DTHEROCK_ENABLE_COMPONENT_NAME=OFF
cmake -DTHEROCK_ENABLE_GROUP_NAME=OFF  # Disables all in group
```

## Version Control Patterns

### Commit Messages

Follow conventional commits style (observed in git log):

```
type(scope): Short summary (≤72 chars)

Detailed explanation if needed.

- Bullet points for changes
- Reference issues: #123

Co-Authored-By: Name <email>
```

**Types:**
- `fix`: Bug fixes
- `feat`: New features
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `ci`: CI/CD changes
- `build`: Build system changes

### Branch Naming

```
users/<username>/<description>   # Personal branches
shared/<description>              # Shared feature branches
```

Examples:
- `users/jsmith/add-rocblas-test`
- `shared/windows-support`

### Submodule Updates

```bash
# Fetch sources resets ALL submodules (destructive)
python3 ./build_tools/fetch_sources.py

# Update specific submodule
cd component/submodule
git fetch origin
git checkout origin/main

# Apply patches
cd ../..
python3 ./build_tools/fetch_sources.py  # Reapplies patches
```

## Style Enforcement

### pre-commit Hooks

TheRock uses pre-commit for automated formatting:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black          # Python formatter
  - repo: https://github.com/pre-commit/mirrors-clang-format
    hooks:
      - id: clang-format   # C++ formatter
  - repo: https://github.com/executablebooks/mdformat
    hooks:
      - id: mdformat        # Markdown formatter
  - repo: https://github.com/rhysd/actionlint
    hooks:
      - id: actionlint      # GitHub Actions linter
```

### Running Checks

```bash
# Install hooks
pip install pre-commit
pre-commit install

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## Platform Considerations

### Cross-Platform Paths

```python
# Use pathlib for all path operations
from pathlib import Path

# Works on Windows and Linux
path = Path("build") / "component" / "output"

# Avoid string concatenation
# Bad: path = "build/" + component + "/output"
```

### Platform-Specific Code

```cmake
# CMake platform detection
if(WIN32)
  # Windows-specific
elseif(UNIX)
  # Linux/Unix-specific
endif()

# Platform in TOML
[artifacts.component]
platform = "linux"  # Only build on Linux
disable_platforms = ["windows"]  # Build everywhere except Windows
```

```python
# Python platform detection
import platform

if platform.system() == "Windows":
    # Windows-specific
elif platform.system() == "Linux":
    # Linux-specific
```

## Key Principles

From style guides in `docs/development/style_guides/`:

1. **Use pathlib.Path** for filesystem operations (Python)
2. **Add type hints** to function signatures (Python)
3. **Use argparse** for CLI with help text (Python)
4. **Don't assume cwd** - use script-relative paths
5. **Dependencies at super-project level** (CMake)
6. **Follow build phases**: configure → build → stage → dist
