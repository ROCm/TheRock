# Unified Logging Integration for GTest/CTest

This document describes the unified logging integration for test and packaging scripts in TheRock.

## Overview

The unified logging framework has been extended to provide consistent logging across:
- **GTest executables** (e.g., `hipblaslt-test`, `rocroller-tests`)
- **CTest test suites** (e.g., `rocwmma`, `rocthrust`)
- **Packaging scripts** (e.g., `build_package.py`)

## Changes Made

### 1. New Test Runner Utility (`test_runner.py`)

A common test runner that provides:
- ✅ Unified logging with GitHub Actions annotations
- ✅ Collapsible log groups for test organization
- ✅ Automatic test result parsing (GTest and CTest)
- ✅ Structured logging with metadata
- ✅ Performance timing tracking
- ✅ Proper error handling with GitHub annotations

**Location:** `build_tools/_therock_utils/test_runner.py`

**Key Features:**
```python
from test_runner import TestRunner

# For CTest
runner = TestRunner(component="rocwmma", test_type="full")
runner.run_ctest(
    test_dir=Path(f"{THEROCK_BIN_DIR}/rocwmma"),
    parallel=8,
    timeout="3600"
)

# For GTest
runner = TestRunner(component="hipblaslt", test_type="smoke")
runner.run_gtest(
    cmd=[f"{THEROCK_BIN_DIR}/hipblaslt-test", "--gtest_filter=*smoke*"],
    cwd=THEROCK_DIR,
    env=environ_vars
)
```

### 2. Updated Test Scripts

#### `test_rocwmma.py` (CTest Example)
**Before:**
```python
import logging
logging.basicConfig(level=logging.INFO)
cmd = ["ctest", "--test-dir", f"{THEROCK_BIN_DIR}/rocwmma", ...]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True)
```

**After:**
```python
from test_runner import TestRunner
from logging_config import configure_root_logger
import logging

configure_root_logger(level=logging.INFO)
runner = TestRunner(component="rocwmma", test_type=test_type)
runner.run_ctest(
    test_dir=Path(f"{THEROCK_BIN_DIR}/rocwmma{test_subdir}"),
    parallel=8,
    timeout=timeout,
    cwd=THEROCK_DIR,
    env=environ_vars
)
```

**Benefits:**
- ✅ GitHub Actions annotations for test failures
- ✅ Collapsible "🧪 Running rocwmma CTest" groups
- ✅ Automatic timing and result logging
- ✅ Structured metadata (component, test_type, etc.)

#### `test_rocroller.py` (GTest Example)
**Before:**
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
cmd = [str(test_bin), test_filter_arg]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=str(THEROCK_DIR), check=True, env=env)
```

**After:**
```python
from test_runner import TestRunner
from logging_config import configure_root_logger, get_logger
import logging

configure_root_logger(level=logging.INFO)
logger = get_logger(__name__, component="rocroller", operation="test")
runner = TestRunner(component="rocroller", test_type=TEST_TYPE)
runner.run_gtest(cmd=cmd, cwd=THEROCK_DIR, env=env, capture_output=True)
```

**Benefits:**
- ✅ Automatic GTest output parsing
- ✅ GitHub annotations for each failed test
- ✅ Structured result reporting
- ✅ DEBUG-level logging of full test output

### 3. Packaging Scripts Integration

#### Updated `packaging_utils.py`
Added unified logging support with INFO level (normal verbosity):

```python
from logging_config import get_logger, configure_root_logger
import logging

# Configure unified logging with INFO level (normal verbosity for packaging)
configure_root_logger(level=logging.INFO)

def get_packaging_logger(operation: str = None):
    """Get a logger instance for packaging operations"""
    return get_logger(__name__, component="packaging", operation=operation)
```

#### Updated `build_package.py`
Integrated unified logging:

```python
from packaging_utils import *

logger = get_packaging_logger(operation="build_package")

def main(argv):
    logger.info("Starting ROCm package build process")
    # ... argument parsing ...
    
    try:
        with logger.github_group(f"📦 Building {args.pkg_type} Packages"):
            run(args)
        logger.github_info("✅ Package build completed successfully")
    except Exception as e:
        logger.github_error(f"❌ Package build failed: {e}")
        raise
```

**Benefits:**
- ✅ INFO level by default (normal verbosity, not DEBUG)
- ✅ GitHub Actions annotations in CI
- ✅ Consistent with other TheRock components
- ✅ Structured logging with metadata

## Log Levels

| Script Type | Default Level | Rationale |
|-------------|---------------|-----------|
| Test Scripts | **INFO** | Show test execution progress, hide detailed test output |
| Packaging Scripts | **INFO** | Normal verbosity for build logs |
| Debug Mode | **DEBUG** | Set via `configure_root_logger(level=logging.DEBUG)` |

## GitHub Actions Integration

When running in GitHub Actions (`GITHUB_ACTIONS=true`):

### Test Failures
```
❌ rocwmma: 2/45 tests failed
   └─ Test failed: WarpScanTest.BasicScan
   └─ Test failed: BlockReduceTest.SimpleSum
```

### Log Groups
```
▼ 🧪 Running rocwmma CTest (full)
  │ Command: ctest --test-dir /path/to/rocwmma --parallel 8
  │ Test directory: /path/to/rocwmma
  │ Parallelism: 8 jobs
  │ Test Results: 43/45 passed, 2 failed
  │ ✅ rocwmma_ctest_execution completed (52.3s)
```

### Package Builds
```
▼ 📦 Building DEB Packages
  │ Package type: deb
  │ ROCm version: 7.1.0
  │ Target: gfx94X-dcgpu
  │ ✅ Package build completed successfully
```

## Migration Guide

### For Test Scripts

1. **Add imports:**
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build_tools" / "_therock_utils"))
   from test_runner import TestRunner
   from logging_config import configure_root_logger
   import logging
   ```

2. **Configure logging:**
   ```python
   configure_root_logger(level=logging.INFO)
   ```

3. **Replace subprocess calls with TestRunner:**
   
   **For CTest:**
   ```python
   runner = TestRunner(component="mycomponent", test_type="full")
   runner.run_ctest(test_dir=Path(f"{THEROCK_BIN_DIR}/mycomponent"))
   ```
   
   **For GTest:**
   ```python
   runner = TestRunner(component="mycomponent", test_type="full")
   runner.run_gtest(cmd=[f"{THEROCK_BIN_DIR}/mytest"], cwd=THEROCK_DIR)
   ```

### For Packaging Scripts

1. **Import the logger:**
   ```python
   from packaging_utils import get_packaging_logger
   logger = get_packaging_logger(operation="my_operation")
   ```

2. **Replace print() with logger methods:**
   ```python
   # Before
   print(f"Building package: {pkg_name}")
   
   # After
   logger.info(f"Building package: {pkg_name}")
   ```

3. **Add GitHub Actions groups for major operations:**
   ```python
   with logger.github_group("📦 Package Creation"):
       create_packages()
   logger.github_info("✅ Packages created successfully")
   ```

## Testing

### Local Testing
Run any updated script locally:
```bash
python build_tools/github_actions/test_executable_scripts/test_rocwmma.py
```

Output will show:
- INFO-level logs to console
- Colored output (on Linux/Mac terminals)
- No GitHub annotations (not in CI)

### GitHub Actions Testing
When run in a workflow:
- GitHub annotations appear in the UI
- Collapsible log groups organize output
- Failures create clickable annotations

## Examples

See the following files for complete examples:
- **CTest:** `build_tools/github_actions/test_executable_scripts/test_rocwmma.py`
- **GTest:** `build_tools/github_actions/test_executable_scripts/test_rocroller.py`
- **Packaging:** `build_tools/packaging/linux/build_package.py`
- **Test Runner:** `build_tools/_therock_utils/test_runner.py`

## Future Improvements

Potential enhancements:
1. **JUnit XML output** - Generate test reports for CI systems
2. **Test result caching** - Speed up re-runs of passing tests
3. **Parallel test execution** - Better utilize multi-GPU systems
4. **Flaky test detection** - Identify unstable tests automatically
5. **Performance regression detection** - Track test execution times

## Questions?

For more information about the unified logging framework, see:
- `build_tools/_therock_utils/logging_config.py` - Core logging implementation
- `build_tools/_therock_utils/sample_github_actions_logging.py` - GitHub Actions examples

