# TheRock Unified Logging Framework

## What is Unified Logging?

Unified logging means **all components of TheRock use the same logging system** with consistent formatting, making logs easier to read, search, and analyze.

---

## The Problem Before (Inconsistent Logging)

### ❌ Multiple Different Logging Styles

Each part of TheRock logged differently:

```python
# Package installer did this:
print(f"Installing package {name}...")

# Build system did this:
logger.info(f"Building component: {component}")

# Test runner did this:
sys.stdout.write("Running tests...\n")
```

### Issues with Old Approach:

1. **Hard to Read**: Different formats made logs confusing
2. **No Timestamps**: Couldn't tell when things happened
3. **No Context**: Missing important details (component name, severity)
4. **Hard to Search**: Different formats = difficult to grep/filter
5. **No Structured Data**: Couldn't easily extract metrics or debug info
6. **Inconsistent Errors**: Exception handling varied across components

**Example of Old Log Output:**
```
Installing packages...
Building component
Test passed
ERROR
Something failed
```
❌ What component? When? Why did it fail?

---

## The Solution (Unified Logging)

### ✅ Single Logging System for Everything

Now **all components** use `TheRockLogger`:

```python
from logging_config import get_logger

logger = get_logger(__name__)
logger.info("Installing package", extra={"package": name, "version": version})
```

### Benefits of New Approach:

1. **Consistent Format**: Every log looks the same
2. **Always Has Timestamps**: Know exactly when things happened
3. **Clear Context**: Component name and severity level in every message
4. **Easy to Search**: Same format = easy grep/filter
5. **Structured Data**: Extra fields for metrics and debugging
6. **Automatic Timing**: Built-in performance measurement
7. **Better Error Handling**: Consistent exception logging with tracebacks

**Example of New Log Output:**
```
2025-12-17 10:30:45 - packaging.installer - INFO - Installing package | package=rocm-core version=6.3.0
2025-12-17 10:30:47 - build.cmake - INFO - Building component | component=rocBLAS duration=2.3s
2025-12-17 10:30:50 - test.runner - INFO - Test passed | test=test_gemm duration=0.5s
2025-12-17 10:30:51 - test.runner - ERROR - Test failed | test=test_trsm error="Assertion failed"
```
✅ Clear who, what, when, and why!

---

## What Changed?

### Components Now Using Unified Logging:

| Component | Before | After |
|-----------|--------|-------|
| **Package Installer** | `print()` statements | `TheRockLogger` |
| **Build System** | Mixed logging | `TheRockLogger` |
| **Test Runner** | Custom output | `TheRockLogger` |
| **GTest Integration** | Raw test output | Parsed & logged |
| **CTest Integration** | Raw test output | Parsed & logged |

### New Files Added:

- **`logging_config.py`**: Core logging configuration
  - `get_logger()`: Get a logger instance
  - `configure_root_logger()`: Set up logging format
  - `timed_operation()`: Context manager for timing

- **`test_runner.py`**: Enhanced test execution with logging
  - Captures GTest/CTest output
  - Parses test results
  - Logs passed/failed/skipped tests
  - Reports timing and statistics

- **Sample Applications**: Demonstrate logging usage
  - `sample_package_installer.py`
  - `sample_build_system.py`

- **Demo Tests**: Show real-world integration
  - `demo_test_rocroller.py` (GTest)
  - `demo_test_rocwmma.py` (CTest)

---

## Key Features

### 1. Automatic Timing

```python
with logger.timed_operation("Installing packages"):
    install_packages()
```

**Output:**
```
2025-12-17 10:30:45 - installer - INFO - Installing packages
2025-12-17 10:30:48 - installer - INFO - Installing packages | duration=3.2s
```

### 2. Structured Data

```python
logger.info("Build completed", extra={
    "component": "rocBLAS",
    "build_type": "Release",
    "duration": 45.3
})
```

**Output:**
```
2025-12-17 10:30:45 - build - INFO - Build completed | component=rocBLAS build_type=Release duration=45.3
```

### 3. Consistent Error Handling

```python
try:
    build_package()
except Exception as e:
    logger.error(f"Build failed: {e}", exc_info=True)
```

**Output:**
```
2025-12-17 10:30:45 - build - ERROR - Build failed: CMake error
Traceback (most recent call last):
  File "build.py", line 42, in build_package
    run_cmake()
  ...
```

### 4. Test Result Parsing

The test runner automatically parses and logs:
- ✅ Passed tests
- ❌ Failed tests (with failure details)
- ⏭️ Skipped tests
- ⏱️ Test timing
- 📊 Summary statistics

---

## How to Use

### Basic Usage

```python
from logging_config import get_logger

# Get a logger for your component
logger = get_logger(__name__)

# Log at different levels
logger.debug("Detailed debug info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")

# Add structured data
logger.info("Processing file", extra={"filename": "data.txt", "size": 1024})

# Time operations
with logger.timed_operation("Long operation"):
    do_something()
```

### Running Tests with Logging

```python
from test_runner import TestRunner

runner = TestRunner()
exit_code = runner.run_test(
    test_name="rocBLAS",
    test_command=["./run_tests"],
    test_type="gtest"  # or "ctest"
)
```

---

## Before vs After Comparison

### Scenario: Building and Testing a Component

**BEFORE (Inconsistent):**
```
Starting build...
rocBLAS
Compiling...
Done
Running tests
[  PASSED  ] 45 tests
[  FAILED  ] 3 tests
```

**AFTER (Unified):**
```
2025-12-17 10:30:00 - build.rocblas - INFO - Starting build | component=rocBLAS variant=Release
2025-12-17 10:30:00 - build.rocblas - INFO - Compiling source files
2025-12-17 10:32:15 - build.rocblas - INFO - Build completed | duration=135.2s
2025-12-17 10:32:15 - test.rocblas - INFO - Running GTest suite | component=rocBLAS
2025-12-17 10:32:20 - test.rocblas - INFO - Test passed | test=gemm_float duration=0.5s
2025-12-17 10:32:21 - test.rocblas - INFO - Test passed | test=gemm_double duration=0.4s
2025-12-17 10:32:22 - test.rocblas - ERROR - Test failed | test=trsm_float error="Assertion failed at line 42"
2025-12-17 10:32:22 - test.rocblas - INFO - Test results | total=48 passed=45 failed=3 skipped=0 duration=7.2s
```

---

## Benefits Summary

| Benefit | Impact |
|---------|--------|
| **Consistent Format** | Easy to read and understand |
| **Searchable** | Filter logs by component, level, time |
| **Debuggable** | Rich context and structured data |
| **Measurable** | Automatic timing for performance tracking |
| **Maintainable** | One system to update, not many |
| **Professional** | Production-ready logging standards |

---

## Demo Workflow

See the unified logging in action:
- **GitHub Actions Workflow**: `.github/workflows/logging_demo.yml`
- Runs automatically on push to `users/rponnuru/logging_poc_3`
- Shows real examples from package installation, builds, and tests

---

## Summary

**Before:** Each component logged differently → confusing, hard to debug
**After:** All components use unified logging → clear, consistent, professional

The unified logging framework makes TheRock's build, test, and packaging systems **easier to understand, debug, and maintain**.

---

**Last Updated:** December 18, 2025

