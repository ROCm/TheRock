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
2025-12-17 10:30:45 - therock.packaging.installer - INFO - Installing package: rocm-core
2025-12-17 10:30:45 - therock.packaging.installer - DEBUG - Starting operation: Installing rocm-core
2025-12-17 10:30:47 - therock.packaging.installer - INFO - ✅ Completed operation: Installing rocm-core (2300.50ms)
2025-12-17 10:30:47 - therock.build.cmake - INFO - Building component: rocBLAS
2025-12-17 10:30:50 - therock.test.runner - INFO - Test passed: test_gemm (0.5s)
2025-12-17 10:30:51 - therock.test.runner - ERROR - Test failed: test_trsm
2025-12-17 10:30:51 - therock.test.runner - ERROR - Assertion failed: expected 1.0, got 0.999
```
✅ Clear who, what, when, and why with automatic timing!

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

### Files Modified/Added:

**Core Framework:**
- **`logging_config.py`** - Core logging framework with standard logging, `timed_operation()` context manager, `log_exception()` for unified error handling, and `log_dict()` for structured data output

**Sample Demonstrations:**
- **`sample_package_installer.py`** - Package installer sample demonstrating timed operations, structured metrics (download, installation, verification), and exception logging
- **`sample_build_system.py`** - Build system sample demonstrating nested timed operations, compilation metrics, test execution, and exception handling with structured data output

**Test Framework Integration:**
- **`test_runner.py`** - Common test runner providing standardized parsing and logging for both GTest and CTest frameworks with failure tracking and performance metrics
- **`demo_test_rocroller.py`** - GTest integration demo showing test result parsing, failure scenarios, and structured logging for rocROLLER component
- **`demo_test_rocwmma.py`** - CTest integration demo showing test result parsing, GPU-specific test logging, and failure scenarios for rocWMMA component

**GitHub Actions Workflow:**
- **`logging_demo.yml`** - CI/CD workflow demonstrating all logging capabilities: package installer, build system, and GTest/CTest integration with failure scenarios

**Documentation:**
- **`UNIFIED_LOGGING_README.md`** - This file: comprehensive documentation with usage examples, best practices, and integration guide

---

## Key Features

### 1. Standard Leveled Logging

```python
logger = get_logger(__name__, component="Build")

logger.debug("Checking dependencies")
logger.info("Starting compilation")
logger.warning("Deprecated flag used")
logger.error("Compilation failed")
```

**Output:**
```
2025-12-17 10:30:45 - therock.build - DEBUG - Checking dependencies
2025-12-17 10:30:46 - therock.build - INFO - Starting compilation
2025-12-17 10:30:47 - therock.build - WARNING - Deprecated flag used
2025-12-17 10:30:48 - therock.build - ERROR - Compilation failed
```

### 2. Automatic Timing with `timed_operation()`

```python
with logger.timed_operation("Installing packages"):
    install_packages()
```

**Output:**
```
2025-12-17 10:30:45 - therock.installer - DEBUG - Starting operation: Installing packages
2025-12-17 10:30:48 - therock.installer - INFO - ✅ Completed operation: Installing packages (3200.45ms)
```
*Automatically logs start (DEBUG) and completion (INFO) with duration in milliseconds*

### 3. Structured Data with `log_dict()`

```python
logger.info("Build completed")

build_metrics = {
    "component": "rocBLAS",
    "build_type": "Release",
    "duration_sec": 45.3,
    "source_files": 250,
    "warnings": 0
}
logger.log_dict(build_metrics, message="📊 Build Metrics:")
```

**Output:**
```
2025-12-17 10:30:45 - therock.build - INFO - Build completed
2025-12-17 10:30:45 - therock.build - INFO - 📊 Build Metrics:
{
  "build_type": "Release",
  "component": "rocBLAS",
  "duration_sec": 45.3,
  "source_files": 250,
  "warnings": 0
}
```
*Clean standard logs + explicit JSON-formatted structured data when needed*

### 4. Unified Exception Handling with `log_exception()`

```python
try:
    build_package()
except Exception as e:
    logger.log_exception(e, "Build failed for component", extra={
        "component": "rocBLAS",
        "build_type": "Release"
    })
```

**Output:**
```
2025-12-17 10:30:45 - therock.build - ERROR - Build failed for component
Traceback (most recent call last):
  File "build.py", line 42, in build_package
    run_cmake()
  File "cmake.py", line 15, in run_cmake
    raise RuntimeError("CMake configuration failed")
RuntimeError: CMake configuration failed
```
*Captures full stack trace with contextual information*

### 5. Test Framework Integration

The test runner automatically parses and logs results for both GTest and CTest:
- ✅ Passed tests with timing
- ❌ Failed tests with detailed error messages
- ⏭️ Skipped tests with reasons
- ⏱️ Individual test and total execution timing
- 📊 Summary statistics (pass rate, failure count)
- 🔍 Structured output for CI/CD analysis

---

## Logging Types Demonstrated

### Type 1: Standard Leveled Logging
**Purpose:** Basic operational messages with consistent formatting  
**Usage:** `logger.info()`, `logger.debug()`, `logger.warning()`, `logger.error()`  
**Example:** Installation progress, configuration steps, warnings about deprecated features

### Type 2: Timed Operations
**Purpose:** Automatic execution duration tracking for performance analysis  
**Usage:** `with logger.timed_operation("operation_name")`  
**Example:** Build compilation time, package download duration, test execution time  
**Metrics:** Automatically logs `duration_ms` for each operation

### Type 3: Structured Data
**Purpose:** Machine-readable metrics and data visualization  
**Usage:** `logger.log_dict(data_dict, message="label")`  
**Example:** File counts, package sizes, test results, success rates  
**Format:** Clean JSON output separate from standard logs

### Type 4: Exception Handling
**Purpose:** Comprehensive error logging with full context  
**Usage:** `logger.log_exception(exception, message, extra={...})`  
**Example:** Build failures, test crashes, installation errors  
**Captures:** Full stack trace, error type, contextual metadata

---

## How to Use

### 1. Standard Logging

```python
from logging_config import get_logger

# Get a logger for your component
logger = get_logger(__name__, component="PackageInstaller", operation="install")

# Log at different levels
logger.debug("Checking package dependencies")
logger.info("Installing package: rocm-core")
logger.warning("Package signature verification skipped")
logger.error("Installation failed: disk space insufficient")
```

### 2. Timed Operations

```python
# Automatic timing for operations
with logger.timed_operation("Package Installation"):
    download_package()
    verify_package()
    install_package()

# Nested timed operations
with logger.timed_operation("Build Process"):
    with logger.timed_operation("Configuration"):
        run_cmake()
    with logger.timed_operation("Compilation"):
        run_make()
```

### 3. Structured Data Display

```python
# Use log_dict() to display structured metrics
logger.info("Installation completed")

install_metrics = {
    "total_packages": 5,
    "total_size_mb": 450,
    "files_installed": 1250,
    "duration_sec": 45.3
}
logger.log_dict(install_metrics, message="📊 Installation Metrics:")
```

### 4. Exception Handling

```python
try:
    install_package(package_name)
except Exception as e:
    logger.log_exception(e, f"Failed to install {package_name}", extra={
        "package_name": package_name,
        "error_type": type(e).__name__
    })
```

### 5. Running Tests with Logging

```python
from test_runner import TestRunner

# GTest
runner = TestRunner(component="rocBLAS", test_type="smoke", operation="gtest")
exit_code = runner.run_gtest(raw_output=gtest_output)

# CTest
runner = TestRunner(component="rocWMMA", test_type="regression", operation="ctest")
exit_code = runner.run_ctest(raw_output=ctest_output)
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
❌ No timestamps, no context, no metrics

**AFTER (Unified Logging):**
```
2025-12-17 10:30:00 - therock.build.rocblas - INFO - Starting build
2025-12-17 10:30:00 - therock.build.rocblas - DEBUG - Starting operation: Configuration
2025-12-17 10:30:05 - therock.build.rocblas - INFO - ✅ Completed operation: Configuration (5200.34ms)
2025-12-17 10:30:05 - therock.build.rocblas - DEBUG - Starting operation: Compilation
2025-12-17 10:32:15 - therock.build.rocblas - INFO - ✅ Completed operation: Compilation (130000.12ms)
2025-12-17 10:32:15 - therock.build.rocblas - INFO - Build completed
2025-12-17 10:32:15 - therock.build.rocblas - INFO - 📊 Build Metrics:
{
  "source_files": 250,
  "object_files": 250,
  "output_size_kb": 2048,
  "total_duration_sec": 135.2,
  "warnings": 0
}
2025-12-17 10:32:15 - therock.test.rocblas - INFO - Running GTest suite
2025-12-17 10:32:20 - therock.test.rocblas - INFO - Test: gemm_float - PASSED (0.5s)
2025-12-17 10:32:21 - therock.test.rocblas - INFO - Test: gemm_double - PASSED (0.4s)
2025-12-17 10:32:22 - therock.test.rocblas - ERROR - Test: trsm_float - FAILED
2025-12-17 10:32:22 - therock.test.rocblas - ERROR - Assertion failed at line 42: expected 1.0, got 0.999
2025-12-17 10:32:22 - therock.test.rocblas - INFO - Test execution completed
2025-12-17 10:32:22 - therock.test.rocblas - INFO - 📊 Test Metrics:
{
  "total_tests": 48,
  "passed": 45,
  "failed": 3,
  "skipped": 0,
  "success_rate_pct": 93.75,
  "total_duration_sec": 7.2
}
```
✅ Clear timestamps, context, automatic timing, structured metrics!

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

See all four logging types in action:

**GitHub Actions Workflow:** `.github/workflows/logging_demo.yml`
- Runs automatically on push to `users/rponnuru/logging_poc_3`
- **Core Samples:**
  - Package Installer (download, install, verify with timed operations and structured metrics)
  - Build System (configure, compile, test with nested timing and exception handling)
- **Test Integration:**
  - rocROLLER GTest demo (failure scenarios with detailed error logging)
  - rocWMMA CTest demo (failure scenarios with GPU-specific test logging)

**What's Demonstrated:**
1. ✅ Standard leveled logging throughout all operations
2. ⏱️ Timed operations with automatic duration tracking (`duration_ms`)
3. 📊 Structured data display using `log_dict()` for metrics
4. ⚠️ Exception handling with full stack traces and context
5. 🧪 Common test framework integration for GTest and CTest

---

## POC Summary

**Problem:** Inconsistent logging across components made debugging difficult and logs hard to analyze

**Solution:** Unified logging framework with four key capabilities:
1. **Standard Logging** - Consistent format with timestamps and severity levels
2. **Timed Operations** - Automatic performance tracking via context manager
3. **Structured Data** - JSON-formatted metrics via `log_dict()` for analysis
4. **Exception Handling** - Comprehensive error logging with full context

**Result:** Clean, consistent, professional logging across all TheRock components

---

## Benefits Summary

| Benefit | Impact |
|---------|--------|
| **Consistent Format** | Easy to read and understand across all components |
| **Searchable** | Filter logs by component, level, time, and operation |
| **Debuggable** | Rich context with structured data and full tracebacks |
| **Measurable** | Automatic timing for performance tracking and optimization |
| **Analyzable** | JSON-formatted metrics for dashboards and monitoring |
| **Maintainable** | One system to update, not many disparate approaches |
| **Professional** | Production-ready logging standards and best practices |

---

**Last Updated:** December 18, 2025

