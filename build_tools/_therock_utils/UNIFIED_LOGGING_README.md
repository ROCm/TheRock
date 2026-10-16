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
2025-12-17 10:30:45 - therock.installer - INFO - Installing package: rocm-core
2025-12-17 10:30:47 - therock.installer - INFO - ✅ Completed operation: Installing rocm-core (2300.50ms)
2025-12-17 10:30:50 - therock.test - INFO - Test passed: test_gemm (0.5s)
2025-12-17 10:30:51 - therock.test - ERROR - Test failed: test_trsm - Assertion failed
```
✅ Clear timestamps, context, and automatic timing!

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
logger.info("Starting compilation")
logger.warning("Deprecated flag used")
logger.error("Compilation failed")
```

### 2. Timed Operations
```python
with logger.timed_operation("Installing packages"):
    install_packages()
# Automatically logs start/completion with duration_ms
```

### 3. Structured Data
```python
metrics = {"total_packages": 5, "duration_sec": 45.3}
logger.log_dict(metrics, message="📊 Metrics:")
# Displays JSON-formatted data
```

### 4. Exception Handling
```python
try:
    build_package()
except Exception as e:
    logger.log_exception(e, "Build failed")
# Captures full stack trace with context
```

### 5. Test Framework Integration
- Common parser for both GTest and CTest
- Automatic result logging (passed/failed/skipped)
- Performance metrics and summary statistics

---

## Four Logging Types

| Type | Purpose | Usage |
|------|---------|-------|
| **Standard Logging** | Operational messages | `logger.info()`, `logger.error()` |
| **Timed Operations** | Automatic duration tracking | `with logger.timed_operation("name")` |
| **Structured Data** | JSON-formatted metrics | `logger.log_dict(metrics)` |
| **Exception Handling** | Full stack traces | `logger.log_exception(e, "msg")` |

---

## How to Use

### Basic Usage
```python
from logging_config import get_logger

logger = get_logger(__name__, component="Build", operation="compile")
logger.info("Starting compilation")
```

### Timed Operations
```python
with logger.timed_operation("Package Installation"):
    install_package()
```

### Structured Data
```python
metrics = {"total_packages": 5, "duration_sec": 45.3}
logger.log_dict(metrics, message="📊 Metrics:")
```

### Exception Handling
```python
try:
    build_package()
except Exception as e:
    logger.log_exception(e, "Build failed")
```

### Test Runner
```python
from test_runner import TestRunner

runner = TestRunner(component="rocBLAS", test_type="smoke")
exit_code = runner.run_gtest(raw_output=gtest_output)
```

---

## Before vs After Comparison

**BEFORE (Inconsistent):**
```
Starting build...
Compiling...
Done
[  PASSED  ] 45 tests
[  FAILED  ] 3 tests
```
❌ No timestamps, no context, no metrics

**AFTER (Unified Logging):**
```
2025-12-17 10:30:00 - therock.build - INFO - Starting build
2025-12-17 10:30:05 - therock.build - INFO - ✅ Completed operation: Configuration (5200ms)
2025-12-17 10:32:15 - therock.build - INFO - ✅ Completed operation: Compilation (130000ms)
2025-12-17 10:32:15 - therock.build - INFO - 📊 Build Metrics:
{
  "source_files": 250,
  "duration_sec": 135.2
}
2025-12-17 10:32:20 - therock.test - INFO - Test: gemm_float - PASSED (0.5s)
2025-12-17 10:32:22 - therock.test - ERROR - Test: trsm_float - FAILED
2025-12-17 10:32:22 - therock.test - INFO - 📊 Test Metrics:
{
  "total_tests": 48,
  "passed": 45,
  "failed": 3
}
```
✅ Clear timestamps, context, timing, and structured metrics!

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

**GitHub Actions:** `.github/workflows/logging_demo.yml`

Demonstrates all four logging types:
- Package installer (download, install, verify)
- Build system (configure, compile, test)
- GTest integration (rocROLLER with failure scenarios)
- CTest integration (rocWMMA with failure scenarios)

---

## POC Summary

**Problem:** Inconsistent logging made debugging difficult

**Solution:** Unified framework with standard logging, timed operations, structured data, and exception handling

**Result:** Clean, consistent, professional logging across all components

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

