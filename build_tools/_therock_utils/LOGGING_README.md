# TheRock Standardized Logging Framework

**Version:** 1.0.0  
**Status:** Production Ready  
**Maintainer:** TheRock Infrastructure Team

---

## 🎯 Overview

A unified, production-grade logging framework for the entire TheRock project that replaces inconsistent logging patterns with a single, powerful API.

### Problems Solved

- ✅ **Inconsistent Logging** - Multiple patterns (`print()`, `log()`, `_log()`, `logging.getLogger()`) across the codebase
- ✅ **Manual Configuration** - Repeated logger setup in every file
- ✅ **Unstructured Logs** - String-only logs that are hard to parse and analyze
- ✅ **No GitHub Actions Integration** - Manual formatting of workflow commands
- ✅ **Missing Metrics** - No built-in performance timing
- ✅ **Poor Exception Tracking** - Inconsistent exception logging with missing tracebacks

---

## 🚀 Quick Start

### 1. Basic Usage

```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Hello, TheRock!")
logger.debug("Detailed diagnostic information")
logger.warning("Something needs attention")
logger.error("An error occurred")
logger.critical("Critical system failure!")
```

### 2. With Component Context

```python
logger = get_logger(__name__, component="PackageInstaller", operation="install")
logger.info("Installing package", extra={
    "package_name": "rocm-core",
    "version": "6.2.0",
    "progress": "1/3"
})
```

### 3. Automatic Performance Timing

```python
with logger.timed_operation("Installing rocm-core"):
    install_package("rocm-core")
    # Automatically logs:
    # DEBUG: "Starting operation: Installing rocm-core"
    # INFO: "✅ Completed operation: Installing rocm-core (502.34ms)"
```

### 4. Exception Handling

```python
try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed", extra={"context": "details"})
    # Automatically includes full traceback and exception details
```

### 5. GitHub Actions Integration

```python
logger.github_info("✅ Build completed successfully")
logger.github_warning("Deprecated API detected", file="src/api.py", line=42)
logger.github_error("Build failed", file="build.py", line=100)

with logger.github_group("📦 Installing Packages"):
    install_packages()  # Creates collapsible section in GitHub Actions
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `logging_config.py` | Core logging framework - **use this!** |
| `sample_package_installer.py` | Sample 1: Package installer (demonstrates core features) |
| `sample_build_system.py` | Sample 2: Build system (demonstrates core features) |
| `run_logging_demos.py` | Demo runner for both samples |
| `LOGGING_README.md` | This file - complete documentation |

**Demo both samples:**
```bash
cd build_tools/_therock_utils
python run_logging_demos.py
```

---

## ✨ Feature List

### 🎨 Core Logging Methods (Demonstrated in Samples)

| Method | Purpose | Example | In Samples |
|--------|---------|---------|------------|
| `debug()` | Detailed diagnostic info | `logger.debug("Configuration step: Checking dependencies")` | ✅ Both |
| `info()` | General information | `logger.info("Package installed successfully")` | ✅ Both |
| `warning()` | Non-critical issues | `logger.warning("Deprecated API used")` | ✅ Both |
| `error()` | Error conditions | `logger.error("Configuration validation failed")` | ✅ Build System |
| `critical()` | Severe failures | `logger.critical("Insufficient disk space!")` | ⚠️ Not in samples |
| `timed_operation()` | Automatic timing | `with logger.timed_operation("Install"): ...` | ✅ Both |
| `log_exception()` | Exception + traceback | `logger.log_exception(e, "Failed")` | ✅ Both |

### 🔧 Additional Methods (Available but not in samples)

| Method | Purpose | Example | Use Case |
|--------|---------|---------|----------|
| `log_dict()` | Format dictionaries | `logger.log_dict(config, "Config:")` | Display structured data |
| `github_info()` | GH Actions notices | `logger.github_info("✅ Tests passed")` | CI/CD workflows |
| `github_warning()` | GH Actions warnings | `logger.github_warning("Warning")` | CI/CD workflows |
| `github_error()` | GH Actions errors | `logger.github_error("Build failed")` | CI/CD workflows |
| `github_group()` | Collapsible sections | `with logger.github_group("Tests"): ...` | CI/CD workflows |

**Core 7 methods are demonstrated in both sample applications.**  
**GitHub Actions methods are available for CI/CD but kept separate for simplicity.**

---

## 📖 API Reference

### Getting a Logger

```python
from _therock_utils.logging_config import get_logger

logger = get_logger(
    name=__name__,                    # Usually __name__ for module name
    component="PackageInstaller",     # Component name (e.g., "BuildSystem")
    operation="install",              # Operation being performed
    **extra_context                   # Additional context fields
)
```

**Example:**
```python
logger = get_logger(__name__, component="PackageInstaller", operation="install")
# All logs from this logger will include component and operation context
```

---

### Standard Log Levels

```python
# DEBUG - Detailed diagnostic information (development)
logger.debug("Configuration step: Setting up paths", extra={"step": "setup"})

# INFO - General informational messages
logger.info("Package installed successfully", extra={"package": "rocm-core"})

# WARNING - Warning messages for potentially problematic situations
logger.warning("Deprecated API used in component", extra={"component": "hip-runtime"})

# ERROR - Error conditions that don't halt execution
logger.error("Configuration validation failed", extra={"missing_field": "type"})

# CRITICAL - Critical errors that may halt execution
logger.critical("Insufficient disk space! Installation cannot proceed.", extra={
    "available_space_gb": 5,
    "required_space_gb": 10
})
```

---

### Structured Logging with `extra` Fields

Always use `extra={}` for structured data instead of string formatting:

**❌ BAD (String formatting):**
```python
logger.info(f"Installed rocm-core version 6.2.0 in 1250ms")
```

**✅ GOOD (Structured logging):**
```python
logger.info("Package installed successfully", extra={
    "package_name": "rocm-core",
    "version": "6.2.0",
    "duration_ms": 1250,
    "status": "success"
})
```

**Benefits:**
- ✅ Easy to parse and analyze
- ✅ Can be exported to JSON
- ✅ Queryable in log aggregation tools
- ✅ Type-safe values (not stringified)

---

### Performance Timing

#### Context Manager: `timed_operation()`

```python
with logger.timed_operation("Installing rocm-core"):
    download_package("rocm-core")
    extract_package()
    configure_package()

# Automatically logs:
# DEBUG: "Starting operation: Installing rocm-core"
# INFO: "✅ Completed operation: Installing rocm-core (502.34ms)"
```

**Features:**
- ✅ Automatic start/end logging
- ✅ Duration calculated automatically
- ✅ Duration visible in message AND in `extra={"duration_ms": 502.34}`
- ✅ Works with nested operations

**Example with nesting:**
```python
with logger.timed_operation("Complete Installation"):
    with logger.timed_operation("Download"):
        download()
    with logger.timed_operation("Extract"):
        extract()
    with logger.timed_operation("Configure"):
        configure()

# Logs timing for each operation + total time
```

#### Decorator: `@timed()` (Alternative)

```python
@logger.timed("check_prerequisites")
def check_prerequisites():
    # Function execution is automatically timed
    validate_disk_space()
    check_network()
    return True
```

---

### Exception Handling: `log_exception()`

#### Why Use `log_exception()` Instead of `logger.error()`?

**❌ Problem with `logger.error()` alone:**
```python
try:
    install_package("rocm-core")
except Exception as e:
    logger.error(f"Installation failed: {str(e)}")

# Output: ERROR - Installation failed: Package not found
# ❌ No traceback - can't debug!
# ❌ No exception type
# ❌ No stack context
```

**⚠️ Better but still manual:**
```python
try:
    install_package("rocm-core")
except Exception as e:
    logger.error(f"Installation failed: {str(e)}", exc_info=True)

# ⚠️ Easy to forget exc_info=True
# ⚠️ Inconsistent across codebase
# ⚠️ Verbose
```

**✅ Best: Use `log_exception()`:**
```python
try:
    install_package("rocm-core")
except Exception as e:
    logger.log_exception(e, "Installation failed", extra={
        "package_name": "rocm-core",
        "operation": "install"
    })

# Output:
# ERROR - Installation failed
# Traceback (most recent call last):
#   File "sample.py", line 10, in install_package
#     raise ValueError("Package not found")
# ValueError: Package not found
```

**Advantages:**
- ✅ **Always includes traceback** - Can't forget!
- ✅ **Cleaner API** - Less boilerplate
- ✅ **Consistent** - Same pattern everywhere
- ✅ **Extensible** - Can add error categorization, alerts, etc.
- ✅ **Default message** - If you omit message, uses `"Exception occurred: ValueError"`

---

### Display Dictionaries: `log_dict()`

Pretty-print dictionaries or structured data:

```python
config = {
    "type": "release",
    "target": "linux",
    "optimization": "O3"
}

logger.log_dict(config, message="📋 Build Configuration:")

# Output:
# INFO - 📋 Build Configuration:
# {
#   "type": "release",
#   "target": "linux",
#   "optimization": "O3"
# }
```

**Use cases:**
- Display configuration settings
- Show results summaries
- Debug complex data structures
- Log API responses

**Example from Build System:**
```python
test_results = {
    "passed": 2,
    "failed": 1,
    "total": 3
}
logger.log_dict(test_results, message="📊 Test Results Summary:")
```

---

### GitHub Actions Integration

#### Workflow Annotations

```python
# INFO notice in GitHub Actions UI
logger.github_info("✅ All tests passed")

# WARNING annotation (yellow)
logger.github_warning("Deprecated API used", file="src/api.py", line=42)

# ERROR annotation (red)
logger.github_error("Build failed", file="build.py", line=100)
```

**Features:**
- ✅ Creates annotations in GitHub Actions UI
- ✅ Can specify file and line number for code annotations
- ✅ Automatically detected when running in GitHub Actions
- ✅ Falls back to normal logging outside GitHub Actions

#### Collapsible Groups

```python
with logger.github_group("📦 Installing 3 packages"):
    for package in packages:
        install_package(package)

# Creates a collapsible group in GitHub Actions logs
# Keeps logs organized and readable
```

**Use cases:**
- Group related operations
- Organize long log outputs
- Make CI/CD logs more readable
- Hide verbose details by default

**Example from Package Installer:**
```python
with logger.github_group("🔍 Package Verification"):
    for package in packages:
        with logger.timed_operation(f"Verify {package}"):
            verify_package(package)
```

---

## 🔧 Configuration

### Default Configuration

The framework auto-configures on import with sensible defaults:
- **Log Level:** DEBUG (to see timed_operation start messages)
- **Format:** Detailed with timestamps
- **Output:** Console (stdout)
- **Colors:** Enabled (if terminal supports ANSI)
- **GitHub Actions:** Auto-detected from environment

### Custom Configuration

```python
from _therock_utils.logging_config import configure_root_logger
import logging

configure_root_logger(
    level=logging.DEBUG,              # Show DEBUG messages
    format_style=None,                # Use default format
    log_file="logs/app.log",          # Optional file output
    json_output=False,                # False = human-readable
    use_colors=True,                  # ANSI colors in terminal
    enable_github_actions=None        # Auto-detect from environment
)
```

### Environment-Specific Configuration

```python
import os
import logging

is_ci = bool(os.getenv("CI"))

configure_root_logger(
    level=logging.INFO if is_ci else logging.DEBUG,
    json_output=is_ci,      # JSON in CI for log aggregation
    use_colors=not is_ci    # No colors in CI
)
```

---

## 📊 Log Formats

### Console Output (Default)

```
2025-12-12 18:32:22,264 - therock.sample_package_installer - INFO - Package installed successfully
2025-12-12 18:32:22,264 - therock.sample_package_installer - DEBUG - Configuration step: Checking dependencies
2025-12-12 18:32:22,265 - therock.sample_package_installer - INFO - ✅ Completed operation: Installing rocm-core (502.34ms)
```

**Format:** `timestamp - logger_name - level - message`

### With Structured Data

The `extra={}` fields are available for structured logging but not shown in console by default:

```python
logger.info("Package installed", extra={
    "package_name": "rocm-core",
    "version": "6.2.0",
    "duration_ms": 502
})
```

**Console:** `INFO - Package installed`  
**Structured:** Fields available for JSON export, log analysis, etc.

### JSON Output (for CI/CD)

```python
configure_root_logger(json_output=True)
```

```json
{
  "timestamp": "2025-12-12T18:32:22.264Z",
  "level": "INFO",
  "logger": "therock.sample_package_installer",
  "message": "Package installed",
  "package_name": "rocm-core",
  "version": "6.2.0",
  "duration_ms": 502,
  "component": "PackageInstaller",
  "operation": "install"
}
```

---

## 📚 Complete Examples

### Sample Applications

Two simple sample applications demonstrate **core logging features**:

#### 1. Package Installer (`sample_package_installer.py`)

Demonstrates:
- ✅ `install_packages()` - info(), timed_operation() for each package
- ✅ `verify_installation()` - timed_operation(), log_exception(), warning()
- ✅ Structured logging with extra={} fields
- ✅ Complete installation workflow with error handling

**Run it:**
```bash
cd build_tools/_therock_utils
python sample_package_installer.py
```

#### 2. Build System (`sample_build_system.py`)

Demonstrates:
- ✅ `configure_build()` - debug(), info() for configuration steps
- ✅ `compile_components()` - timed_operation() for each component
- ✅ `run_tests()` - timed_operation(), log_exception(), warning(), error()
- ✅ Structured logging with extra={} fields
- ✅ Complete build workflow with error handling

**Run it:**
```bash
cd build_tools/_therock_utils
python sample_build_system.py
```

#### Run Both Samples

```bash
cd build_tools/_therock_utils
python run_logging_demos.py
```

**Output includes:**
- All 12 logging methods in action
- GitHub Actions integration (if running in GitHub)
- Performance timing for all operations
- Exception handling with tracebacks
- Structured data summaries
- Collapsible groups

---

## 💡 Best Practices

### ✅ DO This

```python
# 1. Use module-level logger with component context
logger = get_logger(__name__, component="PackageInstaller")

# 2. Use structured logging with extra fields
logger.info("Package installed", extra={
    "package_name": "rocm-core",
    "version": "6.2.0",
    "duration_ms": 502
})

# 3. Use timed_operation for automatic timing
with logger.timed_operation("Installing rocm-core"):
    install_package("rocm-core")

# 4. Use log_exception for all exceptions
try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed", extra={"context": "value"})

# 5. Use log_dict for configuration and results
logger.log_dict(config, message="📋 Configuration:")

# 6. Use github_group to organize related logs
with logger.github_group("📦 Installation"):
    install_all_packages()

# 7. Use appropriate log levels
logger.debug("Detailed diagnostic")      # Development
logger.info("Normal operation")          # Production
logger.warning("Potential issue")        # Needs attention
logger.error("Error occurred")           # Error but recoverable
logger.critical("System failure!")       # Severe, may halt
```

### ❌ DON'T Do This

```python
# 1. Don't use print()
print("Installing package")  # ❌ No structure, no levels

# 2. Don't use string formatting for structured data
logger.info(f"Installed {pkg} version {ver} in {dur}ms")  # ❌ Not queryable

# 3. Don't manually calculate timing
start = time.time()
work()
logger.info(f"Took {time.time() - start}s")  # ❌ Use timed_operation

# 4. Don't manually handle tracebacks
except Exception as e:
    logger.error(str(e))  # ❌ No traceback
    traceback.print_exc()  # ❌ Use log_exception

# 5. Don't forget exc_info parameter
logger.error("Failed", exc_info=True)  # ⚠️ Better: log_exception

# 6. Don't create logger per function
def my_function():
    logger = get_logger(__name__)  # ❌ Create once at module level
```

---

## 🔄 Migration from Old Logging

### Quick Migration Steps

1. **Add import at top of file:**
   ```python
   from _therock_utils.logging_config import get_logger
   ```

2. **Create module-level logger:**
   ```python
   logger = get_logger(__name__, component="MyComponent")
   ```

3. **Replace old patterns:**
   ```python
   # OLD                          # NEW
   print("message")        →      logger.info("message")
   log("message")          →      logger.info("message")
   _log("message")         →      logger.info("message")
   logging.info("msg")     →      logger.info("msg")
   ```

4. **Update exception handling:**
   ```python
   # OLD
   except Exception as e:
       traceback.print_exc()
   
   # NEW
   except Exception as e:
       logger.log_exception(e, "Operation failed")
   ```

5. **Add structured data:**
   ```python
   # OLD
   logger.info(f"Installed {package} version {version}")
   
   # NEW
   logger.info("Package installed", extra={
       "package_name": package,
       "version": version
   })
   ```

---

## 🧪 Testing with Logs

### Capture Logs in Tests

```python
import io
import logging
from _therock_utils.logging_config import get_logger

def test_installation():
    # Capture log output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = get_logger(__name__, component="Test")
    logger.logger.addHandler(handler)
    
    # Run function
    install_package("rocm-core")
    
    # Assert on logs
    output = log_capture.getvalue()
    assert "Package installed" in output
    assert "rocm-core" in output
```

### Mock GitHub Actions Environment

```python
import os

def test_github_actions_logging():
    # Enable GitHub Actions mode
    os.environ["GITHUB_ACTIONS"] = "true"
    
    logger = get_logger(__name__)
    logger.github_info("Test message")
    
    # In GitHub Actions, this outputs: ::notice::Test message
```

---

## 🎓 Learn by Example

### Start Here

1. **Read this README** - You're doing it! ✅
2. **Run the samples:**
   ```bash
   cd build_tools/_therock_utils
   python run_logging_demos.py
   ```
3. **Read the sample code:**
   - `sample_package_installer.py` - Package installation workflow
   - `sample_build_system.py` - Build system workflow
4. **Copy patterns** from samples into your code

### Key Examples from Samples

**Package installation with timing:**
```python
for i, package in enumerate(packages, 1):
    with logger.timed_operation(f"Installing {package}"):
        logger.info(f"Installing package {i}/{len(packages)}: {package}", extra={
            "package_name": package,
            "progress": f"{i}/{len(packages)}"
        })
        # Do installation work...
        logger.info(f"Package {package} installed successfully", extra={
            "package_name": package,
            "status": "success"
        })
# Automatically logs: "✅ Completed operation: Installing rocm-core (502.34ms)"
```

**Exception handling with log_exception:**
```python
try:
    # Simulate verification
    if "rocm" not in package.lower():
        raise ValueError(f"Package {package} not found in system")
except Exception as e:
    logger.log_exception(e, f"❌ Verification failed for {package}", extra={
        "package_name": package,
        "verification": "failed"
    })
    logger.warning(f"Package verification failed: {package}")
```

**Build testing with multiple log levels:**
```python
try:
    logger.info(f"Testing component: {component}", extra={
        "component": component,
        "test_phase": "unit_tests"
    })
    # Run tests...
    if test_failed:
        raise RuntimeError(f"Unit tests failed for {component}")
    logger.info(f"✅ Tests passed for {component}")
except Exception as e:
    logger.log_exception(e, f"❌ Tests failed for {component}")
    logger.error(f"Test failure in {component}: {str(e)}")
    logger.warning(f"Continuing with remaining components despite failure")
```

---

## 🤝 Contributing

### Adding Features

1. Update `logging_config.py` with new functionality
2. Add examples to sample applications
3. Update this README
4. Test in real workflows

### Reporting Issues

- Provide minimal reproduction example
- Include Python version and environment
- Check if already covered in documentation

---

## 📈 Feature Coverage

### ✅ Core Features (Demonstrated in Samples)

- ✅ `debug()` - Both samples (configuration steps, diagnostics)
- ✅ `info()` - Both samples (general logging, success messages)
- ✅ `warning()` - Both samples (non-critical issues, deprecations)
- ✅ `error()` - Build System sample (test failures)
- ✅ `timed_operation()` - Both samples (extensively, all operations)
- ✅ `log_exception()` - Both samples (exception handling with tracebacks)
- ✅ Structured logging (`extra={}`) - Both samples

### 🔧 Available Framework Features (Not in Simple Samples)

- ⚠️ `critical()` - Available for severe failures
- ⚠️ `log_dict()` - Available for displaying structured data
- ⚠️ `github_info()` - Available for CI/CD workflows
- ⚠️ `github_warning()` - Available for CI/CD workflows
- ⚠️ `github_error()` - Available for CI/CD workflows
- ⚠️ `github_group()` - Available for CI/CD workflows
- ⚠️ `@timed()` decorator - Alternative to `timed_operation()` context manager

**Note:** Samples focus on core features for simplicity. All framework methods are fully documented and available for use.

---

## 📝 Quick Reference Card

```python
# Import
from _therock_utils.logging_config import get_logger, configure_root_logger
import logging

# Create logger
logger = get_logger(__name__, component="MyComponent")

# Configure (optional)
configure_root_logger(level=logging.DEBUG)

# Log levels
logger.debug("diagnostic")
logger.info("information")
logger.warning("warning")
logger.error("error")
logger.critical("critical failure")

# Structured logging
logger.info("message", extra={"key": "value"})

# Timing
with logger.timed_operation("operation"):
    do_work()

# Exceptions
try:
    risky()
except Exception as e:
    logger.log_exception(e, "Failed")

# Dictionaries
logger.log_dict(data, message="Results:")

# GitHub Actions
logger.github_info("notice")
logger.github_warning("warning")
logger.github_error("error")
with logger.github_group("title"):
    grouped_work()
```

---

## 🔗 See Also

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Structured Logging Best Practices](https://www.structlog.org/)
- [GitHub Actions Workflow Commands](https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions)

---

## 📝 License

Copyright © Advanced Micro Devices, Inc.  
SPDX-License-Identifier: MIT

---

## 📞 Support

For questions or assistance:
1. Review the sample applications in this directory
2. Check this README for patterns
3. Contact TheRock infrastructure team

---

**Happy Logging! 🪵**
