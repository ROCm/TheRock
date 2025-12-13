# TheRock Logging Framework - Presentation Guide

**Version:** 1.0.0  
**Date:** December 2025  
**Purpose:** Comprehensive guide for presenting the logging framework to teams

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Standard Log Levels (5 Methods)](#standard-log-levels)
3. [Performance Tracking (2 Methods)](#performance-tracking)
4. [Exception Handling (1 Method)](#exception-handling)
5. [Structured Data (1 Method)](#structured-data)
6. [GitHub Actions Integration (4 Methods)](#github-actions-integration)
7. [Configuration (1 Function)](#configuration)
8. [Quick Reference](#quick-reference)

---

## Overview

### What is TheRock Logging Framework?

A **unified, production-grade logging system** that provides:
- ✅ **Consistency** - Same API across all components
- ✅ **Performance** - Built-in timing and metrics
- ✅ **Debugging** - Automatic tracebacks and structured data
- ✅ **CI/CD** - Native GitHub Actions support
- ✅ **Zero Config** - Works out of the box

### Total Methods Available: **13**

| Category | Count | Methods |
|----------|-------|---------|
| Standard Logging | 5 | debug, info, warning, error, critical |
| Performance | 2 | timed_operation, @timed decorator |
| Exception Handling | 1 | log_exception |
| Structured Data | 1 | log_dict |
| GitHub Actions | 4 | github_info, github_warning, github_error, github_group |

---

## Standard Log Levels

### Overview
These are the 5 standard Python logging levels, available through the framework.

---

### 1. `debug()` - Detailed Diagnostic Information

**Purpose:** For development and troubleshooting  
**When to Use:** Detailed diagnostic information that helps developers understand what's happening internally  
**Log Level:** DEBUG (10)

#### Example:
```python
logger.debug("Configuration step: Checking dependencies", extra={"step": "validation"})
logger.debug("Database query: SELECT * FROM users WHERE id=123")
logger.debug("Cache hit for key 'user:123'")
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - DEBUG - Configuration step: Checking dependencies
```

#### Best Practices:
- ✅ Use for internal state tracking
- ✅ Use for variable values during development
- ✅ Include helpful context in extra={}
- ❌ Don't overuse - too much debug logging slows performance

---

### 2. `info()` - General Information

**Purpose:** Normal operation messages  
**When to Use:** Confirming that things are working as expected  
**Log Level:** INFO (20)

#### Example:
```python
logger.info("Application started successfully")
logger.info("Package installed", extra={"package": "rocm-core", "version": "6.2.0"})
logger.info("Processing 150 items", extra={"total": 150, "batch_size": 50})
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - INFO - Package installed
```

#### Best Practices:
- ✅ Use for milestones and progress
- ✅ Use for successful operations
- ✅ Add structured data with extra={}
- ❌ Don't log inside tight loops

---

### 3. `warning()` - Non-Critical Issues

**Purpose:** Something unexpected happened, but not an error  
**When to Use:** Deprecated APIs, recoverable errors, potential problems  
**Log Level:** WARNING (30)

#### Example:
```python
logger.warning("Deprecated API used", extra={"api": "old_function", "replacement": "new_function"})
logger.warning("Slow response time detected", extra={"duration_ms": 5000, "threshold_ms": 1000})
logger.warning("Retrying failed operation (attempt 2/3)")
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - WARNING - Deprecated API used
```

#### Best Practices:
- ✅ Use for deprecation notices
- ✅ Use for performance degradation
- ✅ Use for automatic recovery scenarios
- ❌ Don't use for expected behavior

---

### 4. `error()` - Error Conditions

**Purpose:** An error occurred but the application can continue  
**When to Use:** Recoverable errors, validation failures, operation failures  
**Log Level:** ERROR (40)

#### Example:
```python
logger.error("Configuration validation failed", extra={"missing_field": "database_url"})
logger.error("Failed to send email", extra={"recipient": "user@example.com", "error": "Connection timeout"})
logger.error("Test suite completed with failures", extra={"passed": 45, "failed": 5})
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - ERROR - Configuration validation failed
```

#### Best Practices:
- ✅ Use for recoverable errors
- ✅ Include error context
- ✅ Use log_exception() for exceptions with tracebacks
- ❌ Don't use for expected validation failures (use warning)

---

### 5. `critical()` - Severe Failures

**Purpose:** Critical system failures that may halt execution  
**When to Use:** Unrecoverable errors, system failures, data corruption  
**Log Level:** CRITICAL (50)

#### Example:
```python
logger.critical("Insufficient disk space! Application cannot continue.", extra={
    "available_gb": 0.5,
    "required_gb": 10
})
logger.critical("Database connection lost and all retry attempts failed")
logger.critical("Critical component 'core' failed initialization")
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - CRITICAL - Insufficient disk space! Application cannot continue.
```

#### Best Practices:
- ✅ Use sparingly - only for truly critical issues
- ✅ Include what failed and why
- ✅ Often followed by sys.exit() or raise
- ❌ Don't overuse - dilutes importance

---

## Performance Tracking

### Overview
Built-in methods for tracking operation duration without manual timing calculations.

---

### 6. `timed_operation()` - Context Manager for Automatic Timing

**Purpose:** Automatically track how long an operation takes  
**When to Use:** Wrapping any operation you want to measure  
**Type:** Context Manager (use with `with` statement)

#### How It Works:
1. **Logs START** (DEBUG): `"Starting operation: {name}"`
2. **Executes** your code
3. **Logs COMPLETION** (INFO): `"✅ Completed operation: {name} (502.34ms)"`

#### Example:
```python
with logger.timed_operation("Installing rocm-core"):
    download_package("rocm-core")
    extract_package()
    configure_package()

# Automatically logs:
# DEBUG: Starting operation: Installing rocm-core
# INFO: ✅ Completed operation: Installing rocm-core (1523.45ms)
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - DEBUG - Starting operation: Installing rocm-core
2025-12-12 10:30:46,646 - therock.myapp - INFO - ✅ Completed operation: Installing rocm-core (1523.45ms)
```

#### Why It's Better Than Manual Timing:
```python
# ❌ Manual (old way)
start = time.time()
do_work()
duration = (time.time() - start) * 1000
logger.info(f"Work completed in {duration}ms")

# ✅ Automatic (new way)
with logger.timed_operation("do_work"):
    do_work()
```

#### Best Practices:
- ✅ Use for database queries
- ✅ Use for API calls
- ✅ Use for file operations
- ✅ Use for batch processing
- ✅ Can be nested for detailed timing

#### Nested Example:
```python
with logger.timed_operation("Complete Build"):
    with logger.timed_operation("Compile"):
        compile()
    with logger.timed_operation("Test"):
        test()
# Logs timing for each operation + total time
```

---

### 7. `@timed()` - Decorator for Function Timing

**Purpose:** Time entire function execution  
**When to Use:** When you want to time every call to a function  
**Type:** Decorator

#### Example:
```python
@logger.timed("database_query")
def fetch_user_data(user_id):
    return db.query(f"SELECT * FROM users WHERE id={user_id}")

# Every call automatically timed:
user = fetch_user_data(123)
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - DEBUG - Starting operation: database_query
2025-12-12 10:30:45,345 - therock.myapp - INFO - ✅ Completed operation: database_query (222.15ms)
```

#### Best Practices:
- ✅ Use on functions called multiple times
- ✅ Use for API endpoints
- ✅ Use for critical performance paths
- ❌ Don't overuse - adds overhead

---

## Exception Handling

### 8. `log_exception()` - Unified Exception Logging

**Purpose:** Log exceptions with full traceback automatically  
**When to Use:** In every `except` block  
**Why It's Better:** Ensures consistent exception logging with tracebacks

#### The Problem with Normal Logging:
```python
# ❌ Option 1: No traceback
try:
    install_package()
except Exception as e:
    logger.error(f"Installation failed: {str(e)}")
# Output: ERROR - Installation failed: Package not found
# Problem: No traceback! Can't debug where error originated

# ❌ Option 2: Need to remember exc_info=True
try:
    install_package()
except Exception as e:
    logger.error(f"Installation failed: {str(e)}", exc_info=True)
# Problem: Easy to forget exc_info=True, inconsistent
```

#### The Solution: `log_exception()`
```python
# ✅ Always includes traceback, consistent, simple
try:
    install_package("rocm-core")
except Exception as e:
    logger.log_exception(e, "Installation failed", extra={
        "package": "rocm-core",
        "operation": "install"
    })
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - ERROR - Installation failed
Traceback (most recent call last):
  File "installer.py", line 45, in install_package
    download_file(url)
  File "installer.py", line 78, in download_file
    raise ConnectionError("Network timeout")
ConnectionError: Network timeout
```

#### Advantages:
1. ✅ **Always includes traceback** - Can't forget!
2. ✅ **Consistent** - Same pattern everywhere
3. ✅ **Cleaner API** - Less boilerplate
4. ✅ **Structured data** - Can add extra context
5. ✅ **Default message** - If you forget message, uses exception type

#### Best Practices:
```python
# ✅ Good: Specific message with context
try:
    verify_package(pkg)
except Exception as e:
    logger.log_exception(e, f"Verification failed for {pkg}", extra={
        "package": pkg,
        "phase": "verification"
    })

# ✅ Also good: No message (uses exception type)
try:
    something()
except ValueError as e:
    logger.log_exception(e)  # Logs: "Exception occurred: ValueError"
```

---

## Structured Data

### 9. `log_dict()` - Display Dictionaries in Readable Format

**Purpose:** Pretty-print dictionaries in logs  
**When to Use:** Displaying configuration, results, summaries  
**Format:** JSON with 2-space indentation

#### Example:
```python
config = {
    "type": "release",
    "target": "linux",
    "optimization": "O3",
    "features": ["rocm", "cuda"]
}

logger.log_dict(config, message="📋 Build Configuration:")
```

#### Output:
```
2025-12-12 10:30:45,123 - therock.myapp - INFO - 📋 Build Configuration:
{
  "type": "release",
  "target": "linux",
  "optimization": "O3",
  "features": [
    "rocm",
    "cuda"
  ]
}
```

#### Use Cases:
```python
# Configuration display
logger.log_dict(app_config, message="Application Config:")

# Test results summary
test_results = {"passed": 45, "failed": 5, "skipped": 2}
logger.log_dict(test_results, message="📊 Test Results:")

# API response debugging
logger.log_dict(api_response, level=logging.DEBUG, message="API Response:")

# Prerequisites check
prereqs = {"python": "3.12", "disk_gb": 50, "memory_gb": 16}
logger.log_dict(prereqs, message="System Prerequisites:")
```

#### Best Practices:
- ✅ Use for configuration dumps
- ✅ Use for result summaries
- ✅ Use for debugging complex objects
- ❌ Don't use for huge dictionaries (logs become unreadable)
- ❌ Don't use in tight loops

---

## GitHub Actions Integration

### Overview
Special methods that create GitHub Actions workflow annotations and collapsible groups.

**When to Use:** Only in CI/CD pipelines running in GitHub Actions  
**Local Behavior:** Falls back to normal logging when not in GitHub Actions  
**Auto-Detection:** Framework detects `GITHUB_ACTIONS` environment variable

---

### 10. `github_info()` - GitHub Actions Info Annotation

**Purpose:** Create INFO-level annotations in GitHub Actions UI  
**When to Use:** Success messages, milestones in CI/CD  
**GitHub Output:** Blue "notice" annotation in workflow summary

#### Example:
```python
logger.github_info("✅ All tests passed successfully")
logger.github_info("Build artifacts uploaded to S3")
logger.github_info(f"Deployed to production: v{version}")
```

#### Output in GitHub Actions:
```
::notice::✅ All tests passed successfully
```

#### GitHub UI:
- Shows as blue "info" badge in workflow summary
- Appears in annotations list
- Visible without expanding logs

#### Local Output (non-GitHub):
```
2025-12-12 10:30:45,123 - therock.myapp - INFO - ✅ All tests passed successfully
```

---

### 11. `github_warning()` - GitHub Actions Warning Annotation

**Purpose:** Create WARNING annotations with file/line references  
**When to Use:** Non-critical issues, deprecations in CI/CD  
**GitHub Output:** Yellow warning annotation

#### Example:
```python
logger.github_warning("Deprecated API used", file="src/api.py", line=42)
logger.github_warning("Test flakiness detected")
logger.github_warning("Build took longer than expected", file="build.py", line=156)
```

#### Output in GitHub Actions:
```
::warning file=src/api.py,line=42::Deprecated API used
```

#### GitHub UI:
- Shows as yellow warning badge
- Links directly to file and line number
- Visible in Files Changed tab with annotation

---

### 12. `github_error()` - GitHub Actions Error Annotation

**Purpose:** Create ERROR annotations with file/line references  
**When to Use:** Build failures, test failures in CI/CD  
**GitHub Output:** Red error annotation

#### Example:
```python
logger.github_error("Build failed: missing dependency", file="build.py", line=100)
logger.github_error("Tests failed: 5/50")
logger.github_error("Deployment failed: permission denied", file="deploy.sh", line=23)
```

#### Output in GitHub Actions:
```
::error file=build.py,line=100::Build failed: missing dependency
```

#### GitHub UI:
- Shows as red error badge
- Links directly to file and line number
- Blocks PR merge if configured
- Highly visible in workflow summary

---

### 13. `github_group()` - Collapsible Log Groups

**Purpose:** Create collapsible sections in GitHub Actions logs  
**When to Use:** Grouping related logs for better readability  
**Type:** Context Manager

#### Example:
```python
with logger.github_group("📦 Installing 3 packages"):
    install_package("rocm-core")
    install_package("rocm-hip")
    install_package("pytorch")

with logger.github_group("🧪 Running Tests"):
    run_unit_tests()
    run_integration_tests()
```

#### Output in GitHub Actions:
```
::group::📦 Installing 3 packages
  ... package installation logs ...
::endgroup::

::group::🧪 Running Tests
  ... test execution logs ...
::endgroup::
```

#### GitHub UI:
- Creates collapsible/expandable sections
- Collapsed by default (keeps logs clean)
- Click to expand and see details
- Makes long logs much more readable

#### Best Practices:
```python
# ✅ Good: Group related operations
with logger.github_group("Build Phase"):
    compile()
    link()
    package()

# ✅ Good: Nested groups for hierarchy
with logger.github_group("CI Pipeline"):
    with logger.github_group("Build"):
        build()
    with logger.github_group("Test"):
        test()
    with logger.github_group("Deploy"):
        deploy()
```

---

## Configuration

### `configure_root_logger()` - One-Time Setup

**Purpose:** Configure logging once at application startup  
**When to Use:** Beginning of main(), before any logging  
**Default Behavior:** Works without configuration (sensible defaults)

#### Parameters:
```python
configure_root_logger(
    level=logging.DEBUG,          # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_style=None,            # Log format (default: auto-detect)
    log_file="logs/app.log",      # Optional: write to file
    json_output=False,            # True: JSON format, False: human-readable
    use_colors=True,              # ANSI colors in terminal
    enable_github_actions=None    # Auto-detect from environment
)
```

#### Common Configurations:

**1. Development (default):**
```python
configure_root_logger(level=logging.DEBUG)
```

**2. Production:**
```python
configure_root_logger(
    level=logging.INFO,
    log_file="logs/production.log",
    json_output=True  # For log aggregation systems
)
```

**3. CI/CD:**
```python
configure_root_logger(
    level=logging.INFO,
    json_output=True,
    use_colors=False  # No ANSI colors in CI logs
)
```

**4. Debug with file output:**
```python
configure_root_logger(
    level=logging.DEBUG,
    log_file=f"logs/debug_{datetime.now():%Y%m%d_%H%M%S}.log"
)
```

#### When to Call:
```python
# ✅ Call once at startup
def main():
    configure_root_logger(level=logging.DEBUG)
    logger = get_logger(__name__)
    logger.info("Application started")
    # ... rest of application

# ✅ In test setup
@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    configure_root_logger(level=logging.DEBUG)
```

---

## Quick Reference

### Cheat Sheet

```python
from _therock_utils.logging_config import get_logger, configure_root_logger
import logging

# 1. Configure once at startup
configure_root_logger(level=logging.DEBUG)

# 2. Create logger
logger = get_logger(__name__, component="MyComponent")

# 3. Standard logging
logger.debug("Diagnostic info")
logger.info("General message")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")

# 4. Structured logging
logger.info("Event", extra={"key": "value", "count": 42})

# 5. Performance timing
with logger.timed_operation("operation_name"):
    do_work()

# 6. Exception handling
try:
    risky()
except Exception as e:
    logger.log_exception(e, "Operation failed")

# 7. Display dictionary
logger.log_dict(config, message="Config:")

# 8. GitHub Actions (CI/CD only)
logger.github_info("Success message")
logger.github_warning("Warning", file="file.py", line=42)
logger.github_error("Error", file="file.py", line=100)
with logger.github_group("Group Title"):
    grouped_work()
```

---

## Presentation Tips

### Key Points to Emphasize:

1. **✅ Zero Configuration** - Works out of the box
2. **✅ Automatic Timing** - No manual calculations
3. **✅ Foolproof Exceptions** - Can't forget tracebacks
4. **✅ Structured Data** - Better than string formatting
5. **✅ CI/CD Integration** - Native GitHub Actions support

### Demo Flow:

1. **Show OLD code** - print(), manual timing, no tracebacks
2. **Show NEW code** - Unified logging with all features
3. **Run samples** - Live demonstration
4. **Show GitHub Actions** - Workflow annotations

### Common Questions:

**Q: Do I need to configure it?**  
A: No, it works with defaults. Configure only if you need customization.

**Q: What about performance overhead?**  
A: Minimal. DEBUG logs can be disabled in production.

**Q: Can I use it with existing code?**  
A: Yes! Gradually migrate file by file.

**Q: What about GitHub Actions methods outside GitHub?**  
A: They fall back to normal logging automatically.

---

## Summary

### What You Get:

| Feature | Methods | Benefit |
|---------|---------|---------|
| **Standard Logging** | 5 methods | All Python log levels available |
| **Performance** | 2 methods | Automatic timing, no manual work |
| **Exceptions** | 1 method | Foolproof error logging |
| **Structured Data** | 1 method | Better than string formatting |
| **CI/CD** | 4 methods | Native GitHub Actions support |

### Total Value:

- ✅ **13 powerful methods**
- ✅ **1 configuration function**
- ✅ **Zero learning curve** - familiar Python logging
- ✅ **Production ready** - battle-tested patterns
- ✅ **Extensible** - Easy to add new features

---

**Ready to get started?**

```bash
cd build_tools/_therock_utils
python run_logging_demos.py
```

---

**Questions?**

Contact: TheRock Infrastructure Team  
Documentation: `LOGGING_README.md`  

