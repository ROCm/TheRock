# TheRock Logging Framework - Migration Guide

## Overview

This guide explains how to migrate from various existing logging patterns in TheRock to the standardized logging framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Migration Patterns](#migration-patterns)
3. [Configuration](#configuration)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

The logging framework is available in `build_tools/_therock_utils/logging_config.py`. No additional dependencies are required.

### Basic Usage

```python
from _therock_utils.logging_config import get_logger

# Get a logger instance
logger = get_logger(__name__, component="mycomponent")

# Use it
logger.info("Operation started")
logger.error("Something went wrong", extra={"error_code": 500})
```

---

## Migration Patterns

### Pattern 1: Simple print() statements

**Before:**
```python
print("Starting operation...")
print(f"Processing {count} items")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Starting operation...")
logger.info(f"Processing items", extra={"count": count})
```

---

### Pattern 2: print() with sys.stdout.flush()

**Before:**
```python
import sys

def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

log("Operation completed")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Operation completed")
# flush() is handled automatically
```

---

### Pattern 3: Custom _log() function

**Before:**
```python
def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

_log("Build started")
_log(f"Artifacts: {artifacts}")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="build")
logger.info("Build started")
logger.info("Artifacts generated", extra={"artifacts": artifacts})
```

---

### Pattern 4: Module-level logging.getLogger()

**Before:**
```python
import logging

logger = logging.getLogger("rocm_installer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("Starting installation")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="installer")
logger.info("Starting installation")
# Configuration is handled centrally
```

---

### Pattern 5: GitHub Actions _log()

**Before:**
```python
def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

if os.getenv("GITHUB_ACTIONS"):
    print(f"::warning::{message}")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.github_warning(message)
# GitHub Actions format is automatic
```

---

### Pattern 6: Exception logging

**Before:**
```python
try:
    risky_operation()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed")
    # Traceback is included automatically
```

---

### Pattern 7: Timing operations

**Before:**
```python
import time

start = time.time()
do_work()
duration = time.time() - start
print(f"Operation took {duration:.2f} seconds")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)

with logger.timed_operation("work"):
    do_work()
# Duration is logged automatically in milliseconds
```

---

### Pattern 8: Conditional debug logging

**Before:**
```python
DEBUG = True

if DEBUG:
    print(f"Debug: variable x = {x}")
```

**After:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.debug(f"Variable x = {x}")
# Control via log level, not boolean flag
```

---

## Configuration

### Basic Configuration

```python
from _therock_utils.logging_config import configure_root_logger, LogLevel, LogFormat

# Configure at application startup
configure_root_logger(
    level=LogLevel.INFO,
    format_style=LogFormat.DETAILED,
    log_file="logs/app.log",
    use_colors=True
)
```

### CI/CD Configuration

```python
import os
from _therock_utils.logging_config import configure_root_logger, LogLevel, LogFormat

is_ci = bool(os.getenv("CI"))

configure_root_logger(
    level=LogLevel.INFO,
    format_style=LogFormat.CI if is_ci else LogFormat.DETAILED,
    enable_github_actions=True,
    use_colors=not is_ci
)
```

### JSON Logging Configuration

For log aggregation systems (e.g., ELK, Splunk):

```python
configure_root_logger(
    level=LogLevel.INFO,
    json_output=True,
    log_file="logs/app.json"
)
```

---

## Best Practices

### 1. Use Module-Level Logger

```python
# At top of file
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="packaging")

# Use throughout the file
def install_package(name):
    logger.info(f"Installing package", extra={"package": name})
```

### 2. Include Context in Extra Fields

**Good:**
```python
logger.info("Package installed", extra={
    "package": "rocm-core",
    "version": "6.2.0",
    "duration_ms": 1250
})
```

**Not Recommended:**
```python
logger.info(f"Package rocm-core 6.2.0 installed in 1250ms")
# Harder to parse and query
```

### 3. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about program flow
- **WARNING**: Unexpected but recoverable situations
- **ERROR**: Errors that prevent specific operations
- **CRITICAL**: System-level failures

### 4. Log Exceptions Properly

```python
try:
    operation()
except SpecificError as e:
    logger.log_exception(e, "Specific operation failed")
    # Includes full traceback
except Exception as e:
    logger.log_exception(e, "Unexpected error")
    raise  # Re-raise if fatal
```

### 5. Use Structured Logging

```python
# Good - structured
logger.info("Build completed", extra={
    "artifacts": 42,
    "warnings": 3,
    "errors": 0,
    "duration_seconds": 125
})

# Not ideal - unstructured
logger.info("Build completed: 42 artifacts, 3 warnings, 0 errors, 125 seconds")
```

### 6. Component Naming Convention

Use consistent component names across the codebase:

- `packaging` - Package management
- `build` - Build operations
- `testing` - Test execution
- `ci` - CI/CD operations
- `deployment` - Deployment operations

### 7. GitHub Actions Integration

```python
# Use specialized methods for GitHub Actions
logger.github_warning("Deprecated API", file="src/api.py", line=42)

# Use groups for logical sections
with logger.github_group("Test Results"):
    for test in tests:
        logger.info(f"Test {test.name}: {test.result}")
```

---

## File-by-File Migration Checklist

For each file you migrate:

- [ ] Import the logging framework: `from _therock_utils.logging_config import get_logger`
- [ ] Create module-level logger: `logger = get_logger(__name__, component="...")`
- [ ] Replace all `print()` with `logger.info()` or appropriate level
- [ ] Replace all custom `log()` / `_log()` functions
- [ ] Update exception handling to use `logger.log_exception()`
- [ ] Add timing to performance-critical operations
- [ ] Add structured data via `extra={}` parameter
- [ ] Remove manual flush() calls
- [ ] Remove manual logger configuration
- [ ] Update GitHub Actions integration
- [ ] Test logging output

---

## Common Issues

### Issue: Duplicate log messages

**Cause:** Multiple handlers or propagation enabled

**Solution:**
```python
# Root logger is already configured, don't add more handlers
# Just use get_logger()
logger = get_logger(__name__)
```

### Issue: Logs not appearing

**Cause:** Log level too high

**Solution:**
```python
from _therock_utils.logging_config import set_log_level, LogLevel

set_log_level(LogLevel.DEBUG)
```

### Issue: Colors not working in CI

**Cause:** TTY detection

**Solution:**
```python
configure_root_logger(use_colors=False)  # In CI environments
```

### Issue: JSON parsing errors

**Cause:** Mixed output formats

**Solution:**
```python
# Use consistent format
configure_root_logger(json_output=True)
# OR
configure_root_logger(format_style=LogFormat.DETAILED)
```

---

## Backward Compatibility

If you need to support both old and new logging temporarily:

```python
try:
    from _therock_utils.logging_config import get_logger
    logger = get_logger(__name__)
    USE_NEW_LOGGING = True
except ImportError:
    import sys
    def log(*args):
        print(*args)
        sys.stdout.flush()
    USE_NEW_LOGGING = False

# Then use:
if USE_NEW_LOGGING:
    logger.info("message")
else:
    log("message")
```

---

## Testing

Test your logging changes:

```python
import io
import logging
from _therock_utils.logging_config import get_logger, configure_root_logger

def test_logging():
    # Capture log output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = get_logger(__name__)
    logger.logger.addHandler(handler)
    
    logger.info("Test message")
    
    output = log_capture.getvalue()
    assert "Test message" in output
```

---

## Support

For questions or issues:
1. Review this guide
2. Check `logging_examples.py` for patterns
3. Contact the TheRock infrastructure team

---

## Appendix: Complete Example

**Old Style (packaging_utils.py):**
```python
import logging

logger = logging.getLogger("rocm_installer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

def install_package(pkg_name):
    logger.info(f"Installing {pkg_name}")
    try:
        result = subprocess.run(["dpkg", "-i", pkg_name])
        if result.returncode != 0:
            logger.error(f"Failed to install {pkg_name}")
    except Exception as e:
        logger.error(f"Exception: {e}")
```

**New Style:**
```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="packaging", operation="install")

def install_package(pkg_name):
    logger.info("Installing package", extra={"package": pkg_name})
    
    try:
        with logger.timed_operation(f"install_{pkg_name}"):
            result = subprocess.run(["dpkg", "-i", pkg_name])
            
            if result.returncode != 0:
                logger.error(
                    "Package installation failed",
                    extra={
                        "package": pkg_name,
                        "exit_code": result.returncode
                    }
                )
            else:
                logger.info(
                    "Package installed successfully",
                    extra={"package": pkg_name}
                )
    except Exception as e:
        logger.log_exception(e, f"Failed to install {pkg_name}")
```

---

**End of Migration Guide**

