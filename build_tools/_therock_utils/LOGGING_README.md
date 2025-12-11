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
- ✅ **Poor Exception Tracking** - Inconsistent exception logging

---

## 🚀 Quick Start

### 1. Basic Usage

```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Hello, TheRock!")
```

### 2. With Component Context

```python
logger = get_logger(__name__, component="packaging", operation="install")
logger.info("Installing package", extra={"package": "rocm-core", "version": "6.2.0"})
```

### 3. GitHub Actions Integration

```python
logger.github_info("Build started")
logger.github_warning("Deprecated API detected", file="src/api.py", line=42)
logger.github_error("Build failed")
```

### 4. Performance Timing

```python
with logger.timed_operation("database_query"):
    results = db.execute(query)
# Automatically logs duration in milliseconds
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `logging_config.py` | Core logging framework (use this!) |
| `logging_examples.py` | 12 comprehensive usage examples |
| `logging_demo_migration.py` | Before/after migration demonstration |
| `LOGGING_MIGRATION_GUIDE.md` | Complete migration guide |
| `LOGGING_README.md` | This file |

---

## ✨ Features

### Core Features

- **Zero Configuration** - Works out of the box
- **Structured Logging** - JSON-serializable data via `extra={}`
- **Multiple Formats** - Console, file, JSON output
- **Log Levels** - DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Thread-Safe** - Safe for concurrent operations
- **Context-Aware** - Component and operation tracking

### Advanced Features

- **Performance Timing** - Built-in operation timing
- **Exception Tracking** - Automatic traceback capture
- **GitHub Actions** - Native workflow command support
- **Colored Output** - ANSI colors for terminal
- **Log Rotation** - Automatic file management
- **Dynamic Levels** - Change log level at runtime

---

## 📖 Documentation

### Essential Reading

1. **Start Here:** [Quick Start](#quick-start) (above)
2. **Learn Patterns:** `logging_examples.py` - 12 practical examples
3. **Migration:** `LOGGING_MIGRATION_GUIDE.md` - Step-by-step guide
4. **Demo:** `logging_demo_migration.py` - See before/after

### API Reference

#### Getting a Logger

```python
get_logger(
    name: str,                    # Usually __name__
    component: str = None,        # e.g., "packaging", "build"
    operation: str = None,        # e.g., "install", "compile"
    **extra_context             # Additional context fields
) -> TheRockLogger
```

#### Log Levels

```python
logger.debug("Detailed diagnostic info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

#### Structured Logging

```python
logger.info("Event occurred", extra={
    "user": "john_doe",
    "action": "package_install",
    "duration_ms": 1250,
    "success": True
})
```

#### Timing Operations

```python
# Context manager
with logger.timed_operation("operation_name"):
    do_work()

# Decorator
@logger.timed("function_name")
def my_function():
    do_work()
```

#### Exception Handling

```python
try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed")
    # Includes full traceback automatically
```

#### GitHub Actions

```python
logger.github_info("Build started")
logger.github_warning("Warning", file="src/main.py", line=42)
logger.github_error("Error", file="src/main.py", line=100)

with logger.github_group("Test Results"):
    run_tests()
```

---

## 🔧 Configuration

### Default Configuration

The framework auto-configures on import with sensible defaults:
- **Log Level:** INFO in CI, DEBUG otherwise
- **Format:** CI format in CI, DETAILED otherwise
- **Output:** Console (stdout)
- **Colors:** Enabled (if terminal supports it)

### Custom Configuration

```python
from _therock_utils.logging_config import configure_root_logger, LogLevel, LogFormat

configure_root_logger(
    level=LogLevel.DEBUG,
    format_style=LogFormat.DETAILED,
    log_file="logs/app.log",
    json_output=False,
    use_colors=True,
    enable_github_actions=True
)
```

### Environment-Specific Configuration

```python
import os

is_ci = bool(os.getenv("CI"))

configure_root_logger(
    level=LogLevel.INFO,
    format_style=LogFormat.CI if is_ci else LogFormat.DETAILED,
    json_output=is_ci,  # JSON in CI for log aggregation
    use_colors=not is_ci
)
```

---

## 📊 Log Formats

### SIMPLE
```
INFO - Starting operation
```

### DETAILED (Default)
```
2025-12-11 10:30:45,123 - myapp - INFO - Starting operation
```

### CI (GitHub Actions)
```
2025-12-11 10:30:45,123 [INFO] [packaging] Starting operation
```

### DIAGNOSTIC (File logs)
```
2025-12-11 10:30:45,123 - myapp - INFO - [main.py:42] - main() - Starting operation
```

### JSON (Structured)
```json
{
  "timestamp": "2025-12-11T10:30:45.123Z",
  "level": "INFO",
  "logger": "myapp",
  "message": "Starting operation",
  "module": "main",
  "function": "main",
  "line": 42,
  "component": "packaging",
  "extra_field": "value"
}
```

---

## 🔄 Migration Guide

### Step-by-Step Process

1. **Add Import**
   ```python
   from _therock_utils.logging_config import get_logger
   ```

2. **Create Logger**
   ```python
   logger = get_logger(__name__, component="mycomponent")
   ```

3. **Replace Old Logging**
   - `print()` → `logger.info()`
   - `log()` → `logger.info()`
   - `_log()` → `logger.info()`

4. **Update Exceptions**
   - `traceback.print_exc()` → `logger.log_exception(e)`

5. **Remove Manual Config**
   - Delete `logging.getLogger()` setup code

6. **Add Structure**
   - Add `extra={}` for structured data

See `LOGGING_MIGRATION_GUIDE.md` for complete details.

---

## 💡 Best Practices

### ✅ Do This

```python
# Use module-level logger
logger = get_logger(__name__, component="packaging")

# Use structured logging
logger.info("Package installed", extra={
    "package": "rocm-core",
    "version": "6.2.0",
    "duration_ms": 1250
})

# Use timing for performance
with logger.timed_operation("install"):
    install_package()

# Use proper exception logging
try:
    operation()
except Exception as e:
    logger.log_exception(e, "Context")
```

### ❌ Don't Do This

```python
# Don't use direct print
print("Installing package")

# Don't use string formatting for data
logger.info(f"Installed rocm-core 6.2.0 in 1250ms")

# Don't manually calculate timing
start = time.time()
work()
print(f"Took {time.time() - start}s")

# Don't manually format tracebacks
except Exception as e:
    traceback.print_exc()
```

---

## 🧪 Testing

### Unit Testing Logs

```python
import io
import logging

def test_my_function():
    # Capture logs
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    
    logger = get_logger(__name__)
    logger.logger.addHandler(handler)
    
    # Run function
    my_function()
    
    # Assert logs
    output = log_capture.getvalue()
    assert "Expected message" in output
```

---

## 🎓 Examples

Run the examples to see the framework in action:

```bash
cd build_tools/_therock_utils
python logging_examples.py
```

This will demonstrate:
- Basic logging patterns
- Component-specific logging
- Performance timing
- Exception handling
- Structured logging
- GitHub Actions integration
- File logging
- JSON output
- Multi-threading

---

## 🤝 Contributing

### Adding New Features

1. Add functionality to `logging_config.py`
2. Add examples to `logging_examples.py`
3. Update documentation
4. Test with existing code

### Reporting Issues

- Check if issue is already in migration guide
- Provide minimal reproduction example
- Include Python version and environment

---

## 📈 Adoption Status

Track migration progress across TheRock:

- [ ] `build_tools/packaging/` - **IN PROGRESS** ✓ (installer/uninstaller done)
- [ ] `build_tools/github_actions/`
- [ ] `build_tools/_therock_utils/`
- [ ] `build_tools/` (root scripts)
- [ ] CI/CD workflows

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
1. Review the examples and migration guide
2. Check existing code for patterns
3. Contact TheRock infrastructure team

---

**Happy Logging! 🪵**

