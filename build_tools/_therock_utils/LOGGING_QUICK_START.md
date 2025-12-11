# TheRock Logging - Quick Start Card

**⚡ Copy-paste ready code snippets for immediate use**

---

## 🚀 Get Started in 30 Seconds

```python
# 1. Import
from _therock_utils.logging_config import get_logger

# 2. Create logger
logger = get_logger(__name__, component="mycomponent")

# 3. Use it!
logger.info("Hello, TheRock!")
```

---

## 📝 Common Patterns

### Basic Logging
```python
logger.debug("Detailed diagnostic information")
logger.info("General information")
logger.warning("Something unexpected")
logger.error("An error occurred")
logger.critical("System failure!")
```

### With Data
```python
logger.info("User logged in", extra={
    "user_id": "abc123",
    "ip_address": "192.168.1.1",
    "timestamp": datetime.now()
})
```

### Timing
```python
with logger.timed_operation("database_query"):
    results = db.query(sql)
```

### Exceptions
```python
try:
    risky_operation()
except Exception as e:
    logger.log_exception(e, "Operation failed")
```

### GitHub Actions
```python
logger.github_info("Build started")
logger.github_warning("Deprecated API detected")
logger.github_error("Build failed")
```

---

## 🔄 Migration Cheat Sheet

| Old | New |
|-----|-----|
| `print("msg")` | `logger.info("msg")` |
| `log("msg")` | `logger.info("msg")` |
| `_log("msg")` | `logger.info("msg")` |
| `print(f"User: {user}")` | `logger.info("User action", extra={"user": user})` |
| `traceback.print_exc()` | `logger.log_exception(e)` |
| `time.time()` calculations | `with logger.timed_operation():` |

---

## ⚙️ Configuration (Optional)

```python
from _therock_utils.logging_config import (
    configure_root_logger, 
    LogLevel, 
    LogFormat
)

configure_root_logger(
    level=LogLevel.DEBUG,
    format_style=LogFormat.DETAILED,
    log_file="logs/app.log"
)
```

---

## 🎯 Real-World Example

```python
from _therock_utils.logging_config import get_logger

logger = get_logger(__name__, component="packaging", operation="install")

def install_package(pkg_name, version):
    logger.info("Starting installation", extra={
        "package": pkg_name,
        "version": version
    })
    
    try:
        with logger.timed_operation(f"install_{pkg_name}"):
            # Do installation
            result = subprocess.run([...])
            
            if result.returncode != 0:
                logger.error("Installation failed", extra={
                    "package": pkg_name,
                    "exit_code": result.returncode
                })
            else:
                logger.info("Installation successful", extra={
                    "package": pkg_name
                })
    except Exception as e:
        logger.log_exception(e, "Unexpected error during installation")
```

---

## 📚 Learn More

- **Full Guide:** `LOGGING_README.md`
- **Migration:** `LOGGING_MIGRATION_GUIDE.md`
- **Examples:** `logging_examples.py`
- **Demo:** `logging_demo_migration.py`

---

## 💡 Tips

1. **Always use `__name__`** for the logger name
2. **Add component context** for better organization
3. **Use `extra={}` dict** for structured data
4. **Don't use string formatting** - use extra dict instead
5. **Let the framework handle** timing, exceptions, and formatting

---

## ✅ You're Ready!

That's all you need to know to start using the framework. See the full documentation for advanced features.

**Happy logging!** 🪵

