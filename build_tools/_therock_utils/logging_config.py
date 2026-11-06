#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TheRock Standardized Logging Template
======================================

A unified logging framework for consistent logging across all TheRock components.

Features:
- Centralized configuration
- Multiple output formats (console, file)
- Context-aware logging (component, operation, user)
- Performance/timing tracking
- Structured logging support
- Log level management
- Exception tracking

Usage Examples:
---------------

Basic Usage:
    from _therock_utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting operation")
    logger.error("Operation failed", extra={"error_code": 500})

With Context:
    logger = get_logger(__name__, component="packaging", operation="install")
    logger.info("Installing package", extra={"package": "rocm-core", "version": "6.2.0"})

With Timing:
    with logger.timed_operation("package_installation"):
        install_package()

Exception Handling:
    try:
        risky_operation()
    except Exception as e:
        logger.log_exception(e, "Operation failed")
"""

import logging
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager


# ============================================================================
# Configuration Constants
# ============================================================================

class LogLevel:
    """Standard log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Unified log format for all outputs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Environment detection
IS_CI = bool(os.getenv("CI"))
IS_WINDOWS = sys.platform == "win32"

# Default log directory
DEFAULT_LOG_DIR = Path.cwd() / "logs"


# ============================================================================
# Custom Formatters
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color and not IS_WINDOWS and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            # Add color to level name
            levelname = record.levelname
            color = self.COLORS.get(record.levelno, self.RESET)
            record.levelname = f"{color}{levelname}{self.RESET}"
        
        result = super().format(record)
        
        if self.use_color:
            # Reset level name for other handlers
            record.levelname = logging.getLevelName(record.levelno)
        
        return result


class ContextFilter(logging.Filter):
    """Filter that adds context fields to log records"""
    
    def __init__(self, component: str = None, operation: str = None):
        super().__init__()
        self.component = component or "therock"
        self.operation = operation
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add context fields to record
        if not hasattr(record, "component"):
            record.component = self.component
        if self.operation and not hasattr(record, "operation"):
            record.operation = self.operation
        return True


# ============================================================================
# Enhanced Logger Class
# ============================================================================

class TheRockLogger(logging.LoggerAdapter):
    """
    Enhanced logger with additional functionality for TheRock project
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
        self._timers: Dict[str, float] = {}
    
    def timed(self, operation_name: str):
        """
        Decorator for timing function execution
        
        Usage:
            @logger.timed("package_installation")
            def install_package():
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.timed_operation(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """
        Context manager for timing operations
        
        Usage:
            with logger.timed_operation("database_query"):
                execute_query()
        
        Logs:
            DEBUG: "Starting operation: {operation_name}"
            INFO: "✅ Completed operation: {operation_name} ({duration}ms)"
        """
        start_time = time.time()
        self.debug(f"Starting operation: {operation_name}")
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"✅ Completed operation: {operation_name} ({duration_ms:.2f}ms)",
                extra={"duration_ms": duration_ms, "operation": operation_name}
            )
    
    def log_exception(self, exc: Exception, message: str = None, **kwargs):
        """
        Log an exception with full traceback
        
        Usage:
            try:
                risky_operation()
            except Exception as e:
                logger.log_exception(e, "Operation failed")
        """
        msg = message or f"Exception occurred: {type(exc).__name__}"
        self.exception(msg, exc_info=exc, **kwargs)
    
    def log_dict(self, data: Dict[str, Any], level: int = logging.INFO, message: str = ""):
        """
        Log a dictionary in a readable format
        
        Usage:
            logger.log_dict({"status": "success", "count": 42}, message="Results")
        """
        self.log(level, f"{message}\n{json.dumps(data, indent=2)}")


# ============================================================================
# Logger Configuration
# ============================================================================

_configured_loggers: Dict[str, TheRockLogger] = {}


def configure_root_logger(
    level: int = None,
    log_file: Union[str, Path] = None,
    use_colors: bool = True,
):
    """
    Configure the root logger for TheRock
    
    Parameters:
    -----------
    level : int, optional
        Logging level (default: INFO in CI, DEBUG otherwise)
    log_file : str or Path, optional
        Path to log file (default: None)
    use_colors : bool
        Enable colored console output (default: True)
    """
    root_logger = logging.getLogger("therock")
    
    # Set level
    if level is None:
        level = LogLevel.INFO if IS_CI else LogLevel.DEBUG
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(LOG_FORMAT, use_color=use_colors)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(
    name: str,
    component: str = None,
    operation: str = None,
    **extra_context
) -> TheRockLogger:
    """
    Get or create a logger instance
    
    Parameters:
    -----------
    name : str
        Logger name (typically __name__ from calling module)
    component : str, optional
        Component name (e.g., "packaging", "build", "test")
    operation : str, optional
        Operation name (e.g., "install", "uninstall", "configure")
    **extra_context : dict
        Additional context to include in logs
    
    Returns:
    --------
    TheRockLogger
        Configured logger instance
    
    Example:
    --------
        logger = get_logger(__name__, component="packaging", operation="install")
        logger.info("Installing package", extra={"package_name": "rocm-core"})
    """
    logger_key = f"{name}:{component}:{operation}"
    
    if logger_key not in _configured_loggers:
        # Ensure root logger is configured
        if not logging.getLogger("therock").handlers:
            configure_root_logger()
        
        # Create logger
        base_logger = logging.getLogger(f"therock.{name}")
        
        # Add context filter
        context_filter = ContextFilter(component=component, operation=operation)
        base_logger.addFilter(context_filter)
        
        # Create enhanced logger
        extra = extra_context.copy()
        if component:
            extra["component"] = component
        if operation:
            extra["operation"] = operation
        
        enhanced_logger = TheRockLogger(base_logger, extra=extra)
        _configured_loggers[logger_key] = enhanced_logger
    
    return _configured_loggers[logger_key]


# ============================================================================
# Utility Functions
# ============================================================================

def set_log_level(level: Union[int, str]):
    """
    Change the log level for all TheRock loggers
    
    Parameters:
    -----------
    level : int or str
        Log level (e.g., logging.DEBUG, "DEBUG", "INFO")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    root_logger = logging.getLogger("therock")
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)


def get_log_file_path(component: str, timestamp: bool = True) -> Path:
    """
    Generate a log file path
    
    Parameters:
    -----------
    component : str
        Component name for the log file
    timestamp : bool
        Include timestamp in filename (default: True)
    
    Returns:
    --------
    Path
        Path to log file
    """
    log_dir = DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{component}_{ts}.log"
    else:
        filename = f"{component}.log"
    
    return log_dir / filename


# ============================================================================
# Initialization
# ============================================================================

# Auto-configure on import if not already configured
if not logging.getLogger("therock").handlers:
    configure_root_logger()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    logger = get_logger(__name__, component="example")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error", extra={"error_code": 500})
    
    # Example 2: Timing operations
    with logger.timed_operation("slow_operation"):
        time.sleep(0.1)
    
    # Example 3: Structured logging
    logger.log_dict({"status": "success", "items_processed": 42}, message="Results")
    
    # Example 4: Exception logging
    try:
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.log_exception(e, "Operation failed")

