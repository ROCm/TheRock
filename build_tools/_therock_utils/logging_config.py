#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TheRock Standardized Logging Template
======================================

A unified logging framework for consistent logging across all TheRock components.

Features:
- Centralized configuration
- Multiple output formats (console, file, JSON)
- Context-aware logging (component, operation, user)
- Performance/timing tracking
- GitHub Actions integration
- Structured logging support
- Log level management
- Exception tracking
- Thread-safe operations

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

GitHub Actions:
    logger.github_info("Build completed")
    logger.github_warning("Deprecated API used")
    logger.github_error("Build failed")
"""

import logging
import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import traceback


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


class LogFormat:
    """Predefined log formats"""
    # Simple format for console output
    SIMPLE = "%(levelname)s - %(message)s"
    
    # Detailed format with timestamp and module
    DETAILED = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Format optimized for CI/CD logs
    CI = "%(asctime)s [%(levelname)s] [%(component)s] %(message)s"
    
    # Full diagnostic format
    DIAGNOSTIC = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
    )
    
    # JSON structured format
    JSON = "json"


# Environment detection
IS_CI = bool(os.getenv("CI"))
IS_GITHUB_ACTIONS = bool(os.getenv("GITHUB_ACTIONS"))
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


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "error_code"):
            log_data["error_code"] = record.error_code
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        return json.dumps(log_data)


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
# GitHub Actions Integration
# ============================================================================

class GitHubActionsHandler(logging.Handler):
    """Handler that outputs GitHub Actions workflow commands"""
    
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            
            # Format based on log level
            if record.levelno >= logging.ERROR:
                print(f"::error::{msg}")
            elif record.levelno >= logging.WARNING:
                print(f"::warning::{msg}")
            else:
                print(f"::notice::{msg}")
            
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


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
    
    # GitHub Actions integration
    def github_info(self, message: str, **kwargs):
        """Log an info message in GitHub Actions format"""
        if IS_GITHUB_ACTIONS:
            print(f"::notice::{message}")
            sys.stdout.flush()
        self.info(message, **kwargs)
    
    def github_warning(self, message: str, file: str = None, line: int = None, **kwargs):
        """Log a warning in GitHub Actions format"""
        if IS_GITHUB_ACTIONS:
            annotation = f"::warning"
            if file:
                annotation += f" file={file}"
            if line:
                annotation += f",line={line}"
            annotation += f"::{message}"
            print(annotation)
            sys.stdout.flush()
        self.warning(message, **kwargs)
    
    def github_error(self, message: str, file: str = None, line: int = None, **kwargs):
        """Log an error in GitHub Actions format"""
        if IS_GITHUB_ACTIONS:
            annotation = f"::error"
            if file:
                annotation += f" file={file}"
            if line:
                annotation += f",line={line}"
            annotation += f"::{message}"
            print(annotation)
            sys.stdout.flush()
        self.error(message, **kwargs)
    
    @contextmanager
    def github_group(self, title: str):
        """
        Create a collapsible group in GitHub Actions logs
        
        Usage:
            with logger.github_group("Installation Steps"):
                install_packages()
        """
        if IS_GITHUB_ACTIONS:
            print(f"::group::{title}")
            sys.stdout.flush()
        try:
            yield
        finally:
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                sys.stdout.flush()


# ============================================================================
# Logger Configuration
# ============================================================================

_configured_loggers: Dict[str, TheRockLogger] = {}
_config_lock = threading.Lock()


def configure_root_logger(
    level: int = None,
    format_style: str = None,
    log_file: Union[str, Path] = None,
    json_output: bool = False,
    use_colors: bool = True,
    enable_github_actions: bool = None,
):
    """
    Configure the root logger for TheRock
    
    Parameters:
    -----------
    level : int, optional
        Logging level (default: INFO in CI, DEBUG otherwise)
    format_style : str, optional
        Log format to use (default: CI format in CI, DETAILED otherwise)
    log_file : str or Path, optional
        Path to log file (default: None)
    json_output : bool
        Enable JSON formatted output (default: False)
    use_colors : bool
        Enable colored console output (default: True)
    enable_github_actions : bool, optional
        Enable GitHub Actions integration (default: auto-detect)
    """
    with _config_lock:
        root_logger = logging.getLogger("therock")
        
        # Set level
        if level is None:
            level = LogLevel.INFO if IS_CI else LogLevel.DEBUG
        root_logger.setLevel(level)
        
        # Remove existing handlers
        root_logger.handlers.clear()
        
        # Determine format
        if format_style is None:
            format_style = LogFormat.CI if IS_CI else LogFormat.DETAILED
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if json_output:
            console_formatter = JSONFormatter()
        elif format_style == LogFormat.JSON:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter(format_style, use_color=use_colors)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            
            if json_output or format_style == LogFormat.JSON:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(LogFormat.DIAGNOSTIC)
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # GitHub Actions handler
        if enable_github_actions is None:
            enable_github_actions = IS_GITHUB_ACTIONS
        
        if enable_github_actions:
            gh_handler = GitHubActionsHandler()
            gh_handler.setLevel(logging.WARNING)  # Only warnings and errors
            root_logger.addHandler(gh_handler)
        
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
    
    # Example 3: GitHub Actions integration
    logger.github_info("Build started")
    logger.github_warning("Deprecated API usage detected")
    
    # Example 4: Structured logging
    logger.log_dict({"status": "success", "items_processed": 42}, message="Results")
    
    # Example 5: Exception logging
    try:
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.log_exception(e, "Operation failed")

