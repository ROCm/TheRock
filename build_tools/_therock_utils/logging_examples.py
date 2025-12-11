#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TheRock Logging - Usage Examples
=================================

This file demonstrates how to use the standardized logging framework
across different scenarios in TheRock.
"""

from pathlib import Path
import time
from logging_config import (
    get_logger,
    configure_root_logger,
    set_log_level,
    get_log_file_path,
    LogLevel,
    LogFormat,
)


# ============================================================================
# Example 1: Basic Logger Usage
# ============================================================================

def example_basic_logging():
    """Basic logging across different levels"""
    logger = get_logger(__name__)
    
    logger.debug("Debug information for developers")
    logger.info("General information about program flow")
    logger.warning("Something unusual happened but we can continue")
    logger.error("An error occurred")
    logger.critical("System failure!")


# ============================================================================
# Example 2: Component-Specific Logging
# ============================================================================

def example_component_logging():
    """Logging with component context"""
    # Packaging component
    pkg_logger = get_logger(__name__, component="packaging", operation="install")
    pkg_logger.info("Installing package", extra={"package": "rocm-core", "version": "6.2.0"})
    
    # Build component
    build_logger = get_logger(__name__, component="build", operation="compile")
    build_logger.info("Compiling source", extra={"target": "rocm-smi", "threads": 8})
    
    # Test component
    test_logger = get_logger(__name__, component="testing", operation="run")
    test_logger.info("Running tests", extra={"suite": "unit_tests", "count": 150})


# ============================================================================
# Example 3: Performance Timing
# ============================================================================

def example_timing():
    """Track operation performance"""
    logger = get_logger(__name__, component="performance")
    
    # Method 1: Context manager
    with logger.timed_operation("database_query"):
        time.sleep(0.1)  # Simulate work
        logger.info("Fetched 1000 records")
    
    # Method 2: Decorator
    @logger.timed("complex_calculation")
    def complex_operation():
        time.sleep(0.05)
        return 42
    
    result = complex_operation()
    logger.info(f"Calculation result: {result}")


# ============================================================================
# Example 4: Exception Handling
# ============================================================================

def example_exception_logging():
    """Proper exception logging"""
    logger = get_logger(__name__, component="error_handling")
    
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.log_exception(e, "Division operation failed")
    
    try:
        # File operation error
        with open("/nonexistent/file.txt") as f:
            content = f.read()
    except FileNotFoundError as e:
        logger.error(
            "Configuration file not found",
            extra={
                "file_path": "/nonexistent/file.txt",
                "error_code": "CONFIG_001"
            }
        )
        logger.log_exception(e)


# ============================================================================
# Example 5: Structured Logging
# ============================================================================

def example_structured_logging():
    """Log structured data"""
    logger = get_logger(__name__, component="reporting")
    
    # Log dictionary
    build_results = {
        "status": "success",
        "duration_seconds": 125,
        "artifacts": ["rocm-core.deb", "rocm-dev.deb"],
        "warnings": 3,
        "errors": 0
    }
    logger.log_dict(build_results, message="Build Summary")
    
    # Log with extra fields
    logger.info(
        "Package installation completed",
        extra={
            "package_name": "rocm-hip",
            "version": "6.2.0",
            "size_mb": 45.3,
            "install_time_ms": 1250,
            "dependencies": ["rocm-core", "libc6"]
        }
    )


# ============================================================================
# Example 6: GitHub Actions Integration
# ============================================================================

def example_github_actions():
    """GitHub Actions specific logging"""
    logger = get_logger(__name__, component="ci")
    
    # Regular logs
    logger.github_info("Starting CI pipeline")
    
    # Warnings with file annotations
    logger.github_warning(
        "Deprecated function usage",
        file="src/example.py",
        line=42
    )
    
    # Errors with annotations
    logger.github_error(
        "Build failed: syntax error",
        file="src/main.cpp",
        line=150
    )
    
    # Collapsible log groups
    with logger.github_group("Package Installation"):
        logger.info("Installing rocm-core...")
        time.sleep(0.1)
        logger.info("Installing rocm-hip...")
        time.sleep(0.1)
        logger.info("Installation complete")


# ============================================================================
# Example 7: File Logging
# ============================================================================

def example_file_logging():
    """Write logs to file"""
    # Configure with file output
    log_file = get_log_file_path("packaging", timestamp=True)
    
    configure_root_logger(
        level=LogLevel.DEBUG,
        format_style=LogFormat.DETAILED,
        log_file=log_file,
        json_output=False
    )
    
    logger = get_logger(__name__, component="packaging")
    logger.info(f"Logs will be written to: {log_file}")
    
    # All subsequent logs go to both console and file
    for i in range(5):
        logger.info(f"Processing item {i+1}/5")


# ============================================================================
# Example 8: JSON Logging
# ============================================================================

def example_json_logging():
    """Structured JSON log output"""
    # Configure for JSON output
    configure_root_logger(
        level=LogLevel.INFO,
        json_output=True
    )
    
    logger = get_logger(__name__, component="api")
    
    # These will be output as JSON
    logger.info("API request received", extra={
        "method": "POST",
        "endpoint": "/api/v1/packages",
        "user_id": "user123",
        "ip_address": "192.168.1.100"
    })
    
    logger.error("API error", extra={
        "error_code": "RATE_LIMIT",
        "retry_after": 60,
        "request_id": "abc-123"
    })


# ============================================================================
# Example 9: Dynamic Log Level Changes
# ============================================================================

def example_dynamic_log_level():
    """Change log levels at runtime"""
    logger = get_logger(__name__, component="debug")
    
    # Start with INFO level
    set_log_level(LogLevel.INFO)
    logger.debug("This won't be shown")
    logger.info("This will be shown")
    
    # Switch to DEBUG level
    logger.info("Enabling debug mode...")
    set_log_level(LogLevel.DEBUG)
    logger.debug("Now debug messages are visible")
    
    # Can also use string
    set_log_level("WARNING")
    logger.info("This won't be shown anymore")
    logger.warning("But warnings still show")


# ============================================================================
# Example 10: Real-World Package Installation Scenario
# ============================================================================

def example_package_installation():
    """Complete example: package installation with logging"""
    logger = get_logger(__name__, component="packaging", operation="install")
    
    packages = ["rocm-core", "rocm-hip", "rocm-opencl"]
    
    with logger.github_group("Package Installation"):
        logger.info(f"Starting installation of {len(packages)} packages")
        
        for pkg in packages:
            with logger.timed_operation(f"install_{pkg}"):
                try:
                    logger.info(f"Installing {pkg}...", extra={"package": pkg})
                    
                    # Simulate installation
                    time.sleep(0.1)
                    
                    # Simulate random issues
                    if pkg == "rocm-opencl":
                        raise Exception("Dependency conflict detected")
                    
                    logger.info(f"Successfully installed {pkg}", extra={
                        "package": pkg,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.log_exception(e, f"Failed to install {pkg}")
                    logger.github_error(
                        f"Package installation failed: {pkg}",
                        extra={"package": pkg, "error": str(e)}
                    )
        
        logger.info("Installation process completed")


# ============================================================================
# Example 11: Migration from Old Logging
# ============================================================================

def example_migration_from_old_style():
    """How to migrate from old-style logging"""
    
    # OLD STYLE - Direct print
    # print("Starting operation")
    # sys.stdout.flush()
    
    # NEW STYLE
    logger = get_logger(__name__)
    logger.info("Starting operation")
    
    # OLD STYLE - Custom log function
    # def log(*args):
    #     print(*args)
    #     sys.stdout.flush()
    # log("Processing file:", filename)
    
    # NEW STYLE
    filename = "example.txt"
    logger.info("Processing file", extra={"filename": filename})
    
    # OLD STYLE - GitHub Actions _log
    # def _log(*args, **kwargs):
    #     print(*args, **kwargs)
    #     sys.stdout.flush()
    # _log("Build completed")
    
    # NEW STYLE
    logger.github_info("Build completed")


# ============================================================================
# Example 12: Multi-threaded Logging
# ============================================================================

def example_threaded_logging():
    """Logging in multi-threaded applications"""
    import threading
    
    def worker_task(worker_id: int):
        logger = get_logger(__name__, component="worker", operation=f"task_{worker_id}")
        
        logger.info(f"Worker {worker_id} started")
        time.sleep(0.1)
        logger.info(f"Worker {worker_id} completed", extra={"worker_id": worker_id})
    
    # Create multiple worker threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_task, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all examples"""
    print("=" * 70)
    print("TheRock Logging Framework - Examples")
    print("=" * 70)
    
    examples = [
        ("Basic Logging", example_basic_logging),
        ("Component Logging", example_component_logging),
        ("Performance Timing", example_timing),
        ("Exception Handling", example_exception_logging),
        ("Structured Logging", example_structured_logging),
        ("GitHub Actions", example_github_actions),
        ("File Logging", example_file_logging),
        ("Dynamic Log Levels", example_dynamic_log_level),
        ("Package Installation", example_package_installation),
        ("Migration Guide", example_migration_from_old_style),
        ("Multi-threaded", example_threaded_logging),
    ]
    
    for title, example_func in examples:
        print(f"\n{'─' * 70}")
        print(f"Example: {title}")
        print(f"{'─' * 70}")
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}")
        time.sleep(0.5)  # Brief pause between examples


if __name__ == "__main__":
    # Configure root logger for demo
    configure_root_logger(
        level=LogLevel.INFO,
        format_style=LogFormat.DETAILED,
        use_colors=True
    )
    
    main()

