#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Demonstration: Migrating packaging_utils.py to use standardized logging

This file shows a side-by-side comparison of old vs new logging approaches
for a typical TheRock Python file.
"""

# ==============================================================================
# BEFORE: Old Style (Current packaging_utils.py approach)
# ==============================================================================

"""
OLD CODE - Multiple inconsistent logging patterns:
"""

def old_style_example():
    import logging
    import sys
    
    # Pattern 1: Manual logger configuration (duplicated across files)
    logger = logging.getLogger("rocm_installer")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(ch)
    
    # Pattern 2: Simple print with flush
    def log(*args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()
    
    # Pattern 3: Custom _log function
    def _log(*args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()
    
    # Usage examples - inconsistent
    logger.info("Starting operation")  # Using logging module
    log("Processing file...")         # Using custom log()
    _log("GitHub action step")        # Using _log()
    print("Quick debug")               # Direct print
    
    # Error handling - manual formatting
    try:
        risky_operation()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Timing - manual calculation
    import time
    start = time.time()
    do_work()
    duration = time.time() - start
    print(f"Operation took {duration:.2f}s")


# ==============================================================================
# AFTER: New Style (Using standardized logging framework)
# ==============================================================================

"""
NEW CODE - Consistent, structured, feature-rich logging:
"""

def new_style_example():
    from _therock_utils.logging_config import get_logger
    
    # Single line to get configured logger - no manual setup!
    logger = get_logger(__name__, component="packaging", operation="install")
    
    # All logging uses the same interface
    logger.info("Starting operation")
    logger.info("Processing file...", extra={"filename": "example.txt"})
    logger.github_info("GitHub action step")  # GitHub Actions integration built-in
    logger.debug("Quick debug message")  # Proper log levels
    
    # Error handling - automatic formatting and traceback
    try:
        risky_operation()
    except Exception as e:
        logger.log_exception(e, "Operation failed")
        # Traceback included automatically!
    
    # Timing - built-in, automatic, and includes measurements
    with logger.timed_operation("work"):
        do_work()
        # Duration logged automatically in milliseconds


# ==============================================================================
# REAL EXAMPLE: Package Installation Function
# ==============================================================================

def comparison_install_package():
    """Side-by-side comparison of package installation logging"""
    
    # --------------------------------------------------------------------------
    # OLD STYLE
    # --------------------------------------------------------------------------
    def install_package_old(pkg_name):
        import logging
        import subprocess
        
        logger = logging.getLogger("rocm_installer")
        logger.info(f"Installing {pkg_name}")
        
        try:
            result = subprocess.run(
                ["sudo", "dpkg", "-i", pkg_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Installed {pkg_name} successfully")
                
        except Exception as e:
            logger.error(f"Exception installing {pkg_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # --------------------------------------------------------------------------
    # NEW STYLE
    # --------------------------------------------------------------------------
    def install_package_new(pkg_name):
        from _therock_utils.logging_config import get_logger
        import subprocess
        
        logger = get_logger(__name__, component="packaging", operation="install")
        
        logger.info("Starting package installation", extra={"package": pkg_name})
        
        try:
            with logger.timed_operation(f"install_{pkg_name}"):
                result = subprocess.run(
                    ["sudo", "dpkg", "-i", pkg_name],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode != 0:
                    logger.error(
                        "Package installation failed",
                        extra={
                            "package": pkg_name,
                            "exit_code": result.returncode,
                            "output": result.stdout.strip()
                        }
                    )
                else:
                    logger.info(
                        "Package installed successfully",
                        extra={"package": pkg_name}
                    )
                    
        except subprocess.TimeoutExpired as e:
            logger.error(
                "Installation timeout",
                extra={"package": pkg_name, "timeout_seconds": e.timeout}
            )
        except Exception as e:
            logger.log_exception(e, f"Unexpected error installing {pkg_name}")


# ==============================================================================
# BENEFITS COMPARISON
# ==============================================================================

def show_benefits():
    """Demonstrate the benefits of the new logging system"""
    
    print("\n" + "="*70)
    print("BENEFITS OF STANDARDIZED LOGGING")
    print("="*70)
    
    benefits = {
        "Consistency": [
            "✗ OLD: 3+ different logging patterns in same file",
            "✓ NEW: Single consistent API across entire codebase"
        ],
        "Configuration": [
            "✗ OLD: Manual setup in every file (10+ lines)",
            "✓ NEW: Zero configuration needed (auto-configured)"
        ],
        "Structured Data": [
            "✗ OLD: String formatting only, hard to parse",
            "✓ NEW: JSON-serializable structured logging via extra={}"
        ],
        "Timing": [
            "✗ OLD: Manual time.time() calculations",
            "✓ NEW: Built-in timing with context managers"
        ],
        "Exception Handling": [
            "✗ OLD: Manual traceback.print_exc()",
            "✓ NEW: Automatic traceback with log_exception()"
        ],
        "GitHub Actions": [
            "✗ OLD: Manual ::warning:: formatting",
            "✓ NEW: Built-in github_warning() methods"
        ],
        "Log Levels": [
            "✗ OLD: Mixed print() and logger calls",
            "✓ NEW: Proper DEBUG/INFO/WARNING/ERROR/CRITICAL levels"
        ],
        "Performance": [
            "✗ OLD: No built-in metrics",
            "✓ NEW: Automatic timing and performance tracking"
        ],
        "Testing": [
            "✗ OLD: Hard to test print() statements",
            "✓ NEW: Easy to mock and capture logger output"
        ],
        "Production": [
            "✗ OLD: No structured logs for aggregation",
            "✓ NEW: JSON output for ELK/Splunk integration"
        ]
    }
    
    for category, comparisons in benefits.items():
        print(f"\n{category}:")
        for comparison in comparisons:
            print(f"  {comparison}")


# ==============================================================================
# REAL-WORLD FILE MIGRATION EXAMPLE
# ==============================================================================

def migration_template():
    """Template for migrating an existing file"""
    
    print("\n" + "="*70)
    print("STEP-BY-STEP MIGRATION TEMPLATE")
    print("="*70)
    
    steps = """
    STEP 1: Add import at top of file
    ────────────────────────────────────
    from _therock_utils.logging_config import get_logger
    
    STEP 2: Create module-level logger
    ────────────────────────────────────
    logger = get_logger(__name__, component="your_component")
    
    STEP 3: Replace all custom logging
    ────────────────────────────────────
    # Old:
    def log(*args):
        print(*args)
        sys.stdout.flush()
    
    log("message")
    
    # New:
    logger.info("message")
    
    STEP 4: Update exception handling
    ────────────────────────────────────
    # Old:
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    
    # New:
    except Exception as e:
        logger.log_exception(e, "Operation failed")
    
    STEP 5: Add timing where useful
    ────────────────────────────────────
    # Old:
    start = time.time()
    do_work()
    print(f"Took {time.time() - start}s")
    
    # New:
    with logger.timed_operation("work"):
        do_work()
    
    STEP 6: Add structured data
    ────────────────────────────────────
    # Old:
    logger.info(f"Installed {pkg} v{version}")
    
    # New:
    logger.info("Package installed", extra={"package": pkg, "version": version})
    
    STEP 7: Remove manual logger config
    ────────────────────────────────────
    # Remove all this:
    logger = logging.getLogger()
    logger.setLevel(...)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(...)
    # etc.
    
    STEP 8: Test and verify
    ────────────────────────────────────
    - Run the script
    - Check log output format
    - Verify all messages appear
    - Test with different log levels
    """
    
    print(steps)


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("THEROCK LOGGING MIGRATION DEMONSTRATION")
    print("="*70)
    
    # Show benefits
    show_benefits()
    
    # Show migration steps
    migration_template()
    
    print("\n" + "="*70)
    print("QUICK REFERENCE")
    print("="*70)
    
    quick_ref = """
    Import:
        from _therock_utils.logging_config import get_logger
    
    Setup:
        logger = get_logger(__name__, component="mycomponent")
    
    Usage:
        logger.debug("Debug info")
        logger.info("General info")
        logger.warning("Warning message")
        logger.error("Error occurred", extra={"code": 500})
        logger.critical("Critical failure")
    
    Timing:
        with logger.timed_operation("operation_name"):
            do_work()
    
    Exceptions:
        try:
            ...
        except Exception as e:
            logger.log_exception(e, "Context message")
    
    GitHub Actions:
        logger.github_info("Build started")
        logger.github_warning("Deprecated API")
        logger.github_error("Build failed")
    
    Structured:
        logger.info("Event", extra={"key1": "value1", "key2": 42})
    
    For more examples, see:
        - logging_examples.py
        - LOGGING_MIGRATION_GUIDE.md
    """
    
    print(quick_ref)
    
    print("\n" + "="*70)
    print("Ready to migrate? Follow LOGGING_MIGRATION_GUIDE.md")
    print("="*70 + "\n")

