#!/usr/bin/env python3
"""
Sample Build System - Demonstrates logging framework usage
"""

import time
import logging
from logging_config import get_logger, configure_root_logger


def configure_build(config):
    """Sample build configuration with logging"""
    logger = get_logger(__name__, component="BuildSystem", operation="configure")
    
    logger.info("Configuring build environment", extra={
        "build_type": config.get("type", "release"),
        "target": config.get("target", "all")
    })
    
    # Simulate configuration steps
    steps = ["Checking dependencies", "Setting up paths", "Validating config"]
    
    for step in steps:
        logger.debug(f"Configuration step: {step}", extra={"step": step})
        time.sleep(0.3)
    
    logger.info("Build configuration complete", extra={
        "config": config
    })


def compile_components(components):
    """
    Sample compilation with performance tracking
    
    Demonstrates:
    - timed_operation: Automatic timing for each component compilation
    - Nested operations with automatic duration tracking
    """
    logger = get_logger(__name__, component="BuildSystem", operation="compile")
    
    logger.info(f"Starting compilation of {len(components)} components", extra={
        "component_count": len(components),
        "operation": "compile"
    })
    
    for component in components:
        # timed_operation automatically tracks duration and logs:
        # - DEBUG: "Starting operation: Compiling {component}"
        # - INFO: "Completed operation: Compiling {component}" (duration_ms in extra)
        with logger.timed_operation(f"Compiling {component}"):
            logger.info(f"🔨 Compiling component: {component}", extra={
                "component": component,
                "status": "compiling"
            })
            
            # Simulate compilation
            time.sleep(0.8)
            
            logger.info(f"✅ Component {component} compiled successfully", extra={
                "component": component,
                "status": "success"
            })


def run_tests(components):
    """Sample test execution with error scenarios"""
    logger = get_logger(__name__, component="BuildSystem", operation="test")
    
    logger.info("Running post-build tests")
    
    for i, component in enumerate(components):
        try:
            logger.info(f"Testing component: {component}", extra={
                "component": component,
                "test_phase": "unit_tests"
            })
            
            # Simulate test failure on second component
            if i == 1:
                raise RuntimeError(f"Unit tests failed for {component}")
            
            logger.info(f"Tests passed for {component}", extra={
                "component": component,
                "test_result": "passed"
            })
            
        except Exception as e:
            logger.error(f"Tests failed for {component}", extra={
                "component": component,
                "test_result": "failed",
                "error_type": type(e).__name__
            }, exc_info=True)
            
            # Continue with other components
            logger.warning(f"Continuing with remaining components despite failure")


def main():
    """Main demo function"""
    # Initialize logging with DEBUG level to see timed_operation's automatic logs
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 2: Build System with Logging")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger(__name__, component="BuildSystem")
    logger.info("Build System Demo Started")
    logger.debug("🔧 Logging initialized at DEBUG level - automatic timing enabled")
    
    # Build configuration
    config = {
        "type": "release",
        "target": "linux",
        "optimization": "O3"
    }
    
    # Components to build
    components = [
        "rocm-core",
        "hip-runtime",
        "miopen"
    ]
    
    # Run build process
    with logger.timed_operation("Complete Build Process"):
        configure_build(config)
        compile_components(components)
        run_tests(components)
    
    logger.info("Build System Demo Completed", extra={
        "total_components": len(components),
        "config": config
    })
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()

