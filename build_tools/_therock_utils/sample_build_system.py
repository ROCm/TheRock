#!/usr/bin/env python3
"""
Sample Build System - Demonstrates logging framework usage
"""

import time
import logging
from logging_config import get_logger, configure_root_logger


def configure_build(config):
    """
    Sample build configuration with logging
    
    Demonstrates:
    - Nested timed_operation: Each configuration step is timed individually
    - Structured logging: Build configuration details
    """
    logger = get_logger("sample.build_system", component="BuildSystem", operation="configure")
    
    logger.info("Configuring build environment", extra={
        "build_type": config.get("type", "release"),
        "target": config.get("target", "all"),
        "optimization_level": config.get("optimization", "O3"),
        "compiler": "gcc-11.4.0"
    })
    
    # Simulate configuration steps with individual timing
    steps = [
        ("Checking dependencies", {"cmake": "3.24.0", "gcc": "11.4.0", "python": "3.10"}),
        ("Setting up paths", {"build_dir": "/build", "install_prefix": "/opt/rocm"}),
        ("Validating config", {"arch": "x86_64", "gpu_targets": "gfx942,gfx90a"})
    ]
    
    for step_name, step_data in steps:
        with logger.timed_operation(f"Configure: {step_name}"):
            logger.debug(f"Configuration step: {step_name}", extra={
                "step": step_name,
                "step_data": step_data
            })
            time.sleep(0.3)
            logger.debug(f"Completed: {step_name}", extra={
                "step": step_name,
                "status": "success"
            })
    
    logger.info("Build configuration complete")
    
    # Display structured configuration details
    config_details = {
        "build_type": config.get("type", "release"),
        "target": config.get("target", "all"),
        "optimization_level": config.get("optimization", "O3"),
        "total_steps": len(steps),
        "cmake_flags": ["-DCMAKE_BUILD_TYPE=Release", "-DGPU_TARGETS=gfx942"]
    }
    logger.log_dict(config_details, message="📊 Configuration Details:")


def compile_components(components):
    """
    Sample compilation with performance tracking
    
    Demonstrates:
    - timed_operation: Automatic timing for each component compilation
    - Structured logging with detailed compilation metrics
    - Nested operations: Sub-tasks within compilation
    """
    logger = get_logger("sample.build_system", component="BuildSystem", operation="compile")
    
    logger.info(f"Starting compilation of {len(components)} components", extra={
        "component_count": len(components),
        "operation": "compile",
        "parallel_jobs": 8,
        "compiler_flags": ["-O3", "-march=native", "-DNDEBUG"]
    })
    
    total_source_files = 0
    total_objects = 0
    
    for i, component in enumerate(components, 1):
        source_files = 50 + (i * 30)
        object_files = source_files
        total_source_files += source_files
        total_objects += object_files
        
        # timed_operation automatically tracks duration and logs:
        # - DEBUG: "Starting operation: Compiling {component}"
        # - INFO: "Completed operation: Compiling {component}" (duration_ms in extra)
        with logger.timed_operation(f"Compiling {component}"):
            logger.info(f"🔨 Compiling component {i}/{len(components)}: {component}", extra={
                "component": component,
                "status": "compiling",
                "source_files": source_files,
                "progress_pct": (i / len(components)) * 100
            })
            
            # Simulate compilation phases with nested timed operations
            with logger.timed_operation(f"Preprocessing {component}"):
                time.sleep(0.2)
                logger.debug(f"Preprocessing completed", extra={
                    "component": component,
                    "phase": "preprocess",
                    "headers_processed": source_files * 5
                })
            
            with logger.timed_operation(f"Compiling sources for {component}"):
                time.sleep(0.4)
                logger.debug(f"Source compilation completed", extra={
                    "component": component,
                    "phase": "compile",
                    "objects_created": object_files
                })
            
            with logger.timed_operation(f"Linking {component}"):
                time.sleep(0.2)
                logger.debug(f"Linking completed", extra={
                    "component": component,
                    "phase": "link",
                    "output_lib": f"lib{component}.so"
                })
            
            logger.info(f"✅ Component {component} compiled successfully", extra={
                "component": component,
                "status": "success",
                "source_files": source_files,
                "object_files": object_files,
                "output_size_kb": 1024 + (i * 512),
                "warnings": 0,
                "errors": 0
            })
    
    logger.info("Compilation phase completed")
    
    # Display structured compilation metrics
    compilation_metrics = {
        "total_components": len(components),
        "total_source_files": total_source_files,
        "total_objects": total_objects,
        "all_compiled": True
    }
    logger.log_dict(compilation_metrics, message="📊 Compilation Metrics:")


def run_tests(components):
    """
    Sample test execution with error scenarios
    
    Demonstrates:
    - timed_operation: Automatic timing for test execution
    - log_exception: Unified exception handling with detailed context
    - Structured data: Test metrics, failure details
    """
    logger = get_logger("sample.build_system", component="BuildSystem", operation="test")
    
    logger.info("Running post-build tests", extra={
        "total_components": len(components),
        "test_types": ["unit_tests", "integration_tests"]
    })
    
    passed_tests = 0
    failed_tests = 0
    total_test_count = 0
    
    for i, component in enumerate(components):
        test_count = 25 + (i * 10)
        total_test_count += test_count
        
        # Using timed_operation for automatic timing
        with logger.timed_operation(f"Test {component}"):
            try:
                logger.info(f"Testing component: {component}", extra={
                    "component": component,
                    "test_phase": "unit_tests",
                    "test_count": test_count,
                    "test_framework": "GTest"
                })
                
                # Simulate running unit tests
                with logger.timed_operation(f"Unit tests for {component}"):
                    time.sleep(0.3)
                    logger.debug(f"Unit tests progress", extra={
                        "component": component,
                        "tests_run": test_count,
                        "phase": "unit_tests"
                    })
                
                # Simulate test failure on second component
                if i == 1:
                    failed_count = 3
                    raise RuntimeError(
                        f"Unit tests failed for {component}: "
                        f"{failed_count}/{test_count} tests failed\n"
                        f"Failed tests: TestMemoryAllocation, TestKernelExecution, TestDataTransfer"
                    )
                
                passed_tests += test_count
                logger.info(f"✅ Tests passed for {component}", extra={
                    "component": component,
                    "test_result": "passed",
                    "tests_passed": test_count,
                    "tests_failed": 0,
                    "success_rate": 100.0
                })
                
            except Exception as e:
                failed_tests += test_count
                # Using log_exception for unified error handling
                logger.log_exception(e, f"❌ Tests failed for {component}", extra={
                    "component": component,
                    "test_result": "failed",
                    "tests_attempted": test_count,
                    "error_type": type(e).__name__,
                    "failure_category": "runtime_error"
                })
                
                # Log retry attempt (demonstrating exception handling patterns)
                logger.warning(f"Continuing with remaining components despite failure", extra={
                    "component": component,
                    "action": "continue",
                    "remaining_components": len(components) - i - 1
                })
    
    logger.info("Testing phase completed")
    
    # Display structured test metrics
    test_metrics = {
        "total_components": len(components),
        "total_tests": total_test_count,
        "passed": passed_tests,
        "failed": failed_tests,
        "success_rate_pct": round((passed_tests / total_test_count * 100), 2) if total_test_count > 0 else 0
    }
    logger.log_dict(test_metrics, message="📊 Test Metrics:")


def main():
    """Main demo function"""
    # Explicitly set DEBUG level to see timed_operation's detailed logs
    # Note: Logging auto-configures to INFO by default, but we want DEBUG for this demo
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*60)
    print("  Sample 2: Build System with Logging")
    print("  Demonstrates: Nested Timed Operations, Structured Data, Exception Handling")
    print("="*60 + "\n")
    
    # Setup logger
    logger = get_logger("sample.build_system", component="BuildSystem")
    logger.info("Build System Demo Started", extra={
        "demo_version": "2.0",
        "features": ["nested_timed_operations", "structured_metrics", "exception_handling"]
    })
    logger.debug("🔧 Logging initialized at DEBUG level - automatic timing enabled")
    
    # Build configuration
    config = {
        "type": "release",
        "target": "linux",
        "optimization": "O3",
        "gpu_targets": "gfx942,gfx90a"
    }
    
    # Components to build
    components = [
        "rocm-core",
        "hip-runtime",
        "miopen"
    ]
    
    logger.info("Starting build workflow", extra={
        "components": components,
        "build_config": config,
        "phases": ["configure", "compile", "test"]
    })
    
    # Run build process with nested timed operations
    with logger.timed_operation("Complete Build Process"):
        configure_build(config)
        compile_components(components)
        run_tests(components)
    
    logger.info("Build System Demo Completed", extra={
        "total_components": len(components),
        "config": config,
        "all_phases_completed": True,
        "build_result": "success_with_test_failures"
    })
    
    print("\n" + "="*60)
    print("  ✅ Demo completed - Check logs above for:")
    print("     - Nested timed operation durations (duration_ms)")
    print("     - Structured build metrics (files, sizes, counts)")
    print("     - Exception logs with full stack traces")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

