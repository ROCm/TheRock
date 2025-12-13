#!/usr/bin/env python3
"""
Sample GitHub Actions Logging Demo

Demonstrates how to use GitHub Actions-specific logging methods:
- github_info(): Success notifications
- github_warning(): Warning annotations
- github_error(): Error annotations
- github_group(): Collapsible log sections

This script is designed to run in GitHub Actions workflows.
When run locally, it falls back to normal logging.
"""

import time
import logging
from logging_config import get_logger, configure_root_logger


def check_environment():
    """
    Check environment and dependencies
    
    Demonstrates:
    - github_group(): Organize related checks
    - github_info(): Success notifications
    - github_warning(): Non-critical issues
    """
    logger = get_logger(__name__, component="EnvironmentCheck")
    
    with logger.github_group("🔍 Environment Checks"):
        logger.info("Checking Python version...")
        time.sleep(0.3)
        logger.github_info("✅ Python 3.12 detected")
        
        logger.info("Checking disk space...")
        time.sleep(0.3)
        # Simulate low disk space warning
        disk_space_gb = 15
        if disk_space_gb < 20:
            logger.github_warning(
                f"Low disk space: {disk_space_gb}GB available (recommended: 20GB)",
                file="config/requirements.txt",
                line=5
            )
        else:
            logger.github_info(f"✅ Sufficient disk space: {disk_space_gb}GB")
        
        logger.info("Checking network connectivity...")
        time.sleep(0.3)
        logger.github_info("✅ Network connectivity verified")


def build_components():
    """
    Build project components
    
    Demonstrates:
    - github_group(): Organize build logs
    - github_info(): Build success messages
    - github_error(): Build failure annotations
    """
    logger = get_logger(__name__, component="Build")
    
    components = [
        {"name": "core", "should_fail": False},
        {"name": "runtime", "should_fail": True},  # Simulate failure
        {"name": "tools", "should_fail": False}
    ]
    
    with logger.github_group(f"🔨 Building {len(components)} Components"):
        for i, component in enumerate(components, 1):
            name = component["name"]
            logger.info(f"Building component {i}/{len(components)}: {name}...")
            
            with logger.timed_operation(f"Build {name}"):
                time.sleep(0.5)
                
                if component["should_fail"]:
                    # Simulate build error
                    error_msg = f"Compilation error in {name}: undefined reference to 'hipMalloc'"
                    logger.github_error(
                        error_msg,
                        file=f"src/{name}/device_memory.cpp",
                        line=142
                    )
                    logger.error(f"❌ Build failed for {name}")
                else:
                    logger.github_info(f"✅ Component '{name}' built successfully")


def run_tests():
    """
    Run test suite
    
    Demonstrates:
    - github_group(): Organize test logs
    - github_info(): Test pass notifications
    - github_warning(): Test warnings
    - github_error(): Test failure annotations
    """
    logger = get_logger(__name__, component="Testing")
    
    test_suites = [
        {"name": "unit_tests", "total": 45, "failed": 0, "warnings": 0},
        {"name": "integration_tests", "total": 20, "failed": 2, "warnings": 1},
        {"name": "performance_tests", "total": 10, "failed": 0, "warnings": 3}
    ]
    
    with logger.github_group("🧪 Running Test Suites"):
        for suite in test_suites:
            logger.info(f"Running {suite['name']}...")
            
            with logger.timed_operation(f"Test Suite: {suite['name']}"):
                time.sleep(0.7)
                
                if suite["warnings"] > 0:
                    logger.github_warning(
                        f"{suite['name']}: {suite['warnings']} test(s) with warnings",
                        file=f"tests/{suite['name']}/test_config.py",
                        line=89
                    )
                
                if suite["failed"] > 0:
                    logger.github_error(
                        f"{suite['name']}: {suite['failed']}/{suite['total']} tests failed",
                        file=f"tests/{suite['name']}/test_runner.py",
                        line=156
                    )
                    logger.error(f"❌ {suite['name']}: {suite['failed']} failures")
                else:
                    logger.github_info(
                        f"✅ {suite['name']}: All {suite['total']} tests passed"
                    )


def generate_report():
    """
    Generate and upload test report
    
    Demonstrates:
    - github_group(): Organize report generation
    - github_info(): Report upload confirmation
    """
    logger = get_logger(__name__, component="Reporting")
    
    with logger.github_group("📊 Generating Reports"):
        logger.info("Collecting test results...")
        time.sleep(0.3)
        
        logger.info("Generating HTML report...")
        time.sleep(0.4)
        
        logger.info("Generating JUnit XML...")
        time.sleep(0.3)
        
        logger.info("Uploading artifacts...")
        time.sleep(0.5)
        
        logger.github_info("✅ Reports uploaded: test-results.html, junit.xml")


def main():
    """
    Main workflow demonstrating GitHub Actions logging
    """
    # Configure logging for GitHub Actions
    configure_root_logger(level=logging.DEBUG)
    
    print("\n" + "="*70)
    print("  GitHub Actions Logging Demo")
    print("  Run this in a GitHub Actions workflow to see annotations!")
    print("="*70 + "\n")
    
    logger = get_logger(__name__, component="GitHubActionsDemo")
    
    logger.github_info("🚀 CI/CD Pipeline Started")
    
    try:
        # Phase 1: Environment checks
        check_environment()
        
        # Phase 2: Build
        build_components()
        
        # Phase 3: Test
        run_tests()
        
        # Phase 4: Report
        generate_report()
        
        logger.github_info("✅ Pipeline completed with warnings (see annotations above)")
        
    except Exception as e:
        logger.log_exception(e, "❌ Pipeline failed with critical error")
        logger.github_error("Pipeline execution failed - see logs for details")
        raise
    
    print("\n" + "="*70)
    print("  Demo completed!")
    print("  Check GitHub Actions UI for:")
    print("  - Blue 'info' badges for successes")
    print("  - Yellow 'warning' badges for warnings")
    print("  - Red 'error' badges for failures")
    print("  - Collapsible groups for organized logs")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

