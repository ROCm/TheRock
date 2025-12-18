#!/usr/bin/env python3
"""
Sample CTest Application with Unified Logging

Demonstrates how to use TheRock's unified logging framework with CTest.
Shows logging for:
- CTest configuration and setup
- Parallel test execution
- Test timeout handling
- Result aggregation and reporting
"""

import os
import sys
import time
from pathlib import Path
from logging_config import get_logger, configure_root_logger
from test_runner import TestRunner
import logging


def main():
    """Demo CTest execution with unified logging"""
    
    # Configure logging for CTest demo
    configure_root_logger(
        level=logging.INFO,
        log_file=Path("logs/ctest_demo.log"),
        use_colors=True
    )
    
    logger = get_logger(__name__, component="ctest_demo", operation="test_execution")
    
    logger.info("=" * 70)
    logger.info("🧪 CTEST DEMO - Unified Logging Framework")
    logger.info("=" * 70)
    
    # Simulate CTest configuration
    logger.info("Configuring CTest environment")
    
    ctest_config = {
        "component": "rocfft",
        "test_dir": "/path/to/build/rocfft/test",
        "parallel_jobs": 8,
        "timeout_per_test": 300,
        "test_type": "full"
    }
    
    logger.info(
        "CTest configuration",
        extra=ctest_config
    )
    
    # Simulate test directory validation
    with logger.timed_operation("validate_test_directory"):
        logger.debug(f"Checking test directory: {ctest_config['test_dir']}")
        logger.debug("Validating CTestTestfile.cmake")
        time.sleep(0.05)
    
    logger.info(f"✓ Test directory validated")
    
    # Create test runner
    runner = TestRunner(component="rocfft", test_type="full")
    
    # Simulate CTest discovery
    logger.info("Discovering CTest tests...")
    with logger.timed_operation("ctest_discovery"):
        mock_tests = [
            "rocfft_UnitTest",
            "rocfft_callback_test",
            "rocfft_accuracy_test_pow2",
        ]
        time.sleep(0.1)
    
    logger.info(f"Found {len(mock_tests)} test suites")
    for test in mock_tests:
        logger.debug(f"  - {test}")
    
    # Simulate parallel test execution
    logger.info(f"Starting parallel test execution (jobs={ctest_config['parallel_jobs']})")
    
    test_results = []
    
    with logger.timed_operation("ctest_parallel_execution"):
        for i, test_name in enumerate(mock_tests, 1):
            logger.info(f"[{i}/{len(mock_tests)}] Running: {test_name}")
            
            # Simulate variable test times
            test_duration = 0.1 + (i * 0.05)
            time.sleep(test_duration)
            
            # All tests pass in this demo
            result = {
                "name": test_name,
                "status": "Passed",
                "duration": test_duration,
                "subtests_passed": 100 + (i * 50),
                "subtests_total": 100 + (i * 50)
            }
            test_results.append(result)
            
            logger.info(
                f"✓ {test_name} completed",
                extra={
                    "duration_sec": test_duration,
                    "subtests": f"{result['subtests_passed']}/{result['subtests_total']}"
                }
            )
    
    # Aggregate and log results
    total_tests = len(test_results)
    passed = sum(1 for r in test_results if r["status"] == "Passed")
    total_subtests = sum(r["subtests_total"] for r in test_results)
    passed_subtests = sum(r["subtests_passed"] for r in test_results)
    
    summary = {
        "test_suites": {
            "total": total_tests,
            "passed": passed,
            "failed": total_tests - passed,
            "pass_rate": f"{(passed/total_tests)*100:.1f}%"
        },
        "individual_tests": {
            "total": total_subtests,
            "passed": passed_subtests,
            "failed": total_subtests - passed_subtests,
            "pass_rate": f"{(passed_subtests/total_subtests)*100:.1f}%"
        }
    }
    
    logger.info("=" * 70)
    logger.info("CTEST RESULTS")
    logger.info("=" * 70)
    logger.log_dict(summary, level=logging.INFO, message="Summary:")
    
    if passed == total_tests:
        logger.info(f"✅ All {total_tests} test suites passed!")
        logger.info(f"✅ Total: {passed_subtests}/{total_subtests} individual tests passed")
    else:
        logger.error(f"❌ {total_tests - passed} test suite(s) failed")
    
    # Demonstrate detailed test reporting
    logger.info("\nDetailed Test Results:")
    for result in test_results:
        logger.info(
            f"  {result['name']}: {result['status']} "
            f"({result['subtests_passed']}/{result['subtests_total']} tests, "
            f"{result['duration']:.2f}s)"
        )
    
    logger.info("=" * 70)
    logger.info("CTest demo completed - check logs/ctest_demo.log for details")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


