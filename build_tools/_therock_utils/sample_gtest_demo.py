#!/usr/bin/env python3
"""
Sample GTest Application with Unified Logging

Demonstrates how to use TheRock's unified logging framework with GTest.
Shows logging for:
- Test setup and configuration
- Test execution with sharding
- Result parsing and reporting
- Performance tracking
"""

import os
import sys
import time
from pathlib import Path
from logging_config import get_logger, configure_root_logger
from test_runner import TestRunner
import logging


def main():
    """Demo GTest execution with unified logging"""
    
    # Configure logging for GTest demo
    configure_root_logger(
        level=logging.INFO,
        log_file=Path("logs/gtest_demo.log"),
        use_colors=True
    )
    
    logger = get_logger(__name__, component="gtest_demo", operation="test_execution")
    
    logger.info("=" * 70)
    logger.info("🧪 GTEST DEMO - Unified Logging Framework")
    logger.info("=" * 70)
    
    # Simulate test configuration
    logger.info("Setting up GTest environment")
    
    # Example: Configure test sharding
    test_config = {
        "component": "rocblas",
        "test_type": "smoke",
        "shard_index": 0,
        "total_shards": 4,
        "gpu_family": "gfx1100"
    }
    
    logger.info(
        "Test configuration loaded",
        extra=test_config
    )
    
    # Simulate environment setup
    with logger.timed_operation("environment_setup"):
        logger.debug("Setting GTEST_SHARD_INDEX=0")
        logger.debug("Setting GTEST_TOTAL_SHARDS=4")
        logger.debug("Setting GTEST_BRIEF=1")
        time.sleep(0.05)  # Simulate setup time
    
    # Create test runner
    runner = TestRunner(component="rocblas", test_type="smoke")
    
    # Simulate test discovery
    logger.info("Discovering tests...")
    with logger.timed_operation("test_discovery"):
        mock_tests = [
            "MatrixMultiply.BasicFloat",
            "MatrixMultiply.BasicDouble",
            "GEMM.SquareMatrix",
            "BLAS1.VectorAdd",
        ]
        time.sleep(0.1)  # Simulate discovery
    
    logger.info(f"Found {len(mock_tests)} tests in shard 1/4")
    logger.log_dict(
        {"tests": mock_tests},
        level=logging.DEBUG,
        message="Test list:"
    )
    
    # Simulate test execution
    logger.info("Starting test execution")
    
    passed_tests = 0
    failed_tests = []
    
    with logger.timed_operation("gtest_execution"):
        for i, test_name in enumerate(mock_tests, 1):
            logger.debug(f"[{i}/{len(mock_tests)}] Running: {test_name}")
            
            # Simulate test execution time
            test_duration = 0.05 + (i * 0.01)
            time.sleep(test_duration)
            
            # Simulate 1 failure for demo purposes
            if i == 3:
                logger.warning(f"Test failed: {test_name}")
                failed_tests.append(test_name)
            else:
                logger.debug(f"✓ {test_name} passed ({test_duration*1000:.1f}ms)")
                passed_tests += 1
    
    # Log test results
    total_tests = len(mock_tests)
    test_results = {
        "total": total_tests,
        "passed": passed_tests,
        "failed": len(failed_tests),
        "pass_rate": f"{(passed_tests/total_tests)*100:.1f}%"
    }
    
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    logger.log_dict(test_results, level=logging.INFO, message="Summary:")
    
    if failed_tests:
        logger.error(f"❌ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            logger.error(f"  - {test}")
    else:
        logger.info(f"✅ All {total_tests} tests passed!")
    
    # Demonstrate exception logging
    try:
        logger.info("\nDemonstrating exception logging...")
        if failed_tests:
            raise RuntimeError(f"Test suite failed: {len(failed_tests)} test(s) failed")
    except Exception as e:
        logger.log_exception(
            e,
            "GTest execution completed with failures",
            extra={"failed_count": len(failed_tests), "failed_tests": failed_tests}
        )
    
    logger.info("=" * 70)
    logger.info("GTest demo completed - check logs/gtest_demo.log for details")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

