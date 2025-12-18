#!/usr/bin/env python3
"""
Mock rocWMMA Test Runner - Demo Version
========================================
Demonstrates CTest integration with unified logging framework
without requiring actual GPU hardware or compiled binaries.

Environment Variables:
- TEST_TYPE: smoke|regression|full (default: full)
- SHARD_INDEX: 1-based shard index (default: 1)
- TOTAL_SHARDS: Total number of shards (default: 1)
- AMDGPU_FAMILIES: GPU architecture (default: gfx942)
- DEMO_FAILURES: Comma-separated test indices to fail (e.g., "2,5")
- DEMO_SKIPS: Comma-separated test indices to skip (e.g., "8")
"""

import os
import sys
import time
import random
from pathlib import Path

# Add _therock_utils to path for unified logging
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_therock_utils"))

from logging_config import configure_root_logger, get_logger
import logging

# Configure unified logging with INFO level
configure_root_logger(level=logging.INFO)
logger = get_logger(__name__, component="rocwmma", operation="demo_test")

logger.info("=" * 60)
logger.info("🚀 Starting rocWMMA CTest Demo (Mock)")
logger.info("=" * 60)

# Environment setup
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "gfx942")
platform = os.getenv("RUNNER_OS", "Linux").lower()
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# CTest sharding
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "1")) - 1
TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", "1"))
test_type = os.getenv("TEST_TYPE", "full").lower()

# Mock failure configuration (for demonstrating error logging)
DEMO_FAILURES = os.getenv("DEMO_FAILURES", "")
DEMO_SKIPS = os.getenv("DEMO_SKIPS", "")
fail_indices = set(int(x.strip()) for x in DEMO_FAILURES.split(",") if x.strip())
skip_indices = set(int(x.strip()) for x in DEMO_SKIPS.split(",") if x.strip())

logger.info(f"📋 Test Configuration:")
logger.info(f"   Component: rocWMMA")
logger.info(f"   Test Type: {test_type}")
logger.info(f"   Shard: {SHARD_INDEX + 1} of {TOTAL_SHARDS}")
logger.info(f"   Platform: {platform}")
logger.info(f"   GPU Families: {AMDGPU_FAMILIES}")
if fail_indices:
    logger.warning(f"   ⚠️  Mock failures enabled for test indices: {sorted(fail_indices)}")
if skip_indices:
    logger.warning(f"   ⚠️  Mock skips enabled for test indices: {sorted(skip_indices)}")

# Mock CTest cases (realistic rocWMMA test names)
ALL_CTESTS = [
    "gemm_PGR0_LB0_MP0_MB_CP",
    "gemm_PGR0_LB0_MP0_MB_NC",
    "gemm_PGR0_LB2_MP0_MB_CP",
    "gemm_PGR0_LB2_MP0_MB_NC",
    "gemm_PGR1_LB0_MP0_MB_CP",
    "gemm_PGR1_LB0_MP0_MB_NC",
    "gemm_PGR1_LB2_MP0_MB_CP",
    "gemm_PGR1_LB2_MP0_MB_NC",
    "unit_kernel_base_test",
    "unit_contamination_test",
    "unit_fill_fragment_test",
    "unit_load_store_matrix_sync_test",
    "unit_load_store_matrix_coop_sync_test",
    "dlrm_test_fp16",
    "dlrm_test_bf16",
    "simple_sgemm",
    "simple_dgemm",
    "simple_hgemm",
    "attention_forward",
    "attention_backward",
    "ad_hoc_test_PGR0",
    "ad_hoc_test_PGR1",
    "mma_sync_test",
    "barrier_test",
    "cross_lane_ops",
]

# Determine test subset based on test_type
timeout = 3600
if test_type == "smoke":
    test_patterns = ["simple_", "unit_kernel", "unit_fill"]
    tests_to_run = [t for t in ALL_CTESTS if any(p in t for p in test_patterns)]
    timeout = 720
    logger.info(f"🔥 Running smoke tests ({len(tests_to_run)} tests)")
elif test_type == "regression":
    test_patterns = ["unit_", "simple_", "gemm_PGR0"]
    tests_to_run = [t for t in ALL_CTESTS if any(p in t for p in test_patterns)]
    timeout = 720
    logger.info(f"🔧 Running regression tests ({len(tests_to_run)} tests)")
else:
    tests_to_run = ALL_CTESTS
    logger.info(f"📚 Running full test suite ({len(tests_to_run)} tests)")

logger.info(f"⏱️  Timeout: {timeout}s per test")
logger.info(f"🔧 Parallel jobs: 8 (simulated)")

# Apply sharding
shard_tests = [t for i, t in enumerate(tests_to_run) if i % TOTAL_SHARDS == SHARD_INDEX]
logger.info(f"📊 Tests in this shard: {len(shard_tests)}")

logger.info("")
logger.info("=" * 60)
logger.info("🧪 Executing CTests")
logger.info("=" * 60)

# Simulate CTest execution
passed_tests = 0
failed_tests = []
skipped_tests = []
total_duration = 0

with logger.timed_operation("rocwmma_ctest_execution"):
    for i, test_name in enumerate(shard_tests, 1):
        logger.info(f"")
        logger.info(f"Test #{i}: {test_name}")
        
        # Simulate test execution time (CTests are typically longer)
        test_start = time.time()
        test_duration = random.uniform(0.5, 2.0)
        time.sleep(test_duration / 10)  # Speed up for demo
        
        # Check if this test should fail or skip (for demo purposes)
        if i in fail_indices:
            test_result = "Failed"
            failed_tests.append(test_name)
            logger.error(f"   ❌ {test_result}")
            logger.error(f"      Reason: Matrix dimensions mismatch - expected [16,16], got [16,8]")
            logger.error(f"      Duration: {test_duration:.2f}s")
        elif i in skip_indices:
            test_result = "Skipped"
            skipped_tests.append(test_name)
            logger.warning(f"   ⚠️  {test_result}")
            logger.warning(f"      Reason: GPU architecture {AMDGPU_FAMILIES} not supported for this test")
        else:
            test_result = "Passed"
            passed_tests += 1
            logger.info(f"   ✅ {test_result} ({test_duration:.2f}s)")
        
        total_duration += test_duration

logger.info("")
logger.info("=" * 60)
logger.info("📊 CTest Results Summary")
logger.info("=" * 60)

# Calculate metrics
total_tests = len(shard_tests)
num_failed = len(failed_tests)
num_skipped = len(skipped_tests)
success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
avg_duration = total_duration / total_tests if total_tests > 0 else 0

# Log summary with structured data
test_results = {
    "component": "rocwmma",
    "test_type": test_type,
    "total": total_tests,
    "passed": passed_tests,
    "failed": num_failed,
    "skipped": num_skipped,
    "success_rate": f"{success_rate:.1f}%",
    "total_duration_sec": f"{total_duration:.2f}",
    "avg_duration_sec": f"{avg_duration:.2f}"
}

logger.info(
    f"Results: {passed_tests}/{total_tests} passed, {num_failed} failed, {num_skipped} skipped",
    extra=test_results
)

logger.info(f"   Total Tests: {total_tests}")
logger.info(f"   ✅ Passed: {passed_tests}")
logger.info(f"   ❌ Failed: {num_failed}")
logger.info(f"   ⚠️  Skipped: {num_skipped}")
logger.info(f"   ⏱️  Total Time: {total_duration:.2f}s")
logger.info(f"   Success Rate: {success_rate:.1f}%")

# Log failed test names (matching TestRunner behavior)
if failed_tests:
    logger.info("")
    logger.error(f"❌ {num_failed} test(s) failed:")
    for test_name in failed_tests:
        logger.error(f"   - {test_name}")

# Log skipped test names
if skipped_tests:
    logger.info("")
    logger.warning(f"⚠️  {num_skipped} test(s) skipped:")
    for test_name in skipped_tests:
        logger.warning(f"   - {test_name}")

logger.info("")
logger.info("=" * 60)
logger.info("🎯 Performance Metrics")
logger.info("=" * 60)
logger.info(f"   Average test duration: {avg_duration:.2f}s")
logger.info(f"   Tests per minute: {len(shard_tests) / (total_duration / 60):.1f}")
logger.info(f"   GPU utilization: {random.randint(75, 95)}% (simulated)")
logger.info("=" * 60)

# Exit with appropriate code
if num_failed > 0:
    logger.error("❌ Some tests failed!")
    sys.exit(1)
else:
    logger.info("✅ All tests passed!")
    sys.exit(0)

