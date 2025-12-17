import os
import sys
from pathlib import Path

# Add _therock_utils to path for unified logging
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "_therock_utils"))

from test_runner import TestRunner
from logging_config import configure_root_logger, get_logger
import logging

# Configure unified logging with INFO level
configure_root_logger(level=logging.INFO)
logger = get_logger(__name__, component="rocwmma", operation="test")

logger.info("=" * 60)
logger.info("🚀 Starting rocWMMA CTest Execution")
logger.info("=" * 60)

# Environment setup
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# Enable GTest "brief" output: only show failures and the final results
environ_vars["GTEST_BRIEF"] = str(1)

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

logger.info(f"📋 Test Configuration:")
logger.info(f"   Test Type: {test_type}")
logger.info(f"   Shard: {environ_vars.get('GTEST_SHARD_INDEX', 0)} of {environ_vars.get('GTEST_TOTAL_SHARDS', 1)}")
logger.info(f"   Platform: {platform}")
logger.info(f"   GPU Families: {AMDGPU_FAMILIES}")

# If there are devices for which the full set is too slow, we can
# programatically set test_type to "regression" here.

test_subdir = ""
timeout = "3600"
if test_type == "smoke":
    # The emulator regression tests are very fast.
    # If we need something even faster we can use "/smoke" here.
    test_subdir = "/regression"
    timeout = "720"
elif test_type == "regression":
    test_subdir = "/regression"
    timeout = "720"

# Initialize test runner with unified logging
runner = TestRunner(component="rocwmma", test_type=test_type)

logger.info(f"✅ Test directory: {THEROCK_BIN_DIR}/rocwmma{test_subdir}")
logger.info(f"⏱️  Timeout: {timeout}s per test")
logger.info(f"🔧 Parallel jobs: 8")

# Run CTest with unified logging
runner.run_ctest(
    test_dir=Path(f"{THEROCK_BIN_DIR}/rocwmma{test_subdir}"),
    parallel=8,
    timeout=timeout,
    cwd=THEROCK_DIR,
    env=environ_vars
)
