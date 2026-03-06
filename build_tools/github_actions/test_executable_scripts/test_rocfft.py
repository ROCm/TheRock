# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import output_failed_tests, parse_gtest_json

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    test_filter = ["--smoketest"]
else:
    # "--test_prob" is the probability that a given test will run.
    # Due to the large number of tests for rocFFT, we only run a subset.
    test_filter = [
        "--gtest_filter=-*multi_gpu*",
        "--test_prob",
        "0.02",
    ]

# Create temp file for JSON output
gtest_json_path = Path(tempfile.gettempdir()) / "rocfft_test_results.json"

cmd = [
    f"{THEROCK_BIN_DIR}/rocfft-test",
    f"--gtest_output=json:{gtest_json_path}",
] + test_filter
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
result = subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=False,
)

# Parse and output failed tests
failed_tests = parse_gtest_json(gtest_json_path)
output_failed_tests(failed_tests)

# Exit with the original return code
sys.exit(result.returncode)
