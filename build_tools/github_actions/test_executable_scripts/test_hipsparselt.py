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
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import run_test

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

test_filter = []
if test_type == "smoke":
    test_filter.append("--gtest_filter=*smoke*")
elif test_type == "full":
    test_filter.append("--gtest_filter=*quick*")

# Create temp file for JSON output
gtest_json_path = Path(tempfile.gettempdir()) / "hipsparselt_test_results.json"

cmd = [
    f"{THEROCK_BIN_DIR}/hipsparselt-test",
    f"--gtest_output=json:{gtest_json_path}",
] + test_filter

run_test(
    cmd,
    output_format="gtest",
    output_path=gtest_json_path,
    cwd=THEROCK_DIR,
    env=environ_vars,
)
