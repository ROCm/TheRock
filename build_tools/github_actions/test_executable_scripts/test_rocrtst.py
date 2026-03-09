# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
from github_actions_utils import run_test

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed.
# For shard indexes, we convert to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

cwd_dir = Path(THEROCK_BIN_DIR)

# Create temp file for JSON output
gtest_json_path = Path(tempfile.gettempdir()) / "rocrtst_test_results.json"

cmd = ["./rocrtst64", f"--gtest_output=json:{gtest_json_path}"]

# Excluded tests (flaky or disabled in CI).
EXCLUDED_TESTS = [
    "-rocrtstFunc.Memory_Max_Mem",
]

# If smoke tests are enabled, run smoke tests only. Otherwise, run the full suite.
SMOKE_TESTS = [
    "rocrtst.Test_Example",
    "rocrtstFunc.MemoryAccessTests",
    "rocrtstFunc.GroupMemoryAllocationTest",
    "rocrtstFunc.MemoryAllocateAndFreeTest",
    "rocrtstFunc.Memory_Alignment_Test",
    "rocrtstFunc.Concurrent_Init_Test",
    "rocrtstFunc.Concurrent_Init_Shutdown_Test",
    "rocrtstFunc.Reference_Count",
    "rocrtstFunc.Signal_Create_Concurrently",
    "rocrtstFunc.Signal_Destroy_Concurrently",
    "rocrtstFunc.IPC",
    "rocrtstFunc.AgentProp_UUID",
    "rocrtstFunc.Deallocation_Notifier_Test",
    "rocrtstFunc.Memory_Atomic_Add_Test",
    "rocrtstFunc.Memory_Atomic_Xchg_Test",
]
test_type = os.getenv("TEST_TYPE", "full")
exclude_filter = ":".join(EXCLUDED_TESTS)

if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS) + ":" + exclude_filter
else:
    environ_vars["GTEST_FILTER"] = exclude_filter

run_test(
    cmd, output_format="gtest", output_path=gtest_json_path, cwd=cwd_dir, env=environ_vars
)
