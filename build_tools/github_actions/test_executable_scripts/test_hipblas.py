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

# Importing utilities from github_actions_utils.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import is_asan, run_test

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

if is_asan():
    environ_vars["HSA_XNACK"] = "1"

logging.basicConfig(level=logging.INFO)

tests_to_exclude = [
    "*known_bug*",
    "_/getrs*",
    "_/getri_batched.solver*",
    "_/gels_batched.solver*",
]

exclusion_list = ":".join(tests_to_exclude)

# Create temp file for JSON output
gtest_json_path = Path(tempfile.gettempdir()) / "hipblas_test_results.json"

cmd = [
    f"{THEROCK_BIN_DIR}/hipblas-test",
    f"--gtest_output=json:{gtest_json_path}",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    cmd += [
        "--yaml",
        f"{THEROCK_BIN_DIR}/hipblas_smoke.yaml",
        f"--gtest_filter=-{exclusion_list}",
    ]
else:
    # TODO(#2616): Enable full tests once known machine issues are resolved
    cmd += [f"--gtest_filter=*pre_checkin*-{exclusion_list}"]


run_test(
    cmd,
    output_format="gtest",
    output_path=gtest_json_path,
    cwd=THEROCK_DIR,
    env=environ_vars,
    success_returncodes=[0, 3],
)
