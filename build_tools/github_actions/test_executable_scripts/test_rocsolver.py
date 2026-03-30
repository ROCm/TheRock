# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

import sys
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_api import is_host_asan

if is_host_asan():
    print(
        "SKIP: host-ASAN build (THEROCK_SANITIZER=HOST_ASAN) does not produce "
        "GPU math library test binaries. Run with linux-release-asan for device tests."
    )
    sys.exit(0)

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

cmd = [
    f"{THEROCK_BIN_DIR}/rocsolver-test",
]

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
# Test filter patterns retrieved from https://github.com/ROCm/rocm-libraries/blob/a18b17eef6c24bcd4bcf8dd6a0e36325cbcd11a7/projects/rocsolver/rtest.xml
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    quick_tests = [
        "checkin*BDSQR*",
        "checkin*STEBZ*",
        "checkin*STEIN*",
        "checkin*STERF*",
        "checkin*STEQR*",
        "checkin*SYEVJ*",
        "checkin*HEEVJ*",
        "checkin*LARFG*",
        "checkin*LARF*",
        "checkin*LARFT*",
        "checkin*GETF2*",
        "checkin*POTF2*",
        "checkin*GEQR2*",
        "checkin*GELQ2*",
        "checkin*SPLITLU*",
        "checkin*REFACTLU*",
        "checkin*REFACTCHOL*",
    ]
    cmd.extend([f"--gtest_filter={':'.join(quick_tests)}-*LARFB*:*known_bug*"])
else:
    cmd.extend(
        ["--gtest_filter=checkin*-*known_bug*:checkin_lapack/SYGVDX_INPLACE.__float/41"]
    )

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
