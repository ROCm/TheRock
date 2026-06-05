# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

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
elif test_type == "full":
    # BISECT WORKAROUND: the rocsolver regression under bisect causes ~20-100x
    # slowdown in daily_lapack/* eigenvalue suites, exceeding the 120-minute
    # shard 2/3 budget. Replace the full-shard run with a focused subset that
    #   (a) probes the 11 baseline-failing tests (GESDD/HEEVD/HEEVD_HYBRID on
    #       double_complex), and
    #   (b) samples three slow daily_lapack suites for timing comparison.
    # Only shard 2 runs the focused subset; shards 1 and 3 exit immediately
    # as a no-op so we don't waste runner time re-running the same probe.
    if str(SHARD_INDEX) != "2":
        logging.info(
            f"BISECT WORKAROUND: skipping shard {SHARD_INDEX}/{TOTAL_SHARDS} "
            f"(only shard 2 runs the focused rocsolver subset)"
        )
        sys.exit(0)
    envion_vars["GTEST_SHARD_INDEX"] = "0"
    envion_vars["GTEST_TOTAL_SHARDS"] = "1"
    bisect_filter = ":".join(
        [
            "daily_lapack/GESDD.*double_complex*",
            "daily_lapack/HEEVD.*double_complex*",
            "daily_lapack/HEEVD_HYBRID.*double_complex*",
            "daily_lapack/HEEVDX.*",
            "daily_lapack/HEEVDJ.*",
            "daily_lapack/SYEVDJ.*",
        ]
    )
    cmd.extend([f"--gtest_filter={bisect_filter}"])
    logging.info(f"BISECT WORKAROUND: running focused rocsolver subset: {bisect_filter}")

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
