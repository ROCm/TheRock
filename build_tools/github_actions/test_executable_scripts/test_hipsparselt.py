# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from test_filter_utils import run_ctest

    _has_test_filter_utils = True
except ImportError:
    _has_test_filter_utils = False

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
test_type = os.getenv("TEST_TYPE", "full")

if _has_test_filter_utils:
    logging.info("Using ctest label-based filtering via test_filter_utils")
    sys.exit(
        run_ctest(
            test_dir=str(Path(THEROCK_BIN_DIR) / "hipsparselt"),
            env=environ_vars,
            cwd=str(THEROCK_DIR),
            test_type=test_type,
            amdgpu_families=AMDGPU_FAMILIES,
            shard_index=int(SHARD_INDEX),
            total_shards=int(TOTAL_SHARDS),
        )
    )

# Fallback: use hipsparselt-test when test_filter_utils is not available
logging.info("test_filter_utils not available, falling back to hipsparselt-test")
test_filter = []
if test_type == "quick":
    test_filter.append("--gtest_filter=*smoke*")
else:
    test_filter.append("--gtest_filter=*quick*")

cmd = [f"{THEROCK_BIN_DIR}/hipsparselt-test"] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
