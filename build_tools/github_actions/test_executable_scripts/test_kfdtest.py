# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path
import platform

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()

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

# kfdtest should be run via run_kfdtest.sh script which handles
# platform-specific test exclusions from kfdtest.exclude
cmd = ["./run_kfdtest.sh"]

# Map AMDGPU_FAMILIES to platform names used in kfdtest.exclude
# The script auto-detects the platform from /sys/class/kfd/kfd/topology/nodes/*/name
# but we can override with -p if needed for testing
PLATFORM_MAP = {
    "gfx90a": "gfx90a",
    "gfx940": "gfx942",
    "gfx941": "gfx942",
    "gfx942": "gfx942",
    "gfx950-dcgpu": "gfx950",
    "gfx1100": "gfx1100",
    "gfx1101": "gfx1101",
    "gfx1102": "gfx1102",
    "gfx1103": "gfx1103",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
}

# Additional test exclusions beyond kfdtest.exclude
# These are tests that are known to be flaky or problematic in CI
ADDITIONAL_EXCLUDE = {
    "gfx90a": {
        "linux": [
            "KFDSVMRangeTest.HMMProfilingEvent*",
        ]
    },
    "gfx94X-dcgpu": {
        "linux": [
            "KFDSVMRangeTest.HMMProfilingEvent*",
        ]
    },
    "gfx942": {
        "linux": [
            "KFDSVMRangeTest.HMMProfilingEvent*",
        ]
    },
    "gfx950-dcgpu": {
        "linux": [
            "KFDSVMRangeTest.HMMProfilingEvent*",
        ]
    },
}

# Build additional exclude filter
exclude_tests = []
if AMDGPU_FAMILIES in ADDITIONAL_EXCLUDE and os_type in ADDITIONAL_EXCLUDE[AMDGPU_FAMILIES]:
    exclude_tests = ADDITIONAL_EXCLUDE[AMDGPU_FAMILIES][os_type]

if exclude_tests:
    # Pass additional exclusions to run_kfdtest.sh via -e flag
    exclude_filter = ":".join(exclude_tests)
    cmd.extend(["-e", exclude_filter])

# Check if quick tests are requested
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    # For quick tests, use the core test suite which is a minimal set
    # defined in kfdtest.exclude FILTER[core]
    cmd.extend(["-p", "core"])

# Pass through any gtest sharding args
gtest_args = []
if int(TOTAL_SHARDS) > 1:
    gtest_args.append(f"--gtest_total_shards={TOTAL_SHARDS}")
    gtest_args.append(f"--gtest_shard_index={int(SHARD_INDEX) - 1}")

if gtest_args:
    cmd.extend(gtest_args)

logging.info(f"++ Exec [{cwd_dir}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=cwd_dir, check=True, env=environ_vars)
