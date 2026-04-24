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
cmd = ["./rocrtst64"]

# TODO(#3851): Excluded tests (flaky or disabled in CI).
TEST_TO_IGNORE = {
    "gfx90a": {
        "linux": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx94X-dcgpu": {
        "linux": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx950-dcgpu": {
        "linux": [
            "rocrtstFunc.GpuCoreDump_DefaultPattern",
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx110X-all": {
        "windows": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
    "gfx1151": {
        "windows": [
            "rocrtstFunc.Memory_Max_Mem",
        ]
    },
}

# If quick tests are enabled, run quick tests only. Otherwise, run the full suite.
QUICK_TESTS = [
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

exclude_filter = "-"
if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
    exclude_filter += ":".join(ignored_tests)

test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    environ_vars["GTEST_FILTER"] = ":".join(QUICK_TESTS) + ":" + exclude_filter
else:
    environ_vars["GTEST_FILTER"] = exclude_filter

logging.info(f"++ Exec [{cwd_dir}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=cwd_dir, check=True, env=environ_vars)
