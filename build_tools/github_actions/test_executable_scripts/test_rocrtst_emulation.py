# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the rocrtst runtime test suite under the mirage + rocjitsu GPU emulator.

This is the emulated counterpart of ``test_rocrtst.py``. Instead of executing
``rocrtst64`` against a physical GPU, it wraps the same binary with ``mirage
run`` so the tests execute on top of the ``rocjitsu`` CPU emulator. That lets
the suite run on GPU-less ``rocjitsu-cpu`` runners.

Environment variables (set by the test workflow):
  THEROCK_BIN_DIR: Directory containing ``rocrtst64`` and the ``mirage`` binary.
  AMDGPU_FAMILIES: GPU family under test; selects the matching mirage profile.
  SHARD_INDEX / TOTAL_SHARDS: GTest sharding (1-indexed shard for display).
  EMULATION_TEST_TYPE: Which rocrtst subset to run under emulation. Defaults to
    "quick" (the reduced QUICK_TESTS set) because CPU emulation is much slower
    than native execution. Set to "full" to run the entire suite. This is
    independent of the global TEST_TYPE so emulation stays fast by default while
    remaining configurable per run.
"""

import logging
import os
import shlex
import subprocess
import sys
import platform
from pathlib import Path

# Allow importing the shared emulation helpers regardless of the current
# working directory (the test runs with cwd set to THEROCK_BIN_DIR).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulation_utils import build_mirage_run_command, select_mirage_profile

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)

# TODO(#3851): Excluded tests (flaky or disabled in CI). Mirrors test_rocrtst.py.
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
}

# Reduced set run by default (non-"full"). Emulation is slow, so keep the default
# CI footprint small; set EMULATION_TEST_TYPE=full to run the entire suite.
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


def main():
    if not THEROCK_BIN_DIR:
        raise EnvironmentError("THEROCK_BIN_DIR is not set")

    # rocjitsu only emulates specific agents. If this family has no matching
    # emulator profile, skip rather than run against a mismatched agent.
    profile = select_mirage_profile(AMDGPU_FAMILIES)
    if profile is None:
        logging.warning(
            "Skipping rocrtst emulation: no rocjitsu profile is available for "
            "AMDGPU family '%s'.",
            AMDGPU_FAMILIES,
        )
        return

    exclude_filter = "-"
    if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
        ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
        exclude_filter += ":".join(ignored_tests)

    # Which subset to run. Configurable via EMULATION_TEST_TYPE and defaulting
    # to the reduced (non-full) set so emulation runs stay fast by default.
    emulation_test_type = os.getenv("EMULATION_TEST_TYPE", "quick").lower()
    if emulation_test_type == "full":
        gtest_filter = exclude_filter
    else:
        gtest_filter = ":".join(QUICK_TESTS) + ":" + exclude_filter

    # Forward GTest configuration into the emulated process. For display
    # purposes the shard array is 1-indexed; GTest expects a 0-indexed shard.
    passthrough_env = {
        "GTEST_FILTER": gtest_filter,
        "GTEST_SHARD_INDEX": str(int(SHARD_INDEX) - 1),
        "GTEST_TOTAL_SHARDS": str(TOTAL_SHARDS),
    }

    cwd_dir = Path(THEROCK_BIN_DIR)
    cmd = build_mirage_run_command(
        ["./rocrtst64"],
        profile=profile,
        passthrough_env=passthrough_env,
        bin_dir=THEROCK_BIN_DIR,
    )

    # rocjitsu locates the ROCm runtime via ROCM_HOME. The artifacts are
    # installed one level above THEROCK_BIN_DIR (e.g. build/bin -> build).
    run_env = os.environ.copy()
    run_env.setdefault("ROCM_HOME", str(cwd_dir.resolve().parent))

    logging.info(
        "++ Emulating rocrtst on profile '%s' [%s]$ %s",
        profile,
        cwd_dir,
        shlex.join(cmd),
    )
    subprocess.run(cmd, cwd=cwd_dir, check=True, env=run_env)


if __name__ == "__main__":
    main()
