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

# Resolve paths
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")
ROCPROFILER_COMPUTE_DIRECTORY = THEROCK_PATH / "libexec" / "rocprofiler-compute"

# Set up excluded tests
EXCLUDED_TESTS = [
    "test_profile_live_attach_detach",
]

# quick Tests
QUICK_TESTS = [
    "test_autogen_config",
    "test_utils",
    "test_num_xcds_cli_output",
    "test_num_xcds_spec_class",
    "test_L1_cache_counters",
    "test_analyze_workloads",
    "test_analyze_commands",
    "test_metric_validation",
    "test_profile_iteration_multiplexing_1",
]

environ_vars = os.environ.copy()


def setup_env():
    # Set up ROCM_PATH
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)

    # Set up PATH
    old_path = os.getenv("PATH", "")
    rocm_bin = str(THEROCK_BIN_PATH)
    if old_path:
        environ_vars["PATH"] = f"{rocm_bin}:{old_path}"
    else:
        environ_vars["PATH"] = rocm_bin

    # Set up LD_LIBRARY_PATH
    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
    sysdeps_path = f"{THEROCK_LIB_PATH}/rocm_sysdeps/lib"
    if old_ld_lib_path:
        environ_vars["LD_LIBRARY_PATH"] = (
            f"{THEROCK_LIB_PATH}:{sysdeps_path}:{old_ld_lib_path}"
        )
    else:
        environ_vars["LD_LIBRARY_PATH"] = f"{THEROCK_LIB_PATH}:{sysdeps_path}"


def execute_tests():
    # Sharding
    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

    # Run tests
    cmd = [
        "ctest",
        "--test-dir",
        f"{ROCPROFILER_COMPUTE_DIRECTORY}",
        "--output-on-failure",
        "--verbose",
        "--exclude-regex",
        f"{"|".join(EXCLUDED_TESTS)}",
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]

    # If quick tests are enabled, we run quick tests only.
    # Otherwise, we run the normal test suite
    test_type = os.getenv("TEST_TYPE", "full")
    if test_type == "quick":
        cmd.append("--tests-regex")
        cmd.append("|".join(QUICK_TESTS))

    logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=THEROCK_PATH,
        check=True,
        env=environ_vars,
    )


if __name__ == "__main__":
    setup_env()

    AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
    test_type = os.getenv("TEST_TYPE", "full")
    shard_index = int(os.getenv("SHARD_INDEX", "1"))
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

    if _has_test_filter_utils:
        logging.info("Using ctest label-based filtering via test_filter_utils")
        sys.exit(
            run_ctest(
                test_dir=str(ROCPROFILER_COMPUTE_DIRECTORY),
                env=environ_vars,
                cwd=str(THEROCK_PATH),
                test_type=test_type,
                amdgpu_families=AMDGPU_FAMILIES,
                shard_index=shard_index,
                total_shards=total_shards,
            )
        )

    # Fallback: use existing ctest logic when test_filter_utils is not available
    logging.info("test_filter_utils not available, falling back to existing ctest logic")
    execute_tests()
