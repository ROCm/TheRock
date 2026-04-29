# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")
ROCPROFSYS_TEST_DIR = THEROCK_PATH / "share" / "rocprofiler-systems" / "tests"

# These tests are always excluded until the relevant issue is fixed
EXCLUDED_TESTS = [
    "transferbench-sys-run",  # Requires access to multi-gpu system
    "fork.*",  # May deadlock - Under investigation
]

# Excluded by default
EXCLUDED_LABELS = [
    "mpi",  # TODO: Allow the example binaries to be built with MPI
    "julia",  # Unsupported - Not present in TheRock
    "kfd",  # No SDK version can be found yet
    "attach",  # Fails - Under investigation
    "lulesh",  # Unsupported - Lulesh fails to build on TheRock
    "network",  # NIC unsupported
    "overflow",  # Requires CAPSYS_ADMIN/PERFMON or perf_event_paranoid <= 3
    "fork",  # only the runtime instrument fails, but logs are VERY long
    "pthreads",  # All fail
]

# Limited to 15 minutes
QUICK_TESTS_REGEX = [
    "transpose.*",
    "rocprofiler-systems.*",  # Binary tests
    "config.*",
    "jpegdecode.*",  # TODO: Binary is not built, so test is skipped
    "videodeocde.*",  # TODO: Binary is not built, so test is skipped
    "openmp.*",
    "roctx.*",
    "trace-time-window.*",
]

QUICK_TEST_EXCLUDE_REGEX = [
    "openmp-target.*",  # Requires _omp_dm_init_kernel.kd fix
    "roctx-sampling",  # Need to lower # samples in rocpd validation
]

logging.basicConfig(level=logging.INFO)

environ_vars = os.environ.copy()


def setup_env():
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
    environ_vars["ROCPROFSYS_INSTALL_DIR"] = str(THEROCK_PATH)
    environ_vars["ROCPROFSYS_MAX_THREADS"] = "64"

    old_path = os.getenv("PATH", "")
    rocm_bin = str(THEROCK_BIN_PATH)
    environ_vars["PATH"] = f"{rocm_bin}:{old_path}" if old_path else rocm_bin

    ld_paths = [
        str(THEROCK_PATH / "share" / "rocprofiler-systems" / "examples" / "lib"),
    ]
    ld_paths_str = ":".join(ld_paths)
    old_ld_path = os.getenv("LD_LIBRARY_PATH", "")
    environ_vars["LD_LIBRARY_PATH"] = (
        f"{ld_paths_str}:{old_ld_path}" if old_ld_path else ld_paths_str
    )


def execute_tests():
    shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
    total_shards = int(os.getenv("TOTAL_SHARDS", "1"))
    test_type = os.getenv("TEST_TYPE", "full").lower()

    ctest_base = [
        "ctest",
        "--test-dir",
        str(ROCPROFSYS_TEST_DIR),
    ]

    # Informational test
    config_cmd = ctest_base + [
        "--verbose",
        "--tests-regex",
        "rocprofiler-systems-pytest-config",
    ]
    logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(config_cmd)}")
    subprocess.run(config_cmd, cwd=THEROCK_PATH, check=False, env=environ_vars)

    # Actual tests
    # Cap passed/skipped output to ~8 KiB (tail) so the SKIPPED reason and final
    # lines survive without exploding CI log size.
    excluded_tests = list(EXCLUDED_TESTS)
    if test_type == "quick":
        excluded_tests.extend(QUICK_TEST_EXCLUDE_REGEX)

    cmd = ctest_base + [
        "--verbose",
        "--test-output-size-passed",
        str(8 * 1024),
        "--test-output-truncation",
        "tail",
        "--exclude-regex",
        f"{'|'.join(excluded_tests)}",
        "--label-exclude",
        f"{'|'.join(EXCLUDED_LABELS)}",
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]
    if test_type == "quick":
        cmd.extend(["--tests-regex", "|".join(QUICK_TESTS_REGEX)])

    logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_PATH, check=True, env=environ_vars)


if __name__ == "__main__":
    setup_env()
    execute_tests()
