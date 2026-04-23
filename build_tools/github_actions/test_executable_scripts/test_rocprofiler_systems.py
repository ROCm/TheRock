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

# TODO: Once the corresponding tests are fixed, remove from this list
EXCLUDED_TESTS = [
    "openmp-target",  # Validation test fails (__omp_dm_init_kernel.kd)
    # "transpose-runtime-instrument",  # Requires runtime-instrument optimization PR
    "transferbench-sys-run",  # Requires multi-gpu system
    # "fork.*",  # Requires runtime-instrument optimization PR, or deadlocks because of logger
    "scratch.*",  # Validation test fails (GPU events need to be reduced to 1 instead of 12)
    # "roctx-sampling",  # Requires less strict sampling requirements
    "roctx-runtime-instrument",  # Requires runtime-instrument optimization PR
]

EXCLUDED_LABELS = [
    "mpi",  # Currently unsupported
    "julia",  # Unsupported
    "kfd",  # No SDK version can be found yet
    "attach",  # Fails
]

logging.basicConfig(level=logging.INFO)

environ_vars = os.environ.copy()


def setup_env():
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
    environ_vars["ROCPROFSYS_INSTALL_DIR"] = str(THEROCK_PATH)

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
    cmd = ctest_base + [
        "--output-on-failure",
        "--exclude-regex",
        f"{'|'.join(EXCLUDED_TESTS)}",
        "--label-exclude",
        f"{'|'.join(EXCLUDED_LABELS)}",
        "--tests-information",
        f"{shard_index},,{total_shards}",
    ]

    logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_PATH, check=True, env=environ_vars)


if __name__ == "__main__":
    setup_env()
    execute_tests()
