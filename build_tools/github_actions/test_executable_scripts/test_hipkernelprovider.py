# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

environ_vars = os.environ.copy()
# Some of our runtime kernel compilations have been relying on either ROCM_PATH being set, or ROCm being installed at
# /opt/rocm. Neither of these is true in TheRock so we need to supply ROCM_PATH to our tests.
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

logging.basicConfig(level=logging.INFO)

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hip_kernel_provider",
    "--output-on-failure",
    "--parallel",
    "8",
]

# Determine test filter based on TEST_TYPE environment variable
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)
