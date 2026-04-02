# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import sys
from pathlib import Path

# Import the ctest retry helper
sys.path.append(str(Path(__file__).resolve().parent))
from ctest_retry_helper import run_ctest_with_retry

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipblaslt_plugin",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "600",
]

# Determine test filter based on TEST_TYPE environment variable
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    # Exclude tests that start with "Full" during quick tests
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

exit_code = run_ctest_with_retry(cmd, THEROCK_DIR, environ_vars)
sys.exit(exit_code)
