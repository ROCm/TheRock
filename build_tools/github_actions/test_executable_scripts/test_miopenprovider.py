# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import sys
from pathlib import Path
import platform

# Import the ctest retry helper
sys.path.append(str(Path(__file__).resolve().parent))
from ctest_retry_helper import run_ctest_with_retry

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()

logging.basicConfig(level=logging.INFO)

TEST_TO_IGNORE = {
    # TODO(#3709): Re-enable gfx110X tests once issues are resolved
    "gfx110X-all": {
        "windows": [
            "miopen_plugin_integration_tests",
        ]
    }
}

logging.basicConfig(level=logging.INFO)

# If you increase the timeout here you need to also increase the timeout for the job
# See file build_tools/github_actions/fetch_test_configurations.py and search for miopenprovider
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/miopen_plugin",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "1200",
]

if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
    cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

# Determine test filter based on TEST_TYPE environment variable
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    # Exclude tests that start with "Full" during quick tests
    environ_vars["GTEST_FILTER"] = "-Full*"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

exit_code = run_ctest_with_retry(cmd, THEROCK_DIR, environ_vars)
sys.exit(exit_code)
