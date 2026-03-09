# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
import platform

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import run_test

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

# Create temp file for JUnit XML output
junit_xml_path = Path(tempfile.gettempdir()) / "miopenprovider_test_results.xml"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/miopen_plugin",
    "--output-on-failure",
    "--output-junit",
    str(junit_xml_path),
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

if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"

run_test(
    cmd,
    output_format="ctest",
    output_path=junit_xml_path,
    cwd=THEROCK_DIR,
    env=environ_vars,
)
