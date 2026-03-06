# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import output_failed_tests, parse_ctest_junit_xml

logging.basicConfig(level=logging.INFO)

# Create temp file for JUnit XML output
junit_xml_path = Path(tempfile.gettempdir()) / "hipdnn_samples_test_results.xml"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipdnn_samples",
    "--output-on-failure",
    "--output-junit",
    str(junit_xml_path),
    "--parallel",
    "8",
    "--timeout",
    "60",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

result = subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=False,
)

# Parse and output failed tests
failed_tests = parse_ctest_junit_xml(junit_xml_path)
output_failed_tests(failed_tests)

# Exit with the original return code
sys.exit(result.returncode)
