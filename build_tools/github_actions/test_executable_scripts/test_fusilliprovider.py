# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import run_test

logging.basicConfig(level=logging.INFO)

# Create temp file for JUnit XML output
junit_xml_path = Path(tempfile.gettempdir()) / "fusilliprovider_test_results.xml"

# Build the ctest command
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/fusilli_plugin_test_infra",
    "--output-on-failure",
    "--output-junit",
    str(junit_xml_path),
    "--parallel",
    "8",
    "--timeout",
    "600",
]

# Set up environment variables
environ_vars = os.environ.copy()

# Determine test filter based on TEST_TYPE environment variable
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    # Exclude tests that start with "Full" during smoke tests
    environ_vars["GTEST_FILTER"] = "-Full*"

# As a sanity check, verify libIREECompiler.so is available in the build artifacts.
# TODO: check for .dll on windows
iree_compiler_lib = THEROCK_BIN_DIR.parent / "lib" / "libIREECompiler.so"
if not iree_compiler_lib.exists():
    raise RuntimeError(
        f"libIREECompiler.so not found at {iree_compiler_lib}. "
        "Ensure THEROCK_ENABLE_IREE_COMPILER is ON and iree-compiler is built."
    )
logging.info(f"Verified libIREECompiler.so available at: {iree_compiler_lib}")

# Add THEROCK_BIN_DIR to PATH for rocm_agent_enumerator
environ_vars["PATH"] = f"{THEROCK_BIN_DIR}:{environ_vars['PATH']}"

# Run the tests
if test_type == "smoke":
    logging.info("   TEST_TYPE=smoke: Excluding Full* tests via GTEST_FILTER")

run_test(
    cmd, output_format="ctest", output_path=junit_xml_path, cwd=THEROCK_DIR, env=environ_vars
)
