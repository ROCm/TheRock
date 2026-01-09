#!/usr/bin/env python3
"""
Test script for origami C++ and Python tests.

Origami uses Catch2 for C++ tests and pytest for Python tests.
Both test types are registered with CTest and run via ctest command.
"""

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Environment setup
environ_vars = os.environ.copy()
platform = os.getenv("RUNNER_OS", "linux").lower()

# Test type configuration (smoke, full)
test_type = os.getenv("TEST_TYPE", "full")

# Build the ctest command
# CTest runs both C++ (Catch2) tests and Python (pytest) tests
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/origami",
    "--output-on-failure",
    "--timeout",
    "300",
]

# Test filtering based on test type
if test_type == "smoke":
    # Run only a subset of tests for smoke testing
    # Use CTest's regex to filter test names
    cmd.extend(["-R", "origami_python|GEMM:.*compute"])
else:
    # For"full" test type
    pass

cmd.append("--parallel")

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
