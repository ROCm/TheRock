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

origami_test_dir = Path(THEROCK_BIN_DIR).resolve() / "origami"

# Path separator is different on Windows vs Linux
path_sep = ";" if platform == "windows" else ":"

# Set PYTHONPATH to help Python find the origami module
python_paths = [
    str(origami_test_dir),               # Where origami Python module is staged
    environ_vars.get("PYTHONPATH", ""),
]
environ_vars["PYTHONPATH"] = path_sep.join(p for p in python_paths if p)

logging.info(f"PYTHONPATH: {environ_vars.get('PYTHONPATH', '')}")

# Test type configuration (smoke, full)
test_type = os.getenv("TEST_TYPE", "full")

# CTest runs both C++ (Catch2) tests and Python (pytest) tests
cmd = [
    "ctest",
    "--test-dir",
    str(origami_test_dir),
    "--output-on-failure",
    "--parallel",
    "8",
]

if platform == "windows":
    cmd.extend(["-R", "origami-tests"])
elif test_type == "smoke":
    cmd.extend(["-R", "origami_python|GEMM:.*compute"])

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
