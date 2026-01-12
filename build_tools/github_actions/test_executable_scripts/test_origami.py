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

# Set up library and Python paths for finding liborigami.so and the Python module
bin_dir = Path(THEROCK_BIN_DIR)
lib_dir = bin_dir.parent / "lib"
origami_test_dir = bin_dir / "origami"

# Build LD_LIBRARY_PATH with multiple possible locations
if platform == "linux":
    ld_paths = [
        str(lib_dir),                    # Main lib directory (./build/lib)
        str(origami_test_dir),           # Origami test directory (./build/bin/origami)
        environ_vars.get("LD_LIBRARY_PATH", ""),
    ]
    # Filter empty paths and join
    environ_vars["LD_LIBRARY_PATH"] = ":".join(p for p in ld_paths if p)

# Set PYTHONPATH to help Python find the origami module
python_paths = [
    str(origami_test_dir),               # Where origami Python module is staged
    environ_vars.get("PYTHONPATH", ""),
]
environ_vars["PYTHONPATH"] = ":".join(p for p in python_paths if p)

logging.info(f"LD_LIBRARY_PATH: {environ_vars.get('LD_LIBRARY_PATH', '')}")
logging.info(f"PYTHONPATH: {environ_vars.get('PYTHONPATH', '')}")

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
