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
is_windows = platform == "windows"

bin_dir = Path(THEROCK_BIN_DIR).resolve()
lib_dir = bin_dir.parent / "lib"
origami_test_dir = bin_dir / "origami"

# Path separator is different on Windows vs Linux
path_sep = ";" if is_windows else ":"

# LD_LIBRARY_PATH is needed for Python tests to find liborigami.so
if platform == "linux":
    ld_paths = [
        str(lib_dir),
        str(origami_test_dir),
        environ_vars.get("LD_LIBRARY_PATH", ""),
    ]
    environ_vars["LD_LIBRARY_PATH"] = path_sep.join(p for p in ld_paths if p)
elif is_windows:
    dll_paths = [
        str(bin_dir),
        str(lib_dir),
        str(origami_test_dir),
        environ_vars.get("PATH", ""),
    ]
    environ_vars["PATH"] = path_sep.join(p for p in dll_paths if p)

# Set PYTHONPATH to help Python find the origami module
python_paths = [
    str(origami_test_dir),  # Where origami Python module is staged
    environ_vars.get("PYTHONPATH", ""),
]
environ_vars["PYTHONPATH"] = path_sep.join(p for p in python_paths if p)

logging.info(f"LD_LIBRARY_PATH: {environ_vars.get('LD_LIBRARY_PATH', '')}")
logging.info(f"PYTHONPATH: {environ_vars.get('PYTHONPATH', '')}")

# CTest runs both C++ (Catch2) tests and Python (pytest) tests
cmd = [
    "ctest",
    "--test-dir",
    str(origami_test_dir),
    "--output-on-failure",
    "--parallel",
    "8",
]

if is_windows:
    cmd.extend(["-R", "origami-tests"])

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
