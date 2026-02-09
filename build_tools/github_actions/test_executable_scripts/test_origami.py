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

platform = os.getenv("RUNNER_OS", "linux").lower()
is_windows = platform == "windows"

origami_test_dir = Path(THEROCK_BIN_DIR).resolve() / "origami"

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
subprocess.run(cmd, cwd=THEROCK_DIR, check=True)
