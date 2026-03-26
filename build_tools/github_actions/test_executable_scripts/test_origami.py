# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test script for origami C++ and Python tests.

Origami uses Catch2 for C++ tests and pytest for Python tests.
Both test types are registered with CTest and run via ctest command.
"""

import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
if not THEROCK_BIN_DIR:
    raise RuntimeError("THEROCK_BIN_DIR environment variable is not set")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Environment setup
environ_vars = os.environ.copy()
is_windows = platform.system() == "Windows"

bin_dir = Path(THEROCK_BIN_DIR).resolve()
lib_dir = bin_dir.parent / "lib"
origami_test_dir = bin_dir / "origami"

# The origami Python package is installed to lib/pythonX.Y/site-packages/origami/
# Glob for the actual pythonX.Y directory to be robust across Python versions.
python_dirs = sorted(lib_dir.glob("python*/site-packages"))
if python_dirs:
    site_packages_dir = python_dirs[-1]
else:
    raise RuntimeError(f"No site-packages directory found under {lib_dir}")

# LD_LIBRARY_PATH is needed for Python tests to find liborigami.so
if not is_windows:
    ld_paths = [
        str(lib_dir),
        environ_vars.get("LD_LIBRARY_PATH", ""),
    ]
    environ_vars["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in ld_paths if p)
else:
    dll_paths = [
        str(bin_dir),
        str(lib_dir),
        environ_vars.get("PATH", ""),
    ]
    environ_vars["PATH"] = os.pathsep.join(p for p in dll_paths if p)

# Set PYTHONPATH so Python can find the origami package in site-packages
python_paths = [
    str(site_packages_dir),
    environ_vars.get("PYTHONPATH", ""),
]
environ_vars["PYTHONPATH"] = os.pathsep.join(p for p in python_paths if p)

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
