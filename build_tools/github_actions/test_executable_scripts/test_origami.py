# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test script for origami tests, run via ctest.

Origami uses Catch2 for its C++ tests and pytest for its Python tests. TheRock
builds origami C++-only by default (ORIGAMI_ENABLE_PYTHON=OFF), so only the C++
suite runs unless the optional Python bindings are present in site-packages.
"""

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
if not THEROCK_BIN_DIR:
    raise RuntimeError("THEROCK_BIN_DIR environment variable is not set")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")

environ_vars = os.environ.copy()
is_windows = platform.system() == "Windows"

bin_dir = Path(THEROCK_BIN_DIR).resolve()
lib_dir = bin_dir.parent / "lib"
origami_test_dir = bin_dir / "origami"

# Python bindings are optional: TheRock builds origami C++-only by default
# (ORIGAMI_ENABLE_PYTHON=OFF), so the site-packages module may be absent.
site_packages_candidates = sorted(
    p.parent for p in lib_dir.glob("python*/site-packages/origami") if p.is_dir()
)
python_available = bool(site_packages_candidates)

# LD_LIBRARY_PATH/PATH lets the C++ test binary (and, when present, the Python
# tests) find liborigami.so.
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

# Set PYTHONPATH so Python can find the origami package in site-packages.
if python_available:
    site_packages_dir = site_packages_candidates[0]
    python_paths = [
        str(site_packages_dir),
        environ_vars.get("PYTHONPATH", ""),
    ]
    environ_vars["PYTHONPATH"] = os.pathsep.join(p for p in python_paths if p)

cmd = [
    "ctest",
    "--test-dir",
    str(origami_test_dir),
    "--output-on-failure",
    "--parallel",
    "8",
]

# Restrict to the C++ (Catch2) suite when the Python bindings are absent; the
# Windows path is always C++-only.
if is_windows or not python_available:
    cmd.extend(["-R", "origami-tests"])

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
