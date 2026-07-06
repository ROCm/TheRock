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

site_packages_candidates = sorted(
    p.parent for p in lib_dir.glob("python*/site-packages/origami") if p.is_dir()
)
if not site_packages_candidates:
    raise RuntimeError(
        f"origami package not found under {lib_dir}/python*/site-packages -- "
        "was origami built and installed?"
    )
site_packages_dir = site_packages_candidates[0]

# LD_LIBRARY_PATH is needed for Python tests to find liborigami.so.
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

if is_windows:
    cmd.extend(["-R", "origami-tests"])

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
