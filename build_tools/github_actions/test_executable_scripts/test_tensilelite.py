#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test runner for TensileLite and rocisa Python tests using pre-built artifacts.

Runs against installed artifacts from the hipBLASLt test component:
  share/hipblaslt/tensilelite/Tensile/     — Tensile Python package
  share/hipblaslt/tensilelite/rocisa/       — rocisa Python package + _rocisa.abi3.so
  share/hipblaslt/tensilelite/rocisa_tests/ — rocisa pytest modules

Test order: rocisa first (build dependency of TensileLite), then TensileLite
unit tests. A rocisa failure means TensileLite tests will also fail.

Usage (TheRock CI):
    python test_tensilelite.py

Usage (local, after install):
    THEROCK_BIN_DIR=./build/bin python test_tensilelite.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")

rocm_path = Path(THEROCK_BIN_DIR).resolve().parent
tensilelite_root = rocm_path / "share" / "hipblaslt" / "tensilelite"

if not tensilelite_root.is_dir():
    raise FileNotFoundError(
        f"TensileLite test artifacts not found at {tensilelite_root}. "
        "Ensure the build used -DHIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS=ON."
    )

env = os.environ.copy()
existing_pythonpath = env.get("PYTHONPATH")
env["PYTHONPATH"] = (
    f"{tensilelite_root}{os.pathsep}{existing_pythonpath}"
    if existing_pythonpath
    else str(tensilelite_root)
)
env["ROCM_PATH"] = str(rocm_path)

# _rocisa links libamdhip64.so — ensure HIP libraries are findable.
lib_path = rocm_path / "lib"
existing_ld_path = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = (
    f"{lib_path}{os.pathsep}{existing_ld_path}" if existing_ld_path else str(lib_path)
)

# GPU unit tests use amdclang++ to assemble kernels.
existing_path = env.get("PATH", "")
env["PATH"] = os.pathsep.join(
    filter(
        None,
        [
            str(rocm_path / "bin"),
            str(rocm_path / "lib" / "llvm" / "bin"),
            existing_path,
        ],
    )
)

# Smoke test: verify install layout and stable ABI.
logging.info("=== Verifying artifact install layout ===")
rocisa_dir = tensilelite_root / "rocisa"
logging.info(
    f"rocisa directory contents: {[f.name for f in rocisa_dir.iterdir() if not f.name.startswith('__')]}"
)
abi3_files = list(rocisa_dir.glob("*.abi3.*"))
if abi3_files:
    logging.info(f"Stable ABI confirmed: {[f.name for f in abi3_files]}")
else:
    logging.warning("No .abi3 extension found — stable ABI may not be enabled")
subprocess.check_call(
    [
        sys.executable,
        "-c",
        "import Tensile, rocisa, rocisa.instruction; "
        "print('Tensile:', Tensile.ROOT_PATH); print('rocisa:', rocisa.__file__)",
    ],
    cwd=str(THEROCK_DIR),
    env=env,
)

# rocisa tests (includes GPU tests — runner has GPU access).
logging.info("=== Running rocisa tests ===")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(tensilelite_root / "rocisa_tests"),
    ],
    cwd=str(THEROCK_DIR),
    env=env,
)

# TensileLite Python unit tests (includes GPU subtile tests).
logging.info("=== Running TensileLite unit tests ===")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(tensilelite_root / "Tensile" / "Tests" / "unit"),
    ],
    cwd=str(THEROCK_DIR),
    env=env,
)
