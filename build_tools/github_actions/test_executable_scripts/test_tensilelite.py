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

# Install dependencies from the artifact's requirements.txt.
requirements_txt = tensilelite_root / "requirements.txt"
if requirements_txt.is_file():
    logging.info("=== Installing test dependencies ===")
    subprocess.run(
        ["uv", "pip", "install", "-r", str(requirements_txt)],
        check=True,
        env=env,
    )

# Smoke test: verify install layout allows single-PYTHONPATH imports.
logging.info("=== Verifying artifact install layout ===")
subprocess.run(
    [
        sys.executable,
        "-c",
        "import Tensile, rocisa, rocisa.instruction; "
        "print(Tensile.ROOT_PATH); print(rocisa.__file__)",
    ],
    check=True,
    cwd=str(THEROCK_DIR),
    env=env,
)

# rocisa tests (includes GPU tests — runner has GPU access).
logging.info("=== Running rocisa tests ===")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(tensilelite_root / "rocisa_tests"),
    ],
    check=True,
    cwd=str(THEROCK_DIR),
    env=env,
)

# TensileLite Python unit tests (includes GPU subtile tests).
logging.info("=== Running TensileLite unit tests ===")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        str(tensilelite_root / "Tensile" / "Tests" / "unit"),
    ],
    check=True,
    cwd=str(THEROCK_DIR),
    env=env,
)
