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

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Repository and directory setup
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
platform = os.getenv("RUNNER_OS", "linux").lower()

# Environment setup
env = os.environ.copy()

# Test type configuration (smoke, quick, full)
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()

# Find the origami build directory
BUILD_DIR = Path(os.getenv("THEROCK_BUILD_DIR", THEROCK_DIR / "build"))
ORIGAMI_BUILD_DIR = BUILD_DIR / "math-libs" / "BLAS" / "Origami" / "build"

if not ORIGAMI_BUILD_DIR.exists():
    raise FileNotFoundError(f"Origami build directory not found: {ORIGAMI_BUILD_DIR}")

# Runtime library paths for Linux
if platform == "linux":
    THEROCK_DIST_DIR = BUILD_DIR / "core" / "clr" / "dist"
    ld_parts = [
        str(THEROCK_DIST_DIR / "lib"),
        str(THEROCK_DIST_DIR / "lib64"),
        str(THEROCK_DIST_DIR / "lib" / "llvm" / "lib"),
        # Origami library paths
        str(ORIGAMI_BUILD_DIR),
        str(BUILD_DIR / "math-libs" / "BLAS" / "Origami" / "stage" / "lib"),
    ]
    # De-duplicate while preserving order
    seen, ld_clean = set(), []
    for p in ld_parts:
        if p and p not in seen:
            seen.add(p)
            ld_clean.append(p)

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_clean.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_clean)

    env["ROCM_PATH"] = str(THEROCK_DIST_DIR)
    env["HIP_PATH"] = str(THEROCK_DIST_DIR)

# Build the ctest command
# CTest runs both C++ (Catch2) tests and Python (pytest) tests
cmd = ["ctest", "--output-on-failure"]

# Test filtering based on test type
if TEST_TYPE == "smoke":
    # Run only a subset of tests for smoke testing
    # Use CTest's regex to filter test names
    cmd.extend(["-R", "origami_python|GEMM:.*compute"])
elif TEST_TYPE == "quick":
    # Run fewer tests for quick validation
    cmd.extend(["-R", "origami_python|Origami:"])
# For "full" test type, run all tests (no filter)

# Add extra arguments if provided
extra = os.getenv("EXTRA_CTEST_ARGS", "")
if extra:
    cmd += shlex.split(extra)

logging.info(f"++ Exec [{ORIGAMI_BUILD_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=str(ORIGAMI_BUILD_DIR), check=True, env=env)
