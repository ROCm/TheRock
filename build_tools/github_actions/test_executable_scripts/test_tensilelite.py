#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TensileLite Python unit test runner for TheRock CI.

Runs TensileLite and rocisa Python unit tests using uv for environment
management. Unlike other TheRock test scripts that execute compiled gtest
binaries from build artifacts, this script sets up a Python environment
from the rocm-libraries source checkout and runs pytest.

Required environment:
  - uv (installed by setup_test_environment action)
  - rocm-libraries checked out at ./rocm-libraries/
  - Build artifacts unpacked at ./build/ (provides ROCM_PATH for rocisa build)
"""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Paths
ROCM_LIBRARIES_DIR = THEROCK_DIR / "rocm-libraries"
TENSILELITE_DIR = ROCM_LIBRARIES_DIR / "projects" / "hipblaslt" / "tensilelite"

# Build artifacts provide the ROCm SDK (amdclang++, HIP headers) needed to
# compile the rocisa C++ extension during `uv sync`.
BUILD_DIR = Path(os.getenv("THEROCK_BUILD_DIR", THEROCK_DIR / "build"))
THEROCK_DIST_DIR = BUILD_DIR / "core" / "clr" / "dist"

# Test type (quick vs full) -- currently both run all unit tests since the
# test suite is small. This hook is here for future test filtering.
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()


def run_command(cmd, cwd=None, env=None, check=True):
    """Run a command with logging."""
    logging.info(f"++ Exec [{cwd or '.'}]$ {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, env=env)


def setup_python_environment(env):
    """Install Python 3.13 and sync dependencies using uv."""
    run_command(["uv", "python", "install", "3.13"], cwd=TENSILELITE_DIR, env=env)
    run_command(["uv", "sync", "--group", "dev"], cwd=TENSILELITE_DIR, env=env)


def run_tests(env):
    """Run TensileLite unit tests and rocisa tests via pytest."""
    failed = False

    # Run TensileLite unit tests
    logging.info("=== Running TensileLite unit tests ===")
    result = run_command(
        [
            "uv",
            "run",
            "pytest",
            "-v",
            "--junit-xml=tensilelite-unit-tests.xml",
            "--junit-prefix=tensilelite-unit",
            "-n",
            "auto",
            "Tensile/Tests/unit",
        ],
        cwd=TENSILELITE_DIR,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        logging.error(
            f"TensileLite unit tests failed with return code {result.returncode}"
        )
        failed = True

    # Run rocisa tests
    logging.info("=== Running rocisa tests ===")
    result = run_command(
        [
            "uv",
            "run",
            "pytest",
            "-v",
            "--junit-xml=rocisa-tests.xml",
            "--junit-prefix=rocisa",
            "-n",
            "auto",
            "rocisa/test",
        ],
        cwd=TENSILELITE_DIR,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        logging.error(f"rocisa tests failed with return code {result.returncode}")
        failed = True

    return failed


def main():
    if not TENSILELITE_DIR.is_dir():
        raise FileNotFoundError(
            f"TensileLite source not found at {TENSILELITE_DIR}. "
            "Ensure rocm-libraries is checked out at ./rocm-libraries/"
        )

    env = os.environ.copy()

    # Set ROCM_PATH from build artifacts so rocisa's cmake build can find
    # amdclang++ and HIP headers.
    if THEROCK_DIST_DIR.is_dir():
        env["ROCM_PATH"] = str(THEROCK_DIST_DIR)
        logging.info(f"Using ROCM_PATH from artifacts: {THEROCK_DIST_DIR}")
    else:
        logging.warning(
            f"TheRock dist directory not found at {THEROCK_DIST_DIR}. "
            "rocisa build may fail if ROCM_PATH is not set."
        )

    setup_python_environment(env)
    failed = run_tests(env)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
