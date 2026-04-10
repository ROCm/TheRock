#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TensileLite Python unit test runner for TheRock CI.

Runs TensileLite and rocisa Python unit tests using uv for environment
management. Unlike other TheRock test scripts that execute compiled gtest
binaries from build artifacts, this script sets up a Python environment
from the rocm-libraries source checkout and runs pytest.

This requires rocm-libraries to be checked out alongside TheRock in CI
(via therock-test-component.yml) because the tests are Python source
files, not compiled binaries.

Required environment:
  - uv (installed by setup_test_environment action)
  - rocm-libraries checked out at ./rocm-libraries/
  - Build artifacts unpacked at ./build/ (provides ROCM_PATH for rocisa build)

Environment variables used:
  AMDGPU_FAMILIES: GPU architecture string (e.g., "gfx94X-dcgpu"), logged only
  TEST_TYPE: "smoke", "quick", or "full" (default: "full"), reserved for future filtering
  THEROCK_BIN_DIR: Path to test binaries (default: "./build/bin")
  OUTPUT_ARTIFACTS_DIR: Path to unpacked build artifacts (default: "./build")
"""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Paths — rocm-libraries is checked out alongside TheRock in CI
ROCM_LIBRARIES_DIR = THEROCK_DIR / "rocm-libraries"
TENSILELITE_DIR = ROCM_LIBRARIES_DIR / "projects" / "hipblaslt" / "tensilelite"

# Build artifacts — derive ROCM_PATH the same way as other test scripts
# (test_runner.py, test_rocroller.py): parent of THEROCK_BIN_DIR.
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR", "")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()
platform = os.getenv("RUNNER_OS", "linux").lower()

if THEROCK_BIN_DIR:
    ROCM_PATH = str(Path(THEROCK_BIN_DIR).resolve().parent)
elif OUTPUT_ARTIFACTS_DIR:
    ROCM_PATH = str(Path(OUTPUT_ARTIFACTS_DIR).resolve())
else:
    ROCM_PATH = "/opt/rocm"

# Set up env with ROCM_PATH so rocisa's cmake build can find
# amdclang++ and HIP headers during `uv sync`.
env = os.environ.copy()
env["ROCM_PATH"] = ROCM_PATH


def run_command(cmd, cwd=None, check=True):
    """Run a command with logging."""
    logging.info(f"++ Exec [{cwd or '.'}]$ {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, env=env)


def main():
    if not TENSILELITE_DIR.is_dir():
        raise FileNotFoundError(
            f"TensileLite source not found at {TENSILELITE_DIR}. "
            "Ensure rocm-libraries is checked out at ./rocm-libraries/"
        )

    logging.info(f"# ROCM_PATH: {ROCM_PATH}")
    logging.info(f"# AMDGPU_FAMILIES: {AMDGPU_FAMILIES}")
    logging.info(f"# TEST_TYPE: {TEST_TYPE}")
    logging.info(f"# Platform: {platform}")
    logging.info("")

    # Set up Python environment with uv
    run_command(["uv", "python", "install", "3.13"], cwd=TENSILELITE_DIR)
    run_command(["uv", "sync", "--group", "dev"], cwd=TENSILELITE_DIR)

    failed = False

    # Run TensileLite unit tests (CPU only, no GPU required)
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
        check=False,
    )
    if result.returncode != 0:
        logging.error(
            f"TensileLite unit tests failed with return code {result.returncode}"
        )
        failed = True

    # Run rocisa tests (CPU only, no GPU required)
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
        check=False,
    )
    if result.returncode != 0:
        logging.error(f"rocisa tests failed with return code {result.returncode}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
