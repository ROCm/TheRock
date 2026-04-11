#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TensileLite Python unit test runner for TheRock CI.

Runs TensileLite and rocisa Python unit tests using uv for environment
management. The hipblaslt cmake root and Python source tree are bundled
as test artifacts during the build via CMake install rules and staged
into share/hipblaslt-test-src/.

Environment variables used:
  AMDGPU_FAMILIES: GPU architecture string (e.g., "gfx94X-dcgpu"), logged only
  TEST_TYPE: "smoke", "quick", or "full" (default: "full"), reserved for future filtering
  THEROCK_BIN_DIR: Path to test binaries (e.g., "./build/bin")
  OUTPUT_ARTIFACTS_DIR: Path to unpacked build artifacts (e.g., "./build")
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

OUTPUT_ARTIFACTS_PATH = (
    Path(OUTPUT_ARTIFACTS_DIR).resolve() if OUTPUT_ARTIFACTS_DIR else Path(ROCM_PATH)
)

# hipblaslt cmake root staged as a build artifact.
HIPBLASLT_SOURCE_DIR = OUTPUT_ARTIFACTS_PATH / "share" / "hipblaslt-test-src"
# TensileLite Python sources are staged under the hipblaslt source tree.
TENSILELITE_DIR = HIPBLASLT_SOURCE_DIR / "tensilelite"

# Set up env with ROCM_PATH so rocisa's cmake build can find
# amdclang++ and HIP headers during `uv sync`.
env = os.environ.copy()
env["ROCM_PATH"] = ROCM_PATH
# Tell rocisa/setup.py where the staged hipblaslt cmake root is,
# so `cmake --preset rocisa -S<source_dir>` finds CMakePresets.json.
env["HIPBLASLT_SOURCE_DIR"] = str(HIPBLASLT_SOURCE_DIR)


def run_command(cmd, cwd=None, check=True):
    """Run a command with logging."""
    logging.info(f"++ Exec [{cwd or '.'}]$ {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, env=env)


def main():
    if not TENSILELITE_DIR.is_dir():
        raise FileNotFoundError(
            f"TensileLite test sources not found at {TENSILELITE_DIR}. "
            "Ensure the blas test artifact is unpacked."
        )

    logging.info(f"# TENSILELITE_DIR: {TENSILELITE_DIR}")
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
