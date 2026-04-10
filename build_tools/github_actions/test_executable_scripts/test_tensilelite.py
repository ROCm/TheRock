#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
TensileLite Python test runner for TheRock CI.

Runs TensileLite unit tests, rocisa tests, and GPU-dependent integration
tests using uv for environment management. Unlike other TheRock test scripts
that execute compiled gtest binaries from build artifacts, this script sets
up a Python environment from the rocm-libraries source checkout, builds the
tensilelite-client, and runs pytest.

Required environment:
  - uv (installed by setup_test_environment action)
  - rocm-libraries checked out at ./rocm-libraries/
  - Build artifacts unpacked at ./build/ (provides ROCM_PATH for rocisa build)

Environment variables used:
  AMDGPU_FAMILIES: GPU architecture string (e.g., "gfx94X-dcgpu")
  TEST_TYPE: "smoke", "quick", or "full" (default: "full")
  THEROCK_BIN_DIR: Path to test binaries (default: "./build/bin")
  OUTPUT_ARTIFACTS_DIR: Path to unpacked build artifacts (default: "./build")
"""

import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Paths — rocm-libraries is checked out alongside TheRock in CI
ROCM_LIBRARIES_DIR = THEROCK_DIR / "rocm-libraries"
TENSILELITE_DIR = ROCM_LIBRARIES_DIR / "projects" / "hipblaslt" / "tensilelite"
HIPBLASLT_DIR = ROCM_LIBRARIES_DIR / "projects" / "hipblaslt"

# Build artifacts — derive ROCM_PATH the same way as other test scripts
# (test_runner.py, test_rocroller.py): parent of THEROCK_BIN_DIR or
# OUTPUT_ARTIFACTS_DIR.
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR", "")

if THEROCK_BIN_DIR:
    ROCM_PATH = str(Path(THEROCK_BIN_DIR).resolve().parent)
elif OUTPUT_ARTIFACTS_DIR:
    ROCM_PATH = str(Path(OUTPUT_ARTIFACTS_DIR).resolve())
else:
    ROCM_PATH = "/opt/rocm"

# Client build output directory
CLIENT_BUILD_DIR = TENSILELITE_DIR / "build_tmp"
CLIENT_BINARY = CLIENT_BUILD_DIR / "tensilelite" / "client" / "tensilelite-client"

# CI environment
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()
platform = os.getenv("RUNNER_OS", "linux").lower()

# Set up env with ROCM_PATH
env = os.environ.copy()
env["ROCM_PATH"] = ROCM_PATH
logging.info(f"Using ROCM_PATH: {ROCM_PATH}")

# Extract GPU architecture from AMDGPU_FAMILIES (e.g., "gfx94X-dcgpu" -> "gfx94X")
gpu_arch = ""
if AMDGPU_FAMILIES:
    match = re.search(r"gfx[0-9a-zA-Z]+", AMDGPU_FAMILIES)
    if match:
        gpu_arch = match.group(0)
    else:
        logging.warning(
            f"Could not extract GPU architecture from AMDGPU_FAMILIES='{AMDGPU_FAMILIES}'"
        )


def run_command(cmd, cwd=None, check=True):
    """Run a command with logging."""
    logging.info(f"++ Exec [{cwd or '.'}]$ {shlex.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, env=env)


def build_client():
    """Build the tensilelite-client binary for integration tests."""
    logging.info(f"=== Building tensilelite-client for {gpu_arch} ===")

    compiler = os.path.join(ROCM_PATH, "bin", "amdclang++")

    build_dir = str(CLIENT_BUILD_DIR)
    os.makedirs(build_dir, exist_ok=True)

    # Configure using the tensilelite CMake preset
    run_command(
        [
            "cmake",
            "--preset",
            "tensilelite",
            "-S",
            str(HIPBLASLT_DIR),
            "-B",
            build_dir,
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_PREFIX_PATH={ROCM_PATH}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DGPU_TARGETS={gpu_arch}",
        ],
        cwd=TENSILELITE_DIR,
    )

    # Build
    run_command(
        ["cmake", "--build", build_dir, "--parallel"],
        cwd=TENSILELITE_DIR,
    )

    if not CLIENT_BINARY.exists():
        logging.error(f"Client binary not found at {CLIENT_BINARY}")
        return False

    logging.info(f"Client built successfully: {CLIENT_BINARY}")
    return True


def main():
    if not TENSILELITE_DIR.is_dir():
        raise FileNotFoundError(
            f"TensileLite source not found at {TENSILELITE_DIR}. "
            "Ensure rocm-libraries is checked out at ./rocm-libraries/"
        )

    logging.info(
        f"# AMDGPU_FAMILIES: {AMDGPU_FAMILIES} -> GPU Architecture: {gpu_arch}"
    )
    logging.info(f"# TEST_TYPE: {TEST_TYPE}")
    logging.info(f"# Platform: {platform}")
    logging.info("")

    # Set up Python environment with uv
    run_command(["uv", "python", "install", "3.13"], cwd=TENSILELITE_DIR)
    run_command(["uv", "sync", "--group", "dev"], cwd=TENSILELITE_DIR)

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
        check=False,
    )
    if result.returncode != 0:
        logging.error(f"rocisa tests failed with return code {result.returncode}")
        failed = True

    # Build client and run integration tests (GPU required)
    if gpu_arch:
        client_built = build_client()
        if client_built:
            logging.info("=== Running TensileLite integration tests ===")
            result = run_command(
                [
                    "uv",
                    "run",
                    "pytest",
                    "-v",
                    "--junit-xml=tensilelite-integration-tests.xml",
                    "--junit-prefix=tensilelite-integration",
                    f"--prebuilt-client={CLIENT_BINARY}",
                    "-n",
                    "auto",
                    "Tensile/Tests/common",
                ],
                cwd=TENSILELITE_DIR,
                check=False,
            )
            if result.returncode != 0:
                logging.error(
                    f"TensileLite integration tests failed with return code {result.returncode}"
                )
                failed = True
        else:
            logging.error("Skipping integration tests: client build failed")
            failed = True
    else:
        logging.warning("Skipping integration tests: no GPU architecture specified")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
