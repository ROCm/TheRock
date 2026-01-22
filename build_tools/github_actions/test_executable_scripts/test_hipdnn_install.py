#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN installation consumption test.

This test verifies that hipDNN packages built by TheRock can be properly
consumed by external projects using CMake's find_package. It tests the
CMake packaging/installation correctness, not hipDNN functionality.
"""

import argparse
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
TEST_PROJECT_DIR = SCRIPT_DIR / "hipdnn_install_tests"

logging.basicConfig(level=logging.INFO)


def run_tests(build_dir: Path):
    """Configure, build, and test all hipDNN packages."""
    # Locally, can set OUTPUT_ARTIFACTS_DIR=build/dist/rocm for testing
    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()

    # Set up environment variables for CMake/HIP
    environ_vars = os.environ.copy()
    environ_vars["HIP_PLATFORM"] = "amd"

    # Set library path for runtime (needed when running the test executables)
    rocm_lib = str(artifacts_path / "lib")
    if "LD_LIBRARY_PATH" in environ_vars:
        environ_vars["LD_LIBRARY_PATH"] = (
            f"{rocm_lib}:{environ_vars['LD_LIBRARY_PATH']}"
        )
    else:
        environ_vars["LD_LIBRARY_PATH"] = rocm_lib

    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(TEST_PROJECT_DIR),
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={artifacts_path}",
        f"-DCMAKE_CXX_COMPILER={artifacts_path}/lib/llvm/bin/clang++",
        f"-DCMAKE_C_COMPILER={artifacts_path}/lib/llvm/bin/clang",
        "--log-level=WARNING",
    ]
    logging.info(f"++ Configure: {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)

    build_cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Build: {shlex.join(build_cmd)}")
    subprocess.run(build_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)

    test_cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
    logging.info(f"++ Test: {shlex.join(test_cmd)}")
    subprocess.run(test_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test hipDNN package installation and consumption"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory path (will be created if doesn't exist). "
        "If not specified, uses temporary directory that is auto-deleted.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {OUTPUT_ARTIFACTS_DIR}")

    if args.build_dir:
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using persistent build directory: {build_dir}")
        run_tests(build_dir)
        logging.info(f"Build artifacts retained in: {build_dir}")
    else:
        logging.info("Using temporary build directory (auto-cleanup)")
        with tempfile.TemporaryDirectory() as temp_dir:
            run_tests(Path(temp_dir))

    logging.info("All hipDNN install tests passed!")
