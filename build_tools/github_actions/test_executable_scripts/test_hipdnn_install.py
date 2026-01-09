#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hipDNN installation consumption test.

This test verifies that hipDNN packages built by TheRock can be properly
consumed by external projects using CMake's find_package. It tests the
CMake packaging/installation correctness, not hipDNN functionality.
"""

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

THEROCK_DIST_DIR = os.getenv("THEROCK_DIST_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
TEST_PROJECTS_DIR = SCRIPT_DIR / "hipdnn_install_tests"

TEST_PROJECTS = [
    "test_data_sdk",
    "test_plugin_sdk",
    "test_backend",
    "test_frontend",
    "test_test_sdk",
]

logging.basicConfig(level=logging.INFO)


def run_test(project_name: str, build_dir: Path):
    """Configure, build, and test a single project."""
    project_dir = TEST_PROJECTS_DIR / project_name

    # Configure
    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(project_dir),
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={THEROCK_DIST_DIR}",
    ]
    logging.info(f"++ Configure: {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, cwd=THEROCK_DIR)

    # Build
    build_cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Build: {shlex.join(build_cmd)}")
    subprocess.run(build_cmd, check=True, cwd=THEROCK_DIR)

    # Test
    test_cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
    logging.info(f"++ Test: {shlex.join(test_cmd)}")
    subprocess.run(test_cmd, check=True, cwd=THEROCK_DIR)


if __name__ == "__main__":
    if not THEROCK_DIST_DIR:
        raise RuntimeError("THEROCK_DIST_DIR environment variable not set")

    logging.info(f"Using THEROCK_DIST_DIR: {THEROCK_DIST_DIR}")

    for project in TEST_PROJECTS:
        logging.info(f"=== Testing {project} ===")
        with tempfile.TemporaryDirectory() as build_dir:
            run_test(project, Path(build_dir))
        logging.info(f"=== {project} PASSED ===")

    logging.info("All hipDNN install tests passed!")
