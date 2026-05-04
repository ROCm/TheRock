# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Utility functions for test scripts to generate structured test output.

Provides helpers for:
- CTest JUnit XML output (--output-junit)
- GTest JSON output (--gtest_output=json:)
"""

import os
from pathlib import Path

# Default output directory from environment
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR", "./build")
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")


def get_test_results_dir() -> Path:
    """Get the test results directory, creating it if needed."""
    test_results_dir = Path(OUTPUT_ARTIFACTS_DIR) / "test-results"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    return test_results_dir


def get_ctest_junit_path(component: str) -> Path:
    """Get the path for CTest JUnit XML output.

    Args:
        component: Name of the test component (e.g., "rocprim", "hipblas")

    Returns:
        Path to the JUnit XML file
    """
    return get_test_results_dir() / f"ctest-{component}-shard{SHARD_INDEX}.xml"


def get_gtest_json_path(component: str) -> Path:
    """Get the path for GTest JSON output.

    Args:
        component: Name of the test component (e.g., "rocblas", "rocsolver")

    Returns:
        Path to the GTest JSON file
    """
    return get_test_results_dir() / f"gtest-{component}-shard{SHARD_INDEX}.json"


def add_ctest_junit_args(cmd: list, component: str) -> list:
    """Add JUnit XML output arguments to a ctest command.

    Args:
        cmd: The ctest command list
        component: Name of the test component

    Returns:
        The command list with --output-junit added
    """
    junit_path = get_ctest_junit_path(component)
    return cmd + ["--output-junit", str(junit_path)]


def get_gtest_output_arg(component: str) -> str:
    """Get the --gtest_output argument for GTest JSON output.

    Args:
        component: Name of the test component

    Returns:
        The --gtest_output=json:path argument string
    """
    json_path = get_gtest_json_path(component)
    return f"--gtest_output=json:{json_path}"
