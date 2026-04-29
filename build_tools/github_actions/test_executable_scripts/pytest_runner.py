#!/usr/bin/env python3
"""
Generic pytest test runner for components using pytest-based tests with test_categories.yaml.

This runner implements test sharding similar to GTest's GTEST_SHARD_INDEX to enable
parallel test execution across multiple CI runners with GPU isolation.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g., "tensile", "tensilite")
SHARD_INDEX: Current shard number (1-indexed, like GTest)
TOTAL_SHARDS: Total number of shards for test distribution
"""

import sys
import subprocess
import re
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def collect_pytest_tests(test_dir, marker_expr=None):
    """
    Collect all pytest test IDs from the test directory.

    Args:
        test_dir: Path to directory containing tests
        marker_expr: Optional pytest marker expression (e.g., "pre_checkin and not disabled")

    Returns:
        List of test node IDs (e.g., ["test_file.py::test_name", ...])
    """
    cmd = ["pytest", "--collect-only", "-q", str(test_dir)]

    if marker_expr:
        cmd.extend(["-m", marker_expr])

    logging.info(f"Collecting tests from {test_dir} with markers: {marker_expr or 'none'}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to collect tests: {e.stderr}")
        sys.exit(1)

    # Parse test IDs from pytest collection output
    # Format: "path/to/test_file.py::TestClass::test_method"
    test_ids = []
    for line in result.stdout.splitlines():
        # Test IDs contain "::" and don't start with whitespace or special chars
        if "::" in line and not line.startswith((" ", "<", "=", "-", "!")):
            test_id = line.strip()
            test_ids.append(test_id)

    logging.info(f"Collected {len(test_ids)} tests")
    return test_ids


def shard_tests(test_ids, shard_index, total_shards):
    """
    Distribute tests across shards using modulo arithmetic.

    Mimics GTest's GTEST_SHARD_INDEX behavior:
    - test_number % total_shards == (shard_index - 1)

    Args:
        test_ids: List of pytest test node IDs
        shard_index: Current shard number (1-indexed)
        total_shards: Total number of shards

    Returns:
        List of test IDs assigned to this shard
    """
    if total_shards == 1:
        logging.info("Single shard - running all tests")
        return test_ids

    # Convert to 0-indexed for modulo arithmetic
    shard_idx_zero = shard_index - 1

    sharded_tests = [
        test_id for i, test_id in enumerate(test_ids)
        if i % total_shards == shard_idx_zero
    ]

    logging.info(
        f"Shard {shard_index}/{total_shards}: "
        f"{len(sharded_tests)}/{len(test_ids)} tests assigned"
    )

    return sharded_tests


if __name__ == "__main__":
    # This is a placeholder main - will be extended in subsequent commits
    logging.info("Pytest test collection and sharding utility loaded")

    # Example usage (will be replaced with full runner logic)
    SHARD_INDEX = int(os.getenv("SHARD_INDEX", 1))
    TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", 1))

    logging.info(f"Shard configuration: {SHARD_INDEX}/{TOTAL_SHARDS}")
