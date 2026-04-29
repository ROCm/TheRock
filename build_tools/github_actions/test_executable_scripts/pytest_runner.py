#!/usr/bin/env python3
"""
Generic pytest test runner for components using pytest-based tests with test_categories.yaml.

This runner implements test sharding similar to GTest's GTEST_SHARD_INDEX to enable
parallel test execution across multiple CI runners with GPU isolation.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g., "tensile", "tensilite")
TEST_TYPE: Test category to run (quick, standard, comprehensive, full)
SHARD_INDEX: Current shard number (1-indexed, like GTest)
TOTAL_SHARDS: Total number of shards for test distribution
THEROCK_BIN_DIR: Path to installed binaries
"""

import sys
import subprocess
import re
import os
import logging
import yaml
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


def load_test_categories_yaml(yaml_path):
    """
    Load and parse test_categories.yaml configuration file.

    Args:
        yaml_path: Path to test_categories.yaml

    Returns:
        Dictionary containing parsed YAML configuration
    """
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded test categories from {yaml_path}")
            return config
    except FileNotFoundError:
        logging.error(f"test_categories.yaml not found at {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Invalid YAML syntax in {yaml_path}: {e}")
        sys.exit(1)


def build_marker_expression(category_config):
    """
    Build pytest marker expression from test_categories.yaml category configuration.

    Args:
        category_config: Dictionary for a specific test category (e.g., config['test_categories']['quick'])

    Returns:
        String containing pytest marker expression (e.g., "unit or pre_checkin")
    """
    pytest_markers = category_config.get("pytest_markers", [])
    exclude_markers = category_config.get("exclude_markers", [])

    if not pytest_markers:
        logging.warning("No pytest_markers defined for category - will run all tests")
        marker_expr = ""
    elif len(pytest_markers) == 1:
        marker_expr = pytest_markers[0]
    else:
        # Multiple markers: combine with "or" - (marker1 or marker2 or marker3)
        marker_expr = " or ".join(pytest_markers)

    # Add exclusions with "and not"
    if exclude_markers:
        exclusion_expr = " and not ".join([""] + exclude_markers)  # Leading empty string for first "and not"
        marker_expr = f"({marker_expr}){exclusion_expr}" if marker_expr else exclusion_expr.lstrip(" and ")

    logging.info(f"Built marker expression: {marker_expr or '(none)'}")
    return marker_expr


if __name__ == "__main__":
    # This is a placeholder main - will be extended in subsequent commits
    logging.info("Pytest test collection and sharding utility loaded")

    # Example usage (will be replaced with full runner logic)
    SHARD_INDEX = int(os.getenv("SHARD_INDEX", 1))
    TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", 1))

    logging.info(f"Shard configuration: {SHARD_INDEX}/{TOTAL_SHARDS}")
