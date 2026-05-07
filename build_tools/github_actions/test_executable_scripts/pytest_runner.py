#!/usr/bin/env python3
"""
Generic pytest test runner for components using pytest-based tests with test_categories.yaml.

This runner implements test sharding similar to GTest's GTEST_SHARD_INDEX to enable
parallel test execution across multiple CI runners with GPU isolation.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g., "tensile", "tensilite")
TEST_TYPE: Test category to run (quick, standard, comprehensive, full)
AMDGPU_FAMILIES: GPU architecture for skip marker filtering (e.g., "gfx1151")
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

    logging.info(
        f"Collecting tests from {test_dir} with markers: {marker_expr or 'none'}"
    )

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
        test_id
        for i, test_id in enumerate(test_ids)
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
        exclusion_expr = " and not ".join(
            [""] + exclude_markers
        )  # Leading empty string for first "and not"
        marker_expr = (
            f"({marker_expr}){exclusion_expr}"
            if marker_expr
            else exclusion_expr.lstrip(" and ")
        )

    logging.info(f"Built marker expression: {marker_expr or '(none)'}")
    return marker_expr


def extract_gpu_arch(amdgpu_families):
    """
    Extract GPU architecture from AMDGPU_FAMILIES environment variable.

    Args:
        amdgpu_families: String from AMDGPU_FAMILIES env var (e.g., "gfx1151" or "gfx94X")

    Returns:
        GPU architecture string (e.g., "gfx1151") or None
    """
    if not amdgpu_families:
        return None

    # Extract first GPU architecture (format: gfxNNNN or gfxNNX)
    match = re.search(r"gfx\w+", amdgpu_families)
    if match:
        gpu_arch = match.group(0)
        logging.info(f"Detected GPU architecture: {gpu_arch}")
        return gpu_arch

    logging.warning(f"Could not parse GPU architecture from: {amdgpu_families}")
    return None


def add_gpu_skip_markers(marker_expr, gpu_arch):
    """
    Add GPU-specific skip markers to pytest marker expression.

    Pytest tests use markers like skip-gfx1151 or skip-gfx115X to exclude tests
    on specific GPU architectures. This function adds "and not skip-gfxXXXX" to
    the marker expression.

    Args:
        marker_expr: Existing pytest marker expression
        gpu_arch: GPU architecture (e.g., "gfx1151")

    Returns:
        Updated marker expression with GPU skip exclusions
    """
    if not gpu_arch:
        return marker_expr

    # Build list of skip markers to exclude, from most specific to least
    # e.g., for gfx1151: ["skip-gfx1151", "skip-gfx115X", "skip-gfx11X"]
    skip_markers = [f"skip-{gpu_arch}"]

    # Add wildcard patterns
    for i in range(len(gpu_arch) - 1, 4, -1):  # From end to "gfx11"
        pattern = gpu_arch[:i] + "X"
        skip_markers.append(f"skip-{pattern}")

    # Build exclusion expression
    exclusion_parts = [f"not {marker}" for marker in skip_markers]
    gpu_exclusion = " and ".join(exclusion_parts)

    # Combine with existing marker expression
    if marker_expr:
        combined_expr = f"({marker_expr}) and {gpu_exclusion}"
    else:
        combined_expr = gpu_exclusion

    logging.info(f"Added GPU skip markers: {gpu_exclusion}")
    return combined_expr


def run_pytest_tests(test_dir, test_ids, marker_expr, timeout, num_workers, env_vars):
    """
    Execute pytest with the specified test IDs and configuration.

    Args:
        test_dir: Base test directory
        test_ids: List of pytest test node IDs to run
        marker_expr: Pytest marker expression (for logging purposes)
        timeout: Per-test timeout in seconds
        num_workers: Number of parallel xdist workers
        env_vars: Environment variables dictionary

    Returns:
        Exit code from pytest
    """
    if not test_ids:
        logging.warning("No tests to run in this shard")
        return 0

    cmd = ["pytest", str(test_dir)]

    # Add specific test IDs
    cmd.extend(test_ids)

    # Add pytest options
    cmd.extend(
        [
            "-v",  # Verbose output
            f"--timeout={timeout}",  # Per-test timeout
            f"--numprocesses={num_workers}",  # Parallel workers (pytest-xdist)
            "--color=yes",  # Color output
        ]
    )

    logging.info(
        f"Running pytest with {len(test_ids)} tests, {num_workers} workers, {timeout}s timeout"
    )
    logging.info(f"Marker expression used for collection: {marker_expr or '(none)'}")
    logging.info(
        f"Command: {' '.join(cmd[:3])} <{len(test_ids)} test IDs> {' '.join(cmd[3+len(test_ids):])}"
    )

    result = subprocess.run(cmd, env=env_vars, check=False)
    return result.returncode


if __name__ == "__main__":
    # Component mapping: job name -> directory name
    PYTEST_COMPONENT_MAPPING = {
        "tensile": "Tensile",
        "tensilite": "hipblaslt/tensilite",
    }

    # Source-based components (run from source tree, not installed artifacts)
    SOURCE_BASED_COMPONENTS = {
        "tensilite": "projects/hipblaslt/tensilite",
    }

    # Components that install to OUTPUT_ARTIFACTS_DIR instead of THEROCK_BIN_DIR
    # (e.g., Tensile installs to {prefix}/Tensile/ not {prefix}/bin/Tensile/)
    ARTIFACTS_DIR_COMPONENTS = {
        "tensile",
    }

    # Valid test categories
    VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}

    # Environment variables
    THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
    OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
    GITHUB_WORKSPACE = os.getenv("GITHUB_WORKSPACE")
    TEST_COMPONENT_NAME = os.getenv("TEST_COMPONENT")
    TEST_TYPE = os.getenv("TEST_TYPE", "quick")
    AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
    SHARD_INDEX = int(os.getenv("SHARD_INDEX", 1))
    TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", 1))

    # Validate required environment variables
    if not TEST_COMPONENT_NAME:
        logging.error("TEST_COMPONENT environment variable is required")
        sys.exit(1)

    # Determine component path (source-based or installed)
    if TEST_COMPONENT_NAME in SOURCE_BASED_COMPONENTS:
        # Source-based component: run from checked-out source tree
        if not GITHUB_WORKSPACE:
            logging.error(
                f"GITHUB_WORKSPACE required for source-based component {TEST_COMPONENT_NAME}"
            )
            sys.exit(1)
        source_path = SOURCE_BASED_COMPONENTS[TEST_COMPONENT_NAME]
        component_path = Path(GITHUB_WORKSPACE) / source_path
        logging.info(f"Using source-based tests from: {component_path}")
    else:
        # Installed component: determine base directory
        if TEST_COMPONENT_NAME in ARTIFACTS_DIR_COMPONENTS:
            # Component installs to OUTPUT_ARTIFACTS_DIR (e.g., Tensile -> ./build/Tensile/)
            if not OUTPUT_ARTIFACTS_DIR:
                logging.error("OUTPUT_ARTIFACTS_DIR environment variable is required")
                sys.exit(1)
            base_dir = Path(OUTPUT_ARTIFACTS_DIR)
            logging.info(f"Using artifacts directory base: {base_dir}")
        else:
            # Component installs to THEROCK_BIN_DIR (e.g., hipblaslt/tensilelite -> ./build/bin/hipblaslt/tensilelite/)
            if not THEROCK_BIN_DIR:
                logging.error("THEROCK_BIN_DIR environment variable is required")
                sys.exit(1)
            base_dir = Path(THEROCK_BIN_DIR)
            logging.info(f"Using bin directory base: {base_dir}")

        component_dir = PYTEST_COMPONENT_MAPPING.get(
            TEST_COMPONENT_NAME, TEST_COMPONENT_NAME
        )
        component_path = base_dir / component_dir
        logging.info(f"Using installed tests from: {component_path}")

    if not component_path.exists():
        logging.error(f"Component directory not found: {component_path}")
        sys.exit(1)

    # Validate test category
    if TEST_TYPE not in VALID_TEST_CATEGORIES:
        logging.warning(f"Invalid TEST_TYPE '{TEST_TYPE}', falling back to 'quick'")
        TEST_TYPE = "quick"

    logging.info(f"Component: {TEST_COMPONENT_NAME} ({component_dir})")
    logging.info(f"Test category: {TEST_TYPE}")
    logging.info(f"Shard: {SHARD_INDEX}/{TOTAL_SHARDS}")

    # Load test categories configuration
    yaml_path = component_path / "test_categories.yaml"
    if not yaml_path.exists():
        logging.error(f"test_categories.yaml not found at {yaml_path}")
        sys.exit(1)

    config = load_test_categories_yaml(yaml_path)

    # Get category configuration
    category_config = config.get("test_categories", {}).get(TEST_TYPE)
    if not category_config:
        logging.error(f"No configuration found for test category '{TEST_TYPE}'")
        sys.exit(1)

    # Build marker expression
    marker_expr = build_marker_expression(category_config)

    # Add GPU architecture filtering
    gpu_arch = extract_gpu_arch(AMDGPU_FAMILIES)
    marker_expr = add_gpu_skip_markers(marker_expr, gpu_arch)

    # Get test directory (assume Tests/ subdirectory)
    test_dir = component_path / "Tests"
    if not test_dir.exists():
        logging.error(f"Tests directory not found: {test_dir}")
        sys.exit(1)

    # Collect tests
    all_test_ids = collect_pytest_tests(test_dir, marker_expr)

    # Shard tests
    sharded_test_ids = shard_tests(all_test_ids, SHARD_INDEX, TOTAL_SHARDS)

    # Get execution settings
    exec_settings = config.get("execution_settings", {})
    timeout = exec_settings.get("category_timeouts", {}).get(TEST_TYPE, 300)
    num_workers = exec_settings.get("parallel_workers", 4)

    # Setup environment variables
    env_vars = os.environ.copy()

    # Apply custom environment variables from config
    custom_env = exec_settings.get("environment", {})
    for key, value in custom_env.items():
        # Replace {ROCM_PATH} placeholder
        if "{ROCM_PATH}" in str(value):
            rocm_path = Path(THEROCK_BIN_DIR).parent
            value = value.replace("{ROCM_PATH}", str(rocm_path))
        env_vars[key] = str(value)
        logging.info(f"Set environment variable: {key}={value}")

    # Run pytest
    exit_code = run_pytest_tests(
        test_dir, sharded_test_ids, marker_expr, timeout, num_workers, env_vars
    )

    sys.exit(exit_code)
