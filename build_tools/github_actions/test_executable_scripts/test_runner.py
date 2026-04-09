#!/usr/bin/env python3
"""
This is a generic test runner that can test multiple components.
This works on components in rocm-libraries/rocm-systems which use test_categories.yml for test categorization.

Environment variables used:
TEST_COMPONENT: Job name of the component to test (e.g., "miopen", "rocrand", "hiprand")
    This is automatically set by the GitHub Actions workflow from the job_name field.
    The script maps these job names to actual test directory names (e.g., "miopen" -> "MIOpen")
    Defaults to "miopen" if not set.
TEST_TYPE: Test category to run - one of "quick", "standard", "comprehensive", or "full".
    Defaults to "quick". Invalid values fall back to "quick" with an error message.
AMDGPU_FAMILIES: Parsed to extract GPU architecture (e.g., "gfx1151")

The script discovers GPU-specific labels via ctest --print-labels and runs the appropriate tests for the current GPU architecture.
"""

import sys
import subprocess
import re
import os

import logging
import shlex
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}
TEST_TYPE = os.getenv("TEST_TYPE", "quick")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Map job names to actual test directory names
# The job names come from TEST_COMPONENT env var (set by GitHub Actions workflow)
# and need to be mapped to the actual directory names in THEROCK_BIN_DIR
COMPONENT_DIR_MAPPING = {
    "miopen": "MIOpen",
    "rocblas": "rocblas",
    "rocrand": "rocRAND",
    "hiprand": "hipRAND",
    "rocthrust": "rocthrust",
    "rocprim": "rocprim",
    "rocwmma": "rocwmma",
    "hipcub": "hipcub",
    "hipdnn": "hipdnn",
    "hipdnn-samples": "hipdnn_samples",
    "miopen_plugin": "miopen_legacy_plugin",
    "hipsparselt": "hipsparselt",
    # Add more mappings as needed
}

# Get the test component from environment (required - no default)
test_component_job_name = os.getenv("TEST_COMPONENT")
if not test_component_job_name:
    print(
        "ERROR: TEST_COMPONENT environment variable is required but not set.",
        file=sys.stderr,
    )
    sys.exit(1)

TEST_COMPONENT = COMPONENT_DIR_MAPPING.get(
    test_component_job_name, test_component_job_name
)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)

# CTest parallel jobs (use fewer in less capable platforms)
ctest_parallel_count = 8
if AMDGPU_FAMILIES and "gfx1152" in AMDGPU_FAMILIES:
    ctest_parallel_count = 4
elif AMDGPU_FAMILIES and "gfx1153" in AMDGPU_FAMILIES:
    ctest_parallel_count = 4

# CTest per-test timeout (default 2 hours, in seconds)
# There should be a timeout set from component level, but this can be used as an override
ctest_timeout_seconds = 7200

environ_vars = os.environ.copy()
# Set the GTEST env vars for Gtest based tests
# Set ROCM_PATH for tests that rely on it
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

logging.basicConfig(level=logging.INFO)
##############################################


def find_matching_gpu_arch(gpu_arch: str, available_gpu_archs: set[str]) -> str | None:
    """
    Find the most specific GPU architecture in the set that matches the given GPU.

    Tries in order from most specific to least specific:
    # Example:
    # find_matching_gpu_arch('gfx1151', {'gfx1151', 'gfx115X', 'gfx11X'}) gives 'gfx1151'
    # find_matching_gpu_arch('gfx1151', {'gfx1150', 'gfx94X', 'gfx11X'}) gives 'gfx11X'
    - Wildcard matches (gfx115X, gfx11X, etc.)

    Returns the matching architecture string or None if no match found.
    """
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    # Start matching from the end (gfx115X) and go back till the 5th character (gfx11X)
    # Return the top matching pattern
    for i in range(len(gpu_arch) - 1, 4, -1):
        pattern = gpu_arch[:i] + "X"
        if pattern in available_gpu_archs:
            return pattern

    return None


def get_available_gpu_suite_tests():
    """
    Get all available GPU architecture labels from ctest --print-labels.

    Parses labels of the form ex_gpu_{gpu_arch} (e.g. ex_gpu_gfx110X, ex_gpu_gfx950).
    Returns a set of gpu_arch strings (e.g., 'gfx110X', 'gfx115X', 'gfx950').
    """
    test_dir = Path(THEROCK_BIN_DIR) / TEST_COMPONENT
    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Error: Test directory does not exist: {test_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        # Ensure the component has at least one test
        list_result = subprocess.run(
            ["ctest", "-N", "--test-dir", str(test_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        total_tests = sum(
            1
            for line in list_result.stdout.splitlines()
            if re.search(r"Test\s+#\d+:", line)
        )
        if total_tests == 0:
            print(
                f"Error: No tests found in {test_dir}. Cannot run test suite.",
                file=sys.stderr,
            )
            sys.exit(1)

        result = subprocess.run(
            ["ctest", "--print-labels", "--test-dir", str(test_dir)],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_archs = set()
        prefix = "ex_gpu_"
        for line in result.stdout.splitlines():
            label = line.strip()
            if label.startswith(prefix):
                gpu_arch = label[len(prefix) :]
                if gpu_arch.startswith("gfx"):
                    gpu_archs.add(gpu_arch)

        return gpu_archs
    except subprocess.CalledProcessError as e:
        print(f"Error running ctest --print-labels: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: ctest not found. Make sure CMake/CTest is installed.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_ctest_command(category, gpu_arch, available_gpu_archs):
    """
    Build the appropriate ctest command based on the category and GPU architecture.

    Returns a list of command arguments suitable for subprocess.run()
    """
    cmd = ["ctest", "-L", category]

    # Add common ctest parameters
    cmd.extend(
        [
            "--output-on-failure",
            "--parallel",
            f"{ctest_parallel_count}",
            "--timeout",
            str(ctest_timeout_seconds),
            "--test-dir",
            f"{THEROCK_BIN_DIR}/{TEST_COMPONENT}",
            "-V",  # Always run in verbose mode
            # Shards the tests by running a specific set of tests based on starting test (shard_index) and stride (total_shards)
            "--tests-information",
            f"{SHARD_INDEX},,{TOTAL_SHARDS}",
        ]
    )

    if gpu_arch.lower() in ["generic", "none", ""]:
        # For generic/unspecified GPU, exclude all GPU-specific suite tests
        cmd.extend(["-LE", "ex_gpu"])
        return cmd

    # Find the appropriate GPU suite
    matching_arch = find_matching_gpu_arch(gpu_arch, available_gpu_archs)

    if matching_arch:
        # Run the specific GPU suite using the ex_gpu label
        gpu_label = f"ex_gpu_{matching_arch}"
        cmd.extend(["-L", gpu_label])
        print(f"# Using GPU suite label: {gpu_label}")
    else:
        # No specific GPU suite found, run standard tests excluding all GPU-specific ones
        cmd.extend(["-LE", "ex_gpu"])
        print(f"# No GPU suite found for {gpu_arch}, excluding all ex_gpu tests")

    return cmd


def main():
    category = TEST_TYPE.lower() if TEST_TYPE else "quick"
    if category not in VALID_TEST_CATEGORIES:
        print(
            f"ERROR: Invalid TEST_TYPE '{TEST_TYPE}'. "
            f"Must be one of: {', '.join(sorted(VALID_TEST_CATEGORIES))}. "
            f"Falling back to 'quick'.",
            file=sys.stderr,
        )
        category = "quick"

    # Use AMDGPU_FAMILIES from environment variable, extract gfx<xxx> part
    gpu_arch = ""
    if AMDGPU_FAMILIES:
        # Extract gfx<xxx> pattern from AMDGPU_FAMILIES string
        # Pattern matches: gfx followed by alphanumeric characters (e.g., gfx1151, gfx950, gfx11X)
        match = re.search(r"gfx[0-9a-zA-Z]+", AMDGPU_FAMILIES)
        if match:
            gpu_arch = match.group(0)
        else:
            print(
                f"# Warning: Could not extract GPU architecture from AMDGPU_FAMILIES='{AMDGPU_FAMILIES}', using default '{gpu_arch}'"
            )

    print(
        f"# TEST_COMPONENT: {test_component_job_name} -> Test Directory: {TEST_COMPONENT}"
    )
    print(f"# TEST_TYPE: {TEST_TYPE} -> Category: {category}")
    print(f"# AMDGPU_FAMILIES: {AMDGPU_FAMILIES} -> GPU Architecture: {gpu_arch}")
    print()

    # Get available GPU suite tests from ctest
    print("# Discovering available GPU suite tests...")
    available_gpu_archs = get_available_gpu_suite_tests()

    if available_gpu_archs:
        print(f"# Found {len(available_gpu_archs)} GPU suite test(s)")
        print(f"# Available GPU architectures: {sorted(available_gpu_archs)}")
    else:
        print("# Warning: No GPU specific test suites available")
    print()

    # Build the ctest command
    cmd = build_ctest_command(category, gpu_arch, available_gpu_archs)

    print(f"# Running: {' '.join(cmd)}")
    print()

    # Execute the command
    try:
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=THEROCK_DIR, env=environ_vars, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running ctest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
