#!/usr/bin/env python3
"""
This is a generic test runner that can test multiple components using GPU-based filtering.
This works on top of the PR - Test Filter Standardization Proof of Concept - MIOpen #3513
in rocm-libraries (https://github.com/ROCm/rocm-libraries/pull/3513)

Environment variables used:
- TEST_COMPONENT: Job name of the component to test (e.g., "miopen", "rocrand", "hiprand")
  This is automatically set by the GitHub Actions workflow from the job_name field.
  The script maps these job names to actual test directory names (e.g., "miopen" -> "MIOpen")
  Defaults to "miopen" if not set.
- TEST_TYPE: "smoke" runs tests with "quick" category, otherwise runs "standard" category
- AMDGPU_FAMILIES: Parsed to extract GPU architecture (e.g., "gfx1151")

The script checks the available tests from ctest -N and filters the appropriate tests based on the GPU architecture.
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
TEST_TYPE = os.getenv("TEST_TYPE", "quick")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# Map job names to actual test directory names
# The job names come from TEST_COMPONENT env var (set by GitHub Actions workflow)
# and need to be mapped to the actual directory names in THEROCK_BIN_DIR
COMPONENT_DIR_MAPPING = {
    "miopen": "MIOpen",
    "rocrand": "rocRAND",
    "hiprand": "hipRAND",
    "rocthrust": "rocthrust",
    "rocprim": "rocprim",
    "rocwmma": "rocwmma",
    "hipcub": "hipcub",
    "hipdnn": "hipdnn",
    "hipdnn-samples": "hipdnn_samples",
    "miopen_plugin": "miopen_legacy_plugin",
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
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# Some of our runtime kernel compilations have been relying on either ROCM_PATH being set, or ROCm being installed at
# /opt/rocm. Neither of these is true in TheRock so we need to supply ROCM_PATH to our tests.
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

logging.basicConfig(level=logging.INFO)
##############################################


def get_available_gpu_exclusion_tests():
    """
    Get all available GPU exclusion test architectures from ctest -N.

    Parses test names in the format: {target_name}-{category}-{gpu_arch}-exclude
    Returns a set of gpu_arch strings (e.g., 'gfx1150', 'gfx11X', 'gfx950').
    """
    try:
        result = subprocess.run(
            ["ctest", "-N", "--test-dir", f"{THEROCK_BIN_DIR}/{TEST_COMPONENT}"],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_archs = set()
        # Parse output for test names
        # Looking for pattern: Test #N: name-category-gpuarch-exclude
        # Example: Test #123: miopen_gtest-quick-gfx1150-exclude
        for line in result.stdout.split("\n"):
            # Look for lines containing test names with "-exclude" suffix
            if "-exclude" in line and "Test #" in line:
                # Extract the test name
                # Format: "Test #123: miopen_gtest-quick-gfx1150-exclude"
                match = re.search(r"Test\s+#\d+:\s+(.+)", line)
                if match:
                    test_name = match.group(1).strip()
                    # Extract gpu_arch from pattern: *-{category}-{gpu_arch}-exclude
                    # Split from the right to get the parts
                    parts = test_name.split("-")
                    if len(parts) >= 3 and parts[-1] == "exclude":
                        # The gpu_arch is the second-to-last part
                        gpu_arch = parts[-2]
                        # Verify it looks like a GPU arch (starts with gfx)
                        if gpu_arch.startswith("gfx"):
                            gpu_archs.add(gpu_arch)

        return gpu_archs
    except subprocess.CalledProcessError as e:
        print(f"Error running ctest -N: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: ctest not found. Make sure CMake/CTest is installed.",
            file=sys.stderr,
        )
        sys.exit(1)


def find_matching_gpu_arch(gpu_arch, available_gpu_archs):
    """
    Find the most specific GPU architecture that matches the given GPU.

    Tries in order from most specific to least specific:
    - Exact match (gfx1151)
    - Wildcard matches (gfx115X, gfx11X, etc.)

    Returns the matching architecture string or None if no match found.
    """
    # First, try exact match
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    # Generate possible wildcard patterns from most specific to least specific
    # For gfx1151: try gfx115X, gfx11X, gfx1X
    possible_patterns = []
    arch_str = gpu_arch

    # Generate patterns by replacing characters with X from right to left
    for i in range(len(arch_str) - 1, 0, -1):
        pattern = arch_str[:i] + "X"
        possible_patterns.append(pattern)

    # Try each pattern
    for pattern in possible_patterns:
        if pattern in available_gpu_archs:
            return pattern

    return None


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
            "8",
            "--test-dir",
            f"{THEROCK_BIN_DIR}/{TEST_COMPONENT}",
            "-V",  # Always run in verbose mode
            # Shards the tests by running a specific set of tests based on starting test (shard_index) and stride (total_shards)
            "--tests-information",
            f"{SHARD_INDEX},,{TOTAL_SHARDS}",
        ]
    )

    if gpu_arch.lower() in ["generic", "none", ""]:
        # For generic/unspecified GPU, exclude all GPU-specific tests
        cmd.extend(["-LE", "ex_gpu"])
        return cmd

    # Find the appropriate GPU exclusion architecture
    matching_arch = find_matching_gpu_arch(gpu_arch, available_gpu_archs)

    if matching_arch:
        # Run the specific GPU exclusion test using the ex_gpu label
        gpu_label = f"ex_gpu_{matching_arch}"
        cmd.extend(["-L", gpu_label])
        print(f"# Using GPU exclusion label: {gpu_label}")
    else:
        # No specific GPU exclusion found, run standard tests excluding all GPU-specific ones
        cmd.extend(["-LE", "ex_gpu"])
        print(f"# No GPU exclusion found for {gpu_arch}, excluding all ex_gpu tests")

    return cmd


def main():
    # Use only two categories for now - quick and standard - depending on TEST_TYPE.
    if TEST_TYPE and TEST_TYPE.lower() == "smoke":
        category = "quick"
    else:
        category = "standard"

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

    # Get available GPU exclusion tests from ctest
    print("# Discovering available GPU exclusion tests...")
    available_gpu_archs = get_available_gpu_exclusion_tests()

    if available_gpu_archs:
        print(f"# Found {len(available_gpu_archs)} GPU exclusion test(s)")
        print(f"# Available GPU architectures: {sorted(available_gpu_archs)}")
    else:
        print("# Warning: No GPU exclusion tests found")
    print()

    # Build the ctest command
    cmd = build_ctest_command(category, gpu_arch, available_gpu_archs)

    print(f"# Running: {' '.join(cmd)}")
    print()

    # Execute the command
    try:
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running ctest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
