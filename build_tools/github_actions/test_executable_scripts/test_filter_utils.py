"""
Reusable ctest utilities for label-based test filtering.

This module provides functions to run ctest with category labels (quick, standard,
comprehensive, full) and GPU architecture labels. It is designed to be imported by
individual test_<component>.py scripts so that each component retains its own
environment setup while using standardized ctest label filtering.

Usage:
    from test_filter_utils import run_ctest

    returncode = run_ctest(
        test_dir="/path/to/component/tests",
        env=my_env_vars,
        cwd="/path/to/therock",
    )
"""

import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path

VALID_TEST_CATEGORIES = {"quick", "standard", "comprehensive", "full"}

logging.basicConfig(level=logging.INFO)


def resolve_category(test_type: str | None) -> str:
    """Validate test_type and return the category. Falls back to 'quick' for invalid values."""
    category = test_type.lower() if test_type else "quick"
    if category not in VALID_TEST_CATEGORIES:
        print(
            f"ERROR: Invalid TEST_TYPE '{test_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_TEST_CATEGORIES))}. "
            f"Falling back to 'quick'.",
            file=sys.stderr,
        )
        category = "quick"
    return category


def extract_gpu_arch(amdgpu_families: str | None) -> str:
    """Extract gfx<xxx> pattern from an AMDGPU_FAMILIES string.

    Returns the matched architecture (e.g. 'gfx1151') or '' if not found.
    """
    if not amdgpu_families:
        return ""
    match = re.search(r"gfx[0-9a-zA-Z]+", amdgpu_families)
    if match:
        return match.group(0)
    print(
        f"# Warning: Could not extract GPU architecture from "
        f"AMDGPU_FAMILIES='{amdgpu_families}'"
    )
    return ""


def find_matching_gpu_arch(gpu_arch: str, available_gpu_archs: set[str]) -> str | None:
    """Find the most specific GPU architecture that matches the given GPU.

    Tries exact match first, then progressively shorter wildcard patterns
    (e.g. gfx115X, gfx11X).  Returns None if nothing matches.
    """
    if gpu_arch in available_gpu_archs:
        return gpu_arch

    for i in range(len(gpu_arch) - 1, 4, -1):
        pattern = gpu_arch[:i] + "X"
        if pattern in available_gpu_archs:
            return pattern

    return None


def discover_labels(test_dir: str) -> tuple[set[str], set[str]]:
    """Discover GPU architecture and category-exclude labels via ctest.

    Args:
        test_dir: Path to the component test directory.

    Returns:
        (gpu_archs, exclude_labels) where gpu_archs is a set like {'gfx115X'}
        and exclude_labels is a set like {'quick_exclude'}.

    Raises:
        SystemExit: if the test directory is missing, has no tests, or ctest fails.
    """
    test_path = Path(test_dir)
    if not test_path.exists() or not test_path.is_dir():
        print(f"Error: Test directory does not exist: {test_path}", file=sys.stderr)
        sys.exit(1)

    try:
        list_result = subprocess.run(
            ["ctest", "-N", "--test-dir", str(test_path)],
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
                f"Error: No tests found in {test_path}. Cannot run test suite.",
                file=sys.stderr,
            )
            sys.exit(1)

        result = subprocess.run(
            ["ctest", "--print-labels", "--test-dir", str(test_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        gpu_archs: set[str] = set()
        exclude_labels: set[str] = set()
        gpu_prefix = "ex_gpu_"
        exclude_suffix = "_exclude"
        for line in result.stdout.splitlines():
            label = line.strip()
            if label.startswith(gpu_prefix):
                arch = label[len(gpu_prefix) :]
                if arch.startswith("gfx"):
                    gpu_archs.add(arch)
            elif label.endswith(exclude_suffix):
                exclude_labels.add(label)

        return gpu_archs, exclude_labels
    except subprocess.CalledProcessError as e:
        print(f"Error running ctest: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: ctest not found. Make sure CMake/CTest is installed.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_ctest_command(
    test_dir: str,
    category: str,
    gpu_arch: str,
    available_gpu_archs: set[str],
    exclude_labels: set[str],
    *,
    parallel_count: int = 8,
    timeout_seconds: int = 7200,
    shard_index: int = 1,
    total_shards: int = 1,
) -> list[str]:
    """Build a ctest command with label-based category and GPU filtering.

    Returns a list of command arguments suitable for subprocess.run().
    """
    cmd = ["ctest"]

    le_patterns: list[str] = []
    include_labels = [category]

    category_exclude_label = f"{category}_exclude"
    if category_exclude_label in exclude_labels:
        le_patterns.append(category_exclude_label)
        print(f"# Excluding tests with label: {category_exclude_label}")

    if gpu_arch.lower() in ["generic", "none", ""]:
        le_patterns.append("ex_gpu")
    else:
        matching_arch = find_matching_gpu_arch(gpu_arch, available_gpu_archs)
        if matching_arch:
            gpu_label = f"ex_gpu_{matching_arch}"
            include_labels.append(gpu_label)
            print(f"# Using GPU suite label: {gpu_label}")
        else:
            le_patterns.append("ex_gpu")
            print(f"# No GPU suite found for {gpu_arch}, excluding all ex_gpu tests")

    for label in include_labels:
        cmd.extend(["-L", label])
    if le_patterns:
        cmd.extend(["-LE", "|".join(le_patterns)])

    cmd.extend(
        [
            "--output-on-failure",
            "--parallel",
            str(parallel_count),
            "--timeout",
            str(timeout_seconds),
            "--test-dir",
            test_dir,
            "-V",
            "--tests-information",
            f"{shard_index},,{total_shards}",
        ]
    )

    return cmd


def run_ctest(
    test_dir: str,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    test_type: str | None = None,
    amdgpu_families: str | None = None,
    parallel_count: int = 8,
    timeout_seconds: int = 7200,
    shard_index: int = 1,
    total_shards: int = 1,
) -> int:
    """Run ctest with label-based filtering for a component.

    This is the main entry point for component test scripts. It discovers
    labels, builds the ctest command, and executes it.

    Args:
        test_dir: Path to the component's test directory.
        env: Environment variables for the subprocess (defaults to os.environ).
        cwd: Working directory for ctest execution.
        test_type: Test category string (defaults to TEST_TYPE env var, then 'quick').
        amdgpu_families: GPU family string (defaults to AMDGPU_FAMILIES env var).
        parallel_count: Number of parallel ctest jobs.
        timeout_seconds: Per-test timeout in seconds.
        shard_index: Current shard index (1-based).
        total_shards: Total number of shards.

    Returns:
        Process return code (0 for success).
    """
    import os

    if test_type is None:
        test_type = os.getenv("TEST_TYPE", "quick")
    if amdgpu_families is None:
        amdgpu_families = os.getenv("AMDGPU_FAMILIES")

    category = resolve_category(test_type)
    gpu_arch = extract_gpu_arch(amdgpu_families)

    print(f"# Category: {category}")
    print(f"# GPU Architecture: {gpu_arch}")
    print(f"# Test Directory: {test_dir}")
    print()

    print("# Discovering available test labels...")
    available_gpu_archs, exclude_labels = discover_labels(test_dir)

    if available_gpu_archs:
        print(f"# Found {len(available_gpu_archs)} GPU suite test(s)")
        print(f"# Available GPU architectures: {sorted(available_gpu_archs)}")
    else:
        print("# Warning: No GPU specific test suites available")
    if exclude_labels:
        print(f"# Found exclude labels: {sorted(exclude_labels)}")
    print()

    cmd = build_ctest_command(
        test_dir,
        category,
        gpu_arch,
        available_gpu_archs,
        exclude_labels,
        parallel_count=parallel_count,
        timeout_seconds=timeout_seconds,
        shard_index=shard_index,
        total_shards=total_shards,
    )

    print(f"# Running: {' '.join(cmd)}")
    print()

    try:
        logging.info(f"++ Exec [{cwd or '.'}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running ctest: {e}", file=sys.stderr)
        return 1
