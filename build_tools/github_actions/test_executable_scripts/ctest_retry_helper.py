# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Helper module for retrying ctest tests that were not run due to parallelization issues.

This module provides functionality to parse ctest output and identify tests that
weren't executed (showing "Not Run" or "***Not Run" status), then retry them.
"""

import re
import subprocess
import sys
from pathlib import Path


def parse_not_run_tests(output: str) -> list[str]:
    """
    Parse ctest output to find tests that were "Not Run" or "***Not Run".

    Args:
        output: The stdout from a ctest execution

    Returns:
        A list of test names that weren't executed
    """
    not_run_tests = []
    # Match lines like "  1/100 Test  #42: test_name ........................   Not Run"
    # or "  1/100 Test  #42: test_name ........................***Not Run"
    pattern = re.compile(r"Test\s+#\d+:\s+(\S+)\s+\.+\s*\*{0,3}Not Run", re.IGNORECASE)

    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            test_name = match.group(1)
            not_run_tests.append(test_name)

    return not_run_tests


def retry_not_run_tests(
    base_cmd: list[str],
    not_run_tests: list[str],
    cwd: Path | str,
    environ_vars: dict | None = None,
    max_retries: int = 3,
) -> int:
    """
    Retry tests that were not run due to parallelization issues.

    This function removes the --parallel flag from the base command and runs
    the tests serially to avoid the same parallelization issues.

    Args:
        base_cmd: Base ctest command (list of arguments)
        not_run_tests: List of test names that weren't run
        cwd: Working directory for the command
        environ_vars: Optional environment variables to use (defaults to None)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        0 if all retries succeeded, non-zero otherwise
    """
    if not not_run_tests:
        return 0

    print(f"\n# Found {len(not_run_tests)} test(s) that were not run")
    print(f"# Tests to retry: {', '.join(not_run_tests)}")

    for attempt in range(1, max_retries + 1):
        print(f"\n# Retry attempt {attempt}/{max_retries} for not-run tests")

        # Build retry command - remove --parallel and its value to run tests serially
        retry_cmd = []
        skip_next = False
        for i, part in enumerate(base_cmd):
            if skip_next:
                skip_next = False
                continue
            if part == "--parallel":
                # Skip --parallel and its next value
                skip_next = True
                continue
            retry_cmd.append(part)

        # Add regex to match only the tests that weren't run
        test_regex = "|".join(re.escape(test) for test in not_run_tests)
        retry_cmd.extend(["-R", test_regex])

        print(f"# Running: {' '.join(retry_cmd)}")

        try:
            result = subprocess.run(
                retry_cmd,
                cwd=cwd,
                env=environ_vars,
                check=False,
                capture_output=True,
                text=True,
            )

            # Print output for visibility
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            # Check if any tests are still not run
            still_not_run = parse_not_run_tests(result.stdout)

            if not still_not_run:
                print(f"# All previously not-run tests completed successfully")
                if result.returncode != 0:
                    print(
                        f"# Warning: Some retried tests failed with exit code {result.returncode}"
                    )
                    return result.returncode
                return 0

            not_run_tests = still_not_run
            print(f"# {len(still_not_run)} test(s) still not run after retry")

        except Exception as e:
            print(f"# Error during retry: {e}", file=sys.stderr)
            return 1

    print(
        f"# Failed to run {len(not_run_tests)} test(s) after {max_retries} retries",
        file=sys.stderr,
    )
    return 1


def run_ctest_with_retry(
    cmd: list[str],
    cwd: Path | str,
    environ_vars: dict | None = None,
    max_retries: int = 3,
) -> int:
    """
    Run a ctest command and automatically retry any tests that were not run.

    This is a convenience function that combines running ctest with automatic
    retry of not-run tests.

    Args:
        cmd: ctest command (list of arguments)
        cwd: Working directory for the command
        environ_vars: Optional environment variables to use (defaults to None)
        max_retries: Maximum number of retry attempts for not-run tests (default: 3)

    Returns:
        Exit code from the test run (0 for success, non-zero for failure)
    """
    # Run the initial ctest command
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=environ_vars,
        check=False,
        capture_output=True,
        text=True,
    )

    # Print output for visibility
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Check for tests that were not run and retry them
    not_run_tests = parse_not_run_tests(result.stdout)
    if not_run_tests:
        retry_result = retry_not_run_tests(
            cmd, not_run_tests, cwd, environ_vars, max_retries
        )
        if retry_result != 0:
            return retry_result

    return result.returncode
