#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Parse test output from stdout/stderr for multiple test frameworks.

Supports:
- GTest: Parses [FAILED], [PASSED], [RUN] markers
- CTest: Parses test execution lines, timeouts, and summary

Future support planned for:
- pytest
- Catch2

Usage:
    from parse_test_output import parse_test_output, TestFramework

    results = parse_test_output(log_content)
    for result in results:
        print(f"{result.framework}: {result.name} - {result.status}")
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TestFramework(Enum):
    """Supported test frameworks."""

    GTEST = "gtest"
    CTEST = "ctest"
    PYTEST = "pytest"  # Future
    CATCH2 = "catch2"  # Future
    UNKNOWN = "unknown"


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    RUNNING = "running"  # Started but no completion seen


@dataclass
class TestCase:
    """A single test case result."""

    name: str
    status: TestStatus
    framework: TestFramework
    duration_ms: Optional[float] = None
    failure_message: Optional[str] = None
    # For nested tests (e.g., gtest inside ctest)
    parent_test: Optional[str] = None


@dataclass
class ParsedTestOutput:
    """Complete parsed test output."""

    tests: list[TestCase] = field(default_factory=list)
    # Summary info from test framework
    total_tests: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    # Which frameworks were detected
    frameworks_detected: set[TestFramework] = field(default_factory=set)
    # Overall status
    has_failures: bool = False
    # Raw ctest test that was running when inner tests failed
    ctest_context: Optional[str] = None


# =============================================================================
# GTest Parser
# =============================================================================


def parse_gtest_output(content: str) -> list[TestCase]:
    """Parse GTest output from stdout.

    GTest output format:
        [ RUN      ] TestSuite.TestName
        [       OK ] TestSuite.TestName (123 ms)
        [  FAILED  ] TestSuite.TestName (456 ms)
        [  SKIPPED ] TestSuite.TestName (0 ms)

    Summary:
        [==========] X tests from Y test suites ran. (Z ms total)
        [  PASSED  ] X tests.
        [  FAILED  ] X tests, listed below:
        [  FAILED  ] TestSuite.TestName1
        [  FAILED  ] TestSuite.TestName2

    Args:
        content: Raw stdout/stderr content

    Returns:
        List of TestCase objects
    """
    tests = []
    seen_tests: dict[str, TestCase] = {}

    # Pattern for individual test results with timing
    # [       OK ] TestSuite.TestName (123 ms)
    # [  FAILED  ] TestSuite.TestName (456 ms)
    # Note: Test names contain dots (Suite.Test) so we use a pattern that captures the full name
    result_pattern = re.compile(
        r"\[\s*(OK|FAILED|SKIPPED)\s*\]\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_/]*)+)(?:\s+\((\d+)\s*ms\))?"
    )

    # Pattern for tests listed in the failure summary (no timing)
    # [  FAILED  ] TestSuite.TestName
    # These appear at the end after "X tests, listed below:"
    summary_failed_pattern = re.compile(r"\[\s*FAILED\s*\]\s+(\S+)\s*$", re.MULTILINE)

    # Pattern for [ RUN ] to track started tests
    run_pattern = re.compile(r"\[\s*RUN\s*\]\s+(\S+)")

    # Track which tests started
    started_tests: set[str] = set()
    for match in run_pattern.finditer(content):
        started_tests.add(match.group(1))

    # Parse individual test results
    for match in result_pattern.finditer(content):
        status_str, name, duration_str = match.groups()

        # Skip summary lines like "[  PASSED  ] 42 tests."
        if name.isdigit() or name.endswith("tests") or name.endswith("test"):
            continue

        status_map = {
            "OK": TestStatus.PASSED,
            "FAILED": TestStatus.FAILED,
            "SKIPPED": TestStatus.SKIPPED,
        }
        status = status_map.get(status_str, TestStatus.FAILED)
        duration = float(duration_str) if duration_str else None

        test = TestCase(
            name=name,
            status=status,
            framework=TestFramework.GTEST,
            duration_ms=duration,
        )
        seen_tests[name] = test

    # Also catch failed tests from the summary section
    # Look for the failure summary section
    summary_section = re.search(
        r"\[\s*FAILED\s*\]\s+\d+\s+tests?,\s+listed\s+below:(.*?)(?:\n\n|\Z)",
        content,
        re.DOTALL,
    )
    if summary_section:
        for match in summary_failed_pattern.finditer(summary_section.group(1)):
            name = match.group(1)
            # Skip if it's a count like "1 FAILED TEST"
            if name.isdigit() or "TEST" in name.upper():
                continue
            if name not in seen_tests:
                seen_tests[name] = TestCase(
                    name=name,
                    status=TestStatus.FAILED,
                    framework=TestFramework.GTEST,
                )

    # Check for tests that started but never completed (likely killed/timeout)
    for name in started_tests:
        if name not in seen_tests:
            seen_tests[name] = TestCase(
                name=name,
                status=TestStatus.RUNNING,
                framework=TestFramework.GTEST,
            )

    tests = list(seen_tests.values())
    return tests


# =============================================================================
# CTest Parser
# =============================================================================


def parse_ctest_output(content: str) -> list[TestCase]:
    """Parse CTest output from stdout.

    CTest output format:
        Start  1: test_name
        1/10 Test  #1: test_name ................   Passed    1.23 sec
        2/10 Test  #2: another_test .............***Failed    2.34 sec
        3/10 Test  #3: timeout_test .............***Timeout  600.00 sec

    Summary:
        90% tests passed, 1 tests failed out of 10

        The following tests FAILED:
            2 - another_test (Failed)
            3 - timeout_test (Timeout)

    Args:
        content: Raw stdout/stderr content

    Returns:
        List of TestCase objects
    """
    tests = []
    seen_tests: dict[str, TestCase] = {}

    # Pattern for test execution lines
    # 1/10 Test  #1: test_name ................   Passed    1.23 sec
    # 2/10 Test  #2: test_name ................***Failed    2.34 sec
    # 3/10 Test  #3: test_name ................***Timeout  600.00 sec
    exec_pattern = re.compile(
        r"\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(\*\*\*)?(Passed|Failed|Timeout|Not Run)\s+([\d.]+)\s*sec"
    )

    for match in exec_pattern.finditer(content):
        name, stars, status_str, duration_str = match.groups()

        status_map = {
            "Passed": TestStatus.PASSED,
            "Failed": TestStatus.FAILED,
            "Timeout": TestStatus.TIMEOUT,
            "Not Run": TestStatus.SKIPPED,
        }
        status = status_map.get(status_str, TestStatus.FAILED)
        duration_ms = float(duration_str) * 1000  # Convert to ms

        test = TestCase(
            name=name,
            status=status,
            framework=TestFramework.CTEST,
            duration_ms=duration_ms,
        )
        seen_tests[name] = test

    # Also parse the failure summary section
    # The following tests FAILED:
    #     2 - test_name (Failed)
    #     3 - test_name (Timeout)
    summary_pattern = re.compile(
        r"^\s*\d+\s+-\s+(\S+)\s+\((Failed|Timeout)\)", re.MULTILINE
    )
    for match in summary_pattern.finditer(content):
        name, status_str = match.groups()
        if name not in seen_tests:
            status = (
                TestStatus.TIMEOUT if status_str == "Timeout" else TestStatus.FAILED
            )
            seen_tests[name] = TestCase(
                name=name,
                status=status,
                framework=TestFramework.CTEST,
            )

    tests = list(seen_tests.values())
    return tests


# =============================================================================
# Main Parser
# =============================================================================


def detect_frameworks(content: str) -> set[TestFramework]:
    """Detect which test frameworks are present in the output.

    Args:
        content: Raw stdout/stderr content

    Returns:
        Set of detected frameworks
    """
    frameworks = set()

    # GTest markers
    if re.search(r"\[\s*(RUN|OK|FAILED|PASSED)\s*\]", content):
        frameworks.add(TestFramework.GTEST)

    # CTest markers
    if re.search(r"Test\s+#\d+:", content) or re.search(
        r"ctest", content, re.IGNORECASE
    ):
        frameworks.add(TestFramework.CTEST)

    # Future: pytest markers
    # if re.search(r"={3,}\s*(FAILURES|ERRORS)\s*={3,}", content):
    #     frameworks.add(TestFramework.PYTEST)

    # Future: Catch2 markers
    # if re.search(r"All tests passed|test cases?.*failed", content, re.IGNORECASE):
    #     frameworks.add(TestFramework.CATCH2)

    return frameworks


def get_ctest_context(content: str) -> Optional[str]:
    """Extract the ctest test name that was running when failures occurred.

    Useful for associating inner gtest failures with their outer ctest test.

    Args:
        content: Raw stdout/stderr content

    Returns:
        The ctest test name if found, None otherwise
    """
    # Look for the most recent "Start N: test_name" or test execution line
    # before any gtest output
    start_pattern = re.compile(r"Start\s+\d+:\s+(\S+)")
    matches = list(start_pattern.finditer(content))
    if matches:
        return matches[-1].group(1)

    # Try test execution line
    exec_pattern = re.compile(r"\d+/\d+\s+Test\s+#\d+:\s+(\S+)")
    matches = list(exec_pattern.finditer(content))
    if matches:
        return matches[-1].group(1)

    return None


def parse_test_output(content: str) -> ParsedTestOutput:
    """Parse test output and extract all test results.

    This is the main entry point. It detects frameworks and parses accordingly.
    When both ctest and gtest are detected (nested), it associates gtest failures
    with their parent ctest test.

    Args:
        content: Raw stdout/stderr content from test execution

    Returns:
        ParsedTestOutput with all parsed test cases and summary info
    """
    result = ParsedTestOutput()
    result.frameworks_detected = detect_frameworks(content)

    all_tests: list[TestCase] = []

    # Parse each detected framework
    if TestFramework.GTEST in result.frameworks_detected:
        gtest_tests = parse_gtest_output(content)
        all_tests.extend(gtest_tests)

    if TestFramework.CTEST in result.frameworks_detected:
        ctest_tests = parse_ctest_output(content)

        # If we have both ctest and gtest, the gtest results are the inner tests
        # Associate them with the ctest parent
        if TestFramework.GTEST in result.frameworks_detected:
            ctest_context = get_ctest_context(content)
            result.ctest_context = ctest_context

            # Mark gtest tests with their parent
            for test in all_tests:
                if test.framework == TestFramework.GTEST and ctest_context:
                    test.parent_test = ctest_context

            # Include ctest failures/timeouts alongside inner gtest failures
            # This captures both the specific failing tests AND the outer wrapper status
            for ctest_test in ctest_tests:
                if ctest_test.status in (TestStatus.TIMEOUT, TestStatus.FAILED):
                    all_tests.append(ctest_test)
        else:
            # No gtest, just add ctest results
            all_tests.extend(ctest_tests)

    result.tests = all_tests

    # Calculate summary
    for test in all_tests:
        result.total_tests += 1
        if test.status == TestStatus.PASSED:
            result.passed_count += 1
        elif test.status == TestStatus.FAILED:
            result.failed_count += 1
            result.has_failures = True
        elif test.status == TestStatus.TIMEOUT:
            result.timeout_count += 1
            result.has_failures = True
        elif test.status == TestStatus.SKIPPED:
            result.skipped_count += 1
        elif test.status == TestStatus.RUNNING:
            # Test started but never completed - treat as failure
            result.failed_count += 1
            result.has_failures = True

    return result


def get_failed_test_names(content: str) -> list[str]:
    """Convenience function to get just the names of failed tests.

    Args:
        content: Raw stdout/stderr content

    Returns:
        List of failed test names with status annotations
    """
    result = parse_test_output(content)
    failed_names = []

    for test in result.tests:
        if test.status == TestStatus.FAILED:
            failed_names.append(test.name)
        elif test.status == TestStatus.TIMEOUT:
            failed_names.append(f"{test.name} (Timeout)")
        elif test.status == TestStatus.RUNNING:
            failed_names.append(f"{test.name} (Interrupted)")

    return failed_names
