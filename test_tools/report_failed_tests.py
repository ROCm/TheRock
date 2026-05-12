#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Parse test results and report failed tests with structured metrics output.

This script parses test output from stdout/stderr and produces a metrics report
suitable for CI analysis. It supports multiple test frameworks:

- GTest: Parses [FAILED], [PASSED] markers from stdout
- CTest: Parses test execution lines, timeouts, and failure summaries

When ctest wraps gtest (common pattern), it extracts both the inner gtest failures
and the outer ctest status, using different field names:
- Inner tests (gtest): sub_step_name
- Outer wrapper (ctest): step_name

Usage:
    python report_failed_tests.py --stdout-log <path> --output-file <path> [options]

Examples:
    # Parse stdout log and report failures
    python report_failed_tests.py --stdout-log build/test.log --step-name hip-tests

    # With exit code for fallback when no test output is found
    python report_failed_tests.py --stdout-log build/test.log --step-name hip-tests --exit-code 1
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from parse_test_output import (
    ParsedTestOutput,
    TestFramework,
    TestStatus,
    parse_test_output,
)


@dataclass
class FailedTest:
    """A single failed test with metadata."""

    name: str
    status: str  # "failure", "timeout", "interrupted"
    is_outer: bool = False  # True for ctest wrapper, False for inner tests
    failure_reason: str | None = None


@dataclass
class TestResult:
    """Aggregated test result for a component."""

    component: str
    failed_tests: list[FailedTest] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    exit_code: int = 0
    status: str = "success"
    failure_reason: str | None = None


def parse_stdout_log(log_file: Path, component: str) -> TestResult:
    """Parse test stdout/stderr log using the new parser.

    Args:
        log_file: Path to the stdout/stderr log file
        component: Component name for the result

    Returns:
        TestResult with parsed data
    """
    result = TestResult(component=component)

    try:
        content = log_file.read_text(errors="replace")
        parsed = parse_test_output(content)

        # Convert parsed output to TestResult
        result.total_tests = parsed.total_tests
        result.passed_tests = parsed.passed_count
        result.failed_count = parsed.failed_count
        result.skipped_count = parsed.skipped_count
        result.timeout_count = parsed.timeout_count

        # Build failed test list with framework info
        for test in parsed.tests:
            if test.status == TestStatus.PASSED:
                continue

            # Determine if this is an outer (ctest) or inner (gtest) test
            is_outer = test.framework == TestFramework.CTEST

            if test.status == TestStatus.FAILED:
                failed = FailedTest(
                    name=test.name,
                    status="failure",
                    is_outer=is_outer,
                )
                result.failed_tests.append(failed)
            elif test.status == TestStatus.TIMEOUT:
                failed = FailedTest(
                    name=test.name,
                    status="timeout",
                    is_outer=is_outer,
                    failure_reason="timeout",
                )
                result.failed_tests.append(failed)
                result.timeout_count += 1
            elif test.status == TestStatus.RUNNING:
                failed = FailedTest(
                    name=test.name,
                    status="failure",
                    is_outer=is_outer,
                    failure_reason="interrupted",
                )
                result.failed_tests.append(failed)

        if parsed.has_failures:
            result.status = "failure"
            result.exit_code = 1
            if result.timeout_count > 0:
                result.failure_reason = "timeout"

    except Exception as e:
        print(f"Warning: Error parsing stdout log {log_file}: {e}")
        result.status = "error"
        result.exit_code = 1

    return result


def create_fallback_result(
    component: str, exit_code: int, failure_reason: str | None = None
) -> TestResult:
    """Create a fallback TestResult when no test output is available.

    Used when tests fail without producing parseable output.

    Args:
        component: Component name
        exit_code: Process exit code (non-zero indicates failure)
        failure_reason: Optional reason for failure (e.g., "timeout", "cancelled")

    Returns:
        TestResult indicating the failure
    """
    result = TestResult(component=component)
    result.exit_code = exit_code

    if exit_code != 0:
        result.status = "failure"
        result.failure_reason = failure_reason

        if failure_reason == "timeout":
            failed = FailedTest(
                name=component,
                status="timeout",
                is_outer=True,
                failure_reason="timeout",
            )
            result.timeout_count = 1
        elif failure_reason == "cancelled":
            failed = FailedTest(
                name=component,
                status="failure",
                is_outer=True,
                failure_reason="cancelled",
            )
        else:
            failed = FailedTest(
                name=f"{component} (exit code {exit_code})",
                status="failure",
                is_outer=True,
            )

        result.failed_tests.append(failed)
        result.failed_count = 1

    return result


def find_and_parse_results(
    stdout_log: Path | None = None,
    step_name: str | None = None,
    fallback_exit_code: int | None = None,
) -> list[TestResult]:
    """Parse test results from stdout log.

    Args:
        stdout_log: Path to stdout/stderr log file
        step_name: Step name for the result
        fallback_exit_code: Exit code to use for fallback result

    Returns:
        List of TestResult objects
    """
    results = []
    component = step_name or "unknown"

    # Parse stdout log
    if stdout_log and stdout_log.exists():
        print(f"Parsing stdout log: {stdout_log}")
        result = parse_stdout_log(stdout_log, component)
        if result.total_tests > 0 or result.failed_count > 0 or result.failed_tests:
            results.append(result)
        elif fallback_exit_code is not None and fallback_exit_code != 0:
            # No test output found but we have a failure exit code
            print(f"No test results found in log, using exit code fallback")
            failure_reason = _get_failure_reason(fallback_exit_code)
            result = create_fallback_result(
                component, fallback_exit_code, failure_reason
            )
            results.append(result)
    elif fallback_exit_code is not None and step_name:
        # No log file, use exit code fallback
        print(f"No stdout log found, creating fallback result for {step_name}")
        failure_reason = _get_failure_reason(fallback_exit_code)
        result = create_fallback_result(step_name, fallback_exit_code, failure_reason)
        results.append(result)

    return results


def _get_failure_reason(exit_code: int) -> str | None:
    """Determine failure reason from exit code.

    Args:
        exit_code: Process exit code

    Returns:
        Failure reason string or None
    """
    # Common timeout/kill exit codes
    if exit_code == 124:  # timeout command exit code
        return "timeout"
    elif exit_code == 143:  # SIGTERM (128 + 15)
        return "timeout"
    elif exit_code == 137:  # SIGKILL (128 + 9)
        return "timeout"
    return None


def generate_metrics_output(results: list[TestResult], step_name: str) -> dict:
    """Generate structured metrics output.

    Creates one metric entry per failed test:
    - Inner tests (gtest): uses sub_step_name
    - Outer wrapper (ctest): uses step_name

    Args:
        results: List of TestResult objects
        step_name: Name of the test step

    Returns:
        Dict with metadata and metrics suitable for JSON output
    """
    metrics = []

    for result in results:
        for failed_test in result.failed_tests:
            metric = {
                "exit_code": 1,
                "status": failed_test.status,
            }

            # Use step_name for outer (ctest) tests, sub_step_name for inner tests
            if failed_test.is_outer:
                metric["step_name"] = failed_test.name
            else:
                metric["sub_step_name"] = failed_test.name

            if failed_test.failure_reason:
                metric["failure_reason"] = failed_test.failure_reason

            metrics.append(metric)

    output = {
        "metadata": {
            "exit_code": {"metric_type": "exit_code"},
            "step_name": {"metric_type": "string"},
            "sub_step_name": {"metric_type": "string"},
            "status": {"metric_type": "string"},
            "failure_reason": {"metric_type": "string"},
        },
        "metrics": metrics,
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Parse test results and report failed tests with structured metrics"
    )
    parser.add_argument(
        "--stdout-log",
        type=Path,
        help="Path to stdout/stderr log file to parse",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for metrics JSON",
    )
    parser.add_argument(
        "--step-name",
        type=str,
        default="tests",
        help="Name of the test step for reporting",
    )
    parser.add_argument(
        "--exit-code",
        type=int,
        help="Test process exit code (used for fallback when no test output found)",
    )
    # Legacy argument for backwards compatibility
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="(Deprecated) Directory containing test result files",
    )

    args = parser.parse_args()

    # Handle legacy results-dir argument
    if args.results_dir and not args.output_file:
        args.output_file = args.results_dir / "test-metrics.json"

    results = find_and_parse_results(
        stdout_log=args.stdout_log,
        step_name=args.step_name,
        fallback_exit_code=args.exit_code,
    )

    metrics_output = generate_metrics_output(results, step_name=args.step_name)

    # Determine output file path
    output_file = args.output_file
    if output_file is None:
        output_file = Path("test-metrics.json")

    # Write metrics to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nMetrics written to: {output_file}")

    # Also print to stdout for visibility
    print(f"\n{'='*60}")
    print("TEST METRICS REPORT")
    print(f"{'='*60}")
    if metrics_output["metrics"]:
        print(json.dumps(metrics_output["metrics"], indent=2))
    else:
        print("No test failures detected.")

    # Return non-zero if any tests failed
    any_failures = any(r.status != "success" for r in results)
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
