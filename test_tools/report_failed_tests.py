#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Parse test results and report failed tests with structured metrics output.

This script reads structured test output (JUnit XML, GTest JSON) and produces
a metrics report in a format suitable for CI analysis.

Usage:
    python report_failed_tests.py --results-dir <path> --output-file <path> [options]

Supported formats:
- CTest JUnit XML: ctest-*.xml files (from --output-junit)
- GTest JSON: gtest-*.json files (from --gtest_output=json:)
- Stdout fallback: Parse test output when structured files are missing (e.g., timeout)

Timeout handling:
- CTest timeout: Captured in JUnit XML with failure type containing "Timeout"
- GTest killed mid-run: JSON not written; use --stdout-log for partial results
- GitHub Actions timeout: No files written; use --step-name and --exit-code for fallback
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestResult:
    """Result from parsing a single test result file."""

    component: str
    failed_tests: list[str] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    duration_seconds: float | None = None
    exit_code: int = 0
    status: str = "success"
    failure_reason: str | None = None


def parse_junit_xml(xml_file: Path) -> TestResult:
    """Parse a JUnit XML file and extract test results.

    Handles CTest JUnit XML format, including timeout detection.
    """
    # Extract component name from filename (ctest-<component>-shard*.xml)
    name = xml_file.stem
    component = name.replace("ctest-", "").rsplit("-shard", 1)[0]

    result = TestResult(component=component)

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Handle both <testsuites> and <testsuite> as root
        if root.tag == "testsuites":
            testsuites = root.findall("testsuite")
            # Get aggregate time from root if available
            if root.get("time"):
                result.duration_seconds = float(root.get("time", 0))
        elif root.tag == "testsuite":
            testsuites = [root]
        else:
            return result

        total_time = 0.0
        for testsuite in testsuites:
            suite_name = testsuite.get("name", "")
            suite_tests = int(testsuite.get("tests", 0))
            suite_failures = int(testsuite.get("failures", 0))
            suite_errors = int(testsuite.get("errors", 0))
            suite_skipped = int(testsuite.get("skipped", 0))
            suite_time = float(testsuite.get("time", 0))

            result.total_tests += suite_tests
            result.failed_count += suite_failures + suite_errors
            result.skipped_count += suite_skipped
            total_time += suite_time

            for testcase in testsuite.findall("testcase"):
                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None or error is not None:
                    test_name = testcase.get("name", "unknown")
                    classname = testcase.get("classname", "")

                    # Build full test name
                    if classname and classname != suite_name:
                        full_name = f"{classname}.{test_name}"
                    elif suite_name:
                        full_name = f"{suite_name}.{test_name}"
                    else:
                        full_name = test_name

                    # Check for timeout in failure type or message
                    fail_elem = failure if failure is not None else error
                    fail_type = fail_elem.get("type", "").lower()
                    fail_message = fail_elem.get("message", "").lower()
                    fail_text = (fail_elem.text or "").lower()

                    is_timeout = (
                        "timeout" in fail_type
                        or "timeout" in fail_message
                        or "timeout" in fail_text
                    )

                    if is_timeout:
                        result.timeout_count += 1
                        full_name = f"{full_name} (Timeout)"

                    result.failed_tests.append(full_name)

        if result.duration_seconds is None:
            result.duration_seconds = total_time

        result.passed_tests = (
            result.total_tests - result.failed_count - result.skipped_count
        )

        if result.failed_count > 0:
            result.status = "failure"
            result.exit_code = 1
            if result.timeout_count > 0:
                result.failure_reason = "timeout"

    except ET.ParseError as e:
        print(f"Warning: Failed to parse {xml_file}: {e}")
        result.status = "error"
        result.exit_code = 1
    except Exception as e:
        print(f"Warning: Error reading {xml_file}: {e}")
        result.status = "error"
        result.exit_code = 1

    return result


def parse_gtest_json(json_file: Path) -> TestResult:
    """Parse a GTest JSON file and extract test results.

    Args:
        json_file: Path to the GTest JSON file

    Returns:
        TestResult with parsed data
    """
    # Extract component name from filename (gtest-<component>-shard*.json)
    name = json_file.stem
    component = name.replace("gtest-", "").rsplit("-shard", 1)[0]

    result = TestResult(component=component)

    try:
        with open(json_file) as f:
            data = json.load(f)

        # GTest JSON structure:
        # { "testsuites": [ { "name": "...", "testsuite": [ { "name": "...", "failures": [...] } ] } ] }
        result.total_tests = data.get("tests", 0)
        result.failed_count = data.get("failures", 0)
        result.skipped_count = data.get("disabled", 0)
        result.passed_tests = result.total_tests - result.failed_count

        # Duration is in seconds (GTest reports time as string like "1.234s")
        time_str = data.get("time", "0s")
        if isinstance(time_str, str) and time_str.endswith("s"):
            result.duration_seconds = float(time_str[:-1])
        elif isinstance(time_str, (int, float)):
            result.duration_seconds = float(time_str)

        for testsuite in data.get("testsuites", []):
            suite_name = testsuite.get("name", "")
            for test in testsuite.get("testsuite", []):
                # Check if test has failures
                failures = test.get("failures", [])
                if failures:
                    test_name = test.get("name", "unknown")
                    full_name = f"{suite_name}.{test_name}" if suite_name else test_name
                    result.failed_tests.append(full_name)

        if result.failed_count > 0:
            result.status = "failure"
            result.exit_code = 1

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse {json_file}: {e}")
        result.status = "error"
        result.exit_code = 1
    except Exception as e:
        print(f"Warning: Error reading {json_file}: {e}")
        result.status = "error"
        result.exit_code = 1

    return result


def parse_stdout_log(log_file: Path, component: str) -> TestResult:
    """Parse test stdout/stderr log for test results when structured output is unavailable.

    This is a fallback for when tests are killed mid-run (e.g., timeout) and
    don't produce their normal JSON/XML output.

    Args:
        log_file: Path to the stdout/stderr log file
        component: Component name for the result

    Returns:
        TestResult with parsed data (may be partial)
    """
    result = TestResult(component=component)

    try:
        content = log_file.read_text(errors="replace")

        # Parse GTest-style output
        # [  FAILED  ] TestSuite.TestName (123 ms)
        failed_pattern = r"\[\s*FAILED\s*\]\s+(\S+)"
        failed_matches = re.findall(failed_pattern, content)
        result.failed_tests = list(set(failed_matches))  # Dedupe
        result.failed_count = len(result.failed_tests)

        # [==========] X tests from Y test suites ran. (Z ms total)
        summary_pattern = (
            r"\[==========\]\s+(\d+)\s+tests?\s+from\s+(\d+)\s+test\s+suites?\s+ran"
        )
        summary_match = re.search(summary_pattern, content)
        if summary_match:
            result.total_tests = int(summary_match.group(1))

        # [  PASSED  ] X tests.
        passed_pattern = r"\[\s*PASSED\s*\]\s+(\d+)\s+tests?"
        passed_match = re.search(passed_pattern, content)
        if passed_match:
            result.passed_tests = int(passed_match.group(1))

        # Check for CTest timeout marker
        # 1/1 Test #1: test_name .........***Timeout 600.25 sec
        ctest_timeout_pattern = (
            r"Test\s+#\d+:\s+(\S+)\s+\.+\*\*\*Timeout\s+([\d.]+)\s+sec"
        )
        timeout_matches = re.findall(ctest_timeout_pattern, content)
        if timeout_matches:
            result.timeout_count = len(timeout_matches)
            for test_name, duration in timeout_matches:
                timeout_test = f"{test_name} (Timeout)"
                if timeout_test not in result.failed_tests:
                    result.failed_tests.append(timeout_test)
                    result.failed_count += 1

        # Check for ctest summary
        # 0% tests passed, 1 tests failed out of 1
        ctest_summary_pattern = (
            r"(\d+)%?\s+tests?\s+passed,\s+(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)"
        )
        ctest_summary_match = re.search(ctest_summary_pattern, content)
        if ctest_summary_match:
            passed = int(ctest_summary_match.group(1))
            failed = int(ctest_summary_match.group(2))
            total = int(ctest_summary_match.group(3))
            # Only use if we didn't get better data from GTest output
            if result.total_tests == 0:
                result.total_tests = total
                result.passed_tests = passed
                result.failed_count = max(result.failed_count, failed)

        if result.failed_count > 0:
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
    """Create a fallback TestResult when no structured output is available.

    Used when tests fail without producing output files (e.g., GitHub Actions timeout).

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
            result.failed_tests = [f"{component} (GitHub Actions Timeout)"]
            result.timeout_count = 1
        elif failure_reason == "cancelled":
            result.failed_tests = [f"{component} (Cancelled)"]
        else:
            result.failed_tests = [
                f"{component} (Unknown failure, exit code {exit_code})"
            ]

        result.failed_count = 1

    return result


def find_and_parse_results(
    results_dir: Path,
    stdout_log: Path | None = None,
    step_name: str | None = None,
    fallback_exit_code: int | None = None,
) -> list[TestResult]:
    """Find and parse all test result files in the directory.

    Args:
        results_dir: Directory containing test result files
        stdout_log: Optional path to stdout/stderr log for fallback parsing
        step_name: Step name for fallback result if no files found
        fallback_exit_code: Exit code to use for fallback result

    Returns:
        List of TestResult objects
    """
    results = []
    found_structured_output = False

    if results_dir.exists():
        # Parse JUnit XML files (from ctest)
        for xml_file in sorted(results_dir.glob("ctest-*.xml")):
            print(f"Parsing: {xml_file.name}")
            result = parse_junit_xml(xml_file)
            results.append(result)
            found_structured_output = True

        # Parse GTest JSON files
        for json_file in sorted(results_dir.glob("gtest-*.json")):
            print(f"Parsing: {json_file.name}")
            result = parse_gtest_json(json_file)
            results.append(result)
            found_structured_output = True
    else:
        print(f"Results directory not found: {results_dir}")

    # Fallback: try stdout log if no structured output found
    if not found_structured_output and stdout_log and stdout_log.exists():
        print(f"No structured output found, parsing stdout log: {stdout_log}")
        component = step_name or "unknown"
        result = parse_stdout_log(stdout_log, component)
        if result.total_tests > 0 or result.failed_count > 0:
            results.append(result)
            found_structured_output = True

    # Ultimate fallback: create a failure record if we have exit code but no results
    if not found_structured_output and fallback_exit_code is not None and step_name:
        print(f"No test output found, creating fallback result for {step_name}")
        # Determine failure reason based on exit code patterns
        failure_reason = None
        if fallback_exit_code == 124:  # timeout command exit code
            failure_reason = "timeout"
        elif fallback_exit_code == 143:  # SIGTERM (128 + 15)
            failure_reason = "timeout"
        elif fallback_exit_code == 137:  # SIGKILL (128 + 9)
            failure_reason = "timeout"

        result = create_fallback_result(step_name, fallback_exit_code, failure_reason)
        results.append(result)

    return results


def generate_metrics_output(results: list[TestResult], step_name: str) -> dict:
    """Generate structured metrics output."""
    metrics = []

    for result in results:
        metric = {
            "exit_code": result.exit_code,
            "sub_step_name": f"CUSTOM_STEP_{result.component}",
            "status": result.status,
        }

        if result.duration_seconds is not None:
            metric["duration_seconds"] = round(result.duration_seconds, 2)

        if result.failed_tests:
            metric["failed_tests"] = result.failed_tests

        if result.failure_reason:
            metric["failure_reason"] = result.failure_reason

        if result.timeout_count > 0:
            metric["timeout_count"] = result.timeout_count

        if result.total_tests > 0:
            metric["total_tests"] = result.total_tests
            metric["passed_tests"] = result.passed_tests
            metric["failed_count"] = result.failed_count

        metrics.append(metric)

    output = {
        "metadata": {
            "exit_code": {"metric_type": "exit_code"},
            "duration_seconds": {"metric_type": "duration_s"},
            "sub_step_name": {"metric_type": "string"},
            "status": {"metric_type": "string"},
            "failure_reason": {"metric_type": "string"},
            "timeout_count": {"metric_type": "count"},
            "total_tests": {"metric_type": "count"},
            "passed_tests": {"metric_type": "count"},
            "failed_count": {"metric_type": "count"},
        },
        "metrics": metrics,
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Parse test results and report failed tests with structured metrics"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing test result files (JUnit XML, GTest JSON)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file for metrics JSON (default: results-dir/test-metrics.json)",
    )
    parser.add_argument(
        "--step-name",
        type=str,
        default="tests",
        help="Name of the test step for reporting",
    )
    parser.add_argument(
        "--stdout-log",
        type=Path,
        help="Path to stdout/stderr log for fallback parsing when structured output is missing",
    )
    parser.add_argument(
        "--exit-code",
        type=int,
        help="Test process exit code (used for fallback when no output files exist)",
    )

    args = parser.parse_args()

    results = find_and_parse_results(
        args.results_dir,
        stdout_log=args.stdout_log,
        step_name=args.step_name,
        fallback_exit_code=args.exit_code,
    )

    metrics_output = generate_metrics_output(results, step_name=args.step_name)

    # Determine output file path
    output_file = args.output_file
    if output_file is None:
        output_file = args.results_dir / "test-metrics.json"

    # Write metrics to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nMetrics written to: {output_file}")

    # Also print to stdout for visibility
    print(f"\n{'='*60}")
    print("TEST METRICS REPORT")
    print(f"{'='*60}")
    print(json.dumps(metrics_output, indent=2))

    # Return non-zero if any tests failed
    any_failures = any(r.status != "success" for r in results)
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
