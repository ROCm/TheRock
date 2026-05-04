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
"""

import argparse
import json
import os
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
    duration_seconds: float | None = None
    exit_code: int = 0
    status: str = "success"


def parse_junit_xml(xml_file: Path) -> TestResult:
    """Parse a JUnit XML file and extract test results.

    Args:
        xml_file: Path to the JUnit XML file

    Returns:
        TestResult with parsed data
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
            suite_time = float(testsuite.get("time", 0))

            result.total_tests += suite_tests
            result.failed_count += suite_failures + suite_errors
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

                    result.failed_tests.append(full_name)

        if result.duration_seconds is None:
            result.duration_seconds = total_time

        result.passed_tests = result.total_tests - result.failed_count

        if result.failed_count > 0:
            result.status = "failure"
            result.exit_code = 1

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


def find_and_parse_results(results_dir: Path) -> list[TestResult]:
    """Find and parse all test result files in the directory.

    Args:
        results_dir: Directory containing test result files

    Returns:
        List of TestResult objects
    """
    results = []

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    # Parse JUnit XML files (from ctest)
    for xml_file in sorted(results_dir.glob("ctest-*.xml")):
        print(f"Parsing: {xml_file.name}")
        result = parse_junit_xml(xml_file)
        results.append(result)

    # Parse GTest JSON files
    for json_file in sorted(results_dir.glob("gtest-*.json")):
        print(f"Parsing: {json_file.name}")
        result = parse_gtest_json(json_file)
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

        metrics.append(metric)

    output = {
        "metadata": {
            "exit_code": {"metric_type": "exit_code"},
            "duration_seconds": {"metric_type": "duration_s"},
            "sub_step_name": {"metric_type": "string"},
            "status": {"metric_type": "string"},
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

    args = parser.parse_args()

    results = find_and_parse_results(args.results_dir)

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
