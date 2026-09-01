#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse test output from stdout and report failed tests with structured metrics."""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class TestFramework(Enum):
    GTEST = "gtest"
    CTEST = "ctest"


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    RUNNING = "running"


@dataclass
class TestCase:
    name: str
    status: TestStatus
    framework: TestFramework
    duration_ms: Optional[float] = None
    parent_test: Optional[str] = None


@dataclass
class ParsedTestOutput:
    tests: list[TestCase] = field(default_factory=list)
    total_tests: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    timeout_count: int = 0
    frameworks_detected: set[TestFramework] = field(default_factory=set)
    has_failures: bool = False
    ctest_context: Optional[str] = None


def _parse_gtest_output(content: str) -> list[TestCase]:
    seen_tests: dict[str, TestCase] = {}

    result_pattern = re.compile(
        r"\[\s*(OK|FAILED|SKIPPED)\s*\]\s+(\S+?)(?:,\s*where\s+GetParam|\s+\((\d+)\s*ms\)|$)"
    )
    run_pattern = re.compile(r"\[\s*RUN\s*\]\s+(\S+)")

    started_tests: set[str] = set()
    for match in run_pattern.finditer(content):
        started_tests.add(match.group(1))

    for match in result_pattern.finditer(content):
        status_str, name, duration_str = match.groups()
        if name.isdigit() or name.endswith("tests") or name.endswith("test"):
            continue
        status_map = {
            "OK": TestStatus.PASSED,
            "FAILED": TestStatus.FAILED,
            "SKIPPED": TestStatus.SKIPPED,
        }
        seen_tests[name] = TestCase(
            name=name,
            status=status_map.get(status_str, TestStatus.FAILED),
            framework=TestFramework.GTEST,
            duration_ms=float(duration_str) if duration_str else None,
        )

    summary_section = re.search(
        r"\[\s*FAILED\s*\]\s+\d+\s+tests?,\s+listed\s+below:(.*?)(?:\n\n|\Z)",
        content,
        re.DOTALL,
    )
    if summary_section:
        for match in re.finditer(
            r"\[\s*FAILED\s*\]\s+(\S+)\s*$", summary_section.group(1), re.MULTILINE
        ):
            name = match.group(1)
            if name.isdigit() or "TEST" in name.upper() or name in seen_tests:
                continue
            seen_tests[name] = TestCase(
                name=name, status=TestStatus.FAILED, framework=TestFramework.GTEST
            )

    for name in started_tests:
        if name not in seen_tests:
            seen_tests[name] = TestCase(
                name=name, status=TestStatus.RUNNING, framework=TestFramework.GTEST
            )

    return list(seen_tests.values())


def _parse_ctest_output(content: str) -> list[TestCase]:
    seen_tests: dict[str, TestCase] = {}

    exec_pattern = re.compile(
        r"\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.+\s*(\*\*\*)?(Passed|Failed|Timeout|Not Run)\s+([\d.]+)\s*sec"
    )
    for match in exec_pattern.finditer(content):
        name, _, status_str, duration_str = match.groups()
        status_map = {
            "Passed": TestStatus.PASSED,
            "Failed": TestStatus.FAILED,
            "Timeout": TestStatus.TIMEOUT,
            "Not Run": TestStatus.SKIPPED,
        }
        seen_tests[name] = TestCase(
            name=name,
            status=status_map.get(status_str, TestStatus.FAILED),
            framework=TestFramework.CTEST,
            duration_ms=float(duration_str) * 1000,
        )

    for match in re.finditer(
        r"^\s*\d+\s+-\s+(\S+)\s+\((Failed|Timeout)\)", content, re.MULTILINE
    ):
        name, status_str = match.groups()
        if name not in seen_tests:
            seen_tests[name] = TestCase(
                name=name,
                status=(
                    TestStatus.TIMEOUT if status_str == "Timeout" else TestStatus.FAILED
                ),
                framework=TestFramework.CTEST,
            )

    return list(seen_tests.values())


def _detect_frameworks(content: str) -> set[TestFramework]:
    frameworks = set()
    if re.search(r"\[\s*(RUN|OK|FAILED|PASSED)\s*\]", content):
        frameworks.add(TestFramework.GTEST)
    if re.search(r"Test\s+#\d+:", content) or re.search(
        r"ctest", content, re.IGNORECASE
    ):
        frameworks.add(TestFramework.CTEST)
    return frameworks


def _get_ctest_context(content: str) -> Optional[str]:
    for pattern in [r"Start\s+\d+:\s+(\S+)", r"\d+/\d+\s+Test\s+#\d+:\s+(\S+)"]:
        matches = list(re.finditer(pattern, content))
        if matches:
            return matches[-1].group(1)
    return None


def parse_test_output(content: str) -> ParsedTestOutput:
    result = ParsedTestOutput()
    result.frameworks_detected = _detect_frameworks(content)
    all_tests: list[TestCase] = []

    if TestFramework.GTEST in result.frameworks_detected:
        all_tests.extend(_parse_gtest_output(content))

    if TestFramework.CTEST in result.frameworks_detected:
        ctest_tests = _parse_ctest_output(content)
        if TestFramework.GTEST in result.frameworks_detected:
            ctest_context = _get_ctest_context(content)
            result.ctest_context = ctest_context
            for test in all_tests:
                if test.framework == TestFramework.GTEST and ctest_context:
                    test.parent_test = ctest_context
            for ctest_test in ctest_tests:
                if ctest_test.status in (TestStatus.TIMEOUT, TestStatus.FAILED):
                    all_tests.append(ctest_test)
        else:
            all_tests.extend(ctest_tests)

    result.tests = all_tests
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
            result.failed_count += 1
            result.has_failures = True

    return result


@dataclass
class FailedTest:
    name: str
    status: str
    is_outer: bool = False
    failure_reason: str | None = None


@dataclass
class TestResult:
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
    result = TestResult(component=component)
    try:
        content = log_file.read_text(errors="replace")
        parsed = parse_test_output(content)
        result.total_tests = parsed.total_tests
        result.passed_tests = parsed.passed_count
        result.failed_count = parsed.failed_count
        result.skipped_count = parsed.skipped_count
        result.timeout_count = parsed.timeout_count

        for test in parsed.tests:
            if test.status == TestStatus.PASSED:
                continue
            is_outer = test.framework == TestFramework.CTEST
            if test.status == TestStatus.FAILED:
                result.failed_tests.append(
                    FailedTest(name=test.name, status="failure", is_outer=is_outer)
                )
            elif test.status == TestStatus.TIMEOUT:
                result.failed_tests.append(
                    FailedTest(
                        name=test.name,
                        status="timeout",
                        is_outer=is_outer,
                        failure_reason="timeout",
                    )
                )
                result.timeout_count += 1
            elif test.status == TestStatus.RUNNING:
                result.failed_tests.append(
                    FailedTest(
                        name=test.name,
                        status="failure",
                        is_outer=is_outer,
                        failure_reason="interrupted",
                    )
                )

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
    result = TestResult(component=component, exit_code=exit_code)
    if exit_code != 0:
        result.status = "failure"
        result.failure_reason = failure_reason
        if failure_reason == "timeout":
            result.failed_tests.append(
                FailedTest(
                    name=component,
                    status="timeout",
                    is_outer=True,
                    failure_reason="timeout",
                )
            )
            result.timeout_count = 1
        elif failure_reason == "cancelled":
            result.failed_tests.append(
                FailedTest(
                    name=component,
                    status="failure",
                    is_outer=True,
                    failure_reason="cancelled",
                )
            )
        else:
            result.failed_tests.append(
                FailedTest(
                    name=f"{component} (exit code {exit_code})",
                    status="failure",
                    is_outer=True,
                )
            )
        result.failed_count = 1
    return result


def find_and_parse_results(
    stdout_log: Path | None = None,
    step_name: str | None = None,
    fallback_exit_code: int | None = None,
) -> list[TestResult]:
    results = []
    component = step_name or "unknown"

    if stdout_log and stdout_log.exists():
        print(f"Parsing stdout log: {stdout_log}")
        result = parse_stdout_log(stdout_log, component)
        if result.total_tests > 0 or result.failed_count > 0 or result.failed_tests:
            results.append(result)
        elif fallback_exit_code is not None and fallback_exit_code != 0:
            print("No test results found in log, using exit code fallback")
            failure_reason = (
                "timeout" if fallback_exit_code in (124, 143, 137) else None
            )
            results.append(
                create_fallback_result(component, fallback_exit_code, failure_reason)
            )
    elif fallback_exit_code is not None and step_name:
        print(f"No stdout log found, creating fallback result for {step_name}")
        failure_reason = "timeout" if fallback_exit_code in (124, 143, 137) else None
        results.append(
            create_fallback_result(step_name, fallback_exit_code, failure_reason)
        )

    return results


def generate_metrics_output(results: list[TestResult], step_name: str) -> dict:
    metrics = []
    for result in results:
        for failed_test in result.failed_tests:
            metric = {"exit_code": 1, "status": failed_test.status}
            if failed_test.is_outer:
                metric["step_name"] = failed_test.name
            else:
                metric["sub_step_name"] = failed_test.name
            if failed_test.failure_reason:
                metric["failure_reason"] = failed_test.failure_reason
            metrics.append(metric)

    return {
        "metadata": {
            "exit_code": {"metric_type": "exit_code"},
            "step_name": {"metric_type": "string"},
            "sub_step_name": {"metric_type": "string"},
            "status": {"metric_type": "string"},
            "failure_reason": {"metric_type": "string"},
        },
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse test results and report failures"
    )
    parser.add_argument("--stdout-log", type=Path, help="Path to stdout log file")
    parser.add_argument("--output-file", type=Path, help="Output file for metrics JSON")
    parser.add_argument("--step-name", type=str, default="tests", help="Test step name")
    parser.add_argument("--exit-code", type=int, help="Test exit code for fallback")
    parser.add_argument("--results-dir", type=Path, help="(Deprecated)")
    args = parser.parse_args()

    if args.results_dir and not args.output_file:
        args.output_file = args.results_dir / "test-metrics.json"

    results = find_and_parse_results(
        stdout_log=args.stdout_log,
        step_name=args.step_name,
        fallback_exit_code=args.exit_code,
    )
    metrics_output = generate_metrics_output(results, step_name=args.step_name)

    output_file = args.output_file or Path("test-metrics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\nMetrics written to: {output_file}")
    print(f"\n{'='*60}\nTEST METRICS REPORT\n{'='*60}")
    print(
        json.dumps(metrics_output["metrics"], indent=2)
        if metrics_output["metrics"]
        else "No test failures detected."
    )

    return 1 if any(r.status != "success" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
