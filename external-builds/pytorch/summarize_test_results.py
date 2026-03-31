#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse JUnit XML test reports and write a GitHub Actions step summary.

Scans PyTorch's test-reports directory for JUnit XML files, extracts
failed/errored test cases, and writes a markdown summary table to
$GITHUB_STEP_SUMMARY so it appears in the workflow run UI.

Usage:
    python summarize_test_results.py [--reports-dir DIR] [--test-config CONFIG]
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit_xml(xml_path: Path) -> list[dict]:
    """Extract failed/errored test cases from a JUnit XML file."""
    results = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return results

    root = tree.getroot()

    # JUnit XML can have <testsuites><testsuite>... or just <testsuite>...
    testsuites = root.findall(".//testsuite")
    if root.tag == "testsuite":
        testsuites = [root]
    elif root.tag == "testsuites":
        testsuites = root.findall("testsuite")

    for suite in testsuites:
        for testcase in suite.findall("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time_s = testcase.get("time", "")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None:
                status = "FAILED"
                message = (failure.get("message") or "")[:200]
            elif error is not None:
                status = "ERROR"
                message = (error.get("message") or "")[:200]
            else:
                continue

            results.append(
                {
                    "file": classname,
                    "class": classname.rsplit(".", 1)[-1] if "." in classname else classname,
                    "test": name,
                    "status": status,
                    "message": message,
                    "time": time_s,
                }
            )

    return results


def collect_results(reports_dir: Path) -> list[dict]:
    """Walk the reports directory and collect all failures."""
    all_failures = []
    for xml_file in sorted(reports_dir.rglob("*.xml")):
        failures = parse_junit_xml(xml_file)
        for f in failures:
            f["report_file"] = str(xml_file.relative_to(reports_dir))
        all_failures.extend(failures)
    return all_failures


def derive_test_file(report_path: str) -> str:
    """Derive the PyTorch test file name from the report path.

    e.g. 'python-pytest/distributions.test_distributions/...' -> 'distributions/test_distributions'
    """
    parts = report_path.split("/")
    if len(parts) >= 2:
        test_dir_name = parts[1] if parts[0].startswith("python") else parts[0]
        return test_dir_name.replace(".", "/")
    return report_path


def write_summary(
    failures: list[dict],
    test_config: str,
    amdgpu_family: str,
    summary_file: str,
) -> None:
    """Write markdown summary to the given file (typically $GITHUB_STEP_SUMMARY)."""
    lines = []

    if not failures:
        lines.append("### All tests passed! :white_check_mark:")
        lines.append("")
    else:
        lines.append(f"### Failed Tests: {len(failures)}")
        lines.append("")
        lines.append("| Test File | Test Class | Test Name | Status | Error |")
        lines.append("|-----------|-----------|-----------|--------|-------|")

        seen = set()
        for f in failures:
            test_file = derive_test_file(f["report_file"])
            key = (test_file, f["class"], f["test"])
            if key in seen:
                continue
            seen.add(key)

            msg = f["message"].replace("|", "\\|").replace("\n", " ")[:100]
            lines.append(
                f"| {test_file} | {f['class']} | {f['test']} | {f['status']} | {msg} |"
            )

        lines.append("")
        lines.append(
            f"*Config: {test_config} | GPU: {amdgpu_family} | "
            f"Total failures: {len(seen)}*"
        )

    summary = "\n".join(lines) + "\n"

    if summary_file:
        with open(summary_file, "a") as fh:
            fh.write(summary)

    print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("external-builds/pytorch/pytorch/test/test-reports"),
        help="Path to the JUnit XML test-reports directory",
    )
    parser.add_argument(
        "--test-config",
        default=os.getenv("TEST_CONFIG", "unknown"),
    )
    parser.add_argument(
        "--amdgpu-family",
        default=os.getenv("AMDGPU_FAMILY", "unknown"),
    )
    args = parser.parse_args()

    if not args.reports_dir.is_dir():
        print(f"Reports directory not found: {args.reports_dir}")
        return 0

    failures = collect_results(args.reports_dir)

    summary_file = os.getenv("GITHUB_STEP_SUMMARY", "")

    write_summary(failures, args.test_config, args.amdgpu_family, summary_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
