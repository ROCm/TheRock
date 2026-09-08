#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lists the failed tests in a pytest JSON report, for a fresh-process retry.

Some failures are sticky: the first one leaves the process failing every later
test the same way, so only a new process tells a real failure apart from a
poisoned one. Observed on a pytest-xdist worker's first complex-gemm call and on
a code object failing to load.

A subTest failure is recorded as failed with no message, which costs nothing
here: a retry needs only the nodeid.

Example usage:

    python list_pytest_failed_tests.py logs/pytest_results_single.json
"""

import argparse
import json
import pathlib
import sys


def failed_in_report(report: dict) -> list[str]:
    """The nodeids a parsed JSON report records as failed.

    Separate from failed_tests so that a caller reading other fields of the same
    report, such as its exit code, parses it once. These reports hold a record
    per test and run to tens of megabytes.
    """
    return [
        test["nodeid"]
        for test in report.get("tests", [])
        if test.get("outcome") in ("failed", "error")
    ]


def failed_tests(report_path: pathlib.Path) -> list[str]:
    """The nodeids a JSON report records as failed."""
    return failed_in_report(json.loads(report_path.read_text()))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="list_pytest_failed_tests.py")
    p.add_argument(
        "reports",
        nargs="+",
        type=pathlib.Path,
        help="pytest-json-report files; missing ones are skipped",
    )
    p.add_argument(
        "--max-tests",
        type=int,
        default=40,
        help="Refuse to retry above this many failures (0 disables)",
    )
    args = p.parse_args(argv)

    failed: list[str] = []
    for report in args.reports:
        if not report.exists():
            print(f"{report}: not found, skipping", file=sys.stderr)
            continue
        failed += failed_tests(report)

    # The retry pass is serial, and a configuration this broken is something
    # other than a few poisoned workers.
    if args.max_tests and len(failed) > args.max_tests:
        print(
            f"{len(failed)} failures is more than --max-tests"
            f" {args.max_tests}, so nothing will be retried",
            file=sys.stderr,
        )
        return 1

    print("\n".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
