#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fail a component test job when its CSAN report files contain diagnostics.

The pinned GPU CSAN reporter does not fail its process when it reports a race.
Scanning log_path files also catches reports from successful tests whose output
CTest suppresses. A clean run need not produce any report files.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")
CSAN_DIAGNOSTIC = re.compile(
    r"\b(?:WARNING|ERROR|FATAL|SUMMARY):\s*ConcurrencySanitizer:"
    r"|\bConcurrencySanitizer:\s*(?:CHECK failed|DEADLYSIGNAL)\b"
)
# Keep a large failure useful in the Actions console; the complete reports are
# retained for the workflow's artifact upload.
MAX_PRINTED_FINDINGS = 20


@dataclass(frozen=True)
class CsanFinding:
    path: Path
    line_number: int
    message: str


def scan_csan_reports(reports_dir: Path) -> tuple[list[CsanFinding], int]:
    """Return diagnostic lines and the number of per-process reports scanned."""
    if not reports_dir.is_dir():
        raise FileNotFoundError(
            f"CSAN report directory is missing: {reports_dir}; "
            "run configure_csan_env.py before the component tests"
        )
    findings = []
    report_count = 0
    for path in sorted(reports_dir.rglob("csan.*")):
        if not path.is_file():
            continue
        report_count += 1
        with path.open(encoding="utf-8", errors="replace") as report:
            for line_number, line in enumerate(report, 1):
                message = ANSI_COLOR.sub("", line.rstrip())
                if CSAN_DIAGNOSTIC.search(message):
                    findings.append(CsanFinding(path, line_number, message))
    return findings, report_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        findings, report_count = scan_csan_reports(args.reports_dir)
    except OSError as exc:
        print(f"Could not check CSAN reports: {exc}", file=sys.stderr)
        return 1
    if not findings:
        print(f"No CSAN diagnostics found in {report_count} report files.")
        return 0
    print(
        f"CSAN reported {len(findings)} diagnostic lines in {report_count} report "
        "files. Review the retained reports before treating this component as clean.",
        file=sys.stderr,
    )
    for finding in findings[:MAX_PRINTED_FINDINGS]:
        print(
            f"{finding.path}:{finding.line_number}: {finding.message}",
            file=sys.stderr,
        )
    if len(findings) > MAX_PRINTED_FINDINGS:
        print("Additional diagnostics remain in the report files.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
