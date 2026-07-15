#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Validate Python wheel artifacts before publishing.

Checks include:
  * Wheel files must be valid ZIP archives.
  * Wheel size must be greater than a configurable minimal threshold.
  * Optional metadata is collected for later inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import zipfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel-dir",
        required=True,
        help="Directory containing built wheel files",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional output JSON report path",
    )
    parser.add_argument(
        "--min-size-bytes",
        type=int,
        default=1024,
        help="Minimum acceptable wheel size (defaults to 1 KiB)",
    )
    return parser.parse_args()


def _collect_wheels(wheel_dir: Path) -> List[Path]:
    if not wheel_dir.exists():
        raise FileNotFoundError(f"Wheel directory '{wheel_dir}' does not exist")
    return [path for path in wheel_dir.rglob("*.whl") if path.is_file()]


def _validate_wheel(path: Path, min_size: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "filename": str(path),
        "size_bytes": path.stat().st_size,
        "valid_zip": False,
        "error": None,
    }

    if record["size_bytes"] < min_size:
        record["error"] = f"Wheel size {record['size_bytes']} bytes is below threshold ({min_size} bytes)"
        return record

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                record["error"] = f"Corrupt member '{bad_member}'"
                return record
    except zipfile.BadZipFile as exc:
        record["error"] = f"Bad ZIP file: {exc}"
        return record

    record["valid_zip"] = True
    return record


def main() -> int:
    args = _parse_args()
    wheel_dir = Path(args.wheel_dir).resolve()
    wheels = _collect_wheels(wheel_dir)

    if not wheels:
        print(f"[validate_wheels] No wheel files found under '{wheel_dir}'", file=sys.stderr)
        return 1

    results: List[Dict[str, Any]] = []
    failures = 0

    for wheel in sorted(wheels):
        record = _validate_wheel(wheel, args.min_size_bytes)
        results.append(record)
        status = "OK" if record["valid_zip"] else "FAIL"
        print(f"[validate_wheels] {status:<4} {wheel.name} ({record['size_bytes']} bytes)")
        if not record["valid_zip"]:
            failures += 1

    report_payload = {
        "wheel_dir": str(wheel_dir),
        "total_wheels": len(results),
        "min_size_bytes": args.min_size_bytes,
        "results": results,
        "failures": failures,
    }

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[validate_wheels] Report written to {report_path}")

    if failures:
        print(f"[validate_wheels] Detected {failures} invalid wheel(s)", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
