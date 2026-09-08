#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detects failed teatime build logs and creates a small 0.error.*.log file
that contains only the important failure context.

The original log is preserved unchanged.

Log directory resolution order:
1. --log-dir
2. OUTPUT_DIR/build/logs on Linux CI
3. BUILD_DIR/logs as fallback
4. build/logs as the final default

Summary output resolution order:
1. --summary-path
2. GITHUB_STEP_SUMMARY
3. skip summary generation if neither is available

This script can also be run manually:
    python build_tools/github_actions/detect_failed_logs.py \
        --log-dir build/logs \
        --summary-path /tmp/summary.md

"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

IMPORTANT_RE = re.compile(
    r"(FAILED:|error:|CMake Error|Traceback|FileNotFoundError|ninja: build stopped|subcommand failed)",
)


# teatime END records are tab-separated.
# We only care that the line starts with END and that the last field is a
# non-zero exit code, so the parser is resilient if teatime adds fields later.
def is_failed_end_line(line: str) -> bool:
    fields = line.rstrip("\n").split("\t")

    if len(fields) < 4:
        return False

    if fields[0] != "END":
        return False

    try:
        return int(fields[-1]) != 0
    except ValueError:
        return False


def get_failed_end_line(path: Path, tail_bytes: int = 4096) -> str | None:
    """Read only the tail of the log when searching for a failed END record.
    Teatime writes the END line at the end of the log, so scanning the last
    few KiB avoids reading large log files into memory.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError as e:
        print(f"Warning: failed to read log {path}: {e}", file=sys.stderr)
        return None

    for line in reversed(tail.splitlines()):
        if is_failed_end_line(line):
            return line

    return None


def find_failed_logs(log_dir: Path, tail_bytes: int = 4096) -> list[tuple[Path, str]]:
    failed: list[tuple[Path, str]] = []

    for path in sorted(log_dir.glob("*.log")):
        if path.name.startswith("0.error."):
            continue
        failure_end = get_failed_end_line(path, tail_bytes=tail_bytes)
        if failure_end:
            failed.append((path, failure_end))

    return failed


def build_excerpt(
    lines: list[str],
    window_before: int = 12,
    window_after: int = 20,
) -> list[str]:
    """
    Return a small, deterministic excerpt centered around the most recent
    important failure line. Falls back to the tail of the log if nothing matches.
    """
    important_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if IMPORTANT_RE.search(lines[i]):
            important_idx = i
            break

    if important_idx is None:
        excerpt = lines[-(window_before + window_after) :]
    else:
        start = max(0, important_idx - window_before)
        end = min(len(lines), important_idx + window_after + 1)
        excerpt = lines[start:end]

    return excerpt


def write_failure_summary_log(
    src: Path,
    dst: Path,
    failure_end: str,
) -> None:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    excerpt = build_excerpt(lines)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        out.write(f"Failure summary for: {src.name}\n")
        out.write(f"Source log: {src}\n")
        out.write("\n")

        out.write("Failed END line:\n")
        out.write(f"{failure_end}\n\n")

        out.write("Important excerpt:\n")
        out.write("\n".join(excerpt))
        out.write("\n\n")
        out.write(f"See original log: {src.name}\n")


def resolve_logs_dir(args: argparse.Namespace) -> Path:
    if args.log_dir:
        return args.log_dir

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        return Path(output_dir) / "build" / "logs"

    build_dir = os.environ.get("BUILD_DIR", "build")
    return Path(build_dir) / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect failed teatime logs and generate 0.error.*.log summaries."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory containing *.log files. Overrides env-based fallback.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="GitHub step summary file path. Overrides GITHUB_STEP_SUMMARY.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=4096,
        help="Number of bytes to read from the end of each log when searching for END.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_dir = resolve_logs_dir(args)
    if not logs_dir.exists():
        print(f"Log directory does not exist: {logs_dir}", file=sys.stderr)
        return 0

    summary_path = args.summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        print("GITHUB_STEP_SUMMARY is not set; skipping failed log summary generation.")
        return 0

    summary_path = Path(summary_path)
    failed_logs = find_failed_logs(logs_dir, tail_bytes=args.tail_bytes)

    if not failed_logs:
        print("No failed log found.")
        return 0

    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write("## Build failure\n")
        summary.write(f"**Error logs:** {len(failed_logs)}\n")
        summary.write("\n")

        for src, failure_end in failed_logs:
            dst = src.with_name(f"0.error.{src.name}")
            try:
                write_failure_summary_log(src, dst, failure_end)
                print(f"Created {dst.name} from {src.name}")
                summary.write(f"- `{dst.name}`\n")
            except OSError as e:
                print(
                    f"Failed to create failure summary log for {src}: {e}",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
