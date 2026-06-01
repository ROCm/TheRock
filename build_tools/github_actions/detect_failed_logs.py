#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detects failed teatime build logs and creates a small companion file named
0.error.<original>.log that contains only the important failure context.

The original log is preserved unchanged. The companion file is meant to sort
first in CI artifact/log listings and make failures easier to spot quickly.

A log is considered failed if it contains an END line with a non-zero exit code.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

FAILED_END_RE = re.compile(r"^END\t[0-9.]*\t[0-9.]*\t[1-9][0-9]*$")
IMPORTANT_RE = re.compile(
    r"(FAILED:|error:|CMake Error|Traceback|FileNotFoundError|ninja: build stopped|subcommand failed)",
    re.IGNORECASE,
)


def find_failed_logs(log_dir: Path) -> list[Path]:
    failed: list[Path] = []

    for path in sorted(log_dir.glob("*.log")):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if FAILED_END_RE.match(line.rstrip("\n")):
                        failed.append(path)
                        break
        except OSError:
            continue

    return failed


def build_excerpt(
    lines: list[str],
    window_before: int = 12,
    window_after: int = 20,
    max_lines: int = 60,
) -> list[str]:
    """
    Return a small, deterministic excerpt centered around the first important
    failure line. Falls back to the tail of the log if nothing matches.
    """
    important_idx = None
    for i, line in enumerate(lines):
        if IMPORTANT_RE.search(line):
            important_idx = i
            break

    if important_idx is None:
        excerpt = lines[-max_lines:]
    else:
        start = max(0, important_idx - window_before)
        end = min(len(lines), important_idx + window_after)
        excerpt = lines[start:end]

    if len(excerpt) > max_lines:
        excerpt = excerpt[:max_lines]

    return excerpt


def write_companion_log(src: Path, dst: Path) -> None:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()

    failure_end = next((line for line in lines if FAILED_END_RE.match(line)), None)
    excerpt = build_excerpt(lines)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        out.write(f"Failure companion for: {src.name}\n")
        out.write(f"Source log: {src}\n")
        out.write("\n")

        if failure_end:
            out.write("Failed END line:\n")
            out.write(f"{failure_end}\n\n")

        out.write("Important excerpt:\n")
        out.write("\n".join(excerpt))
        out.write("\n\n")
        out.write(f"See original log: {src.name}\n")


def main() -> int:
    output_dir = os.environ.get("OUTPUT_DIR")
    build_dir = os.environ.get("BUILD_DIR", "build")

    if output_dir:
        logs_dir = Path(output_dir) / "build" / "logs"
    else:
        logs_dir = Path(build_dir) / "logs"

    summary_env = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_env:
        print("GITHUB_STEP_SUMMARY is not set; skipping failed log summary generation.")
        return 0

    summary_path = Path(summary_env)
    failed_logs = find_failed_logs(logs_dir)

    if not failed_logs:
        print("No failed log found.")
        return 0

    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write("## Build failure\n")
        summary.write(f"**Error logs:** {len(failed_logs)}\n")
        summary.write("\n")

        for src in failed_logs:
            dst = src.with_name(f"0.error.{src.name}")
            try:
                write_companion_log(src, dst)
                print(f"Created {dst.name} from {src.name}")
                summary.write(f"- `{dst.name}`\n")
                summary.write("\n")
            except OSError as e:
                print(f"Failed to create companion log for {src}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
