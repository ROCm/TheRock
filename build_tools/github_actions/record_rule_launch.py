#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Light-weight wrapper invoked via CMAKE_RULE_LAUNCH_COMPILE / LINK to capture
per-command timing during TheRock stage builds.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _resolve_log_path() -> Path | None:
    target = os.environ.get("RUN_STAGE_TIMING_LOG")
    if not target:
        return None
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _extract_output_flag(args: Sequence[str]) -> str | None:
    for index, token in enumerate(args):
        if token in {"-o", "--output"} and index + 1 < len(args):
            return args[index + 1]
        if token.startswith("-o") and token != "-o":
            return token[2:]
    return None


def _append_record(log_path: Path, record: dict[str, object]) -> None:
    serialized = json.dumps(record, sort_keys=True) + "\n"
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(serialized)


def main() -> int:
    if len(sys.argv) < 3:
        print("record_rule_launch requires mode ('compile' or 'link') and the underlying command", file=sys.stderr)
        return 97

    mode = sys.argv[1]
    if mode not in {"compile", "link"}:
        print(f"record_rule_launch: unexpected mode '{mode}'", file=sys.stderr)
        return 98

    command = sys.argv[2:]
    log_path = _resolve_log_path()
    stage = os.environ.get("STAGE_NAME") or None
    variant = os.environ.get("THEROCK_BUILD_VARIANT") or os.environ.get("RUN_STAGE_VARIANT") or None

    start = time.time()
    result = subprocess.run(command)
    end = time.time()

    if log_path:
        record = {
            "timestamp_start": start,
            "timestamp_end": end,
            "duration_seconds": end - start,
            "mode": mode,
            "command": command,
            "command_display": " ".join(shlex.quote(arg) for arg in command),
            "exit_code": result.returncode,
            "stage": stage,
            "variant": variant,
            "hostname": os.uname().nodename,
            "pid": os.getpid(),
            "output": _extract_output_flag(command),
        }
        try:
            _append_record(log_path, record)
        except OSError as exc:
            print(f"record_rule_launch: failed to append timing record: {exc}", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
