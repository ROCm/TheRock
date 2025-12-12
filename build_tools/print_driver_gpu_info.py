#!/usr/bin/env python3
"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"

On Windows:
  - run "hipInfo.exe"

This script prints only raw command output.
"""

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def exec(args: List[str | Path], cwd: Optional[Path] = None) -> None:
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()

    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")

    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        log(proc.stdout.rstrip())
    except FileNotFoundError:
        log(f"{args[0]}: command not found")


def run_candidates(label: str, candidates: List[List[str]]) -> None:
    """
    Try a list of commands and run the first one that exists.

    Uses shutil.which() for PATH resolution, otherwise accepts an explicit path.
    """
    for cmd in candidates:
        exe = cmd[0]

        resolved = shutil.which(exe)
        if resolved:
            cmd_to_run = [resolved] + cmd[1:]
            log(f"\n=== {label} ===")
            exec(cmd_to_run)
            return

        resolved_path = Path(exe)
        if resolved_path.exists():
            cmd_to_run = [str(resolved_path)] + cmd[1:]
            log(f"\n=== {label} ===")
            exec(cmd_to_run)
            return

    # Nothing found
    exe_name = candidates[0][0] if candidates else "<unknown>"
    log(f"\n=== {label} ===")
    log(f"{exe_name}: command not found")


def run_sanity(os_name: str) -> None:

    THIS_SCRIPT_DIR = Path(__file__).resolve().parent
    THEROCK_DIR = THIS_SCRIPT_DIR.parent
    bin_dir = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))
    log("=== Sanity check: driver / GPU info ===")

    if os_name.lower() == "windows":
        hipinfo_candidates = [
            [str(bin_dir / "hipInfo.exe")],
            ["hipInfo.exe"],
        ]
        run_candidates("hipInfo.exe", hipinfo_candidates)
    else:
        amd_smi_candidates = [
            [str(bin_dir / "amd-smi"), "static"],
            ["amd-smi", "static"],
        ]
        rocminfo_candidates = [
            [str(bin_dir / "rocminfo")],
            ["rocminfo"],
        ]
        run_candidates("amd-smi static", amd_smi_candidates)
        run_candidates("rocminfo", rocminfo_candidates)

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check script to log driver / GPU info on CI runners."
    )
    parser.add_argument(
        "--os",
        dest="os_name",
        help="Override OS (Linux or Windows).",
    )
    args = parser.parse_args(argv)

    detected = platform.system().lower()
    os_name = (
        args.os_name
        if args.os_name
        else ("Windows" if detected == "windows" else "Linux")
    )

    run_sanity(os_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
