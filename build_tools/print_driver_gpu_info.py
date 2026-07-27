#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"

On Windows:
  - run "hipInfo.exe"

This script prints only raw command output.
"""

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
unsupported_amdsmi_families = ["gfx1151"]


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(
    args: List[str | Path], cwd: Optional[Path] = None, env: Optional[dict] = None
) -> None:
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
            env=env,
        )
        log(proc.stdout.rstrip())
    except FileNotFoundError:
        log(f"{args[0]}: command not found")


def run_command_with_search(
    label: str,
    command: str,
    args: List[str],
    extra_command_search_paths: List[Path],
    env: Optional[dict] = None,
) -> None:
    """
    Run a command, searching in extra paths first, then PATH.

    Example:
        run_command_with_search(
            label="amd-smi static",
            command="amd-smi",
            args=["static"],
            extra_command_search_paths=[bin_dir],
        )
    """
    # Try explicit directories first (e.g. THEROCK_DIR/build/bin)
    for base in extra_command_search_paths:
        candidate = base / command
        if candidate.exists():
            log(f"\n=== {label} ===")
            run_command([candidate] + args, env=env)
            return

    # Then fall back to PATH
    resolved = shutil.which(command)
    if resolved:
        log(f"\n=== {label} ===")
        run_command([resolved] + args, env=env)
        return

    # Nothing found
    log(f"\n=== {label} ===")
    log(f"{command}: command not found")


def run_sanity(os_name: str) -> None:
    THIS_SCRIPT_DIR = Path(__file__).resolve().parent
    THEROCK_DIR = THIS_SCRIPT_DIR.parent
    bin_dir = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))

    # Set up environment with LD_LIBRARY_PATH for newly built libraries
    env = os.environ.copy()
    if os_name.lower() != "windows":
        lib_dir = bin_dir.parent / "lib"
        if lib_dir.exists():
            existing_ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{lib_dir}:{existing_ld_path}" if existing_ld_path else str(lib_dir)
            )
            log(f"=== Setting LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']} ===\n")

    log("=== Sanity check: driver / GPU info ===")

    if os_name.lower() == "windows":
        # Windows: only hipInfo.exe
        run_command_with_search(
            label="hipInfo.exe",
            command="hipInfo.exe",
            args=[],
            extra_command_search_paths=[bin_dir],
            env=env,
        )
    else:
        # Linux: amd-smi static + rocminfo
        # TODO(#2789): Remove conditional once amdsmi supports gfx1151
        if AMDGPU_FAMILIES not in unsupported_amdsmi_families:
            run_command_with_search(
                label="amd-smi static",
                command="amd-smi",
                args=["static"],
                extra_command_search_paths=[bin_dir],
                env=env,
            )
        run_command_with_search(
            label="rocminfo",
            command="rocminfo",
            args=[],
            extra_command_search_paths=[bin_dir],
            env=env,
        )
        run_command_with_search(
            label="Kernel version",
            command="uname",
            args=["-r"],
            extra_command_search_paths=[bin_dir],
            env=env,
        )

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
