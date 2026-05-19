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
# TODO(#2964): Remove gfx950-dcgpu once amdsmi static does not timeout
unsupported_amdsmi_families = ["gfx1151", "gfx950-dcgpu"]


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(args: List[str | Path], cwd: Optional[Path] = None) -> None:
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


def run_command_with_search(
    label: str,
    command: str,
    args: List[str],
    extra_command_search_paths: List[Path],
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
            run_command([candidate] + args)
            return

    # Then fall back to PATH
    resolved = shutil.which(command)
    if resolved:
        log(f"\n=== {label} ===")
        run_command([resolved] + args)
        return

    # Nothing found
    log(f"\n=== {label} ===")
    log(f"{command}: command not found")


def print_sysfs_firmware_versions() -> None:
    """
    Print per-component firmware versions from /sys/class/drm/card*/device/fw_version/.

    These sysfs files are world-readable and exposed by amdgpu without
    needing debugfs, so they are reliable inside CI containers. They include
    mes_fw_version and mes_kiq_fw_version, which are useful when diagnosing
    MES-related dispatch hangs.
    """
    log("\n=== sysfs firmware versions ===")
    cards = sorted(Path("/sys/class/drm").glob("card*/device/fw_version"))
    if not cards:
        log("/sys/class/drm/card*/device/fw_version: not found")
        return
    for fw_dir in cards:
        card = fw_dir.parent.parent.name
        log(f"-- {card} ({fw_dir})")
        for entry in sorted(fw_dir.iterdir()):
            try:
                value = entry.read_text().strip()
            except OSError as e:
                value = f"<read error: {e}>"
            log(f"  {entry.name}: {value}")


def run_sanity(os_name: str) -> None:
    THIS_SCRIPT_DIR = Path(__file__).resolve().parent
    THEROCK_DIR = THIS_SCRIPT_DIR.parent
    bin_dir = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))

    log("=== Sanity check: driver / GPU info ===")

    if os_name.lower() == "windows":
        # Windows: only hipInfo.exe
        run_command_with_search(
            label="hipInfo.exe",
            command="hipInfo.exe",
            args=[],
            extra_command_search_paths=[bin_dir],
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
            )
        run_command_with_search(
            label="rocminfo",
            command="rocminfo",
            args=[],
            extra_command_search_paths=[bin_dir],
        )
        run_command_with_search(
            label="Kernel version",
            command="uname",
            args=["-r"],
            extra_command_search_paths=[bin_dir],
        )
        # Print per-component firmware versions (useful for debugging hangs).
        # Read sysfs first: it works regardless of amd-smi support and always
        # includes mes_fw_version / mes_kiq_fw_version for diagnosing MES hangs.
        print_sysfs_firmware_versions()
        if AMDGPU_FAMILIES not in unsupported_amdsmi_families:
            run_command_with_search(
                label="amd-smi firmware",
                command="amd-smi",
                args=["firmware"],
                extra_command_search_paths=[bin_dir],
            )

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
