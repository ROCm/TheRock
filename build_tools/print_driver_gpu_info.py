#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"
  - run "uname -r"
  - read "/var/lib/dkms" for the AMDGPU DKMS package version

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


DKMS_ROOT = Path("/var/lib/dkms")


def log_dkms_status(module: str = "amdgpu") -> None:
    """
    Report a DKMS module's version(s) by reading /var/lib/dkms/<module>/.

    This avoids invoking "dkms status", which requires root on some systems.
    The on-disk layout mirrors the "dkms status" output:

        /var/lib/dkms/<module>/<version>/<kernel>/<arch>/module

    A module is reported as "installed" when the "module" directory exists for
    that version/kernel/arch, otherwise "built".
    """
    label = f"{module} DKMS package version"
    log(f"\n=== {label} ===")

    module_root = DKMS_ROOT / module
    if not module_root.is_dir():
        log(f"{module}: no DKMS entry found under {DKMS_ROOT}")
        return

    found = False
    for version_dir in sorted(module_root.iterdir()):
        # Skip the "kernel-*" convenience symlinks and the "original_module"
        # backup directory; only walk real version directories.
        if (
            version_dir.is_symlink()
            or version_dir.name == "original_module"
            or not version_dir.is_dir()
        ):
            continue
        version = version_dir.name
        for kernel_dir in sorted(version_dir.iterdir()):
            # "source" is a symlink into /usr/src, not a kernel build.
            if kernel_dir.name == "source" or not kernel_dir.is_dir():
                continue
            kernel = kernel_dir.name
            for arch_dir in sorted(kernel_dir.iterdir()):
                if not arch_dir.is_dir():
                    continue
                arch = arch_dir.name
                state = "installed" if (arch_dir / "module").is_dir() else "built"
                log(f"{module}/{version}, {kernel}, {arch}: {state}")
                found = True

    if not found:
        log(f"{module}: no built modules found under {module_root}")


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
        log_dkms_status("amdgpu")

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
