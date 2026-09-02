#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"
  - report the KFD IOCTL version and warn if outside the range required by rocdbgapi (>= 1.13 and < 2.0)

On Windows:
  - run "hipInfo.exe"

Driver commands (amd-smi, rocminfo, hipInfo) are run with check=True and will
cause this script to exit non-zero if they fail. The KFD version check also
causes a non-zero exit when the version is below the supported minimum or
cannot be queried. A version above the supported maximum produces a warning
but does not fail.
"""

import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
from typing import List, Optional, Tuple


# AMDKFD_IOC_GET_VERSION = _IOR('K', 0x01, struct { u32 major; u32 minor; })
# _IOR: direction=0x80, size=8, type='K'=0x4b, nr=0x01 -> 0x80084b01
_AMDKFD_IOC_GET_VERSION = 0x80084B01
_KFD_DEVICE = "/dev/kfd"
_KFD_VERSION_MIN = (1, 13)
_KFD_VERSION_MAX = (2, 0)  # exclusive


def _get_kfd_version() -> Tuple[int, int]:
    # fcntl is a Unix-only stdlib module and is only needed for this Linux
    # KFD ioctl query. Import it lazily so the Windows sanity check does not
    # crash at import time with ModuleNotFoundError: No module named 'fcntl'.
    import fcntl

    fd = os.open(_KFD_DEVICE, os.O_RDWR)
    try:
        buf = bytearray(8)
        fcntl.ioctl(fd, _AMDKFD_IOC_GET_VERSION, buf)
        major, minor = struct.unpack("II", buf)
        return major, minor
    finally:
        os.close(fd)


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
            check=True,
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


def run_sanity(os_name: str) -> int:
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

        log("\n=== KFD IOCTL version ===")
        if not os.path.exists(_KFD_DEVICE):
            log(f"error: {_KFD_DEVICE} not found — is the AMDGPU driver loaded?")
            return 1
        try:
            major, minor = _get_kfd_version()
            too_old = (major, minor) < _KFD_VERSION_MIN
            too_new = (major, minor) >= _KFD_VERSION_MAX
            if too_old:
                status = "NOT supported (too old)"
            elif too_new:
                status = "NOT supported (warning: newer than tested range)"
            else:
                status = "supported"
            log(f"KFD IOCTL version: {major}.{minor} ({status})")
            log(
                f"Required range for rocdbgapi: "
                f">= {_KFD_VERSION_MIN[0]}.{_KFD_VERSION_MIN[1]}"
                f" and < {_KFD_VERSION_MAX[0]}.{_KFD_VERSION_MAX[1]}"
            )
            if too_old:
                return 1
        except OSError as e:
            log(f"error: failed to query KFD version: {e}")
            return 1

    log("\n=== End of sanity check ===")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    return run_sanity(detected)


if __name__ == "__main__":
    sys.exit(main())
