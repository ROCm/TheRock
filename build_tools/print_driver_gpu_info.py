#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"
  - report the KFD IOCTL version and warn if outside the range required by rocdbgapi (>= 1.13 and < 2.0)

On Linux inside WSL (kernel release contains "microsoft"):
  - as above, except the KFD IOCTL check: WSL has no amdgpu kernel driver and
    no /dev/kfd. The GPU is /dev/dxg, and amd-smi reports the WDDM driver version.

On Windows:
  - run "hipInfo.exe"

Driver commands (amd-smi, rocminfo, hipInfo) are run with check=True and will
cause this script to exit non-zero if they fail. amd-smi and rocminfo are each
skipped for the GPU families listed below, for unrelated reasons noted there. The KFD version check also
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
_DXG_DEVICE = "/dev/dxg"
_KFD_VERSION_MIN = (1, 13)
_KFD_VERSION_MAX = (2, 0)  # exclusive

# TODO(#7659): Re-enable once rocminfo is fixed for gfx125X-dcgpu
_ROCMINFO_EXCLUDED_FAMILIES = ["gfx125X-dcgpu"]

# amd-smi runs CPU discovery during init, before it parses which processors
# were asked for, and treats finding none as fatal -- it exits 196 having
# printed no GPU information at all, even when invoked as `amd-smi static -g
# all`. CPU discovery goes through ESMI, which needs /dev/hsmp (CAP_SYS_ADMIN
# to open, regardless of mode) or /dev/cpu/*/msr (CAP_SYS_RAWIO). Neither is
# available to a container that does not run as root, so amd-smi cannot
# report on a GPU it can otherwise see perfectly well.
#
# This is not a GPU problem and the GPU checks here do not depend on it:
# initialising the amdsmi library directly with INIT_AMD_GPUS on an affected
# runner returns the expected handle and BDF, and rocminfo, which reaches the
# device over KFD rather than ESMI, is unaffected and still checked below.
_AMDSMI_EXCLUDED_FAMILIES = ["gfx125X-dcgpu"]


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
        run_command_with_search(
            label="rocm-smi",
            command="rocm-smi",
            args=[],
            extra_command_search_paths=[bin_dir],
        )
        run_command_with_search(
            label="rocminfo",
            command="rocminfo",
            args=[],
            extra_command_search_paths=[bin_dir],
        )
        
        # Linux: amd-smi static + rocminfo
        amdgpu_families = os.getenv("AMDGPU_FAMILIES", "")
        skip_amdsmi = any(
            family in amdgpu_families for family in _AMDSMI_EXCLUDED_FAMILIES
        )
        if skip_amdsmi:
            log(f"\n=== amd-smi static ===")
            log(
                f"Skipping amd-smi: disabled for {amdgpu_families} "
                f"(excluded families: {_AMDSMI_EXCLUDED_FAMILIES})"
            )
        else:
            run_command_with_search(
                label="amd-smi static",
                command="amd-smi",
                args=["static"],
                extra_command_search_paths=[bin_dir],
            )
        # Check if rocminfo should be skipped for this GPU family
        skip_rocminfo = any(
            family in amdgpu_families for family in _ROCMINFO_EXCLUDED_FAMILIES
        )
        if skip_rocminfo:
            log(f"\n=== rocminfo ===")
            log(
                f"Skipping rocminfo: disabled for {amdgpu_families} "
                f"(excluded families: {_ROCMINFO_EXCLUDED_FAMILIES}), see #7659"
            )
        else:
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

        # WSL kernel releases look like "6.6.87.2-microsoft-standard-WSL2"
        # (containers share the host kernel).
        if "microsoft" in platform.release().lower():
            log("\n=== GPU-PV (WSL) ===")
            if not os.path.exists(_DXG_DEVICE):
                log(
                    f"error: {_DXG_DEVICE} not found - is the GPU passed through to WSL?"
                )
                return 1
            log(f"{_DXG_DEVICE} present; KFD IOCTL check does not apply under WSL")
            log("\n=== End of sanity check ===")
            return 0

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
