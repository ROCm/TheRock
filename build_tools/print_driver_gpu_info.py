#!/usr/bin/env python3
"""
Sanity check script for CI runners.

On Linux:
  - print Linux kernel version
  - print linux-firmware package version
  - print amdgpu driver source (in-kernel vs DKMS/external)
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


def print_kernel_version() -> None:
    """Print the Linux kernel version."""
    log("\n=== Linux Kernel Version ===")
    log(f"Kernel: {platform.release()}")
    log(f"Full uname: {platform.platform()}")


def print_linux_firmware_version() -> None:
    """Print the version of the installed linux-firmware package."""
    log("\n=== Linux Firmware Package ===")

    # Try dpkg (Debian/Ubuntu)
    dpkg = shutil.which("dpkg-query")
    if dpkg:
        try:
            proc = subprocess.run(
                [dpkg, "-W", "-f", "${Version}", "linux-firmware"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                log(f"linux-firmware (dpkg): {proc.stdout.strip()}")
                return
        except Exception:
            pass

    # Try rpm (Fedora/RHEL)
    rpm = shutil.which("rpm")
    if rpm:
        try:
            proc = subprocess.run(
                [rpm, "-q", "linux-firmware", "--queryformat", "%{VERSION}-%{RELEASE}"],
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                log(f"linux-firmware (rpm): {proc.stdout.strip()}")
                return
        except Exception:
            pass

    log("linux-firmware: package not found or unsupported package manager")


def print_amdgpu_driver_source() -> None:
    """Check whether the amdgpu driver is from the kernel or external (DKMS)."""
    log("\n=== AMDGPU Driver Source ===")

    modinfo = shutil.which("modinfo")
    if not modinfo:
        log("modinfo: command not found")
        return

    try:
        proc = subprocess.run(
            [modinfo, "amdgpu"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            log("amdgpu: module not found")
            return

        # Parse modinfo output
        info = {}
        for line in proc.stdout.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                # Only keep first occurrence of each key
                if key.strip() not in info:
                    info[key.strip()] = value.strip()

        filename = info.get("filename", "")
        version = info.get("version", "unknown")

        # Determine source based on module path
        if "updates/dkms" in filename or "/dkms/" in filename:
            source = "DKMS (external)"
        elif "extra/" in filename or "/extra/" in filename:
            source = "External (extra modules)"
        elif "kernel/drivers" in filename:
            source = "In-kernel"
        elif "(builtin)" in filename.lower() or not filename:
            source = "Built-in (compiled into kernel)"
        else:
            source = "Unknown"

        log(f"Driver version: {version}")
        log(f"Driver source: {source}")
        log(f"Module path: {filename}")

    except Exception as e:
        log(f"Error checking amdgpu driver: {e}")


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
        # Linux: kernel version, firmware, driver source, amd-smi static, rocminfo
        print_kernel_version()
        print_linux_firmware_version()
        print_amdgpu_driver_source()

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

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
