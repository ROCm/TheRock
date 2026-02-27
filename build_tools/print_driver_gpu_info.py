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

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import re

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
# TODO(#2964): Remove gfx950-dcgpu once amdsmi static does not timeout
unsupported_amdsmi_families = ["gfx1151", "gfx950-dcgpu"]


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(args: list[str | Path], cwd: Path | None = None) -> None:
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


# Capture stdout while preserving existing "raw output" printing behavior.
def run_command_capture_output(
    args: list[str | Path], cwd: Path | None = None
) -> str | None:
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
        return proc.stdout
    except FileNotFoundError:
        log(f"{args[0]}: command not found")
        return None


def run_command_with_search(
    label: str,
    command: str,
    args: list[str],
    extra_command_search_paths: list[Path],
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


def run_command_with_search_capture(
    label: str,
    command: str,
    args: list[str],
    extra_command_search_paths: list[Path],
) -> str | None:
    for base in extra_command_search_paths:
        candidate = base / command
        if candidate.exists():
            log(f"\n=== {label} ===")
            return run_command_capture_output([candidate] + args)

    resolved = shutil.which(command)
    if resolved:
        log(f"\n=== {label} ===")
        return run_command_capture_output([resolved] + args)

    log(f"\n=== {label} ===")
    log(f"{command}: command not found")
    return None


# BAR info printing (Linux only)
_BDF_RE = re.compile(r"^\s*BDF:\s*(\S+)\s*$")
_BAR0_RE = re.compile(r"Region 0:.*\[\s*size=([^\]]+)\]", re.IGNORECASE)


def _parse_bdfs_from_amd_smi_static(output: str) -> list[str]:
    bdfs: list[str] = []
    for line in output.splitlines():
        m = _BDF_RE.match(line)
        if m:
            bdfs.append(m.group(1))
    # preserve order; de-dup
    seen: set[str] = set()
    out: list[str] = []
    for b in bdfs:
        if b not in seen:
            out.append(b)
            seen.add(b)
    return out


def _get_bar0_size(bdf: str) -> str | None:
    lspci = shutil.which("lspci")
    if not lspci:
        return None

    # Prefer non-interactive sudo; fall back if not available.
    proc = subprocess.run(
        ["sudo", "-n", lspci, "-vv", "-s", bdf],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    out = proc.stdout or ""
    if not out:
        proc = subprocess.run(
            [lspci, "-vv", "-s", bdf],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        out = proc.stdout or ""

    m = _BAR0_RE.search(out)
    return m.group(1) if m else None


def _is_large_bar(size: str | None) -> bool:
    return bool(size and re.search(r"[GT]", size, re.IGNORECASE))


# Log BAR0 only: it reflects the GPU memory aperture (small vs large BAR)
def print_linux_bar_info_from_amd_smi_output(amd_smi_output: str) -> None:
    log("\n=== PCI BAR info (BAR0 / large BAR proxy) ===")

    lspci = shutil.which("lspci")
    if not lspci:
        log("lspci: command not found (install pciutils)")
        return

    bdfs = _parse_bdfs_from_amd_smi_static(amd_smi_output)
    if not bdfs:
        log("No GPU BDFs found in amd-smi output")
        return

    for bdf in bdfs:
        bar0_size = _get_bar0_size(bdf)
        large = "YES" if _is_large_bar(bar0_size) else "NO"
        log(f"{bdf}: BAR0_size={bar0_size or 'UNKNOWN'} large_BAR={large}")


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
            amd_smi_out = run_command_with_search_capture(
                label="amd-smi static",
                command="amd-smi",
                args=["static"],
                extra_command_search_paths=[bin_dir],
            )
            if amd_smi_out:
                print_linux_bar_info_from_amd_smi_output(amd_smi_out)

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

    log("\n=== End of sanity check ===")


def main(argv: list[str] | None = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
