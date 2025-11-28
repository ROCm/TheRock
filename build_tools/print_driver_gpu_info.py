#!/usr/bin/env python3
"""
Sanity check script for CI runners.

Prints a small set of driver / GPU information from:
  - amd-smi
  - rocminfo

Usage:
    ./build_tools/print_driver_gpu_info.py
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.info


def run_cmd(cmd: list[str]) -> Tuple[bool, str]:
    """
    Run a command and return (ok, combined_output).

    ok = True  -> command executed and returned success
    ok = False -> command missing or returned an error
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout
    except FileNotFoundError:
        return False, f"{cmd[0]}: command not found"


def parse_amd_smi(output: str) -> Dict[str, str]:
    """
    Parse a few useful fields from amd-smi text output.
    """
    info: Dict[str, str] = {}

    # Header: AMD-SMI version, amdgpu driver version, ROCm version
    header_match = re.search(
        r"AMD-SMI\s+([^\|]+?)\s+amdgpu version:\s*(\S+)\s+ROCm version:\s*(\S+)",
        output,
    )
    if header_match:
        info["amd_smi_version"] = header_match.group(1).strip()
        info["amdgpu_driver_version"] = header_match.group(2).strip()
        info["rocm_version"] = header_match.group(3).strip()

    # Try to find the first real GPU line
    for line in output.splitlines():
        # Example:
        # | 0000:05:00.0    AMD Instinct MI300X |
        m = re.search(
            r"^\|\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]\s+(.+?)\s+\|",
            line,
        )
        if not m:
            continue
        candidate = m.group(1).strip()
        if candidate == "GPU-Name":
            # This is the header row, skip it
            continue
        info["gpu_name"] = candidate
        break

    return info


def parse_rocminfo(output: str) -> Dict[str, str]:
    """
    Parse a minimal set of fields from rocminfo
    """
    info: Dict[str, str] = {}

    # GFX target from ISA line: amdgcn-amd-amdhsa--gfx942
    isa_match = re.search(r"amdgcn-amd-amdhsa--(gfx[0-9a-zA-Z]+)", output)
    if isa_match:
        info["gpu_target"] = isa_match.group(1).strip()
    else:
        # Fallback: rocminfo might print "Name: gfx942"
        name_match = re.search(r"\bName:\s*(gfx[0-9a-zA-Z]+)", output)
        if name_match:
            info["gpu_target"] = name_match.group(1).strip()

    return info


def collect_info() -> Dict[str, object]:
    """Run tools and collect structured info."""
    info: Dict[str, object] = {}

    # amd-smi
    ok, out = run_cmd(["amd-smi"])
    info["amd_smi_available"] = ok
    if ok:
        info.update(parse_amd_smi(out))
    else:
        info["amd_smi_error"] = out.strip()

    # rocminfo
    ok, out = run_cmd(["rocminfo"])
    info["rocminfo_available"] = ok
    if ok:
        info.update(parse_rocminfo(out))
    else:
        info["rocminfo_error"] = out.strip()

    return info


def print_human_readable(info: Dict[str, object]) -> None:
    """Emit log for CI."""
    log("=== Sanity check: driver / GPU info ===")

    rocm_version = info.get("rocm_version")
    amdgpu_driver_version = info.get("amdgpu_driver_version")
    amd_smi_version = info.get("amd_smi_version")
    gpu_name = info.get("gpu_name")
    gpu_target = info.get("gpu_target")

    if amd_smi_version:
        log(f"AMD-SMI version      : {amd_smi_version}")
    if rocm_version:
        log(f"ROCm version         : {rocm_version}")
    if amdgpu_driver_version:
        log(f"amdgpu driver        : {amdgpu_driver_version}")
    if gpu_name:
        log(f"GPU name             : {gpu_name}")
    if gpu_target:
        log(f"GPU target           : {gpu_target}")

    if not info.get("amd_smi_available", False):
        log("")
        log("[warning] amd-smi not available or failed.")
        err = info.get("amd_smi_error")
        if err:
            log(f"  amd-smi output: {err.splitlines()[0]}")

    if not info.get("rocminfo_available", False):
        log("")
        log("[warning] rocminfo not available or failed.")
        err = info.get("rocminfo_error")
        if err:
            log(f"  rocminfo output: {err.splitlines()[0]}")

    log("=== End of sanity check ===")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check script to log driver / GPU info on CI runners."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    info = collect_info()

    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print_human_readable(info)

    # Do not fail the build based on this script
    return 0


if __name__ == "__main__":
    sys.exit(main())
