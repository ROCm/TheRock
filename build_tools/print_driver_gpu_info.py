#!/usr/bin/env python3
"""
Sanity check script for CI runners.

Prints a small set of driver / GPU information from:
  - amd-smi
  - rocminfo

It prefers binaries installed under ./build/bin (as produced by
install_rocm_from_artifacts.py), and falls back to the PATH.

Usage:
    ./build_tools/print_driver_gpu_info.py
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from typing import Dict, Tuple, Optional, List, Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.info


def run_cmd(cmd: list[str]) -> Tuple[bool, str]:
    """
    Run a command and return (ok, combined_output).

    ok = True  -> command executed and returned success (exit code 0)
    ok = False -> command missing or returned a non-zero exit code
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


def run_tool_with_candidates(
    tool_name: str, candidates: list[str]
) -> Tuple[Optional[str], bool, str]:
    """
    Try running a tool from a list of candidate paths / names.

    Returns (resolved_candidate, ok, output).

    - resolved_candidate: the candidate string that was used, or None
    - ok: True if the command ran and returned exit code 0
    - output: combined stdout/stderr from the last attempt
    """
    last_out = ""
    for cand in candidates:
        ok, out = run_cmd([cand])
        last_out = out
        # If the tool is literally not found, try the next candidate
        if "command not found" in out:
            continue
        # We found *something* (even if it failed with non-zero rc)
        return cand, ok, out

    # Nothing worked
    return None, False, last_out or f"{tool_name}: command not found"


def parse_amd_smi(output: str) -> Dict[str, Any]:
    """
    Parse a few useful fields from amd-smi / amdsmi text output.

    Supports both the older AMD-SMI header and the newer AMDSMI Tool header.
    """
    info: Dict[str, Any] = {}

    # New-style amdsmi header, e.g.:
    # AMDSMI Tool: 26.1.0+65125c66 | AMDSMI Library version: 26.1.0
    # | ROCm version: 7.11.0 | amdgpu version: 6.12.12 | ...
    new_header = re.search(
        r"AMDSMI Tool:\s*([^\|]+?)\s*\|"
        r"\s*AMDSMI Library version:\s*([^\|]+?)\s*\|"
        r"\s*ROCm version:\s*([^\|]+?)\s*\|"
        r"\s*amdgpu version:\s*([^\|]+?)\s*\|",
        output,
    )
    if new_header:
        info["amd_smi_version"] = new_header.group(1).strip()
        info["amd_smi_library_version"] = new_header.group(2).strip()
        info["rocm_version"] = new_header.group(3).strip()
        info["amdgpu_driver_version"] = new_header.group(4).strip()
    else:
        # Older header style, e.g.:
        # | AMD-SMI 26.1.0+b6840895      amdgpu version: 6.16.6   ROCm version: 7.10.0 |
        old_header = re.search(
            r"AMD-SMI\s+([^\|]+?)\s+amdgpu version:\s*(\S+)\s+ROCm version:\s*(\S+)",
            output,
        )
        if old_header:
            info["amd_smi_version"] = old_header.group(1).strip()
            info["amdgpu_driver_version"] = old_header.group(2).strip()
            info["rocm_version"] = old_header.group(3).strip()

    # (Optional) summary GPU name from the table, e.g.:
    # | 0000:05:00.0    AMD Instinct MI300X |
    for line in output.splitlines():
        m = re.search(
            r"^\|\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]\s+(.+?)\s+\|",
            line,
        )
        if not m:
            continue
        candidate = m.group(1).strip()
        if candidate == "GPU-Name":
            # Header row, skip it
            continue
        info["gpu_name"] = candidate
        break

    return info


def parse_rocminfo(output: str) -> Dict[str, Any]:
    """
    Parse a minimal set of fields from rocminfo.

    We collect:
      - A representative gpu_target (first GPU)
      - A list of all GPU agents with (agent_index, gpu_name, gpu_target)
        under the key "gpu_agents".
    """
    info: Dict[str, Any] = {}
    gpu_agents: List[Dict[str, Any]] = []

    # Match blocks like:
    # *******
    # Agent N
    # *******
    #   ...block...
    agent_pattern = re.compile(
        r"\*{7}\s*\nAgent\s+(\d+)\s*\n\*{7}\s*\n(.*?)(?=\*{7}\s*\nAgent\s+\d+\s*\n\*{7}|\Z)",
        re.DOTALL,
    )

    for m in agent_pattern.finditer(output):
        agent_id_str = m.group(1)
        block = m.group(2)

        # Only care about GPU agents
        if "Device Type:             GPU" not in block:
            continue

        try:
            agent_index = int(agent_id_str)
        except ValueError:
            agent_index = None

        # Marketing name, e.g.:
        #   Marketing Name:          AMD Instinct MI300X
        gpu_name_match = re.search(r"^\s*Marketing Name:\s*(.+)$", block, re.MULTILINE)
        gpu_name = gpu_name_match.group(1).strip() if gpu_name_match else None

        # GFX target from ISA line: amdgcn-amd-amdhsa--gfx942
        isa_match = re.search(r"amdgcn-amd-amdhsa--(gfx[0-9a-zA-Z]+)", block)
        if isa_match:
            gpu_target = isa_match.group(1).strip()
        else:
            # Fallback: "Name: gfx942" inside the GPU block
            name_match = re.search(
                r"^\s*Name:\s*(gfx[0-9a-zA-Z]+)", block, re.MULTILINE
            )
            gpu_target = name_match.group(1).strip() if name_match else None

        gpu_agents.append(
            {
                "agent_index": agent_index,
                "gpu_name": gpu_name,
                "gpu_target": gpu_target,
            }
        )

    if gpu_agents:
        info["gpu_agents"] = gpu_agents
        # Representative gpu_target for backwards compatibility
        first_target = gpu_agents[0].get("gpu_target")
        if first_target:
            info["gpu_target"] = first_target

    return info


def collect_info() -> Dict[str, Any]:
    """Run tools and collect structured info."""
    info: Dict[str, Any] = {}

    # Prefer SDK-installed binaries in ./build/bin (your local + CI pattern),
    # then fall back to PATH.
    amd_smi_candidates = [
        "./build/bin/amd-smi",
        "amd-smi",
    ]
    rocminfo_candidates = [
        "./build/bin/rocminfo",
        "rocminfo",
    ]

    # amd-smi / amdsmi
    amd_path, amd_ok, amd_out = run_tool_with_candidates("amd-smi", amd_smi_candidates)
    if amd_path:
        info["amd_smi_path"] = amd_path
    info["amd_smi_available"] = amd_ok
    if amd_ok:
        info.update(parse_amd_smi(amd_out))
    else:
        info["amd_smi_error"] = (amd_out or "").strip()

    # rocminfo
    rocminfo_path, roc_ok, roc_out = run_tool_with_candidates(
        "rocminfo", rocminfo_candidates
    )
    if rocminfo_path:
        info["rocminfo_path"] = rocminfo_path
    info["rocminfo_available"] = roc_ok
    if roc_ok:
        info.update(parse_rocminfo(roc_out))
    else:
        info["rocminfo_error"] = (roc_out or "").strip()

    return info


def print_human_readable(info: Dict[str, object]) -> None:
    """Emit log for CI."""
    log("=== Sanity check: driver / GPU info ===")

    amd_smi_version = info.get("amd_smi_version")
    amd_smi_library_version = info.get("amd_smi_library_version")
    rocm_version = info.get("rocm_version")
    amdgpu_driver_version = info.get("amdgpu_driver_version")
    amd_smi_path = info.get("amd_smi_path")
    rocminfo_path = info.get("rocminfo_path")
    gpu_agents = info.get("gpu_agents")

    if amd_smi_version:
        log(f"AMD-SMI version      : {amd_smi_version}")
    if amd_smi_library_version:
        log(f"AMD-SMI lib version  : {amd_smi_library_version}")
    if rocm_version:
        log(f"ROCm version         : {rocm_version}")
    if amdgpu_driver_version:
        log(f"amdgpu driver        : {amdgpu_driver_version}")
    if amd_smi_path:
        log(f"amd-smi path         : {amd_smi_path}")
    if rocminfo_path:
        log(f"rocminfo path        : {rocminfo_path}")

    # Per-GPU details from rocminfo
    if isinstance(gpu_agents, list) and gpu_agents:
        log("")
        log("Per-GPU agents (from rocminfo):")
        for entry in sorted(gpu_agents, key=lambda x: x.get("agent_index") or 0):
            agent_idx = entry.get("agent_index")
            name = entry.get("gpu_name") or "Unknown"
            target = entry.get("gpu_target") or "Unknown"

            if agent_idx is not None:
                log(f"GPU name   (Agent {agent_idx}) : {name}")
                log(f"GPU target (Agent {agent_idx}) : {target}")
            else:
                log(f"GPU name   : {name}")
                log(f"GPU target : {target}")

    if not info.get("amd_smi_available", False):
        log("")
        log("[warning] amd-smi not available or failed.")
        err = info.get("amd_smi_error")
        if err:
            log(f"  amd-smi output: {str(err).splitlines()[0]}")

    if not info.get("rocminfo_available", False):
        log("")
        log("[warning] rocminfo not available or failed.")
        err = info.get("rocminfo_error")
        if err:
            log(f"  rocminfo output: {str(err).splitlines()[0]}")

    log("=== End of sanity check ===")


def main(argv: Optional[list[str]] = None) -> int:
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
