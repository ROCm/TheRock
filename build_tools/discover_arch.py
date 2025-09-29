#!/usr/bin/env python

"""discover_arch.py

This script is a Python port of the Linux version of the amdgpu-arch binary.
It's meant to serve first-time users without an existing ROCm installation in discovering the gfx target(s)
they have on their system.

Currently it's only available to Linux users.

TODO: Implement a Python function with minimal dependenices that queries GPUs on Windows and
maps them to the appropriate gfx target.
"""

import sys
import platform
import argparse

from pathlib import Path

# Path to the sysfs directory provided by the AMD KFD driver
KFD_SYSFS_NODE_PATH = Path("/sys/devices/virtual/kfd/kfd/topology/nodes")


def get_major(ver):
    return (ver // 10000) % 100


def get_minor(ver):
    return (ver // 100) % 100


def get_step(ver):
    return ver % 100


def parse_properties_file(path):
    """Reads the 'gfx_target_version' value from a given properties file."""
    try:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("gfx_target_version"):
                    _, version_str = line.split("gfx_target_version", 1)
                    version_str = version_str.strip()
                    if version_str.isdigit():
                        return int(version_str)
    except Exception:
        pass
    return 0  # Return 0 if parsing fails or if file isn't found


def get_amd_gpus_by_kfd():
    """Parses the KFD sysfs interface to find installed AMD GPUs."""
    if not KFD_SYSFS_NODE_PATH.exists():
        print("KFD sysfs path not found.")
        return []

    devices = []

    for entry in KFD_SYSFS_NODE_PATH.iterdir():
        if not entry.is_dir():
            continue

        try:
            node = int(entry.name)
        except ValueError:
            continue  # Skip non-numeric node names

        properties_path = entry / "properties"
        gfx_version = parse_properties_file(properties_path)

        if gfx_version == 0:
            continue  # Likely a CPU node

        devices.append((node, gfx_version))

    # Sort devices by node number
    devices.sort(key=lambda x: x[0])

    # Format output: gfx<major><minor><step in hex>
    gpu_names = [
        f"gfx{get_major(ver)}{get_minor(ver)}{get_step(ver):x}" for _, ver in devices
    ]
    return gpu_names


def main(args: list[str]):
    p = argparse.ArgumentParser(
        "discover_arch.py",
        usage="discover_arch.py",
        description="Prints all the available gfx architectures on the system. (Currently Linux only)",
    )
    p.parse_args(args)
    if platform.system() == "Windows":
        print("The Windows version of this script is still being developed.")
    else:
        gpus = get_amd_gpus_by_kfd()
        if gpus:
            for gpu in gpus:
                print(f"{gpu}")
        else:
            print("No AMD GPUs detected or no access to KFD sysfs.")


if __name__ == "__main__":
    main(sys.argv[1:])
