#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import re

import rewrite_torch_runpath


def check_wheel(wheel: Path, expected_name: str):
    # Check 1: Filename starts with expected prefix
    if not wheel.name.startswith(f"{expected_name}-"):
        print(f"ERROR: Unexpected wheel name: {wheel.name}")
        sys.exit(1)

    # Check 2: File size sanity
    size = wheel.stat().st_size
    if size < 100 * 1024:  # minimum 100 KB
        print(f"ERROR: Wheel {wheel.name} is too small ({size} bytes)")
        sys.exit(1)

    # Check 3: Wheel name format (e.g. torch-2.1.0+rocmsdk20250529-cp312-cp312-linux_x86_64.whl)
    wheel_name_re = re.compile(rf"^{re.escape(expected_name)}-[\d\.]+.*\.whl$")
    if not wheel_name_re.match(wheel.name):
        print(f"WARNING: Wheel name {wheel.name} does not match typical pattern")

    print(f"Valid wheel: {wheel.name} ({size} bytes)")
    if expected_name == "torch" and sys.platform != "win32":
        check_torch_rpath(wheel)


def check_torch_rpath(wheel: Path) -> None:
    """Fail if torch still has manylinux-builder _rocm_sdk_devel RUNPATHs."""
    try:
        patchelf = rewrite_torch_runpath.find_patchelf()
    except FileNotFoundError as exc:
        print(f"WARNING: skipping RPATH check: {exc}")
        return

    with tempfile.TemporaryDirectory(prefix="therock-rpath-check-") as td:
        root = Path(td)
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(root)
        shared_objects = rewrite_torch_runpath.iter_shared_objects(root)
        if not shared_objects:
            print(f"WARNING: {wheel.name} contains no shared objects to RPATH-check")
            return
        checked = 0
        for sofile in shared_objects:
            result = subprocess.run(
                [str(patchelf), "--print-rpath", str(sofile)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue
            rpath = result.stdout.strip()
            checked += 1
            rel = sofile.relative_to(root).as_posix()
            if rewrite_torch_runpath.rpath_contains_builder_path(rpath):
                print(f"ERROR: {wheel.name}:{rel} still has a builder-absolute RPATH:")
                print(f"  {rpath}")
                sys.exit(1)
            if "_rocm_sdk_core" not in rpath:
                print(
                    f"ERROR: {wheel.name}:{rel} RPATH does not mention _rocm_sdk_core:"
                )
                print(f"  {rpath}")
                sys.exit(1)
        if checked == 0:
            print(
                f"WARNING: patchelf could not read RPATH from any .so in {wheel.name}"
            )
            return
        print(f"RPATH ok on {checked} shared object(s) in {wheel.name}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <wheel-dir>")
        sys.exit(1)

    wheel_dir = Path(sys.argv[1])
    if not wheel_dir.is_dir():
        print(f"ERROR: {wheel_dir} is not a directory")
        sys.exit(1)

    # Expected names: torch, torchaudio, torchvision
    for expected_name in ["torch", "torchaudio", "torchvision"]:
        wheels = list(wheel_dir.glob(f"{expected_name}-*.whl"))
        for wheel in wheels:
            check_wheel(wheel, expected_name)


if __name__ == "__main__":
    main()
