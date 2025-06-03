#!/usr/bin/env python

import sys
from pathlib import Path
import re


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
