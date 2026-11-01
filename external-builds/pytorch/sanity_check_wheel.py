#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
import re
import zipfile

# The error torchvision compiles in for each codec it was built without.
CODEC_MISSING_RE = re.compile(
    rb"torchvision not compiled with (libPNG|libpng|libjpeg|libwebp) support"
)


def missing_image_codecs(wheel: Path) -> list[str]:
    """Returns the codecs torchvision's image extension was built without."""
    missing = set()
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if re.match(r"torchvision/image.*\.so$", name):
                found = CODEC_MISSING_RE.findall(zf.read(name))
                missing.update(m.decode().lower() for m in found)
    return sorted(missing)


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

    # Check 4: torchvision's image extension has its codecs. setup.py builds it
    # without them (with only a warning) when the libraries aren't found.
    if expected_name == "torchvision" and "linux" in wheel.name:
        missing = missing_image_codecs(wheel)
        if missing:
            print(f"ERROR: Wheel {wheel.name} was built without {', '.join(missing)}")
            sys.exit(1)

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
