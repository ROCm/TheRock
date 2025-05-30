#!/usr/bin/env python

import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: sanity_check_wheel.py <wheel_directory>")
        sys.exit(2)

    wheel_dir = Path(sys.argv[1])
    if not wheel_dir.is_dir():
        print(f"ERROR: {wheel_dir} is not a directory")
        sys.exit(1)

    wheels = list(wheel_dir.glob("torch-*.whl"))
    if not wheels:
        print("ERROR: No torch-*.whl file found")
        sys.exit(1)

    found_valid = False
    for wheel in wheels:
        size = wheel.stat().st_size
        if size < 100:
            print(f"ERROR: Wheel {wheel.name} is too small ({size} bytes)")
            sys.exit(1)
        print(f" Found valid wheel: {wheel.name} ({size} bytes)")
        found_valid = True

    if not found_valid:
        print("ERROR: Invalid torch wheel found")
        sys.exit(1)


if __name__ == "__main__":
    main()
