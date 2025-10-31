#!/usr/bin/env python3
"""
Rename JAX ROCm plugin wheels to include ROCm version.
"""
import argparse
import re
import sys
from pathlib import Path


def rename_wheels(wheel_dir: Path, rocm_version: str) -> None:
    """
    Rename wheels to replace version with ROCm version.

    Args:
        wheel_dir: Directory containing wheel files
        rocm_version: ROCm version string (e.g., "7.10.0a20251021")
    """
    # Pattern for standard wheels (e.g., cp312-cp312)
    wheel_pattern = re.compile(r"^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+)-(cp\d+-[^.]+\.whl)$")

    # Pattern for universal wheels (e.g., py3-none)
    universal_pattern = re.compile(
        r"^([^-]+)-([0-9]+\.[0-9]+\.[0-9]+)-(py\d+-[^.]+\.whl)$"
    )

    renamed_count = 0
    skipped_count = 0

    for wheel_file in wheel_dir.glob("*.whl"):
        match = wheel_pattern.match(wheel_file.name) or universal_pattern.match(
            wheel_file.name
        )

        if match:
            package_name, old_version, rest = match.groups()
            new_name = f"{package_name}-{rocm_version}-{rest}"
            new_path = wheel_dir / new_name

            if wheel_file.name != new_name:
                wheel_file.rename(new_path)
                print(f"Renamed {wheel_file.name} to {new_name}")
                renamed_count += 1
            else:
                print(f"Skipping {wheel_file.name} (no change needed)")
                skipped_count += 1
        else:
            print(
                f"Warning: {wheel_file.name} doesn't match expected pattern",
                file=sys.stderr,
            )
            skipped_count += 1

    print(f"\nSummary: Renamed {renamed_count} file(s), skipped {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename JAX ROCm plugin wheels with ROCm version"
    )
    parser.add_argument("wheel_dir", type=Path, help="Directory containing wheel files")
    parser.add_argument(
        "rocm_version", type=str, help="ROCm version to use (e.g., 7.10.0a20251021)"
    )

    args = parser.parse_args()

    if not args.wheel_dir.is_dir():
        print(f"Error: {args.wheel_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    rename_wheels(args.wheel_dir, args.rocm_version)


if __name__ == "__main__":
    main()
