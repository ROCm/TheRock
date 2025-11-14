#!/usr/bin/env python3
"""
Script Name: mapping_built_wheels.py

Usage:
  python mapping_built_wheels.py --dir <directory_path>

Description:
  This script scans the specified directory for Python wheel (.whl) files,
  collects their filenames, and prints a JSON array of those filenames to stdout.

Requirements:
  - the directory specified by --dir must exist else the script will raise a FileNotFoundError
Example:
  python mapping_built_wheels.py --dir wheelhouse
  Output: ["jax_rocm7_pjrt-0.7.1-py3-none-manylinux_2_28_x86_64.whl","jax_rocm7_plugin-0.7.1-cp312-cp312-manylinux_2_28_x86_64.whl","jaxlib-0.7.1.dev0+selfbuilt-cp312-cp312-manylinux_2_27_x86_64.whl"]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

def collect_wheel_filenames(directory: Path) -> List[str]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(whl.name for whl in directory.glob("*.whl"))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    array = collect_wheel_filenames(Path(args.dir))
    json_array = json.dumps(array, separators=(",", ","))
    print(json_array)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
