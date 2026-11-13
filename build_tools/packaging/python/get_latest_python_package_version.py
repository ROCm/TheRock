#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Find the latest compatible wheel version in a Python package index.

Uses pip's JSON output without downloading or installing packages. Compatibility
is checked for the running interpreter. Pre-release versions are included.
"""

import argparse
import json
import subprocess
import sys


def get_latest_python_package_version(package: str, index_url: str) -> str:
    """Return the latest version, failing if pip cannot find a compatible wheel."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "index",
            "versions",
            package,
            "--index-url",
            index_url,
            "--pre",
            "--only-binary=:all:",
            "--json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    version = json.loads(result.stdout).get("latest")
    if not isinstance(version, str) or not version:
        raise ValueError(f"No latest version returned for {package} at {index_url}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="Package name to look up")
    parser.add_argument("--index-url", required=True, help="Python package index URL")
    args = parser.parse_args()
    print(get_latest_python_package_version(args.package, args.index_url))


if __name__ == "__main__":
    main()
