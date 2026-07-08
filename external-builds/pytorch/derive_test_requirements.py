#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Print pinned test requirements sourced from the vendored PyTorch checkout.

Extracts the pins for the requested packages (default: ``numpy``) from the
checked-out PyTorch's ``.ci/docker/requirements-ci.txt`` and prints them, so
TheRock installs the exact versions upstream validates for the torch ref under
test instead of a hand-copied pin that silently drifts. Environment markers
(e.g. ``; python_version ...``) are preserved so pip selects the right line.

Usage:
    python derive_test_requirements.py
    python derive_test_requirements.py --package numpy --package scipy
    python derive_test_requirements.py | python -m pip install -r /dev/stdin
"""

import argparse
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REQUIREMENTS_CI = (
    THIS_DIR / "pytorch" / ".ci" / "docker" / "requirements-ci.txt"
)


def extract_pins(requirements_ci: Path, packages: list[str]) -> list[str]:
    """Return the requirement lines for each package, in the given order."""
    lines = requirements_ci.read_text().splitlines()
    pins: list[str] = []
    for package in packages:
        pattern = re.compile(rf"^\s*{re.escape(package)}\s*(==|>=|<=|<|>|~=|;|$)")
        matches = [line.rstrip() for line in lines if pattern.match(line)]
        if not matches:
            raise ValueError(f"no pin for '{package}' found in {requirements_ci}")
        pins.extend(matches)
    return pins


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help="Package to extract (repeatable). Defaults to numpy.",
    )
    parser.add_argument(
        "--requirements-ci",
        type=Path,
        default=DEFAULT_REQUIREMENTS_CI,
        help=f"PyTorch requirements-ci.txt to read. Default: {DEFAULT_REQUIREMENTS_CI}",
    )
    args = parser.parse_args(argv)

    if not args.requirements_ci.exists():
        parser.error(f"requirements-ci.txt not found: {args.requirements_ci}")

    print("\n".join(extract_pins(args.requirements_ci, args.packages or ["numpy"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
