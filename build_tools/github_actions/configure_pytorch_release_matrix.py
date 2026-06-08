#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch release build matrix for workflows."""

import argparse
import json
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

PYTORCH_REFS_LINUX = [
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "release/2.12",
    "nightly",
]

PYTORCH_REFS_WINDOWS = [
    "release/2.9",
    "release/2.10",
    "release/2.11",
    "release/2.12",
    "nightly",
]


def generate_pytorch_matrix(
    python_versions: list[str] | None,
    platform: str = "linux",
) -> list[dict]:
    versions = python_versions if python_versions else PYTHON_VERSIONS
    pytorch_refs = PYTORCH_REFS_WINDOWS if platform == "windows" else PYTORCH_REFS_LINUX
    matrix = []
    for py in versions:
        for ref in pytorch_refs:
            matrix.append({"python_version": py, "pytorch_git_ref": ref})
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch release build matrix"
    )
    parser.add_argument(
        "--python-versions",
        type=str,
        default="",
        help="Comma or semicolon separated list of Python versions (default: all)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        choices=["linux", "windows"],
        help="Platform to generate matrix for (default: current system)",
    )
    args = parser.parse_args(argv)

    python_versions = None
    if args.python_versions:
        sep = ";" if ";" in args.python_versions else ","
        python_versions = [
            v.strip() for v in args.python_versions.split(sep) if v.strip()
        ]

    matrix = generate_pytorch_matrix(python_versions, args.platform)
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
