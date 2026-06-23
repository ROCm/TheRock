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

# PyTorch git refs per platform.
# Each ref may optionally carry an `exclude_amdgpu_families` set: GPU families
# that should be omitted for this ref (e.g. because upstream support is not yet
# merged). When `exclude_amdgpu_families` is absent the ref inherits whatever
# families are passed in via --amdgpu-families.
#
# Format: list of dicts with keys:
#   pytorch_git_ref         – git branch/tag passed to the build job
#   exclude_amdgpu_families – (optional) set[str] of family names to drop

PYTORCH_REFS_LINUX: list[dict] = [
    {
        "pytorch_git_ref": "release/2.9",
        # gfx125x not supported for PyTorch 2.9.
        "exclude_amdgpu_families": {"gfx125x"},
    },
    {
        "pytorch_git_ref": "release/2.10",
        # gfx125x not supported for PyTorch 2.10.
        "exclude_amdgpu_families": {"gfx125x"},
    },
    {
        "pytorch_git_ref": "release/2.11",
        # gfx125x not yet upstreamed to pytorch/pytorch.
        # See https://github.com/ROCm/TheRock/issues/5833.
        "exclude_amdgpu_families": {"gfx125x"},
    },
    {
        "pytorch_git_ref": "release/2.12",
        # gfx125x not yet upstreamed to pytorch/pytorch. Upstream expected
        # 2026-06-26, but the ROCm 7.14 release is cut before that date.
        # See https://github.com/ROCm/TheRock/issues/5833.
        "exclude_amdgpu_families": {"gfx125x"},
    },
    {
        "pytorch_git_ref": "nightly",
        # gfx125x not yet upstreamed to pytorch/pytorch.
        # See https://github.com/ROCm/TheRock/issues/5833.
        "exclude_amdgpu_families": {"gfx125x"},
    },
]

# gfx125x is Linux-only; no exclusion needed for Windows.
PYTORCH_REFS_WINDOWS: list[dict] = [
    {"pytorch_git_ref": "release/2.9"},
    {"pytorch_git_ref": "release/2.10"},
    {"pytorch_git_ref": "release/2.11"},
    {"pytorch_git_ref": "release/2.12"},
    {"pytorch_git_ref": "nightly"},
]


def _filter_families(families_str: str, exclude: set[str]) -> str:
    """Remove excluded family names from a semicolon-separated families string.

    Family names are matched case-insensitively against the exclude set so that
    e.g. ``gfx125x`` matches ``gfx125X-dcgpu`` style entries.
    """
    if not exclude:
        return families_str
    result = []
    for fam in families_str.split(";"):
        fam = fam.strip()
        if not fam:
            continue
        # Match by checking whether any excluded name is a case-insensitive
        # prefix/substring of the family token (e.g. "gfx125x" in "gfx125X-dcgpu").
        skip = any(exc.lower() in fam.lower() for exc in exclude)
        if not skip:
            result.append(fam)
    return ";".join(result)


def generate_pytorch_matrix(
    python_versions: list[str] | None,
    amdgpu_families: str,
    platform: str = "linux",
) -> list[dict]:
    versions = python_versions if python_versions else PYTHON_VERSIONS
    pytorch_refs = PYTORCH_REFS_WINDOWS if platform == "windows" else PYTORCH_REFS_LINUX
    matrix = []
    for py in versions:
        for ref_cfg in pytorch_refs:
            ref = ref_cfg["pytorch_git_ref"]
            exclude = ref_cfg.get("exclude_amdgpu_families", set())
            families = _filter_families(amdgpu_families, exclude)
            row: dict = {
                "python_version": py,
                "pytorch_git_ref": ref,
                "amdgpu_families": families,
            }
            matrix.append(row)
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
    parser.add_argument(
        "--amdgpu-families",
        type=str,
        default="",
        help=(
            "Semicolon-separated AMD GPU families to build PyTorch for. "
            "Families that are not supported for a given PyTorch ref will be "
            "filtered out of this list for that ref's matrix entry."
        ),
    )
    args = parser.parse_args(argv)

    python_versions = None
    if args.python_versions:
        sep = ";" if ";" in args.python_versions else ","
        python_versions = [
            v.strip() for v in args.python_versions.split(sep) if v.strip()
        ]

    matrix = generate_pytorch_matrix(
        python_versions, args.amdgpu_families, args.platform
    )
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
