#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch build matrices for CI and release workflows."""

import argparse
import json
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

RELEASE_TYPES = ["ci", "dev", "nightly", "prerelease"]

# TODO: add opt-ins for CI runs to use python versions and pytorch refs normally
#       only included in release runs

RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
CI_PYTHON_VERSIONS = {
    "linux": ["3.12"],
    "windows": ["3.12"],
}

# Refs for the "prerelease" release type. The "nightly" release type extends
# this set with additional refs (see RELEASE_PYTORCH_REFS).
RELEASE_STABLE_PYTORCH_REFS = {
    "linux": [
        "release/2.10",
        "release/2.11",
        "release/2.12",
        "release/2.13",
    ],
    "windows": [
        "release/2.10",
        "release/2.11",
        "release/2.12",
        "release/2.13",
    ],
}

# Refs for the "nightly" release type: stable refs + "nightly" branch.
RELEASE_PYTORCH_REFS = {
    platform: [*refs, "nightly"]
    for platform, refs in RELEASE_STABLE_PYTORCH_REFS.items()
}

CI_PYTORCH_REFS = {
    "linux": ["release/2.10", "release/2.11", "release/2.12", "release/2.13"],
    "windows": ["release/2.10"],
}

# Unknown explicit refs are left unfiltered so bring-up branches can opt into
# new GPU families before the default PyTorch refs support them.
UNSUPPORTED_AMDGPU_FAMILIES = {
    "linux": {
        # gfx125x not supported for PyTorch 2.10.
        "release/2.10": {"gfx125X-dcgpu"},
        # gfx125x supported for PyTorch 2.11 via https://github.com/ROCm/pytorch/pull/3346.
        "release/2.11": {},
        # gfx125x supported for PyTorch 2.12 via https://github.com/ROCm/pytorch/pull/3421.
        "release/2.12": {},
        # gfx125x not yet enabled for PyTorch release/2.13 (ROCm/pytorch fork).
        # See https://github.com/ROCm/TheRock/issues/5833.
        "release/2.13": {"gfx125X-dcgpu", "gfx90c"},
        # gfx125x supported on upstream pytorch/pytorch nightly via pytorch#188597.
        # gfx90c excluded: blocked until CK submodule bump in pytorch nightly.
        "nightly": {"gfx90c"},
    },
    "windows": {
        "release/2.10": {"gfx90c"},
        "release/2.11": {"gfx90c"},
        "release/2.12": {"gfx90c"},
        "release/2.13": {"gfx90c"},
    },
}


def _split_values(raw: str) -> list[str]:
    """Split comma, semicolon, or whitespace-separated workflow input values."""
    return [
        value.strip()
        for value in raw.replace(",", " ").replace(";", " ").split()
        if value.strip()
    ]


def _split_families(raw: str) -> list[str]:
    return [family.strip() for family in raw.split(";") if family.strip()]


def _default_python_versions(*, release_type: str, platform: str) -> list[str]:
    if release_type == "ci":
        return list(CI_PYTHON_VERSIONS[platform])
    return list(RELEASE_PYTHON_VERSIONS)


def _default_pytorch_git_refs(*, release_type: str, platform: str) -> list[str]:
    if release_type == "ci":
        return list(CI_PYTORCH_REFS[platform])
    if release_type == "prerelease":
        return list(RELEASE_STABLE_PYTORCH_REFS[platform])
    return list(RELEASE_PYTORCH_REFS[platform])


def _filter_families(families_str: str, exclude: set[str]) -> str:
    """Remove excluded canonical family names from a semicolon-separated list."""
    if not exclude:
        return ";".join(_split_families(families_str))

    exclude_lower = {family.lower() for family in exclude}
    return ";".join(
        family
        for family in _split_families(families_str)
        if family.lower() not in exclude_lower
    )


def generate_pytorch_matrix_for_release_type(
    *,
    release_type: str,
    amdgpu_families: str,
    platform: str,
    python_versions: list[str] | None = None,
    pytorch_git_refs: list[str] | None = None,
) -> list[dict[str, str]]:
    if release_type not in RELEASE_TYPES:
        raise ValueError(f"Unknown release_type: {release_type!r}")
    if platform not in ["linux", "windows"]:
        raise ValueError(f"Unknown platform: {platform!r}")

    versions = python_versions or _default_python_versions(
        release_type=release_type, platform=platform
    )
    refs = pytorch_git_refs or _default_pytorch_git_refs(
        release_type=release_type, platform=platform
    )

    # Build one matrix row per requested Python version and PyTorch ref. Each
    # row carries the AMDGPU families that the child build workflow should use
    # for that ref after filtering out families that are not supported yet.
    #
    # Example Linux output for release_type="dev" and
    # amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu":
    #
    # [
    #   {
    #     "python_version": "3.10",
    #     "pytorch_git_ref": "release/2.12",
    #     "amdgpu_families": "gfx94X-dcgpu"
    #   },
    #   ...
    #   {
    #     "python_version": "3.14",
    #     "pytorch_git_ref": "nightly",
    #     "amdgpu_families": "gfx94X-dcgpu"
    #   }
    # ]
    matrix: list[dict[str, str]] = []
    for py in versions:
        for ref in refs:
            exclude = UNSUPPORTED_AMDGPU_FAMILIES[platform].get(ref, set())
            families = _filter_families(amdgpu_families, exclude)
            if not families:
                continue
            # These row keys are the contract with workflow files which use them
            # via matrix.<key> expressions. Empty values are allowed when the
            # workflow handles them explicitly, but undefined keys are not.
            row: dict[str, str] = {
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
        help=(
            "Comma, semicolon, or whitespace separated list of Python versions "
            "(default depends on --release-type)"
        ),
    )
    parser.add_argument(
        "--pytorch-git-refs",
        type=str,
        default="",
        help=(
            "Comma, semicolon, or whitespace separated list of PyTorch refs "
            "(default depends on --release-type and --platform)"
        ),
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        choices=["linux", "windows"],
        help="Platform to generate matrix for (default: current system)",
    )
    parser.add_argument(
        "--release-type",
        type=str,
        default="dev",
        choices=RELEASE_TYPES,
        help="Release type selecting default PyTorch/Python matrix (default: dev)",
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

    python_versions = _split_values(args.python_versions) or None
    pytorch_git_refs = _split_values(args.pytorch_git_refs) or None

    matrix = generate_pytorch_matrix_for_release_type(
        release_type=args.release_type,
        python_versions=python_versions,
        pytorch_git_refs=pytorch_git_refs,
        amdgpu_families=args.amdgpu_families,
        platform=args.platform,
    )
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
