#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate PyTorch build matrices for release and CI workflows."""

import argparse
from dataclasses import dataclass
import json
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

RELEASE_TYPES = ["ci", "dev", "nightly", "prerelease"]

RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
CI_PYTHON_VERSIONS = ["3.12"]


@dataclass(frozen=True)
class PyTorchRefConfig:
    pytorch_git_ref: str
    exclude_amdgpu_families: frozenset[str] = frozenset()


LINUX_RELEASE_2_9 = PyTorchRefConfig(
    pytorch_git_ref="release/2.9",
    exclude_amdgpu_families=frozenset({"gfx125x"}),
)
LINUX_RELEASE_2_10 = PyTorchRefConfig(
    pytorch_git_ref="release/2.10",
    exclude_amdgpu_families=frozenset({"gfx125x"}),
)
LINUX_RELEASE_2_11 = PyTorchRefConfig(pytorch_git_ref="release/2.11")
LINUX_RELEASE_2_12 = PyTorchRefConfig(
    pytorch_git_ref="release/2.12",
    exclude_amdgpu_families=frozenset({"gfx125x"}),
)
LINUX_NIGHTLY = PyTorchRefConfig(
    pytorch_git_ref="nightly",
    exclude_amdgpu_families=frozenset({"gfx125x"}),
)

RELEASE_PYTORCH_REFS_LINUX = [
    LINUX_RELEASE_2_9,
    LINUX_RELEASE_2_10,
    LINUX_RELEASE_2_11,
    LINUX_RELEASE_2_12,
    LINUX_NIGHTLY,
]

RELEASE_PYTORCH_REFS_WINDOWS = [
    PyTorchRefConfig(pytorch_git_ref="release/2.9"),
    PyTorchRefConfig(pytorch_git_ref="release/2.10"),
    PyTorchRefConfig(pytorch_git_ref="release/2.11"),
    PyTorchRefConfig(pytorch_git_ref="release/2.12"),
    PyTorchRefConfig(pytorch_git_ref="nightly"),
]

CI_PYTORCH_REFS_LINUX = [
    LINUX_RELEASE_2_10,
    LINUX_RELEASE_2_11,
    LINUX_RELEASE_2_12,
]

CI_PYTORCH_REFS_WINDOWS = [
    PyTorchRefConfig(pytorch_git_ref="release/2.10"),
]


def _split_list(raw: str) -> list[str]:
    """Split comma- or semicolon-separated workflow inputs."""
    return [item.strip() for item in raw.replace(",", ";").split(";") if item.strip()]


def _default_python_versions(release_type: str) -> list[str]:
    if release_type == "ci":
        return CI_PYTHON_VERSIONS
    return RELEASE_PYTHON_VERSIONS


def _default_pytorch_refs(release_type: str, platform: str) -> list[PyTorchRefConfig]:
    if release_type == "ci":
        if platform == "windows":
            return CI_PYTORCH_REFS_WINDOWS
        return CI_PYTORCH_REFS_LINUX

    if platform == "windows":
        return RELEASE_PYTORCH_REFS_WINDOWS
    return RELEASE_PYTORCH_REFS_LINUX


def _is_excluded_family(family: str, excluded_family: str) -> bool:
    """Match either canonical family names or their base family prefix."""
    normalized_family = family.casefold()
    normalized_excluded = excluded_family.casefold()
    return (
        normalized_family == normalized_excluded
        or normalized_family.split("-", 1)[0] == normalized_excluded
    )


def filter_pytorch_amdgpu_families(
    amdgpu_families: str,
    exclude_amdgpu_families: frozenset[str],
) -> str:
    """Remove AMDGPU families that a PyTorch ref should not build yet."""
    families = _split_list(amdgpu_families)
    return ";".join(
        family
        for family in families
        if not any(
            _is_excluded_family(family, excluded_family)
            for excluded_family in exclude_amdgpu_families
        )
    )


def generate_pytorch_matrix(
    *,
    release_type: str,
    platform: str = "linux",
    python_versions: list[str] | None = None,
    amdgpu_families: str = "",
) -> list[dict[str, str]]:
    versions = (
        python_versions if python_versions else _default_python_versions(release_type)
    )
    pytorch_refs = _default_pytorch_refs(release_type, platform)

    matrix = []
    for py in versions:
        for ref_config in pytorch_refs:
            filtered_amdgpu_families = filter_pytorch_amdgpu_families(
                amdgpu_families=amdgpu_families,
                exclude_amdgpu_families=ref_config.exclude_amdgpu_families,
            )
            if amdgpu_families and not filtered_amdgpu_families:
                continue
            matrix.append(
                {
                    "python_version": py,
                    "pytorch_git_ref": ref_config.pytorch_git_ref,
                    "amdgpu_families": filtered_amdgpu_families,
                }
            )
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PyTorch build matrix")
    parser.add_argument(
        "--release-type",
        type=str,
        choices=RELEASE_TYPES,
        default="dev",
        help="Release type selecting default PyTorch refs and Python versions",
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
            "Families unsupported by a given PyTorch ref are filtered from "
            "that ref's matrix row."
        ),
    )
    args = parser.parse_args(argv)

    python_versions = None
    if args.python_versions:
        python_versions = _split_list(args.python_versions)

    matrix = generate_pytorch_matrix(
        release_type=args.release_type,
        platform=args.platform,
        python_versions=python_versions,
        amdgpu_families=args.amdgpu_families,
    )
    gha_set_output(
        {
            "build_pytorch": json.dumps(bool(matrix)),
            "pytorch_matrix": json.dumps(matrix),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
