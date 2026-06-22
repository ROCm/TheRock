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

RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
CI_PYTHON_VERSIONS = {
    "linux": ["3.12"],
    "windows": ["3.12"],
}

GFX125X_FAMILY = "gfx125X-dcgpu"
GFX125X_UNSUPPORTED = {GFX125X_FAMILY}

# Known PyTorch refs per platform. Each ref may carry canonical AMDGPU
# families that should be omitted while support is not available.
PYTORCH_REFS_LINUX: list[dict[str, object]] = [
    {
        "pytorch_git_ref": "release/2.9",
        "exclude_amdgpu_families": GFX125X_UNSUPPORTED,
    },
    {
        "pytorch_git_ref": "release/2.10",
        "exclude_amdgpu_families": GFX125X_UNSUPPORTED,
    },
    {
        "pytorch_git_ref": "release/2.11",
        "exclude_amdgpu_families": GFX125X_UNSUPPORTED,
    },
    {
        "pytorch_git_ref": "release/2.12",
        "exclude_amdgpu_families": GFX125X_UNSUPPORTED,
    },
    {
        "pytorch_git_ref": "nightly",
        "exclude_amdgpu_families": GFX125X_UNSUPPORTED,
    },
]

# gfx125X-dcgpu is Linux-only, so Windows currently needs no exclusions.
PYTORCH_REFS_WINDOWS: list[dict[str, object]] = [
    {"pytorch_git_ref": "release/2.9"},
    {"pytorch_git_ref": "release/2.10"},
    {"pytorch_git_ref": "release/2.11"},
    {"pytorch_git_ref": "release/2.12"},
    {"pytorch_git_ref": "nightly"},
]

CI_PYTORCH_REFS = {
    "linux": ["release/2.10", "release/2.11", "release/2.12"],
    "windows": ["release/2.10"],
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


def _ref_configs_for_platform(platform: str) -> list[dict[str, object]]:
    if platform == "windows":
        return PYTORCH_REFS_WINDOWS
    return PYTORCH_REFS_LINUX


def _default_ref_configs(
    *, release_type: str, platform: str
) -> list[dict[str, object]]:
    known_configs = _ref_configs_for_platform(platform)
    if release_type != "ci":
        return list(known_configs)

    ci_refs = set(CI_PYTORCH_REFS[platform])
    return [
        config for config in known_configs if str(config["pytorch_git_ref"]) in ci_refs
    ]


def _ref_config_for_ref(*, platform: str, pytorch_git_ref: str) -> dict[str, object]:
    for config in _ref_configs_for_platform(platform):
        if config["pytorch_git_ref"] == pytorch_git_ref:
            return config
    return {"pytorch_git_ref": pytorch_git_ref}


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


def generate_pytorch_matrix(
    *,
    python_versions: list[str] | None,
    pytorch_git_refs: list[str] | None,
    amdgpu_families: str,
    platform: str = "linux",
) -> list[dict[str, str]]:
    return generate_pytorch_matrix_for_release_type(
        release_type="dev",
        python_versions=python_versions,
        pytorch_git_refs=pytorch_git_refs,
        amdgpu_families=amdgpu_families,
        platform=platform,
    )


def generate_pytorch_matrix_for_release_type(
    *,
    release_type: str,
    python_versions: list[str] | None,
    pytorch_git_refs: list[str] | None,
    amdgpu_families: str,
    platform: str,
) -> list[dict[str, str]]:
    if release_type not in RELEASE_TYPES:
        raise ValueError(f"Unknown release_type: {release_type!r}")
    if platform not in ["linux", "windows"]:
        raise ValueError(f"Unknown platform: {platform!r}")

    versions = python_versions or _default_python_versions(
        release_type=release_type, platform=platform
    )
    ref_configs = (
        [
            _ref_config_for_ref(platform=platform, pytorch_git_ref=ref)
            for ref in pytorch_git_refs
        ]
        if pytorch_git_refs
        else _default_ref_configs(release_type=release_type, platform=platform)
    )

    matrix: list[dict[str, str]] = []
    for py in versions:
        for ref_cfg in ref_configs:
            ref = str(ref_cfg["pytorch_git_ref"])
            exclude = set(ref_cfg.get("exclude_amdgpu_families", set()))
            families = _filter_families(amdgpu_families, exclude)
            if not families:
                continue
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
