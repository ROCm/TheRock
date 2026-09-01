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

# Build matrix configuration.
RELEASE_TYPES = [
    "ci",
    "dev",
    "dev-bkc",
    "nightly",
    "nightly-bkc",
    "prerelease",
]

# TODO: add opt-ins for CI runs to use python versions and pytorch refs normally
#       only included in release runs

RELEASE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# Refs for the "prerelease" release type. The "nightly" release type extends
# this set with additional refs (see RELEASE_PYTORCH_REFS).
RELEASE_STABLE_PYTORCH_REFS = {
    "linux": [
        "release/2.12",
        "release/2.13",
        "release/2.14",
    ],
    "windows": [
        "release/2.12",
        "release/2.13",
        "release/2.14",
    ],
}

# Refs for release types: stable refs + "nightly" branch.
RELEASE_PYTORCH_REFS = {
    platform: [*refs, "nightly"]
    for platform, refs in RELEASE_STABLE_PYTORCH_REFS.items()
}

CI_PYTORCH_REFS = {
    "linux": ["release/2.12", "release/2.13"],
    "windows": ["release/2.12"],
}

# Unknown explicit refs are left unfiltered so bring-up branches can opt into
# new GPU families before the default PyTorch refs support them.
UNSUPPORTED_AMDGPU_FAMILIES = {
    "linux": {
        "release/2.12": {},
        "release/2.13": {"gfx90c"},
        "release/2.14": {"gfx90c"},
        "nightly": {},
    },
    "windows": {
        "release/2.12": {"gfx90c"},
        "release/2.13": {"gfx90c"},
        "release/2.14": {"gfx90c"},
    },
}

# Test coverage configuration.
#
# PyTorch test levels are additive:
#
# * sanity runs only sanity_check_wheel.py in the wheel build job. This checks
#   the wheel on a CPU runner without scheduling self-hosted GPU tests.
# * standard also runs test_pytorch_wheels.yml on each selected AMDGPU family.
# * full also dispatches test_pytorch_wheels_full.yml, which runs the much
#   larger upstream PyTorch test suite and can take several hours.
PYTORCH_TEST_LEVELS = ["sanity", "standard", "full"]

# CI and release workflows limit the bulk of testing (standard/full test
# levels) to the oldest Python version supported by each PyTorch ref. Match
# upstream's trunk test version and release support policy:
# https://github.com/pytorch/pytorch/blob/main/.github/workflows/trunk.yml
# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#python
PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS = {
    "release/2.12": "3.10",
    "release/2.13": "3.10",
    "release/2.14": "3.10",
    "nightly": "3.10",
}

# The full PyTorch test suite can take several hours, so workflows by default
# run it on only one representative configuration.
PYTORCH_FULL_TEST_PLATFORM = "linux"
PYTORCH_FULL_TEST_AMDGPU_FAMILY = "gfx94X-dcgpu"


def _split_values(raw: str) -> list[str]:
    """Split comma, semicolon, or whitespace-separated workflow input values."""
    return [
        value.strip()
        for value in raw.replace(",", " ").replace(";", " ").split()
        if value.strip()
    ]


def _split_families(raw: str) -> list[str]:
    return [family.strip() for family in raw.split(";") if family.strip()]


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


def _primary_test_python_version(pytorch_git_ref: str) -> str:
    """Return the primary test version for a PyTorch ref.

    Unknown explicit refs use the nightly policy because bring-up branches
    generally track upstream main.
    """
    return PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS.get(
        pytorch_git_ref, PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS["nightly"]
    )


def _select_release_test_level(
    *,
    python_version: str,
    pytorch_git_ref: str,
    platform: str,
    amdgpu_families: list[str],
    run_full_pytorch_tests: bool,
) -> str:
    """Select the shared PyTorch test level for a scheduled release row.

    Use CPU-only sanity coverage outside the ref's primary test Python version.
    That version receives standard per-family GPU coverage, or the
    hours-long full suite on the one representative configuration selected by
    the release policy above.
    """
    primary_python_version = _primary_test_python_version(pytorch_git_ref)
    if python_version != primary_python_version:
        return "sanity"
    if (
        run_full_pytorch_tests
        and platform == PYTORCH_FULL_TEST_PLATFORM
        and PYTORCH_FULL_TEST_AMDGPU_FAMILY.lower()
        in {family.lower() for family in amdgpu_families}
    ):
        return "full"
    return "standard"


def generate_pytorch_matrix_for_release_type(
    *,
    release_type: str,
    amdgpu_families: str,
    platform: str,
    python_versions: list[str] | None = None,
    pytorch_git_refs: list[str] | None = None,
    run_full_pytorch_tests: bool = False,
) -> list[dict[str, str]]:
    if release_type not in RELEASE_TYPES:
        raise ValueError(f"Unknown release_type: {release_type!r}")
    if platform not in ["linux", "windows"]:
        raise ValueError(f"Unknown platform: {platform!r}")

    refs = pytorch_git_refs or _default_pytorch_git_refs(
        release_type=release_type, platform=platform
    )
    if release_type == "ci" and not python_versions:
        # The reduced CI matrix uses each ref's primary test version instead of
        # building the full Python-version x PyTorch-ref product.
        version_ref_pairs = [(_primary_test_python_version(ref), ref) for ref in refs]
    else:
        versions = python_versions or RELEASE_PYTHON_VERSIONS
        version_ref_pairs = [
            (python_version, ref) for python_version in versions for ref in refs
        ]

    # Build one matrix row per selected Python-version/PyTorch-ref pair. Each
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
    #     "amdgpu_families": "gfx94X-dcgpu",
    #     "test_level": "standard"
    #   },
    #   ...
    #   {
    #     "python_version": "3.14",
    #     "pytorch_git_ref": "nightly",
    #     "amdgpu_families": "gfx94X-dcgpu",
    #     "test_level": "sanity"
    #   }
    # ]
    matrix: list[dict[str, str]] = []
    for py, ref in version_ref_pairs:
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
            "test_level": _select_release_test_level(
                python_version=py,
                pytorch_git_ref=ref,
                platform=platform,
                amdgpu_families=_split_families(families),
                run_full_pytorch_tests=run_full_pytorch_tests,
            ),
            # TODO(#7185): PyTorch nightly's requirements-ci.txt pins
            # scikit-image==0.22.0, which has no cp314 wheel and fails to
            # build from source. Build those wheels but skip their tests
            # until that is fixed.
            "test_amdgpu_families": (
                "none"
                if (platform, ref, py) == ("windows", "nightly", "3.14")
                else "auto"
            ),
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
        "--run-full-pytorch-tests",
        action="store_true",
        help="Use the full test level for eligible Python matrix rows",
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
        run_full_pytorch_tests=args.run_full_pytorch_tests,
    )
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
