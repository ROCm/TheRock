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

from github_actions.github_actions_api import gha_append_step_summary, gha_set_output
from github_actions.manifest_utils import GitSourceInfo

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

# Release matrices start with this shared version range, then omit unsupported
# ref/version combinations below.
RELEASE_PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
CI_PYTHON_VERSIONS = {
    "linux": ["3.12"],
    "windows": ["3.12"],
}

UNSUPPORTED_PYTHON_VERSIONS = {
    "release/2.12": {"3.15"},
}

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
# * none schedules no self-hosted GPU tests. The wheel build job still runs
#   its build-time wheel validation.
# * standard also runs test_pytorch_wheels.yml on each selected AMDGPU family.
PYTORCH_TEST_LEVELS = ["none", "standard"]

# Release workflows limit standard GPU testing to one primary Python version
# per PyTorch ref. Use Python 3.11 to align with upstream's ROCm trunk testing.
# Revisit each ref's selection as upstream's test and support versions change:
# https://github.com/pytorch/pytorch/blob/main/.github/workflows/trunk.yml
# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#python
PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS = {
    "release/2.12": "3.11",
    "release/2.13": "3.11",
    "release/2.14": "3.11",
    "nightly": "3.11",
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


def _primary_test_python_version(pytorch_git_ref: str) -> str:
    """Return the primary test version for a PyTorch ref.

    Unknown explicit refs use the nightly policy because bring-up branches
    generally track upstream main.
    """
    return PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS.get(
        pytorch_git_ref, PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS["nightly"]
    )


def _select_test_level(
    *, release_type: str, python_version: str, pytorch_git_ref: str
) -> str:
    # multi_arch_ci does not schedule PyTorch GPU tests yet. Preserve its
    # standard level until that workflow's test policy is selected explicitly.
    if release_type == "ci":
        return "standard"
    if python_version == _primary_test_python_version(pytorch_git_ref):
        return "standard"
    return "none"


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
    #     "python_version": "3.11",
    #     "pytorch_git_ref": "release/2.12",
    #     "amdgpu_families": "gfx94X-dcgpu",
    #     "test_level": "standard"
    #   },
    #   ...
    #   {
    #     "python_version": "3.14",
    #     "pytorch_git_ref": "nightly",
    #     "amdgpu_families": "gfx94X-dcgpu",
    #     "test_level": "none"
    #   }
    # ]
    matrix: list[dict[str, str]] = []
    for py in versions:
        for ref in refs:
            if py in UNSUPPORTED_PYTHON_VERSIONS.get(ref, set()):
                continue
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
                "test_level": _select_test_level(
                    release_type=release_type,
                    python_version=py,
                    pytorch_git_ref=ref,
                ),
                "test_amdgpu_families": "auto",
            }
            matrix.append(row)
    return matrix


def check_source_versions(ref: str, sources: dict[str, GitSourceInfo]) -> list[str]:
    """Return version-policy errors for resolved source entries.

    Release refs require final versions; other refs require prereleases, except
    for Triton. Returning all errors lets the matrix check report them together.
    """
    # Matrix generation without version checks needs only the standard library.
    from packaging.version import InvalidVersion, Version

    errors: list[str] = []
    for project, source in sources.items():
        context = (
            f"{ref}: {project} {source.version!r} ({source.repo}/tree/{source.commit})"
        )
        try:
            version = Version(source.version or "")
        except InvalidVersion:
            errors.append(f"{context}: invalid package version")
            continue
        # is_prerelease includes dev releases as well as alpha, beta, and RC.
        # https://packaging.pypa.io/en/stable/version.html#packaging.version.Version.is_prerelease
        if ref.startswith("release/") and version.is_prerelease:
            errors.append(f"{context}: expected a stable package version")
        # Triton may use a final base version on nightly: torch pins the exact
        # Triton wheel, including its git/ROCm local version, for compatibility.
        elif (
            not ref.startswith("release/")
            and project != "triton"
            and not version.is_prerelease
        ):
            errors.append(f"{context}: expected a prerelease package version")
        else:
            print(f"Checked {context}")
    return errors


def check_matrix_versions(matrix: list[dict[str, str]], *, platform: str) -> None:
    """Check package versions in source repositories once per selected ref.

    Refs starting with "release/" require final (non-prerelease) package versions.
    Other refs require prerelease versions, except for Triton. All versions must
    be valid version strings. This checks the source version policy before builds.

    Note:
      * This queries GitHub at resolved commits, without checking out code.
      * Future scripts/steps may check out different code from floating refs.
    """

    # The manifest resolver also imports packaging transitively.
    from github_actions import generate_pytorch_source_manifest as source_manifest

    # TODO(https://github.com/ROCm/TheRock/issues/5110): move this code into
    #     generate_pytorch_source_manifest.py once that is used in workflows?

    errors: list[str] = []
    for ref in dict.fromkeys(row["pytorch_git_ref"] for row in matrix):
        print(f"Checking '{ref}' ref for valid versions")
        projects = source_manifest.default_projects_for_pytorch_ref(platform, ref)
        sources = source_manifest.resolve_sources(
            pytorch_ref=ref, version_suffix="", platform=platform, projects=projects
        )
        sources = source_manifest.fetch_versions(sources=sources, version_suffix="")
        errors.extend(check_source_versions(ref, sources))
        print()
    if errors:
        raise ValueError(
            "PyTorch release matrix contains unexpected package versions:\n  "
            + "\n  ".join(errors)
        )


def format_matrix_summary(
    *,
    release_type: str,
    platform: str,
    python_versions: list[str] | None,
    pytorch_git_refs: list[str] | None,
    amdgpu_families: str,
    matrix: list[dict[str, str]],
) -> str:
    """Format the resolved release matrix for logs and the job summary."""

    level_counts = {
        level: sum(row["test_level"] == level for row in matrix)
        for level in PYTORCH_TEST_LEVELS
    }
    count_summary = (
        ", ".join(
            f"`{level}`: {count}" for level, count in level_counts.items() if count
        )
        or "none"
    )
    selected_refs = list(dict.fromkeys(row["pytorch_git_ref"] for row in matrix))
    primary_versions = (
        ", ".join(
            f"`{ref}`: `{_primary_test_python_version(ref)}`" for ref in selected_refs
        )
        or "none"
    )
    lines = [
        "## PyTorch Release Matrix",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Release type | `{release_type}` |",
        f"| Platform | `{platform}` |",
        "| Python selection | "
        + (
            ", ".join(f"`{version}`" for version in python_versions)
            if python_versions
            else "default"
        )
        + " |",
        "| PyTorch ref selection | "
        + (
            ", ".join(f"`{ref}`" for ref in pytorch_git_refs)
            if pytorch_git_refs
            else "default"
        )
        + " |",
        f"| Requested AMDGPU families | `{amdgpu_families or 'none'}` |",
        f"| Generated rows | {len(matrix)} ({count_summary}) |",
    ]

    if not matrix:
        lines.extend(
            [
                "",
                "**Decision:** No build rows remain after applying the "
                "PyTorch-ref AMDGPU-family support filters.",
            ]
        )
    elif release_type == "ci":
        lines.extend(
            [
                "",
                "CI rows remain `standard` until multi-arch CI schedules "
                "PyTorch GPU tests and selects its test policy explicitly.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Standard GPU testing is assigned only to each PyTorch ref's "
                f"primary Python version: {primary_versions}. Other versions "
                "use `none`.",
            ]
        )

    if matrix and level_counts["none"] == len(matrix):
        lines.extend(
            [
                "",
                "All generated rows use `none` because none uses its "
                "PyTorch ref's primary test Python version.",
            ]
        )

    lines.extend(
        [
            "",
            "| Python | PyTorch ref | AMDGPU families | Test level |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in matrix:
        families = ", ".join(
            f"`{family}`" for family in _split_families(row["amdgpu_families"])
        )
        lines.append(
            f"| `{row['python_version']}` | `{row['pytorch_git_ref']}` | "
            f"{families} | `{row['test_level']}` |"
        )
    if not matrix:
        lines.append("| none | none | none | none |")
    return "\n".join(lines)


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
    parser.add_argument(
        "--check-versions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checks current package versions in the GitHub sources for each "
            "project in the matrix and fails if any unexpected versions are detected. "
            "Refs starting with release/ require final (non-prerelease) versions; "
            "other refs require prerelease versions except for Triton."
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
    if args.check_versions:
        check_matrix_versions(matrix, platform=args.platform)
    gha_append_step_summary(
        format_matrix_summary(
            release_type=args.release_type,
            platform=args.platform,
            python_versions=python_versions,
            pytorch_git_refs=pytorch_git_refs,
            amdgpu_families=args.amdgpu_families,
            matrix=matrix,
        )
    )
    gha_set_output({"pytorch_matrix": json.dumps(matrix)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
