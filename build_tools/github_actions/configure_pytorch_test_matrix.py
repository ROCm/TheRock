#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure quick PyTorch wheel test jobs for multi-arch build workflows."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.amdgpu_family_matrix import get_all_families_for_trigger_types
from github_actions.github_actions_api import gha_set_output


@dataclass(frozen=True)
class PyTorchTestMatrixEntry:
    amdgpu_family: str
    test_runs_on: str
    package_index_url: str

    def to_dict(self) -> dict[str, str]:
        return {
            "amdgpu_family": self.amdgpu_family,
            "test_runs_on": self.test_runs_on,
            "package_index_url": self.package_index_url,
        }


SKIP_TEST_VALUES = {"none", "skip", "false", "off", "0"}
AUTO_TEST_VALUES = {"", "auto", "built"}


def split_families(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", " ").split() if item.strip()]


def find_amdgpu_family(
    *, requested_family: str, platform: str
) -> tuple[str, dict[str, object]]:
    """Return canonical family and platform info for a requested family/target."""
    requested_lower = requested_family.lower()
    matrix = get_all_families_for_trigger_types(["presubmit", "postsubmit", "nightly"])
    for key, info_for_key in matrix.items():
        platform_info = info_for_key.get(platform)
        if not platform_info:
            continue
        family = platform_info.get("family")
        fetch_targets = platform_info.get("fetch-gfx-targets", [])
        if not isinstance(family, str):
            continue
        if not isinstance(fetch_targets, list):
            fetch_targets = []

        if (
            requested_lower == key.lower()
            or requested_lower == family.lower()
            or requested_family in fetch_targets
        ):
            return family, platform_info

    raise ValueError(
        f"No {platform} AMDGPU family entry found for {requested_family!r}"
    )


def build_test_matrix(
    *,
    amdgpu_families: list[str],
    platform: str,
    package_index_url: str,
) -> dict[str, list[dict[str, str]]]:
    requested_families = amdgpu_families
    if not requested_families:
        return {"include": []}
    if not package_index_url:
        raise ValueError("--package-index-url is required when tests are enabled")

    include: list[dict[str, str]] = []
    seen_families: set[str] = set()
    for requested_family in requested_families:
        family, platform_info = find_amdgpu_family(
            requested_family=requested_family,
            platform=platform,
        )
        if family in seen_families:
            continue
        seen_families.add(family)
        test_runs_on = platform_info.get("test-runs-on")
        if not test_runs_on:
            print(f"Skipping {family}: no {platform} test runner is configured")
            continue
        if not isinstance(test_runs_on, str):
            raise ValueError(f"{family}: test-runs-on must be a string")
        include.append(
            PyTorchTestMatrixEntry(
                amdgpu_family=family,
                test_runs_on=test_runs_on,
                package_index_url=package_index_url,
            ).to_dict()
        )

    return {"include": include}


def resolve_requested_test_families(
    *, build_amdgpu_families: str, test_amdgpu_families: str
) -> tuple[list[str], list[str]]:
    """Return build families and explicit test families.

    Empty/auto/built test input means "test all built families". Explicit
    none/skip disables quick tests.
    """
    build_families = split_families(build_amdgpu_families)
    test_value = test_amdgpu_families.strip()
    test_value_lower = test_value.lower()
    if test_value_lower in SKIP_TEST_VALUES:
        return build_families, []
    if test_value_lower in AUTO_TEST_VALUES:
        return build_families, build_families
    return build_families, split_families(test_amdgpu_families)


def emit_outputs(matrix: dict[str, list[dict[str, str]]]) -> None:
    include = matrix["include"]
    gha_set_output(
        {
            "enabled": str(bool(include)).lower(),
            "matrix": json.dumps(matrix),
        }
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-amdgpu-families",
        required=True,
        help="Semicolon- or space-separated AMDGPU families that were built.",
    )
    parser.add_argument(
        "--test-amdgpu-families",
        default="",
        help=(
            "Semicolon- or space-separated AMDGPU families to quick-test. "
            "Use 'auto' or leave empty to test built families. Use 'none' "
            "to skip quick tests."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=["linux"],
        default="linux",
        help="Test platform.",
    )
    parser.add_argument(
        "--package-index-url",
        default="",
        help="Package index URL for tests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    _build_amdgpu_families, test_amdgpu_families = resolve_requested_test_families(
        build_amdgpu_families=args.build_amdgpu_families,
        test_amdgpu_families=args.test_amdgpu_families,
    )
    matrix = build_test_matrix(
        amdgpu_families=test_amdgpu_families,
        platform=args.platform,
        package_index_url=args.package_index_url,
    )
    emit_outputs(matrix)


if __name__ == "__main__":
    main(sys.argv[1:])
