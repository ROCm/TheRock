#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure PyTorch wheel test jobs for multi-arch build workflows.

TODO(#5110): Extract the AMDGPU family -> test runner policy once JAX needs
the same flow. Standalone workflow_dispatch runs can keep using raw family
inputs, while coordinated CI/release runs should be able to pass the
per-family test policy produced by configure_multi_arch_ci.py. That shared
policy should also support named defaults such as "release" and "presubmit",
plus explicit opt-in/opt-out family lists.
"""

import argparse
import json
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.amdgpu_family_matrix import get_all_families_for_trigger_types
from github_actions.github_actions_api import gha_set_output


def split_families(value: str) -> list[str]:
    return list(dict.fromkeys(f.strip() for f in value.split(";") if f.strip()))


def find_test_runs_on(*, amdgpu_family: str, platform: str) -> str:
    matrix = get_all_families_for_trigger_types(["presubmit", "postsubmit", "nightly"])
    for info_for_key in matrix.values():
        platform_info = info_for_key.get(platform)
        if not platform_info:
            continue

        family = platform_info["family"]
        if amdgpu_family.lower() == family.lower():
            return platform_info["test-runs-on"]

    raise ValueError(f"No {platform} AMDGPU family entry found for {amdgpu_family!r}")


def build_test_matrix(
    *,
    amdgpu_families: list[str],
    platform: str,
) -> dict[str, list[dict[str, str]]]:
    print(f"Requested {platform} AMDGPU families: {amdgpu_families}")
    include: list[dict[str, str]] = []
    for requested_family in amdgpu_families:
        test_runs_on = find_test_runs_on(
            amdgpu_family=requested_family,
            platform=platform,
        )

        if not test_runs_on:
            print(
                f"Skipping {requested_family}: no {platform} test runner is configured"
            )
            continue

        print(f"Including {requested_family}: testing on {test_runs_on}")
        include.append(
            {
                "amdgpu_family": requested_family,
                "test_runs_on": test_runs_on,
            }
        )

    return {"include": include}


def emit_outputs(matrix: dict[str, list[dict[str, str]]]) -> None:
    gha_set_output(
        {
            "enabled": str(bool(matrix["include"])).lower(),
            "matrix": json.dumps(matrix, separators=(",", ":")),
        }
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-amdgpu-families",
        required=True,
        help="Semicolon-separated AMDGPU families that were built.",
    )
    parser.add_argument(
        "--test-amdgpu-families",
        default="",
        help=(
            "Semicolon-separated AMDGPU families to test. Use 'auto' or leave "
            "empty to test built families. Use 'none' to skip tests."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default=platform_module.system().lower(),
        help="Test platform (default: current system).",
    )
    args = parser.parse_args(argv)

    built_families = split_families(args.build_amdgpu_families)
    test_families_arg = args.test_amdgpu_families.strip().lower()
    if test_families_arg in ("", "auto", "built"):
        test_amdgpu_families = built_families
    elif test_families_arg in ("none", "skip"):
        test_amdgpu_families = []
    else:
        test_amdgpu_families = split_families(args.test_amdgpu_families)
        for test_family in test_amdgpu_families:
            if test_family.lower() in ("auto", "built", "none", "skip"):
                raise ValueError(
                    f"Test family control value {test_family!r} cannot be mixed "
                    "with explicit AMDGPU families"
                )

    matrix = build_test_matrix(
        amdgpu_families=test_amdgpu_families,
        platform=args.platform,
    )
    emit_outputs(matrix)


if __name__ == "__main__":
    main(sys.argv[1:])
