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
from github_actions.github_actions_api import gha_append_step_summary, gha_set_output
from configure_pytorch_release_matrix import PYTORCH_TEST_LEVELS


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
    test_level: str,
) -> dict[str, list[dict[str, str]]]:
    if test_level not in PYTORCH_TEST_LEVELS:
        raise ValueError(f"Unknown PyTorch test level: {test_level!r}")
    if test_level == "none":
        print(
            "Test level 'none': no self-hosted GPU test jobs will be scheduled; "
            "the build job still runs its build-time wheel validation"
        )
        return {"include": []}

    print(f"Resolved {platform} GPU test families: {amdgpu_families or 'none'}")
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


def format_test_summary(
    *,
    platform: str,
    test_level: str,
    built_families: list[str],
    requested_test_families: str,
    resolved_test_families: list[str],
    matrix: dict[str, list[dict[str, str]]],
) -> str:
    """Format the resolved test policy for logs and the job summary."""

    def format_families(families: list[str]) -> str:
        return ", ".join(f"`{family}`" for family in families) or "none"

    include = matrix["include"]
    lines = [
        "## PyTorch Test Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Platform | `{platform}` |",
        f"| Test level | `{test_level}` |",
        f"| Built AMDGPU families | {format_families(built_families)} |",
        f"| Requested test families | `{requested_test_families}` |",
        f"| Resolved test families | {format_families(resolved_test_families)} |",
        f"| Self-hosted GPU test jobs | {len(include)} |",
        "",
    ]

    if test_level == "none":
        lines.append(
            "**Decision:** No self-hosted GPU test jobs are scheduled because "
            "the test level is `none`."
        )
    elif not resolved_test_families:
        lines.append(
            "**Decision:** No self-hosted GPU test jobs are scheduled because "
            "the test-family selection resolved to none."
        )
    elif not include:
        lines.append(
            "**Decision:** No self-hosted GPU test jobs are scheduled because "
            f"none of the selected families has a configured {platform} runner."
        )
    else:
        lines.extend(
            [
                "**Decision:** Schedule the following self-hosted GPU tests:",
                "",
                "| AMDGPU family | Runner |",
                "| --- | --- |",
                *[
                    f"| `{row['amdgpu_family']}` | `{row['test_runs_on']}` |"
                    for row in include
                ],
            ]
        )

    lines.extend(
        [
            "",
            "The build job always runs `sanity_check_wheel.py` as build-time "
            "artifact validation before any GPU tests.",
        ]
    )
    if test_level == "full":
        lines.append(
            "The build workflow dispatches the additional full suite "
            "separately from this standard GPU test matrix."
        )
    return "\n".join(lines)


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
        "--test-level",
        choices=PYTORCH_TEST_LEVELS,
        default="standard",
        help=(
            "Test coverage level. 'none' stops after build-time wheel "
            "validation; 'standard' and 'full' run the standard GPU matrix."
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
        test_level=args.test_level,
    )
    gha_append_step_summary(
        format_test_summary(
            platform=args.platform,
            test_level=args.test_level,
            built_families=built_families,
            requested_test_families=args.test_amdgpu_families.strip() or "auto",
            resolved_test_families=test_amdgpu_families,
            matrix=matrix,
        )
    )
    emit_outputs(matrix)


if __name__ == "__main__":
    main(sys.argv[1:])
