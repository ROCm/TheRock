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


def _find_platform_info(*, amdgpu_family: str, platform: str) -> dict[str, object]:
    matrix = get_all_families_for_trigger_types(["presubmit", "postsubmit", "nightly"])
    for info_for_key in matrix.values():
        platform_info = info_for_key.get(platform)
        if not platform_info:
            continue

        family = platform_info["family"]
        if amdgpu_family.lower() == family.lower():
            return platform_info

    raise ValueError(f"No {platform} AMDGPU family entry found for {amdgpu_family!r}")


def find_test_runs_on(*, amdgpu_family: str, platform: str) -> str:
    platform_info = _find_platform_info(amdgpu_family=amdgpu_family, platform=platform)
    return platform_info["test-runs-on"]


def find_multi_gpu_runs_on(*, amdgpu_family: str, platform: str) -> str:
    """Return the multi-GPU runner label for a family, or "" if none exists.

    Only data-center families (e.g. gfx942, gfx950) currently have multi-GPU
    runner pools; consumer families have single-GPU runners only.
    """
    platform_info = _find_platform_info(amdgpu_family=amdgpu_family, platform=platform)
    return str(platform_info.get("test-runs-on-multi-gpu") or "")


def select_test_configs(*, default_test_configs: str, has_multi_gpu: bool) -> str:
    """Drop multi-GPU-only configs when a family has no multi-GPU runner.

    `distributed` requires more than one GPU, so families without a multi-GPU
    runner can only run the single-GPU configs.
    """
    configs = [c for c in default_test_configs.split() if c]
    if not has_multi_gpu:
        configs = [c for c in configs if c != "distributed"]
    return " ".join(configs)


def build_test_matrix(
    *,
    amdgpu_families: list[str],
    platform: str,
    include_multi_gpu: bool = False,
    default_test_configs: str = "",
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

        entry: dict[str, str] = {
            "amdgpu_family": requested_family,
            "test_runs_on": test_runs_on,
        }

        if include_multi_gpu:
            multi_gpu_runs_on = find_multi_gpu_runs_on(
                amdgpu_family=requested_family,
                platform=platform,
            )
            entry["test_runs_on_multi_gpu"] = multi_gpu_runs_on
            entry["test_configs"] = select_test_configs(
                default_test_configs=default_test_configs,
                has_multi_gpu=bool(multi_gpu_runs_on),
            )

        print(
            f"Including {requested_family}: testing on {test_runs_on}"
            + (
                f" (multi-GPU: {entry['test_runs_on_multi_gpu'] or 'none'}, "
                f"configs: {entry['test_configs'] or 'none'})"
                if include_multi_gpu
                else ""
            )
        )
        include.append(entry)

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
    parser.add_argument(
        "--include-multi-gpu",
        action="store_true",
        help=(
            "Also emit 'test_runs_on_multi_gpu' and 'test_configs' per family. "
            "Families without a multi-GPU runner drop multi-GPU-only configs "
            "(e.g. 'distributed')."
        ),
    )
    parser.add_argument(
        "--default-test-configs",
        default="",
        help=(
            "Space-separated test configs to request when --include-multi-gpu "
            "is set (e.g. 'default distributed inductor')."
        ),
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
        include_multi_gpu=args.include_multi_gpu,
        default_test_configs=args.default_test_configs,
    )
    emit_outputs(matrix)


if __name__ == "__main__":
    main(sys.argv[1:])
