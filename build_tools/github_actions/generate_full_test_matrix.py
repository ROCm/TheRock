#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate a GitHub Actions matrix of GPU families for full PyTorch UT runs.

Reads amdgpu_family_matrix.py across all trigger types (presubmit,
postsubmit, nightly) and selects families that have:
  - A non-empty test-runs-on runner label for the requested platform
  - The run-full-tests-only flag set to True

Outputs a ``matrix`` variable via $GITHUB_OUTPUT suitable for
``fromJSON()`` in a workflow strategy block.

Usage (in a workflow step)::

    python build_tools/github_actions/generate_full_test_matrix.py \\
        --platform linux

Example output (written to $GITHUB_OUTPUT as ``matrix=<json>``)::

    {"include":[
      {"amdgpu_family":"gfx94X-dcgpu","test_runs_on":"linux-gfx942-1gpu-ossci-rocm",
       "test_runs_on_multi_gpu":"linux-gfx942-8gpu-ossci-rocm"},
      {"amdgpu_family":"gfx950-dcgpu","test_runs_on":"linux-mi355-1gpu-ossci-rocm",
       "test_runs_on_multi_gpu":""}
    ]}
"""

import argparse
import json
import os
import sys

from amdgpu_family_matrix import get_all_families_for_trigger_types


def generate_matrix(platform: str) -> dict:
    all_families = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit", "nightly"]
    )

    entries = []
    for _key, info in all_families.items():
        platform_info = info.get(platform)
        if not platform_info:
            continue

        runner = platform_info.get("test-runs-on", "")
        if not runner:
            continue

        if not platform_info.get("run-full-tests-only"):
            continue

        family = platform_info.get("family", "")
        multi_gpu = platform_info.get("test-runs-on-multi-gpu", "")

        entries.append(
            {
                "amdgpu_family": family,
                "test_runs_on": runner,
                "test_runs_on_multi_gpu": multi_gpu,
            }
        )

    return {"include": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        default=os.getenv("PLATFORM", "linux"),
        help="Platform to filter by (linux or windows). Default: $PLATFORM or linux.",
    )
    args = parser.parse_args()

    matrix = generate_matrix(args.platform)
    matrix_json = json.dumps(matrix, separators=(",", ":"))

    print(f"Generated matrix ({len(matrix['include'])} entries):")
    print(json.dumps(matrix, indent=2))

    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"matrix={matrix_json}\n")
    else:
        print(f"\nmatrix={matrix_json}")

    if not matrix["include"]:
        print(
            f"WARNING: No families with run-full-tests-only=True and a runner "
            f"found for platform '{args.platform}'",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
