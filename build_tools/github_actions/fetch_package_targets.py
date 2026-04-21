# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This file helps generate a package target matrix for workflows.

Environment variable inputs:
    * 'AMDGPU_FAMILIES': A comma separated list of AMD GPU families, e.g.
                    `gfx94X,gfx103x`, or empty for the default list
    * 'THEROCK_PACKAGE_PLATFORM': "linux" or "windows"

Outputs written to GITHUB_OUTPUT:
    * 'package_targets': JSON list of the form
        [
            {
                "amdgpu_family": "gfx94X-dcgpu",
                "test_machine": "linux-mi300-1gpu-ossci-rocm",
                "expect_failure": false,
                "expect_pytorch_failure": false
            },
            {
                "amdgpu_family": "gfx110X-all",
                "test_machine": "",
                "expect_failure": false,
                "expect_pytorch_failure": true
            }
        ]

Example usage:

```yml
jobs:
  setup_metadata:
    runs-on: ubuntu-24.04
    outputs:
      package_targets: ${{ steps.configure.outputs.package_targets }}

    steps:
      - name: Generating package target matrix
        id: configure
        env:
          AMDGPU_FAMILIES: ${{ inputs.families }}
          THEROCK_PACKAGE_PLATFORM: "windows"
        run: python ./build_tools/github_actions/fetch_package_targets.py

  windows_packages:
    name: ${{ matrix.target_bundle.amdgpu_family }}::Build Windows
    runs-on: 'windows-2022'
    needs: [setup_metadata]
    strategy:
      matrix:
        target_bundle: ${{ fromJSON(needs.setup_metadata.outputs.package_targets) }}
```
"""

import os
import json
import random
from amdgpu_family_matrix import (
    get_all_families_for_trigger_types,
)
import string

from github_actions_api import *


def _select_weighted_label(labels_config: list[dict], context_name: str) -> str:
    """Select a runner label based on weighted random selection.

    Args:
        labels_config: List of dicts with "label" and "weight" keys.
                       Weights should sum to 1.0.
        context_name: Name for logging context (e.g. family name).

    Returns:
        Selected label string.
    """
    rand_val = random.random()
    cumulative = 0.0
    for config in labels_config:
        cumulative += config["weight"]
        if rand_val < cumulative:
            print(
                f"  {context_name}: selected runner (weight={config['weight']}): "
                f"{config['label']}"
            )
            return config["label"]
    # Fallback to last label if rounding errors
    selected = labels_config[-1]
    print(
        f"  {context_name}: selected runner (weight={selected['weight']}): "
        f"{selected['label']}"
    )
    return selected["label"]


def determine_package_targets(args):
    amdgpu_families = args.get("AMDGPU_FAMILIES")
    package_platform = args.get("THEROCK_PACKAGE_PLATFORM")
    test_harness_target_fetch = args.get("TEST_HARNESS_TARGET_FETCH", False)

    # Use trigger-specific matrix lookup with presubmit priority.
    # When a family appears in multiple trigger types (e.g., gfx110x in both presubmit and nightly),
    # the presubmit configuration takes priority. This ensures consistent behavior across all
    # packaging workflows.
    matrix = get_all_families_for_trigger_types(["presubmit", "postsubmit", "nightly"])
    family_matrix = matrix
    package_targets = []
    # If the workflow does specify AMD GPU family, package those. Otherwise, then package all families
    if amdgpu_families:
        # Sanitizing the string to remove any punctuation from the input
        # After replacing punctuation with spaces, turning string input to an array
        # (ex: ",gfx94X ,|.gfx1201" -> "gfx94X   gfx1201" -> ["gfx94X", "gfx1201"])
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        family_matrix = [
            item.lower() for item in amdgpu_families.translate(translator).split()
        ]

    for key in family_matrix:
        info_for_key = matrix.get(key)

        # In case an invalid target is requested and returns null, we continue to the next target
        if not info_for_key:
            continue

        platform_for_key = info_for_key.get(package_platform)

        if not platform_for_key:
            # Some AMDGPU families are only supported on certain platforms.
            continue

        family = platform_for_key.get("family")
        test_machine = platform_for_key.get("test-runs-on")

        # Handle multi-label configuration with weighted random selection.
        if "test-runs-on-labels" in platform_for_key:
            test_machine = _select_weighted_label(
                platform_for_key["test-runs-on-labels"], family
            )

        sanity_check_only_for_family = platform_for_key.get(
            "sanity_check_only_for_family", False
        )

        # Due to the long test times for the test harness, we only want to use highly available test machines.
        # TODO(#1920): Remove this logic and use direct communication with test machines (instead of using GH runners)
        if (test_harness_target_fetch and not test_machine) or (
            test_harness_target_fetch and sanity_check_only_for_family
        ):
            continue

        expect_failure = platform_for_key.get("expect_failure", False)
        expect_pytorch_failure = platform_for_key.get("expect_pytorch_failure", False)

        package_targets.append(
            {
                "amdgpu_family": family,
                "test_machine": test_machine,
                "expect_failure": expect_failure,
                "expect_pytorch_failure": expect_pytorch_failure,
            }
        )

    return package_targets


def main(args):
    package_targets = determine_package_targets(args)
    gha_set_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    args["THEROCK_PACKAGE_PLATFORM"] = os.getenv("THEROCK_PACKAGE_PLATFORM")
    args["TEST_HARNESS_TARGET_FETCH"] = str2bool(os.getenv("TEST_HARNESS_TARGET_FETCH"))
    main(args)
