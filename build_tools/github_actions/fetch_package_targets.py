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
                "test_machine": "linux-mi300-1gpu-ossci-rocm"
            },
            {
                "amdgpu_family": "gfx110X-dgpu",
                "test_machine": ""
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
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
)
import string

from github_actions_utils import *


def determine_package_targets(args):
    amdgpu_families = args.get("AMDGPU_FAMILIES")
    package_platform = args.get("THEROCK_PACKAGE_PLATFORM")

    matrix = amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit
    family_matrix = (
        amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit
    )
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

        package_targets.append({"amdgpu_family": family, "test_machine": test_machine})

    return package_targets


def main(args):
    package_targets = determine_package_targets(args)
    gha_set_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    args["THEROCK_PACKAGE_PLATFORM"] = os.getenv("THEROCK_PACKAGE_PLATFORM")
    main(args)
