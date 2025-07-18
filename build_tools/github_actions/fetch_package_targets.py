import os
import json
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
)
import string

from github_actions_utils import *

# This file helps generate a package target matrix for workflows.


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
