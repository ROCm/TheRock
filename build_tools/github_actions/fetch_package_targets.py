import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
)
import string

# This file helps generate a package target matrix for portable_linux_package_matrix.yml and publish_pytorch_dev_docker.yml


def determine_package_targets(args):
    amdgpu_families = args.get("AMDGPU_FAMILIES")
    pytorch_dev_docker = args.get("PYTORCH_DEV_DOCKER") == "true"
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

        if pytorch_dev_docker:
            if not "pytorch-target" in platform_for_key:
                # Some AMDGPU families may not have an associated PyTorch targets.
                continue
            family = platform_for_key.get("pytorch-target")
        else:
            family = platform_for_key.get("family")

        package_targets.append({"amdgpu_family": family})

    return package_targets


def main(args):
    package_targets = determine_package_targets(args)
    set_github_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    args["PYTORCH_DEV_DOCKER"] = os.getenv("PYTORCH_DEV_DOCKER")
    args["THEROCK_PACKAGE_PLATFORM"] = os.getenv("THEROCK_PACKAGE_PLATFORM")
    main(args)
