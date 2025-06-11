import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import amdgpu_family_info_matrix
import string

# This file helps generate a package target matrix for portable_linux_package_matrix.yml and publish_pytorch_dev_docker.yml


def main(args):
    amdgpu_families = args.get("AMDGPU_FAMILIES")
    pytorch_dev_docker = args.get("PYTORCH_DEV_DOCKER") == "true"
    package_platform = args.get("THEROCK_PACKAGE_PLATFORM")

    family_matrix = amdgpu_family_info_matrix
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
        info_for_key = amdgpu_family_info_matrix.get(key)
        if pytorch_dev_docker:
            # If there is not a target specified for the family
            if not "pytorch-target" in info_for_key.get(package_platform):
                continue
            family = info_for_key.get(package_platform).get("pytorch-target")
        else:
            family = info_for_key.get(package_platform).get("family")

        package_targets.append({"amdgpu_family": family})

    set_github_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    args["PYTORCH_DEV_DOCKER"] = os.getenv("PYTORCH_DEV_DOCKER")
    args["THEROCK_PACKAGE_PLATFORM"] = os.getenv("THEROCK_PACKAGE_PLATFORM")
    main(args)
