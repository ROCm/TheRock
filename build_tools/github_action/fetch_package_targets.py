import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import amdgpu_family_info_matrix
import string

# This file helps generate a package target matrix for portable_linux_package_matrix.yml and publish_pytorch_dev_docker.yml


def main(args):
    pytorch_dev_docker = args.get("PYTORCH_DEV_DOCKER") == "true"
    amdgpu_families = args.get("AMDGPU_FAMILIES")
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
        if pytorch_dev_docker:
            # If there is not a target specified for the family
            if not "pytorch-target" in amdgpu_family_info_matrix.get(key).get("linux"):
                continue
            family = (
                amdgpu_family_info_matrix.get(key).get("linux").get("pytorch-target")
            )
        else:
            family = amdgpu_family_info_matrix.get(key).get("linux").get("family")

        experimental_amdgpu_family = (
            amdgpu_family_info_matrix.get(key).get("linux").get("experimental")
        )
        if experimental_amdgpu_family:
            experimental = True
        else:
            experimental = False

        package_targets.append(
            {"amdgpu_family": family, "experimental": json.dumps(experimental)}
        )

    set_github_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["PYTORCH_DEV_DOCKER"] = os.getenv("PYTORCH_DEV_DOCKER")
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    main(args)
