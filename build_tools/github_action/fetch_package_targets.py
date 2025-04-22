import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import amdgpu_family_info_matrix

# This file helps generate a package target matrix for portable_linux_package_matrix.yml and publish_pytorch_dev_docker.yml


def main(args):
    pytorch_dev_docker = args.get("PYTORCH_DEV_DOCKER") == "true"
    package_targets = []
    for key in amdgpu_family_info_matrix:
        if pytorch_dev_docker:
            # If there is not a target specified for the family
            if not "pytorch-target" in amdgpu_family_info_matrix.get(key).get("linux"):
                continue
            family = (
                amdgpu_family_info_matrix.get(key).get("linux").get("pytorch-target")
            )
        else:
            family = amdgpu_family_info_matrix.get(key).get("linux").get("family")

        package_targets.append({"amdgpu_family": family})

    set_github_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    args = {}
    args["PYTORCH_DEV_DOCKER"] = os.getenv("PYTORCH_DEV_DOCKER")
    main(args)
