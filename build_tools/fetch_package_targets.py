import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import amdgpu_family_info_matrix

# This file helps generate a package target matrix for portable_linux_package_matrix.yml and publish_pytorch_dev_docker.yml


def main():
    package_targets = []
    for key in amdgpu_family_info_matrix:
        target = amdgpu_family_info_matrix.get(key).get("linux").get("target")
        package_targets.append({"amdgpu_family": target})

    set_github_output({"package_targets": json.dumps(package_targets)})


if __name__ == "__main__":
    main()
