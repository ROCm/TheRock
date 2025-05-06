import os
import json
from configure_ci import set_github_output
from amdgpu_family_matrix import amdgpu_family_info_matrix

# This file helps determine if an amdgpu_families is experimental or not

def main(args):
    amdgpu_families = args.get("AMDGPU_FAMILIES").lower()
    platform = args.get("PLATFORM")
    family_matrix = amdgpu_family_info_matrix

    for key in family_matrix:
        if key in amdgpu_families:
            amdgpu_family_object = family_matrix.get(key).get(platform)
            if amdgpu_family_object and amdgpu_family_object.get("experimental"):
                set_github_output({"experimental": json.dumps(True)})
                return

    set_github_output({"experimental": json.dumps(False)})

if __name__ == "__main__":
    args = {}
    args["AMDGPU_FAMILIES"] = os.getenv("AMDGPU_FAMILIES")
    args["PLATFORM"] = os.getenv("PLATFORM")
    main(args)
