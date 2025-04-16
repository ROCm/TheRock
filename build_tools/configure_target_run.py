import os
import json
from configure_ci import set_github_output, amdgpu_family_info_matrix

# This file helps configure which target to run

# TODO (geomin12): this is very hard-coded to a very specific use-case.
# Once portable_linux_package_matrix.yml matures, this will mature as well

def main(args):
    target = args.get("target")
    for key in amdgpu_family_info_matrix.keys():
        # If the amdgpu_family matrix key is inside the target (ex: gfx94X in gfx94X-dcgpu)
        if key in target:
            test_runs_on_machine = amdgpu_family_info_matrix.get(key).get("linux").get("test-runs-on")
            # if there is a test machine available for this target
            if test_runs_on_machine:
                set_github_output({"test-runs-on": test_runs_on_machine})


if __name__ == "__main__":
    args = {}
    args["target"] = os.environ.get("TARGET", "")
    main(args)
