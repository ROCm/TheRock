import os
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
)

from github_actions_utils import *

# This file helps configure which target to run

# TODO (geomin12): this is very hard-coded to a very specific use-case.
# Once portable_linux_package_matrix.yml matures, this will mature as well
# Some logic is duplicated with fetch_package_targets.py


def main(target: str, platform: str):
    amdgpu_family_info_matrix = (
        amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit
    )
    for key, info_for_key in amdgpu_family_info_matrix.items():
        # Only consider items containing the amdgpu_family (ex: gfx94X in gfx94X-dcgpu)
        if key not in target.lower():
            continue

        platform_for_key = info_for_key.get(platform)

        if not platform_for_key:
            # Some AMDGPU families are only supported on certain platforms.
            continue

        # If there is a test machine available for this target, run on it.
        test_runs_on_machine = platform_for_key.get("test-runs-on")
        if test_runs_on_machine:
            gha_set_output({"test-runs-on": test_runs_on_machine})


if __name__ == "__main__":
    target = os.getenv("TARGET", "")
    platform = os.getenv("PLATFORM", "")
    main(target=target, platform=platform)
