"""This file looks up the test-runs-on runner label for a given GPU family.

Environment variable inputs:
    * 'TARGET': A GPU family like 'gfx95X-dcgpu' or 'gfx1151', corresponding
                to a release index.
    * 'PLATFORM': "linux" or "windows"
"""

import os
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
)

from github_actions_utils import *


def get_runner_label(target: str, platform: str) -> str:
    print(f"Searching for a runner for target '{target}' on platform '{platform}'")
    amdgpu_family_info_matrix = (
        amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit
    )
    for key, info_for_key in amdgpu_family_info_matrix.items():
        print(f"Cheecking key '{key}' with info:\n  {info_for_key}")
        platform_for_key = info_for_key.get(platform)

        if not platform_for_key:
            # Some AMDGPU families are only supported on certain platforms.
            print(f"  Skipping since this entry has no platform '{platform}'")
            continue

        # Check against both the inner "family" and the outer "key". If neither
        # match then skip. Workflows are expected to use the inner "family"
        # but manually triggered runs may use the outer "key" instead, so we'll
        # be a bit lenient here.
        # This needs a rework, see https://github.com/ROCm/TheRock/issues/1097.
        family_for_platform = platform_for_key.get("family")
        if target != family_for_platform and key not in target.lower():
            print(
                f"  Skipping since the target '{target}' does not match the family '{family_for_platform}'"
            )
            continue

        # If there is a test machine available for this target, run on it.
        test_runs_on_machine = platform_for_key.get("test-runs-on")
        if test_runs_on_machine:
            print(f"  Found runner: '{test_runs_on_machine}'")
            return test_runs_on_machine
    return ""


def get_upload_label(target: str, platform: str) -> str:
    print(f"Searching for a runner for target '{target}' on platform '{platform}'")
    amdgpu_family_info_matrix = (
        amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit
    )
    for key, info_for_key in amdgpu_family_info_matrix.items():
        print(f"Cheecking key '{key}' with info:\n  {info_for_key}")
        platform_for_key = info_for_key.get(platform)

        if not platform_for_key:
            # Some AMDGPU families are only supported on certain platforms.
            print(f"  Skipping since this entry has no platform '{platform}'")
            continue

        # Check against both the inner "family" and the outer "key". If neither
        # match then skip. Workflows are expected to use the inner "family"
        # but manually triggered runs may use the outer "key" instead, so we'll
        # be a bit lenient here.
        # This needs a rework, see https://github.com/ROCm/TheRock/issues/1097.
        family_for_platform = platform_for_key.get("family")
        if target != family_for_platform and key not in target.lower():
            print(
                f"  Skipping since the target '{target}' does not match the family '{family_for_platform}'"
            )
            continue

        # If there is no test machine available and bypass_tests_for_releases flag is True for GPU family and platform, output bypass_tests_for_releases as True
        bypass_tests_for_releases = platform_for_key.get("bypass_tests_for_releases")
        if bypass_tests_for_releases:
            print(f"  bypass_tests_for_releases: True")
            return bypass_tests_for_releases
    return ""


def main(target: str, platform: str):
    runner_label = get_runner_label(target, platform)
    if runner_label:
        gha_set_output({"test-runs-on": runner_label})
    upload_label = get_upload_label(target, platform)
    if upload_label:
        gha_set_output({"bypass_tests_for_releases": upload_label})


if __name__ == "__main__":
    target = os.getenv("TARGET", "")
    platform = os.getenv("PLATFORM", "")
    main(target=target, platform=platform)
