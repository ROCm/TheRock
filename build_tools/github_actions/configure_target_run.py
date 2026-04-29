# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""This file looks up the test-runs-on runner label for a given GPU family.

Environment variable inputs:
    * 'TARGET': A GPU family like 'gfx95X-dcgpu' or 'gfx1151', corresponding
                to a release index.
    * 'PLATFORM': "linux" or "windows"

Command-line:
    * `--test-project-name`: When set to `pytorch`, use `pytorch-ci-test-runs-on`
      instead of `test-runs-on` label. Workflows need specific runners for
      PyTorch testing should pass this explicitly.
"""

import argparse
import os
from amdgpu_family_matrix import get_all_families_for_trigger_types

from github_actions_api import *

test_project_runs_on_label = {
    "pytorch": "pytorch-ci-test-runs-on",
}


def validate_test_project_name(project_name: str) -> str:
    """Validate the test project name.

    Empty input returns ``""`` (use default ``test-runs-on`` in the matrix).
    Unknown names raise ``argparse.ArgumentTypeError``.
    """
    if not project_name:
        return ""

    if project_name in test_project_runs_on_label:
        return project_name

    raise argparse.ArgumentTypeError(
        f"Project '{project_name}' does not have a dedicated test runner label."
    )


def get_runner_label(target: str, platform: str, *, test_project_name: str = "") -> str:
    print(f"Searching for a runner for target '{target}' on platform '{platform}'")
    if test_project_name:
        print(f"Using test project name: '{test_project_name}'")
    amdgpu_family_info_matrix = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit"]
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

        # Optional per-project matrix key (e.g. pytorch-ci-test-runs-on); missing
        # or empty dedicated label falls back to test-runs-on.
        if test_project_name:
            test_runs_on_machine = platform_for_key.get(
                test_project_runs_on_label[test_project_name]
            ) or platform_for_key.get("test-runs-on")
        else:
            test_runs_on_machine = platform_for_key.get("test-runs-on")

        if test_runs_on_machine:
            print(f"  Found runner: '{test_runs_on_machine}'")
            return test_runs_on_machine
    return ""


def get_upload_label(target: str, platform: str) -> str:
    print(f"Searching for a runner for target '{target}' on platform '{platform}'")
    amdgpu_family_info_matrix = get_all_families_for_trigger_types(
        ["presubmit", "postsubmit"]
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


def main(target: str, platform: str, *, test_project_name: str = ""):
    runner_label = get_runner_label(
        target, platform, test_project_name=test_project_name
    )
    if runner_label:
        gha_set_output({"test-runs-on": runner_label})
    upload_label = get_upload_label(target, platform)
    if upload_label:
        gha_set_output({"bypass_tests_for_releases": upload_label})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-project-name",
        default="",
        type=validate_test_project_name,
        help=(
            "Request project specific test runner label. e.g. 'pytorch' for `pytorch-ci-test-runs-on` label."
        ),
    )
    args = parser.parse_args()
    target = os.getenv("TARGET", "")
    platform = os.getenv("PLATFORM", "")
    main(
        target=target,
        platform=platform,
        test_project_name=args.test_project_name,
    )
