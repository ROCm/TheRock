#!/usr/bin/env python3

"""Configures metadata for a CI workflow run.

----------
| Inputs |
----------

  Environment variables (for all triggers):
  * GITHUB_EVENT_NAME    : GitHub event name, e.g. pull_request.
  * GITHUB_OUTPUT        : path to write workflow output variables.
  * GITHUB_STEP_SUMMARY  : path to write workflow summary output.
  * INPUT_LINUX_AMDGPU_FAMILIES (optional): Comma-separated string of Linux AMD GPU families
  * LINUX_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Linux tests
  * INPUT_WINDOWS_AMDGPU_FAMILIES (optional): Comma-separated string of Windows AMD GPU families
  * WINDOWS_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Windows tests
  * BRANCH_NAME (optional): The branch name
  * INPUT_TASKS_FORCE_BUILD (optional)  : Always return build targets,
                                          independent of if files were changed or not
  * INPUT_TASKS_NO_TESTS (optional)     : Never return test targets
  * INPUT_TASKS_MAKE_RELEASE (optional) : Return release targets

  Environment variables (for pull requests):
  * PR_LABELS (optional) : JSON list of PR label names.
  * BASE_REF  (required) : base commit SHA of the PR.

  Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

  Written to GITHUB_OUTPUT:
  * linux_amdgpu_families : List of valid Linux AMD GPU families to execute build and test jobs
  * windows_amdgpu_families : List of valid Windows AMD GPU families to execute build and test jobs
  * enable_build_jobs: If true, builds will be enabled

  Written to GITHUB_STEP_SUMMARY:
  * Human-readable summary for most contributors

  Written to stdout/stderr:
  * Detailed information for CI maintainers
"""

import json
import os
import pprint
import sys
from enum import Flag, auto
from typing import Dict, List, Optional

from new_amdgpu_family_matrix_data import (
    all_build_variants,
    amdgpu_family_predefined_groups,
    amdgpu_family_info_matrix_all,
)
from new_amdgpu_family_matrix_types import BuildConfig, TestConfig
from configure_ci_path_filters import (
    get_git_modified_paths,
    is_ci_run_required,
)
from github_actions_utils import *

# --------------------------------------------------------------------------- #
# Matrix creation logic based on PR, push, or workflow_dispatch
# --------------------------------------------------------------------------- #


def get_pr_labels(github_event_args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(github_event_args["pr_labels"])
    labels = []
    if data:
        for label in data.get("labels", []):
            labels.append(label["name"])
    return labels


def print_github_info(github_event_args):
    workflow_label = {
        "pull_request": "[PULL_REQUEST]",
        "push": "[PUSH - MAIN]",
        "schedule": "[SCHEDULE]",
        "workflow_dispatch": "[WORKFLOW_DISPATCH]",
    }

    github_event_name = github_event_args.get("github_event_name")

    print(f"{workflow_label[github_event_name]} Generating build matrix with:")
    print(json.dumps(github_event_args, indent=4, sort_keys=True))


class TaskMask(Flag):
    BUILD = auto()
    TEST = auto()
    RELEASE = auto()

    @property
    def label(self) -> str:
        return self.name.lower()


class PlatformMask(Flag):
    LINUX = auto()
    WINDOWS = auto()

    @property
    def label(self) -> str:
        return self.name.lower()


def get_build_config(
    build_config: BuildConfig,
    build_variant: str,
    platform_str: str,
    arch: str,
    overwrite_values=None,
    enforce_overwrite=False,
) -> Optional[dict]:
    # We have custom build variants for specific CI flows.
    # For CI, we use the release build variant (for PRs, pushes to main, nightlies)
    # For CI ASAN, we use the ASAN build variant (for pushes to main)
    # In the case that the build variant is not requested, we skip it

    if build_variant not in build_config.build_variants:
        print(
            f"[WARNING] Build variant {build_variant} is not available for {arch} on {platform_str}"
        )
        return None

    variant_info = all_build_variants.get(platform_str, build_variant)
    if variant_info is None:
        print(f"[ERROR] Build variant {build_variant} does not exist on {platform_str}")
        return None

    config = variant_info.to_dict()

    # Assign a computed "artifact_group" combining the family and variant suffix.
    artifact_group = arch
    if variant_info.suffix:
        artifact_group += f"-{variant_info.suffix}"
    config["artifact_group"] = artifact_group

    # expect_failure: True if either the variant or the family entry sets it
    config["expect_failure"] = (
        variant_info.expect_failure or build_config.expect_failure
    )

    return config


def get_test_config(
    test_config: TestConfig,
    platform_str: str,
    target: str,
    overwrite_values=None,
    enforce_overwrite=False,
    orgwide_test_runner_dict={},
) -> Optional[dict]:
    """
    Layout overwrite_values:
    {
        "test_runner": "oem"
        "benchmark_runner": "some_benchmark_label"
    }

    The value (e.g. "oem") is looked up as a key in the entry's runs_on extra dict
    and used to replace the standard runner for that label ("test" or "benchmark").

    enforce_overwrite: If False, warn and continue if the value is not found in runs_on.
                       If True, return None (fail the config) if the value is not found.
    """
    if not test_config.run_tests:
        return None

    # Serialize runs_on to a mutable dict; includes extra keys for overwrite lookups.
    # Defaults for all fields are guaranteed by the TestConfig/GpuRunners dataclasses.
    runs_on = test_config.runs_on.to_dict()

    if overwrite_values:
        for key, value in overwrite_values.items():
            if "test_runner" in key or "benchmark_runner" in key:
                label = key.split("_")[0]
                if value in runs_on:
                    runs_on[label] = runs_on[value]
                else:
                    if not enforce_overwrite:
                        print(
                            f"[WARNING] Value {value} not found in test_config['runs_on'] for target {target} on {platform_str}"
                        )
                    else:
                        print(
                            f"[ERROR] Value {value} not found in test_config['runs_on'] for target {target} on {platform_str}. Skipping testing!"
                        )
                        return None

    # For external TheRock-CI:
    # As test runner names are frequently updated, we are pulling the runner label data from the ROCm organization variable
    # called "ROCM_THEROCK_TEST_RUNNERS"
    # For more info, go to 'docs/development/test_runner_info.md'
    for overwrite_arch in orgwide_test_runner_dict:
        # we need to do partial matching of the orgwide_test_runner_dict keys with the target
        if target.upper().startswith(overwrite_arch.upper()):
            runs_on["test"] = orgwide_test_runner_dict[overwrite_arch][platform_str]
            break

    # Build output with only standard runner keys; extra keys are for overwrite lookups only.
    output_runs_on = {
        "test": runs_on.get("test", ""),
        "test-multi-gpu": runs_on.get("test-multi-gpu", ""),
        "benchmark": runs_on.get("benchmark", ""),
    }

    # Sanity check: skip test job if no machine to run on
    if not output_runs_on["test"] and not output_runs_on["benchmark"]:
        return None

    result = test_config.to_dict()
    result["runs_on"] = output_runs_on
    return result


def matrix_generator(
    platform_mask: PlatformMask,
    task_mask: TaskMask,
    req_gpu_families_or_targets: Dict[PlatformMask, set],
    build_variant: str = "release",
    overwrite_values: Dict[str, str] = {},
    enforce_overwrite: bool = False,
    orgwide_test_runner_dict: Dict[str, str] = {},
) -> Dict[str, List[dict]]:
    # TODO TODO move to main() those checks
    if not bool(platform_mask):
        print("No platform set. Exiting")
        sys.exit(1)
    if not bool(task_mask):
        print("No task (build, test, release) set. Exiting")
        sys.exit(1)
    tasks = [mask.label for mask in task_mask]
    print(f"Creating AMDGPU matrix for tasks {tasks}...")

    # assign to platform, task, and build variant
    full_matrix = {}
    for platform in platform_mask:  # linux, windows
        platform_str = platform.label
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            req_gpu_families_or_targets[platform]
        )
        for key in result.unmatched_keys:
            print(
                f"ERROR! No entry for {key} on {platform_str} found - Skipping!",
                file=sys.stderr,
            )
        # select only entries matching the platform
        platform_entries = [
            (entry.key, entry.platform_config(platform_str))
            for entry in result.entries
            if entry.platform_config(platform_str) is not None
        ]
        print(f"    {platform_str}: {[target for target, _ in platform_entries]}")

        platform_matrix = []
        for target, platform_config in platform_entries:
            # target = gfx94X-dcgpu, gfx1151, ...
            data = {"amdgpu_family": target}

            if TaskMask.BUILD in task_mask:
                build_config = get_build_config(
                    platform_config.build,
                    build_variant,
                    platform_str,
                    target,
                    overwrite_values,
                    enforce_overwrite,
                )
                if build_config:
                    data[TaskMask.BUILD.label] = build_config
                else:
                    print(
                        f"[WARNING] Skipping build job for {target} on {platform_str} since build variant {build_variant} is not supported"
                    )
            if TaskMask.TEST in task_mask:
                test_config = get_test_config(
                    platform_config.test,
                    platform_str,
                    target,
                    overwrite_values,
                    enforce_overwrite,
                    orgwide_test_runner_dict,
                )
                if test_config:
                    data[TaskMask.TEST.label] = test_config
                else:
                    print(
                        f"[WARNING] Skipping test job for {target} on {platform_str}. Test config returned empty."
                    )
            if TaskMask.RELEASE in task_mask:
                # release_config always exists and needs no special treatement (yet)
                release_config = platform_config.release.to_dict()
                data[TaskMask.RELEASE.label] = release_config

            # more than amdgpu_family as entry? Means we want to run some tasks, so add it
            if len(data) > 1:
                platform_matrix += [data]
        full_matrix[platform_str] = platform_matrix

    print("done")
    return full_matrix


def get_github_event_args() -> dict:
    github_event_args = {}
    # Ensure pr_labels is a proper JSON string with a "labels" key (list of dicts)
    github_event_args["pr_labels"] = os.environ.get("PR_LABELS", "[]")

    # TODO TODO Remove after testing
    github_event_args["pr_labels"] = os.environ.get(
        "PR_LABELS", '{"labels":[{"name":"test_runner:oem"}]}'
    )
    github_event_args["branch_name"] = os.environ.get("GITHUB_REF_NAME", "")
    # TODO TODO Remove after testing
    github_event_args["branch_name"] = os.environ.get("GITHUB_REF_NAME", "main")
    if github_event_args["branch_name"] == "":
        print(
            "[ERROR] GITHUB_REF_NAME is not set! No branch name detected. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    # TODO TODO Remove after testing
    github_event_args["github_event_name"] = os.environ.get(
        "GITHUB_EVENT_NAME", "workflow_dispatch"
    )
    # github_event_args["github_event_name"] = os.environ.get("GITHUB_EVENT_NAME", "")
    github_event_args["base_ref"] = os.environ.get("BASE_REF", "HEAD^1")
    github_event_args["linux_use_prebuilt_artifacts"] = (
        os.environ.get("LINUX_USE_PREBUILT_ARTIFACTS") == "true"
    )
    github_event_args["windows_use_prebuilt_artifacts"] = (
        os.environ.get("WINDOWS_USE_PREBUILT_ARTIFACTS") == "true"
    )
    github_event_args["build_variant"] = os.getenv("BUILD_VARIANT", "release")
    github_event_args["multi_arch"] = os.environ.get("MULTI_ARCH", "false") == "true"

    github_event_args["force_build"] = os.environ.get("INPUT_TASKS_FORCE_BUILD", "")
    github_event_args["no_tests"] = os.environ.get("INPUT_TASKS_NO_TESTS", "NO")
    github_event_args["make_release"] = os.environ.get("INPUT_TASKS_MAKE_RELEASE", "")

    github_event_args["req_linux_amdgpus"] = os.environ.get(
        "INPUT_LINUX_AMDGPU_FAMILIES", ""
    )
    github_event_args["req_windows_amdgpus"] = os.environ.get(
        "INPUT_WINDOWS_AMDGPU_FAMILIES", ""
    )
    github_event_args["req_linux_amdgpus_predef"] = os.environ.get(
        "INPUT_LINUX_AMDGPU_PREDEFINED_GROUP", ""
    )
    github_event_args["req_windows_amdgpus_predef"] = os.environ.get(
        "INPUT_WINDOWS_AMDGPU_PREDEFINED_GROUP", ""
    )
    github_event_args["use_runner_label_for_test"] = os.environ.get(
        "INPUT_USE_RUNNER_LABEL_FOR_TEST", ""
    )
    github_event_args["use_runner_label_for_benchmark"] = os.environ.get(
        "INPUT_USE_RUNNER_LABEL_FOR_BENCHMARK", ""
    )

    github_event_args["enforce_overwrite"] = (
        os.environ.get("INPUT_ENFORCE_OVERWRITE", "false") == "true"
    )

    # For external TheRock-CI:
    # As test runner names are frequently updated, we are pulling the runner label data from the ROCm organization variable
    # called "ROCM_THEROCK_TEST_RUNNERS"
    # For more info, go to 'docs/development/test_runner_info.md'
    if os.environ.get("LOAD_TEST_RUNNERS_FROM_VAR", "false") == "true":
        test_runner_json_str = os.getenv("ROCM_THEROCK_TEST_RUNNERS", "{}")
        github_event_args["orgwide_test_runner_dict"] = json.loads(test_runner_json_str)
    else:
        github_event_args["orgwide_test_runner_dict"] = {}

    return github_event_args


def get_requested_amdgpu_families(github_event_args) -> Dict[PlatformMask, set[str]]:
    req_gpu_families_or_targets = {PlatformMask.LINUX: [], PlatformMask.WINDOWS: []}
    if github_event_args["req_linux_amdgpus"]:
        for target in github_event_args["req_linux_amdgpus"].split(","):
            req_gpu_families_or_targets[PlatformMask.LINUX].append(target.strip())
    if github_event_args["req_windows_amdgpus"]:
        for target in github_event_args["req_windows_amdgpus"].split(","):
            req_gpu_families_or_targets[PlatformMask.WINDOWS].append(target.strip())
    if github_event_args["req_linux_amdgpus_predef"]:
        for raw_target in github_event_args["req_linux_amdgpus_predef"].split(","):
            target = raw_target.strip()
            if target in amdgpu_family_predefined_groups:
                req_gpu_families_or_targets[
                    PlatformMask.LINUX
                ] += amdgpu_family_predefined_groups[target]
            else:
                print(
                    f"[WARNING] Predefined group '{target}' not found in amdgpu_family_predefined_groups for {PlatformMask.LINUX.label}",
                    file=sys.stderr,
                )
    if github_event_args["req_windows_amdgpus_predef"]:
        for raw_target in github_event_args["req_windows_amdgpus_predef"].split(","):
            target = raw_target.strip()
            if target in amdgpu_family_predefined_groups:
                req_gpu_families_or_targets[
                    PlatformMask.WINDOWS
                ] += amdgpu_family_predefined_groups[target]
            else:
                print(
                    f"[WARNING] Predefined group '{target}' not found in amdgpu_family_predefined_groups for {PlatformMask.WINDOWS.label}",
                    file=sys.stderr,
                )

    if github_event_args["github_event_name"] == "pull_request":
        pr_labels = get_pr_labels(github_event_args)
        for label in pr_labels:
            if "gfx" in label:
                req_gpu_families_or_targets[PlatformMask.LINUX].append(label.strip())
                req_gpu_families_or_targets[PlatformMask.WINDOWS].append(label.strip())

    # remove duplicates
    for platform in req_gpu_families_or_targets:
        req_gpu_families_or_targets[platform] = set(
            req_gpu_families_or_targets[platform]
        )

    return req_gpu_families_or_targets


def get_overwrite_values(github_event_args) -> dict[str, str]:
    overwrite_values = {}

    value_to_overwrite_labels = ["test_runner", "benchmark_runner"]

    for label in get_pr_labels(github_event_args):
        key, value = label.split(":", 1)
        if key in value_to_overwrite_labels:
            overwrite_values[key] = value.strip()

    return overwrite_values


# TODO TODO add test_type and test_type_reason. update entire selection process.
def get_task_mask(github_event_args):
    force_build = github_event_args["force_build"].upper() in ["YES", "ON", "TRUE", "1"]
    enable_build_jobs = True
    if not force_build and github_event_args["github_event_name"] == "pull_request":
        print(
            "[PULL REQUEST] Checking if build jobs should be enabled based on file changes"
        )
        modified_paths = get_git_modified_paths(github_event_args["base_ref"])
        print("    Modified_paths (max 200):", modified_paths[:200])
        # TODO TODO this pr should close #199
        # TODO(#199): other behavior changes
        #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
        enable_build_jobs = is_ci_run_required(modified_paths)

    task_mask = TaskMask(0)
    if enable_build_jobs:
        task_mask |= TaskMask.BUILD
    if github_event_args["no_tests"].upper() in ["NO", "OFF", "FALSE"]:
        task_mask |= TaskMask.TEST
    if github_event_args["make_release"].upper() in ["YES", "ON", "TRUE"]:
        task_mask |= TaskMask.RELEASE
    return task_mask


if __name__ == "__main__":
    # Setup variables
    github_event_args = get_github_event_args()
    print_github_info(github_event_args)

    req_gpu_families_or_targets = get_requested_amdgpu_families(github_event_args)
    task_mask = get_task_mask(github_event_args)

    platform_mask = PlatformMask(0)
    if req_gpu_families_or_targets[PlatformMask.LINUX]:
        platform_mask |= PlatformMask.LINUX
    if req_gpu_families_or_targets[PlatformMask.WINDOWS]:
        platform_mask |= PlatformMask.WINDOWS

    if github_event_args["multi_arch"]:
        print("Multi-arch mode not supported yet. Exiting")
        sys.exit(1)

    # linux only "gfx94X-dcgpu", "gfx110X-dgpu",
    # export INPUT_LINUX_AMDGPU_FAMILIES="gfx94X-dcgpu, gfx110X-dgpu"
    full_matrix = matrix_generator(
        platform_mask=platform_mask,
        task_mask=task_mask,
        req_gpu_families_or_targets=req_gpu_families_or_targets,
        build_variant=github_event_args["build_variant"],
        overwrite_values=get_overwrite_values(github_event_args),
        enforce_overwrite=github_event_args["enforce_overwrite"],
        orgwide_test_runner_dict=github_event_args["orgwide_test_runner_dict"],
    )

    print("")
    gha_append_step_summary(
        f"""
[ Workflow Config: AMDGPU Family ]

linux_use_prebuilt_artifacts: {json.dumps(github_event_args.get("linux_use_prebuilt_artifacts"))}
windows_use_prebuilt_artifacts: {json.dumps(github_event_args.get("windows_use_prebuilt_artifacts"))}
amdgpu_family_matrix:
{pprint.pformat(full_matrix)}
    """
    )

    gha_set_output({"amdgpu_family_matrix": json.dumps(full_matrix)})
