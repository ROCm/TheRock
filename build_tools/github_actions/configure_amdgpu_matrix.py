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

import copy
import fnmatch
import json
import os
import subprocess
import sys
from typing import Iterable, List, Optional
from new_amdgpu_family_matrix import (
    amdgpu_family_predefined_groups,
    amdgpu_family_info_matrix_all,
)

from github_actions_utils import *

from enum import Flag, auto
import pprint

# --------------------------------------------------------------------------- #
# Filtering by modified paths
# --------------------------------------------------------------------------- #


def get_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of modified files relative to the base reference."""
    try:
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except TimeoutError:
        print(
            "Computing modified files timed out. Not using PR diff to determine"
            " jobs to run.",
            file=sys.stderr,
        )
        return None


# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*.pre-commit-config.*",
    "*LICENSE",
    # Changes to 'external-builds/' (e.g. PyTorch) do not affect "CI" workflows.
    # At time of writing, workflows run in this sequence:
    #   `ci.yml`
    #   `ci_linux.yml`
    #   `build_linux_packages.yml`
    #   `test_linux_packages.yml`
    #   `test_[rocm subproject].yml`
    # If we add external-builds tests there, we can revisit this, maybe leaning
    # on options like LINUX_USE_PREBUILT_ARTIFACTS or sufficient caching to keep
    # workflows efficient when only nodes closer to the edges of the build graph
    # are changed.
    "external-builds/*",
    # Changes to experimental code do not run standard build/test workflows.
    "experimental/*",
]


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


GITHUB_WORKFLOWS_CI_PATTERNS = [
    "setup.yml",
    "ci*.yml",
    "build*package*.yml",
    "test*packages.yml",
    "test*.yml",  # This may be too broad, but there are many test workflows.
]


def is_path_workflow_file_related_to_ci(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    )


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(is_path_workflow_file_related_to_ci(p) for p in paths)


def should_ci_run_given_modified_paths(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if CI workflows should run given a list of modified paths."""

    if paths is None:
        print("    No files were modified, skipping build jobs")
        return False

    paths_set = set(paths)
    github_workflows_paths = set(
        [p for p in paths if p.startswith(".github/workflows")]
    )
    other_paths = paths_set - github_workflows_paths

    related_to_ci = check_for_workflow_file_related_to_ci(github_workflows_paths)
    contains_other_non_skippable_files = check_for_non_skippable_path(other_paths)

    print(f"    Modified paths/files related to ci: {related_to_ci}")
    print(
        f"    PR contains other non-skippable files: {contains_other_non_skippable_files}"
    )

    if related_to_ci:
        print("--> Enabling build jobs since a related workflow file was modified")
        return True
    elif contains_other_non_skippable_files:
        print("--> Enabling build jobs since a non-skippable path was modified")
        return True
    else:
        print(
            "--> Skipping build jobs since only unrelated and/or skippable paths were modified"
        )
        return False


# --------------------------------------------------------------------------- #
# Matrix creation logic based on PR, push, or workflow_dispatch
# --------------------------------------------------------------------------- #


def get_pr_labels(github_event_args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(github_event_args["pr_labels"])
    labels = []
    if not len(data) == 0:
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


taskLabel = {
    TaskMask.BUILD: {"label": "build", "amdgpu_family_label": ""},
    TaskMask.TEST: {"label": "test", "amdgpu_family_label": "test"},
    TaskMask.RELEASE: {"label": "release", "amdgpu_family_label": "release"},
}


class PlatformMask(Flag):
    LINUX = auto()
    WINDOWS = auto()


platformLabel = {PlatformMask.LINUX: "linux", PlatformMask.WINDOWS: "windows"}


def get_build_config(amdgpu_matrix_entry, build_variant, platform_str, arch):
    # amdgpu_matrix_entry is the "build" entry of a single arch and a given platform from new_amdgpu_family_matrix.py

    # We have custom build variants for specific CI flows.
    # For CI, we use the release build variant (for PRs, pushes to main, nightlies)
    # For CI ASAN, we use the ASAN build variant (for pushes to main)
    # In the case that the build variant is not requested, we skip it

    # check if build_variant is available
    if build_variant not in amdgpu_matrix_entry["build_variants"]:
        print(
            f"[WARNING] Build variant {build_variant} is not available for {arch} on {platform_str}"
        )
        return None

    # get all build variants settings for the given platform
    # Import build variants from new_amdgpu_family_matrix
    from new_amdgpu_family_matrix import all_build_variants

    build_variants_settings = all_build_variants[platform_str][build_variant]
    if not build_variants_settings:
        print(f"[ERROR] Build variant {build_variant} does not exist on {platform_str}")
        return None

    # copy build variants settings to the build_config
    build_config = copy.deepcopy(build_variants_settings)

    # Assign a computed "artifact_group" combining the family and variant.
    artifact_group = arch
    build_variant_suffix = build_variants_settings["build_variant_suffix"]
    # only add build_variant_suffix if not empty
    if build_variants_settings["build_variant_suffix"]:
        artifact_group += f"-{build_variant_suffix}"
    build_config["artifact_group"] = artifact_group

    # set expect_failure
    # If it is not set, default to False
    # Otherwise any True overwrites it (either in the arch or by the build_variant)
    build_config["expect_failure"] = build_variants_settings.get(
        "expect_failure", False
    ) or amdgpu_matrix_entry.get("expect_failure", False)

    # Make future-proof: copy all other keys from amdgpu_matrix_entry to build_config
    for key in amdgpu_matrix_entry.keys():
        if key not in build_config.keys() and key != "build_variants":
            build_config[key] = amdgpu_matrix_entry[key]

    return build_config


def get_test_config(amdgpu_matrix_entry, plat_str, target):
    # only run test if run_tests is True
    if not amdgpu_matrix_entry.get("run_tests", False):
        return None

    # copy test config
    test_config = copy.deepcopy(amdgpu_matrix_entry)

    # TODO TODO automate this. have default values in new_amdgpu_family_matrix.py
    # set default values if not set
    if "benchmark_runs_on" not in test_config.keys():
        test_config["benchmark_runs_on"] = ""
    if "runs_on" not in test_config.keys():
        test_config["runs_on"] = ""
    if "sanity_check_only_for_family" not in test_config.keys():
        test_config["sanity_check_only_for_family"] = False
    if "expect_pytorch_failure" not in test_config.keys():
        test_config["expect_pytorch_failure"] = False

    # sanity check: check if we have some machine to run on, otherwise skip test job
    if not test_config["runs_on"] and not test_config["benchmark_runs_on"]:
        return None

    return test_config


def get_release_config(amdgpu_matrix_entry, plat_str, target):
    # only run releases if we want to also push them
    if not amdgpu_matrix_entry.get("push_on_success", False):
        return None

    # copy release config
    release_config = copy.deepcopy(amdgpu_matrix_entry)

    if "bypass_tests_for_releases" not in release_config.keys():
        release_config["bypass_tests_for_releases"] = False

    return release_config


def new_matrix_generator(
    platformMask: PlatformMask,
    taskMask: TaskMask,
    req_gpu_families_or_targets: List[str],
    build_variant: str = "release",
):
    # TODO TODO move to main() those checks
    if not bool(platformMask):
        print("No platform set. Exiting")
        sys.exit(1)
    if not bool(taskMask):
        print("No task (build, test, release) set. Exiting")
        sys.exit(1)

    # extract proper arch names for the requested famillies and targets, and platforms based on
    # the amdgpu_family_info_matrix_all from new_amdgpu_family_matrix.py
    gpu_matrix = {PlatformMask.LINUX: {}, PlatformMask.WINDOWS: {}}
    for platform in gpu_matrix.keys():
        plat_str = platformLabel[platform]
        # get existing build targets
        for target in req_gpu_families_or_targets[platform]:
            # single gpu architecture
            if target[-1].isdigit():
                print(f"target is a single gpu architecture {target}")
                family = target[:-1] + "x"
                gpu = amdgpu_family_info_matrix_all[family][target]
                if plat_str in gpu.keys():
                    gpu_matrix[platform][target] = gpu[plat_str]
            elif "-" in target:
                # gpu family like gfx110x-dgpu
                family_parts = target.split("-")
                family = amdgpu_family_info_matrix_all[family_parts[0]][family_parts[1]]
                if plat_str in family.keys():
                    gpu_matrix[platform][target] = family[plat_str]
            else:
                print(
                    f"ERROR! No entry for {target} on {platformLabel[platform]} found - Skipping!",
                    file=sys.stderr,
                )
            # TODO TODO should we add also support for "gfx115x" and then add ALL subtargets of it? e.g. gfx1151, gfx115x-dcgpu, ..
            # This would needd to change it to add the build target names as key to gpu_matrix

    print("")
    print(f"Found the following AMDGPU targets (family/arch):")
    print(f"    Linux:   {[ele for ele in gpu_matrix[PlatformMask.LINUX].keys()]}")
    print(f"    Windows: {[ele for ele in gpu_matrix[PlatformMask.WINDOWS].keys()]}")
    print("")
    tasks = [taskLabel[mask]["label"] for mask in taskMask]
    print(
        f"Creating AMDGPU matrix for the AMDGPU targets and tasks {tasks}... ", end=""
    )

    print(gpu_matrix)

    # assign to platform, task, and build variant
    full_matrix = {}
    for platform in platformMask:  # linux, windows
        platform_matrix = []
        plat_str = platformLabel[platform]
        for target in gpu_matrix[platform]:  # gfx94X-dcgpu, gfx1151, ...
            data = {"amdgpu_family": target}

            if TaskMask.BUILD in taskMask:
                build_config = get_build_config(
                    gpu_matrix[platform][target]["build"],
                    build_variant,
                    plat_str,
                    target,
                )
                if build_config:
                    data[taskLabel[TaskMask.BUILD]["label"]] = build_config
                else:
                    print(
                        f"[WARNING] Skipping build job for {target} on {platformLabel[platform]} since build variant {build_variant} is not supported"
                    )
            if TaskMask.TEST in taskMask:
                test_config = get_test_config(
                    gpu_matrix[platform][target]["test"], plat_str, target
                )
                if test_config:
                    data[taskLabel[TaskMask.TEST]["label"]] = test_config
                else:
                    print(
                        f"[WARNING] Skipping test job for {target} on {platformLabel[platform]}. Test config returned empty."
                    )
            if TaskMask.RELEASE in taskMask:
                release_config = get_release_config(
                    gpu_matrix[platform][target]["release"], plat_str, target
                )
                if release_config:
                    data[taskLabel[TaskMask.RELEASE]["label"]] = release_config
                else:
                    print(
                        f"[WARNING] Skipping release job for {target} on {platformLabel[platform]}. Release config returned empty."
                    )

            # more than amdgpu_family as entry? Means we want to run some tasks, so add it
            if len(data.keys()) > 1:
                platform_matrix += [data]
        full_matrix[plat_str] = platform_matrix

    print("done")
    return full_matrix


def get_github_event_args():
    github_event_args = {}
    github_event_args["pr_labels"] = os.environ.get("PR_LABELS", "[]")
    github_event_args["branch_name"] = os.environ.get(
        "GITHUB_REF", "not/a/notaref"
    ).split("/")[-1]
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

    return github_event_args


def get_requested_amdgpu_families(github_event_args):
    req_gpu_families_or_targets = {PlatformMask.LINUX: [], PlatformMask.WINDOWS: []}
    if len(github_event_args["req_linux_amdgpus"]) > 0:
        for target in github_event_args["req_linux_amdgpus"].split(","):
            req_gpu_families_or_targets[PlatformMask.LINUX].append(target.strip())
    if len(github_event_args["req_windows_amdgpus"]) > 0:
        for target in github_event_args["req_windows_amdgpus"].split(","):
            req_gpu_families_or_targets[PlatformMask.WINDOWS].append(target.strip())
    if len(github_event_args["req_linux_amdgpus_predef"]) > 0:
        for target in github_event_args["req_linux_amdgpus_predef"].split(","):
            target = target.strip()
            if target in amdgpu_family_predefined_groups.keys():
                req_gpu_families_or_targets[
                    PlatformMask.LINUX
                ] += amdgpu_family_predefined_groups[target]
    if len(github_event_args["req_windows_amdgpus_predef"]) > 0:
        for target in github_event_args["req_windows_amdgpus_predef"].split(","):
            target = target.strip()
            if target in amdgpu_family_predefined_groups.keys():
                req_gpu_families_or_targets[
                    PlatformMask.WINDOWS
                ] += amdgpu_family_predefined_groups[target]

    if github_event_args["github_event_name"] == "pull_request":
        pr_labels = get_pr_labels(github_event_args)
        for label in pr_labels:
            if "gfx" in label:
                req_gpu_families_or_targets[PlatformMask.LINUX] += label
                req_gpu_families_or_targets[PlatformMask.WINDOWS] += label

    # remove duplicates
    for platform in req_gpu_families_or_targets.keys():
        req_gpu_families_or_targets[platform] = set(
            req_gpu_families_or_targets[platform]
        )

    return req_gpu_families_or_targets


def getTaskMask(github_event_args):
    enable_build_jobs = True
    if github_event_args["force_build"].upper() in ["YES", "ON", "TRUE"]:
        enable_build_jobs = True
    else:
        if github_event_args["github_event_name"] == "pull_request":
            print(
                "[PULL REQUEST] Checking if build jobs should be enabled based on file changes"
            )
            modified_paths = get_modified_paths(github_event_args["base_ref"])
            print("    Modified_paths (max 200):", modified_paths[:200])
            # TODO TODO this pr should close #199
            # TODO(#199): other behavior changes
            #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
            enable_build_jobs = should_ci_run_given_modified_paths(modified_paths)

    taskMask = TaskMask(0)
    if enable_build_jobs:
        taskMask |= TaskMask.BUILD
    if github_event_args["no_tests"].upper() in ["NO", "OFF", "FALSE"]:
        taskMask |= TaskMask.TEST
    if github_event_args["make_release"].upper() in ["YES", "ON", "TRUE"]:
        taskMask |= TaskMask.RELEASE
    return taskMask


if __name__ == "__main__":
    # Setup variables
    github_event_args = get_github_event_args()
    print_github_info(github_event_args)

    req_gpu_families_or_targets = get_requested_amdgpu_families(github_event_args)
    taskMask = getTaskMask(github_event_args)

    platformMask = PlatformMask(0)
    if len(req_gpu_families_or_targets[PlatformMask.LINUX]) > 0:
        platformMask |= PlatformMask.LINUX
    if len(req_gpu_families_or_targets[PlatformMask.WINDOWS]) > 0:
        platformMask |= PlatformMask.WINDOWS

    if github_event_args["multi_arch"]:
        print("Multi-arch mode not supported yet. Exiting")
        sys.exit(1)

    # linux only "gfx94X-dcgpu", "gfx110X-dgpu",
    # export INPUT_LINUX_AMDGPU_FAMILIES="gfx94X-dcgpu, gfx110X-dgpu"
    full_matrix = new_matrix_generator(
        platformMask=platformMask,
        taskMask=taskMask,
        req_gpu_families_or_targets=req_gpu_families_or_targets,
        build_variant=github_event_args["build_variant"],
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
