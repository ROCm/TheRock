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

import fnmatch
import json
import os
import subprocess
import sys
from typing import Iterable, List, Optional
import string
from amdgpu_family_matrix import (
    amdgpu_family_info_matrix_presubmit,
    amdgpu_family_info_matrix_postsubmit,
    amdgpu_family_matrix_xfail,
)

from github_actions_utils import *


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
        print("No files were modified, skipping build jobs")
        return False

    paths_set = set(paths)
    github_workflows_paths = set(
        [p for p in paths if p.startswith(".github/workflows")]
    )
    other_paths = paths_set - github_workflows_paths

    related_to_ci = check_for_workflow_file_related_to_ci(github_workflows_paths)
    contains_other_non_skippable_files = check_for_non_skippable_path(other_paths)

    print("should_ci_run_given_modified_paths findings:")
    print(f"  related_to_ci: {related_to_ci}")
    print(f"  contains_other_non_skippable_files: {contains_other_non_skippable_files}")

    if related_to_ci:
        print("Enabling build jobs since a related workflow file was modified")
        return True
    elif contains_other_non_skippable_files:
        print("Enabling build jobs since a non-skippable path was modified")
        return True
    else:
        print(
            "Only unrelated and/or skippable paths were modified, skipping build jobs"
        )
        return False


# --------------------------------------------------------------------------- #
# Matrix creation logic based on PR, push or workflow_dispatch
# --------------------------------------------------------------------------- #


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(args.get("pr_labels"))
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def discover_targets(potential_targets, matrix):
    # iterate through each potential target, validate it exists in our matrix and then append target to run on
    targets = []

    for target in potential_targets:
        # For workflow dispatch triggers, this helps prevent potential user-input errors
        target = target.lower()
        if target in matrix:
            targets.append(target)

    return targets


def matrix_generator(
    is_pull_request=False,
    is_workflow_dispatch=False,
    is_push=False,
    is_schedule=False,
    base_args={},
    families={},
    platform="linux",
):
    """Parses and generates build matrix with build requirements"""
    targets = []
    matrix = amdgpu_family_info_matrix_presubmit | amdgpu_family_info_matrix_postsubmit

    # For the specific event trigger, parse linux and windows target information
    # if the trigger is a workflow_dispatch, parse through the inputs and retrieve the list
    if is_workflow_dispatch:
        print(f"[WORKFLOW_DISPATCH] Generating build matrix with {str(base_args)}")
        # For workflow dispatch, user can select an "expect_failure" family or regular family
        matrix = matrix | amdgpu_family_matrix_xfail

        input_gpu_targets = families.get("amdgpu_families")

        # Sanitizing the string to remove any punctuation from the input
        # After replacing punctuation with spaces, turning string input to an array
        # (ex: ",gfx94X ,|.gfx1201" -> "gfx94X   gfx1201" -> ["gfx94X", "gfx1201"])
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        potential_targets = input_gpu_targets.translate(translator).split()
        targets.extend(discover_targets(potential_targets, matrix))

    # if the trigger is a pull_request label, parse through the labels and retrieve the list
    if is_pull_request:
        print(f"[PULL_REQUEST] Generating build matrix with {str(base_args)}")
        potential_targets = []
        pr_labels = get_pr_labels(base_args)
        for label in pr_labels:
            if "gfx" in label:
                target, _ = label.split("-")
                potential_targets.append(target)
        targets.extend(discover_targets(potential_targets, matrix))

        # Add the presubmit targets
        for target in amdgpu_family_info_matrix_presubmit:
            targets.append(target)

    if is_push and base_args.get("branch_name") == "main":
        print(f"[PUSH - MAIN] Generating build matrix with {str(base_args)}")
        # Add all options except for families that allow failures
        for key in matrix:
            targets.append(key)

    if is_schedule:
        print(f"[SCHEDULE] Generating build matrix with {str(base_args)}")
        # For schedule runs, we will run build and tests for only expect_failure families
        matrix = matrix | amdgpu_family_matrix_xfail

        # Add all options that allow failures
        for key in amdgpu_family_matrix_xfail:
            targets.append(key)

    # Ensure the targets in the list are unique
    unique_targets = list(set(targets))

    target_output = []
    for target in unique_targets:
        if platform in matrix.get(target):
            target_output.append(matrix.get(target).get(platform))

    print(f"Generated build matrix: {str(target_output)}")

    return target_output


# --------------------------------------------------------------------------- #
# Core script logic
# --------------------------------------------------------------------------- #


def main(base_args, linux_families, windows_families):
    github_event_name = base_args.get("github_event_name")
    is_push = github_event_name == "push"
    is_workflow_dispatch = github_event_name == "workflow_dispatch"
    is_pull_request = github_event_name == "pull_request"
    is_schedule = github_event_name == "schedule"

    base_ref = base_args.get("base_ref")
    print("Found metadata:")
    print(f"  github_event_name: {github_event_name}")
    print(f"  is_push: {is_push}")
    print(f"  is_workflow_dispatch: {is_workflow_dispatch}")
    print(f"  is_pull_request: {is_pull_request}")

    print(f"Generating build matrix for Linux: {str(linux_families)}")
    linux_target_output = matrix_generator(
        is_pull_request,
        is_workflow_dispatch,
        is_push,
        is_schedule,
        base_args,
        linux_families,
        platform="linux",
    )

    print(f"Generating test matrix for Windows: {str(windows_families)}")
    windows_target_output = matrix_generator(
        is_pull_request,
        is_workflow_dispatch,
        is_push,
        is_schedule,
        base_args,
        windows_families,
        platform="windows",
    )

    # In the case of a scheduled run, we always want to build
    if is_schedule:
        enable_build_jobs = True
    else:
        modified_paths = get_modified_paths(base_ref)
        print("modified_paths (max 200):", modified_paths[:200])
        print(f"Checking modified files since this had a {github_event_name} trigger")
        # TODO(#199): other behavior changes
        #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
        enable_build_jobs = should_ci_run_given_modified_paths(modified_paths)

    gha_append_step_summary(
        f"""## Workflow configure results

* `linux_amdgpu_families`: {str([item.get("family") for item in linux_target_output])}
* `linux_use_prebuilt_artifacts`: {json.dumps(base_args.get("linux_use_prebuilt_artifacts"))}
* `windows_amdgpu_families`: {str([item.get("family") for item in windows_target_output])}
* `windows_use_prebuilt_artifacts`: {json.dumps(base_args.get("windows_use_prebuilt_artifacts"))}
* `enable_build_jobs`: {json.dumps(enable_build_jobs)}
    """
    )

    output = {
        "linux_amdgpu_families": json.dumps(linux_target_output),
        "windows_amdgpu_families": json.dumps(windows_target_output),
        "enable_build_jobs": json.dumps(enable_build_jobs),
    }
    gha_set_output(output)


if __name__ == "__main__":
    base_args = {}
    linux_families = {}
    windows_families = {}

    linux_families["amdgpu_families"] = os.environ.get(
        "INPUT_LINUX_AMDGPU_FAMILIES", ""
    )

    windows_families["amdgpu_families"] = os.environ.get(
        "INPUT_WINDOWS_AMDGPU_FAMILIES", ""
    )

    # For now, add default run for gfx94X-linux
    base_args["pr_labels"] = os.environ.get("PR_LABELS", "[]")
    base_args["branch_name"] = os.environ.get("GITHUB_REF").split("/")[-1]
    base_args["github_event_name"] = os.environ.get("GITHUB_EVENT_NAME", "")
    base_args["base_ref"] = os.environ.get("BASE_REF", "HEAD^1")
    base_args["linux_use_prebuilt_artifacts"] = (
        os.environ.get("LINUX_USE_PREBUILT_ARTIFACTS") == "true"
    )
    base_args["windows_use_prebuilt_artifacts"] = (
        os.environ.get("WINDOWS_USE_PREBUILT_ARTIFACTS") == "true"
    )

    main(base_args, linux_families, windows_families)
