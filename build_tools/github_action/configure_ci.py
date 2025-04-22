#!/usr/bin/env python3

"""Configures metadata for a CI workflow run.

----------
| Inputs |
----------

  Environment variables (for all triggers):
  * GITHUB_EVENT_NAME    : GitHub event name, e.g. pull_request.
  * GITHUB_OUTPUT        : path to write workflow output variables.
  * GITHUB_STEP_SUMMARY  : path to write workflow summary output.
  * INPUT_BUILD_LINUX_AMDGPU_FAMILIES (optional): Comma-separated string of Linux AMD GPU families to build
  * INPUT_BUILD_WINDOWS_AMDGPU_FAMILIES (optional): Comma-separated string of Windows AMD GPU families to build
  * INPUT_TEST_LINUX_AMDGPU_FAMILIES (optional): Comma-separated string of Linux AMD GPU families to test
  * INPUT_TEST_WINDOWS_AMDGPU_FAMILIES (optional): Comma-separated string of Windows AMD GPU families to test
  * BRANCH_NAME (optional): The branch name

  Environment variables (for pull requests):
  * PR_LABELS (optional) : JSON list of PR label names.
  * BASE_REF  (required) : base commit SHA of the PR.

  Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

  Written to GITHUB_OUTPUT:
  * build_linux_amdgpu_families : List of valid Linux AMD GPU families to execute build jobs
  * build_windows_amdgpu_families : List of valid Windows AMD GPU families to execute build jobs
  * test_linux_amdgpu_families : List of valid Linux AMD GPU families to execute test jobs
  * test_windows_amdgpu_families : List of valid Windows AMD GPU families to execute test jobs

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
from typing import Iterable, List, Mapping, Optional
import string
from amdgpu_family_matrix import amdgpu_family_info_matrix

# --------------------------------------------------------------------------- #
# General utilities
# --------------------------------------------------------------------------- #


def set_github_output(d: Mapping[str, str]):
    """Sets GITHUB_OUTPUT values.
    See https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    print(f"Setting github output:\n{d}")
    step_output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not step_output_file:
        print("Warning: GITHUB_OUTPUT env var not set, can't set github outputs")
        return
    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={v}" + "\n" for k, v in d.items())


def write_job_summary(summary: str):
    """Appends a string to the GitHub Actions job summary.
    See https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
    """
    print(f"Writing job summary:\n{summary}")
    step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not step_summary_file:
        print("Warning: GITHUB_STEP_SUMMARY env var not set, can't write job summary")
        return
    with open(step_summary_file, "a") as f:
        # Use double newlines to split sections in markdown.
        f.write(summary + "\n\n")


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
]


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


# TODO(#199): rename all of these to `ci_*.yml` so this is easier to understand?
GITHUB_WORKFLOWS_CI_PATTERNS = [
    "ci.yml",
    "setup.yml",
    "build_*_packages.yml",
    "test_*_packages.yml",
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

DEFAULT_LINUX_CONFIGURATIONS = ["gfx94X", "gfx110X"]
DEFAULT_WINDOWS_CONFIGURATIONS = ["gfx110X"]


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    data = json.loads(args.get("pr_labels"))
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def matrix_generator(
    is_pull_request, is_workflow_dispatch, is_push, base_args, families, is_test
):
    """Parses and generates build matrix with build requirements"""
    potential_linux_targets = []
    potential_windows_targets = []

    # For the specific event trigger, parse linux and windows target information
    # if the trigger is a workflow_dispatch, parse through the inputs and retrieve the list
    if is_workflow_dispatch:
        print(f"[WORKFLOW_DISPATCH] Generating build matrix with {str(base_args)}")

        input_linux_gpu_targets = families.get("input_linux_amdgpu_families")
        input_windows_gpu_targets = families.get("input_windows_amdgpu_families")

        # Sanitizing the string to remove any punctuation from the input
        # After replacing punctuation with spaces, turning string input to an array
        # (ex: ",gfx94X ,|.gfx1201" -> "gfx94X   gfx1201" -> ["gfx94X", "gfx1201"])
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        potential_linux_targets = input_linux_gpu_targets.translate(translator).split()
        potential_windows_targets = input_windows_gpu_targets.translate(
            translator
        ).split()

    # if the trigger is a pull_request label, parse through the labels and retrieve the list
    if is_pull_request:
        print(f"[PULL_REQUEST] Generating build matrix with {str(base_args)}")
        pr_labels = get_pr_labels(base_args)
        for label in pr_labels:
            if "gfx" in label:
                target, operating_system = label.split("-")
                if operating_system == "linux":
                    potential_linux_targets.append(target)
                if operating_system == "windows":
                    potential_windows_targets.append(target)

        # Add the linux and windows default labels to the potential target lists
        potential_linux_targets.extend(DEFAULT_LINUX_CONFIGURATIONS)
        potential_windows_targets.extend(DEFAULT_WINDOWS_CONFIGURATIONS)

    if is_push and base_args.get("branch_name") == "main":
        print(f"[PUSH - MAIN] Generating build matrix with {str(base_args)}")
        # Add all options
        for key in amdgpu_family_info_matrix:
            if "linux" in amdgpu_family_info_matrix[key]:
                potential_linux_targets.append(key)
            if "windows" in amdgpu_family_info_matrix[key]:
                potential_windows_targets.append(key)

    # Ensure the targets in the list are unique
    potential_linux_targets = list(set(potential_linux_targets))
    potential_windows_targets = list(set(potential_windows_targets))

    # iterate through each potential target, validate it exists in our matrix and then append target to run on
    linux_target_output = []
    windows_target_output = []

    for linux_target in potential_linux_targets:
        # For workflow dispatch triggers, this helps prevent potential user-input errors
        linux_target = linux_target.lower()
        if (
            linux_target in amdgpu_family_info_matrix
            and "linux" in amdgpu_family_info_matrix.get(linux_target)
        ):
            linux_target_output.append(
                amdgpu_family_info_matrix.get(linux_target).get("linux")
            )

    for windows_target in potential_windows_targets:
        # For workflow dispatch triggers, this helps prevent potential user-input errors
        windows_target = windows_target.lower()
        if (
            windows_target in amdgpu_family_info_matrix
            and "windows" in amdgpu_family_info_matrix.get(windows_target)
        ):
            windows_target_output.append(
                amdgpu_family_info_matrix.get(windows_target).get("windows")
            )

    print("Generated build matrix:")
    print(f"   Linux targets: {str(linux_target_output)}")
    print(f"   Windows targets: {str(windows_target_output)}")

    return linux_target_output, windows_target_output


# --------------------------------------------------------------------------- #
# Core script logic
# --------------------------------------------------------------------------- #


def main(base_args, build_families, test_families):
    github_event_name = base_args.get("github_event_name")
    is_push = github_event_name == "push"
    is_workflow_dispatch = github_event_name == "workflow_dispatch"
    is_pull_request = github_event_name == "pull_request"

    base_ref = base_args.get("base_ref")
    print("Found metadata:")
    print(f"  github_event_name: {github_event_name}")
    print(f"  is_push: {is_push}")
    print(f"  is_workflow_dispatch: {is_workflow_dispatch}")
    print(f"  is_pull_request: {is_pull_request}")

    modified_paths = get_modified_paths(base_ref)
    print("modified_paths (max 200):", modified_paths[:200])

    print(f"Generating build matrix for {str(build_families)}")
    build_linux_target_output, build_windows_target_output = matrix_generator(
        is_pull_request, is_workflow_dispatch, is_push, base_args, build_families, False
    )

    print(f"Generating test matrix for {str(test_families)}")
    test_linux_target_output, test_windows_target_output = matrix_generator(
        is_pull_request, is_workflow_dispatch, is_push, base_args, test_families, True
    )

    enable_build_jobs = False
    if not is_workflow_dispatch:
        print(
            f"Checking modified files since this had a {github_event_name} trigger, not workflow_dispatch"
        )
        # TODO(#199): other behavior changes
        #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
        enable_build_jobs = should_ci_run_given_modified_paths(modified_paths)

    # If job trigger is workflow dispatch and user specifies valid build target, build jobs becomes enabled
    if is_workflow_dispatch and (
        build_linux_target_output or build_windows_target_output
    ):
        enable_build_jobs = True

    if not enable_build_jobs:
        build_linux_target_output = []
        build_windows_target_output = []

        # If this enable_build_jobs flag is set to false and the trigger is either a main push or pull request,
        # skip the tests since there is no build to use.
        if not is_workflow_dispatch:
            test_linux_target_output = []
            test_windows_target_output = []

    write_job_summary(
        f"""## Workflow configure results

* `build_linux_amdgpu_families`: {str([item.get("target") for item in build_linux_target_output])}
* `build_windows_amdgpu_families`: {str([item.get("target") for item in build_windows_target_output])}
* `test_linux_amdgpu_families`: {str([item.get("target") for item in test_linux_target_output])}
* `test_windows_amdgpu_families`: {str([item.get("target") for item in test_windows_target_output])}
    """
    )

    output = {
        "build_linux_amdgpu_families": json.dumps(build_linux_target_output),
        "build_windows_amdgpu_families": json.dumps(build_windows_target_output),
        "test_linux_amdgpu_families": json.dumps(test_linux_target_output),
        "test_windows_amdgpu_families": json.dumps(test_windows_target_output),
    }
    set_github_output(output)


if __name__ == "__main__":
    base_args = {}
    build_families = {}
    test_families = {}

    build_families["input_linux_amdgpu_families"] = os.environ.get(
        "INPUT_BUILD_LINUX_AMDGPU_FAMILIES", ""
    )
    build_families["input_windows_amdgpu_families"] = os.environ.get(
        "INPUT_BUILD_WINDOWS_AMDGPU_FAMILIES", ""
    )

    test_families["input_linux_amdgpu_families"] = os.environ.get(
        "INPUT_TEST_LINUX_AMDGPU_FAMILIES", ""
    )
    test_families["input_windows_amdgpu_families"] = os.environ.get(
        "INPUT_TEST_WINDOWS_AMDGPU_FAMILIES", ""
    )

    # For now, add default run for gfx94X-linux
    base_args["pr_labels"] = os.environ.get("PR_LABELS", "[]")
    base_args["branch_name"] = os.environ.get("GITHUB_REF").split("/")[-1]
    base_args["github_event_name"] = os.environ.get("GITHUB_EVENT_NAME", "")
    base_args["base_ref"] = os.environ.get("BASE_REF", "HEAD^1")

    main(base_args, build_families, test_families)
