#!/usr/bin/env python3

"""Configures metadata for a CI workflow run.

----------
| Inputs |
----------

  Environment variables (for all triggers):
  * GITHUB_EVENT_NAME    : GitHub event name, e.g. pull_request.
  * GITHUB_OUTPUT        : path to write workflow output variables.
  * GITHUB_STEP_SUMMARY  : path to write workflow summary output.

  Environment variables (for pull requests):
  * PR_LABELS (optional) : JSON list of PR label names.
  * BASE_REF  (required) : base commit SHA of the PR.

  Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

  Written to GITHUB_OUTPUT:
  * enable_build_jobs : true/false

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
# Matrix creation logic and determinator of PR, workflow_dispatch or push
# --------------------------------------------------------------------------- #

amdgpu_family_info_matrix = {
    "gfx942X": {
        "linux": {"runs-on": "linux-mi300-1gpu-ossci-rocm", "target": "gfx942X-dcgpu"}
    }
}


def get_pr_labels() -> List[str]:
    """Gets a list of labels applied to a pull request."""
    labels = json.loads(os.environ.get("PR_LABELS", "[]"))
    return labels


def matrix_generator(is_pull_request, is_workflow_dispatch, is_push):
    """Parses and generates build matrix with build requirements"""
    potential_linux_targets = []
    potential_windows_targets = []

    # For the specific event trigger, parse linux and windows target information
    if is_workflow_dispatch:
        input_linux_gpu_targets = os.environ.get("INPUT_LINUX_AMDGPU_FAMILIES", "")
        input_windows_gpu_targets = os.environ.get("INPUT_WINDOWS_AMDGPU_FAMILIES", "")

        potential_linux_targets = input_linux_gpu_targets.replace(",", " ").split()
        potential_windows_targets = input_windows_gpu_targets.replace(",", " ").split()

    if is_pull_request:
        for label in get_pr_labels():
            if "gfx" in label:
                target, operating_system = label.split("-")
                if operating_system == "linux":
                    potential_linux_targets.append(target)
                if operating_system == "windows":
                    potential_windows_targets.append(target)

    if is_push and os.environ.get("BRANCH_NAME") == "main":
        # TODO: do we want to run all machines for main branch push? need to figure this out
        pass

    # iterate through each potential target, validate it exists and then append target to run on
    linux_target_output = []
    windows_target_output = []

    for linux_target in potential_linux_targets:
        if (
            linux_target in amdgpu_family_info_matrix
            and "linux" in amdgpu_family_info_matrix.get(linux_target)
        ):
            linux_target_output.append(
                amdgpu_family_info_matrix.get(linux_target).get("linux")
            )

    for windows_target in potential_windows_targets:
        if (
            windows_target in amdgpu_family_info_matrix
            and "windows" in amdgpu_family_info_matrix.get(windows_target)
        ):
            windows_target_output.append(
                amdgpu_family_info_matrix.get(windows_target).get("windows")
            )

    return linux_target_output, windows_target_output


# --------------------------------------------------------------------------- #
# Core script logic
# --------------------------------------------------------------------------- #


def main():
    github_event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    is_push = github_event_name == "push"
    is_workflow_dispatch = github_event_name == "workflow_dispatch"
    is_pull_request = github_event_name == "pull_request"

    base_ref = os.environ.get("BASE_REF", "HEAD^1")
    print("Found metadata:")
    print(f"  github_event_name: {github_event_name}")
    print(f"  is_push: {is_push}")
    print(f"  is_workflow_dispatch: {is_workflow_dispatch}")
    print(f"  is_pull_request: {is_pull_request}")

    modified_paths = get_modified_paths(base_ref)
    print("modified_paths (max 200):", modified_paths[:200])

    enable_build_jobs = False
    if is_workflow_dispatch:
        print("Enabling build jobs since this had a workflow_dispatch trigger")
        enable_build_jobs = True
    else:
        print(
            f"Checking modified files since this had a {github_event_name} trigger, not workflow_dispatch"
        )
        # TODO(#199): other behavior changes
        #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
        enable_build_jobs = should_ci_run_given_modified_paths(modified_paths)

    linux_target_output, windows_target_output = matrix_generator(
        is_pull_request, is_workflow_dispatch, is_push
    )

    write_job_summary(
        f"""## Workflow configure results

* `enable_build_jobs`: {enable_build_jobs}
* `linux_amdgpu_families`: {str([item.get("target") for item in linux_target_output])}
* `windows_amdgpu_families`: {str([item.get("target") for item in windows_target_output])}
    """
    )

    output = {
        "enable_build_jobs": json.dumps(enable_build_jobs),
        "linux_amdgpu_families": json.dumps(linux_target_output),
        "windows_amdgpu_families": json.dumps(windows_target_output),
    }
    set_github_output(output)


if __name__ == "__main__":
    main()
