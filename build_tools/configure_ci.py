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


# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
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


def get_pr_labels() -> List[str]:
    """Gets a list of labels applied to a pull request."""
    labels = json.loads(os.environ.get("PR_LABELS", "[]"))
    return labels


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


def main():
    github_event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    is_pr = github_event_name == "pull_request"
    is_push = github_event_name == "push"
    is_workflow_dispatch = github_event_name == "workflow_dispatch"

    labels = get_pr_labels() if is_pr else []
    # TODO(#199): Use labels or remove the code for handling them

    base_ref = os.environ.get("BASE_REF", "HEAD^1")
    print("Found metadata:")
    print(f"  github_event_name: {github_event_name}")
    print(f"  is_pr: {is_pr}")
    print(f"  is_push: {is_push}")
    print(f"  is_workflow_dispatch: {is_workflow_dispatch}")
    print(f"  labels: {labels}")

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
        includes_non_skippable_path = check_for_non_skippable_path(modified_paths)
        if includes_non_skippable_path:
            print("Enabling build jobs since a non-skippable path was modified")
            enable_build_jobs = True
        else:
            print("Only skippable paths were modified, keeping build jobs disabled")

    write_job_summary(
        f"""## Workflow configure results

* `enable_build_jobs`: {enable_build_jobs}
    """
    )

    output = {
        "enable_build_jobs": json.dumps(enable_build_jobs),
    }
    set_github_output(output)


if __name__ == "__main__":
    main()
