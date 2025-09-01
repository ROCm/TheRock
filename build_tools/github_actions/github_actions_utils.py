"""Utilities for working with GitHub Actions from Python.

See also https://pypi.org/project/github-action-utils/.
"""

import os
from pathlib import Path
import sys
from typing import Mapping


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def gha_warn_if_not_running_on_ci():
    # https://docs.github.com/en/actions/reference/variables-reference
    if not os.getenv("CI"):
        _log("Warning: 'CI' env var not set, not running under GitHub Actions?")


def gha_add_to_path(new_path: str | Path):
    """Adds an entry to the system PATH for future GitHub Actions workflow run steps.

    This appends to the file located at the $GITHUB_PATH environment variable.

    See
      * https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#example-of-adding-a-system-path
    """
    _log(f"Adding to path by appending to $GITHUB_PATH:\n  '{new_path}'")

    path_file = os.getenv("GITHUB_PATH")
    if not path_file:
        _log("  Warning: GITHUB_PATH env var not set, can't add to path")
        return

    with open(path_file, "a") as f:
        f.write(str(new_path))


def gha_set_env(vars: Mapping[str, str | Path]):
    """Sets environment variables for future GitHub Actions workflow run steps.

    This appends to the file located at the $GITHUB_ENV environment variable.

    See
      * https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#environment-files
    """
    _log(f"Setting environment variable by appending to $GITHUB_ENV:\n  {vars}")

    env_file = os.getenv("GITHUB_ENV")
    if not env_file:
        _log("  Warning: GITHUB_ENV env var not set, can't set environment variable")
        return

    with open(env_file, "a") as f:
        f.writelines(f"{k}={str(v)}" + "\n" for k, v in vars.items())


def gha_set_output(vars: Mapping[str, str | Path]):
    """Sets values in a step's output parameters.

    This appends to the file located at the $GITHUB_OUTPUT environment variable.

    See
      * https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-output-parameter
      * https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/passing-information-between-jobs
    """
    _log(f"Setting github output:\n{vars}")

    step_output_file = os.getenv("GITHUB_OUTPUT")
    if not step_output_file:
        _log("  Warning: GITHUB_OUTPUT env var not set, can't set github outputs")
        return

    with open(step_output_file, "a") as f:
        f.writelines(f"{k}={str(v)}" + "\n" for k, v in vars.items())


def gha_append_step_summary(summary: str):
    """Appends a string to the GitHub Actions job summary.

    This appends to the file located at the $GITHUB_STEP_SUMMARY environment variable.

    See
      * https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#adding-a-job-summary
      * https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#adding-a-job-summary
    """
    _log(f"Writing job summary:\n{summary}")

    step_summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if not step_summary_file:
        _log("  Warning: GITHUB_STEP_SUMMARY env var not set, can't write job summary")
        return

    with open(step_summary_file, "a") as f:
        # Use double newlines to split sections in markdown.
        f.write(summary + "\n\n")


def retrieve_bucket_info() -> tuple[str, str]:
    """Retrieves bucket information based on env variables

    Returns a tuple [EXTERNAL_REPO, BUCKET], where:
    - EXTERNAL_REPO = if CI is run on an external repo, we create a S3 sub-folder
                      to avoid conflicting run IDs
    - BUCKET = the name of the S3 bucket if it's an external repo

    Environment variables:
    - GITHUB_REPOSITORY
    - IS_PR_FROM_FORK
    """
    github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    is_pr_from_fork = os.getenv("IS_PR_FROM_FORK", "false") == "true"
    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )

    # TODO: We should probably change this logic to a default and allow passing in an
    # environment variable.
    if external_repo == "":
        bucket = "therock-artifacts"
    elif repo_name == "therock-releases" and owner == "ROCm" and not is_pr_from_fork:
        bucket = "therock-artifacts-internal"
    else:
        bucket = "therock-artifacts-external"
    return (external_repo, bucket)


def str2bool(value: str | None) -> bool:
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")
