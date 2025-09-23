"""Utilities for working with GitHub Actions from Python.

See also https://pypi.org/project/github-action-utils/.
"""

import json
import os
from pathlib import Path
import sys
from typing import Mapping
from urllib.request import urlopen, Request


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


def gha_get_request_headers():
    """Gets common request heaers for use with the GitHub REST API.

    See https://docs.github.com/en/rest.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # If GITHUB_TOKEN environment variable is available, include it in the API request to avoid a lower rate limit
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authentication"] = f"Bearer {gh_token}"

    return headers


def gha_send_request(url: str) -> object:
    """Sents a request to the given GitHub REST API URL and returns the response if successful."""
    headers = gha_get_request_headers()

    _log(f"Sending request to URL: {url}")

    request = Request(url, headers=headers)
    with urlopen(request) as response:
        if response.status == 403:
            raise Exception(
                f"Access denied (403 Forbidden). "
                f"Check if your token has the necessary permissions (e.g., `repo`, `workflow`)."
            )
        elif response.status != 200:
            raise Exception(
                f"Received unexpected status code: {response.status}. Please verify the URL or check GitHub API status {response.status}."
            )

        return json.loads(response.read().decode("utf-8"))


def gha_query_workflow_run_information(github_repository: str, workflow_run_id: str):
    """Gets metadata for a workflow run from the GitHub REST API.

    https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28
    """

    url = f"https://api.github.com/repos/{github_repository}/actions/runs/{workflow_run_id}"
    workflow_run = gha_send_request(url)
    return workflow_run


def retrieve_bucket_info(
    github_repository: str | None = None,
    workflow_run_id: str | None = None,
) -> tuple[str, str]:
    """Given a github repository and a workflow run, retrieves bucket information.

    This is intended to segment artifacts by repository and trust level, with
    artifacts split across three buckets:
      * therock-artifacts
      * therock-artifacts-internal
      * therock-artifacts-external

    Typically while run as a continious CI/CD pipeline, this function should
    return the same bucket information for each stage of the pipeline. While
    testing workflows, it can be useful to reference artifacts from other
    repositories and arbitrary buckets to avoid rebuilding. Those test cases
    can use the explicit |github_repository| and |workflow_run_id| parameters.

    Priority for |github_repository| is:
      1. The function argument
      2. The GITHUB_REPOSITORY environment variable
      3. The default, "ROCm/TheRock"

    If |workflow_run_id| is provided, the function will check if that workflow
    run was triggered from different repository than |github_repository| and
    will set |is_pr_from_fork| accordingly. Otherwise, that value is populated
    from the |IS_PR_FROM_FORK| environment variable.

    Returns a tuple [EXTERNAL_REPO, BUCKET], where:
    - EXTERNAL_REPO = if CI is run on an external repo, we create a S3 sub-folder
                      to avoid conflicting run IDs
    - BUCKET = the name of the S3 bucket
    """

    _log("Retrieving bucket info...")

    if github_repository:
        _log(f"  (explicit) github_repository: {github_repository}")
    if not github_repository:
        # Default to the current repository (if any), else ROCm/TheRock.
        github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
        _log(f"  (implicit) github_repository: {github_repository}")

    if workflow_run_id:
        _log(f"  workflow_run_id             : {workflow_run_id}")
        workflow_run = gha_query_workflow_run_information(
            github_repository, workflow_run_id
        )
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository      : {head_github_repository}")
        _log(f"  is_pr_from_fork             : {is_pr_from_fork}")
    else:
        is_pr_from_fork = os.getenv("IS_PR_FROM_FORK", "false") == "true"
        _log(f"  (implicit) is_pr_from_fork  : {is_pr_from_fork}")

    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )

    if external_repo == "":
        bucket = "therock-artifacts"
    elif repo_name == "therock-releases" and owner == "ROCm" and not is_pr_from_fork:
        bucket = "therock-artifacts-internal"
    else:
        bucket = "therock-artifacts-external"

    _log("Retrieved bucket info:")
    _log(f"  external_repo: {external_repo}")
    _log(f"  bucket       : {bucket}")
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
