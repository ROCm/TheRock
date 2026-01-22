"""Utilities for working with GitHub Actions from Python.

See also https://pypi.org/project/github-action-utils/.
"""

from datetime import datetime, timezone
from enum import Enum, auto
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request


def _log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class GitHubAPIError(Exception):
    """Error from a GitHub API request.

    Raised when a GitHub API request fails, whether via REST API or gh CLI.
    """

    pass


class GitHubAPI:
    """Client for making GitHub API requests.

    Handles authentication automatically:
    1. If GITHUB_TOKEN env var is set, uses that (CI environment)
    2. If gh CLI is installed and authenticated, uses `gh api` (local dev)
    3. Falls back to unauthenticated requests (rate limited)

    CI workflows are expected to set the GITHUB_TOKEN environment variable from
    ${{ secrets.GITHUB_TOKEN }} or ${{ github.token }}.

    Developers are encouraged to install the `gh` CLI and authenticate with
    `gh auth login`. They _can_ also use a Personal Access Token in the
    GITHUB_TOKEN env var but this is less secure.

    References:
      * https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?apiVersion=2022-11-28
      * https://docs.github.com/en/rest/using-the-rest-api/getting-started-with-the-rest-api?apiVersion=2022-11-28#authentication
      * https://cli.github.com/manual/gh_auth_login
      * https://docs.github.com/en/actions/tutorials/authenticate-with-github_token

    The authentication method is detected once and cached for the lifetime
    of the instance.

    Usage:
        api = GitHubAPI()
        response = api.send_request("https://api.github.com/repos/owner/repo")

    For most use cases, use the module-level functions which use a shared
    singleton instance:
        response = gha_send_request("https://api.github.com/repos/owner/repo")
    """

    class AuthMethod(Enum):
        """Authentication method for GitHub API requests."""

        # Use GITHUB_TOKEN env var (CI environment) with the GitHub REST API.
        GITHUB_TOKEN = auto()

        # Use `gh api` command (local dev with OAuth).
        GH_CLI = auto()

        # Use the GitHub REST API without authenticating, subject to rate limits.
        UNAUTHENTICATED = auto()

    def __init__(self):
        self._auth_method: GitHubAPI.AuthMethod | None = None
        self._github_token: str | None = None
        self._gh_cli_path: str | None = None

    def _detect_auth_method(self) -> AuthMethod:
        """Detects the best available GitHub API authentication method.

        Always performs fresh detection. Use get_auth_method() for cached access.
        """
        # Check for GITHUB_TOKEN (CI environment or PAT)
        token = os.getenv("GITHUB_TOKEN", "")
        if token:
            self._github_token = token
            return GitHubAPI.AuthMethod.GITHUB_TOKEN

        # Check for gh CLI
        gh_path = shutil.which("gh")
        if gh_path:
            # Verify gh is authenticated by checking auth status
            try:
                result = subprocess.run(
                    [gh_path, "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=30,  # seconds
                )
                if result.returncode == 0:
                    self._gh_cli_path = gh_path
                    return GitHubAPI.AuthMethod.GH_CLI
            except (subprocess.TimeoutExpired, OSError):
                pass  # Fall through to unauthenticated

        return GitHubAPI.AuthMethod.UNAUTHENTICATED

    def get_auth_method(self) -> AuthMethod:
        """Gets the current GitHub API authentication method.

        Returns the detected auth method (GITHUB_TOKEN, GH_CLI, or UNAUTHENTICATED).
        The result is cached after the first call.
        """
        if self._auth_method is None:
            self._auth_method = self._detect_auth_method()
        return self._auth_method

    def is_authenticated(self) -> bool:
        """Checks if authenticated GitHub API access is available.

        Returns True if either GITHUB_TOKEN is set or gh CLI is authenticated.
        """
        return self.get_auth_method() != GitHubAPI.AuthMethod.UNAUTHENTICATED

    def _get_request_headers(self) -> dict[str, str]:
        """Gets common request headers for use with the GitHub REST API.

        Note: This is only used for direct REST API calls (GITHUB_TOKEN or
        unauthenticated). When using gh CLI, headers are handled by gh.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        if self.get_auth_method() == GitHubAPI.AuthMethod.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {self._github_token}"

        return headers

    def _send_request_via_gh_cli(self, url: str, timeout_seconds: int) -> object:
        """Sends a GitHub API request using the gh CLI.

        Raises:
            GitHubAPIError: If the request fails for any reason.
        """
        assert self._gh_cli_path is not None, (
            "_send_request_via_gh_cli called without gh CLI path set. "
            "Call get_auth_method() first."
        )

        # Strip the base URL to get the API path
        api_path = url.removeprefix("https://api.github.com")

        try:
            result = subprocess.run(
                [self._gh_cli_path, "api", api_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            raise GitHubAPIError(
                f"gh api request timed out after {timeout_seconds}s for {api_path}"
            ) from e
        except OSError as e:
            raise GitHubAPIError(
                f"Failed to execute gh CLI at {self._gh_cli_path}: {e}"
            ) from e

        if result.returncode != 0:
            stderr = result.stderr or "(no error message)"
            raise GitHubAPIError(f"gh api request failed: {stderr}")

        if not result.stdout:
            raise GitHubAPIError("gh api returned empty response")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise GitHubAPIError(
                f"gh api returned invalid JSON: {e.msg} at position {e.pos}"
            ) from e

    def _send_request_via_rest_api(self, url: str, timeout_seconds: int) -> object:
        """Sends a GitHub API request using the REST API directly.

        Raises:
            GitHubAPIError: If the request fails for any reason.
        """
        headers = self._get_request_headers()
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as e:
            if e.code == 403:
                raise GitHubAPIError(
                    f"Access denied (403 Forbidden) for {url}. "
                    f"Check if your token has the necessary permissions (e.g., `repo`, `workflow`)."
                ) from e
            elif e.code == 404:
                raise GitHubAPIError(
                    f"Resource not found (404) for {url}. "
                    f"Verify the repository, workflow, or run ID exists."
                ) from e
            else:
                raise GitHubAPIError(
                    f"HTTP {e.code} error for {url}: {e.reason}"
                ) from e
        except URLError as e:
            raise GitHubAPIError(f"Network error for {url}: {e.reason}") from e
        except TimeoutError as e:
            raise GitHubAPIError(
                f"Request timed out after {timeout_seconds}s for {url}"
            ) from e

        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise GitHubAPIError(
                f"Invalid JSON response from {url}: {e.msg} at position {e.pos}"
            ) from e

    def send_request(self, url: str, timeout_seconds: int = 300) -> object:
        """Sends a request to the given GitHub REST API URL.

        Args:
            url: Full GitHub API URL (e.g., https://api.github.com/repos/...)
            timeout_seconds: Request timeout in seconds (default 300).

        Returns:
            Parsed JSON response.

        Raises:
            GitHubAPIError: If the request fails (network error, HTTP error,
                timeout, invalid JSON, etc.). The original exception is
                available via the __cause__ attribute.
        """
        auth_method = self.get_auth_method()

        if auth_method == GitHubAPI.AuthMethod.GH_CLI:
            return self._send_request_via_gh_cli(url, timeout_seconds)

        if auth_method == GitHubAPI.AuthMethod.UNAUTHENTICATED:
            _log("Warning: No GitHub auth available, requests may be rate limited")

        return self._send_request_via_rest_api(url, timeout_seconds)


# Module-level singleton with cached state.
# Tests may opt to use separate instances to exercise the state more precisely.
_default_github_api = GitHubAPI()


def is_authenticated_github_api_available() -> bool:
    """Checks if authenticated GitHub API access is available.

    Returns True if either GITHUB_TOKEN is set or gh CLI is authenticated.
    Useful for tests to decide whether to skip network-dependent tests.
    """
    return _default_github_api.is_authenticated()


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
    _log(f"Setting github output:\n{json.dumps(vars, indent=2)}")

    step_output_file = os.getenv("GITHUB_OUTPUT")
    if not step_output_file:
        _log("  Warning: GITHUB_OUTPUT env var not set, can't set github outputs")
        return

    with open(step_output_file, "a") as f:
        for k, v in vars.items():
            print(f"OUTPUT {k}={str(v)}")
            f.write(f"{k}={str(v)}\n")


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


def gha_send_request(url: str, timeout_seconds: int = 300) -> object:
    """Sends a request to the given GitHub REST API URL and returns the response.

    Authentication is handled automatically:
    1. If GITHUB_TOKEN env var is set, uses that (CI environment)
    2. If gh CLI is installed and authenticated, uses `gh api` (local dev)
    3. Falls back to unauthenticated requests (rate limited)

    Args:
        url: Full GitHub API URL (e.g., https://api.github.com/repos/...)
        timeout_seconds: Request timeout in seconds (default 300).

    Returns:
        Parsed JSON response.

    Raises:
        GitHubAPIError: If the request fails (network error, HTTP error,
            timeout, invalid JSON, etc.). The original exception is available
            via the __cause__ attribute.
    """
    return _default_github_api.send_request(url, timeout_seconds=timeout_seconds)


def gha_query_workflow_run_by_id(github_repository: str, workflow_run_id: str) -> dict:
    """Gets metadata for a workflow run by its run ID.

    Uses the GitHub REST API endpoint: /actions/runs/{run_id}

    Args:
        github_repository: Repository in "owner/repo" format (e.g., "ROCm/TheRock")
        workflow_run_id: The workflow run ID (e.g., "12345678901")

    Returns:
        Workflow run metadata dict from GitHub API.

    See: https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run
    """
    url = f"https://api.github.com/repos/{github_repository}/actions/runs/{workflow_run_id}"
    return gha_send_request(url)


def gha_query_workflow_runs_for_commit(
    github_repository: str,
    workflow_file_name: str,
    git_commit_sha: str,
) -> list[dict]:
    """Gets all workflow runs for a specific commit.

    Uses the GitHub REST API endpoint: /actions/workflows/{workflow}/runs?head_sha={sha}

    A commit may have multiple workflow runs if the workflow was retriggered.
    The list is ordered by most recent first.

    Note: The API returns up to 30 results by default (first page only).
    For a single commit this is typically sufficient.

    Args:
        github_repository: Repository in "owner/repo" format (e.g., "ROCm/TheRock")
        workflow_file_name: Workflow filename (e.g., "ci.yml")
        git_commit_sha: Full git commit SHA

    Returns:
        List of workflow run metadata dicts, ordered most recent first.
        Empty list if no runs exist for this commit.

    See: https://docs.github.com/en/rest/actions/workflow-runs#list-workflow-runs-for-a-workflow
    """
    url = (
        f"https://api.github.com/repos/{github_repository}"
        f"/actions/workflows/{workflow_file_name}/runs?head_sha={git_commit_sha}"
    )
    response = gha_send_request(url)
    return response.get("workflow_runs", [])


def gha_query_last_successful_workflow_run(
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "ci.yml",
    branch: str = "main",
) -> dict | None:
    """Find the last successful run of a specific workflow on the specified branch.

    Args:
        github_repository: Repository in format "owner/repo"
        workflow_name: Name of the workflow file (e.g., "ci_nightly.yml")
        branch: Branch to filter by (defaults to "main")

    Returns:
        The full workflow run object of the most recent successful run on the specified branch,
        or None if no successful runs are found.
    """
    # Use GitHub API query parameters to pre-filter for successful runs on the specified branch
    url = f"https://api.github.com/repos/{github_repository}/actions/workflows/{workflow_name}/runs?status=success&branch={branch}&per_page=100"
    response = gha_send_request(url)

    # Return the first (most recent) successful run
    if response and response.get("workflow_runs"):
        return response["workflow_runs"][0]
    return None


def retrieve_bucket_info(
    github_repository: str | None = None,
    workflow_run_id: str | None = None,
    workflow_run: dict | None = None,
) -> tuple[str, str]:
    """Given a github repository and a workflow run, retrieves bucket information.

    This is intended to segment artifacts by repository and trust level, with
    artifacts split across several buckets:
      * therock-ci-artifacts
      * therock-ci-artifacts-external
      * therock-artifacts-internal
      * therock-dev-artifacts
      * therock-nightly-artifacts
      * therock-release-artifacts

    Typically while run as a continious CI/CD pipeline, this function should
    return the same bucket information for each stage of the pipeline. While
    testing workflows, it can be useful to reference artifacts from other
    repositories and arbitrary buckets to avoid rebuilding. Those test cases
    can use the explicit |github_repository| and |workflow_run_id| parameters.

    Priority for |github_repository| is:
      1. The function argument
      2. The GITHUB_REPOSITORY environment variable
      3. The default, "ROCm/TheRock"

    If |workflow_run| is provided, uses it directly without making an API call.
    Otherwise, if |workflow_run_id| is provided, fetches the workflow run data
    from the GitHub API.

    If neither is provided, |is_pr_from_fork| is populated from the
    IS_PR_FROM_FORK environment variable.

    Returns a tuple [EXTERNAL_REPO, BUCKET], where:
    - EXTERNAL_REPO = if CI is run on an external repo, we create a S3 sub-folder
                      to avoid conflicting run IDs
    - BUCKET = the name of the S3 bucket
    """

    _log("Retrieving bucket info...")

    if github_repository:
        _log(f"  (explicit) github_repository: {github_repository}")
    else:
        # Default to the current repository (if any), else ROCm/TheRock.
        github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
        _log(f"  (implicit) github_repository: {github_repository}")

    # Fetch workflow_run from API if not provided but workflow_run_id is set
    if workflow_run is None and workflow_run_id is not None:
        workflow_run = gha_query_workflow_run_by_id(github_repository, workflow_run_id)

    # Extract metadata from workflow_run if available
    curr_commit_dt = None
    if workflow_run is not None:
        _log(f"  workflow_run_id             : {workflow_run['id']}")
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository      : {head_github_repository}")
        _log(f"  is_pr_from_fork             : {is_pr_from_fork}")

        # From TheRock #2046 onward, a new S3 bucket was used.
        # This datetime comparison will determine whether to download from older bucket or newer bucket.
        curr_commit_dt = datetime.strptime(
            workflow_run["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
        )
        curr_commit_dt = curr_commit_dt.replace(tzinfo=timezone.utc)
        commit_to_compare_dt = datetime.fromisoformat("2025-11-11T16:18:48+00:00")
    else:
        is_pr_from_fork = os.getenv("IS_PR_FROM_FORK", "false") == "true"
        _log(f"  (implicit) is_pr_from_fork  : {is_pr_from_fork}")

    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )

    release_type = os.getenv("RELEASE_TYPE")
    if release_type:
        _log(f"  (implicit) RELEASE_TYPE: {release_type}")
        bucket = f"therock-{release_type}-artifacts"
    else:
        if external_repo == "":
            bucket = "therock-ci-artifacts"
            if curr_commit_dt and curr_commit_dt <= commit_to_compare_dt:
                bucket = "therock-artifacts"
        elif (
            repo_name == "therock-releases-internal"
            and owner == "ROCm"
            and not is_pr_from_fork
        ):
            bucket = "therock-artifacts-internal"
        else:
            bucket = "therock-ci-artifacts-external"
            if curr_commit_dt and curr_commit_dt <= commit_to_compare_dt:
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


def get_visible_gpu_count(env=None, therock_bin_dir: str | None = None) -> int:
    rocminfo = Path(therock_bin_dir) / "rocminfo"
    rocminfo_cmd = str(rocminfo) if rocminfo.exists() else "rocminfo"

    result = subprocess.run(
        [rocminfo_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )

    pattern = re.compile(r"^\s*Name:\s+gfx[0-9a-z]+$", re.IGNORECASE)

    return sum(1 for line in result.stdout.splitlines() if pattern.match(line.strip()))
