# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small GitHub REST helpers used by therock-tools artifact probes.

The TheRock-specific GitHub Actions workflow helpers live under
``build_tools/github_actions``. This module keeps the installed package limited
to generic GitHub API operations needed by the artifact lookup tools.
"""

import base64
import binascii
from enum import Enum, auto
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def _log(*args: object, **kwargs: object):
    print(*args, **kwargs)
    sys.stdout.flush()


class GitHubAPIError(Exception):
    """Error from a GitHub API request."""


class GitHubAPI:
    """Client for making authenticated GitHub REST API requests."""

    class AuthMethod(Enum):
        """Authentication method for GitHub API requests."""

        GITHUB_TOKEN = auto()
        GH_CLI = auto()
        UNAUTHENTICATED = auto()

    def __init__(self):
        self._auth_method: GitHubAPI.AuthMethod | None = None
        self._github_token: str | None = None
        self._gh_cli_path: str | None = None

    def _detect_auth_method(self) -> AuthMethod:
        token = os.getenv("GITHUB_TOKEN", "")
        if token:
            self._github_token = token
            return GitHubAPI.AuthMethod.GITHUB_TOKEN

        gh_path = shutil.which("gh")
        if gh_path:
            try:
                result = subprocess.run(
                    [gh_path, "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self._gh_cli_path = gh_path
                    return GitHubAPI.AuthMethod.GH_CLI
            except (subprocess.TimeoutExpired, OSError):
                pass

        return GitHubAPI.AuthMethod.UNAUTHENTICATED

    def get_auth_method(self) -> AuthMethod:
        if self._auth_method is None:
            self._auth_method = self._detect_auth_method()
        return self._auth_method

    def is_authenticated(self) -> bool:
        return self.get_auth_method() != GitHubAPI.AuthMethod.UNAUTHENTICATED

    def _get_request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.get_auth_method() == GitHubAPI.AuthMethod.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {self._github_token}"
        return headers

    def _send_request_via_gh_cli(self, url: str, timeout_seconds: int) -> object:
        assert self._gh_cli_path is not None, (
            "_send_request_via_gh_cli called without gh CLI path set. "
            "Call get_auth_method() first."
        )
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
        request = Request(url, headers=self._get_request_headers())
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            if e.code == 403:
                if "rate limit" in error_body.lower():
                    raise GitHubAPIError(
                        f"GitHub API rate limit exceeded for {url}. "
                        "Authenticate with `gh auth login` or set GITHUB_TOKEN "
                        "to increase limits."
                    ) from e
                raise GitHubAPIError(
                    f"Access denied (403 Forbidden) for {url}. "
                    "Check if your token has the necessary permissions."
                ) from e
            if e.code == 404:
                raise GitHubAPIError(
                    f"Resource not found (404) for {url}. "
                    "Verify the repository, workflow, or run ID exists."
                ) from e
            raise GitHubAPIError(f"HTTP {e.code} error for {url}: {e.reason}") from e
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
        auth_method = self.get_auth_method()
        if auth_method == GitHubAPI.AuthMethod.GH_CLI:
            return self._send_request_via_gh_cli(url, timeout_seconds)
        if auth_method == GitHubAPI.AuthMethod.UNAUTHENTICATED:
            _log("Warning: No GitHub auth available, requests may be rate limited")
        return self._send_request_via_rest_api(url, timeout_seconds)


_default_github_api = GitHubAPI()


def _expect_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GitHubAPIError(f"Expected GitHub API object response for {context}")
    return value


def is_authenticated_github_api_available() -> bool:
    return _default_github_api.is_authenticated()


def gha_load_github_event() -> dict[str, object]:
    path = os.environ["GITHUB_EVENT_PATH"]
    with open(path, encoding="utf-8") as f:
        return _expect_dict(json.load(f), path)


def gha_send_request(url: str, timeout_seconds: int = 300) -> object:
    return _default_github_api.send_request(url, timeout_seconds=timeout_seconds)


def gha_query_workflow_run_by_id(
    github_repository: str, workflow_run_id: str
) -> dict[str, object]:
    url = f"https://api.github.com/repos/{github_repository}/actions/runs/{workflow_run_id}"
    return _expect_dict(gha_send_request(url), url)


def gha_query_workflow_runs_for_commit(
    github_repository: str,
    workflow_file_name: str,
    git_commit_sha: str,
) -> list[dict[str, object]]:
    url = (
        f"https://api.github.com/repos/{github_repository}"
        f"/actions/workflows/{workflow_file_name}/runs"
        f"?head_sha={git_commit_sha}&sort=created&direction=desc"
    )
    response = _expect_dict(gha_send_request(url), url)
    runs = response.get("workflow_runs", [])
    if not isinstance(runs, list):
        raise GitHubAPIError(f"Expected workflow_runs list in response for {url}")

    result: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            raise GitHubAPIError(f"Expected workflow run object in response for {url}")
        result.append(run)
    result.sort(key=lambda run: str(run.get("created_at", "")), reverse=True)
    return result


def gha_query_last_successful_workflow_run(
    github_repository: str = "ROCm/TheRock",
    workflow_name: str = "multi_arch_ci.yml",
    branch: str = "main",
) -> dict[str, object] | None:
    url = (
        f"https://api.github.com/repos/{github_repository}"
        f"/actions/workflows/{workflow_name}/runs"
        f"?status=success&branch={branch}&per_page=100&sort=created&direction=desc"
    )
    response = _expect_dict(gha_send_request(url), url)
    runs = response.get("workflow_runs", [])
    if not isinstance(runs, list) or not runs:
        return None
    first_run = runs[0]
    if not isinstance(first_run, dict):
        raise GitHubAPIError(f"Expected workflow run object in response for {url}")
    return first_run


def gha_query_recent_branch_commits(
    github_repository_name: str = "ROCm/TheRock",
    branch: str = "main",
    max_count: int = 50,
) -> list[str]:
    if max_count > 100:
        _log(
            f"Warning: max_count of {max_count} commits to query exceeds "
            "API per_page limit of 100"
        )

    url = (
        f"https://api.github.com/repos/{github_repository_name}/commits"
        f"?sha={branch}&per_page={max_count}"
    )
    response = gha_send_request(url)
    if not isinstance(response, list):
        raise GitHubAPIError(f"Expected commits list in response for {url}")

    result: list[str] = []
    for commit in response:
        if not isinstance(commit, dict) or not isinstance(commit.get("sha"), str):
            raise GitHubAPIError(
                f"Expected commit object with SHA in response for {url}"
            )
        result.append(commit["sha"])
    return result


def gha_resolve_git_ref(github_repository: str, ref: str) -> str:
    encoded_ref = quote(ref, safe="")
    url = f"https://api.github.com/repos/{github_repository}/commits/{encoded_ref}"
    response = _expect_dict(gha_send_request(url), url)
    sha = response.get("sha")
    if not isinstance(sha, str):
        raise GitHubAPIError(f"Expected SHA in response for {url}")
    return sha


def gha_fetch_file_contents(github_repository: str, path: str, ref: str) -> bytes:
    encoded_path = quote(path, safe="/")
    encoded_ref = quote(ref, safe="")
    url = (
        f"https://api.github.com/repos/{github_repository}/contents/"
        f"{encoded_path}?ref={encoded_ref}"
    )
    response = _expect_dict(gha_send_request(url), url)
    if response.get("type") != "file":
        raise GitHubAPIError(
            f"Expected GitHub contents response for a file at {path!r}"
        )
    content = response.get("content")
    if response.get("encoding") != "base64" or not isinstance(content, str):
        raise GitHubAPIError(
            f"Expected base64 GitHub contents response for {path!r}; "
            "use the Git blobs API for larger files"
        )
    try:
        return base64.b64decode(content)
    except binascii.Error as e:
        raise GitHubAPIError(f"Failed to decode GitHub contents for {path!r}") from e


def gha_fetch_text_file_contents(
    github_repository: str, path: str, ref: str, *, encoding: str = "utf-8"
) -> str:
    contents = gha_fetch_file_contents(github_repository, path, ref)
    try:
        return contents.decode(encoding)
    except UnicodeDecodeError as e:
        raise GitHubAPIError(
            f"Failed to decode GitHub contents for {path!r} as {encoding}"
        ) from e
