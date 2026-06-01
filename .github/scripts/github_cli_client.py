#!/usr/bin/env python3

"""
GitHub Client Utility
---------------------
This utility provides a GitHubClient class that wraps GitHub REST API operations
used across automation scripts, such as retrieving pull request file changes and labels.

When doing manual testing, you can run the same REST API calls through curl in the terminal.
These REST API URLs, without the authentication header, will be output by the debug logging.

This includes:
- Fetching PR details
- Creating PRs
- Closing PRs

Requirements:
    - NOTE: GH_TOKEN environment variable hands authentication token to this script in a runner.
    - The token is created by the GitHub App and is passed to the script via the environment variable.

Manual curl testing:

To fetch PR details:
  curl -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/OWNER/REPO/pulls/NUMBER

To list PRs by head branch:
  curl -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/OWNER/REPO/pulls?head=OWNER:branch-name&state=open"

To fetch changed files in a PR:
  curl -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/OWNER/REPO/pulls/NUMBER/files

To create a PR:
  curl -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/OWNER/REPO/pulls \
    -d '{"title":"Title","body":"Description","head":"branch-name","base":"main"}'

To apply labels:
  curl -X POST -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/OWNER/REPO/issues/NUMBER/labels \
    -d '{"labels": ["bug", "needs-review"]}'
"""

import os
import requests
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GitHubCLIClient:
    def __init__(self) -> None:
        """Initialize the GitHub API client using GitHub App authentication."""
        self.api_url = "https://api.github.com"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self._get_token()}",
                "Accept": "application/vnd.github+json",
            }
        )

    def _get_token(self) -> str:
        """Helper method to retrieve the GitHub token from environment variable."""
        token = os.getenv("GH_TOKEN")
        if not token:
            raise EnvironmentError("GH_TOKEN environment variable is not set")
        return token

    def _get_with_retries(
        self,
        url: str,
        error_msg: str,
        retries: int = 3,
        backoff: int = 2,
        timeout: int = 10,
    ) -> Optional[requests.Response]:
        """Internal helper to retry a GET request with exponential backoff."""
        # no logging the actual request to avoid leaking sensitive information
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response
                # for api rate limiting, we check the headers for remaining requests and reset time
                elif (
                    response.status_code == 403
                    and response.headers.get("X-RateLimit-Remaining") == "0"
                ):
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    sleep_seconds = max(1, reset_time - int(time.time()) + 1)
                    logger.warning(
                        f"Rate limited. Sleeping for {sleep_seconds} seconds..."
                    )
                    time.sleep(sleep_seconds)
                    continue
                # other errors will use exponential backoff timeout
                elif response.status_code in {403, 429, 500, 502, 503, 504}:
                    logger.warning(
                        f"Retryable error {response.status_code} on attempt {attempt}."
                    )
                else:
                    response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt}: {e}")
            logger.error(f"{error_msg} for {url} (Attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(backoff**attempt)  # Exponential backoff
            else:
                logger.error(f"Max retries reached for GET at {url}. Giving up.")
        return None

    def _get_json(self, url: str, error_msg: str) -> dict:
        """Helper method to perform a simple GET request and return a single JSON object."""
        response = self._get_with_retries(url, error_msg)
        return response.json() if response else {}

    def _get_paginated_json(self, url: str, error_msg: str) -> List[dict]:
        """Helper method to perform a sequence of GET requests with pagination."""
        results = []
        while url:
            response = self._get_with_retries(url, error_msg)
            if not response:
                return results
            results.extend(response.json())
            url = response.links.get("next", {}).get("url")
        return results

    def _request_json(
        self,
        method: str,
        url: str,
        json: Optional[dict] = None,
        error_msg: str = "",
        retries: int = 3,
        backoff: int = 2,
    ) -> dict:
        """Helper method to perform a request with retries and return JSON response."""
        # no logging the actual request to avoid leaking sensitive information
        for attempt in range(retries):
            response = self.session.request(method, url, json=json)
            if response.ok:
                if response.status_code == 204 or not response.text.strip():
                    return {}  # DELETE requests have no json content
                else:
                    return response.json()
            else:
                # for api rate limiting, we check the headers for remaining requests and reset time
                if (
                    response.status_code == 403
                    and response.headers.get("X-RateLimit-Remaining") == "0"
                ):
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    sleep_seconds = max(1, reset_time - int(time.time()) + 1)
                    logger.warning(
                        f"Rate limited. Sleeping for {sleep_seconds} seconds..."
                    )
                    time.sleep(sleep_seconds)
                # other errors will use exponential backoff timeout
                else:
                    logger.error(
                        f"{error_msg} for method {method} at {url} (Attempt {attempt + 1}/{retries})"
                    )
                    if attempt < retries - 1:
                        time.sleep(backoff**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Max retries reached for method {method} at {url}. Giving up."
                        )
                        return {}

    def get_changed_files(self, repo: str, pr: int) -> List[str]:
        """Fetch the changed files in a pull request using GitHub API."""
        url = f"{self.api_url}/repos/{repo}/pulls/{pr}/files?per_page=50"
        logger.debug(f"Request URL: {url}")
        files_data = self._get_paginated_json(
            url, f"Failed to fetch files for PR #{pr} in {repo}"
        )
        files = [file["filename"] for file in files_data]
        logger.debug(f"Changed files in PR #{pr}: {files}")
        return files

    def get_defined_labels(self, repo: str) -> List[str]:
        """Get all labels defined in the given repository."""
        url = f"{self.api_url}/repos/{repo}/labels?per_page=100"
        logger.debug(f"Request URL: {url}")
        labels_data = self._get_paginated_json(
            url, f"Failed to fetch labels from {repo}"
        )
        labels = [label["name"] for label in labels_data]
        logger.debug(f"Defined labels in {repo}: {labels}")
        return labels

    def get_existing_labels_on_pr(self, repo: str, pr: int) -> List[str]:
        """Fetch current labels on a PR."""
        url = f"{self.api_url}/repos/{repo}/issues/{pr}/labels?per_page=100"
        logger.debug(f"Request URL: {url}")
        labels_data = self._get_paginated_json(
            url, f"Failed to fetch labels for PR #{pr} in {repo}"
        )
        labels = [label["name"] for label in labels_data]
        logger.debug(f"Existing labels on PR #{pr}: {labels}")
        return labels

    def pr_view(self, repo: str, head: str) -> Optional[int]:
        """Check if a PR exists for the given repo and branch."""
        # This is similar to get_pr_by_head_branch but returns only the PR number directly
        url = f"{self.api_url}/repos/{repo}/pulls?head={repo.split('/')[0]}:{head}&per_page=100"
        logger.debug(f"Request URL: {url}")
        result = self._get_paginated_json(
            url, f"Failed to retrieve PR for head branch {head} in repo {repo}"
        )
        return result[0]["number"] if result else None

    def get_pr_by_head_branch(self, repo: str, head: str) -> Optional[dict]:
        """Fetch the PR object for a given head branch in a repository, if it exists."""
        # This is similar to pr_view but returns the full PR object
        url = f"{self.api_url}/repos/{repo}/pulls?head={repo.split('/')[0]}:{head}&state=open&per_page=100"
        logger.debug(f"Request URL: {url}")
        data = self._get_paginated_json(
            url, f"Failed to get PRs for {repo} with head {head}"
        )
        return data[0] if data else None

    def get_pr_by_number(self, repo: str, pr_number: int) -> Optional[dict]:
        """Fetch the PR object for a given PR number in a repository."""
        url = f"{self.api_url}/repos/{repo}/pulls/{pr_number}"
        logger.debug(f"Fetching PR #{pr_number} from {repo}")
        response = self._get_json(url, f"Failed to get PR #{pr_number} from {repo}")
        return response

    def pr_create(
        self,
        repo: str,
        base: str,
        head: str,
        title: str,
        body: str,
        dry_run: bool = False,
    ) -> None:
        """Create a new pull request."""
        url = f"{self.api_url}/repos/{repo}/pulls"
        payload = {"title": title, "body": body, "head": head, "base": base}
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request Payload: {payload}")
        if dry_run:
            logger.info(
                f"Dry run: The pull request would be created from {head} to {base} in {repo}"
            )
            return
        self._request_json(
            "POST", url, payload, f"Failed to create PR from {head} to {base} in {repo}"
        )
        logger.info(f"Created PR from {head} to {base} in {repo}.")

    def close_pr_and_delete_branch(
        self, repo: str, pr_number: int, dry_run: bool = False
    ) -> None:
        """Close a pull request and delete the associated branch using the GitHub API."""
        pr_url = f"{self.api_url}/repos/{repo}/pulls/{pr_number}"
        logger.debug(f"Request URL: {pr_url}")
        pr_data = self._get_json(pr_url, f"Failed to fetch PR #{pr_number} in {repo}")
        head_ref = pr_data.get("head", {}).get("ref")
        if not head_ref:
            logger.error(
                f"Could not determine head branch for PR #{pr_number} in {repo}"
            )
            return
        logger.debug(f"PR #{pr_number} head branch: {head_ref}")
        close_payload = {"state": "closed"}
        logger.debug(f"Request Payload: {close_payload}")
        if dry_run:
            logger.info(
                f"Dry run: The pull request #{pr_number} would be closed and the branch '{head_ref}' would be deleted in repo '{repo}'"
            )
            return
        self._request_json(
            "PATCH", pr_url, close_payload, f"Failed to close PR #{pr_number} in {repo}"
        )
        branch_url = f"{self.api_url}/repos/{repo}/git/refs/heads/{head_ref}"
        logger.debug(f"Branch DELETE URL: {branch_url}")
        self._request_json(
            "DELETE",
            branch_url,
            None,
            f"Failed to delete branch '{head_ref}' for PR #{pr_number}",
        )
        logger.info(
            f"Closed pull request #{pr_number} and deleted the branch '{head_ref}' in {repo}."
        )

    def sync_labels(
        self, target_repo: str, pr_number: int, labels: List[str], dry_run: bool = False
    ) -> None:
        """Sync labels from the source repo to the target repo (only apply existing labels)."""
        url = f"{self.api_url}/repos/{target_repo}/labels?per_page=100"
        logger.debug(f"Request URL: {url}")
        target_repo_labels = {
            label["name"]
            for label in self._get_paginated_json(
                url, f"Failed to fetch labels for {target_repo}"
            )
        }
        labels_set = set(labels)
        labels_to_apply = labels_set & target_repo_labels
        labels_for_logging = ",".join(labels_to_apply)
        if labels_to_apply:
            # note: using issues endpoint for labels as PRs are a subset of issues
            url = f"{self.api_url}/repos/{target_repo}/issues/{pr_number}/labels"
            payload = {"labels": list(labels_to_apply)}
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Request Payload: {payload}")
            if not dry_run:
                self._request_json(
                    "POST",
                    url,
                    payload,
                    f"Failed to apply labels to PR #{pr_number} in {target_repo}",
                )
                logger.info(
                    f"Applied labels '{labels_for_logging}' to PR #{pr_number} in {target_repo}."
                )
            else:
                logger.info(
                    f"Dry run: Labels '{labels_for_logging}' would be applied to PR #{pr_number} in {target_repo}."
                )
        else:
            logger.info(
                f"No valid labels to apply to PR #{pr_number} in {target_repo}."
            )

    def get_squash_merge_commit(self, repo: str, pr_number: int) -> Optional[str]:
        """Get the squash merge commit SHA of a merged pull request."""
        url = f"{self.api_url}/repos/{repo}/pulls/{pr_number}"
        logger.debug(f"Request URL: {url}")
        data = self._get_json(url, f"Failed to fetch PR #{pr_number} from {repo}")
        if not data:
            logger.error(f"No data returned for PR #{pr_number}")
            return None
        if data.get("merged") and data.get("merge_commit_sha"):
            logger.debug(f"PR #{pr_number} merged commit: {data['merge_commit_sha']}")
            return data["merge_commit_sha"]
        logger.warning(f"PR #{pr_number} is not merged or missing merge commit SHA.")
        return None

    def get_user(self, username: str) -> tuple[str, str]:
        """Fetch the name and email of a GitHub user. Falls back to login and no-reply email."""
        url = f"{self.api_url}/users/{username}"
        logger.debug(f"Fetching user profile for @{username}")
        data = self._get_json(url, f"Failed to fetch user profile for @{username}")
        name = data.get("name") or username
        email = data.get("email")
        if not email:
            user_id = data.get("id")
            if user_id:
                email = f"{user_id}+{username}@users.noreply.github.com"
            else:
                email = f"{username}@users.noreply.github.com"
        return name, email
