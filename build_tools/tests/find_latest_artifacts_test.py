import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from find_latest_artifacts import find_latest_artifacts
from github_actions.github_actions_utils import (
    GitHubAPIError,
    is_authenticated_github_api_available,
)


def _skip_unless_authenticated_github_api_is_available(test_func):
    """Decorator to skip tests unless GitHub API is available."""
    return unittest.skipUnless(
        is_authenticated_github_api_available(),
        "No authenticated GitHub API available (need GITHUB_TOKEN or authenticated gh CLI)",
    )(test_func)


# --- Mocking strategy ---
#
# These tests mock two layers:
#
# 1. gha_query_recent_branch_commits() — Mocked to return a fixed list of
#    known commit SHAs. This avoids dependence on the evolving tip of any
#    branch and lets us control which commits are "searched".
#
# 2. check_if_artifacts_exist() — Mocked because S3 artifacts are subject to
#    a retention policy and may be deleted for older runs. By controlling this
#    mock's return value per-commit, we can test commits/workflows that have
#    already expired and we can also simulate missing artifacts due to failed
#    builds.
#
# The GitHub API calls within find_artifacts_for_commit() (querying workflow
# runs by commit SHA, retrieving bucket info) are NOT mocked — they hit the
# real API. The pinned commits below have stable workflow run history that is
# unlikely to change. If tests become brittle, we can re-evaluate.

# Two consecutive commits on TheRock main with CI workflow runs, simulating
# what gha_query_recent_branch_commits() would return (most recent first).
#
#   https://github.com/ROCm/TheRock/commit/5ea91c38d19237716ba0c9382928da12a6fc9b08
#   CI run: https://github.com/ROCm/TheRock/actions/runs/21249928112
TEST_THEROCK_COMMIT_NEWER = "5ea91c38d19237716ba0c9382928da12a6fc9b08"
#   https://github.com/ROCm/TheRock/commit/02946b2295f8fae31fd506c1be6735b5911cdc6b
#   CI run: https://github.com/ROCm/TheRock/actions/runs/21243829022
TEST_THEROCK_COMMIT_OLDER = "02946b2295f8fae31fd506c1be6735b5911cdc6b"


class FindLatestArtifactsTest(unittest.TestCase):
    """Tests for find_latest_artifacts() with real GitHub API calls."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_returns_first_commit_with_artifacts(self, mock_commits, mock_check):
        """Returns the first commit that has artifacts."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.git_commit_sha, TEST_THEROCK_COMMIT_NEWER)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_skips_commits_missing_artifacts(self, mock_commits, mock_check):
        """Skips commits whose artifacts are missing (e.g. flaky build)."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        # First commit's artifacts are missing, second commit's are present
        def check_by_commit(info):
            return info.git_commit_sha != TEST_THEROCK_COMMIT_NEWER

        mock_check.side_effect = check_by_commit

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.git_commit_sha, TEST_THEROCK_COMMIT_OLDER)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist", return_value=False
    )
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_returns_none_when_all_artifacts_missing(self, mock_commits, mock_check):
        """Returns None when no commits have artifacts available."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNone(info)

    def test_rate_limit_error_on_commit_list_raises_exception(self):
        """Rate limit when fetching commits raises GitHubAPIError."""
        rate_limit_error = GitHubAPIError(
            "GitHub API rate limit exceeded. "
            "Authenticate with `gh auth login` or set GITHUB_TOKEN to increase limits."
        )

        with mock.patch(
            "find_latest_artifacts.gha_query_recent_branch_commits",
            side_effect=rate_limit_error,
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                find_latest_artifacts(
                    artifact_group="gfx110X-all",
                    github_repository_name="ROCm/TheRock",
                )

            self.assertIn("rate limit", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
