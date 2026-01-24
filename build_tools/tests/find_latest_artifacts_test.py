import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from find_latest_artifacts import find_latest_artifacts
from github_actions.github_actions_utils import is_authenticated_github_api_available


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
# 1. get_recent_branch_commits_via_api() — Mocked to return a fixed list of
#    known commit SHAs. This avoids dependence on the evolving tip of any
#    branch and lets us control which commits are "searched".
#
# 2. check_if_artifacts_exist() — Mocked because S3 artifacts are subject to
#    a retention policy and may be deleted for older runs. By controlling this
#    mock's return value per-commit, we simulate scenarios like flaky builds
#    where some commits are missing artifacts for a given artifact group.
#
# The GitHub API calls within find_artifacts_for_commit() (querying workflow
# runs by commit SHA, retrieving bucket info) are NOT mocked — they hit the
# real API. The pinned commits below have stable workflow run history that is
# unlikely to change. If tests become brittle, we can re-evaluate.

# Known commits with CI workflow runs in ROCm/TheRock:
#   https://github.com/ROCm/TheRock/commit/77f0cb2112d1d0aaae0de6088a6e4337f2488233
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20083647898
THEROCK_MAIN_COMMIT = "77f0cb2112d1d0aaae0de6088a6e4337f2488233"

#   https://github.com/ROCm/TheRock/commit/62bc1eaa02e6ad1b49a718eed111cf4c9f03593a
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20384488184
#   (PR from fork: ScottTodd/TheRock)
THEROCK_FORK_COMMIT = "62bc1eaa02e6ad1b49a718eed111cf4c9f03593a"


class FindLatestArtifactsTest(unittest.TestCase):
    """Tests for find_latest_artifacts() with real GitHub API calls."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    @mock.patch("find_latest_artifacts.get_recent_branch_commits_via_api")
    def test_returns_first_commit_with_artifacts(self, mock_commits, mock_check):
        """Returns the first commit that has artifacts."""
        mock_commits.return_value = [
            THEROCK_MAIN_COMMIT,
            THEROCK_FORK_COMMIT,
        ]

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.git_commit_sha, THEROCK_MAIN_COMMIT)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_latest_artifacts.get_recent_branch_commits_via_api")
    def test_skips_commits_missing_artifacts(self, mock_commits, mock_check):
        """Skips commits whose artifacts are missing (e.g. flaky build)."""
        mock_commits.return_value = [
            THEROCK_MAIN_COMMIT,
            THEROCK_FORK_COMMIT,
        ]

        # First commit's artifacts are missing, second commit's are present
        def check_by_commit(info):
            return info.git_commit_sha != THEROCK_MAIN_COMMIT

        mock_check.side_effect = check_by_commit

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.git_commit_sha, THEROCK_FORK_COMMIT)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist", return_value=False
    )
    @mock.patch("find_latest_artifacts.get_recent_branch_commits_via_api")
    def test_returns_none_when_all_artifacts_missing(self, mock_commits, mock_check):
        """Returns None when no commits have artifacts available."""
        mock_commits.return_value = [
            THEROCK_MAIN_COMMIT,
            THEROCK_FORK_COMMIT,
        ]

        info = find_latest_artifacts(
            artifact_group="gfx110X-all",
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
