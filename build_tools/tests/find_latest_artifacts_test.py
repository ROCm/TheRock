# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from find_latest_artifacts import (
    find_latest_artifacts,
    main,
)
from github_actions.github_actions_api import (
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

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].git_commit_sha, TEST_THEROCK_COMMIT_NEWER)

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

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].git_commit_sha, TEST_THEROCK_COMMIT_OLDER)

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

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNone(results)

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
                    artifact_groups=["gfx110X-all"],
                    github_repository_name="ROCm/TheRock",
                )

            self.assertIn("rate limit", str(ctx.exception).lower())


class FindLatestArtifactsMultiGroupTest(unittest.TestCase):
    """Tests for multi-group behavior of find_latest_artifacts()."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_multiple_groups_all_found(self, mock_commits, mock_check):
        """Returns results when all requested groups have artifacts."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")
        self.assertEqual(results[1].artifact_group, "gfx120X-all")
        # Both from the same commit
        self.assertEqual(results[0].git_commit_sha, results[1].git_commit_sha)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_skips_commit_with_partial_groups(self, mock_commits, mock_check):
        """Skips a commit where only some groups have artifacts, returns next."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        # Newer commit: only gfx110X has artifacts
        # Older commit: both groups have artifacts
        def check_by_commit_and_group(info):
            if info.git_commit_sha == TEST_THEROCK_COMMIT_NEWER:
                return info.artifact_group == "gfx110X-all"
            return True  # older commit has everything

        mock_check.side_effect = check_by_commit_and_group

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)
        # Should come from the older commit (newer was partial)
        self.assertEqual(results[0].git_commit_sha, TEST_THEROCK_COMMIT_OLDER)
        self.assertEqual(results[1].git_commit_sha, TEST_THEROCK_COMMIT_OLDER)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_returns_none_when_no_commit_has_all_groups(self, mock_commits, mock_check):
        """Returns None when no single commit has all requested groups."""
        mock_commits.return_value = [
            TEST_THEROCK_COMMIT_NEWER,
            TEST_THEROCK_COMMIT_OLDER,
        ]

        # Each commit only has one of the two groups
        def check_alternating(info):
            if info.git_commit_sha == TEST_THEROCK_COMMIT_NEWER:
                return info.artifact_group == "gfx110X-all"
            return info.artifact_group == "gfx120X-all"

        mock_check.side_effect = check_alternating

        results = find_latest_artifacts(
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            workflow_file_name="ci.yml",
            platform="linux",
        )

        self.assertIsNone(results)


class ExactSelectorTest(unittest.TestCase):
    @mock.patch("find_latest_artifacts.find_artifacts_for_commit")
    @mock.patch("find_latest_artifacts.gha_resolve_git_ref")
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_ref_resolves_one_exact_commit(
        self,
        mock_query_commits,
        mock_resolve_ref,
        mock_find,
    ):
        mock_resolve_ref.return_value = "resolved-sha"
        mock_find.return_value = []

        result = find_latest_artifacts(
            artifact_groups=["gfx94X-dcgpu"],
            ref="feature-branch",
        )

        self.assertIsNone(result)
        mock_resolve_ref.assert_called_once_with(
            github_repository="ROCm/TheRock",
            ref="feature-branch",
        )
        mock_query_commits.assert_not_called()
        mock_find.assert_called_once()
        self.assertEqual(
            mock_find.call_args.kwargs["commit"],
            "resolved-sha",
        )

    @mock.patch("find_latest_artifacts.find_artifacts_for_run")
    @mock.patch("find_latest_artifacts.gha_query_recent_branch_commits")
    def test_run_id_bypasses_branch_search(
        self,
        mock_query_commits,
        mock_find_run,
    ):
        mock_find_run.return_value = []

        result = find_latest_artifacts(
            artifact_groups=["gfx94X-dcgpu"],
            run_id="123456",
        )

        self.assertIsNone(result)
        mock_query_commits.assert_not_called()
        mock_find_run.assert_called_once()
        self.assertEqual(
            mock_find_run.call_args.kwargs["workflow_run_id"],
            "123456",
        )

    def test_ref_and_run_id_are_mutually_exclusive(self):
        with self.assertRaisesRegex(
            ValueError,
            "mutually exclusive",
        ):
            find_latest_artifacts(
                artifact_groups=["gfx94X-dcgpu"],
                ref="main",
                run_id="123",
            )

    @mock.patch(
        "find_latest_artifacts.find_latest_artifacts",
        return_value=None,
    )
    def test_cli_forwards_require_successful_run(
        self,
        mock_find_latest,
    ):
        exit_code = main(
            [
                "--artifact-group",
                "gfx94X-dcgpu",
                "--require-successful-run",
            ]
        )

        self.assertEqual(exit_code, 1)

        mock_find_latest.assert_called_once()
        self.assertTrue(mock_find_latest.call_args.kwargs["require_successful_run"])

    @mock.patch("find_latest_artifacts.find_latest_artifacts")
    def test_json_output_contains_only_json_on_stdout(
        self,
        mock_find_latest,
    ):
        info = mock.Mock()
        info.to_dict.return_value = {
            "workflow_run_id": "123",
        }

        def find_with_progress_output(**kwargs):
            print("lookup progress")
            return [info]

        mock_find_latest.side_effect = find_with_progress_output

        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--artifact-group",
                    "gfx94X-dcgpu",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "lookup progress\n")
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "status": "found",
                "results": [
                    {
                        "workflow_run_id": "123",
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
