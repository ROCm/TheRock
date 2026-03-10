# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from find_artifacts_for_commit import (
    ArtifactRunInfo,
    find_artifacts_for_commit,
)
from _therock_utils.workflow_outputs import WorkflowOutputRoot
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
# These tests make real GitHub API calls to query workflow run metadata, but
# mock check_if_artifacts_exist() which does HTTP HEAD requests to S3. This is
# because:
#
# 1. S3 retention: Artifacts will be subject to a retention policy, so older
#    runs' artifacts may be deleted. Mocking the S3 check avoids false failures
#    when artifacts are cleaned up.
#
# 2. Workflow run stability: The GitHub API workflow run history for these
#    pinned commits is unlikely to change (runs probably won't be re-triggered
#    or deleted for old commits). If tests become brittle we can re-evaluate.

# Known commits with CI workflow runs in ROCm/TheRock:
#   https://github.com/ROCm/TheRock/commit/77f0cb2112d1d0aaae0de6088a6e4337f2488233
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20083647898
TEST_THEROCK_MAIN_COMMIT = "77f0cb2112d1d0aaae0de6088a6e4337f2488233"
TEST_THEROCK_MAIN_RUN_ID = "20083647898"

#   https://github.com/ROCm/TheRock/commit/62bc1eaa02e6ad1b49a718eed111cf4c9f03593a
#   CI run: https://github.com/ROCm/TheRock/actions/runs/20384488184
#   (PR from fork: ScottTodd/TheRock)
#   (attribution is fuzzy here, since branches from forks are often deleted,
#    we really just want to test that therock-ci-artifacts-external is used)
TEST_THEROCK_FORK_COMMIT = "62bc1eaa02e6ad1b49a718eed111cf4c9f03593a"
TEST_THEROCK_FORK_RUN_ID = "20384488184"

# Known commit with CI workflow run in ROCm/rocm-libraries:
#   https://github.com/ROCm/rocm-libraries/commit/ab692342ac4d00268ac8a5a4efbc144c194cb45a
#   CI run: https://github.com/ROCm/rocm-libraries/actions/runs/21365647639
TEST_ROCM_LIBRARIES_COMMIT = "ab692342ac4d00268ac8a5a4efbc144c194cb45a"
TEST_ROCM_LIBRARIES_RUN_ID = "21365647639"


class FindArtifactsForCommitTest(unittest.TestCase):
    """Tests for find_artifacts_for_commit() with real GitHub API calls."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_therock_main_commit(self, mock_check):
        """Known main commit returns ArtifactRunInfo with correct metadata."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIsInstance(info, ArtifactRunInfo)
        self.assertEqual(info.git_commit_sha, TEST_THEROCK_MAIN_COMMIT)
        self.assertEqual(info.github_repository_name, "ROCm/TheRock")
        self.assertEqual(info.workflow_file_name, "ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_MAIN_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts")
        self.assertEqual(info.external_repo, "")
        self.assertEqual(info.platform, "linux")
        self.assertEqual(info.artifact_group, "gfx110X-all")

        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_therock_fork_commit(self, mock_check):
        """Fork commit returns ArtifactRunInfo with external bucket."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_FORK_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertEqual(info.workflow_run_id, TEST_THEROCK_FORK_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts-external")
        self.assertEqual(info.external_repo, "ROCm-TheRock/")

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch(
        "find_artifacts_for_commit.check_if_artifacts_exist", return_value=False
    )
    def test_commit_with_runs_but_no_artifacts(self, mock_check):
        """Commit with workflow runs but no S3 artifacts returns empty list."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(results, [])
        mock_check.assert_called()

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_platform_windows(self, mock_check):
        """Check that we can find artifacts for Windows as well as Linux."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="windows",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertEqual(info.platform, "windows")
        self.assertIn("windows", info.s3_path)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_rocm_libraries_commit(self, mock_check):
        """rocm-libraries commit uses therock-ci.yml and external bucket."""
        results = find_artifacts_for_commit(
            commit=TEST_ROCM_LIBRARIES_COMMIT,
            artifact_groups=["gfx94X-dcgpu"],
            github_repository_name="ROCm/rocm-libraries",
            workflow_file_name="therock-ci.yml",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        info = results[0]
        self.assertIsInstance(info, ArtifactRunInfo)
        self.assertEqual(info.git_commit_sha, TEST_ROCM_LIBRARIES_COMMIT)
        self.assertEqual(info.github_repository_name, "ROCm/rocm-libraries")
        self.assertEqual(info.workflow_file_name, "therock-ci.yml")
        self.assertEqual(info.workflow_run_id, TEST_ROCM_LIBRARIES_RUN_ID)
        self.assertEqual(info.s3_bucket, "therock-ci-artifacts-external")
        self.assertEqual(info.external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(info.platform, "linux")
        self.assertEqual(info.artifact_group, "gfx94X-dcgpu")

        mock_check.assert_called()

    def test_rate_limit_error_raises_exception(self):
        """Rate limit errors raise GitHubAPIError (not silently return None)."""
        rate_limit_error = GitHubAPIError(
            "GitHub API rate limit exceeded. "
            "Authenticate with `gh auth login` or set GITHUB_TOKEN to increase limits."
        )

        with mock.patch(
            "find_artifacts_for_commit.gha_query_workflow_runs_for_commit",
            side_effect=rate_limit_error,
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                find_artifacts_for_commit(
                    commit="abc123",
                    artifact_groups=["gfx110X-all"],
                    github_repository_name="ROCm/TheRock",
                )

            self.assertIn("rate limit", str(ctx.exception).lower())


class FindArtifactsForCommitMultiGroupTest(unittest.TestCase):
    """Tests for multi-group behavior of find_artifacts_for_commit()."""

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multiple_groups_all_found(self, mock_check):
        """All requested groups are returned when all have artifacts."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")
        self.assertEqual(results[1].artifact_group, "gfx120X-all")
        # Both should come from the same workflow run
        self.assertEqual(results[0].workflow_run_id, results[1].workflow_run_id)

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    def test_multiple_groups_partial(self, mock_check):
        """Only groups with artifacts are returned (partial result)."""

        def only_gfx110x(info):
            return info.artifact_group == "gfx110X-all"

        mock_check.side_effect = only_gfx110x

        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")

    @_skip_unless_authenticated_github_api_is_available
    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist", return_value=True)
    def test_multiple_groups_preserves_requested_order(self, mock_check):
        """Results are returned in the same order as requested."""
        results = find_artifacts_for_commit(
            commit=TEST_THEROCK_MAIN_COMMIT,
            artifact_groups=["gfx120X-all", "gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx120X-all")
        self.assertEqual(results[1].artifact_group, "gfx110X-all")


class FindArtifactsCrossRunTest(unittest.TestCase):
    """Tests for cross-run artifact accumulation (fully mocked, no API).

    When a workflow is retriggered for the same commit (e.g. manual re-dispatch),
    GitHub creates a new workflow run with a distinct run ID. The API returns
    all runs for the commit, most recent first. These tests verify that
    find_artifacts_for_commit() accumulates groups across those distinct runs.

    Note: This is different from "re-run failed jobs" which creates a new
    *attempt* under the same run ID. Since attempts share a run ID (and thus
    the same S3 path), cross-attempt accumulation happens implicitly.
    """

    # Fake workflow runs representing two retriggered CI runs for the same
    # commit, ordered most-recent-first (as the GitHub API returns them).
    FAKE_RUN_NEWER = {
        "id": 99999999902,
        "status": "completed",
        "conclusion": "failure",
        "html_url": "https://github.com/ROCm/TheRock/actions/runs/99999999902",
    }
    FAKE_RUN_OLDER = {
        "id": 99999999901,
        "status": "completed",
        "conclusion": "success",
        "html_url": "https://github.com/ROCm/TheRock/actions/runs/99999999901",
    }

    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_accumulates_groups_across_runs(
        self, mock_query_runs, mock_from_wfr, mock_check
    ):
        """Groups found across different retriggered runs are accumulated."""
        mock_query_runs.return_value = [self.FAKE_RUN_NEWER, self.FAKE_RUN_OLDER]
        mock_from_wfr.return_value = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="0",
            platform="linux",
        )

        # Newer run only built gfx120X; older run built gfx110X.
        def check_by_run_and_group(info):
            if info.workflow_run_id == str(self.FAKE_RUN_NEWER["id"]):
                return info.artifact_group == "gfx120X-all"
            if info.workflow_run_id == str(self.FAKE_RUN_OLDER["id"]):
                return info.artifact_group == "gfx110X-all"
            return False

        mock_check.side_effect = check_by_run_and_group

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx110X-all", "gfx120X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].artifact_group, "gfx110X-all")
        self.assertEqual(results[1].artifact_group, "gfx120X-all")
        # Each group comes from a different run
        self.assertEqual(results[0].workflow_run_id, str(self.FAKE_RUN_OLDER["id"]))
        self.assertEqual(results[1].workflow_run_id, str(self.FAKE_RUN_NEWER["id"]))

    @mock.patch("find_artifacts_for_commit.check_if_artifacts_exist")
    @mock.patch("find_artifacts_for_commit.WorkflowOutputRoot.from_workflow_run")
    @mock.patch("find_artifacts_for_commit.gha_query_workflow_runs_for_commit")
    def test_newer_run_takes_priority(self, mock_query_runs, mock_from_wfr, mock_check):
        """When multiple retriggered runs have the same group, the newer wins."""
        mock_query_runs.return_value = [self.FAKE_RUN_NEWER, self.FAKE_RUN_OLDER]
        mock_from_wfr.return_value = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="0",
            platform="linux",
        )
        mock_check.return_value = True  # both runs have all groups

        results = find_artifacts_for_commit(
            commit="abc123",
            artifact_groups=["gfx110X-all"],
            github_repository_name="ROCm/TheRock",
            platform="linux",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].workflow_run_id, str(self.FAKE_RUN_NEWER["id"]))


if __name__ == "__main__":
    unittest.main()
