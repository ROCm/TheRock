import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import *

# Note: these tests use the network and require GITHUB_TOKEN to avoid rate limits.


class GitHubActionsUtilsTest(unittest.TestCase):
    def setUp(self):
        # Save environment state
        self._saved_env = {}
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]
        # Clean environment for tests
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        # Restore environment state
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._saved_env.items():
            os.environ[key] = value

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_workflow_run_by_id(self):
        """Test querying a workflow run by its ID."""
        workflow_run = gha_query_workflow_run_by_id("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Verify fields we depend on in retrieve_bucket_info and find_artifacts_for_commit
        self.assertIn("id", workflow_run)
        self.assertIn("head_repository", workflow_run)
        self.assertIn("full_name", workflow_run["head_repository"])
        self.assertIn("updated_at", workflow_run)
        self.assertIn("status", workflow_run)
        self.assertIn("html_url", workflow_run)

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_workflow_run_by_id_not_found(self):
        """Test querying a workflow run by its ID where the ID is not found."""
        with self.assertRaises(Exception):
            gha_query_workflow_run_by_id("ROCm/TheRock", "00000000000")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_workflow_runs_for_commit_found(self):
        """Test querying workflow runs for a commit that has runs."""
        # https://github.com/ROCm/TheRock/commit/77f0cb2112d1d0aaae0de6088a6e4337f2488233
        runs = gha_query_workflow_runs_for_commit(
            "ROCm/TheRock", "ci.yml", "77f0cb2112d1d0aaae0de6088a6e4337f2488233"
        )
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        # Verify fields we depend on in retrieve_bucket_info and find_artifacts_for_commit
        run = runs[0]
        self.assertIn("id", run)
        self.assertIn("head_repository", run)
        self.assertIn("full_name", run["head_repository"])
        self.assertIn("updated_at", run)
        self.assertIn("status", run)
        self.assertIn("html_url", run)

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_workflow_runs_for_commit_not_found(self):
        """Test querying workflow runs for a commit with no runs returns empty list."""
        runs = gha_query_workflow_runs_for_commit(
            "ROCm/TheRock", "ci.yml", "0000000000000000000000000000000000000000"
        )
        self.assertIsInstance(runs, list)
        self.assertEqual(len(runs), 0)

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_gha_query_last_successful_workflow_run(self):
        """Test querying for the last successful workflow run on a branch."""
        # Test successful run found on main branch
        result = gha_query_last_successful_workflow_run(
            "ROCm/TheRock", "ci_nightly.yml", "main"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["head_branch"], "main")
        self.assertEqual(result["conclusion"], "success")
        self.assertIn("id", result)

        # Test no matching branch - should return None
        result = gha_query_last_successful_workflow_run(
            "ROCm/TheRock", "ci_nightly.yml", "nonexistent-branch-12345"
        )
        self.assertIsNone(result)

        # Test non-existent workflow - should raise an exception
        with self.assertRaises(Exception):
            gha_query_last_successful_workflow_run(
                "ROCm/TheRock", "nonexistent_workflow_12345.yml", "main"
            )

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_older_bucket_info(self):
        # TODO(geomin12): work on pulling these run IDs more dynamically
        # https://github.com/ROCm/TheRock/actions/runs/18022609292?pr=1597
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18022609292")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-artifacts")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_newer_bucket_info(self):
        # https://github.com/ROCm/TheRock/actions/runs/19680190301
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "19680190301")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_from_fork(self):
        # https://github.com/ROCm/TheRock/actions/runs/18023442478?pr=1596
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18023442478")
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/18020401326?pr=1828
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "18020401326"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_newer_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/19784318631
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "19784318631"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

    @unittest.skipUnless(
        os.getenv("GITHUB_TOKEN"),
        "GITHUB_TOKEN not set, skipping test that requires GitHub API access",
    )
    def test_retrieve_bucket_info_for_release(self):
        # https://github.com/ROCm/TheRock/actions/runs/19157864140
        os.environ["RELEASE_TYPE"] = "nightly"
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "19157864140")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-nightly-artifacts")

    def test_retrieve_bucket_info_without_workflow_id(self):
        """Test bucket info retrieval without making API calls."""
        # Test default case (no workflow_run_id, no API call)
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["IS_PR_FROM_FORK"] = "false"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

        # Test external repo case
        os.environ["GITHUB_REPOSITORY"] = "SomeOrg/SomeRepo"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "SomeOrg-SomeRepo/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

        # Test fork case
        os.environ["GITHUB_REPOSITORY"] = "ROCm/TheRock"
        os.environ["IS_PR_FROM_FORK"] = "true"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

        # Test release case
        os.environ["RELEASE_TYPE"] = "nightly"
        os.environ["IS_PR_FROM_FORK"] = "false"
        external_repo, bucket = retrieve_bucket_info()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-nightly-artifacts")

    def test_retrieve_bucket_info_with_workflow_run_skips_api_call(self):
        """Test that providing workflow_run skips the API call."""
        # Mock workflow_run data matching the structure from GitHub API
        mock_workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-12-01T12:00:00Z",  # After the bucket cutover date
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        with patch("github_actions_utils.gha_send_request") as mock_send_request, patch(
            "github_actions_utils.gha_query_workflow_run_by_id"
        ) as mock_query_by_id:
            external_repo, bucket = retrieve_bucket_info(
                github_repository="ROCm/TheRock",
                workflow_run=mock_workflow_run,
            )

            # Verify no API calls were made
            mock_send_request.assert_not_called()
            mock_query_by_id.assert_not_called()

            # Verify correct bucket info based on mock data
            self.assertEqual(external_repo, "")
            self.assertEqual(bucket, "therock-ci-artifacts")

    def test_retrieve_bucket_info_with_workflow_run_from_fork(self):
        """Test workflow_run from a fork returns external bucket."""
        mock_workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "SomeUser/TheRock"},  # Fork
            "updated_at": "2025-12-01T12:00:00Z",
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        with patch("github_actions_utils.gha_send_request") as mock_send_request, patch(
            "github_actions_utils.gha_query_workflow_run_by_id"
        ) as mock_query_by_id:
            external_repo, bucket = retrieve_bucket_info(
                github_repository="ROCm/TheRock",
                workflow_run=mock_workflow_run,
            )

            # Verify no API calls were made
            mock_send_request.assert_not_called()
            mock_query_by_id.assert_not_called()

            # Fork PRs go to external bucket with repo prefix
            self.assertEqual(external_repo, "ROCm-TheRock/")
            self.assertEqual(bucket, "therock-ci-artifacts-external")

    def test_retrieve_bucket_info_with_workflow_run_old_date(self):
        """Test workflow_run with old date returns legacy bucket."""
        mock_workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-10-01T12:00:00Z",  # Before the bucket cutover date
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        with patch("github_actions_utils.gha_send_request") as mock_send_request, patch(
            "github_actions_utils.gha_query_workflow_run_by_id"
        ) as mock_query_by_id:
            external_repo, bucket = retrieve_bucket_info(
                github_repository="ROCm/TheRock",
                workflow_run=mock_workflow_run,
            )

            # Verify no API calls were made
            mock_send_request.assert_not_called()
            mock_query_by_id.assert_not_called()

            # Old runs use legacy bucket
            self.assertEqual(external_repo, "")
            self.assertEqual(bucket, "therock-artifacts")


if __name__ == "__main__":
    unittest.main()
