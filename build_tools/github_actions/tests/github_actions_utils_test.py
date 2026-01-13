import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import (
    GitHubAPI,
    gha_query_last_successful_workflow_run,
    gha_query_workflow_run_information,
    is_authenticated_github_api_available,
    retrieve_bucket_info,
)


def _skip_unless_authenticated_github_api_is_available(test_func):
    """Decorator to skip tests unless GitHub API is available.

    Checks for GITHUB_TOKEN env var or authenticated gh CLI.
    """
    return unittest.skipUnless(
        is_authenticated_github_api_available(),
        "No authenticated GitHub API auth available (need GITHUB_TOKEN or authenticated gh CLI)",
    )(test_func)


class GitHubAPITest(unittest.TestCase):
    """Tests for GitHubAPI class."""

    def setUp(self):
        # Save and clear GITHUB_TOKEN
        self._saved_token = os.environ.get("GITHUB_TOKEN")
        if "GITHUB_TOKEN" in os.environ:
            del os.environ["GITHUB_TOKEN"]

    def tearDown(self):
        # Restore GITHUB_TOKEN
        if self._saved_token is not None:
            os.environ["GITHUB_TOKEN"] = self._saved_token
        elif "GITHUB_TOKEN" in os.environ:
            del os.environ["GITHUB_TOKEN"]

    def test_github_token_takes_priority(self):
        """GITHUB_TOKEN should be used when available, even if gh CLI is present."""
        os.environ["GITHUB_TOKEN"] = "test-token-12345"

        # Mock gh CLI as available and authenticated
        mock_result = mock.Mock()
        mock_result.returncode = 0

        with mock.patch(
            "github_actions_utils.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_utils.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(api.get_auth_method(), GitHubAPI.AuthMethod.GITHUB_TOKEN)

    def test_gh_cli_used_when_no_token(self):
        """gh CLI should be used when GITHUB_TOKEN is not set and gh is authenticated."""
        # Mock gh CLI as available and authenticated
        mock_result = mock.Mock()
        mock_result.returncode = 0

        with mock.patch(
            "github_actions_utils.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_utils.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(api.get_auth_method(), GitHubAPI.AuthMethod.GH_CLI)

    def test_gh_cli_not_authenticated(self):
        """Should fall back to unauthenticated when gh CLI is not logged in."""
        # Mock gh CLI as available but not authenticated (non-zero return code)
        mock_result = mock.Mock()
        mock_result.returncode = 1

        with mock.patch(
            "github_actions_utils.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_utils.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(
                api.get_auth_method(), GitHubAPI.AuthMethod.UNAUTHENTICATED
            )

    def test_unauthenticated_fallback(self):
        """Should fall back to unauthenticated when no auth is available."""
        with mock.patch("github_actions_utils.shutil.which", return_value=None):
            api = GitHubAPI()
            self.assertEqual(
                api.get_auth_method(), GitHubAPI.AuthMethod.UNAUTHENTICATED
            )

    def test_auth_method_is_cached(self):
        """Auth method should be cached after first call to get_auth_method()."""
        os.environ["GITHUB_TOKEN"] = "test-token-12345"
        api = GitHubAPI()
        first_result = api.get_auth_method()

        # Change env, but cached result should persist
        del os.environ["GITHUB_TOKEN"]
        second_result = api.get_auth_method()

        self.assertEqual(first_result, second_result)
        self.assertEqual(second_result, GitHubAPI.AuthMethod.GITHUB_TOKEN)

    def test_fresh_instance_detects_new_env(self):
        """A new GitHubAPI instance should detect changed environment."""
        os.environ["GITHUB_TOKEN"] = "test-token-12345"
        api1 = GitHubAPI()
        self.assertEqual(api1.get_auth_method(), GitHubAPI.AuthMethod.GITHUB_TOKEN)

        # New instance with no token should detect unauthenticated
        del os.environ["GITHUB_TOKEN"]
        with mock.patch("github_actions_utils.shutil.which", return_value=None):
            api2 = GitHubAPI()
            self.assertEqual(
                api2.get_auth_method(), GitHubAPI.AuthMethod.UNAUTHENTICATED
            )

    def test_is_authenticated_with_token(self):
        """is_authenticated should return True with GITHUB_TOKEN."""
        os.environ["GITHUB_TOKEN"] = "test-token-12345"
        api = GitHubAPI()
        self.assertTrue(api.is_authenticated())

    def test_is_authenticated_without_auth(self):
        """is_authenticated should return False without any auth."""
        with mock.patch("github_actions_utils.shutil.which", return_value=None):
            api = GitHubAPI()
            self.assertFalse(api.is_authenticated())

    def test_get_auth_method_returns_enum(self):
        """get_auth_method should return a GitHubAPI.AuthMethod enum."""
        api = GitHubAPI()
        auth_method = api.get_auth_method()
        self.assertIsInstance(auth_method, GitHubAPI.AuthMethod)


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

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_run_information(self):
        workflow_run = gha_query_workflow_run_information("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Useful for debugging
        # import json
        # print(json.dumps(workflow_run, indent=2))

    @_skip_unless_authenticated_github_api_is_available
    def test_retrieve_bucket_info(self):
        # TODO(geomin12): work on pulling these run IDs more dynamically
        # https://github.com/ROCm/TheRock/actions/runs/18022609292?pr=1597
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18022609292")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-artifacts")

    @_skip_unless_authenticated_github_api_is_available
    def test_retrieve_newer_bucket_info(self):
        # https://github.com/ROCm/TheRock/actions/runs/19680190301
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "19680190301")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    @_skip_unless_authenticated_github_api_is_available
    def test_retrieve_bucket_info_from_fork(self):
        # https://github.com/ROCm/TheRock/actions/runs/18023442478?pr=1596
        external_repo, bucket = retrieve_bucket_info("ROCm/TheRock", "18023442478")
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @_skip_unless_authenticated_github_api_is_available
    def test_retrieve_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/18020401326?pr=1828
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "18020401326"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-artifacts-external")

    @_skip_unless_authenticated_github_api_is_available
    def test_retrieve_newer_bucket_info_from_rocm_libraries(self):
        # https://github.com/ROCm/rocm-libraries/actions/runs/19784318631
        external_repo, bucket = retrieve_bucket_info(
            "ROCm/rocm-libraries", "19784318631"
        )
        self.assertEqual(external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

    @_skip_unless_authenticated_github_api_is_available
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

    @_skip_unless_authenticated_github_api_is_available
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


if __name__ == "__main__":
    unittest.main()
