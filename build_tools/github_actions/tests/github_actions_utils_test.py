import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import (
    GitHubAPI,
    GitHubAPIError,
    gha_query_last_successful_workflow_run,
    gha_query_workflow_run_by_id,
    gha_query_workflow_runs_for_commit,
    is_authenticated_github_api_available,
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

    # -------------------------------------------------------------------------
    # Authentication method selection tests
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Successful request tests
    # -------------------------------------------------------------------------

    def test_rest_api_success(self):
        """REST API successful request should return parsed JSON."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'{"id": 12345, "name": "test"}'
        mock_response.__enter__.return_value = mock_response

        with mock.patch("github_actions_utils.urlopen", return_value=mock_response):
            result = api.send_request("https://api.github.com/repos/test/test")

        self.assertEqual(result, {"id": 12345, "name": "test"})

    def test_gh_cli_success(self):
        """gh CLI successful request should return parsed JSON."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"id": 12345, "name": "test"}'

        with mock.patch(
            "github_actions_utils.subprocess.run", return_value=mock_result
        ):
            result = api.send_request("https://api.github.com/repos/test/test")

        self.assertEqual(result, {"id": 12345, "name": "test"})

    # -------------------------------------------------------------------------
    # gh CLI error handling tests
    # -------------------------------------------------------------------------

    def test_gh_cli_timeout_raises_github_api_error(self):
        """gh CLI timeout should raise GitHubAPIError with TimeoutExpired cause."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        with mock.patch(
            "github_actions_utils.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=10),
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("timed out", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, subprocess.TimeoutExpired)

    def test_gh_cli_oserror_raises_github_api_error(self):
        """gh CLI OSError should raise GitHubAPIError with OSError cause."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/nonexistent/gh"

        with mock.patch(
            "github_actions_utils.subprocess.run",
            side_effect=OSError("No such file or directory"),
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Failed to execute gh CLI", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_gh_cli_nonzero_exit_raises_github_api_error(self):
        """gh CLI non-zero exit should raise GitHubAPIError."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stderr = "gh: Not Found (HTTP 404)"

        with mock.patch(
            "github_actions_utils.subprocess.run", return_value=mock_result
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("gh api request failed", str(ctx.exception))
            self.assertIn("Not Found", str(ctx.exception))

    def test_gh_cli_empty_response_raises_github_api_error(self):
        """gh CLI empty response should raise GitHubAPIError."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with mock.patch(
            "github_actions_utils.subprocess.run", return_value=mock_result
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("empty response", str(ctx.exception))

    def test_gh_cli_invalid_json_raises_github_api_error(self):
        """gh CLI invalid JSON should raise GitHubAPIError with JSONDecodeError cause."""
        import json

        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json {"

        with mock.patch(
            "github_actions_utils.subprocess.run", return_value=mock_result
        ):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("invalid JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)

    # -------------------------------------------------------------------------
    # REST API error handling tests
    # -------------------------------------------------------------------------

    def test_rest_api_http_403_raises_github_api_error(self):
        """REST API 403 should raise GitHubAPIError with HTTPError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_error = HTTPError(
            url="https://api.github.com/repos/test/test",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with mock.patch("github_actions_utils.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("403", str(ctx.exception))
            self.assertIn("Access denied", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, HTTPError)

    def test_rest_api_http_404_raises_github_api_error(self):
        """REST API 404 should raise GitHubAPIError with HTTPError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_error = HTTPError(
            url="https://api.github.com/repos/test/test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        with mock.patch("github_actions_utils.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("404", str(ctx.exception))
            self.assertIn("not found", str(ctx.exception).lower())
            self.assertIsInstance(ctx.exception.__cause__, HTTPError)

    def test_rest_api_http_500_raises_github_api_error(self):
        """REST API 500 should raise GitHubAPIError with HTTPError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_error = HTTPError(
            url="https://api.github.com/repos/test/test",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )

        with mock.patch("github_actions_utils.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("500", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, HTTPError)

    def test_rest_api_network_error_raises_github_api_error(self):
        """REST API network error should raise GitHubAPIError with URLError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_error = URLError(reason="Connection refused")

        with mock.patch("github_actions_utils.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Network error", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, URLError)

    def test_rest_api_timeout_raises_github_api_error(self):
        """REST API timeout should raise GitHubAPIError with TimeoutError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        with mock.patch("github_actions_utils.urlopen", side_effect=TimeoutError()):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("timed out", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, TimeoutError)

    def test_rest_api_invalid_json_raises_github_api_error(self):
        """REST API invalid JSON should raise GitHubAPIError with JSONDecodeError cause."""
        import json

        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b"not valid json {"
        mock_response.__enter__.return_value = mock_response

        with mock.patch("github_actions_utils.urlopen", return_value=mock_response):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Invalid JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)


class GitHubActionsUtilsTest(unittest.TestCase):
    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_run_by_id(self):
        """Test querying a workflow run by its ID."""
        workflow_run = gha_query_workflow_run_by_id("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Verify fields we depend on in RunOutputRoot and find_artifacts_for_commit
        self.assertIn("id", workflow_run)
        self.assertIn("head_repository", workflow_run)
        self.assertIn("full_name", workflow_run["head_repository"])
        self.assertIn("updated_at", workflow_run)
        self.assertIn("status", workflow_run)
        self.assertIn("html_url", workflow_run)

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_run_by_id_not_found(self):
        """Test querying a workflow run by its ID where the ID is not found."""
        with self.assertRaises(Exception):
            gha_query_workflow_run_by_id("ROCm/TheRock", "00000000000")

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_runs_for_commit_found(self):
        """Test querying workflow runs for a commit that has runs."""
        # https://github.com/ROCm/TheRock/commit/77f0cb2112d1d0aaae0de6088a6e4337f2488233
        runs = gha_query_workflow_runs_for_commit(
            "ROCm/TheRock", "ci.yml", "77f0cb2112d1d0aaae0de6088a6e4337f2488233"
        )
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        # Verify fields we depend on in RunOutputRoot and find_artifacts_for_commit
        run = runs[0]
        self.assertIn("id", run)
        self.assertIn("head_repository", run)
        self.assertIn("full_name", run["head_repository"])
        self.assertIn("updated_at", run)
        self.assertIn("status", run)
        self.assertIn("html_url", run)

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_runs_for_commit_not_found(self):
        """Test querying workflow runs for a commit with no runs returns empty list."""
        runs = gha_query_workflow_runs_for_commit(
            "ROCm/TheRock", "ci.yml", "0000000000000000000000000000000000000000"
        )
        self.assertIsInstance(runs, list)
        self.assertEqual(len(runs), 0)

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
