# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_api import (
    GitHubAPI,
    GitHubAPIError,
    gha_append_step_summary,
    gha_fetch_file_contents,
    gha_fetch_text_file_contents,
    gha_job_summary_mirror_path,
    gha_load_github_event,
    gha_query_last_workflow_run,
    gha_query_prs_for_commit,
    gha_query_recent_branch_commits,
    gha_query_workflow_run_by_id,
    gha_query_workflow_runs_for_commit,
    gha_resolve_git_ref,
    gha_set_job_summary_output,
    gha_set_output,
    gha_update_pr_comment,
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


class GhaLoadGitHubEventTest(unittest.TestCase):
    """Tests for gha_load_github_event."""

    def test_loads_utf8_curly_quotes_from_github_event_path(self):
        """UTF-8 bytes must decode correctly (GitHub writes UTF-8 event files)."""
        payload = {"body": "“smart quotes” in PR description"}
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".json") as f:
            f.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            path = f.name
        saved = os.environ.get("GITHUB_EVENT_PATH")
        os.environ["GITHUB_EVENT_PATH"] = path
        try:
            loaded = gha_load_github_event()
            self.assertEqual(loaded["body"], "“smart quotes” in PR description")
        finally:
            os.unlink(path)
            if saved is None:
                del os.environ["GITHUB_EVENT_PATH"]
            else:
                os.environ["GITHUB_EVENT_PATH"] = saved

    def test_loads_from_github_event_path_env(self):
        """GITHUB_EVENT_PATH must be read as UTF-8."""
        payload = {"action": "opened", "number": 42}
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".json") as f:
            f.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            path = f.name
        saved = os.environ.get("GITHUB_EVENT_PATH")
        os.environ["GITHUB_EVENT_PATH"] = path
        try:
            loaded = gha_load_github_event()
            self.assertEqual(loaded["action"], "opened")
            self.assertEqual(loaded["number"], 42)
        finally:
            os.unlink(path)
            if saved is None:
                del os.environ["GITHUB_EVENT_PATH"]
            else:
                os.environ["GITHUB_EVENT_PATH"] = saved


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
            "github_actions_api.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(api.get_auth_method(), GitHubAPI.AuthMethod.GITHUB_TOKEN)

    def test_gh_cli_used_when_no_token(self):
        """gh CLI should be used when GITHUB_TOKEN is not set and gh is authenticated."""
        # Mock gh CLI as available and authenticated
        mock_result = mock.Mock()
        mock_result.returncode = 0

        with mock.patch(
            "github_actions_api.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(api.get_auth_method(), GitHubAPI.AuthMethod.GH_CLI)

    def test_gh_cli_not_authenticated(self):
        """Should fall back to unauthenticated when gh CLI is not logged in."""
        # Mock gh CLI as available but not authenticated (non-zero return code)
        mock_result = mock.Mock()
        mock_result.returncode = 1

        with mock.patch(
            "github_actions_api.shutil.which", return_value="/usr/bin/gh"
        ), mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
            api = GitHubAPI()
            self.assertEqual(
                api.get_auth_method(), GitHubAPI.AuthMethod.UNAUTHENTICATED
            )

    def test_unauthenticated_fallback(self):
        """Should fall back to unauthenticated when no auth is available."""
        with mock.patch("github_actions_api.shutil.which", return_value=None):
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
        with mock.patch("github_actions_api.shutil.which", return_value=None):
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
        with mock.patch("github_actions_api.shutil.which", return_value=None):
            api = GitHubAPI()
            self.assertFalse(api.is_authenticated())

    def test_get_auth_method_returns_enum(self):
        """get_auth_method should return a GitHubAPI.AuthMethod enum."""
        api = GitHubAPI()
        auth_method = api.get_auth_method()
        self.assertIsInstance(auth_method, GitHubAPI.AuthMethod)

    def test_explicit_github_token_skips_auto_detection(self):
        """An explicit github_token should be used without env/gh-CLI detection."""
        # No GITHUB_TOKEN env var and gh CLI unavailable: auto-detection alone
        # would land on UNAUTHENTICATED, but the explicit token should win.
        with mock.patch("github_actions_api.shutil.which", return_value=None):
            api = GitHubAPI(github_token="explicit-token")
            self.assertEqual(api.get_auth_method(), GitHubAPI.AuthMethod.GITHUB_TOKEN)
            self.assertEqual(api._github_token, "explicit-token")

    def test_explicit_github_token_used_in_request_headers(self):
        """An explicit github_token should be sent as the Authorization header."""
        api = GitHubAPI(github_token="explicit-token")
        headers = api._get_request_headers()
        self.assertEqual(headers["Authorization"], "Bearer explicit-token")

    def test_explicit_github_token_independent_of_env_token(self):
        """Two instances with different explicit tokens should not interfere."""
        os.environ["GITHUB_TOKEN"] = "env-token"
        default_api = GitHubAPI()
        app_api = GitHubAPI(github_token="app-token")

        self.assertEqual(
            default_api._get_request_headers()["Authorization"], "Bearer env-token"
        )
        self.assertEqual(
            app_api._get_request_headers()["Authorization"], "Bearer app-token"
        )

    def test_no_github_token_falls_back_to_auto_detection(self):
        """Without an explicit github_token, auto-detection still runs as before."""
        api = GitHubAPI(github_token=None)
        self.assertIsNone(api._auth_method)

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

        with mock.patch("github_actions_api.urlopen", return_value=mock_response):
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

        with mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
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
            "github_actions_api.subprocess.run",
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
            "github_actions_api.subprocess.run",
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

        with mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("gh api request failed", str(ctx.exception))
            self.assertIn("Not Found", str(ctx.exception))

    def test_gh_cli_rate_limit_error_passes_through_message(self):
        """gh CLI rate limit error should pass through the stderr message."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stderr = "gh: API rate limit exceeded for user ID 123."

        with mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            error_msg = str(ctx.exception)
            # gh CLI stderr message should be preserved in error
            self.assertIn("rate limit", error_msg.lower())

    def test_gh_cli_empty_response_raises_github_api_error(self):
        """gh CLI empty response should raise GitHubAPIError."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
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

        with mock.patch("github_actions_api.subprocess.run", return_value=mock_result):
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

        with mock.patch("github_actions_api.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("403", str(ctx.exception))
            self.assertIn("Access denied", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, HTTPError)

    def test_rest_api_rate_limit_error_provides_helpful_message(self):
        """REST API rate limit (403 with rate limit body) should provide actionable guidance."""
        import io

        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        # GitHub returns 403 with a JSON body containing the rate limit message
        rate_limit_body = b'{"message": "API rate limit exceeded for user ID 123."}'

        mock_error = HTTPError(
            url="https://api.github.com/repos/test/test",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=io.BytesIO(rate_limit_body),
        )

        with mock.patch("github_actions_api.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            error_msg = str(ctx.exception)
            # Should mention rate limit, not just "Access denied"
            self.assertIn("rate limit", error_msg.lower())
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

        with mock.patch("github_actions_api.urlopen", side_effect=mock_error):
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

        with mock.patch("github_actions_api.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("500", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, HTTPError)

    def test_rest_api_network_error_raises_github_api_error(self):
        """REST API network error should raise GitHubAPIError with URLError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_error = URLError(reason="Connection refused")

        with mock.patch("github_actions_api.urlopen", side_effect=mock_error):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Network error", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, URLError)

    def test_rest_api_timeout_raises_github_api_error(self):
        """REST API timeout should raise GitHubAPIError with TimeoutError cause."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        with mock.patch("github_actions_api.urlopen", side_effect=TimeoutError()):
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

        with mock.patch("github_actions_api.urlopen", return_value=mock_response):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Invalid JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)

    def test_rest_api_post_sends_json_body(self):
        """REST API POST should send JSON body with Content-Type header."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'{"id": 99}'
        mock_response.__enter__.return_value = mock_response

        with mock.patch(
            "github_actions_api.urlopen", return_value=mock_response
        ) as urlopen:
            result = api.send_request(
                "https://api.github.com/repos/test/test/issues/1/comments",
                method="POST",
                body={"body": "hello"},
            )

        self.assertEqual(result, {"id": 99})
        request = urlopen.call_args[0][0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.data, b'{"body": "hello"}')
        self.assertEqual(request.headers["Content-type"], "application/json")

    def test_rest_api_patch_sends_json_body(self):
        """REST API PATCH should send JSON body."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b'{"id": 42, "body": "updated"}'
        mock_response.__enter__.return_value = mock_response

        with mock.patch(
            "github_actions_api.urlopen", return_value=mock_response
        ) as urlopen:
            result = api.send_request(
                "https://api.github.com/repos/test/test/issues/comments/42",
                method="PATCH",
                body={"body": "updated"},
            )

        self.assertEqual(result["id"], 42)
        request = urlopen.call_args[0][0]
        self.assertEqual(request.method, "PATCH")

    def test_gh_cli_post_uses_method_and_stdin_body(self):
        """gh CLI POST should pass --method and JSON on stdin."""
        api = GitHubAPI()
        api._auth_method = GitHubAPI.AuthMethod.GH_CLI
        api._gh_cli_path = "/usr/bin/gh"

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"id": 7}'

        with mock.patch(
            "github_actions_api.subprocess.run", return_value=mock_result
        ) as subprocess_run:
            result = api.send_request(
                "https://api.github.com/repos/test/test/issues/1/comments",
                method="POST",
                body={"body": "hello"},
            )

        self.assertEqual(result, {"id": 7})
        cmd = subprocess_run.call_args[0][0]
        self.assertEqual(cmd[0:4], ["/usr/bin/gh", "api", "--method", "POST"])
        self.assertEqual(cmd[4], "/repos/test/test/issues/1/comments")
        self.assertIn("--input", cmd)
        self.assertEqual(subprocess_run.call_args[1]["input"], '{"body": "hello"}')

    def test_rest_api_get_empty_body_raises_github_api_error(self):
        """REST API GET with empty body should raise GitHubAPIError (unchanged behavior)."""
        import json

        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__.return_value = mock_response

        with mock.patch("github_actions_api.urlopen", return_value=mock_response):
            with self.assertRaises(GitHubAPIError) as ctx:
                api.send_request("https://api.github.com/repos/test/test")

            self.assertIn("Invalid JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)

    def test_rest_api_post_empty_body_returns_empty_dict(self):
        """REST API POST with empty body may return {} without parsing JSON."""
        os.environ["GITHUB_TOKEN"] = "test-token"
        api = GitHubAPI()

        mock_response = mock.MagicMock()
        mock_response.read.return_value = b""
        mock_response.__enter__.return_value = mock_response

        with mock.patch("github_actions_api.urlopen", return_value=mock_response):
            result = api.send_request(
                "https://api.github.com/repos/test/test/issues/1/comments",
                method="POST",
                body={"body": "hello"},
            )

        self.assertEqual(result, {})


class GhaUpdatePrCommentTest(unittest.TestCase):
    """Tests for gha_update_pr_comment."""

    def test_posts_new_comment_when_marker_not_found(self):
        marker = "<!-- therock-report-manifest-diff -->"
        body = f"{marker}\n### Report\n\n[View report](https://example.com)\n"
        comments_url = "https://api.github.com/repos/ROCm/TheRock/issues/42/comments"

        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            side_effect=[
                [{"id": 1, "body": "unrelated comment"}],
                {"id": 99, "body": body},
            ],
        ) as send_request:
            result = gha_update_pr_comment(
                pr_number=42,
                marker=marker,
                body=body,
            )

        self.assertEqual(result["id"], 99)
        self.assertEqual(send_request.call_count, 2)
        send_request.assert_any_call(
            f"{comments_url}?per_page=100&page=1",
        )
        send_request.assert_any_call(
            comments_url,
            method="POST",
            body={"body": body},
        )

    def test_patches_existing_comment_when_marker_found(self):
        marker = "<!-- therock-report-manifest-diff -->"
        body = f"{marker}\n### Report\n\n[View report](https://example.com/v2)\n"
        comments_url = "https://api.github.com/repos/ROCm/TheRock/issues/42/comments"

        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            side_effect=[
                [{"id": 10, "body": f"{marker}\nold content"}],
                {"id": 10, "body": body},
            ],
        ) as send_request:
            result = gha_update_pr_comment(
                pr_number=42,
                marker=marker,
                body=body,
            )

        self.assertEqual(result["id"], 10)
        self.assertEqual(send_request.call_count, 2)
        send_request.assert_any_call(
            "https://api.github.com/repos/ROCm/TheRock/issues/comments/10",
            method="PATCH",
            body={"body": body},
        )
        send_request.assert_any_call(f"{comments_url}?per_page=100&page=1")

    def test_paginates_until_marker_found(self):
        marker = "<!-- therock-breadcrumb-unmapped-6057 -->"
        body = f"{marker}\n### Unmapped\n"
        comments_url = (
            "https://api.github.com/repos/ROCm/rocm-libraries/issues/7/comments"
        )

        page_1 = [{"id": i, "body": f"comment {i}"} for i in range(100)]
        page_2 = [{"id": 200, "body": f"{marker}\nstale"}]

        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            side_effect=[
                page_1,
                page_2,
                {"id": 200, "body": body},
            ],
        ) as send_request:
            result = gha_update_pr_comment(
                pr_number=7,
                marker=marker,
                body=body,
                github_repository="ROCm/rocm-libraries",
            )

        self.assertEqual(result["id"], 200)
        self.assertEqual(send_request.call_count, 3)
        send_request.assert_any_call(f"{comments_url}?per_page=100&page=1")
        send_request.assert_any_call(f"{comments_url}?per_page=100&page=2")

    def test_raises_when_comment_response_is_not_dict(self):
        marker = "<!-- therock-report-manifest-diff -->"
        body = f"{marker}\nbody"

        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            side_effect=[[], "not-a-dict"],
        ):
            with self.assertRaisesRegex(GitHubAPIError, "Expected comment object"):
                gha_update_pr_comment(pr_number=1, marker=marker, body=body)

    def test_uses_provided_github_api_instance_instead_of_singleton(self):
        """A github_api override should be used instead of the singleton."""
        marker = "<!-- therock-bump-breadcrumb -->"
        body = f"{marker}\nThis PR is now part of TheRock."

        custom_api = mock.create_autospec(GitHubAPI, instance=True)
        custom_api.send_request.side_effect = [
            [],
            {"id": 55, "body": body},
        ]

        with mock.patch(
            "github_actions_api._default_github_api.send_request"
        ) as singleton_send_request:
            result = gha_update_pr_comment(
                pr_number=9,
                marker=marker,
                body=body,
                github_repository="ROCm/rocm-systems",
                github_api=custom_api,
            )

        self.assertEqual(result["id"], 55)
        self.assertEqual(custom_api.send_request.call_count, 2)
        singleton_send_request.assert_not_called()


class GhaQueryPrsForCommitTest(unittest.TestCase):
    """Tests for gha_query_prs_for_commit."""

    def test_returns_prs_for_commit(self):
        sha = "a" * 40
        prs = [{"number": 42, "title": "Some PR"}]

        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            return_value=prs,
        ) as send_request:
            result = gha_query_prs_for_commit("ROCm/TheRock", sha)

        self.assertEqual(result, prs)
        send_request.assert_called_once_with(
            f"https://api.github.com/repos/ROCm/TheRock/commits/{sha}/pulls"
        )

    def test_returns_empty_list_when_no_prs_found(self):
        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            return_value=[],
        ):
            result = gha_query_prs_for_commit("ROCm/TheRock", "b" * 40)

        self.assertEqual(result, [])

    def test_raises_when_response_is_not_list(self):
        with mock.patch(
            "github_actions_api._default_github_api.send_request",
            return_value={"message": "not found"},
        ):
            with self.assertRaisesRegex(GitHubAPIError, "Expected a list"):
                gha_query_prs_for_commit("ROCm/TheRock", "c" * 40)

    def test_uses_provided_github_api_instance_instead_of_singleton(self):
        """A github_api override should be used instead of the singleton."""
        sha = "d" * 40
        prs = [{"number": 7}]
        custom_api = mock.create_autospec(GitHubAPI, instance=True)
        custom_api.send_request.return_value = prs

        with mock.patch(
            "github_actions_api._default_github_api.send_request"
        ) as singleton_send_request:
            result = gha_query_prs_for_commit(
                "ROCm/rocm-libraries", sha, github_api=custom_api
            )

        self.assertEqual(result, prs)
        custom_api.send_request.assert_called_once_with(
            f"https://api.github.com/repos/ROCm/rocm-libraries/commits/{sha}/pulls"
        )
        singleton_send_request.assert_not_called()


class GitHubActionsUtilsTest(unittest.TestCase):
    def test_resolve_git_ref_returns_sha_from_commit_api(self):
        with mock.patch(
            "github_actions_api.gha_send_request",
            return_value={"sha": "1" * 40},
        ) as gha_send_request:
            sha = gha_resolve_git_ref("ROCm/pytorch", "release/2.12")

        self.assertEqual(sha, "1" * 40)
        gha_send_request.assert_called_once_with(
            "https://api.github.com/repos/ROCm/pytorch/commits/release%2F2.12"
        )

    def test_fetch_file_contents_decodes_contents_api_response(self):
        content = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(content).decode("ascii")
        with mock.patch(
            "github_actions_api.gha_send_request",
            return_value={"type": "file", "encoding": "base64", "content": encoded},
        ) as gha_send_request:
            contents = gha_fetch_file_contents(
                "ROCm/pytorch", "some path/version.txt", "release/2.12"
            )

        self.assertEqual(contents, content)
        gha_send_request.assert_called_once_with(
            "https://api.github.com/repos/ROCm/pytorch/contents/some%20path/version.txt?ref=release%2F2.12"
        )

    def test_fetch_text_file_contents_decodes_text(self):
        content = "2.13.0a0\n".encode("utf-8")
        encoded = base64.b64encode(content).decode("ascii")
        with mock.patch(
            "github_actions_api.gha_send_request",
            return_value={"type": "file", "encoding": "base64", "content": encoded},
        ):
            text = gha_fetch_text_file_contents(
                "pytorch/pytorch", "version.txt", "nightly"
            )

        self.assertEqual(text, "2.13.0a0\n")

    def test_fetch_file_contents_rejects_non_file_response(self):
        with mock.patch("github_actions_api.gha_send_request", return_value=[]):
            with self.assertRaisesRegex(GitHubAPIError, "Expected GitHub contents"):
                gha_fetch_file_contents("ROCm/pytorch", "ci", "abc123")

    def test_fetch_file_contents_rejects_non_base64_response(self):
        with mock.patch(
            "github_actions_api.gha_send_request",
            return_value={"type": "file", "encoding": "none", "content": ""},
        ):
            with self.assertRaisesRegex(GitHubAPIError, "Expected base64"):
                gha_fetch_file_contents("ROCm/pytorch", "large.bin", "abc123")

    def setUp(self):
        # Save environment state
        self._saved_env = {}
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY"]:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]
        # Clean environment for tests
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY"]:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        # Restore environment state
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY"]:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._saved_env.items():
            os.environ[key] = value

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_workflow_run_by_id(self):
        """Test querying a workflow run by its ID."""
        workflow_run = gha_query_workflow_run_by_id("ROCm/TheRock", "18022609292")
        self.assertEqual(workflow_run["repository"]["full_name"], "ROCm/TheRock")

        # Verify fields depended on by WorkflowOutputRoot and find_artifacts_for_commit
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

        # Verify fields depended on by WorkflowOutputRoot and find_artifacts_for_commit
        run = runs[0]
        self.assertIn("id", run)
        self.assertIn("head_repository", run)
        self.assertIn("full_name", run["head_repository"])
        self.assertIn("created_at", run)
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

    def test_gha_query_workflow_runs_for_commit_sorts_by_created_at(self):
        """Runs are sorted most-recent-first by created_at (ISO 8601)."""
        # API returns ISO 8601 timestamps like "2026-01-15T10:00:00Z" which
        # are lexicographically sortable. Simulate an API response where the
        # runs arrive in the wrong order.
        older_run = {"id": 1, "created_at": "2026-01-10T08:00:00Z"}
        newer_run = {"id": 2, "created_at": "2026-01-15T10:00:00Z"}

        with mock.patch(
            "github_actions_api.gha_send_request",
            return_value={"workflow_runs": [older_run, newer_run]},
        ):
            runs = gha_query_workflow_runs_for_commit(
                "ROCm/TheRock", "ci.yml", "abc123"
            )

        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["id"], 2, "Newer run should be first")
        self.assertEqual(runs[1]["id"], 1, "Older run should be second")

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_last_workflow_run(self):
        """Test querying for the last workflow run on a branch."""
        # Test successful run found on main branch
        result = gha_query_last_workflow_run(
            "ROCm/TheRock", "multi_arch_ci.yml", "main"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["head_branch"], "main")
        self.assertEqual(result["conclusion"], "success")
        self.assertIn("id", result)

        # Test multi-status set: accept success or failure
        result = gha_query_last_workflow_run(
            "ROCm/TheRock",
            "multi_arch_ci.yml",
            "main",
            accepted_statuses={"success", "failure"},
        )
        self.assertIsNotNone(result)
        self.assertIn(result["conclusion"], {"success", "failure"})

        # Test no matching branch - should return None
        result = gha_query_last_workflow_run(
            "ROCm/TheRock", "multi_arch_ci.yml", "nonexistent-branch-12345"
        )
        self.assertIsNone(result)

        # Test non-existent workflow - should raise an exception
        with self.assertRaises(Exception):
            gha_query_last_workflow_run(
                "ROCm/TheRock", "nonexistent_workflow_12345.yml", "main"
            )

    @_skip_unless_authenticated_github_api_is_available
    def test_gha_query_recent_branch_commits(self):
        """Test querying recent commits on a branch."""
        import re

        sha_pattern = re.compile(r"^[0-9a-f]{40}$")

        # Test default parameters (main branch)
        commits = gha_query_recent_branch_commits("ROCm/TheRock")
        self.assertIsInstance(commits, list)
        self.assertGreater(len(commits), 0)

        # Verify each commit SHA is a valid 40-character hex string
        for sha in commits:
            self.assertIsInstance(sha, str)
            self.assertRegex(sha, sha_pattern, f"Invalid SHA format: {sha}")

        # Test max_count parameter limits results
        commits_limited = gha_query_recent_branch_commits(
            "ROCm/TheRock", branch="main", max_count=5
        )
        self.assertIsInstance(commits_limited, list)
        self.assertLessEqual(len(commits_limited), 5)
        self.assertGreater(len(commits_limited), 0)

        # Each limited result should also be a valid SHA
        for sha in commits_limited:
            self.assertRegex(sha, sha_pattern)


class JobSummaryTest(unittest.TestCase):
    """Tests for the job-summary mirror and output helpers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.runner_temp = self.tmp_path / "runner_temp"
        self.runner_temp.mkdir()
        self.step_summary_file = self.tmp_path / "step_summary.md"
        self.github_output_file = self.tmp_path / "github_output"
        self.env = {
            "RUNNER_TEMP": os.fspath(self.runner_temp),
            "GITHUB_STEP_SUMMARY": os.fspath(self.step_summary_file),
            "GITHUB_OUTPUT": os.fspath(self.github_output_file),
        }

    def test_mirror_path_uses_runner_temp(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            self.assertEqual(
                gha_job_summary_mirror_path(),
                self.runner_temp / "job_summary.md",
            )

    def test_mirror_path_none_without_runner_temp(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(gha_job_summary_mirror_path())

    def test_append_writes_both_step_summary_and_mirror(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary("### Section A")
            gha_append_step_summary("### Section B")

        expected = "### Section A\n\n### Section B\n\n"
        self.assertEqual(self.step_summary_file.read_text(encoding="utf-8"), expected)
        mirror = self.runner_temp / "job_summary.md"
        self.assertEqual(mirror.read_text(encoding="utf-8"), expected)

    def test_append_multiline_summary(self):
        summary = "### Heading\n\n- bullet a\n- bullet b\n\n| col |\n| --- |"
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary(summary)

        expected = summary + "\n\n"
        self.assertEqual(self.step_summary_file.read_text(encoding="utf-8"), expected)
        mirror = self.runner_temp / "job_summary.md"
        self.assertEqual(mirror.read_text(encoding="utf-8"), expected)

    def test_append_can_skip_mirror(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary("### Only step summary", mirror_to_job_file=False)

        self.assertEqual(
            self.step_summary_file.read_text(encoding="utf-8"),
            "### Only step summary\n\n",
        )
        self.assertFalse((self.runner_temp / "job_summary.md").exists())

    def test_append_without_step_summary_does_not_mirror(self):
        # The mirror must only ever hold what was written to GITHUB_STEP_SUMMARY.
        # When that env var is unset there is nothing to mirror, so the mirror
        # file must not be created.
        env = {k: v for k, v in self.env.items() if k != "GITHUB_STEP_SUMMARY"}
        with mock.patch.dict(os.environ, env, clear=True):
            gha_append_step_summary("### Mirror only")

        self.assertFalse((self.runner_temp / "job_summary.md").exists())

    def test_set_output_publishes_with_custom_name(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary("### Line 1")
            gha_append_step_summary("### Line 2")
            gha_set_job_summary_output(output_name="job_summary")

        # The accumulated mirror is published under the requested name using the
        # multiline heredoc form written by gha_set_output.
        output = self.github_output_file.read_text(encoding="utf-8")
        self.assertIn("job_summary<<EOF_mag1c\n", output)
        self.assertIn("### Line 1", output)
        self.assertIn("### Line 2", output)

    def _parse_heredoc_output(self, output_name):
        # Parse the GITHUB_OUTPUT heredoc the way GitHub does and return the
        # published value for output_name.
        output = self.github_output_file.read_text(encoding="utf-8")
        # first == "output_name<<EOF_mag1c", rest == "value\nEOF_mag1c\n"
        first, _, rest = output.partition("\n")
        key, _, delimiter = first.partition("<<")
        self.assertEqual(key, output_name)
        closing = f"\n{delimiter}\n"
        self.assertTrue(rest.endswith(closing))
        return rest[: -len(closing)]

    def test_set_output_round_trips_single_line_summary(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary("### Only section")
            gha_set_job_summary_output()

        # gha_append_step_summary always appends a blank-line separator, so the
        # published value carries the same trailing newlines as the mirror.
        self.assertEqual(self._parse_heredoc_output("summary"), "### Only section\n\n")

    def test_set_output_round_trips_multi_line_summary(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_append_step_summary("### Heading\n\n- bullet a\n- bullet b")
            gha_append_step_summary("| col |\n| --- |\n| val |")
            gha_set_job_summary_output()

        expected = (
            "### Heading\n\n- bullet a\n- bullet b\n\n" "| col |\n| --- |\n| val |\n\n"
        )
        self.assertEqual(self._parse_heredoc_output("summary"), expected)

    def test_set_output_skips_when_no_mirror(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            gha_set_job_summary_output()

        # No mirror written -> nothing published to GITHUB_OUTPUT.
        output = (
            self.github_output_file.read_text(encoding="utf-8")
            if self.github_output_file.exists()
            else ""
        )
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
