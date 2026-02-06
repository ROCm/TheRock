"""Tests for generate_manifest_diff_report.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from generate_manifest_diff_report import (
    compare_manifests,
    determine_status,
    fetch_commits_in_range,
    format_commit_date,
    get_api_base_from_url,
    is_revert,
    parse_args,
    parse_gitmodules,
    resolve_commits,
)
from github_actions.github_actions_utils import is_authenticated_github_api_available


def _skip_unless_authenticated_github_api_is_available(test_func):
    """Decorator to skip tests unless GitHub API is available."""
    return unittest.skipUnless(
        is_authenticated_github_api_available(),
        "No authenticated GitHub API auth available (need GITHUB_TOKEN or authenticated gh CLI)",
    )(test_func)


# =============================================================================
# Pure Function Unit Tests
# =============================================================================


class ParseGitmodulesTest(unittest.TestCase):
    """Tests for parse_gitmodules function."""

    def test_parse_single_submodule(self):
        """Parse a single submodule entry."""
        content = """[submodule "llvm-project"]
    path = llvm-project
    url = https://github.com/ROCm/llvm-project.git
"""
        result = parse_gitmodules(content)

        self.assertEqual(len(result), 1)
        self.assertIn("llvm-project", result)
        self.assertEqual(result["llvm-project"]["name"], "llvm-project")
        self.assertEqual(
            result["llvm-project"]["url"], "https://github.com/ROCm/llvm-project.git"
        )
        self.assertEqual(result["llvm-project"]["branch"], "main")  # Default

    def test_parse_multiple_submodules_with_branch(self):
        """Parse multiple submodules including one with explicit branch."""
        content = """[submodule "llvm-project"]
    path = llvm-project
    url = https://github.com/ROCm/llvm-project.git

[submodule "rocm-libraries"]
    path = rocm-libraries
    url = https://github.com/ROCm/rocm-libraries.git
    branch = develop
"""
        result = parse_gitmodules(content)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["llvm-project"]["branch"], "main")
        self.assertEqual(result["rocm-libraries"]["branch"], "develop")


class GetApiBaseFromUrlTest(unittest.TestCase):
    """Tests for get_api_base_from_url function."""

    def test_https_url(self):
        """Convert HTTPS GitHub URL to API base."""
        url = "https://github.com/ROCm/rocBLAS.git"
        result = get_api_base_from_url(url, "rocBLAS")

        self.assertEqual(result, "https://api.github.com/repos/ROCm/rocBLAS")

    def test_ssh_url(self):
        """Convert SSH GitHub URL to API base."""
        url = "git@github.com:ROCm/MIOpen.git"
        result = get_api_base_from_url(url, "MIOpen")

        self.assertEqual(result, "https://api.github.com/repos/ROCm/MIOpen")


class FormatCommitDateTest(unittest.TestCase):
    """Tests for format_commit_date function."""

    def test_valid_iso_date(self):
        """Format valid ISO date string."""
        date_str = "2025-01-15T10:30:00Z"
        result = format_commit_date(date_str)

        self.assertEqual(result, "Jan 15, 2025")

    def test_invalid_date(self):
        """Handle invalid/empty date strings."""
        self.assertEqual(format_commit_date("Unknown"), "Unknown")
        self.assertEqual(format_commit_date(""), "Unknown")
        self.assertEqual(format_commit_date("not-a-date"), "not-a-date")


class DetermineStatusTest(unittest.TestCase):
    """Tests for determine_status function."""

    def test_removed_status(self):
        """Old SHA exists, new SHA doesn't -> removed."""
        status, fetch_start, fetch_end = determine_status(
            "abc123", None, "https://api.github.com/repos/ROCm/test"
        )

        self.assertEqual(status, "removed")
        self.assertEqual(fetch_start, "")
        self.assertEqual(fetch_end, "")

    def test_added_status(self):
        """New SHA exists, old SHA doesn't -> added."""
        status, fetch_start, fetch_end = determine_status(
            None, "def456", "https://api.github.com/repos/ROCm/test"
        )

        self.assertEqual(status, "added")
        self.assertEqual(fetch_start, "")
        self.assertEqual(fetch_end, "def456")

    def test_unchanged_status(self):
        """Same SHA returns unchanged status without API calls."""
        # This should not make any API calls since SHAs are equal
        status, fetch_start, fetch_end = determine_status(
            "abc123", "abc123", "https://api.github.com/repos/ROCm/test"
        )

        self.assertEqual(status, "unchanged")
        self.assertEqual(fetch_start, "")
        self.assertEqual(fetch_end, "")


# =============================================================================
# Mocked API Tests
# =============================================================================


class IsRevertTest(unittest.TestCase):
    """Tests for is_revert function with mocked API calls."""

    def test_is_revert_ahead_status(self):
        """Returns True when old_sha is ahead of new_sha (revert)."""
        with mock.patch(
            "generate_manifest_diff_report.gha_send_request"
        ) as mock_request:
            mock_request.return_value = {"status": "ahead"}
            result = is_revert(
                "old_sha", "new_sha", "https://api.github.com/repos/ROCm/test"
            )

        self.assertTrue(result)

    def test_is_revert_behind_status(self):
        """Returns False when old_sha is behind new_sha (forward progress)."""
        with mock.patch(
            "generate_manifest_diff_report.gha_send_request"
        ) as mock_request:
            mock_request.return_value = {"status": "behind"}
            result = is_revert(
                "old_sha", "new_sha", "https://api.github.com/repos/ROCm/test"
            )

        self.assertFalse(result)

    def test_is_revert_http_404(self):
        """Returns False on 404 (orphaned commits - can't determine)."""
        mock_error = HTTPError(
            url="https://api.github.com/repos/ROCm/test/compare/new...old",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        with mock.patch(
            "generate_manifest_diff_report.gha_send_request", side_effect=mock_error
        ):
            result = is_revert(
                "old_sha", "new_sha", "https://api.github.com/repos/ROCm/test"
            )

        self.assertFalse(result)


class FetchCommitsInRangeTest(unittest.TestCase):
    """Tests for fetch_commits_in_range function with mocked API calls."""

    def test_fetch_commits_success(self):
        """Successfully fetch commits between two SHAs."""
        mock_commits = [
            {"sha": "commit3", "commit": {"message": "Third"}},
            {"sha": "commit2", "commit": {"message": "Second"}},
            {"sha": "start_sha", "commit": {"message": "Start"}},
        ]

        with mock.patch(
            "generate_manifest_diff_report.gha_send_request"
        ) as mock_request:
            mock_request.return_value = mock_commits
            result = fetch_commits_in_range(
                repo_name="test-repo",
                start_sha="start_sha",
                end_sha="commit3",
                api_base="https://api.github.com/repos/ROCm/test",
            )

        # Should return commits up to but not including start_sha
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["sha"], "commit3")
        self.assertEqual(result[1]["sha"], "commit2")

    def test_fetch_commits_diverged_fallback(self):
        """Falls back to compare API when commits diverged."""
        diverged_commits = [
            {"sha": "diverged1"},
            {"sha": "diverged2"},
        ]

        def mock_request_side_effect(url):
            if "compare" in url:
                return {"status": "diverged", "commits": diverged_commits}
            # Return empty list to trigger fallback
            return []

        with mock.patch(
            "generate_manifest_diff_report.gha_send_request",
            side_effect=mock_request_side_effect,
        ):
            result = fetch_commits_in_range(
                repo_name="test-repo",
                start_sha="start_sha",
                end_sha="end_sha",
                api_base="https://api.github.com/repos/ROCm/test",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["sha"], "diverged1")


# =============================================================================
# Integration Tests (Real API Calls)
# =============================================================================


class ManifestDiffIntegrationTest(unittest.TestCase):
    """Integration tests using real API calls with documented test cases."""

    @_skip_unless_authenticated_github_api_is_available
    def test_superrepo_changed(self):
        """Test Case 1: rocm-libraries changed with component commits."""
        diff = compare_manifests("e3fb7163", "5ff856ba")

        self.assertIn("rocm-libraries", diff.superrepos)
        superrepo = diff.superrepos["rocm-libraries"]
        self.assertEqual(superrepo.status, "changed")
        self.assertTrue(len(superrepo.all_commits) > 0)

    @_skip_unless_authenticated_github_api_is_available
    def test_superrepo_unchanged(self):
        """Test Case 4: All superrepos unchanged."""
        diff = compare_manifests("c7bc0b40", "cf13cfdc")

        for name, superrepo in diff.superrepos.items():
            self.assertEqual(
                superrepo.status, "unchanged", f"{name} should be unchanged"
            )

    @_skip_unless_authenticated_github_api_is_available
    def test_submodule_changed(self):
        """Test Case 5: libhipcxx submodule changed."""
        diff = compare_manifests("3a6cbc2a", "02946b22")

        self.assertIn("libhipcxx", diff.submodules)
        self.assertEqual(diff.submodules["libhipcxx"].status, "changed")
        self.assertTrue(len(diff.submodules["libhipcxx"].commits) > 0)

    @_skip_unless_authenticated_github_api_is_available
    def test_submodule_reverted(self):
        """Test Case 7: libhipcxx submodule reverted (swapped commits from Case 5)."""
        diff = compare_manifests("02946b22", "3a6cbc2a")

        self.assertIn("libhipcxx", diff.submodules)
        self.assertEqual(diff.submodules["libhipcxx"].status, "reverted")

    @_skip_unless_authenticated_github_api_is_available
    def test_submodule_added(self):
        """Test Case 10: libhipcxx submodule added."""
        diff = compare_manifests("f5552032", "bcc9df4b")

        self.assertIn("libhipcxx", diff.submodules)
        self.assertEqual(diff.submodules["libhipcxx"].status, "added")


# =============================================================================
# CLI Options Tests (Mocked)
# =============================================================================


class ResolveCommitsTest(unittest.TestCase):
    """Tests for resolve_commits() with mocked API calls."""

    def test_workflow_mode_resolves_both_commits(self):
        """--workflow-mode resolves both start and end from workflow run IDs."""
        args = parse_args(["--start", "123", "--end", "456", "--workflow-mode"])

        with mock.patch(
            "generate_manifest_diff_report.gha_query_workflow_run_by_id"
        ) as mock_query:
            mock_query.side_effect = [
                {"head_sha": "abc123def456"},  # start workflow
                {"head_sha": "789xyz000111"},  # end workflow
            ]
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "abc123def456")
        self.assertEqual(end_sha, "789xyz000111")
        self.assertEqual(mock_query.call_count, 2)

    def test_find_last_successful_resolves_start(self):
        """--find-last-successful finds last successful run for start commit."""
        args = parse_args(["--end", "def456", "--find-last-successful", "ci.yml"])

        with mock.patch(
            "generate_manifest_diff_report.gha_query_last_successful_workflow_run"
        ) as mock_query:
            mock_query.return_value = {"head_sha": "last_successful_sha"}
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "last_successful_sha")
        self.assertEqual(end_sha, "def456")
        mock_query.assert_called_once()

    def test_direct_commit_shas_no_api_calls(self):
        """Direct commit SHAs don't require API calls."""
        args = parse_args(["--start", "abc123", "--end", "def456"])

        # No mocking needed - should work without API calls
        start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "abc123")
        self.assertEqual(end_sha, "def456")


# =============================================================================
# CLI Options Integration Tests (Real API Calls)
# =============================================================================


class ResolveCommitsIntegrationTest(unittest.TestCase):
    """Integration tests for CLI options with real API calls."""

    @_skip_unless_authenticated_github_api_is_available
    def test_workflow_mode_with_real_run_id(self):
        """Test --workflow-mode with a real workflow run ID."""
        # Using workflow run ID from existing tests in github_actions_utils_test.py
        # https://github.com/ROCm/TheRock/actions/runs/18022609292
        args = parse_args(
            ["--start", "18022609292", "--end", "18022609292", "--workflow-mode"]
        )

        start_sha, end_sha = resolve_commits(args)

        # Should resolve to actual commit SHAs (40 hex chars)
        self.assertEqual(len(start_sha), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in start_sha))
        self.assertEqual(start_sha, end_sha)  # Same workflow = same SHA

    @_skip_unless_authenticated_github_api_is_available
    def test_find_last_successful_with_real_workflow(self):
        """Test --find-last-successful with ci_nightly.yml."""
        args = parse_args(["--end", "main", "--find-last-successful", "ci_nightly.yml"])

        start_sha, end_sha = resolve_commits(args)

        # Should resolve start to a commit SHA from last successful nightly
        self.assertEqual(len(start_sha), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in start_sha))


if __name__ == "__main__":
    unittest.main()
