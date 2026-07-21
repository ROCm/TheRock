"""Tests for generate_manifest_diff_report.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from generate_manifest_diff_report import (
    build_commit_range_summary,
    create_table,
    determine_status,
    fetch_commits_in_range,
    format_commit_date,
    generate_non_superrepo_html,
    generate_step_summary,
    get_api_base_from_url,
    is_revert,
    ManifestDiff,
    parse_args,
    resolve_commits,
    Submodule,
)


# =============================================================================
# Pure Function Unit Tests
# =============================================================================


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
                {"head_sha": "789xyz000111"},  # end workflow (resolved first)
                {"head_sha": "abc123def456"},  # start workflow
            ]
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "abc123def456")
        self.assertEqual(end_sha, "789xyz000111")
        self.assertEqual(mock_query.call_count, 2)

    def test_find_last_run_resolves_start(self):
        """--find-last-run finds the most recent matching run for start commit."""
        args = parse_args(["--end", "def456", "--find-last-run", "multi_arch_ci.yml"])

        with mock.patch(
            "generate_manifest_diff_report.gha_query_last_workflow_run"
        ) as mock_query:
            mock_query.return_value = {"head_sha": "last_matching_sha"}
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "last_matching_sha")
        self.assertEqual(end_sha, "def456")
        mock_query.assert_called_once()

    def test_find_last_run_uses_terminal_statuses(self):
        """--find-last-run hardcodes accepted statuses to {success, failure}."""
        args = parse_args(["--end", "def456", "--find-last-run", "ci.yml"])

        with mock.patch(
            "generate_manifest_diff_report.gha_query_last_workflow_run"
        ) as mock_query:
            mock_query.return_value = {"head_sha": "abc"}
            resolve_commits(args)

        _, kwargs = mock_query.call_args
        self.assertEqual(kwargs["accepted_statuses"], {"success", "failure"})

    def test_pr_base_ref_resolves_start_via_compare(self):
        """--pr-base-ref resolves start as the merge-base via the Compare API."""
        args = parse_args(["--end", "deadbeef", "--pr-base-ref", "main"])

        with mock.patch(
            "generate_manifest_diff_report.gha_send_request"
        ) as mock_request:
            mock_request.return_value = {"merge_base_commit": {"sha": "base_sha_xyz"}}
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "base_sha_xyz")
        self.assertEqual(end_sha, "deadbeef")
        # Verify the URL hit Compare API with base...end ordering.
        called_url = mock_request.call_args.args[0]
        self.assertIn("/compare/main...deadbeef", called_url)

    def test_find_last_run_no_match_returns_none(self):
        """--find-last-run with no matching prior run → (None, None)."""
        args = parse_args(["--end", "def456", "--find-last-run", "ci.yml"])
        with mock.patch(
            "generate_manifest_diff_report.gha_query_last_workflow_run",
            return_value=None,
        ):
            self.assertEqual(resolve_commits(args), (None, None))

    def test_pr_base_ref_takes_precedence_over_find_last_run(self):
        """When both --pr-base-ref and --find-last-run are set, Compare wins.

        This pins the precedence ladder documented in resolve_commits():
        pr_base_ref > find_last_run > workflow_mode/start_ref. If a future
        refactor reorders the branches, this test catches it.
        """
        args = parse_args(
            [
                "--end",
                "deadbeef",
                "--pr-base-ref",
                "main",
                "--find-last-run",
                "ci.yml",
            ]
        )

        with mock.patch(
            "generate_manifest_diff_report.gha_send_request"
        ) as mock_compare, mock.patch(
            "generate_manifest_diff_report.gha_query_last_workflow_run"
        ) as mock_last_run:
            mock_compare.return_value = {"merge_base_commit": {"sha": "merge_base"}}
            start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "merge_base")
        self.assertEqual(end_sha, "deadbeef")
        mock_compare.assert_called_once()
        mock_last_run.assert_not_called()

    def test_direct_commit_shas_no_api_calls(self):
        """Direct commit SHAs don't require API calls."""
        args = parse_args(["--start", "abc123", "--end", "def456"])

        # No mocking needed - should work without API calls
        start_sha, end_sha = resolve_commits(args)

        self.assertEqual(start_sha, "abc123")
        self.assertEqual(end_sha, "def456")

    def test_output_dir_argument_parsed(self):
        """--output-dir argument is parsed as Path."""
        args = parse_args(["--start", "abc", "--end", "def", "--output-dir", "reports"])
        self.assertEqual(args.output_dir, Path("reports"))

    def test_output_dir_defaults_to_none(self):
        """--output-dir defaults to None when not specified."""
        args = parse_args(["--start", "abc", "--end", "def"])
        self.assertIsNone(args.output_dir)


# =============================================================================
# HTML Report Structure Tests
# =============================================================================


class HtmlReportStructureTest(unittest.TestCase):
    """Tests that generated HTML includes semantic row classes and data attributes."""

    def test_create_table_includes_header_row_class(self):
        """Report tables have header row with class report-table-header-row."""
        html = create_table(["Component", "Commits"], [])
        self.assertIn("report-table-header-row", html)

    def test_non_superrepo_html_includes_component_row_and_data_component(self):
        """Non-superrepo table rows have component-row class and data-component attribute."""
        sub = Submodule(
            name="test-submodule",
            sha="abc123",
            api_base="https://api.github.com/repos/ROCm/test",
            branch="main",
            status="unchanged",
        )
        diff = ManifestDiff(
            start_commit="start",
            end_commit="end",
            submodules={"test-submodule": sub},
        )
        html = generate_non_superrepo_html(diff)
        self.assertIn("component-row", html)
        self.assertIn("data-component=", html)
        self.assertIn("test-submodule", html)


class BuildCommitRangeSummaryTest(unittest.TestCase):
    """Tests for build_commit_range_summary()."""

    def test_includes_short_shas_and_singular_changed_count(self):
        diff = ManifestDiff(
            start_commit="a" * 40,
            end_commit="b" * 40,
            submodules={
                "changed-sub": Submodule(
                    name="changed-sub",
                    sha="c" * 40,
                    api_base="https://api.github.com/repos/ROCm/changed-sub",
                    branch="main",
                    status="changed",
                ),
                "unchanged-sub": Submodule(
                    name="unchanged-sub",
                    sha="d" * 40,
                    api_base="https://api.github.com/repos/ROCm/unchanged-sub",
                    branch="main",
                    status="unchanged",
                ),
            },
        )

        summary = build_commit_range_summary(diff)

        self.assertIn(f"`{diff.start_commit[:8]}`", summary)
        self.assertIn(f"`{diff.end_commit[:8]}`", summary)
        self.assertIn("1 submodule changed", summary)

    def test_zero_and_plural_changed_counts(self):
        diff = ManifestDiff(start_commit="a" * 40, end_commit="b" * 40)
        self.assertIn("0 submodules changed", build_commit_range_summary(diff))

        for name in ("s1", "s2"):
            diff.submodules[name] = Submodule(
                name=name,
                sha="e" * 40,
                api_base="https://api.github.com/repos/ROCm/" + name,
                branch="main",
                status="changed",
            )
        self.assertIn("2 submodules changed", build_commit_range_summary(diff))


class GenerateStepSummaryReusesCommitRangeSummaryTest(unittest.TestCase):
    """generate_step_summary() should embed build_commit_range_summary()'s text
    verbatim rather than re-deriving equivalent formatting, so both surfaces
    (the job step summary and the bump-PR comment) agree exactly."""

    def test_step_summary_includes_commit_range_summary_line(self):
        diff = ManifestDiff(start_commit="a" * 40, end_commit="b" * 40)
        expected_line = build_commit_range_summary(diff)

        with mock.patch(
            "generate_manifest_diff_report.gha_append_step_summary"
        ) as append_summary:
            generate_step_summary(diff)

        append_summary.assert_called_once()
        posted_summary = append_summary.call_args[0][0]
        self.assertIn(expected_line, posted_summary)


if __name__ == "__main__":
    unittest.main()
