# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add github_actions to path so workflow_summary is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from workflow_summary import (
    FailedJobInfo,
    evaluate_results,
    fetch_failed_jobs,
    main,
    parse_needs_json,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic needs JSON blobs
# ---------------------------------------------------------------------------

ALL_SUCCESS = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "success", "outputs": {}},
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

ONE_FAILURE = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "failure", "outputs": {}},
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

FAILURE_WITH_CONTINUE_ON_ERROR = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {
        "result": "failure",
        "outputs": {"continue_on_error": "true"},
    },
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

ONE_CANCELLED = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "cancelled", "outputs": {}},
}

ONE_SKIPPED = {
    "setup": {"result": "success", "outputs": {}},
    "windows_build_and_test": {"result": "skipped", "outputs": {}},
}


# ---------------------------------------------------------------------------
# parse_needs_json
# ---------------------------------------------------------------------------


class TestParseNeedsJson:
    def test_all_success(self):
        jobs = parse_needs_json(json.dumps(ALL_SUCCESS))
        assert len(jobs) == 3
        assert all(j.result == "success" for j in jobs)
        assert all(not j.continue_on_error for j in jobs)

    def test_continue_on_error_parsed(self):
        jobs = parse_needs_json(json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR))
        by_name = {j.name: j for j in jobs}
        assert by_name["linux_build_and_test"].continue_on_error is True
        assert by_name["setup"].continue_on_error is False

    def test_missing_outputs_key(self):
        """Jobs with no outputs key should still parse (continue_on_error=False)."""
        needs = {"build": {"result": "success"}}
        jobs = parse_needs_json(json.dumps(needs))
        assert len(jobs) == 1
        assert jobs[0].continue_on_error is False

    def test_null_outputs(self):
        """Jobs with null outputs should still parse."""
        needs = {"build": {"result": "success", "outputs": None}}
        jobs = parse_needs_json(json.dumps(needs))
        assert len(jobs) == 1
        assert jobs[0].continue_on_error is False

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_needs_json("not json")

    def test_non_dict_top_level_raises(self):
        with pytest.raises(AssertionError, match="Expected a JSON object"):
            parse_needs_json("[]")

    def test_non_dict_job_raises(self):
        with pytest.raises(AssertionError, match="JSON object for job"):
            parse_needs_json(json.dumps({"build": "not a dict"}))


# ---------------------------------------------------------------------------
# evaluate_results
# ---------------------------------------------------------------------------


class TestEvaluateResults:
    def test_all_success(self):
        jobs = parse_needs_json(json.dumps(ALL_SUCCESS))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 3

    def test_one_failure(self):
        jobs = parse_needs_json(json.dumps(ONE_FAILURE))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 1
        assert failed[0].name == "linux_build_and_test"
        assert len(ok) == 2

    def test_continue_on_error_not_failed(self):
        jobs = parse_needs_json(json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 3

    def test_cancelled_is_failure(self):
        jobs = parse_needs_json(json.dumps(ONE_CANCELLED))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 1
        assert failed[0].name == "linux_build_and_test"
        assert failed[0].result == "cancelled"

    def test_skipped_is_ok(self):
        jobs = parse_needs_json(json.dumps(ONE_SKIPPED))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 2

    def test_empty_needs(self):
        failed, ok = evaluate_results([])
        assert len(failed) == 0
        assert len(ok) == 0


# ---------------------------------------------------------------------------
# fetch_failed_jobs
# ---------------------------------------------------------------------------


class TestFetchFailedJobs:
    def test_parses_failed_jobs(self):
        api_response = {
            "jobs": [
                {
                    "name": "Build",
                    "conclusion": "success",
                    "html_url": "https://github.com/test/run/1",
                },
                {
                    "name": "Test hip-tests (shard 1/1)",
                    "conclusion": "failure",
                    "html_url": "https://github.com/test/run/2",
                },
                {
                    "name": "Test rocthrust (shard 1/1)",
                    "conclusion": "failure",
                    "html_url": "https://github.com/test/run/3",
                },
            ]
        }
        with patch("workflow_summary.gha_send_request", return_value=api_response):
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert len(failed) == 2
        assert failed[0] == FailedJobInfo(
            name="Test hip-tests (shard 1/1)", html_url="https://github.com/test/run/2"
        )
        assert failed[1] == FailedJobInfo(
            name="Test rocthrust (shard 1/1)", html_url="https://github.com/test/run/3"
        )

    def test_no_failures(self):
        api_response = {
            "jobs": [
                {"name": "Build", "conclusion": "success", "html_url": ""},
            ]
        }
        with patch("workflow_summary.gha_send_request", return_value=api_response):
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert len(failed) == 0

    def test_empty_jobs_list(self):
        with patch("workflow_summary.gha_send_request", return_value={"jobs": []}):
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert len(failed) == 0

    def test_paginates_when_more_than_100_jobs(self):
        """Runs with >100 jobs require multiple API pages."""
        # Page 1: 100 successful jobs.
        page1 = {
            "jobs": [
                {"name": f"Job {i}", "conclusion": "success", "html_url": ""}
                for i in range(100)
            ]
        }
        # Page 2: 2 jobs, one failed.
        page2 = {
            "jobs": [
                {"name": "Job 100", "conclusion": "success", "html_url": ""},
                {
                    "name": "Job 101",
                    "conclusion": "failure",
                    "html_url": "https://github.com/test/run/101",
                },
            ]
        }
        with patch("workflow_summary.gha_send_request", side_effect=[page1, page2]):
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert len(failed) == 1
        assert failed[0] == FailedJobInfo(
            name="Job 101", html_url="https://github.com/test/run/101"
        )


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_all_success_returns_zero(self, capsys):
        rc = main(["--needs-json", json.dumps(ALL_SUCCESS)])
        assert rc == 0
        assert "succeeded" in capsys.readouterr().out

    def test_failure_returns_one(self, capsys):
        rc = main(["--needs-json", json.dumps(ONE_FAILURE)])
        assert rc == 1
        assert "failed" in capsys.readouterr().out

    def test_continue_on_error_returns_zero(self, capsys):
        rc = main(["--needs-json", json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR)])
        assert rc == 0
        assert "succeeded" in capsys.readouterr().out

    def test_failure_with_api_prints_urls(self, capsys):
        api_response = {
            "jobs": [
                {
                    "name": "Test hip-tests",
                    "conclusion": "failure",
                    "html_url": "https://github.com/test/run/2",
                },
            ]
        }
        with patch("workflow_summary.gha_send_request", return_value=api_response):
            rc = main(
                [
                    "--needs-json",
                    json.dumps(ONE_FAILURE),
                    "--github-repository",
                    "owner/repo",
                    "--github-run-id",
                    "12345",
                ]
            )

        assert rc == 1
        out = capsys.readouterr().out
        assert "Test hip-tests" in out
        assert "https://github.com/test/run/2" in out

    def test_failure_without_api_args_still_works(self, capsys, monkeypatch):
        """Without repo/run-id, the script should still report failures."""
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        rc = main(["--needs-json", json.dumps(ONE_FAILURE)])
        assert rc == 1
        assert "failed" in capsys.readouterr().out

    def test_failure_with_api_error_still_fails(self, capsys):
        """API errors should not prevent the script from reporting failures."""
        from github_actions_api import GitHubAPIError

        with patch(
            "workflow_summary.gha_send_request",
            side_effect=GitHubAPIError("test error"),
        ):
            rc = main(
                [
                    "--needs-json",
                    json.dumps(ONE_FAILURE),
                    "--github-repository",
                    "owner/repo",
                    "--github-run-id",
                    "12345",
                ]
            )

        assert rc == 1
        out = capsys.readouterr().out
        assert "Could not fetch job details" in out
