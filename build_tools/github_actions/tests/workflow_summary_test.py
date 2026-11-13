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
    JobResult,
    LogicalFailure,
    evaluate_results,
    fetch_failed_jobs,
    group_failed_jobs,
    main,
    normalize_failed_job,
    parse_needs_json,
    render_step_summary,
)


@pytest.fixture(autouse=True)
def clear_github_environment(monkeypatch):
    """Keep main() tests from using or modifying the enclosing Actions run."""
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


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

    def test_includes_root_problem_conclusions_but_not_cascade_cancellation(self):
        api_response = {
            "jobs": [
                {"name": "Timed out", "conclusion": "timed_out", "html_url": ""},
                {
                    "name": "Could not start",
                    "conclusion": "startup_failure",
                    "html_url": "",
                },
                {"name": "Cancelled", "conclusion": "cancelled", "html_url": ""},
            ]
        }
        with patch("workflow_summary.gha_send_request", return_value=api_response):
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert [(job.name, job.conclusion) for job in failed] == [
            ("Timed out", "timed_out"),
            ("Could not start", "startup_failure"),
        ]

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
        with patch(
            "workflow_summary.gha_send_request", side_effect=[page1, page2]
        ) as request:
            failed = fetch_failed_jobs("owner/repo", "12345")

        assert len(failed) == 1
        assert failed[0] == FailedJobInfo(
            name="Job 101", html_url="https://github.com/test/run/101"
        )
        assert "filter=latest&per_page=100&page=1" in request.call_args_list[0].args[0]
        assert "filter=latest&per_page=100&page=2" in request.call_args_list[1].args[0]


# ---------------------------------------------------------------------------
# compact summary rendering
# ---------------------------------------------------------------------------


class TestNormalizeFailedJob:
    def test_normalizes_standalone_shard_name(self):
        assert normalize_failed_job("Test hip-tests (shard 1/1)") == LogicalFailure(
            normalized_name="Test hip-tests",
            total_shards=1,
            display_name="Test hip-tests",
        )

    @pytest.mark.parametrize("component", ["hip-tests (PAL)", "hip-tests (ROCR)"])
    def test_friendly_name_preserves_component_parentheses(self, component):
        name = (
            "Linux::release / Test gfx94X-dcgpu / "
            f"Test {component} / Test {component} (shard 2/4) (gfx94X-dcgpu)"
        )

        failure = normalize_failed_job(name)

        assert failure.normalized_name == (
            "Linux::release / Test gfx94X-dcgpu / "
            f"Test {component} / Test {component} (gfx94X-dcgpu)"
        )
        assert failure.total_shards == 4
        assert failure.display_name == f"Linux · gfx94X-dcgpu · {component}"

    def test_friendly_name_supports_legacy_platform_segment(self):
        name = (
            "Windows::gfx110X-all::release / Test Artifacts / Test rocthrust / "
            "Test rocthrust (shard 1/1) (gfx110X-all)"
        )

        failure = normalize_failed_job(name)

        assert failure.display_name == "Windows · gfx110X-all · rocthrust"

    def test_friendly_name_keeps_xfail_as_a_suffix(self):
        name = (
            "Linux::asan / Test gfx94X / Test hip-tests (PAL) / "
            "Test hip-tests (PAL) (shard 1/4) (gfx94X) (xfail)"
        )

        failure = normalize_failed_job(name)

        assert failure.display_name == "Linux · gfx94X · hip-tests (PAL) (xfail)"

    @pytest.mark.parametrize(
        "name",
        [
            "Build Multi-Arch Stages",
            "Test hipblaslt (shard 2/1) (gfx94X-dcgpu)",
            "Test hipblaslt (shard 0/4) (gfx94X-dcgpu)",
            "Test hipblaslt (shard 1/4)unexpected (gfx94X-dcgpu)",
            "Test hipblaslt (shard 1/4) then (shard 2/4) (gfx94X-dcgpu)",
        ],
    )
    def test_unknown_or_invalid_shapes_are_lossless(self, name):
        assert normalize_failed_job(name) == LogicalFailure(name, None, name)


class TestGroupFailedJobs:
    def test_collapses_shards_by_normalized_full_name_and_total(self):
        prefix = "Linux::release / Test gfx94X-dcgpu / Test hipblaslt / Test hipblaslt"
        jobs = [
            FailedJobInfo(f"{prefix} (shard 1/6) (gfx94X-dcgpu)", "shard-1"),
            FailedJobInfo(f"{prefix} (shard 5/6) (gfx94X-dcgpu)", "shard-5"),
        ]

        assert group_failed_jobs(jobs) == [
            LogicalFailure(
                normalized_name=f"{prefix} (gfx94X-dcgpu)",
                total_shards=6,
                display_name="Linux · gfx94X-dcgpu · hipblaslt",
            )
        ]

    def test_different_totals_and_full_paths_remain_distinct(self):
        jobs = [
            FailedJobInfo(
                "Linux::release / Path A / Test hipblaslt (shard 1/4) (gfx94X)",
                "",
            ),
            FailedJobInfo(
                "Linux::release / Path A / Test hipblaslt (shard 1/6) (gfx94X)",
                "",
            ),
            FailedJobInfo(
                "Linux::release / Path B / Test hipblaslt (shard 1/4) (gfx94X)",
                "",
            ),
        ]

        failures = group_failed_jobs(jobs)

        assert len(failures) == 3
        assert all(
            failure.display_name == failure.normalized_name for failure in failures
        )

    def test_does_not_collapse_same_named_non_sharded_jobs(self):
        jobs = [
            FailedJobInfo("Build", "first"),
            FailedJobInfo("Build", "second"),
        ]

        assert len(group_failed_jobs(jobs)) == 2

    def test_omits_aggregate_summary_job(self):
        jobs = [
            FailedJobInfo("Linux test", ""),
            FailedJobInfo("Multi-Arch CI Summary", ""),
        ]

        assert group_failed_jobs(jobs) == [
            LogicalFailure("Linux test", None, "Linux test")
        ]


class TestRenderStepSummary:
    def test_success_is_terse(self):
        summary = render_step_summary(parse_needs_json(json.dumps(ALL_SUCCESS)), None)

        assert summary == (
            "## Workflow result\n\n" "✅ Workflow jobs completed: 3 succeeded."
        )

    def test_success_distinguishes_skips_and_allowed_failures(self):
        jobs = parse_needs_json(
            json.dumps(
                {
                    "build": {"result": "success"},
                    "optional": {
                        "result": "failure",
                        "outputs": {"continue_on_error": "true"},
                    },
                    "not_selected": {"result": "skipped"},
                }
            )
        )

        summary = render_step_summary(jobs, None)

        assert "1 succeeded · 1 skipped · 1 allowed failure" in summary
        assert "All required jobs succeeded" not in summary

    def test_groups_shards_and_links_only_to_overall_run(self):
        jobs = parse_needs_json(json.dumps(ONE_FAILURE))
        failed_jobs = [
            FailedJobInfo(
                "Linux::release / Test gfx94X / Test hipblaslt / "
                "Test hipblaslt (shard 1/6) (gfx94X)",
                "https://github.com/test/shard/1",
            ),
            FailedJobInfo(
                "Linux::release / Test gfx94X / Test hipblaslt / "
                "Test hipblaslt (shard 4/6) (gfx94X)",
                "https://github.com/test/shard/4",
            ),
        ]

        summary = render_step_summary(jobs, failed_jobs, "ROCm/TheRock", "123")

        assert summary.count("Linux · gfx94X · hipblaslt") == 1
        assert "shard" not in summary
        assert "https://github.com/test/shard" not in summary
        assert summary.count("[View workflow run]") == 1
        assert "https://github.com/ROCm/TheRock/actions/runs/123" in summary

    def test_retains_cancelled_job_with_granular_failures(self):
        jobs = [
            JobResult("linux", "failure", False),
            JobResult("windows", "cancelled", False),
        ]
        failed_jobs = [FailedJobInfo("Linux build", "")]

        summary = render_step_summary(jobs, failed_jobs)

        assert "Linux build" in summary
        assert "windows — cancelled" in summary
        assert "1 cancelled · 1 failure" in summary

    def test_surfaces_timed_out_logical_failure(self):
        jobs = [JobResult("linux", "failure", False)]
        failed_jobs = [
            FailedJobInfo(
                "Linux::release / Test gfx94X / Test hipblaslt / "
                "Test hipblaslt (shard 1/6) (gfx94X)",
                "",
                conclusion="timed_out",
            )
        ]

        summary = render_step_summary(jobs, failed_jobs)

        assert "Linux · gfx94X · hipblaslt — timed out" in summary

    def test_falls_back_to_top_level_jobs_when_details_unavailable(self):
        jobs = [JobResult("linux_build_and_test", "failure", False)]

        summary = render_step_summary(jobs, None)

        assert "linux\\_build\\_and\\_test — failure" in summary

    def test_escapes_untrusted_job_names(self):
        jobs = [JobResult("top", "failure", False)]
        failed_jobs = [
            FailedJobInfo("<b>@team [x](url) | *boom*\n# heading", "ignored")
        ]

        summary = render_step_summary(jobs, failed_jobs)

        assert "<b>" not in summary
        assert "@team" not in summary
        assert "&lt;b&gt;&#64;team" in summary
        assert r"\[x\]\(url\) \| \*boom\*" in summary
        assert "&#10;\\# heading" in summary

    def test_caps_logical_failures(self):
        jobs = [JobResult("top", "failure", False)]
        failed_jobs = [
            FailedJobInfo(f"Failed job {index:02}", "") for index in range(22)
        ]

        summary = render_step_summary(jobs, failed_jobs)

        assert "Failed job 19" in summary
        assert "Failed job 20" not in summary
        assert "and 2 more failure(s) not shown" in summary

    def test_caps_untrusted_job_name_length(self):
        jobs = [JobResult("top", "failure", False)]
        failed_jobs = [FailedJobInfo("x" * 500, "")]

        summary = render_step_summary(jobs, failed_jobs)

        assert "x" * 200 not in summary
        assert "…" in summary


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_all_success_returns_zero(self, capsys):
        rc = main(["--needs-json", json.dumps(ALL_SUCCESS)])
        assert rc == 0
        assert "succeeded" in capsys.readouterr().out

    def test_all_success_writes_step_summary(self, tmp_path, monkeypatch):
        summary_path = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", os.fspath(summary_path))

        rc = main(["--needs-json", json.dumps(ALL_SUCCESS)])

        assert rc == 0
        assert summary_path.read_text() == (
            "## Workflow result\n\n" "✅ Workflow jobs completed: 3 succeeded.\n\n"
        )

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
        with patch(
            "workflow_summary.gha_send_request", return_value=api_response
        ) as request:
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
        request.assert_called_once()
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

    @pytest.mark.parametrize(
        ("needs", "expected_rc"),
        [(ALL_SUCCESS, 0), (ONE_FAILURE, 1)],
    )
    @pytest.mark.parametrize(
        "failing_function", ["render_step_summary", "gha_append_step_summary"]
    )
    def test_summary_errors_do_not_change_gate_result(
        self, needs, expected_rc, failing_function, capsys
    ):
        with patch(
            f"workflow_summary.{failing_function}", side_effect=OSError("disk full")
        ):
            rc = main(["--needs-json", json.dumps(needs)])

        assert rc == expected_rc
        assert "Could not write workflow summary: disk full" in capsys.readouterr().out
