# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest

# Add github_actions to path so wait_for_baseline_stage is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from github_actions_api import GitHubAPIError
from wait_for_baseline_stage import (
    MAX_CONSECUTIVE_API_ERRORS,
    MAX_PAGES,
    STAGE_JOB_NAMES,
    State,
    evaluate,
    expected_job_names,
    list_run_jobs,
    main,
    parse_families,
    wait_for_stage,
)
from workflow_utils import WORKFLOWS_DIR, load_workflow

LINUX_STAGES = "Linux::release / Build Multi-Arch Stages"
WINDOWS_STAGES = "Windows::release / Build Multi-Arch Stages"
COVERAGE = "Linux::CodeCoverage"
GFX94X = "Stage - Math Libs (gfx94X-dcgpu)"
GFX950 = "Stage - Math Libs (gfx950-dcgpu)"


def job(name, status="completed", conclusion="success", job_id=1):
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion if status == "completed" else None,
        "html_url": f"https://github.com/ROCm/rocm-libraries/actions/runs/1/job/{job_id}",
    }


def math_libs(family, prefix=LINUX_STAGES, **kwargs):
    return job(
        f"{prefix} / math-libs ({family}, linux-gfx942-1gpu) / Stage - Math Libs ({family})",
        **kwargs,
    )


SETUP = job("setup / setup", job_id=100)
COMPILER_RUNTIME = job(
    f"{LINUX_STAGES} / compiler-runtime / Stage - Compiler Runtime", job_id=101
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class TestParseFamilies:
    def test_accepts_semicolons_and_commas(self):
        assert parse_families("gfx94X-dcgpu; gfx950-dcgpu,gfx1151") == [
            "gfx94X-dcgpu",
            "gfx950-dcgpu",
            "gfx1151",
        ]

    def test_empty(self):
        assert parse_families(" ; ") == []


class TestExpectedJobNames:
    def test_math_libs_names_a_job_per_family(self):
        assert expected_job_names("math-libs", ["gfx94X-dcgpu", "gfx950-dcgpu"]) == [
            GFX94X,
            GFX950,
        ]

    def test_compiler_runtime_is_one_job_for_every_family(self):
        assert expected_job_names("compiler-runtime", ["gfx94X-dcgpu"]) == [
            "Stage - Compiler Runtime"
        ]


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_ready_once_the_job_succeeds(self):
        state, lines = evaluate(
            [SETUP, COMPILER_RUNTIME, math_libs("gfx94X-dcgpu")], "math-libs", [GFX94X]
        )
        assert state is State.READY
        assert lines == [
            f"{GFX94X}: success https://github.com/ROCm/rocm-libraries/actions/runs/1/job/1"
        ]

    def test_waits_while_the_job_runs(self):
        jobs = [SETUP, math_libs("gfx94X-dcgpu", status="in_progress")]
        state, lines = evaluate(jobs, "math-libs", [GFX94X])
        assert state is State.WAITING
        assert lines[0].startswith(f"{GFX94X}: in_progress ")

    def test_waits_before_the_stage_starts(self):
        state, lines = evaluate(
            [SETUP, job(COMPILER_RUNTIME["name"], status="in_progress")],
            "math-libs",
            [GFX94X],
        )
        assert state is State.WAITING
        assert lines == [f"{GFX94X}: not started"]

    @pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped"])
    def test_fails_when_the_job_does_not_succeed(self, conclusion):
        jobs = [math_libs("gfx94X-dcgpu", conclusion=conclusion)]
        state, lines = evaluate(jobs, "math-libs", [GFX94X])
        assert state is State.FAILED
        assert lines[0].startswith(f"{GFX94X}: {conclusion} ")

    def test_ignores_windows_jobs_with_the_same_name(self):
        windows_done = math_libs("gfx94X-dcgpu", prefix=WINDOWS_STAGES, job_id=2)
        linux_running = math_libs("gfx94X-dcgpu", status="in_progress")
        state, _ = evaluate([windows_done, linux_running], "math-libs", [GFX94X])
        assert state is State.WAITING

        windows_failed = math_libs(
            "gfx94X-dcgpu", prefix=WINDOWS_STAGES, conclusion="failure", job_id=2
        )
        state, _ = evaluate(
            [windows_failed, math_libs("gfx94X-dcgpu")], "math-libs", [GFX94X]
        )
        assert state is State.READY

    @pytest.mark.parametrize(
        "caller",
        ["math-libs", "math-libs (gfx94X-dcgpu, linux-gfx942-1gpu)"],
    )
    def test_fails_when_the_stage_did_not_run(self, caller):
        skipped_call = job(f"{LINUX_STAGES} / {caller}", conclusion="skipped")
        state, lines = evaluate(
            [SETUP, job(COMPILER_RUNTIME["name"], conclusion="failure"), skipped_call],
            "math-libs",
            [GFX94X],
        )
        assert state is State.FAILED
        assert lines[0].startswith(f"{GFX94X}: will not run; the math-libs stage was")

    def test_fails_when_the_run_builds_only_other_families(self):
        state, lines = evaluate(
            [math_libs("gfx950-dcgpu", status="in_progress")], "math-libs", [GFX94X]
        )
        assert state is State.FAILED
        assert lines == [f"{GFX94X}: not built by this run, which builds {GFX950}"]

    def test_coverage_stage_jobs_are_not_the_baseline(self):
        coverage_job = job(
            f"{COVERAGE} / Build Instrumented Math Libs (gfx94X-dcgpu)"
            " / Stage - Coverage Math Libs (gfx94X-dcgpu)",
            job_id=3,
        )
        state, lines = evaluate([coverage_job], "math-libs", [GFX94X])
        assert state is State.WAITING
        assert lines == [f"{GFX94X}: not started"]

    def test_waits_for_every_family(self):
        names = [GFX94X, GFX950]
        partial = [
            math_libs("gfx94X-dcgpu"),
            math_libs("gfx950-dcgpu", status="queued", job_id=2),
        ]
        assert evaluate(partial, "math-libs", names)[0] is State.WAITING

        done = [math_libs("gfx94X-dcgpu"), math_libs("gfx950-dcgpu", job_id=2)]
        assert evaluate(done, "math-libs", names)[0] is State.READY

    def test_compiler_runtime_stage(self):
        windows_running = job(
            f"{WINDOWS_STAGES} / compiler-runtime / Stage - Compiler Runtime",
            status="in_progress",
            job_id=2,
        )
        state, _ = evaluate(
            [COMPILER_RUNTIME, windows_running],
            "compiler-runtime",
            ["Stage - Compiler Runtime"],
        )
        assert state is State.READY


# ---------------------------------------------------------------------------
# list_run_jobs()
# ---------------------------------------------------------------------------


class TestListRunJobs:
    def test_paginates(self):
        page1 = {"jobs": [job(f"job {i}", job_id=i) for i in range(100)]}
        page2 = {"jobs": [job("last", job_id=100)]}
        with patch(
            "wait_for_baseline_stage.gha_send_request", side_effect=[page1, page2]
        ) as send:
            jobs = list_run_jobs("ROCm/rocm-libraries", "123")
        assert len(jobs) == 101
        urls = [c.args[0] for c in send.call_args_list]
        assert urls == [
            "https://api.github.com/repos/ROCm/rocm-libraries/actions/runs/123/jobs"
            f"?filter=latest&per_page=100&page={page}"
            for page in (1, 2)
        ]

    def test_raises_past_the_page_limit(self):
        full_page = {"jobs": [job(f"job {i}", job_id=i) for i in range(100)]}
        with patch(
            "wait_for_baseline_stage.gha_send_request", return_value=full_page
        ) as send:
            with pytest.raises(GitHubAPIError):
                list_run_jobs("ROCm/rocm-libraries", "123")
        assert send.call_count == MAX_PAGES


# ---------------------------------------------------------------------------
# wait_for_stage()
# ---------------------------------------------------------------------------

WAITING = [math_libs("gfx94X-dcgpu", status="in_progress")]
READY = [math_libs("gfx94X-dcgpu")]
FAILED = [math_libs("gfx94X-dcgpu", conclusion="failure")]


def run_wait(snapshots, monotonic=None, timeout_minutes=60):
    clock = {"side_effect": monotonic} if monotonic else {"return_value": 0}
    with (
        patch("wait_for_baseline_stage.list_run_jobs", side_effect=snapshots),
        patch("wait_for_baseline_stage.time.monotonic", **clock),
        patch("wait_for_baseline_stage.time.sleep") as sleep,
    ):
        result = wait_for_stage(
            "ROCm/rocm-libraries",
            "123",
            "math-libs",
            [GFX94X],
            poll_seconds=180,
            timeout_minutes=timeout_minutes,
        )
    return result, sleep


class TestWaitForStage:
    def test_returns_zero_once_ready(self, capsys):
        result, sleep = run_wait([WAITING, WAITING, READY])
        assert result == 0
        assert sleep.call_args_list == [call(180), call(180)]
        # Only status changes are logged.
        assert capsys.readouterr().out.count("Baseline math-libs") == 2

    def test_fails_once_the_next_poll_confirms_a_failure(self, capsys):
        result, sleep = run_wait([FAILED, FAILED])
        assert result == 1
        assert sleep.call_count == 1
        assert "::error::The baseline math-libs build did not succeed" in (
            capsys.readouterr().out
        )

    def test_unconfirmed_failure_keeps_waiting(self):
        other_family_only = [math_libs("gfx950-dcgpu", status="in_progress")]
        result, _ = run_wait([other_family_only, WAITING, READY])
        assert result == 0

    def test_times_out(self, capsys):
        # Deadline at 600s; the first check passes, the second is past it.
        result, sleep = run_wait(
            [WAITING, WAITING], monotonic=[0, 0, 601], timeout_minutes=10
        )
        assert result == 1
        assert sleep.call_count == 1
        assert "::error::Timed out after 10 minutes" in capsys.readouterr().out

    def test_gives_up_after_consecutive_api_errors(self):
        errors = [GitHubAPIError("boom")] * MAX_CONSECUTIVE_API_ERRORS
        result, sleep = run_wait(errors)
        assert result == 1
        assert sleep.call_count == MAX_CONSECUTIVE_API_ERRORS - 1

    def test_recovers_from_api_errors(self):
        result, _ = run_wait([GitHubAPIError("boom"), READY])
        assert result == 0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_math_libs_needs_families(self):
        with pytest.raises(SystemExit) as exc:
            main(
                ["--stage=math-libs", "--repository=ROCm/rocm-libraries", "--run-id=1"]
            )
        assert exc.value.code == 2

    def test_needs_repository_and_run_id(self, monkeypatch):
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        with pytest.raises(SystemExit) as exc:
            main(["--stage=compiler-runtime"])
        assert exc.value.code == 2

    def test_waits_for_the_named_jobs(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "ROCm/rocm-libraries")
        monkeypatch.setenv("GITHUB_RUN_ID", "123")
        with patch(
            "wait_for_baseline_stage.wait_for_stage", return_value=0
        ) as wait_mock:
            assert (
                main(
                    ["--stage=math-libs", "--amdgpu-families=gfx94X-dcgpu;gfx950-dcgpu"]
                )
                == 0
            )
        wait_mock.assert_called_once_with(
            "ROCm/rocm-libraries", "123", "math-libs", [GFX94X, GFX950], 180, 300
        )


# ---------------------------------------------------------------------------
# Workflow drift
# ---------------------------------------------------------------------------


class TestStageJobNames:
    def test_match_the_regular_linux_build(self):
        jobs = load_workflow(WORKFLOWS_DIR / "multi_arch_build_portable_linux.yml")[
            "jobs"
        ]
        for stage, template in STAGE_JOB_NAMES.items():
            stage_inputs = jobs[stage]["with"]
            assert stage_inputs["stage_name"] == stage
            assert stage_inputs["stage_display_name"] == template.format(
                amdgpu_family="${{ matrix.family_info.amdgpu_family }}"
            )

    def test_coverage_stage_jobs_are_named_apart(self):
        jobs = load_workflow(WORKFLOWS_DIR / "multi_arch_ci_coverage_linux.yml")["jobs"]
        prefixes = [template.split("{")[0] for template in STAGE_JOB_NAMES.values()]
        coverage_names = [
            job_def["with"]["stage_display_name"]
            for job_def in jobs.values()
            if "stage_display_name" in job_def.get("with", {})
        ]
        assert coverage_names
        for name in coverage_names:
            assert not any(name.startswith(prefix) for prefix in prefixes), name
