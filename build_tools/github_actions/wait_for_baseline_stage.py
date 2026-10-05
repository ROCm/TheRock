# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Waits for this run's regular Linux build to finish a stage.

multi_arch_ci_coverage_linux.yml tests each instrumented project against a
non-instrumented baseline that it installs from S3. When the caller builds that
baseline in the same run (the rocm-libraries nightly does), coverage can start
alongside the regular build instead of after it, as long as its tests wait for
the baseline. A reusable workflow cannot `needs:` a job inside another one, so
this polls the run's jobs and exits 0 once the regular build's stage job has
succeeded for every requested family.

It exits 1 once one of those jobs fails, is cancelled or is skipped, or once
the run shows the job will never start: the regular build skipped the stage
(an earlier stage failed, or the caller marked it prebuilt or skipped), or
built it only for other families.

On a public repository, listing the jobs needs no `actions` permission.

Usage:

    python build_tools/github_actions/wait_for_baseline_stage.py \\
        --stage=math-libs --amdgpu-families="gfx94X-dcgpu"
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import os
import re
import sys
import time

from github_actions_api import GitHubAPIError, gha_send_request

# stage_display_name of each stage's job in multi_arch_build_portable_linux.yml,
# which keys every stage's job by the stage name.
STAGE_JOB_NAMES = {
    "compiler-runtime": "Stage - Compiler Runtime",
    "math-libs": "Stage - Math Libs ({amdgpu_family})",
}

JOBS_PER_PAGE = 100
MAX_PAGES = 20
REQUEST_TIMEOUT_SECONDS = 60
MAX_CONSECUTIVE_API_ERRORS = 5


class State(Enum):
    READY = "ready"
    WAITING = "waiting"
    FAILED = "failed"


def log(*args):
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{timestamp}]", *args, flush=True)


def parse_families(value: str) -> list[str]:
    """Splits a comma- or semicolon-separated family list."""
    return [family.strip() for family in re.split(r"[,;]", value) if family.strip()]


def expected_job_names(stage: str, amdgpu_families: list[str]) -> list[str]:
    template = STAGE_JOB_NAMES[stage]
    if "{amdgpu_family}" not in template:
        return [template]
    return [template.format(amdgpu_family=family) for family in amdgpu_families]


def list_run_jobs(repository: str, run_id: str) -> list[dict]:
    """Returns the latest attempt of every job in the run."""
    jobs: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = (
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs"
            f"?filter=latest&per_page={JOBS_PER_PAGE}&page={page}"
        )
        response = gha_send_request(url, timeout_seconds=REQUEST_TIMEOUT_SECONDS)
        page_jobs = response.get("jobs", [])
        jobs.extend(page_jobs)
        if len(page_jobs) < JOBS_PER_PAGE:
            return jobs
    raise GitHubAPIError(f"Run {run_id} has more than {len(jobs)} jobs")


def _name_parts(job: dict) -> list[str]:
    # Jobs inside reusable workflows are named "<caller> / ... / <job>".
    return job.get("name", "").split(" / ")


def _is_windows(job: dict) -> bool:
    # Windows builds reuse the stage job names. multi_arch_ci.yml and the
    # rocm-libraries nightly name the top-level job Windows::<variant>.
    return _name_parts(job)[0].lower().startswith("windows")


def _describe(job: dict) -> str:
    return f"{job.get('conclusion') or job.get('status')} {job.get('html_url', '')}"


def _did_not_succeed(job: dict) -> bool:
    return job.get("status") == "completed" and job.get("conclusion") != "success"


def evaluate(
    jobs: list[dict], stage: str, job_names: list[str]
) -> tuple[State, list[str]]:
    """Returns whether every job in job_names succeeded, with a line per job."""
    jobs_by_leaf: dict[str, list[dict]] = {}
    for job in jobs:
        if not _is_windows(job):
            jobs_by_leaf.setdefault(_name_parts(job)[-1], []).append(job)

    # A reusable-workflow call that did not run is listed as a single job named
    # after the caller: the stage name, plus the matrix values once expanded.
    stage_not_run = [
        job
        for leaf, leaf_jobs in jobs_by_leaf.items()
        if leaf == stage or leaf.startswith(f"{stage} (")
        for job in leaf_jobs
        if _did_not_succeed(job)
    ]
    template = STAGE_JOB_NAMES[stage]
    family_prefix = template.split("{")[0] if "{" in template else None
    built_names = sorted(
        leaf
        for leaf in jobs_by_leaf
        if family_prefix and leaf.startswith(family_prefix)
    )

    lines = []
    failed = waiting = False
    for name in job_names:
        matches = jobs_by_leaf.get(name, [])
        unsuccessful = [job for job in matches if _did_not_succeed(job)]
        unfinished = [job for job in matches if job.get("status") != "completed"]
        if unsuccessful:
            failed = True
            lines.extend(f"{name}: {_describe(job)}" for job in unsuccessful)
        elif unfinished:
            waiting = True
            lines.extend(f"{name}: {_describe(job)}" for job in unfinished)
        elif matches:
            lines.extend(f"{name}: {_describe(job)}" for job in matches)
        elif stage_not_run:
            failed = True
            lines.append(
                f"{name}: will not run; the {stage} stage was "
                f"{_describe(stage_not_run[0])}"
            )
        elif built_names:
            failed = True
            lines.append(
                f"{name}: not built by this run, which builds {', '.join(built_names)}"
            )
        else:
            waiting = True
            lines.append(f"{name}: not started")

    if failed:
        return State.FAILED, lines
    if waiting:
        return State.WAITING, lines
    return State.READY, lines


def wait_for_stage(
    repository: str,
    run_id: str,
    stage: str,
    job_names: list[str],
    poll_seconds: int,
    timeout_minutes: int,
) -> int:
    deadline = time.monotonic() + timeout_minutes * 60
    api_errors = 0
    last_lines: list[str] | None = None
    unconfirmed_failure: list[str] | None = None
    while True:
        try:
            jobs = list_run_jobs(repository, run_id)
        except GitHubAPIError as e:
            api_errors += 1
            log(f"Could not list jobs ({api_errors}/{MAX_CONSECUTIVE_API_ERRORS}): {e}")
            if api_errors >= MAX_CONSECUTIVE_API_ERRORS:
                print("::error::Gave up listing this run's jobs")
                return 1
        else:
            api_errors = 0
            state, lines = evaluate(jobs, stage, job_names)
            if lines != last_lines:
                log(f"Baseline {stage} {state.value}:")
                for line in lines:
                    print(f"  {line}", flush=True)
                last_lines = lines
            if state is State.READY:
                return 0
            if state is State.FAILED:
                # A snapshot spans several API pages, so act on a failure only
                # once the next poll shows it too.
                if lines == unconfirmed_failure:
                    print(
                        f"::error::The baseline {stage} build did not succeed;"
                        " coverage tests would run against missing artifacts"
                    )
                    return 1
                unconfirmed_failure = lines
            else:
                unconfirmed_failure = None

        if time.monotonic() >= deadline:
            print(
                f"::error::Timed out after {timeout_minutes} minutes waiting for"
                f" the baseline {stage} build"
            )
            return 1
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wait for this run's regular Linux build to finish a stage."
    )
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_JOB_NAMES))
    parser.add_argument(
        "--amdgpu-families",
        default="",
        help="Comma- or semicolon-separated families; required for math-libs.",
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="Repository in owner/repo format (default: $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", ""),
        help="Workflow run ID (default: $GITHUB_RUN_ID).",
    )
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--timeout-minutes", type=int, default=300)
    args = parser.parse_args(argv)

    if not args.repository or not args.run_id:
        parser.error("--repository and --run-id are required outside GitHub Actions")
    job_names = expected_job_names(args.stage, parse_families(args.amdgpu_families))
    if not job_names:
        parser.error(f"--stage={args.stage} needs --amdgpu-families")

    return wait_for_stage(
        args.repository,
        args.run_id,
        args.stage,
        job_names,
        args.poll_seconds,
        args.timeout_minutes,
    )


if __name__ == "__main__":
    sys.exit(main())
