# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evaluate GitHub Actions workflow job results and produce a summary.

Call this script from a "_summary" job at the end of a workflow to get an
"anchor" job that can be used as a required check that includes all jobs in
the "needs:" array. Jobs can then be added or removed without needing to update
branch protection settings.

Usage in a workflow:

    ci_summary:
      if: always()
      needs: [setup, build, test]
      runs-on: ubuntu-24.04
      steps:
        - uses: actions/checkout@<sha>
        - name: Evaluate workflow results
          env:
            GITHUB_TOKEN: ${{ github.token }}
          run: |
            python build_tools/github_actions/workflow_summary.py \
              --needs-json '${{ toJSON(needs) }}'

Local testing (https://github.com/ROCm/TheRock/actions/runs/22879205184?pr=3865):

    ```
    python build_tools/github_actions/workflow_summary.py \
        --needs-json="{ \"setup\": { \"result\": \"success\" }, \"linux_build_and_test\": { \"result\": \"cancelled\" }, \"windows_build_and_test\": { \"result\": \"failure\" } }" \
        --github-repository=ROCm/TheRock \
        --github-run-id=22879205184
    ```

Notes:
  * Choose a name for the summary step that is unique across workflow files.
    ci.yml should use ci_summary, unit_tests.yml should use unit_tests_summary, etc.
    This ensures that required checks can be added in the github UI without
    the ambiguity of names overlapping.
  * Jobs skipped by "if" conditions are okay - they will not fail here.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

from github_actions_utils import GitHubAPIError, gha_send_request, str2bool


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    """Parsed result for a single upstream job."""

    name: str
    result: str
    continue_on_error: bool


@dataclass
class FailedJobInfo:
    """A failed sub-job from the GitHub API with a link to its log."""

    name: str
    html_url: str


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

# Job results that are treated as acceptable (not a failure).
_ACCEPTABLE_RESULTS = frozenset({"success", "skipped"})


def parse_needs_json(needs_json: str) -> list[JobResult]:
    """Parse the ``needs`` context JSON emitted by GitHub Actions.

    Args:
        needs_json: Raw JSON string from ``${{ toJSON(needs) }}``.

    Returns:
        A list of `JobResult` for each upstream job.
    """
    data = json.loads(needs_json)
    assert isinstance(data, dict), f"Expected a JSON object, got {type(data).__name__}"

    results: list[JobResult] = []
    for job_name, job_info in data.items():
        assert isinstance(job_info, dict), (
            f"Expected a JSON object for job '{job_name}', "
            f"got {type(job_info).__name__}"
        )
        result = job_info.get("result", "unknown")
        # The continue_on_error flag is conveyed as a job output string.
        outputs = job_info.get("outputs") or {}
        continue_on_error = str2bool(outputs.get("continue_on_error"))
        results.append(
            JobResult(
                name=job_name,
                result=result,
                continue_on_error=continue_on_error,
            )
        )
    return results


def evaluate_results(jobs: list[JobResult]) -> tuple[list[JobResult], list[JobResult]]:
    """Partition jobs into failed and ok lists.

    A job is considered *failed* if its result is not in
    ``{"success", "skipped"}`` and it did not set the ``continue_on_error``
    output to a truthy value.

    Returns:
        A ``(failed, ok)`` tuple of job lists.
    """
    failed: list[JobResult] = []
    ok: list[JobResult] = []
    for job in jobs:
        if job.result in _ACCEPTABLE_RESULTS:
            ok.append(job)
        elif job.continue_on_error:
            ok.append(job)
        else:
            failed.append(job)
    return failed, ok


def fetch_failed_jobs(
    github_repository: str, github_run_id: str
) -> list[FailedJobInfo]:
    """Fetch the list of failed sub-jobs from the GitHub Actions API.

    This gives more granular information than the ``needs`` context, which
    only sees top-level reusable workflow callers. The API returns all
    sub-jobs including those inside reusable workflows.

    Args:
        github_repository: Repository in "owner/repo" format.
        github_run_id: The workflow run ID.

    Returns:
        A list of failed jobs with names and URLs.
    """
    max_pages = 10
    all_jobs: list[dict] = []
    page = 1
    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{github_repository}"
            f"/actions/runs/{github_run_id}/jobs"
            f"?filter=latest&per_page=100&page={page}"
        )
        response = gha_send_request(url)
        jobs = response.get("jobs", [])
        all_jobs.extend(jobs)
        if len(jobs) < 100:
            break
        page += 1
    else:
        print(
            f"  Warning: fetched {len(all_jobs)} jobs across {max_pages} pages;"
            f" results may be incomplete."
        )

    failed: list[FailedJobInfo] = []
    for job in all_jobs:
        if job.get("conclusion") == "failure":
            failed.append(
                FailedJobInfo(
                    name=job.get("name", "unknown"),
                    html_url=job.get("html_url", ""),
                )
            )
    return failed


def _print_failed_sub_job_urls(github_repository: str, github_run_id: str) -> None:
    """Best-effort: fetch and print URLs for failed sub-jobs."""
    try:
        failed_jobs = fetch_failed_jobs(github_repository, github_run_id)
    except GitHubAPIError as e:
        print(f"\n  (Could not fetch job details: {e})")
        return

    if not failed_jobs:
        return

    print(f"\n{_RED}Failed sub-jobs:{_RESET}")
    for job in failed_jobs:
        print(f"  {_RED}{job.name}{_RESET}")
        if job.html_url:
            print(f"    {job.html_url}")


# ---------------------------------------------------------------------------
# ANSI colors (supported by GitHub Actions log output)
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"

_RESULT_COLORS: dict[str, str] = {
    "success": _GREEN,
    "skipped": _YELLOW,
    "failure": _RED,
    "cancelled": _RED,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate GitHub Actions workflow job results and produce a summary.",
    )
    parser.add_argument(
        "--needs-json",
        required=True,
        help="Raw JSON string from ${{ toJSON(needs) }}.",
    )
    parser.add_argument(
        "--github-repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="Repository in owner/repo format (default: $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--github-run-id",
        default=os.environ.get("GITHUB_RUN_ID", ""),
        help="Workflow run ID (default: $GITHUB_RUN_ID).",
    )
    args = parser.parse_args(argv)

    jobs = parse_needs_json(args.needs_json)
    failed, ok = evaluate_results(jobs)

    print(f"Checking status for {len(jobs)} job(s):")
    for job in jobs:
        color = _RESULT_COLORS.get(job.result, _RED)
        print(f"  {color}{job.name}: {job.result}{_RESET}")

    if failed:
        print(f"\n{_RED}The following job(s) failed:{_RESET}")
        for job in failed:
            print(f"  {_RED}{job.name}{_RESET}")

        # Try to fetch granular failure info from the API.
        if args.github_repository and args.github_run_id:
            _print_failed_sub_job_urls(args.github_repository, args.github_run_id)

        return 1

    print(f"\n{_GREEN}All required jobs succeeded.{_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
