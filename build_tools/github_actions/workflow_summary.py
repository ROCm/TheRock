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
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from urllib.parse import quote

from github_actions_api import (
    GitHubAPIError,
    gha_append_step_summary,
    gha_send_request,
    str2bool,
)


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
    conclusion: str = "failure"


@dataclass(frozen=True)
class LogicalFailure:
    """A failed job with shard-specific details removed."""

    normalized_name: str
    total_shards: int | None
    display_name: str
    conclusions: tuple[str, ...] = ("failure",)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

# Job results that are treated as acceptable (not a failure).
_ACCEPTABLE_RESULTS = frozenset({"success", "skipped"})
_PROBLEM_JOB_CONCLUSIONS = frozenset(
    {"action_required", "failure", "stale", "startup_failure", "timed_out"}
)
_MAX_SUMMARY_FAILURES = 20
_MAX_SUMMARY_NAME_LENGTH = 200
_SHARD_TOKEN_RE = re.compile(
    r" \(shard (?P<shard>[1-9][0-9]*)/(?P<total>[1-9][0-9]*)\)(?= |$)"
)
_FRIENDLY_TEST_RE = re.compile(r"^Test (?P<component>.+) \((?P<family>[^()\r\n]+)\)$")
_PLATFORM_RE = re.compile(r"^(?P<platform>Linux|Windows)(?:::[^/\r\n]+)+$")
_AGGREGATE_SUMMARY_JOB_NAMES = frozenset(
    {
        "CI Summary",
        "Multi-Arch CI ASAN Nightly Summary",
        "Multi-Arch CI ASAN Summary",
        "Multi-Arch CI Summary",
        "Unit Tests Summary",
    }
)


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
        conclusion = job.get("conclusion")
        if conclusion in _PROBLEM_JOB_CONCLUSIONS:
            failed.append(
                FailedJobInfo(
                    name=job.get("name", "unknown"),
                    html_url=job.get("html_url", ""),
                    conclusion=conclusion,
                )
            )
    return failed


def _print_failed_sub_job_urls(
    github_repository: str, github_run_id: str
) -> list[FailedJobInfo] | None:
    """Best-effort: fetch and print failed sub-jobs, returning what was found."""
    try:
        failed_jobs = fetch_failed_jobs(github_repository, github_run_id)
    except GitHubAPIError as e:
        print(f"\n  (Could not fetch job details: {e})")
        return None

    if not failed_jobs:
        return []

    print(f"\n{_RED}Failed sub-jobs:{_RESET}")
    for job in failed_jobs:
        detail = "" if job.conclusion == "failure" else f": {job.conclusion}"
        print(f"  {_RED}{job.name}{detail}{_RESET}")
        if job.html_url:
            print(f"    {job.html_url}")
    return failed_jobs


def normalize_failed_job(name: str) -> LogicalFailure:
    """Remove one valid shard token from a failed job name.

    Grouping uses the full normalized name and total shard count. A friendly
    platform/family/component label is used only when the normalized terminal
    segment exactly matches the test job convention. Unknown shapes otherwise
    retain their complete names.
    """
    matches = list(_SHARD_TOKEN_RE.finditer(name))
    if len(matches) != 1:
        return LogicalFailure(name, None, name)

    match = matches[0]
    shard = int(match.group("shard"))
    total = int(match.group("total"))
    if shard > total:
        return LogicalFailure(name, None, name)

    normalized_name = name[: match.start()] + name[match.end() :]
    parts = normalized_name.split(" / ")
    terminal = parts[-1]
    xfail = ""
    if terminal.endswith(" (xfail)"):
        terminal = terminal.removesuffix(" (xfail)")
        xfail = " (xfail)"
    terminal_match = _FRIENDLY_TEST_RE.fullmatch(terminal)

    platform_match = _PLATFORM_RE.fullmatch(parts[0])
    if platform_match is None or terminal_match is None:
        display_name = normalized_name
    else:
        platform = platform_match.group("platform")
        family = terminal_match.group("family")
        component = terminal_match.group("component")
        display_name = f"{platform} · {family} · {component}{xfail}"

    return LogicalFailure(normalized_name, total, display_name)


def group_failed_jobs(failed_jobs: list[FailedJobInfo]) -> list[LogicalFailure]:
    """Collapse failed shards without conflating distinct job configurations."""
    shard_groups: dict[tuple[str, int], tuple[LogicalFailure, set[str]]] = {}
    failures: list[LogicalFailure] = []
    for job in failed_jobs:
        # A completed run reports this summary job as failed because the script
        # exits non-zero. It is the consequence of the failures below, not a
        # separate failed area.
        if job.name in _AGGREGATE_SUMMARY_JOB_NAMES:
            continue
        logical_failure = normalize_failed_job(job.name)
        if logical_failure.total_shards is None:
            # Duplicate names are legal for non-sharded jobs. Preserve each
            # one; only recognized shard siblings may be collapsed.
            failures.append(
                LogicalFailure(
                    logical_failure.normalized_name,
                    None,
                    logical_failure.display_name,
                    (job.conclusion,),
                )
            )
            continue

        key = (logical_failure.normalized_name, logical_failure.total_shards)
        if key not in shard_groups:
            shard_groups[key] = (logical_failure, set())
        shard_groups[key][1].add(job.conclusion)

    failures.extend(
        LogicalFailure(
            failure.normalized_name,
            failure.total_shards,
            failure.display_name,
            tuple(sorted(conclusions)),
        )
        for failure, conclusions in shard_groups.values()
    )

    display_counts: dict[str, int] = {}
    for failure in failures:
        display_counts[failure.display_name] = (
            display_counts.get(failure.display_name, 0) + 1
        )
    failures = [
        LogicalFailure(
            failure.normalized_name,
            failure.total_shards,
            (
                failure.normalized_name
                if display_counts[failure.display_name] > 1
                else failure.display_name
            ),
            failure.conclusions,
        )
        for failure in failures
    ]

    return sorted(
        failures,
        key=lambda failure: (
            failure.display_name.casefold(),
            failure.normalized_name.casefold(),
            failure.total_shards or 0,
        ),
    )


def _escape_markdown(value: str) -> str:
    """Escape untrusted job names for a Markdown/HTML summary."""
    if len(value) > _MAX_SUMMARY_NAME_LENGTH:
        value = value[: _MAX_SUMMARY_NAME_LENGTH - 1] + "…"
    escaped = re.sub(r"([\\`*_[\]{}()#+.!|\-])", r"\\\1", value)
    escaped = html.escape(escaped, quote=False)
    escaped = escaped.replace("@", "&#64;")
    escaped = escaped.replace("\r", "&#13;").replace("\n", "&#10;")
    return escaped


def _workflow_run_url(github_repository: str, github_run_id: str) -> str:
    """Build a safe URL for the overall workflow run."""
    repository = quote(github_repository, safe="/")
    run_id = quote(github_run_id, safe="")
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def _count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    """Return a compact, grammatically correct result count."""
    return f"{count} {singular if count == 1 else plural or singular}"


def _logical_failure_label(failure: LogicalFailure) -> str:
    non_failure_conclusions = [
        conclusion.replace("_", " ")
        for conclusion in failure.conclusions
        if conclusion != "failure"
    ]
    if not non_failure_conclusions:
        return failure.display_name
    return f"{failure.display_name} — {', '.join(non_failure_conclusions)}"


def render_step_summary(
    jobs: list[JobResult],
    failed_sub_jobs: list[FailedJobInfo] | None,
    github_repository: str = "",
    github_run_id: str = "",
) -> str:
    """Render a compact Markdown workflow result."""
    failed, _ = evaluate_results(jobs)
    if not failed:
        succeeded = sum(job.result == "success" for job in jobs)
        skipped = sum(job.result == "skipped" for job in jobs)
        allowed = sum(
            job.result not in _ACCEPTABLE_RESULTS and job.continue_on_error
            for job in jobs
        )
        counts = []
        if succeeded:
            counts.append(_count_phrase(succeeded, "succeeded"))
        if skipped:
            counts.append(_count_phrase(skipped, "skipped"))
        if allowed:
            counts.append(_count_phrase(allowed, "allowed failure", "allowed failures"))
        result_counts = " · ".join(counts) if counts else "no jobs reported"
        return f"## Workflow result\n\n✅ Workflow jobs completed: {result_counts}."

    result_counts: dict[str, int] = {}
    for job in failed:
        result_counts[job.result] = result_counts.get(job.result, 0) + 1
    result_text = " · ".join(
        _count_phrase(
            count,
            "failure" if result == "failure" else result,
            "failures" if result == "failure" else result,
        )
        for result, count in sorted(result_counts.items())
    )
    lines = [
        "## Workflow result",
        "",
        f"❌ Workflow jobs did not succeed: {_escape_markdown(result_text)}.",
        "",
    ]

    logical_failures = group_failed_jobs(failed_sub_jobs or [])
    entries: list[str] = []

    # The jobs API reports only conclusions of "failure". Preserve other
    # unacceptable top-level results, especially cancellations, alongside any
    # granular failures returned by the API.
    for job in failed:
        if job.result != "failure":
            entries.append(f"{job.name} — {job.result}")

    entries.extend(_logical_failure_label(job) for job in logical_failures)

    # If granular details were unavailable or empty, retain every failed
    # top-level job and its result rather than producing an empty report.
    if not logical_failures:
        entries = [f"{job.name} — {job.result}" for job in failed]

    visible_entries = entries[:_MAX_SUMMARY_FAILURES]
    lines.extend(f"- {_escape_markdown(entry)}" for entry in visible_entries)
    omitted = len(entries) - len(visible_entries)
    if omitted:
        lines.append(f"- … and {omitted} more failure(s) not shown")

    if github_repository and github_run_id:
        lines.extend(
            [
                "",
                f"[View workflow run]({_workflow_run_url(github_repository, github_run_id)})",
            ]
        )

    return "\n".join(lines)


def _write_step_summary(
    jobs: list[JobResult],
    failed_sub_jobs: list[FailedJobInfo] | None,
    github_repository: str = "",
    github_run_id: str = "",
) -> None:
    """Best-effort: render and publish the compact workflow result."""
    try:
        summary = render_step_summary(
            jobs,
            failed_sub_jobs,
            github_repository,
            github_run_id,
        )
        gha_append_step_summary(summary, mirror_to_job_file=False)
    except Exception as e:
        print(f"\n  (Could not write workflow summary: {e})")


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
        failed_sub_jobs = None
        if args.github_repository and args.github_run_id:
            failed_sub_jobs = _print_failed_sub_job_urls(
                args.github_repository, args.github_run_id
            )

        _write_step_summary(
            jobs,
            failed_sub_jobs,
            args.github_repository,
            args.github_run_id,
        )

        return 1

    print(f"\n{_GREEN}All required jobs succeeded.{_RESET}")
    _write_step_summary(jobs, None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
