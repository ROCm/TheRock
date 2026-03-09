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
          run: |
            python build_tools/github_actions/workflow_summary.py \
              --needs-json '${{ toJSON(needs) }}'

Notes:
  * Choose a name for the summary step that is unique across workflow files.
    ci.yml should use ci_summary, unit_tests.yml should use unit_tests_summary, etc.
    This ensures that required checks can be added in the github UI without
    the ambiguity of names overlapping.
  * Jobs skipped by "if" conditions are okay - they will not fail here.
"""

import argparse
import json
import sys
from dataclasses import dataclass

from github_actions_utils import str2bool


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    """Parsed result for a single upstream job."""

    name: str
    result: str
    continue_on_error: bool


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
    args = parser.parse_args(argv)

    jobs = parse_needs_json(args.needs_json)
    failed, ok = evaluate_results(jobs)

    print(f"Checking status for {len(jobs)} job(s):")
    for job in jobs:
        color = _RESULT_COLORS.get(job.result, _RED)
        print(f"  {color}{job.name}: {job.result}{_RESET}")

    if failed:
        print(f"\n{_RED}The following jobs failed:{_RESET}")
        for job in failed:
            print(f"  {_RED}{job.name}{_RESET}")
        print(f"\n{_RED}Check those jobs to see what failed{_RESET}")
        return 1

    print(f"\n{_GREEN}All required jobs succeeded.{_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
