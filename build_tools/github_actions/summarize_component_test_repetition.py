# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare component/shard repetition jobs with the expected manifest.

Only the requested workflow attempt is read. Job and Test-step outcomes are
reported separately: setup failures, advisory failures, and missing executions
must not be counted as passing tests. Internal ProLense attempts are not exposed
by the jobs API, so a successful Test step is not proof of an initially clean pass.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re

from github_actions_api import gha_send_request


def fetch_attempt_jobs(repository: str, run_id: str, run_attempt: int) -> list[dict]:
    """Read every page rather than imposing a cap on large repetition runs."""
    jobs = []
    page = 1
    while True:
        response = gha_send_request(
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
            f"/attempts/{run_attempt}/jobs?per_page=100&page={page}"
        )
        batch = response["jobs"]
        jobs.extend(batch)
        if len(batch) < 100:
            return jobs
        page += 1


def classify_job(job: dict) -> str:
    test = next((step for step in job.get("steps", []) if step["name"] == "Test"), None)
    if not test or test.get("conclusion") == "skipped":
        if job.get("conclusion") in ("failure", "timed_out"):
            return "infrastructure_failure"
        return "test_not_run"
    if test.get("conclusion") in ("failure", "timed_out"):
        return "test_failure"
    if test.get("conclusion") != "success":
        return "test_incomplete"
    if job.get("conclusion") != "success":
        return "post_test_failure"
    return "passed"


def summarize_jobs(manifest: dict, jobs: list[dict]) -> dict:
    expected = manifest["expected"]
    observed = defaultdict(list)
    unexpected = []
    for job in jobs:
        # Disabled caller jobs retain their unevaluated matrix expressions.
        # Ignore only skipped, empty placeholders for targets absent from the
        # manifest; skipped requested executions must still be accounted for.
        placeholder = re.fullmatch(
            r"(repeat-(?:linux|windows)-(?:gfx120x|gfx110x|gfx1151))"
            r"-\$\{\{ matrix\.iteration \}\}-\$\{\{ matrix\.component\.job_name \}\}",
            job["name"],
        )
        if (
            placeholder
            and job.get("conclusion") == "skipped"
            and job.get("steps") == []
            and not any(
                execution["sample_id"].startswith(placeholder[1] + "-")
                for execution in expected
            )
        ):
            continue
        # Reusable-workflow jobs are named "<caller name> / <child job name>".
        # The caller name contains the target, iteration, and component identity.
        match = re.search(
            r"(?:^| / )(repeat-(?:linux|windows)-(?:gfx120x|gfx110x|gfx1151)"
            r"-\d+-[A-Za-z0-9_-]+) / Test .*?\(shard (\d+)/(\d+)\)",
            job["name"],
        )
        if match:
            observed[(match[1], int(match[2]))].append((job, int(match[3])))
        elif "repeat-" in job["name"]:
            unexpected.append(job)
    results = []
    for execution in expected:
        matches = observed.pop((execution["sample_id"], execution["shard"]), [])
        result = dict(execution)
        result["jobs"] = [
            {
                "id": job["id"],
                "url": job.get("html_url", ""),
                "name": job["name"],
                "conclusion": job.get("conclusion"),
                "runner_name": job.get("runner_name", ""),
                "runner_id": job.get("runner_id"),
                "labels": job.get("labels", []),
                "steps": job.get("steps", []),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
            }
            for job, _ in matches
        ]
        if not matches:
            result["outcome"] = "missing"
        elif len(matches) > 1:
            result["outcome"] = "duplicate"
        elif matches[0][1] != execution["total_shards"]:
            result["outcome"] = "shard_mismatch"
        elif execution["runner_label"] not in matches[0][0].get("labels", []):
            result["outcome"] = "runner_mismatch"
        else:
            result["outcome"] = classify_job(matches[0][0])
        results.append(result)
    unexpected.extend(job for matches in observed.values() for job, _ in matches)
    incomplete = any(
        result["outcome"]
        in (
            "missing",
            "duplicate",
            "shard_mismatch",
            "runner_mismatch",
            "test_not_run",
            "test_incomplete",
        )
        for result in results
    )
    return {
        "schema_version": 1,
        "baseline": manifest.get("baseline"),
        "therock_sha": manifest.get("therock_sha"),
        "ci_config_sha": manifest.get("ci_config_sha"),
        "test_type": manifest.get("test_type"),
        "omitted_platforms": manifest.get("omitted_platforms", []),
        "repetitions": manifest.get("repetitions"),
        "counts": dict(Counter(result["outcome"] for result in results)),
        "results": results,
        "unexpected_jobs": unexpected,
        "incomplete": incomplete or bool(unexpected),
        "errors": [],
    }


def render_report(report: dict, *, include_details: bool = True) -> str:
    lines = ["# Component test repetition", ""]
    lines.append(
        "Report incomplete."
        if report["incomplete"]
        else "All expected shard jobs accounted for."
    )
    lines.extend(["", f"Workflow attempt: {report['run_attempt']}", ""])
    if report.get("baseline"):
        baseline = report["baseline"]
        lines.extend(
            [
                f"Baseline: [{baseline['run_id']}]({baseline['url']})",
                f"TheRock SHA: `{report['therock_sha']}`",
                f"CI-config SHA: `{report['ci_config_sha']}`",
                "",
            ]
        )
    if report.get("omitted_platforms"):
        lines.extend(
            [
                "Platforms omitted because the selected tests have no configured tests there: "
                + ", ".join(report["omitted_platforms"])
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            f"Tier: {report.get('test_type') or 'unavailable'}. Each sample is one component/shard job execution.",
            "The existing reusable workflow retains its internal retries and expected-failure handling. "
            "Passed counts reflect the final Test-step outcome; individual tests and initial retry attempts are not measured.",
            "",
            "## Outcomes",
            "",
            "| Platform | Family | Component | Expected shards | Passed | Test failures | Infrastructure failures | Other outcomes |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    groups = defaultdict(Counter)
    for result in report["results"]:
        groups[(result["platform"], result["family"], result["component"])][
            result["outcome"]
        ] += 1
    for (platform, family, component), counts in sorted(groups.items()):
        other = (
            ", ".join(
                f"{key}: {count}"
                for key, count in sorted(counts.items())
                if key not in ("passed", "test_failure", "infrastructure_failure")
            )
            or "—"
        )
        lines.append(
            f"| {platform} | {family} | {component} | {sum(counts.values())} | "
            f"{counts['passed']} | {counts['test_failure']} | {counts['infrastructure_failure']} | {other} |"
        )
    failures = [result for result in report["results"] if result["outcome"] != "passed"]
    if failures and include_details:
        lines.extend(["", "## Execution details", ""])
        for result in failures:
            links = (
                ", ".join(f"[job {job['id']}]({job['url']})" for job in result["jobs"])
                or "no matching job"
            )
            advisory = " (expected failure)" if result.get("expect_failure") else ""
            lines.append(
                f"- `{result['sample_id']}`, shard {result['shard']}: "
                f"{result['outcome']}{advisory}; {links}."
            )
    if report["errors"]:
        lines.extend(["", "## Collection errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["unexpected_jobs"]:
        lines.extend(
            [
                "",
                f"Unexpected repetition jobs: {len(report['unexpected_jobs'])}. See report.json.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("component-repetition/manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("component-repetition"))
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID"),
        required=not os.environ.get("GITHUB_RUN_ID"),
    )
    parser.add_argument(
        "--run-attempt",
        type=int,
        default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
    )
    options = parser.parse_args()
    errors = []
    manifest = {"expected": []}
    try:
        candidate = json.loads(options.manifest.read_text())
        if not isinstance(candidate, dict) or not isinstance(
            candidate.get("expected"), list
        ):
            raise ValueError("Invalid expected execution manifest")
        if not candidate["expected"]:
            raise ValueError("Expected execution manifest is empty")
        required_fields = {
            "sample_id",
            "shard",
            "total_shards",
            "runner_label",
            "platform",
            "family",
            "component",
        }
        if any(
            not isinstance(row, dict) or not required_fields.issubset(row)
            for row in candidate["expected"]
        ):
            raise ValueError("Expected execution manifest contains invalid rows")
        manifest = candidate
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"Cannot read expected execution manifest: {error}")
    if os.environ.get("CONFIGURATION_RESULT", "success") != "success":
        errors.append(f"Configuration job result: {os.environ['CONFIGURATION_RESULT']}")
    jobs = []
    try:
        jobs = fetch_attempt_jobs(
            options.repository, options.run_id, options.run_attempt
        )
    except Exception as error:
        errors.append(f"Cannot collect workflow jobs: {error}")
    report = summarize_jobs(manifest, jobs)
    report.update(
        repository=options.repository,
        run_id=options.run_id,
        run_attempt=options.run_attempt,
    )
    report["errors"] = errors
    report["incomplete"] |= bool(errors)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_report(report)
    (options.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (options.output_dir / "report.md").write_text(markdown)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        # GitHub limits each step summary to 1 MiB. Preserve the complete
        # execution details in the artifacts even for a large failure campaign.
        step_markdown = markdown
        if len(step_markdown.encode("utf-8")) > 900_000:
            step_markdown = render_report(report, include_details=False)
            step_markdown += "\nFull execution details are in the component-repetition-report artifact.\n"
        with open(summary, "a") as stream:
            stream.write(step_markdown)
    print(markdown)
    return int(report["incomplete"])


if __name__ == "__main__":
    raise SystemExit(main())
