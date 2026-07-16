#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence
import json
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class TimingRecord:
    workflow_run_id: str
    run_attempt: str
    job_name: str
    runner_label: str
    runner_pool: str | None
    runner_instance: str | None
    platform: str | None
    workflow_phase: str | None
    component: str | None
    job_type: str | None
    queued_at: str | None
    started_at: str | None
    completed_at: str | None
    decision: str
    stage_or_test_family: str | None = None
    queue_seconds: float | None = None
    run_seconds: float | None = None
    total_seconds: float | None = None


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    # GitHub returns timestamps like "2026-07-09T18:12:34Z"
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _http_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "therock-ci",
        },
    )
    with urllib.request.urlopen(request) as response:  # nosec: B310
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _workflow_run_url(repository: str, run_id: str) -> str:
    owner_repo = repository.strip()
    return f"https://api.github.com/repos/{owner_repo}/actions/runs/{run_id}"


def _workflow_jobs_url(repository: str, run_id: str, page: int = 1) -> str:
    owner_repo = repository.strip()
    query = urllib.parse.urlencode({"per_page": 100, "page": page})
    return (
        f"https://api.github.com/repos/{owner_repo}/actions/runs/{run_id}/jobs?{query}"
    )


def _split_runner_label(runner_label: str) -> tuple[str | None, str | None]:
    """
    Split a runner label into a stable pool name and a temporary instance name.

    Examples:
      - "GitHub Actions 1005172268" -> ("GitHub Actions", "1005172268")
      - "aws-linux-scale-rocm-prod-qv2xv-runner-tx9kw" -> ("aws-linux-scale-rocm-prod", "qv2xv")
    """
    label = runner_label.strip()
    if not label:
        return None, None

    if "-runner-" in label:
        prefix, _suffix = label.rsplit("-runner-", 1)
        if "-" in prefix:
            pool, instance = prefix.rsplit("-", 1)
            return pool or None, instance or None
        return prefix or None, None

    if " " in label:
        pool, maybe_instance = label.rsplit(" ", 1)
        if maybe_instance.isdigit():
            return pool or None, maybe_instance

    return label, None


def _normalize_status(raw: str | None) -> str:
    if not raw:
        return "unknown"
    return raw.strip().lower()


def _status_symbol(status: str) -> str:
    status = _normalize_status(status)
    if status in {"success", "completed"}:
        return "✅"
    if status in {"failure", "cancelled", "timed_out", "action_required"}:
        return "❌"
    if status in {
        "skipped",
        "neutral",
        "queued",
        "in_progress",
        "waiting",
        "requested",
    }:
        return "⏭️" if status == "skipped" else "⏳"
    return "—"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"

    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _infer_platform(job_name: str, runner_label: str) -> str:
    text = f"{job_name} {runner_label}".lower()
    if "windows" in text or runner_label.lower().startswith("azure-"):
        return "Windows"
    if "linux" in text or runner_label.lower().startswith(("aws-", "github actions")):
        return "Linux"
    return "—"


def _infer_workflow_phase(job_name: str, status: str) -> str:
    name = job_name.lower()
    if "manifest" in name:
        return "Manifest Diff"
    if "setup" in name:
        return "Setup"
    if "build" in name or "stage" in name:
        return "Build Stages"
    if "test" in name:
        return "Tests"
    if status in {"queued", "in_progress"}:
        return "Running"
    return "Other"


def _infer_job_type(job_name: str, status: str) -> str:
    name = job_name.lower()
    if "stage" in name or "build" in name:
        return "Stage"
    if "test" in name:
        return "Test"
    if "setup" in name:
        return "Job"
    if status in {"queued", "in_progress"}:
        return "Job"
    return "Job"


def _infer_component(job_name: str) -> str:
    """
    Best-effort component name from the job title.

    For simple names like 'setup / setup' or 'Manifest Diff / Generate ...',
    this returns the first meaningful chunk.
    """
    if " / " in job_name:
        left, right = job_name.split(" / ", 1)
        # Prefer the more specific right-hand side when it is not just a repeat.
        if right and right.lower() != left.lower():
            return right.strip()
        return left.strip()

    if "::" in job_name:
        return job_name.split("::", 1)[0].strip()

    return job_name.strip()


def collect_timing_records(
    *,
    repository: str,
    run_id: str,
    run_attempt: str,
    token: str,
    decision_lookup: Mapping[str, str] | None = None,
    family_lookup: Mapping[str, str] | None = None,
) -> list[TimingRecord]:
    """
    Fetch workflow/job timing from the GitHub API.

    queued_at is approximated from the workflow run's created_at timestamp.
    """
    decision_lookup = decision_lookup or {}
    family_lookup = family_lookup or {}

    run_payload = _http_json(_workflow_run_url(repository, run_id), token)
    workflow_queued_at = run_payload.get("created_at")

    queued_dt = _parse_iso8601(workflow_queued_at)
    records: list[TimingRecord] = []

    page = 1
    while True:
        jobs_payload = _http_json(
            _workflow_jobs_url(repository, run_id, page=page), token
        )
        jobs = jobs_payload.get("jobs", [])
        if not jobs:
            break

        for job in jobs:
            job_name = str(job.get("name") or job.get("id") or "unknown-job")
            started_at = job.get("started_at")
            completed_at = job.get("completed_at")
            labels = job.get("labels") or []
            runner_label = job.get("runner_name") or (
                labels[0] if labels else "unknown"
            )
            runner_label = str(runner_label)
            runner_pool, runner_instance = _split_runner_label(runner_label)

            status = str(job.get("conclusion") or job.get("status") or "unknown")
            platform = _infer_platform(job_name, runner_label)
            workflow_phase = _infer_workflow_phase(job_name, status)
            component = _infer_component(job_name)
            job_type = _infer_job_type(job_name, status)

            started_dt = _parse_iso8601(started_at)
            completed_dt = _parse_iso8601(completed_at)

            queue_seconds = None
            if queued_dt is not None and started_dt is not None:
                queue_seconds = (started_dt - queued_dt).total_seconds()

            run_seconds = None
            if started_dt is not None and completed_dt is not None:
                run_seconds = (completed_dt - started_dt).total_seconds()

            records.append(
                TimingRecord(
                    workflow_run_id=str(run_id),
                    run_attempt=str(run_attempt),
                    job_name=job_name,
                    runner_label=runner_label,
                    runner_pool=runner_pool,
                    runner_instance=runner_instance,
                    platform=platform,
                    workflow_phase=workflow_phase,
                    component=component,
                    job_type=job_type,
                    queued_at=workflow_queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    decision=decision_lookup.get(job_name, status),
                    stage_or_test_family=family_lookup.get(job_name),
                    queue_seconds=queue_seconds,
                    run_seconds=run_seconds,
                    total_seconds=(
                        (queue_seconds + run_seconds)
                        if queue_seconds is not None and run_seconds is not None
                        else None
                    ),
                )
            )

        total_count = int(jobs_payload.get("total_count") or 0)
        if len(jobs) >= total_count:
            break
        if len(jobs) < 100:
            break
        page += 1

    return records


def format_timing_summary(records: Sequence[TimingRecord]) -> str:
    if not records:
        return "### CI Job Timing Summary\n\n_no timing data_"

    def sort_key(record: TimingRecord) -> tuple[int, float, str]:
        status = _normalize_status(record.decision)
        if status in {"failure", "cancelled", "timed_out"}:
            rank = 0
        elif status in {"queued", "in_progress", "waiting", "requested"}:
            rank = 1
        else:
            rank = 2

        total = record.total_seconds
        total_sort = -(total if total is not None else -1.0)
        return (rank, total_sort, record.job_name.lower())

    sorted_records = sorted(records, key=sort_key)

    lines = ["### CI Job Timing Summary", ""]
    lines.append(
        "| Platform | Workflow Phase | Component | Variant | Job Type | Runner Pool | Runner Instance | Status | Queue | Run | Total |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"
    )

    for record in sorted_records:
        status = _normalize_status(record.decision)
        status_label = _status_symbol(status)
        queue = _format_duration(record.queue_seconds)
        run = _format_duration(record.run_seconds)
        total = _format_duration(record.total_seconds)
        runner_pool = record.runner_pool or "—"
        runner_instance = record.runner_instance or "—"
        platform = record.platform or "—"
        phase = record.workflow_phase or "—"
        component = record.component or "—"
        variant = record.stage_or_test_family or "—"
        job_type = record.job_type or "—"

        lines.append(
            "| "
            f"{platform} | "
            f"{phase} | "
            f"{component} | "
            f"{variant} | "
            f"{job_type} | "
            f"{runner_pool} | "
            f"{runner_instance} | "
            f"{status_label} {record.decision} | "
            f"{queue} | "
            f"{run} | "
            f"{total} |"
        )

    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Exact timestamps</summary>")
    lines.append("")
    lines.append("| Job | Started At | Completed At |")
    lines.append("| --- | --- | --- |")
    for record in sorted_records:
        lines.append(
            f"| {record.job_name} | {record.started_at or '—'} | {record.completed_at or '—'} |"
        )
    lines.append("</details>")
    return "\n".join(lines)
