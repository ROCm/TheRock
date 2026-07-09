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
    queued_at: str | None
    started_at: str | None
    completed_at: str | None
    decision: str
    stage_or_test_family: str | None = None
    queue_seconds: float | None = None
    run_seconds: float | None = None


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
                    queued_at=workflow_queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    decision=decision_lookup.get(
                        job_name,
                        str(job.get("conclusion") or job.get("status") or "unknown"),
                    ),
                    stage_or_test_family=family_lookup.get(job_name),
                    queue_seconds=queue_seconds,
                    run_seconds=run_seconds,
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
        return "### Queue-time / execution-time\n\n_no timing data_"

    def fmt_seconds(value: float | None) -> str:
        if value is None:
            return "_n/a_"
        return f"{value:.1f}s"

    lines = ["### Queue-time / execution-time", ""]
    lines.append("| job | runner | decision | queue | run | started | completed |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for record in records:
        lines.append(
            "| "
            f"{record.job_name} | "
            f"{record.runner_label} | "
            f"{record.decision} | "
            f"{fmt_seconds(record.queue_seconds)} | "
            f"{fmt_seconds(record.run_seconds)} | "
            f"{record.started_at or '_n/a_'} | "
            f"{record.completed_at or '_n/a_'} |"
        )
    return "\n".join(lines)
