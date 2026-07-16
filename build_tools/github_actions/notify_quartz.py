#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Dispatch CI event payloads to the Quartz repository for ingestion.

Reads the GitHub Actions webhook event from GITHUB_EVENT_PATH and dispatches
a structured payload to a Quartz ingest workflow via workflow_dispatch.

Payload ``event_type`` values and top-level keys align with
``quartz_ingest.ingest_dispatch`` in ROCm/Quartz:

- ``workflow_run_requested`` / ``workflow_run_completed``: ``repository``,
  ``workflow_run`` (GitHub run object plus optional ``jobs`` from the Actions API).
- ``pull_request_event``: ``action``, ``pull_request``.
- ``push_event``: push payload fields (``ref``, ``before``, ``after``, ``commits``, etc.).
- ``workflow_dispatch_test``: manual connectivity check (ingest skips DB writes).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


# ── GitHub API helpers ───────────────────────────────────────────────────────


def _api(
    token: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> Any:
    """Make an authenticated GitHub API request."""
    url = f"{GITHUB_API}{path}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Authorization", f"token {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    with urlopen(req) as resp:
        return json.loads(resp.read()) if resp.status != 204 else None


def _api_get(token: str, path: str) -> Any:
    return _api(token, "GET", path)


def _api_post(token: str, path: str, body: dict[str, Any]) -> Any:
    return _api(token, "POST", path, body=body)


def _workflow_job_dispatch_fields(job: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "conclusion": job.get("conclusion"),
    }
    for key in ("created_at", "started_at", "completed_at", "runner_name"):
        val = job.get(key)
        if val is not None:
            row[key] = val
    labels = job.get("labels")
    if labels:
        # Hosted runners: list[str]. Some payloads use objects; normalize to strings.
        normalized: list[str] = []
        for lb in labels:
            if isinstance(lb, str):
                normalized.append(lb)
            elif isinstance(lb, dict) and lb.get("name") is not None:
                normalized.append(str(lb["name"]))
        if normalized:
            row["labels"] = normalized
    return row


def _paginate(token: str, path: str, *, key: str, per_page: int = 100) -> list:
    """Fetch all pages from a paginated GitHub API endpoint."""
    results: list = []
    for page in range(1, 200):
        sep = "&" if "?" in path else "?"
        data = _api_get(token, f"{path}{sep}per_page={per_page}&page={page}")
        batch = data.get(key) or []
        results.extend(batch)
        if len(batch) < per_page:
            break
    return results


# ── Payload builders ─────────────────────────────────────────────────────────


def _build_workflow_run_payload(
    event_payload: dict[str, Any],
    repo: str,
    token: str,
) -> dict[str, Any]:
    wr = event_payload["workflow_run"]
    action = event_payload["action"]
    event_type = f"workflow_run_{action}"
    owner, repo_name = repo.split("/")

    run_inputs: dict = {}
    referenced_workflows: list = []
    check_suite_id: int | None = None
    try:
        run_data = _api_get(
            token, f"/repos/{repo}/actions/runs/{wr['id']}"
        )
        run_inputs = run_data.get("inputs") or {}
        referenced_workflows = [
            {"path": rw["path"], "sha": rw["sha"], "ref": rw.get("ref")}
            for rw in (run_data.get("referenced_workflows") or [])
        ]
        check_suite_id = run_data.get("check_suite_id")
    except Exception as exc:
        log.warning("Failed to fetch run details: %s", exc)

    jobs: list[dict[str, Any]] = []
    if action == "completed":
        try:
            raw_jobs = _paginate(
                token,
                f"/repos/{repo}/actions/runs/{wr['id']}/jobs",
                key="jobs",
            )
            jobs = [_workflow_job_dispatch_fields(j) for j in raw_jobs]
        except Exception as exc:
            log.warning("Failed to fetch jobs: %s", exc)

    parent_workflow: dict[str, Any] | None = None
    if check_suite_id:
        try:
            suite_data = _api_get(
                token,
                f"/repos/{repo}/actions/runs?check_suite_id={check_suite_id}",
            )
            for run in suite_data.get("workflow_runs") or []:
                if run["id"] != wr["id"]:
                    parent_workflow = {
                        "id": run["id"],
                        "name": run["name"],
                        "workflow_id": run["workflow_id"],
                        "path": run.get("path"),
                        "event": run.get("event"),
                        "html_url": run.get("html_url"),
                        "status": run.get("status"),
                        "conclusion": run.get("conclusion"),
                    }
                    break
        except Exception as exc:
            log.warning("Failed to detect parent workflow: %s", exc)

    actor = wr.get("actor") or {}
    triggering = wr.get("triggering_actor") or {}

    return {
        "event_type": event_type,
        "repository": repo,
        "workflow_run": {
            "id": wr["id"],
            "name": wr.get("name"),
            "display_title": wr.get("display_title"),
            "path": wr.get("path"),
            "workflow_id": wr.get("workflow_id"),
            "status": wr.get("status"),
            "conclusion": wr.get("conclusion"),
            "event": wr.get("event"),
            "head_branch": wr.get("head_branch"),
            "head_sha": wr.get("head_sha"),
            "run_number": wr.get("run_number"),
            "run_attempt": wr.get("run_attempt"),
            "html_url": wr.get("html_url"),
            "created_at": wr.get("created_at"),
            "updated_at": wr.get("updated_at"),
            "run_started_at": wr.get("run_started_at"),
            "actor": {"login": actor["login"], "id": actor["id"]}
            if actor.get("login")
            else None,
            "triggering_actor": {
                "login": triggering["login"],
                "id": triggering["id"],
            }
            if triggering.get("login")
            else None,
            "pull_requests": [
                {
                    "number": pr["number"],
                    "head": {"ref": pr["head"]["ref"], "sha": pr["head"]["sha"]},
                    "base": {"ref": pr["base"]["ref"], "sha": pr["base"]["sha"]},
                }
                for pr in (wr.get("pull_requests") or [])
            ],
            "inputs": run_inputs,
            "release_type": run_inputs.get("release_type"),
            "jobs": jobs,
            "parent_workflow": parent_workflow,
            "referenced_workflows": referenced_workflows,
        },
    }


def _build_pull_request_payload(
    event_payload: dict[str, Any], repo: str
) -> dict[str, Any]:
    pr = event_payload["pull_request"]
    user = pr.get("user") or {}
    return {
        "event_type": "pull_request_event",
        "repository": repo,
        "action": event_payload.get("action"),
        "pull_request": {
            "id": pr["id"],
            "number": pr["number"],
            "state": pr.get("state"),
            "title": pr.get("title"),
            "draft": pr.get("draft"),
            "merged": pr.get("merged"),
            "merge_commit_sha": pr.get("merge_commit_sha"),
            "head": {"ref": pr["head"]["ref"], "sha": pr["head"]["sha"]},
            "base": {"ref": pr["base"]["ref"], "sha": pr["base"]["sha"]},
            "user": {"login": user["login"]} if user.get("login") else None,
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "changed_files": pr.get("changed_files"),
            "commits": pr.get("commits"),
        },
    }


def _build_push_payload(
    event_payload: dict[str, Any], repo: str
) -> dict[str, Any]:
    pusher = event_payload.get("pusher") or {}
    head_commit = event_payload.get("head_commit")
    return {
        "event_type": "push_event",
        "repository": repo,
        "ref": event_payload.get("ref"),
        "before": event_payload.get("before"),
        "after": event_payload.get("after"),
        "forced": event_payload.get("forced", False),
        "pusher": pusher,
        "head_commit": {"timestamp": head_commit["timestamp"]}
        if head_commit
        else None,
        "commits": [
            {"id": c["id"], "timestamp": c.get("timestamp")}
            for c in (event_payload.get("commits") or [])
        ],
    }


def _build_workflow_dispatch_payload(repo: str) -> dict[str, Any]:
    return {
        "event_type": "workflow_dispatch_test",
        "repository": repo,
        "triggered_by": os.environ.get("GITHUB_ACTOR", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Dispatch ─────────────────────────────────────────────────────────────────


def dispatch_to_quartz(
    token: str,
    quartz_repo: str,
    workflow_ref: str,
    payload: dict[str, Any],
) -> None:
    """Trigger the ingest workflow in Quartz via workflow_dispatch."""
    owner, repo = quartz_repo.split("/")
    workflow_file = "ingest.yml"

    _api_post(
        token,
        f"/repos/{quartz_repo}/actions/workflows/{workflow_file}/dispatches",
        body={
            "ref": workflow_ref,
            "inputs": {
                "payload_json": json.dumps(payload),
                "fetch_jobs": "true",
            },
        },
    )
    log.info("Dispatched workflow_dispatch to %s (ref: %s)", quartz_repo, workflow_ref)


# ── Main ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quartz-repo",
        default=os.environ.get("QUARTZ_REPO", "ROCm/Quartz"),
        help="Target Quartz repository (owner/repo)",
    )
    p.add_argument(
        "--quartz-workflow-ref",
        default=os.environ.get("QUARTZ_WORKFLOW_REF", "main"),
        help="Branch in Quartz repo where the ingest workflow lives",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("QUARTZ_DISPATCH_TOKEN", ""),
        help="GitHub token for dispatching (defaults to QUARTZ_DISPATCH_TOKEN env var)",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    args = build_parser().parse_args()

    token = args.token
    if not token:
        log.error("No GitHub token provided (--token or QUARTZ_DISPATCH_TOKEN)")
        return 1

    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not event_name or not event_path or not repo:
        log.error(
            "Missing required GitHub Actions environment variables "
            "(GITHUB_EVENT_NAME, GITHUB_EVENT_PATH, GITHUB_REPOSITORY)"
        )
        return 1

    event_payload = json.loads(Path(event_path).read_text(encoding="utf-8"))

    payload: dict[str, Any]
    if event_name == "workflow_run":
        payload = _build_workflow_run_payload(event_payload, repo, token)
    elif event_name == "pull_request":
        payload = _build_pull_request_payload(event_payload, repo)
    elif event_name == "push":
        payload = _build_push_payload(event_payload, repo)
    elif event_name == "workflow_dispatch":
        payload = _build_workflow_dispatch_payload(repo)
    else:
        log.info("Unhandled event: %s, skipping dispatch.", event_name)
        return 0

    try:
        dispatch_to_quartz(token, args.quartz_repo, args.quartz_workflow_ref, payload)
    except HTTPError as exc:
        log.error("Dispatch failed: %s %s", exc.code, exc.read().decode())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
