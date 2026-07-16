#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Dispatch CI run status to the Quartz repository for ingestion.

This is a self-report bridge: a calling workflow invokes this script
from its own jobs to push status to Quartz at start
(`--run-phase=started`) and end (`--run-phase=completed`). The payload
is built from the currently running workflow — resolved via
`GITHUB_RUN_ID` + the GitHub Actions API — and dispatched to a Quartz
ingest workflow via `workflow_dispatch`.

Required runtime context
------------------------
* Standard GitHub Actions env: `GITHUB_RUN_ID`, `GITHUB_REPOSITORY`.

* A token from the **Quartz Hauly** GitHub App installation, passed via
  `--token` or `QUARTZ_HAULY_TOKEN`. Personal access tokens are NOT
  expected; route through Hauly so install scopes are managed
  centrally. The token must grant `actions:read` on the source repo
  (to inspect the run / jobs / parent run) and `actions:write` on the
  target Quartz repo (to dispatch the ingest workflow).

* `--embedded-inputs` / `WORKFLOW_INPUTS` — JSON of the caller's
  `toJSON(inputs)` context, because the Actions API does not expose it.

* `--captured-outputs` / `WORKFLOW_CAPTURED_OUTPUTS` — JSON object of
  caller-supplied job context. Canonical pattern is `${{ toJSON(needs) }}`
  from a `notify_completed` job whose `needs:` lists every job the
  caller wants recorded; the receiver then has per-job `result` +
  `outputs` without any plucking, and `--run-conclusion` is derived
  automatically. Values are stored verbatim; authors are responsible
  for not including secrets.

* `--run-conclusion` / `RUN_CONCLUSION` — final conclusion for the run
  (success / failure / cancelled). Optional when
  `--captured-outputs` is a `toJSON(needs)` blob (the script derives
  from the per-job `result` values). Required when `--run-phase=completed`
  and no derivable captured-outputs was provided; we refuse to silently
  assume success.

Receiving side
--------------
The Quartz repo (default `ROCm/Quartz`; override with
`--quartz-repo`) must expose a workflow file (default
`receive_therock_data.yml`; override with `--quartz-workflow-file`) on
the chosen ref (default `main`; override with `--quartz-workflow-ref`).
That workflow must accept two `workflow_dispatch` string inputs:

* `payload_json` — JSON-encoded payload described below.
* `fetch_jobs`   — `"true"` / `"false"`. Set to `"true"` when this
  script had to strip the `jobs` array to stay under GitHub's
  65,535-byte cap on serialized dispatch `inputs`; the receiver is
  then responsible for re-fetching jobs from the Actions API.

Payload contract
----------------
Top-level keys align with
`scripts/receive_therock/therock_parse_input.py` in ROCm/Quartz.

`event_type` is the dispatch envelope's lifecycle marker, used by the
receiver to route to the correct ingest path and to record whether the
run is still in flight. It is one of:

* `workflow_run_in_progress`  — emitted with `run_phase=started`.
  Carries `repository` and a `workflow_run` object with live status
  (typically `in_progress`) and no `jobs` yet.
* `workflow_run_completed`  — emitted with `run_phase=completed`. Carries
  `repository` and a `workflow_run` object including the final
  `conclusion` and the `jobs` array fetched from the Actions API.

`event_type` is intentionally distinct from two unrelated fields that
also live on the payload and are easy to confuse with it:

* `workflow_run.event`  — the GitHub Actions trigger that started the
  run (`push`, `pull_request`, `schedule`, `workflow_dispatch`, ...).
* `workflow_run.status` / `workflow_run.conclusion`  — GitHub's own
  in-flight / final state on the run itself.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# workflow_dispatch caps the *serialized* `inputs` object at 65,535 bytes. We
# measure the encoded inputs dict (after JSON re-escaping) against this; a
# raw `payload_json` of ~50 k can blow the cap once re-escaped.
GITHUB_INPUTS_MAX_CHARS = 65_535
GITHUB_API_TIMEOUT_SECONDS = 30
# RFC 5988 Link header used by GitHub for paginated responses, e.g.
#   <https://api.github.com/...?page=2>; rel="next", <...>; rel="last"
# We follow rel="next" until it disappears; see `_iter_paginated`.
_NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


class _DispatchConfigError(RuntimeError):
    """Raised when the script cannot proceed due to missing config or env vars."""


@dataclass
class _GithubApiResponse:
    """Decoded body and response headers from a GitHub REST API call."""

    body: Any
    headers: dict[str, str]


def _github_api_request(
    token: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> _GithubApiResponse:
    """Make an authenticated GitHub REST API request.

    `path` may be a leading-slash path like `/repos/{owner}/{repo}/...`
    (joined to the GitHub API base URL) or a full `https://...` URL — the
    latter is what `_iter_paginated` follows from `Link: rel="next"`.
    """
    url = path if path.startswith("http") else f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Authorization", f"token {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=GITHUB_API_TIMEOUT_SECONDS) as resp:
            # 204 No Content: GitHub returns this for endpoints that succeed
            # but have nothing to return (e.g. `workflow_dispatch`). Body is
            # empty, so we surface `None` to callers and skip JSON parsing.
            decoded = None if resp.status == 204 else json.loads(resp.read())
            return _GithubApiResponse(body=decoded, headers=dict(resp.headers))
    except HTTPError:
        raise
    except URLError as exc:
        log.error("GitHub API %s %s failed: %s", method, url, exc)
        raise


def _workflow_job_dispatch_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Project a GitHub Actions Job object down to the fields we ship to Quartz.

    `conclusion` is kept even when null because downstream consumers
    distinguish "not yet concluded" from "no conclusion". The timing /
    runner fields, by contrast, are simply absent for jobs that haven't
    progressed that far — we omit them when null to keep the payload
    small rather than emit a row of nullable noise.
    """
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
    # Hosted runners: list[str]. Some payloads use list[dict] with a `name`
    # field; normalize both shapes to a flat list of strings.
    labels = [
        lb if isinstance(lb, str) else str(lb["name"])
        for lb in (job.get("labels") or [])
        if isinstance(lb, str) or (isinstance(lb, dict) and lb.get("name") is not None)
    ]
    if labels:
        row["labels"] = labels
    return row


def _next_link_url(link_header: str | None) -> str | None:
    """Return the rel="next" URL from a `Link` header, or `None` if absent."""
    if not link_header:
        return None
    match = _NEXT_LINK_RE.search(link_header)
    return match.group(1) if match else None


def _iter_paginated(
    token: str,
    path: str,
    *,
    key: str,
    per_page: int = 100,
) -> Iterator[dict[str, Any]]:
    """Yield items from every page of a paginated GitHub API endpoint.

    Follows `Link: rel="next"` until exhausted, so this works regardless of
    GitHub's per-page or total-result caps and needs no page-counter bound.
    """
    sep = "&" if "?" in path else "?"
    url: str | None = f"{path}{sep}per_page={per_page}"
    while url:
        resp = _github_api_request(token, "GET", url)
        yield from (resp.body.get(key) or [])
        url = _next_link_url(resp.headers.get("Link"))


def _get_list_referenced_workflows(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the `referenced_workflows` list from a run or run-details dict."""
    return [
        {"path": rw["path"], "sha": rw["sha"], "ref": rw.get("ref")}
        for rw in (source.get("referenced_workflows") or [])
    ]


def _fetch_jobs(
    *,
    token: str,
    repo: str,
    run_id: int | str,
) -> list[dict[str, Any]]:
    """Fetch and normalize the jobs list for a run. Returns [] and logs on API errors."""
    try:
        return [
            _workflow_job_dispatch_fields(j)
            for j in _iter_paginated(
                token,
                f"/repos/{repo}/actions/runs/{run_id}/jobs",
                key="jobs",
            )
        ]
    except (URLError, json.JSONDecodeError, KeyError) as exc:
        log.warning(
            "Failed to fetch jobs (%s): %s",
            type(exc).__name__,
            exc,
        )
        return []


def _fetch_parent_workflow(
    *,
    token: str,
    repo: str,
    check_suite_id: int,
    self_run_id: int,
) -> dict[str, Any] | None:
    """Find the parent (caller) workflow run sharing this check suite, if any."""
    try:
        suite_data = _github_api_request(
            token,
            "GET",
            f"/repos/{repo}/actions/runs?check_suite_id={check_suite_id}",
        ).body
        for run in suite_data.get("workflow_runs") or []:
            if run["id"] != self_run_id:
                return {
                    "id": run["id"],
                    "name": run["name"],
                    "workflow_id": run["workflow_id"],
                    "path": run.get("path"),
                    "event": run.get("event"),
                    "html_url": run.get("html_url"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                }
    except (URLError, json.JSONDecodeError, KeyError) as exc:
        log.warning(
            "Failed to detect parent workflow (%s): %s",
            type(exc).__name__,
            exc,
        )
    return None


def _normalize_actor(actor: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return `{login, id}` for a GitHub actor object, or `None` if missing.

    Guards against the API returning `null`, `{}`, or an object missing
    `login` for ghost/deleted users.
    """
    if not actor or not actor.get("login"):
        return None
    return {"login": actor["login"], "id": actor["id"]}


def _build_workflow_run_dict(
    wr: dict[str, Any],
    *,
    status: str | None,
    conclusion: str | None,
    inputs: dict[str, Any],
    captured_outputs: dict[str, Any],
    jobs: list[dict[str, Any]],
    parent_workflow: dict[str, Any] | None,
    referenced_workflows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the `workflow_run` sub-dict for the dispatch payload."""
    return {
        "id": wr["id"],
        "name": wr.get("name"),
        "display_title": wr.get("display_title"),
        "path": wr.get("path"),
        "workflow_id": wr.get("workflow_id"),
        "status": status,
        "conclusion": conclusion,
        "event": wr.get("event"),
        "head_branch": wr.get("head_branch"),
        "head_sha": wr.get("head_sha"),
        "run_number": wr.get("run_number"),
        "run_attempt": wr.get("run_attempt"),
        "html_url": wr.get("html_url"),
        "created_at": wr.get("created_at"),
        "updated_at": wr.get("updated_at"),
        "run_started_at": wr.get("run_started_at"),
        "actor": _normalize_actor(wr.get("actor")),
        "triggering_actor": _normalize_actor(wr.get("triggering_actor")),
        "pull_requests": [
            {
                "number": pr["number"],
                "head": {"ref": pr["head"]["ref"], "sha": pr["head"]["sha"]},
                "base": {"ref": pr["base"]["ref"], "sha": pr["base"]["sha"]},
            }
            for pr in (wr.get("pull_requests") or [])
        ],
        "inputs": inputs,
        # `release_type` is kept inside `inputs` and ALSO promoted to the top
        # level: the receiver pivots on it (nightly vs ad-hoc vs release) and
        # storing it twice avoids forcing every consumer to dig into `inputs`.
        # Null when the calling workflow has no `release_type` input.
        "release_type": inputs.get("release_type"),
        # Empty dict collapses to null on the wire so the receiver can
        # treat "no captured outputs" with a single truthiness check.
        "captured_outputs": captured_outputs or None,
        "jobs": jobs,
        "parent_workflow": parent_workflow,
        "referenced_workflows": referenced_workflows,
    }


def _normalize_reporting_path(reporting_workflow: str) -> str:
    """Normalize a caller-supplied workflow identifier to a `.github/workflows`
    path whose basename is the workflow file.

    Accepts a bare filename (`multi_arch_build_portable_linux.yml`), a workflow
    path, or a full `github.workflow_ref`-style value
    (`owner/repo/.github/workflows/x.yml@refs/heads/main`); the `@ref` suffix
    and any leading directories are stripped. Returns `""` for an empty input,
    signalling the caller to keep the run's API-reported path.
    """
    value = reporting_workflow.strip()
    if not value:
        return ""
    value = value.split("@", 1)[0]  # drop any @ref suffix
    name = PurePosixPath(value).name
    return f".github/workflows/{name}" if name else ""


def _build_payload(
    *,
    token: str,
    repo: str,
    embedded_inputs: dict[str, Any],
    captured_outputs: dict[str, Any],
    run_conclusion: str,
    run_phase: str = "completed",
    reporting_workflow: str = "",
) -> dict[str, Any]:
    """Build a payload for the current run using GITHUB_RUN_ID.

    Workflow inputs and the calling workflow's hand-picked job outputs
    (see `--embedded-inputs` / `--captured-outputs`) are threaded in by
    the caller, since the Actions API does not expose it.

    `reporting_workflow` (when set) overrides the `workflow_run.path` the
    receiver classifies on. Reusable workflows (`uses:`) all share the
    top-level run's `GITHUB_RUN_ID`, so the Actions API returns the entry
    workflow's path for every nested call; the caller passes its own file's
    basename so each leaf is classified by its true workflow rather than
    collapsing onto the orchestrator. `run_id` deliberately stays the shared
    run's id (it links the leaf back to its parent run).

    When *run_phase* is `"started"`, the payload uses
    `workflow_run_in_progress` as the `event_type` with the live run
    status from the API (typically `in_progress`). Jobs are skipped
    because they have not completed yet.
    """
    is_started = run_phase == "started"

    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not run_id:
        raise _DispatchConfigError(
            "GITHUB_RUN_ID not set — must run inside GitHub Actions"
        )

    try:
        wr = _github_api_request(
            token, "GET", f"/repos/{repo}/actions/runs/{run_id}"
        ).body
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise _DispatchConfigError(
            f"Failed to fetch workflow run {run_id} from GitHub API: {exc}"
        ) from exc

    referenced_workflows = _get_list_referenced_workflows(wr)
    jobs = [] if is_started else _fetch_jobs(token=token, repo=repo, run_id=run_id)
    check_suite_id = wr.get("check_suite_id")
    parent_workflow = (
        _fetch_parent_workflow(
            token=token,
            repo=repo,
            check_suite_id=check_suite_id,
            self_run_id=wr["id"],
        )
        if check_suite_id
        else None
    )

    if is_started:
        event_type = "workflow_run_in_progress"
        # The Actions API briefly returns null for `status` on a newly-queued
        # run before the runner picks it up. We're announcing the run at the
        # very start of its first job, so any null here means "in_progress" —
        # the alternative would be losing the start signal to a race.
        status = wr.get("status") or "in_progress"
        conclusion = None
    else:
        event_type = "workflow_run_completed"
        # We deliberately do NOT trust `wr.get("status")` here: this script
        # runs from inside the workflow it's reporting on, so the API still
        # shows the run as `in_progress` even though we know we're at the end.
        status = "completed"
        conclusion = run_conclusion

    workflow_run = _build_workflow_run_dict(
        wr,
        status=status,
        conclusion=conclusion,
        inputs=embedded_inputs,
        captured_outputs=captured_outputs,
        jobs=jobs,
        parent_workflow=parent_workflow,
        referenced_workflows=referenced_workflows,
    )

    reporting_path = _normalize_reporting_path(reporting_workflow)
    if reporting_path and reporting_path != workflow_run.get("path"):
        log.info(
            "Overriding reported workflow path %r -> %r (reporting_workflow=%r); "
            "run_id %s is the shared entry run.",
            workflow_run.get("path"),
            reporting_path,
            reporting_workflow,
            workflow_run.get("id"),
        )
        workflow_run["path"] = reporting_path

    return {
        "event_type": event_type,
        "repository": repo,
        "workflow_run": workflow_run,
    }


def dispatch_to_quartz(
    token: str,
    quartz_repo: str,
    workflow_file: str,
    workflow_ref: str,
    payload: dict[str, Any],
) -> None:
    """Trigger the ingest workflow in Quartz via workflow_dispatch.

    If the encoded `inputs` dict exceeds `GITHUB_INPUTS_MAX_CHARS`, the jobs
    array is stripped and `fetch_jobs` is set so the receiving side
    re-fetches them from the GitHub API.
    """
    payload_json = json.dumps(payload)
    fetch_jobs = False
    # workflow_dispatch inputs are strings; serialize the bool at the boundary.
    inputs = {"payload_json": payload_json, "fetch_jobs": str(fetch_jobs).lower()}

    original_inputs_size = len(json.dumps(inputs))
    if original_inputs_size > GITHUB_INPUTS_MAX_CHARS:
        original_payload_size = len(payload_json)
        payload.get("workflow_run", {}).pop("jobs", None)
        payload_json = json.dumps(payload)
        fetch_jobs = True
        inputs = {"payload_json": payload_json, "fetch_jobs": str(fetch_jobs).lower()}
        log.warning(
            "Inputs too large (%d encoded chars), stripped jobs (%d -> %d chars); "
            "receiving side will re-fetch",
            original_inputs_size,
            original_payload_size,
            len(payload_json),
        )

        # Belt-and-braces: if we're STILL over the cap after dropping jobs,
        # fail loud with diagnostic sizes for the two caller-controlled
        # fields (`inputs` and `captured_outputs`). We deliberately don't
        # truncate further — silently dropping more user-supplied data
        # would mean a lossy ingest with no audit trail.
        final_size = len(json.dumps(inputs))
        if final_size > GITHUB_INPUTS_MAX_CHARS:
            wr = payload.get("workflow_run", {})
            inputs_size = len(json.dumps(wr.get("inputs") or {}))
            captured_size = len(json.dumps(wr.get("captured_outputs") or {}))
            raise _DispatchConfigError(
                f"Inputs still exceed GitHub's {GITHUB_INPUTS_MAX_CHARS}-char "
                f"cap after stripping jobs ({final_size} chars). Likely culprits: "
                f"`inputs` ({inputs_size} chars), "
                f"`captured_outputs` ({captured_size} chars). "
                f"Reduce --embedded-inputs / --captured-outputs in the caller."
            )

    _github_api_request(
        token,
        "POST",
        f"/repos/{quartz_repo}/actions/workflows/{workflow_file}/dispatches",
        body={"ref": workflow_ref, "inputs": inputs},
    )
    log.info(
        "Dispatched to %s (ref: %s, payload: %d chars, fetch_jobs: %s)",
        quartz_repo,
        workflow_ref,
        len(payload_json),
        fetch_jobs,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quartz-repo",
        default=os.environ.get("QUARTZ_REPO", "ROCm/Quartz"),
        help="Target Quartz repository (owner/repo)",
    )
    p.add_argument(
        "--quartz-workflow-file",
        default=os.environ.get("QUARTZ_WORKFLOW_FILE", "receive_therock_data.yml"),
        help="Filename of the Quartz ingest workflow to dispatch to",
    )
    p.add_argument(
        "--quartz-workflow-ref",
        default=os.environ.get("QUARTZ_WORKFLOW_REF", "main"),
        help="Branch in Quartz repo where the ingest workflow lives",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("QUARTZ_HAULY_TOKEN", ""),
        help=(
            "GitHub token for dispatching. Must be a token issued by the "
            "Quartz Hauly GitHub App (see GH_APP_HAULY_ID / "
            "GH_APP_HAULY_PRIVATE_KEY). Defaults to QUARTZ_HAULY_TOKEN env var."
        ),
    )
    p.add_argument(
        "--embedded-inputs",
        default=os.environ.get("WORKFLOW_INPUTS", ""),
        help="JSON string of workflow inputs captured at dispatch time",
    )
    p.add_argument(
        "--captured-outputs",
        default=os.environ.get("WORKFLOW_CAPTURED_OUTPUTS", ""),
        help=(
            "JSON object of caller-supplied job context. Canonical pattern "
            "is `${{ toJSON(needs) }}` from a `notify_completed` job whose "
            "`needs:` lists every job whose result/outputs you want "
            "recorded; the receiver then has per-job `result` + `outputs` "
            "without any plucking on the caller side, and `--run-conclusion` "
            "is derived automatically (see below). NOT a dump of `env` / "
            "`vars` / `secrets`: anything passed here is stored verbatim, "
            "so the workflow author is responsible for not including secrets."
        ),
    )
    p.add_argument(
        "--run-conclusion",
        default=os.environ.get("RUN_CONCLUSION", ""),
        help=(
            "Final conclusion for the run (e.g. success/failure/cancelled). "
            "Optional when --captured-outputs carries a `${{ toJSON(needs) }}` "
            "blob; in that case the script derives it from the per-job "
            "`result` values: any unknown result -> failure (safe default), "
            "any failure -> failure, any cancelled -> cancelled, else "
            "success. Required when --run-phase=completed and no derivable "
            "captured-outputs is provided; we refuse to silently assume "
            "success. Ignored when --run-phase=started."
        ),
    )
    p.add_argument(
        "--run-phase",
        default=os.environ.get("RUN_PHASE", "completed"),
        choices=("started", "completed"),
        help="Phase of the workflow run: 'started' (in-progress) or 'completed' (default)",
    )
    p.add_argument(
        "--reporting-workflow",
        default=os.environ.get("REPORTING_WORKFLOW", ""),
        help=(
            "Filename of the workflow doing the reporting (e.g. "
            "'multi_arch_build_portable_linux.yml'). Overrides the "
            "API-reported workflow_run.path so a reusable workflow (which "
            "shares the top-level run's GITHUB_RUN_ID) is classified by its "
            "own file rather than the entry workflow's. Empty = keep the "
            "API path. Defaults to the REPORTING_WORKFLOW env var."
        ),
    )
    return p


def _derive_run_conclusion_from_captured_outputs(
    captured_outputs: dict[str, Any],
) -> str | None:
    """Derive a workflow-run conclusion from a `toJSON(needs)` blob.

    When callers pass `${{ toJSON(needs) }}` as `--captured-outputs`, every
    value carries a `result` key with one of the four values GitHub
    documents for `needs.<job>.result`: `success`, `failure`, `cancelled`,
    `skipped` (see https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#job-context).

    Precedence used to roll multiple per-job results up to a single
    workflow conclusion:

      * any `failure`   -> `failure`
      * any `cancelled` -> `cancelled`
      * any `success`   -> `success` (skipped jobs alongside successes
        don't drag the verdict down; matches the inline
        `contains(needs.*.result, 'failure') || ...` ternary callers
        used to write)
      * all `skipped`   -> `skipped` (every dependency was skipped, so
        the run did effectively nothing — don't lie about success;
        receiver maps `skipped` to a cancelled-style row state)

    Off-spec entries (not a dict, or missing `result` — e.g. future GHA
    additions of metadata top-level keys) are skipped with a warning so
    a single weird entry can't sink derivation from the rest. Returns
    `None` when `captured_outputs` is empty or no entry survives the
    shape check (e.g. a flat `{KEY: value}` map), so the strict
    "no silent success" rule still applies in that case.
    """
    if not captured_outputs:
        return None
    results: list[str] = []
    for key, entry in captured_outputs.items():
        if not isinstance(entry, dict) or "result" not in entry:
            log.warning(
                "captured-outputs entry %r isn't a `toJSON(needs)` shape "
                "(missing 'result' or not a dict); skipping.",
                key,
            )
            continue
        results.append(entry["result"])
    if not results:
        return None
    if "failure" in results:
        return "failure"
    if "cancelled" in results:
        return "cancelled"
    if "success" in results:
        return "success"
    return "skipped"


def _load_payload(
    args: argparse.Namespace,
    token: str,
    repo: str,
) -> dict[str, Any]:
    """Build the dispatch payload from CLI args + GITHUB_RUN_ID."""
    embedded = json.loads(args.embedded_inputs) if args.embedded_inputs else {}
    captured_outputs = (
        json.loads(args.captured_outputs) if args.captured_outputs else {}
    )
    return _build_payload(
        token=token,
        repo=repo,
        embedded_inputs=embedded,
        captured_outputs=captured_outputs,
        run_conclusion=args.run_conclusion,
        run_phase=args.run_phase,
        reporting_workflow=args.reporting_workflow,
    )


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    token = args.token
    if not token:
        parser.error("No GitHub token provided (--token or QUARTZ_HAULY_TOKEN)")

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        log.error("GITHUB_REPOSITORY not set")
        return 1

    if args.run_phase == "completed" and not args.run_conclusion:
        # Try deriving from --captured-outputs when caller passed a
        # `toJSON(needs)` blob — that's the canonical pattern under
        # notify_quartz.yml. Fall back to the strict refusal if the
        # captured-outputs payload isn't shaped that way (e.g. empty,
        # or a flat KEY/value map a legacy caller still provides).
        try:
            captured = (
                json.loads(args.captured_outputs) if args.captured_outputs else {}
            )
        except json.JSONDecodeError as exc:
            log.error("--captured-outputs is not valid JSON: %s", exc)
            return 1
        derived = _derive_run_conclusion_from_captured_outputs(captured)
        if derived is None:
            log.error(
                "--run-conclusion (or RUN_CONCLUSION env) is required when "
                "--run-phase=completed and --captured-outputs isn't a "
                "`toJSON(needs)` blob the script can derive from. "
                "Refusing to write an ambiguous status row."
            )
            return 1
        log.info(
            "Derived run_conclusion=%s from captured-outputs needs.*.result.",
            derived,
        )
        args.run_conclusion = derived

    try:
        payload = _load_payload(args, token, repo)
    except _DispatchConfigError as exc:
        log.error("%s", exc)
        return 1

    log.info("Constructed dispatch payload:")
    log.info("%s", json.dumps(payload, indent=2))

    try:
        dispatch_to_quartz(
            token=token,
            quartz_repo=args.quartz_repo,
            workflow_file=args.quartz_workflow_file,
            workflow_ref=args.quartz_workflow_ref,
            payload=payload,
        )
    except _DispatchConfigError as exc:
        log.error("%s", exc)
        return 1
    except HTTPError as exc:
        log.error("Dispatch failed: %s %s", exc.code, exc.read().decode())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
