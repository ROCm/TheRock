#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report which CI labels each run of a PR used, and what changed since the last tested pass.

Part of the `pr-ci-labels` skill (see reference.md, "The ci_labels.py report"). For the PR's
main CI workflow it replays the PR's label events to give each run the labels it used, reads
whether passing runs actually built or tested anything, and classifies the commits since the
last tested pass with the repository's skip-ci patterns. The verdict is go, caution, or hold.

Advisory: it always exits 0 unless gh fails. Whether a label set was safe for the diff is a
judgment for the agent and the human, not this tool.

Usage:
    python ci_labels.py --pr <number|url> [--repo owner/name] [--repo-path <clone>] [--json]

Requires: gh (authenticated). Python 3.11+ reads the skip-ci TOML files; older versions fall
back to a docs heuristic.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except (
    ModuleNotFoundError
):  # Python < 3.11: skip-pattern loading falls back to a heuristic
    tomllib = None

# Main PR CI workflows: "Multi-Arch CI" in TheRock, "TheRock Multi-Arch CI" in repositories
# that call it. The ASAN and nightly variants differ.
CI_WORKFLOWS = [r"\bMulti-Arch CI$"]
PR_URL_RE = re.compile(r"https?://[^/]+/([^/]+)/([^/]+)/pull/(\d+)")


def run(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    # Force UTF-8 decoding; gh/git output is UTF-8 even on Windows (default cp1252 corrupts it).
    p = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def gh_pr_json(pr: str, repo: str | None) -> dict:
    cmd = ["gh", "pr", "view", pr, "--json", "number,url,baseRefName,commits,labels"]
    if repo:
        cmd += ["--repo", repo]
    code, out, err = run(cmd)
    if code != 0:
        sys.stderr.write(f"ERROR: gh pr view failed: {err.strip()}\n")
        sys.exit(3)
    return json.loads(out)


def gh_api_json(path: str):
    code, out, _ = run(["gh", "api", path])
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def gh_api_pages(path: str) -> list | None:
    """Fetch every page of a list endpoint via gh (100 per page); None if any page fails."""
    items: list = []
    page = 1
    sep = "&" if "?" in path else "?"
    while True:
        data = gh_api_json(f"{path}{sep}per_page=100&page={page}")
        if not isinstance(data, list):
            return None
        items.extend(data)
        if len(data) < 100:
            return items
        page += 1


def repo_from_pr_url(url: str) -> str | None:
    """`owner/name` from a PR URL, or None."""
    m = PR_URL_RE.match(url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def resolve_repo(repo: str | None) -> str | None:
    if repo:
        return repo
    code, out, _ = run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    )
    return out.strip() if code == 0 and out.strip() else None


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _parse_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp, or None if unset, unparseable, or gh's zero date."""
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.year > 2000 else None


# Labels that make CI run less than its default selection.
NARROWING_LABELS = {
    "ci:skip",
    "ci:no-downstream-test",
    "ci:cpu-test-only",
    "ci:exactly",
    "test_filter:quick",
}
NARROWING_PREFIXES = ("ci:platform:", "test:")
# Labels that change what CI runs; others (for example "project: hipblaslt") are ignored.
CI_LABEL_PREFIXES = ("ci:", "test:", "test_filter:")
# The report looks up CI runs for at most this many of the PR's newest commits.
CI_LABELS_MAX_COMMITS = 30
THEROCK_REPO = "ROCm/TheRock"
SKIP_CI_BASE = "build_tools/github_actions/skip-ci-base.toml"
SKIP_CI_THEROCK = "build_tools/github_actions/skip-ci-config.toml"
SKIP_CI_EXTERNAL = ".github/skip-ci-config.toml"
# Used only when no skip-ci config can be read.
HEURISTIC_SKIP_SUFFIXES = (".md", ".rst")
# A label change this close to a run's start may or may not be among the labels its setup
# job read.
LABEL_RACE_SECONDS = 60
# Multi-Arch CI build and test jobs ("Linux::release / Build Multi-Arch Stages / ...",
# "Linux::release / Test gfx94X-dcgpu / ..."). Skip-only runs have only setup and summary jobs.
BUILD_TEST_JOB_RE = re.compile(r"::\S+ / (Build|Test)\b")
JOBS_PAGE_SIZE = 30  # larger pages of a big run's jobs can time out (HTTP 502)
JOBS_MAX_PAGES = 10
JOBS_MAX_RUNS = 10  # passing runs to check, newest first, for build or test evidence
# The commits API paginates a commit's files and stops listing at 3000.
COMMIT_FILES_PAGE_SIZE = 100
COMMIT_FILES_LIMIT = 3000


def is_narrowing_label(label: str) -> bool:
    return label in NARROWING_LABELS or label.startswith(NARROWING_PREFIXES)


def is_ci_label(label: str) -> bool:
    return label.startswith(CI_LABEL_PREFIXES)


def labels_at(events: list[dict], when: str | None) -> list[str]:
    """CI labels on the PR at time `when` (all events when None), replayed from label events.

    `events` are dicts with event ("labeled" or "unlabeled"), label, and created_at. Labels
    that do not affect CI are ignored.
    """
    cutoff = _parse_timestamp(when)
    labels: set[str] = set()
    ci_events = [e for e in events if is_ci_label(e.get("label", ""))]
    for ev in sorted(ci_events, key=lambda e: e.get("created_at") or ""):
        ts = _parse_timestamp(ev.get("created_at"))
        if cutoff and ts and ts > cutoff:
            break
        if ev.get("event") == "labeled":
            labels.add(ev.get("label", ""))
        elif ev.get("event") == "unlabeled":
            labels.discard(ev.get("label", ""))
    return sorted(labels)


def parse_skip_patterns(toml_text: str) -> list[str]:
    """The `[skip_ci] common` patterns of a skip-ci config: paths that skip CI everywhere."""
    if tomllib is None or not toml_text:
        return []
    data = tomllib.loads(toml_text)
    return list((data.get("skip_ci") or {}).get("common") or [])


def is_ci_skippable(path: str, patterns: list[str] | None) -> bool:
    """Whether CI ignores `path`. Matches CI's own rule (fnmatch, where `*` crosses `/`)."""
    if patterns:
        return any(fnmatch.fnmatch(path, p) for p in patterns)
    return (
        path.endswith(HEURISTIC_SKIP_SUFFIXES)
        or path.startswith("docs/")
        or "/docs/" in path
    )


def _read_repo_file(
    repo: str | None, path: str, ref: str | None, repo_path: str | None
) -> str | None:
    if repo_path:
        local = Path(repo_path) / path
        if local.is_file():
            return local.read_text(encoding="utf-8")
    if not repo:
        return None
    data = gh_api_json(f"repos/{repo}/contents/{path}" + (f"?ref={ref}" if ref else ""))
    if isinstance(data, dict) and data.get("content"):
        return base64.b64decode(data["content"]).decode("utf-8", "replace")
    return None


def load_skip_patterns(
    repo: str | None, base_ref: str | None, repo_path: str | None
) -> tuple[list[str] | None, str]:
    """(patterns, method): TheRock's base skip list plus the repository's own extension.

    Patterns are None when no config could be read; callers then use a docs heuristic.
    """
    if tomllib is None:
        return None, "docs heuristic (this Python has no tomllib)"
    is_therock = (repo or "").lower() == THEROCK_REPO.lower()
    base = _read_repo_file(
        THEROCK_REPO, SKIP_CI_BASE, None, repo_path if is_therock else None
    )
    extension = _read_repo_file(
        repo, SKIP_CI_THEROCK if is_therock else SKIP_CI_EXTERNAL, base_ref, repo_path
    )
    if base is None and extension is None:
        return None, "docs heuristic (no skip-ci config found)"
    return (
        parse_skip_patterns(base or "") + parse_skip_patterns(extension or ""),
        "skip-ci patterns",
    )


def uncertain_labels(events: list[dict], when: str | None) -> list[str]:
    """CI labels that changed within LABEL_RACE_SECONDS of `when`.

    Multi-Arch CI setup reads the PR's labels when it starts, and the API does not say which
    labels it saw. Replaying label events up to the run's start is exact only when no label
    changed close to it.
    """
    created = _parse_timestamp(when)
    if not created:
        return []
    near = set()
    for ev in events:
        ts = _parse_timestamp(ev.get("created_at"))
        label = ev.get("label", "")
        if (
            ts
            and is_ci_label(label)
            and abs((ts - created).total_seconds()) <= LABEL_RACE_SECONDS
        ):
            near.add(label)
    return sorted(near)


def labels_read_at(ci_run: dict) -> str | None:
    """When the run's setup job read the PR's labels: the start of its latest attempt.

    A re-run reads the labels present when it starts, not when the run was created.
    """
    return ci_run.get("labels_read_at") or ci_run.get("created_at")


def is_coverage_baseline(ci_run: dict) -> bool:
    """A passing run with job evidence that it built or tested something.

    Under `ci:skip`, or when every path is auto-skipped, the setup and summary jobs still
    succeed, so a passing conclusion alone is not evidence of coverage, and neither is a run
    whose jobs could not be read. The job evidence decides, not the replayed labels, which
    are an estimate.
    """
    return ci_run.get("conclusion") == "success" and ci_run.get("tested") is True


def analyze_ci_labels(
    runs: list[dict],
    events: list[dict] | None,
    commits: list[dict],
    commit_files: dict[str, list[str] | None],
    patterns: list[str] | None,
    current_labels: list[str] | None = None,
) -> dict:
    """Which labels each CI run used, and whether anything since the last tested pass escaped CI.

    `runs` are the PR's main-CI runs (id, name, head_sha, created_at, labels_read_at (when the
    latest attempt started; created_at when absent), status, conclusion, url, and `tested`:
    True or False from job evidence, None or absent when unknown). `events` are
    label events (see labels_at), or None when the label history could not be read; then
    `current_labels` (the PR's labels now) stands in and no run's labels are known.
    `commits` are the PR's commits, oldest first. `commit_files` maps a commit to its changed
    paths (None when unknown) for commits after the baseline.
    The verdict is go, caution, or hold (reference.md, "The ci_labels.py report").
    """
    history = events is not None
    events = events or []
    timeline = [
        {
            **r,
            "labels": labels_at(events, labels_read_at(r)),
            "uncertain_labels": uncertain_labels(events, labels_read_at(r)),
            "labels_unknown": not history,
        }
        for r in sorted(runs, key=lambda r: r.get("created_at") or "")
    ]
    if history:
        current = labels_at(events, None)
    else:
        current = sorted(lbl for lbl in current_labels or [] if is_ci_label(lbl))
    report = {
        "current_labels": current,
        "label_history_known": history,
        "runs": timeline,
        "last_passing_run": None,
        "commits_since": [],
        "label_changes_since": [],
        "verdict": "caution",
        "reason": "",
    }
    baselines = [r for r in timeline if is_coverage_baseline(r)]
    last = baselines[-1] if baselines else None
    report["last_passing_run"] = last
    unverified = [
        r
        for r in timeline
        if r.get("conclusion") == "success"
        and "ci:skip" not in r["labels"]
        and r.get("tested") is not True
        and r.get("tested") is not False
        and (not last or (r.get("created_at") or "") > (last.get("created_at") or ""))
    ]
    oids = [c.get("oid", "") for c in commits]
    if last and last.get("head_sha") not in oids:
        report["reason"] = (
            "The last passing run's commit is not in the PR's current history (rebased or "
            "force-pushed). Judge the diff since that run by hand."
        )
        return _note_gaps(report, history, unverified)

    start = oids.index(last["head_sha"]) + 1 if last else 0
    later = []
    for oid in oids[start:]:
        files = commit_files.get(oid)
        unknown = files is None
        code = [] if unknown else [f for f in files if not is_ci_skippable(f, patterns)]
        later.append(
            {
                "oid": oid,
                "files": files or [],
                "non_skippable": code,
                "files_unknown": unknown,
            }
        )
    report["commits_since"] = later
    code_commits = [c for c in later if c["non_skippable"] or c["files_unknown"]]

    if not last:
        if later and not code_commits:
            report["verdict"] = "go"
            report["reason"] = (
                "CI skips every change in this PR, so no build or test run is needed."
            )
        elif "ci:skip" in current:
            report["verdict"] = "hold"
            report["reason"] = (
                "No passing run built or tested this PR's code, and ci:skip is set. Remove "
                "it and run CI."
            )
        elif not timeline:
            report["reason"] = "No CI runs found for this PR's commits."
        else:
            report["reason"] = "No passing CI run built or tested this PR's code."
        return _note_gaps(report, history, unverified)

    started = _parse_timestamp(labels_read_at(last))
    report["label_changes_since"] = [
        e
        for e in sorted(events, key=lambda e: e.get("created_at") or "")
        if is_ci_label(e.get("label", ""))
        and started
        and (_parse_timestamp(e.get("created_at")) or started) > started
    ]
    narrowing_added = [
        lbl for lbl in current if is_narrowing_label(lbl) and lbl not in last["labels"]
    ]
    other_added = [
        lbl
        for lbl in current
        if not is_narrowing_label(lbl) and lbl not in last["labels"]
    ]
    if not code_commits and other_added and history:
        report["reason"] = (
            f"Label(s) {', '.join(other_added)} were added after the last passing run, "
            "and no run has used them. Re-run CI if they add coverage the change needs."
        )
    elif not later:
        report["verdict"] = "go"
        report["reason"] = "The PR's head commit passed a run that built or tested it."
    elif not code_commits:
        report["verdict"] = "go"
        report["reason"] = (
            f"Only CI-skippable changes in the {len(later)} commit(s) since the last passing "
            "run."
        )
    elif narrowing_added:
        report["verdict"] = "hold"
        report["reason"] = (
            f"{len(code_commits)} commit(s) since the last passing run change code, and "
            f"narrowing label(s) {', '.join(narrowing_added)} were added since. No passing "
            "run covers that code. Remove the label(s) and re-run, or confirm they are safe "
            "for the later diff."
        )
    else:
        report["reason"] = (
            f"{len(code_commits)} commit(s) since the last passing run change code, and no "
            "passing run covers them yet."
        )

    # A go that credits the run with coverage labels it may not have seen is only a caution.
    unsure = [
        lbl
        for lbl in last["uncertain_labels"]
        if lbl in current and not is_narrowing_label(lbl)
    ]
    if report["verdict"] == "go" and unsure:
        report["verdict"] = "caution"
        report["reason"] += (
            f" But {', '.join(unsure)} changed within {LABEL_RACE_SECONDS} seconds of that "
            "run starting, so it may not have used them. Check the run's setup log for "
            "the matching \"Label '...'\" line, or re-run CI."
        )
    return _note_gaps(report, history, unverified)


def _note_gaps(report: dict, history: bool, unverified: list[dict]) -> dict:
    """Say what evidence was missing.

    Without the label history, a go that rests on a run's labels is only a caution. A go for
    a fully CI-skipped PR does not depend on labels, so it stands.
    """
    if not history:
        if report["verdict"] == "go" and report["last_passing_run"]:
            report["verdict"] = "caution"
        report["reason"] += (
            " The PR's label history could not be read, so the labels each run used are "
            "unknown; the current labels were used instead."
        )
    if unverified and report["verdict"] != "go":
        report["reason"] += (
            f" {len(unverified)} newer passing run(s) could not be checked for build or test "
            "jobs, so they were not counted as coverage."
        )
    return report


def run_built_or_tested(repo: str, run_id) -> bool | None:
    """Whether a run has a successful build or test job; None if the jobs can't be read."""
    for page in range(1, JOBS_MAX_PAGES + 1):
        data = gh_api_json(
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=latest"
            f"&per_page={JOBS_PAGE_SIZE}&page={page}"
        )
        if not isinstance(data, dict):
            return None
        jobs = data.get("jobs") or []
        if any(
            j.get("conclusion") == "success"
            and BUILD_TEST_JOB_RE.search(j.get("name", ""))
            for j in jobs
        ):
            return True
        if len(jobs) < JOBS_PAGE_SIZE:
            return False
    return None


def fetch_commit_files(repo: str, oid: str) -> list[str] | None:
    """Every path a commit changed, or None if the list is unavailable or truncated."""
    files: list[str] = []
    page = 1
    while True:
        data = gh_api_json(
            f"repos/{repo}/commits/{oid}?per_page={COMMIT_FILES_PAGE_SIZE}&page={page}"
        )
        if not isinstance(data, dict):
            return None
        batch = data.get("files") or []
        files.extend(f.get("filename", "") for f in batch)
        if len(batch) < COMMIT_FILES_PAGE_SIZE:
            return files
        if len(files) >= COMMIT_FILES_LIMIT:
            return None
        page += 1


def fetch_ci_label_report(
    data: dict,
    repo: str | None,
    repo_path: str | None,
    workflows: list[str] = CI_WORKFLOWS,
) -> dict:
    """Gather label events, CI runs, and later commits' files via gh, then analyze them."""
    repo = repo or repo_from_pr_url(data.get("url", "")) or resolve_repo(None)
    if not repo:
        return {
            "verdict": "caution",
            "reason": "CI-label report skipped (repository unknown).",
            "current_labels": sorted(
                lbl.get("name", "")
                for lbl in data.get("labels") or []
                if is_ci_label(lbl.get("name", ""))
            ),
            "runs": [],
        }
    number = data.get("number")
    current_labels = [lbl.get("name", "") for lbl in data.get("labels") or []]
    raw_events = gh_api_pages(f"repos/{repo}/issues/{number}/events")
    events = (
        None
        if raw_events is None
        else [
            {
                "event": ev.get("event"),
                "label": (ev.get("label") or {}).get("name", ""),
                "created_at": ev.get("created_at"),
            }
            for ev in raw_events
            if ev.get("event") in ("labeled", "unlabeled")
        ]
    )
    commits = data.get("commits") or []
    runs = []
    for c in commits[-CI_LABELS_MAX_COMMITS:]:
        result = gh_api_json(
            f"repos/{repo}/actions/runs?head_sha={c.get('oid')}&event=pull_request&per_page=100"
        )
        for r in (result or {}).get("workflow_runs", []):
            if not matches_any(r.get("name", ""), workflows):
                continue
            runs.append(
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "head_sha": r.get("head_sha"),
                    "created_at": r.get("created_at"),
                    # Setup reads the PR's labels when each attempt starts, so a re-run
                    # uses the labels present at the re-run.
                    "labels_read_at": r.get("run_started_at") or r.get("created_at"),
                    "status": r.get("status"),
                    "conclusion": r.get("conclusion"),
                    "url": r.get("html_url"),
                }
            )
    # Job evidence for passing runs, newest first, until one that built or tested is found.
    # Runs whose jobs cannot be read are skipped, never counted as coverage.
    checked = 0
    for r in sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True):
        if r["conclusion"] != "success":
            continue
        if checked >= JOBS_MAX_RUNS:
            break
        checked += 1
        r["tested"] = run_built_or_tested(repo, r["id"])
        if r["tested"] is True:
            break
    patterns, method = load_skip_patterns(repo, data.get("baseRefName"), repo_path)
    first = analyze_ci_labels(runs, events, commits, {}, patterns, current_labels)
    files = {
        c["oid"]: fetch_commit_files(repo, c["oid"])
        for c in first.get("commits_since", [])
    }
    report = analyze_ci_labels(runs, events, commits, files, patterns, current_labels)
    report["skip_method"] = method
    report["commits_checked"] = min(len(commits), CI_LABELS_MAX_COMMITS)
    report["commits_total"] = len(commits)
    return report


def print_ci_label_report(report: dict) -> None:
    print(
        f"CI labels now: {', '.join(report.get('current_labels') or []) or 'none'} "
        f"(skip classification: {report.get('skip_method', 'n/a')})"
    )
    for r in report.get("runs", []):
        state = r.get("conclusion") or r.get("status")
        if state == "success" and "ci:skip" in (r.get("labels") or []):
            state = "success (ci:skip)"
        elif state == "success" and r.get("tested") is False:
            state = "success (built and tested nothing)"
        elif state == "success" and "tested" in r and r["tested"] is None:
            state = "success (jobs unreadable)"
        if r.get("labels_unknown"):
            labels = "labels unknown"
        else:
            labels = ", ".join(r.get("labels") or []) or "no labels"
        unsure = r.get("uncertain_labels") or []
        note = f" (uncertain: {', '.join(unsure)})" if unsure else ""
        print(
            f"  run {r.get('created_at')} {str(r.get('head_sha'))[:9]} {state}: "
            f"{labels}{note}  {r.get('url')}"
        )
    since = (
        "since last tested pass" if report.get("last_passing_run") else "no tested pass"
    )
    for c in report.get("commits_since", []):
        if c["files_unknown"]:
            what = "files unknown"
        elif c["non_skippable"]:
            what = f"{len(c['non_skippable'])} non-skippable file(s)"
        else:
            what = "CI-skippable only"
        print(f"  {since}: commit {c['oid'][:9]} ({what})")
    for e in report.get("label_changes_since", []):
        print(f"  {since}: {e['event']} {e['label']} at {e['created_at']}")
    if report.get("commits_total", 0) > report.get("commits_checked", 0):
        print(
            f"  Only the newest {report['commits_checked']} of {report['commits_total']} "
            "commits were checked for CI runs."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Which CI labels each run of a PR used, and what changed since the "
        "last tested pass (advisory)."
    )
    ap.add_argument("--pr", required=True, help="PR number or URL")
    ap.add_argument(
        "--repo", help="owner/name (default: from the PR URL, then the checkout)"
    )
    ap.add_argument(
        "--repo-path", help="local clone to read skip-ci configs from (default: GitHub)"
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    data = gh_pr_json(args.pr, args.repo)
    report = fetch_ci_label_report(data, args.repo, args.repo_path)
    if args.json:
        print(
            json.dumps(
                {"pr": data.get("number"), "url": data.get("url"), **report}, indent=2
            )
        )
    else:
        print(f"PR #{data.get('number')} - {data.get('url')}")
        print_ci_label_report(report)
        print(f"Verdict: {report.get('verdict', '').upper()} - {report.get('reason')}")


if __name__ == "__main__":
    main()
