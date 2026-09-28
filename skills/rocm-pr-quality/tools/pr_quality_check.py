#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic subset of the ROCm PR-quality MUST checks.

The skill invokes it locally for fast feedback. It covers only the *mechanical* checks --
description completeness, work-tracking presence (M4), related-link presence/resolution (M5),
a product-code-without-tests heuristic (M2), CI currency, and stale-base/adjacent-file overlap.
Semantic judgments -- test substance, the mutation question, and M3 (never disable tests to
green CI) -- stay with the agent and the human reviewer; they are intentionally NOT enforced here.

Advisory by default (always exits 0). Pass --enforce to exit non-zero on blockers.

Usage:
    python pr_quality_check.py --pr <number|url> [--repo owner/name] [--profile hipblaslt]
                               [--repo-path .] [--check-links] [--enforce] [--json]

Requires: gh (authenticated). The stale-base check uses local git when --repo-path is given
(exact), otherwise falls back to the GitHub compare API (no clone needed; the API caps the file
list at 300, so on a long-diverged base it reports "inconclusive" rather than a false "fine").
M4 distinguishes a real work-tracking item (Jira key / closing-issue / issue link) from an
incidental cross-reference or timeline mention, which it surfaces as a soft note, not a blocker.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

# --- Profiles: generic defaults plus component overlays ------------------------------------

PROFILES = {
    "base": {
        "product_globs": [r"src/", r"library/", r"include/"],
        "test_globs": [r"test", r"clients/", r"/tests?/"],
        "high_coupling": [],
        # Strong work-tracking signals (a real ticket/issue), vs. incidental cross-references.
        "tracker_regexes": [r"\b[A-Z][A-Z0-9]+-\d+\b"],
        "issue_link_regexes": [r"https?://\S+/issues/\d+"],
    },
    "hipblaslt": {
        "product_globs": [r"projects/hipblaslt/", r"tensilelite/"],
        "test_globs": [r"clients/tests", r"test", r"/tests?/"],
        "high_coupling": [
            r"KernelWriter.*\.py",
            r"KernelWriterAssembly\.py",
            r"Components/",
            r"sgpr",
            r"register",
        ],
        "tracker_regexes": [r"\bAIHPBLAS-\d+\b", r"\bROCM-\d+\b"],
        "issue_link_regexes": [r"https?://\S+/issues/\d+"],
    },
}

REQUIRED_SECTIONS = ["Summary", "Risk", "Related", "Testing"]
WAIVER_RE = re.compile(r"\bW-[A-Z\-]+\b")
CI_STALE_DAYS = 3


class Finding:
    def __init__(self, rule: str, severity: str, message: str):
        self.rule = rule
        self.severity = severity  # BLOCKING | IMPORTANT | INFO
        self.message = message

    def as_dict(self):
        return {"rule": self.rule, "severity": self.severity, "message": self.message}


def run(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    # Force UTF-8 decoding; gh/git output is UTF-8 even on Windows (default cp1252 corrupts it).
    p = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def gh_pr_json(pr: str, repo: str | None) -> dict:
    fields = (
        "title,body,files,baseRefName,headRefName,headRefOid,url,statusCheckRollup,"
        "commits,number,closingIssuesReferences"
    )
    cmd = ["gh", "pr", "view", pr, "--json", fields]
    if repo:
        cmd += ["--repo", repo]
    code, out, err = run(cmd)
    if code != 0:
        sys.stderr.write(f"ERROR: gh pr view failed: {err.strip()}\n")
        sys.exit(3)
    return json.loads(out)


def gh_api_json(path: str):
    code, out, err = run(["gh", "api", path])
    if code != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def resolve_repo(repo: str | None) -> str | None:
    if repo:
        return repo
    code, out, _ = run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    )
    return out.strip() if code == 0 and out.strip() else None


def cross_reference_numbers(pr: str, repo: str | None) -> list[str]:
    """Best-effort: PR/issue numbers that cross-reference this PR via the timeline.

    A cross-reference (e.g. 'mentioned this pull request #7326') is NOT a work-tracking
    item, but it is worth surfacing so M4 degrades to a soft note rather than a hard block.
    """
    repo = resolve_repo(repo)
    if not repo:
        return []
    m = re.search(r"(\d+)$", pr)
    num = m.group(1) if m else pr
    data = gh_api_json(f"repos/{repo}/issues/{num}/timeline?per_page=100")
    if not isinstance(data, list):
        return []
    refs = []
    for ev in data:
        if ev.get("event") == "cross-referenced":
            src = (ev.get("source") or {}).get("issue") or {}
            if src.get("number"):
                refs.append(str(src["number"]))
    return sorted(set(refs))


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(re.search(p, path, re.IGNORECASE) for p in patterns)


def check_description(body: str, findings: list[Finding]) -> None:
    body = body or ""
    missing = [
        s for s in REQUIRED_SECTIONS if not re.search(rf"#+\s*{s}", body, re.IGNORECASE)
    ]
    if missing:
        findings.append(
            Finding(
                "DESC",
                "IMPORTANT",
                f"PR body missing section(s): {', '.join(missing)}.",
            )
        )
    for empty in re.findall(r"(Devices?|Link):\s*N/?A", body, re.IGNORECASE):
        findings.append(
            Finding(
                "DESC",
                "INFO",
                "Empty placeholder field present; omit it instead of 'N/A'.",
            )
        )
        break


def check_tracking(
    data: dict, prof: dict, findings: list[Finding], cross_refs: list[str]
) -> None:
    haystack = " ".join(
        [data.get("title", ""), data.get("body", ""), data.get("headRefName", "")]
    )
    body = data.get("body", "") or ""
    has_jira = any(re.search(rx, haystack) for rx in prof["tracker_regexes"])
    has_issue_link = any(
        re.search(rx, body) for rx in prof.get("issue_link_regexes", [])
    )
    closing = data.get("closingIssuesReferences") or []
    has_waiver = bool(WAIVER_RE.search(body))
    body_refs = re.findall(r"(?<!\w)#(\d+)\b", body)
    mentions = sorted(set(body_refs) | set(cross_refs))

    if has_jira or has_issue_link or closing:
        findings.append(Finding("M4", "INFO", "Work-tracking reference found."))
    elif has_waiver:
        findings.append(
            Finding(
                "M4",
                "INFO",
                "No tracker, but a waiver is declared; reviewer to adjudicate.",
            )
        )
    elif mentions:
        # A cross-reference / mention is not itself a tracker -- soft note, not a hard block.
        shown = ", ".join("#" + m for m in mentions[:5])
        findings.append(
            Finding(
                "M4",
                "IMPORTANT",
                f"No work-tracking ticket, but cross-reference(s)/mention(s) found "
                f"({shown}). A related PR or mention is not work tracking -- confirm "
                "a ticket/issue or declare a credible no-tracker waiver.",
            )
        )
    else:
        findings.append(
            Finding(
                "M4",
                "BLOCKING",
                "No work-tracking reference and no cross-reference or declared "
                "waiver. Add a tracker, or a credible no-tracker waiver.",
            )
        )


def check_related_links(
    data: dict, findings: list[Finding], check_links: bool, repo: str | None
) -> None:
    body = data.get("body", "") or ""
    urls = re.findall(r"https?://[^\s)]+", body)
    issue_refs = re.findall(r"(?<!\w)#(\d+)\b", body)
    if not urls and not issue_refs:
        findings.append(
            Finding(
                "M5",
                "IMPORTANT",
                "No related links found in the PR body. Link the work item, the "
                "defect fixed, and directly related PRs.",
            )
        )
        return
    if check_links:
        for num in set(issue_refs):
            cmd = ["gh", "issue", "view", num, "--json", "number"]
            if repo:
                cmd += ["--repo", repo]
            code, _, _ = run(cmd)
            if code != 0:
                # could be a PR rather than an issue
                cmd2 = ["gh", "pr", "view", num, "--json", "number"]
                if repo:
                    cmd2 += ["--repo", repo]
                code2, _, _ = run(cmd2)
                if code2 != 0:
                    findings.append(
                        Finding(
                            "M5",
                            "IMPORTANT",
                            f"Reference #{num} did not resolve to an issue or PR.",
                        )
                    )


def check_tests(data: dict, prof: dict, findings: list[Finding]) -> None:
    files = [f.get("path", "") for f in data.get("files", [])]
    product = [f for f in files if matches_any(f, prof["product_globs"])]
    tests = [f for f in files if matches_any(f, prof["test_globs"])]
    docs_only = files and all(f.endswith((".md", ".rst", ".txt")) for f in files)
    has_waiver = bool(WAIVER_RE.search(data.get("body", "") or ""))
    if product and not tests and not docs_only and not has_waiver:
        findings.append(
            Finding(
                "M2",
                "BLOCKING",
                f"{len(product)} product file(s) changed but no test files touched "
                "and no waiver declared. Add tests, a safe-default flag + tracker, "
                "or a written waiver. (Substance is judged separately by the agent.)",
            )
        )


def check_ci_currency(data: dict, findings: list[Finding]) -> None:
    rollup = data.get("statusCheckRollup") or []
    times = []
    for c in rollup:
        ts = c.get("completedAt") or c.get("startedAt")
        if ts:
            try:
                times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
    if not times:
        findings.append(
            Finding(
                "CI", "INFO", "No CI completion timestamps found to assess currency."
            )
        )
        return
    newest = max(times)
    age_days = (datetime.now(timezone.utc) - newest).total_seconds() / 86400.0
    if age_days > CI_STALE_DAYS:
        findings.append(
            Finding(
                "CI",
                "IMPORTANT",
                f"Newest CI result is ~{age_days:.1f} days old (> {CI_STALE_DAYS}). "
                "Consider a rebase + re-run before merge.",
            )
        )
    else:
        findings.append(
            Finding("CI", "INFO", f"Newest CI result ~{age_days:.1f} days old.")
        )


def _report_overlap(
    overlap: set[str], prof: dict, findings: list[Finding], how: str
) -> None:
    if not overlap:
        findings.append(
            Finding(
                "STALE-BASE",
                "INFO",
                f"No adjacent-file overlap with base ({how}); merge is fine.",
            )
        )
        return
    hot = (
        [f for f in overlap if matches_any(f, prof["high_coupling"])]
        if prof["high_coupling"]
        else []
    )
    if hot:
        findings.append(
            Finding(
                "STALE-BASE",
                "BLOCKING",
                "Overlap on high-coupling file(s) since divergence "
                f"({how}): {', '.join(sorted(hot))}. Rebase + re-run before merge.",
            )
        )
    else:
        findings.append(
            Finding(
                "STALE-BASE",
                "IMPORTANT",
                f"Overlap with base on {len(overlap)} file(s) ({how}); "
                "consider rebase + re-run.",
            )
        )


def check_stale_base_api(
    data: dict, prof: dict, repo: str | None, findings: list[Finding]
) -> None:
    """No local clone: use the compare API to find the merge-base, then the files the base
    branch advanced with since divergence, and intersect with the PR's files."""
    repo = resolve_repo(repo)
    base = data.get("baseRefName", "develop")
    head = data.get("headRefOid")
    pr_files = {f.get("path", "") for f in data.get("files", [])}
    if not repo or not head:
        findings.append(
            Finding(
                "STALE-BASE", "INFO", "Stale-base check skipped (need repo + head SHA)."
            )
        )
        return
    cmp1 = gh_api_json(f"repos/{repo}/compare/{base}...{head}")
    if not cmp1 or not (cmp1.get("merge_base_commit") or {}).get("sha"):
        findings.append(
            Finding(
                "STALE-BASE",
                "INFO",
                "Stale-base check skipped (merge-base not resolvable via API; "
                "head may be from a deleted/fork branch -- use --repo-path).",
            )
        )
        return
    mb = cmp1["merge_base_commit"]["sha"]
    cmp2 = gh_api_json(f"repos/{repo}/compare/{mb}...{base}")
    if not cmp2:
        findings.append(
            Finding(
                "STALE-BASE", "INFO", "Stale-base check skipped (base compare failed)."
            )
        )
        return
    base_changed = {f.get("filename", "") for f in (cmp2.get("files") or [])}
    overlap = pr_files & base_changed
    truncated = len(cmp2.get("files") or []) >= 300
    if truncated and not overlap:
        # The base advanced past the API's 300-file cap, so "no overlap" is not trustworthy.
        findings.append(
            Finding(
                "STALE-BASE",
                "IMPORTANT",
                "Inconclusive: base advanced past the API's 300-file limit, so the "
                "adjacent-file check is incomplete. Re-run nearer merge time, or use "
                "--repo-path <clone> for an exact result.",
            )
        )
        return
    _report_overlap(overlap, prof, findings, "API (truncated)" if truncated else "API")


def check_stale_base(
    data: dict,
    prof: dict,
    repo_path: str | None,
    repo: str | None,
    findings: list[Finding],
) -> None:
    if not repo_path:
        check_stale_base_api(data, prof, repo, findings)
        return
    base = data.get("baseRefName", "develop")
    pr_files = {f.get("path", "") for f in data.get("files", [])}
    code, mb, err = run(["git", "merge-base", f"origin/{base}", "HEAD"], cwd=repo_path)
    if code != 0:
        code, mb, err = run(["git", "merge-base", base, "HEAD"], cwd=repo_path)
    if code != 0:
        findings.append(
            Finding(
                "STALE-BASE", "INFO", f"Could not compute merge-base: {err.strip()}"
            )
        )
        return
    merge_base = mb.strip()
    code, out, err = run(
        ["git", "diff", "--name-only", f"{merge_base}..origin/{base}"], cwd=repo_path
    )
    if code != 0:
        findings.append(
            Finding("STALE-BASE", "INFO", f"Could not diff base: {err.strip()}")
        )
        return
    base_changed = set(filter(None, out.splitlines()))
    _report_overlap(pr_files & base_changed, prof, findings, "git")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ROCm PR quality deterministic checks (advisory)."
    )
    ap.add_argument("--pr", required=True, help="PR number or URL")
    ap.add_argument("--repo", help="owner/name (gh infers from cwd if omitted)")
    ap.add_argument(
        "--profile", default="base", choices=sorted(PROFILES), help="overlay profile"
    )
    ap.add_argument("--repo-path", help="local clone path for the stale-base check")
    ap.add_argument(
        "--check-links", action="store_true", help="resolve #refs via gh (network)"
    )
    ap.add_argument(
        "--enforce", action="store_true", help="exit non-zero on BLOCKING findings"
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    data = gh_pr_json(args.pr, args.repo)
    cross_refs = cross_reference_numbers(args.pr, args.repo)
    findings: list[Finding] = []

    check_description(data.get("body", ""), findings)
    check_tracking(data, prof, findings, cross_refs)
    check_related_links(data, findings, args.check_links, args.repo)
    check_tests(data, prof, findings)
    check_ci_currency(data, findings)
    check_stale_base(data, prof, args.repo_path, args.repo, findings)

    blockers = [f for f in findings if f.severity == "BLOCKING"]
    warns = [f for f in findings if f.severity == "IMPORTANT"]

    if args.json:
        print(
            json.dumps(
                {
                    "pr": data.get("number"),
                    "url": data.get("url"),
                    "profile": args.profile,
                    "blockers": len(blockers),
                    "warnings": len(warns),
                    "findings": [f.as_dict() for f in findings],
                },
                indent=2,
            )
        )
    else:
        print(f"PR #{data.get('number')} [{args.profile}] - {data.get('url')}")
        order = {"BLOCKING": 0, "IMPORTANT": 1, "INFO": 2}
        for f in sorted(findings, key=lambda x: order[x.severity]):
            print(f"  [{f.severity:9}] {f.rule:11} {f.message}")
        print(
            f"Summary: {len(blockers)} blocking, {len(warns)} important. "
            "Substance/M3 judged separately by the agent + human reviewer."
        )

    sys.exit(2 if (args.enforce and blockers) else 0)


if __name__ == "__main__":
    main()
