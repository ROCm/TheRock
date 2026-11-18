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

With --check-attestation it instead reports whether the PR carries a valid review attestation
(see reference.md); there the exit status is the answer (0 verified, 4 none, 5 invalid).

With --ci-labels it also reports the PR's CI-label timeline: the labels each CI run used, the
last passing run, and the commits and label changes since then (reference.md, "Choosing CI
labels"). Whether a label set was safe for the diff stays a judgment for the agent and human.

Usage:
    python pr_quality_check.py --pr <number|url> [--repo owner/name] [--profile hipblaslt]
                               [--repo-path .] [--check-links] [--enforce] [--json]
                               [--check-attestation] [--ci-labels]

Requires: gh (authenticated). The stale-base check uses local git when --repo-path is given
(exact), otherwise falls back to the GitHub compare API (no clone needed; the API caps the file
list at 300, so on a long-diverged base it reports "inconclusive" rather than a false "fine").
M4 distinguishes a real work-tracking item (Jira key / closing-issue / issue link) from an
incidental cross-reference or timeline mention, which it surfaces as a soft note, not a blocker.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except (
    ModuleNotFoundError
):  # Python < 3.11: skip-pattern loading falls back to a heuristic
    tomllib = None

# --- Profiles: generic defaults plus component overlays ------------------------------------

# Test-file patterns match whole path components or filename conventions, never a bare
# substring: a plain "test" would also match production files such as `src/contest.cpp`.
# Matching is case-insensitive except where noted, so CamelCase suffixes use a scoped
# case-sensitive group (`FooTest.cpp` matches; `contest.cpp` does not).
TEST_GLOBS = [
    r"(^|/)g?tests?/",  # test/, tests/, gtest/ directories (any case: Tests/)
    r"(^|/)test_[^/]+$",  # test_foo.py
    r"[^/]_tests?\.[^/]+$",  # foo_test.cpp, foo_tests.py
    r"\.(test|spec)\.[^/]+$",  # foo.test.ts, foo.spec.js
    r"(?-i:[a-z0-9]Tests?)\.[^/]+$",  # FooTest.cpp, FooTests.cs
    r"(^|/)conftest\.py$",
]

# Main PR CI workflows whose runs the --ci-labels report follows: "Multi-Arch CI" in TheRock,
# "TheRock Multi-Arch CI" in repositories that call it. The ASAN and nightly variants differ.
CI_WORKFLOWS = [r"\bMulti-Arch CI$"]

PROFILES = {
    "base": {
        "product_globs": [r"src/", r"library/", r"include/"],
        "test_globs": TEST_GLOBS + [r"(^|/)clients/"],
        "high_coupling": [],
        "ci_workflows": CI_WORKFLOWS,
        # Strong work-tracking signals (a real ticket/issue), vs. incidental cross-references.
        "tracker_regexes": [r"\b[A-Z][A-Z0-9]+-\d+\b"],
        "issue_link_regexes": [r"https?://\S+/issues/\d+"],
    },
    "therock": {
        # The build repo: python build tooling is the product code that should carry tests;
        # CMake and descriptor changes are judged by the overlay, not by this heuristic.
        "product_globs": [r"^build_tools/.*\.py$"],
        "test_globs": TEST_GLOBS,
        # Mirrors the overlay's "stale-base on high-coupling build files" list.
        "high_coupling": [
            r"^CMakeLists\.txt$",
            r"^cmake/",
            r"(^|/)\.gitmodules$",
            r"(^|/)artifact-[^/]*\.toml$",
            r"(^|/)BUILD_TOPOLOGY\.toml$",
            r"^\.github/workflows/",
        ],
        "tracker_regexes": [r"\b[A-Z][A-Z0-9]+-\d+\b"],
        "issue_link_regexes": [r"https?://\S+/issues/\d+"],
        "ci_workflows": CI_WORKFLOWS,
    },
    "hipblaslt": {
        "product_globs": [r"projects/hipblaslt/", r"tensilelite/"],
        "test_globs": TEST_GLOBS + [r"(^|/)clients/tests"],
        "high_coupling": [
            r"KernelWriter.*\.py",
            r"KernelWriterAssembly\.py",
            r"Components/",
            r"sgpr",
            r"register",
        ],
        "tracker_regexes": [r"\bAIHPBLAS-\d+\b", r"\bROCM-\d+\b"],
        "issue_link_regexes": [r"https?://\S+/issues/\d+"],
        "ci_workflows": CI_WORKFLOWS,
    },
}

REQUIRED_SECTIONS = ["Summary", "Risk", "Related", "Testing"]
WAIVER_RE = re.compile(r"\bW-[A-Z][A-Z0-9\-]*\b")
# Which rule each base waiver code addresses (reference.md, "Waiver codes"). A waiver only
# relaxes the rule it names; any other code leaves that obligation open.
TEST_WAIVERS = {"W-DOC", "W-BUILD", "W-REVERT", "W-HOTFIX", "W-FLAG"}
TRACKING_WAIVERS = {"W-NOTICKET", "W-DOC", "W-REVERT"}
BASE_WAIVERS = TEST_WAIVERS | TRACKING_WAIVERS
DOC_SUFFIXES = (".md", ".rst", ".txt")
CI_STALE_DAYS = 3


PR_URL_RE = re.compile(r"https?://[^/]+/([^/]+)/([^/]+)/pull/(\d+)")


def attestation_digest(pr_url: str) -> str | None:
    """Review-attestation digest: first 8 hex of sha256("<owner>/<repo>#<number>", lowercased).

    Deliberately keyed on the PR identity only (not the head SHA), so later pushes do not
    invalidate it. It is a checksum, not a secret: it stops a pasted template or a block copied
    from another PR, not a deliberate forgery. Keep in sync with reference.md.
    """
    m = PR_URL_RE.match(pr_url or "")
    if not m:
        return None
    key = f"{m.group(1)}/{m.group(2)}#{m.group(3)}".lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


# --- Attestation verification ---------------------------------------------------------------

ATTESTATION_MARKER = "<!-- rocm-pr-quality:attestation v1 -->"
# How much text after the marker belongs to one block (a full block is ~600 chars).
_ATTESTATION_SPAN = 1000
_ATT_DIGEST_RE = re.compile(r"\battestation\b[:\s`*]*([0-9a-f]{8})\b", re.IGNORECASE)
_ATT_COMMIT_RE = re.compile(r"\bcommit\b[:\s`*]*([0-9a-f]{7,40})\b", re.IGNORECASE)
# Full block: "Result: **APPROVED**"; one-line form: "rocm-pr-quality review APPROVED".
_ATT_APPROVED_RE = re.compile(r"(?:\bResult:?|\breview)\s*[`*]*APPROVED\b")

# Exit codes for --check-attestation (the exit status is the answer in that mode).
EXIT_ATTESTATION_VERIFIED = 0
EXIT_ATTESTATION_NONE = 4
EXIT_ATTESTATION_INVALID = 5


def parse_attestations(body: str) -> list[dict]:
    """Parse every attestation block (full or one-line form) in a comment/review body."""
    found = []
    for part in (body or "").split(ATTESTATION_MARKER)[1:]:
        part = part[:_ATTESTATION_SPAN]
        digest = _ATT_DIGEST_RE.search(part)
        commit = _ATT_COMMIT_RE.search(part)
        found.append(
            {
                "digest": digest.group(1).lower() if digest else None,
                "commit": commit.group(1).lower() if commit else None,
                "approved": bool(_ATT_APPROVED_RE.search(part)),
            }
        )
    return found


def verify_attestations(pr_url: str, sources: list[dict]) -> dict:
    """Check the comments/reviews of a PR for a skill attestation generated for *this* PR.

    `sources` are dicts with kind, author, association, created_at, url, and body. A block is
    valid when it records an APPROVED result and its digest matches the PR's own digest.
    The reviewed commit is informational and is never compared to the PR's head.
    """
    expected = attestation_digest(pr_url)
    found = []
    for src in sources:
        for blk in parse_attestations(src.get("body", "")):
            if not blk["approved"]:
                problem = "not an APPROVED result"
            elif not blk["digest"]:
                problem = "missing digest"
            elif blk["digest"] != expected:
                problem = "digest does not match this PR"
            else:
                problem = None
            found.append(
                {
                    **blk,
                    "valid": problem is None,
                    "problem": problem,
                    "kind": src.get("kind"),
                    "author": src.get("author"),
                    "association": src.get("association"),
                    "created_at": src.get("created_at"),
                    "updated_at": src.get("updated_at"),
                    "url": src.get("url"),
                }
            )
    if any(f["valid"] for f in found):
        status = "VERIFIED"
    elif found:
        status = "INVALID"
    else:
        status = "NONE"
    return {"status": status, "expected_digest": expected, "attestations": found}


def commits_since(commits: list[dict] | None, sha: str | None) -> int | None:
    """Commits pushed after `sha` in the PR's commit list, or None if it is not in the list
    (for example after a rebase or force-push)."""
    if not sha:
        return None
    oids = [c.get("oid", "") for c in commits or []]
    for i, oid in enumerate(oids):
        if oid.startswith(sha):
            return len(oids) - i - 1
    return None


def gh_api_pages(path: str) -> list:
    """Fetch every page of a list endpoint via gh (100 per page)."""
    items: list = []
    page = 1
    sep = "&" if "?" in path else "?"
    while True:
        data = gh_api_json(f"{path}{sep}per_page=100&page={page}")
        if not isinstance(data, list) or not data:
            break
        items.extend(data)
        if len(data) < 100:
            break
        page += 1
    return items


def fetch_attestation_sources(pr_url: str) -> list[dict]:
    """Collect PR conversation comments and review bodies, the places a block may be posted."""
    m = PR_URL_RE.match(pr_url or "")
    if not m:
        return []
    owner, repo, number = m.groups()
    sources = []
    for c in gh_api_pages(f"repos/{owner}/{repo}/issues/{number}/comments"):
        sources.append(
            {
                "kind": "comment",
                "author": (c.get("user") or {}).get("login"),
                "association": c.get("author_association"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "url": c.get("html_url"),
                "body": c.get("body") or "",
            }
        )
    for r in gh_api_pages(f"repos/{owner}/{repo}/pulls/{number}/reviews"):
        sources.append(
            {
                "kind": "review",
                "author": (r.get("user") or {}).get("login"),
                "association": r.get("author_association"),
                "created_at": r.get("submitted_at"),
                "url": r.get("html_url"),
                "body": r.get("body") or "",
            }
        )
    return sources


def run_attestation_check(data: dict, as_json: bool) -> int:
    """Report whether the PR carries a valid skill attestation; return the exit code."""
    url = data.get("url", "")
    result = verify_attestations(url, fetch_attestation_sources(url))
    for a in result["attestations"]:
        a["commits_since"] = commits_since(data.get("commits"), a["commit"])
    result = {"pr": data.get("number"), "url": url, **result}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"PR #{data.get('number')} - skill attestation: {result['status']}")
        for a in result["attestations"]:
            who = f"{a['author']} ({a['association']})"
            tag = "valid" if a["valid"] else f"invalid: {a['problem']}"
            when = a["created_at"]
            if a.get("updated_at") and a["updated_at"] != a["created_at"]:
                when += f", edited {a['updated_at']}"
            line = f"  [{tag}] {who} {a['kind']} {when} {a['url']}"
            if a["commit"]:
                since = a["commits_since"]
                note = (
                    "not in current history"
                    if since is None
                    else f"{since} commit(s) pushed since"
                )
                line += f"\n      reviewed commit {a['commit'][:9]} ({note}; allowed)"
            print(line)
        if result["status"] == "NONE":
            print("  No attestation block found in comments or review bodies.")
        print(
            "Note: the digest is a checksum, not a secret. It shows the block was "
            "generated for this PR, not who ran the skill."
        )
    return {
        "VERIFIED": EXIT_ATTESTATION_VERIFIED,
        "NONE": EXIT_ATTESTATION_NONE,
        "INVALID": EXIT_ATTESTATION_INVALID,
    }[result["status"]]


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
        "commits,number,closingIssuesReferences,labels"
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


def declared_waivers(body: str | None) -> set[str]:
    return set(WAIVER_RE.findall(body or ""))


def waiver_status(body: str | None, covers: set[str]) -> tuple[bool, set[str]]:
    """(covered, unknown): whether a declared waiver addresses this rule, and any declared
    codes the base does not define (overlay codes, which a reviewer must adjudicate)."""
    found = declared_waivers(body)
    return bool(found & covers), found - BASE_WAIVERS


def is_docs_only(data: dict) -> bool:
    files = [f.get("path", "") for f in data.get("files", [])]
    return bool(files) and all(f.endswith(DOC_SUFFIXES) for f in files)


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
    has_waiver, unknown_waivers = waiver_status(body, TRACKING_WAIVERS)
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
    elif is_docs_only(data):
        # Self-evident class (reference.md, "Self-evident changes"): no tracker required.
        findings.append(
            Finding(
                "M4",
                "INFO",
                "Docs-only change: the base policy does not require a tracker. A "
                "repository bot or overlay may still require one; that is its own rule.",
            )
        )
    elif unknown_waivers:
        findings.append(
            Finding(
                "M4",
                "IMPORTANT",
                f"No tracker; waiver code(s) {', '.join(sorted(unknown_waivers))} are "
                "not defined by the base skill. Pending adjudication by the reviewer.",
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
        if urls:
            findings.append(
                Finding(
                    "M5",
                    "INFO",
                    f"{len(set(urls))} full URL(s) in the PR body were not link-checked; "
                    "--check-links resolves bare #refs only.",
                )
            )
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
    docs_only = is_docs_only(data)
    has_waiver, unknown_waivers = waiver_status(data.get("body", ""), TEST_WAIVERS)
    if product and not tests and not docs_only and unknown_waivers and not has_waiver:
        findings.append(
            Finding(
                "M2",
                "IMPORTANT",
                f"{len(product)} product file(s) changed with no test files; waiver "
                f"code(s) {', '.join(sorted(unknown_waivers))} are not defined by the base "
                "skill. Pending adjudication by the reviewer.",
            )
        )
    elif product and not tests and not docs_only and not has_waiver:
        findings.append(
            Finding(
                "M2",
                "BLOCKING",
                f"{len(product)} product file(s) changed but no test files touched "
                "and no waiver declared. Add tests, a safe-default flag + tracker, "
                "or a written waiver. (Substance is judged separately by the agent.)",
            )
        )


def _parse_ci_timestamp(ts: str | None) -> datetime | None:
    """Parse a check timestamp, or None if unset or unparseable.

    gh reports an unset time (for example `completedAt` of a queued or running check) as
    the zero date `0001-01-01T00:00:00Z`, which is not a real completion time.
    """
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.year > 2000 else None


def check_ci_currency(data: dict, findings: list[Finding]) -> None:
    rollup = data.get("statusCheckRollup") or []
    times = []
    for c in rollup:
        ts = _parse_ci_timestamp(c.get("completedAt")) or _parse_ci_timestamp(
            c.get("startedAt")
        )
        if ts:
            times.append(ts)
    if not times:
        findings.append(
            Finding(
                "CI", "INFO", "No CI completion timestamps found to assess currency."
            )
        )
        return
    newest = max(times)
    # Clamp: clock skew can put a just-started check slightly in the future.
    age_days = max(0.0, (datetime.now(timezone.utc) - newest).total_seconds() / 86400.0)
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


def _git_has_commit(repo_path: str | None, ref: str) -> bool:
    code, _, _ = run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=repo_path
    )
    return code == 0


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
    head = data.get("headRefOid")
    pr_files = {f.get("path", "") for f in data.get("files", [])}
    # Measure from the reviewed PR's own head, never the clone's checked-out HEAD: the clone
    # may be on another branch, and the answer would then describe the wrong change.
    if not head or not _git_has_commit(repo_path, head):
        findings.append(
            Finding(
                "STALE-BASE",
                "INFO",
                "PR head commit is not in the local clone (fetch it, e.g. "
                "`git fetch origin pull/<n>/head`); using the GitHub compare API instead.",
            )
        )
        check_stale_base_api(data, prof, repo, findings)
        return
    # Pick one base ref and use it for both the merge-base and the diff.
    base_ref = next(
        (r for r in (f"origin/{base}", base) if _git_has_commit(repo_path, r)), None
    )
    if not base_ref:
        findings.append(
            Finding(
                "STALE-BASE",
                "INFO",
                f"Base branch '{base}' is not in the local clone; using the GitHub "
                "compare API instead.",
            )
        )
        check_stale_base_api(data, prof, repo, findings)
        return
    code, mb, err = run(["git", "merge-base", base_ref, head], cwd=repo_path)
    if code != 0:
        findings.append(
            Finding(
                "STALE-BASE", "INFO", f"Could not compute merge-base: {err.strip()}"
            )
        )
        return
    merge_base = mb.strip()
    code, out, err = run(
        ["git", "diff", "--name-only", f"{merge_base}..{base_ref}"], cwd=repo_path
    )
    if code != 0:
        findings.append(
            Finding("STALE-BASE", "INFO", f"Could not diff base: {err.strip()}")
        )
        return
    base_changed = set(filter(None, out.splitlines()))
    _report_overlap(pr_files & base_changed, prof, findings, "git")


# --- CI labels (reference.md, "Choosing CI labels") -----------------------------------------

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
# A label change this close to a run's creation may or may not be in its event payload.
LABEL_RACE_SECONDS = 60
# Multi-Arch CI build and test jobs ("Linux::release / Build Multi-Arch Stages / ...",
# "Linux::release / Test gfx94X-dcgpu / ..."). Skip-only runs have only setup and summary jobs.
BUILD_TEST_JOB_RE = re.compile(r"::\S+ / (Build|Test)\b")
JOBS_PAGE_SIZE = 30  # larger pages of a big run's jobs can time out (HTTP 502)
JOBS_MAX_PAGES = 10
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
    cutoff = _parse_ci_timestamp(when)
    labels: set[str] = set()
    ci_events = [e for e in events if is_ci_label(e.get("label", ""))]
    for ev in sorted(ci_events, key=lambda e: e.get("created_at") or ""):
        ts = _parse_ci_timestamp(ev.get("created_at"))
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

    A run's labels come from its triggering event, which the API does not expose. Replaying
    label events up to the run's creation is exact only when no label changed close to it.
    """
    created = _parse_ci_timestamp(when)
    if not created:
        return []
    near = set()
    for ev in events:
        ts = _parse_ci_timestamp(ev.get("created_at"))
        label = ev.get("label", "")
        if (
            ts
            and is_ci_label(label)
            and abs((ts - created).total_seconds()) <= LABEL_RACE_SECONDS
        ):
            near.add(label)
    return sorted(near)


def is_coverage_baseline(run: dict) -> bool:
    """A passing run that built or tested something.

    Under `ci:skip`, or when every path is auto-skipped, the setup and summary jobs still
    succeed, so a passing conclusion alone is not evidence of coverage.
    """
    return (
        run.get("conclusion") == "success"
        and "ci:skip" not in run.get("labels", [])
        and run.get("tested") is not False
    )


def analyze_ci_labels(
    runs: list[dict],
    events: list[dict],
    commits: list[dict],
    commit_files: dict[str, list[str] | None],
    patterns: list[str] | None,
) -> dict:
    """Which labels each CI run used, and whether anything since the last tested pass escaped CI.

    `runs` are the PR's main-CI runs (id, name, head_sha, created_at, status, conclusion, url,
    and optionally `tested`: True or False from job evidence, None when unknown). `events` are
    label events (see labels_at). `commits` are the PR's commits, oldest first. `commit_files`
    maps a commit to its changed paths (None when unknown) for commits after the baseline.
    The verdict is go, caution, or hold (reference.md, "Pre-merge: was CI bypassed since review?").
    """
    timeline = [
        {
            **r,
            "labels": labels_at(events, r.get("created_at")),
            "uncertain_labels": uncertain_labels(events, r.get("created_at")),
        }
        for r in sorted(runs, key=lambda r: r.get("created_at") or "")
    ]
    current = labels_at(events, None)
    report = {
        "current_labels": current,
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
    oids = [c.get("oid", "") for c in commits]
    if last and last.get("head_sha") not in oids:
        report["reason"] = (
            "The last passing run's commit is not in the PR's current history (rebased or "
            "force-pushed). Judge the diff since that run by hand."
        )
        return report

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
        return report

    started = _parse_ci_timestamp(last.get("created_at"))
    report["label_changes_since"] = [
        e
        for e in sorted(events, key=lambda e: e.get("created_at") or "")
        if is_ci_label(e.get("label", ""))
        and started
        and (_parse_ci_timestamp(e.get("created_at")) or started) > started
    ]
    narrowing_added = [
        lbl for lbl in current if is_narrowing_label(lbl) and lbl not in last["labels"]
    ]
    other_added = [
        lbl
        for lbl in current
        if not is_narrowing_label(lbl) and lbl not in last["labels"]
    ]
    if not later:
        if other_added:
            report["reason"] = (
                f"Label(s) {', '.join(other_added)} were added after the last passing run, "
                "and no run has used them. Re-run CI if they add coverage the change needs."
            )
        else:
            report["verdict"] = "go"
            report["reason"] = (
                "The PR's head commit passed a run that built or tested it."
            )
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
            "run being created, so it may not have used them. Check the run's setup log for "
            "the matching \"Label '...'\" line, or re-run CI."
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
    data: dict, repo: str | None, prof: dict, repo_path: str | None
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
    events = [
        {
            "event": ev.get("event"),
            "label": (ev.get("label") or {}).get("name", ""),
            "created_at": ev.get("created_at"),
        }
        for ev in gh_api_pages(f"repos/{repo}/issues/{number}/events")
        if ev.get("event") in ("labeled", "unlabeled")
    ]
    commits = data.get("commits") or []
    runs = []
    for c in commits[-CI_LABELS_MAX_COMMITS:]:
        result = gh_api_json(
            f"repos/{repo}/actions/runs?head_sha={c.get('oid')}&event=pull_request&per_page=100"
        )
        for r in (result or {}).get("workflow_runs", []):
            if not matches_any(
                r.get("name", ""), prof.get("ci_workflows", CI_WORKFLOWS)
            ):
                continue
            runs.append(
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "head_sha": r.get("head_sha"),
                    # A re-run reuses its original event payload, so the labels it used are
                    # the ones present when the run was created, not when it was re-run.
                    "created_at": r.get("created_at"),
                    "status": r.get("status"),
                    "conclusion": r.get("conclusion"),
                    "url": r.get("html_url"),
                }
            )
    # Job evidence for passing runs, newest first, until one that built or tested is found.
    for r in sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True):
        if r["conclusion"] != "success":
            continue
        if "ci:skip" in labels_at(events, r["created_at"]):
            continue
        r["tested"] = run_built_or_tested(repo, r["id"])
        if r["tested"] is not False:
            break
    patterns, method = load_skip_patterns(repo, data.get("baseRefName"), repo_path)
    first = analyze_ci_labels(runs, events, commits, {}, patterns)
    files = {
        c["oid"]: fetch_commit_files(repo, c["oid"])
        for c in first.get("commits_since", [])
    }
    report = analyze_ci_labels(runs, events, commits, files, patterns)
    report["skip_method"] = method
    report["commits_checked"] = min(len(commits), CI_LABELS_MAX_COMMITS)
    report["commits_total"] = len(commits)
    return report


def ci_label_finding(report: dict) -> Finding:
    severity = "INFO" if report.get("verdict") == "go" else "IMPORTANT"
    return Finding(
        "CI-LABELS",
        severity,
        f"{report.get('verdict', '').upper()}: {report.get('reason')}",
    )


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
    ap.add_argument(
        "--check-attestation",
        action="store_true",
        help="only report whether the PR carries a valid skill attestation (skips the "
        "quality checks); exit 0 if verified, 4 if none found, 5 if invalid",
    )
    ap.add_argument(
        "--ci-labels",
        action="store_true",
        help="also report the labels each CI run used and what changed since the last "
        "passing run (pre-merge 'CI not bypassed' check; network)",
    )
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    data = gh_pr_json(args.pr, args.repo)
    # Prefer the PR's own repository over the checkout gh would otherwise infer.
    args.repo = args.repo or repo_from_pr_url(data.get("url", ""))
    if args.check_attestation:
        sys.exit(run_attestation_check(data, args.json))
    cross_refs = cross_reference_numbers(args.pr, args.repo)
    findings: list[Finding] = []

    check_description(data.get("body", ""), findings)
    check_tracking(data, prof, findings, cross_refs)
    check_related_links(data, findings, args.check_links, args.repo)
    check_tests(data, prof, findings)
    check_ci_currency(data, findings)
    check_stale_base(data, prof, args.repo_path, args.repo, findings)
    ci_labels = None
    if args.ci_labels:
        ci_labels = fetch_ci_label_report(data, args.repo, prof, args.repo_path)
        findings.append(ci_label_finding(ci_labels))

    blockers = [f for f in findings if f.severity == "BLOCKING"]
    warns = [f for f in findings if f.severity == "IMPORTANT"]

    if args.json:
        result = {
            "pr": data.get("number"),
            "url": data.get("url"),
            "profile": args.profile,
            "attestation": attestation_digest(data.get("url", "")),
            "blockers": len(blockers),
            "warnings": len(warns),
            "findings": [f.as_dict() for f in findings],
        }
        if ci_labels is not None:
            result["ci_labels"] = ci_labels
        print(json.dumps(result, indent=2))
    else:
        print(f"PR #{data.get('number')} [{args.profile}] - {data.get('url')}")
        order = {"BLOCKING": 0, "IMPORTANT": 1, "INFO": 2}
        for f in sorted(findings, key=lambda x: order[x.severity]):
            print(f"  [{f.severity:9}] {f.rule:11} {f.message}")
        if ci_labels is not None:
            print_ci_label_report(ci_labels)
        print(
            f"Summary: {len(blockers)} blocking, {len(warns)} important. "
            "Substance/M3 judged separately by the agent + human reviewer."
        )
        print(
            f"Attestation digest (for an APPROVED review): {attestation_digest(data.get('url', ''))}"
        )

    sys.exit(2 if (args.enforce and blockers) else 0)


if __name__ == "__main__":
    main()
