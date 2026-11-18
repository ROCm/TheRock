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

Usage:
    python pr_quality_check.py --pr <number|url> [--repo owner/name] [--profile hipblaslt]
                               [--repo-path .] [--check-links] [--enforce] [--json]
                               [--check-attestation]

Requires: gh (authenticated). The stale-base check uses local git when --repo-path is given
(exact), otherwise falls back to the GitHub compare API (no clone needed; the API caps the file
list at 300, so on a long-diverged base it reports "inconclusive" rather than a false "fine").
M4 distinguishes a real work-tracking item (Jira key / closing-issue / issue link) from an
incidental cross-reference or timeline mention, which it surfaces as a soft note, not a blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

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

PROFILES = {
    "base": {
        "product_globs": [r"src/", r"library/", r"include/"],
        "test_globs": TEST_GLOBS + [r"(^|/)clients/"],
        "high_coupling": [],
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
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    data = gh_pr_json(args.pr, args.repo)
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

    blockers = [f for f in findings if f.severity == "BLOCKING"]
    warns = [f for f in findings if f.severity == "IMPORTANT"]

    if args.json:
        print(
            json.dumps(
                {
                    "pr": data.get("number"),
                    "url": data.get("url"),
                    "profile": args.profile,
                    "attestation": attestation_digest(data.get("url", "")),
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
        print(
            f"Attestation digest (for an APPROVED review): {attestation_digest(data.get('url', ''))}"
        )

    sys.exit(2 if (args.enforce and blockers) else 0)


if __name__ == "__main__":
    main()
