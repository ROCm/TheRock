"""
Main test script to policy-check PRs and report results in a comment. This is the core of the bot's
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
import yaml

NOT_READY_LABEL = "Not ready to Review"

# CodeQL is disabled at the bot level for now. When False, the bot never queries
# the Code Scanning API and the CodeQL row always renders as "To Be Enabled".
CODEQL_ENABLED = False

# Only these policy checks trigger the "Not ready to Review" label when they
# fail. Failures of other checks (Branch Name, PR Size, Draft PR, pre-commit,
# CodeQL, …) do NOT add the label.
LABEL_TRIGGER_CHECKS = {
    "PR Title/Description",
    "Unit Test",
    "Forbidden Files",
}

# Fixed display order for rows in the results table (by check name). Any row
# whose name is not listed here is appended after these, in its original order.
TABLE_ORDER = [
    "Branch Name",
    "PR Title/Description",
    "Forbidden Files",
    "Unit Test",
    "pre-commit",
    "Draft PR",
    "PR Size",
    "Feature Flag",
    "Code Coverage",
    "CodeQL",
    "therock-pr-bot",
]


@dataclass(frozen=True)
class FailureComment:
    title: str
    body: str


@dataclass
class CheckResult:
    name: str
    icon: str
    passed: bool
    details: List[str]
    pending: bool = False
    wip: bool = False
    tbe: bool = False


@dataclass(frozen=True)
class Policy:
    branch_patterns: List[re.Pattern[str]]
    title_patterns: List[re.Pattern[str]]
    title_min_length: int
    title_max_length: int
    description_min_length: int
    description_issue_patterns: List[re.Pattern[str]]
    block_draft: bool
    forbidden_title_patterns: List[re.Pattern[str]]
    max_files_changed: int
    max_total_changes: int
    max_single_file_changes: int
    forbidden_paths: List[str]
    unit_test_code_extensions: List[str]
    unit_test_patterns: List[str]
    unit_test_exempt_paths: List[str]
    required_checks: List[str]
    precommit_failure_comment: Optional[FailureComment]


def find_repo_root(start: Path) -> Path:
    """
    Location-independent: works from any directory.
    """
    cur = start.resolve()
    for _ in range(50):
        if (cur / ".git").exists() or (cur / ".github").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(f"Could not locate repo root from: {start}")


def load_policy(policy_path: Path) -> Policy:
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("policy.yml must be a mapping/object")

    pr = raw.get("pr", {}) or {}
    diff = raw.get("diff", {}) or {}
    checks = raw.get("checks", {}) or {}

    patterns_raw = pr.get("branch_name_patterns", []) or []
    branch_patterns = [re.compile(str(p)) for p in patterns_raw]

    # PR title rules now live under the nested `title:` mapping.
    title_cfg = pr.get("title", {}) or {}
    title_patterns_raw = title_cfg.get("pattern", []) or []
    title_patterns = [re.compile(str(p)) for p in title_patterns_raw]
    title_min_length = int(title_cfg.get("title_min_length", 0) or 0)
    title_max_length = int(title_cfg.get("title_max_length", 0) or 0)

    # PR description rules.
    description_cfg = pr.get("description", {}) or {}
    description_min_length = int(description_cfg.get("min_length", 0) or 0)
    description_issue_raw = description_cfg.get("issue_reference_patterns", []) or []
    description_issue_patterns = [re.compile(str(p)) for p in description_issue_raw]

    # Block drafts / WIP titles.
    block_draft = bool(pr.get("block_draft", False))
    forbidden_title_raw = pr.get("forbidden_title_patterns", []) or []
    forbidden_title_patterns = [re.compile(str(p)) for p in forbidden_title_raw]

    # PR "reviewable shape" limits live under the diff: section.
    max_files_changed = int(diff.get("max_files_changed", 0) or 0)
    max_total_changes = int(diff.get("max_total_changes", 0) or 0)
    max_single_file_changes = int(diff.get("max_single_file_changes", 0) or 0)

    forbidden_paths = [str(p) for p in (diff.get("forbidden_paths", []) or [])]

    # Unit test rules live under pr.unit_tests.
    unit_cfg = pr.get("unit_tests", {}) or {}
    unit_test_code_extensions = [
        str(e).lower() for e in (unit_cfg.get("code_extensions", []) or [])
    ]
    unit_test_patterns = [
        str(p) for p in (unit_cfg.get("test_file_patterns", []) or [])
    ]
    unit_test_exempt_paths = [str(p) for p in (unit_cfg.get("exempt_paths", []) or [])]

    required_checks = [str(x) for x in (checks.get("required_check_runs", []) or [])]

    fc = ((checks.get("failure_comments", {}) or {}).get("pre-commit")) or None
    precommit_failure_comment = None
    if isinstance(fc, dict) and "title" in fc and "body" in fc:
        precommit_failure_comment = FailureComment(
            title=str(fc["title"]),
            body=str(fc["body"]),
        )

    return Policy(
        branch_patterns=branch_patterns,
        title_patterns=title_patterns,
        title_min_length=title_min_length,
        title_max_length=title_max_length,
        description_min_length=description_min_length,
        description_issue_patterns=description_issue_patterns,
        block_draft=block_draft,
        forbidden_title_patterns=forbidden_title_patterns,
        max_files_changed=max_files_changed,
        max_total_changes=max_total_changes,
        max_single_file_changes=max_single_file_changes,
        forbidden_paths=forbidden_paths,
        unit_test_code_extensions=unit_test_code_extensions,
        unit_test_patterns=unit_test_patterns,
        unit_test_exempt_paths=unit_test_exempt_paths,
        required_checks=required_checks,
        precommit_failure_comment=precommit_failure_comment,
    )


def gh_headers(token: str) -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "therock-pr-bot",
    }


def gh_get(url: str, token: str) -> Any:
    r = requests.get(url, headers=gh_headers(token), timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text}")
    return r.json()


def gh_post(url: str, token: str, payload: Dict[str, Any]) -> Any:
    r = requests.post(url, headers=gh_headers(token), json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {url} -> {r.status_code}: {r.text}")
    return r.json()


def gh_patch(url: str, token: str, payload: Dict[str, Any]) -> Any:
    r = requests.patch(url, headers=gh_headers(token), json=payload, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"PATCH {url} -> {r.status_code}: {r.text}")
    return r.json()


def get_pr(owner: str, repo: str, pr_number: int, token: str) -> Dict[str, Any]:
    data = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}", token
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected PR payload")
    return data


def iter_pr_files(
    owner: str, repo: str, pr_number: int, token: str
) -> Iterable[Dict[str, Any]]:
    page = 1
    while True:
        data = gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files?per_page=100&page={page}",
            token,
        )
        if not isinstance(data, list):
            raise RuntimeError("Unexpected PR files payload")
        if not data:
            return
        for item in data:
            if isinstance(item, dict):
                yield item
        page += 1


def get_check_runs(owner: str, repo: str, sha: str, token: str) -> List[Dict[str, Any]]:
    data = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/check-runs?per_page=100",
        token,
    )
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected check-runs payload")
    runs = data.get("check_runs", [])
    return runs if isinstance(runs, list) else []


def ensure_branch_name(policy: Policy, branch_name: str, errors: List[str]) -> None:
    if not policy.branch_patterns:
        return
    if any(p.match(branch_name) for p in policy.branch_patterns):
        return

    allowed = "\n".join([f"- `{p.pattern}`" for p in policy.branch_patterns])
    errors.append(
        "Branch name does not match allowed patterns.\n"
        f"Branch: `{branch_name}`\n"
        "Allowed patterns:\n"
        f"{allowed}"
    )


def _short(value: str, limit: int = 80) -> str:
    """Truncate a value for display so one long field can't bloat the table."""
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def ensure_pr_title(policy: Policy, title: str, errors: List[str]) -> None:
    title = (title or "").strip()
    fmt = "**Desired format:** `type(optional-scope): short description`"

    if policy.title_min_length and len(title) < policy.title_min_length:
        errors.append(
            f"**Error:** Title is too short ({len(title)} characters).\n"
            f"**Expected:** at least {policy.title_min_length} characters.\n"
            f"{fmt}"
        )

    if policy.title_max_length and len(title) > policy.title_max_length:
        errors.append(
            f"**Error:** Title is too long ({len(title)} characters).\n"
            f"**Expected:** at most {policy.title_max_length} characters.\n"
            f"{fmt}"
        )

    if policy.title_patterns and not any(
        p.search(title) for p in policy.title_patterns
    ):
        errors.append(
            "**Error:** Title does not follow Conventional Commits style.\n"
            "**Expected:** start with a valid type (feat, fix, docs, …).\n"
            f"{fmt}"
        )

    if policy.forbidden_title_patterns:
        matched = [
            p.pattern for p in policy.forbidden_title_patterns if p.search(title)
        ]
        if matched:
            blocked = ", ".join([f"`{m}`" for m in matched])
            errors.append(
                "**Error:** Title contains forbidden text (e.g. WIP / do not merge).\n"
                f"**Expected:** remove the matched term(s): {blocked}.\n"
                f"{fmt}"
            )


def ensure_pr_not_draft(policy: Policy, is_draft: bool, errors: List[str]) -> None:
    if policy.block_draft and is_draft:
        errors.append(
            "This PR is a draft. Please mark it as 'Ready for review' before "
            "it can pass policy checks."
        )


def ensure_pr_description(policy: Policy, body: str, errors: List[str]) -> None:
    body = (body or "").strip()
    if policy.description_min_length and len(body) < policy.description_min_length:
        errors.append(
            f"**Error:** PR description is too short ({len(body)} characters).\n"
            f"**Expected:** at least {policy.description_min_length} characters.\n"
            "**Current:** please provide a meaningful description of your changes"
        )

    if policy.description_issue_patterns and not any(
        p.search(body) for p in policy.description_issue_patterns
    ):
        errors.append(
            "**Error:** PR description must reference a JIRA ID or ISSUE ID.\n"
            "**Expected:** include a `JIRA ID` or `ISSUE ID` line. The separator "
            "may be `:` or `-` (or omitted), and the value can be a JIRA key, a "
            "number (with or without `#`), or a link. Accepted examples:\n"
            "• `JIRA ID : TESTAUTO-6039`\n"
            "• `JIRA ID -` [#330](https://github.com/<org_name>/<repo_name>/issues/330)\n"
            "• `JIRA ID` [#330](https://github.com/<org_name>/<repo_name>/issues/330)\n"
            "• `ISSUE ID : TESTUTO-3334`\n"
            "• `ISSUE ID` [#3334](https://github.com/<org_name>/<repo_name>/issues/3334)\n"
            "• `ISSUE ID - TESTAUTO-3433`\n"
            "• `ISSUE ID : https://github.com/<org_name>/<repo_name>/issues/1234`\n"
            "**Current:** no valid JIRA/ISSUE reference found"
        )


def _matches_forbidden(filename: str, pattern: str) -> bool:
    # GitHub returns POSIX-style paths.
    if fnmatch.fnmatch(filename, pattern):
        return True
    # Allow '**/<x>' patterns to also match root-level files (e.g. '.env').
    if pattern.startswith("**/") and fnmatch.fnmatch(filename, pattern[3:]):
        return True
    return False


def ensure_no_forbidden_files(
    policy: Policy, pr_files: Iterable[Dict[str, Any]], errors: List[str]
) -> None:
    if not policy.forbidden_paths:
        return
    for f in pr_files:
        filename = str(f.get("filename") or "")
        status = str(f.get("status") or "")
        if not filename or status == "removed":
            continue
        norm = Path(filename).as_posix()
        for pattern in policy.forbidden_paths:
            if _matches_forbidden(norm, pattern):
                errors.append(
                    f"Forbidden file present in PR: `{norm}` (matched `{pattern}`)."
                )
                break


def ensure_unit_tests(
    policy: Policy, pr_files: Iterable[Dict[str, Any]], errors: List[str]
) -> None:
    """
    Require a unit test when real source code changes.

    - Doc/config files (anything NOT in code_extensions, e.g. .md/.txt/.yml/.ini)
      never trigger the requirement — a doc/config-only PR passes automatically.
    - If any code file is changed, the PR must also add/modify at least one
      test file (basename matching test_file_patterns, e.g. test_xxx.py).
    """
    if not policy.unit_test_code_extensions:
        return

    code_files: List[str] = []
    has_test = False

    for f in pr_files:
        status = str(f.get("status") or "")
        if status == "removed":
            continue
        filename = Path(str(f.get("filename") or "")).as_posix()
        if not filename:
            continue

        # Files under exempt paths never require an accompanying unit test.
        if any(
            _matches_forbidden(filename, pat) for pat in policy.unit_test_exempt_paths
        ):
            continue

        base = Path(filename).name
        ext = Path(filename).suffix.lower()

        # A test file satisfies the requirement.
        if any(fnmatch.fnmatch(base, pat) for pat in policy.unit_test_patterns):
            has_test = True
            continue

        # A real source/code file triggers the requirement.
        if ext in policy.unit_test_code_extensions:
            code_files.append(filename)

    if code_files and not has_test:
        listed = ", ".join(f"`{c}`" for c in code_files[:5])
        more = "" if len(code_files) <= 5 else f" (+{len(code_files) - 5} more)"
        errors.append(
            "**Error:** Source/code files changed without an accompanying unit test.\n"
            "**Expected:** add at least one test file named like "
            "`test_<name>.py` / `test_<name>.cpp` (or `<name>_test.*`).\n"
            f"**Current:** code file(s) changed: {listed}{more}; no test file found"
        )


def ensure_pr_reviewable(
    policy: Policy, pr_files: List[Dict[str, Any]], errors: List[str]
) -> None:
    """Keep PRs small enough to review: file count, total churn, per-file churn."""
    if not (
        policy.max_files_changed
        or policy.max_total_changes
        or policy.max_single_file_changes
    ):
        return

    num_files = len(pr_files)
    total_changes = 0

    for f in pr_files:
        additions = int(f.get("additions") or 0)
        deletions = int(f.get("deletions") or 0)
        changes = int(f.get("changes") or (additions + deletions))
        total_changes += changes

        filename = Path(str(f.get("filename") or "")).as_posix()
        if policy.max_single_file_changes and changes > policy.max_single_file_changes:
            errors.append(
                "**Error:** A single file changes too much to review easily.\n"
                f"**Expected:** at most {policy.max_single_file_changes} changes "
                "in one file.\n"
                f"**Current:** `{filename}` has {changes} changes"
            )

    if policy.max_files_changed and num_files > policy.max_files_changed:
        errors.append(
            "**Error:** Too many files changed in one PR.\n"
            f"**Expected:** at most {policy.max_files_changed} files.\n"
            f"**Current:** {num_files} files changed"
        )

    if policy.max_total_changes and total_changes > policy.max_total_changes:
        errors.append(
            "**Error:** Total diff is too large to review easily.\n"
            f"**Expected:** at most {policy.max_total_changes} total "
            "additions + deletions.\n"
            f"**Current:** {total_changes} total changes"
        )


def summarize_required_checks(
    policy: Policy,
    check_runs: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns:
      - missing: required checks not present
      - failing: required checks that concluded not-success
      - conc_by_name: name -> conclusion (string; 'null' if none)
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for r in check_runs:
        name = r.get("name")
        if isinstance(name, str):
            by_name[name] = r

    conc_by_name: Dict[str, str] = {}
    for name, r in by_name.items():
        conc = r.get("conclusion")
        conc_by_name[name] = str(conc) if conc is not None else "null"

    missing = [n for n in policy.required_checks if n not in by_name]

    ok = {"success", "neutral", "skipped"}
    failing: List[str] = []
    for n in policy.required_checks:
        r = by_name.get(n)
        if not r:
            continue
        conc = r.get("conclusion")
        if conc is None:
            continue  # still running
        if str(conc) not in ok:
            failing.append(f"{n}={conc}")

    return missing, failing, conc_by_name


def upsert_comment(
    owner: str, repo: str, pr_number: int, token: str, marker: str, body: str
) -> None:
    comments = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100",
        token,
    )
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict) and marker in str(c.get("body", "")):
                gh_patch(c["url"], token, {"body": body})
                return
    gh_post(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments",
        token,
        {"body": body},
    )


def delete_comment_by_marker(
    owner: str, repo: str, pr_number: int, token: str, marker: str
) -> None:
    """Delete any comment whose body contains the given marker."""
    comments = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100",
        token,
    )
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict) and marker in str(c.get("body", "")):
                r = requests.delete(c["url"], headers=gh_headers(token), timeout=30)
                if r.status_code not in (200, 204, 404):
                    print(
                        f"⚠️  Could not delete comment: {r.status_code}: {r.text}",
                        file=sys.stderr,
                    )


def _fetch_cs_alerts_for_ref(
    owner: str, repo: str, ref: str, token: str
) -> Optional[List[Dict[str, Any]]]:
    """Return alerts for a ref, or None on 403 (stop retrying — permission problem)."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/code-scanning/alerts"
        f"?state=open&tool_name=CodeQL&per_page=100&ref={ref}"
    )
    r = requests.get(url, headers=gh_headers(token), timeout=30)
    if r.status_code == 403:
        print(
            f"⚠️  Code Scanning API 403 for ref={ref} — missing "
            "'security-events: read' permission or wrong token.",
            file=sys.stderr,
        )
        return None
    if r.status_code == 404:
        print(f"ℹ️  Code Scanning: 404 for ref={ref} (no analysis stored there yet).")
        return []
    if r.status_code >= 300:
        print(
            f"⚠️  Code Scanning API {r.status_code} for ref={ref}: {r.text}",
            file=sys.stderr,
        )
        return []
    data = r.json()
    alerts = data if isinstance(data, list) else []
    if alerts:
        print(f"ℹ️  Code Scanning: {len(alerts)} alert(s) from ref={ref}.")
        for a in alerts[:3]:  # print first 3 for debug
            rule = a.get("rule", {}) or {}
            print(
                f"    rule.id={rule.get('id')} "
                f"severity={rule.get('severity')} "
                f"security_severity_level={rule.get('security_severity_level')}"
            )
    return alerts


def get_code_scanning_alerts(
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: Optional[str],
    token: str,
    retries: int = 4,
    delay: int = 8,
) -> Optional[List[Dict[str, Any]]]:
    """
    Returns:
      None  — 403 permission error; caller must not override the job conclusion.
      []    — no open error-level alerts (all clear).
      [...] — open alerts found; caller should check severity.
    """
    refs = [f"refs/pull/{pr_number}/merge", f"refs/pull/{pr_number}/head"]
    if head_sha:
        refs.append(str(head_sha))

    for attempt in range(retries):
        for ref in refs:
            alerts = _fetch_cs_alerts_for_ref(owner, repo, ref, token)
            if alerts is None:
                return None  # 403 — permission denied, retrying won't help
            if alerts:
                print(
                    f"ℹ️  Code Scanning: {len(alerts)} open alert(s) found "
                    f"(ref={ref}, attempt {attempt + 1})."
                )
                return alerts
        if attempt < retries - 1:
            print(
                f"ℹ️  Code Scanning: no alerts yet — retrying in {delay}s "
                f"(attempt {attempt + 1}/{retries})."
            )
            time.sleep(delay)

    print(
        "ℹ️  Code Scanning: no open CodeQL alerts found for this PR after all retries."
    )
    return []


def build_policy_table_comment(
    results: List[CheckResult],
    marker: str,
    ready: bool = False,
) -> str:
    # Render rows in a fixed, human-friendly order regardless of the order in
    # which they were appended (policy rows + required-check rows).
    order_index = {name: i for i, name in enumerate(TABLE_ORDER)}
    results = sorted(results, key=lambda r: order_index.get(r.name, len(TABLE_ORDER)))

    all_passed = all(r.passed for r in results)
    if all_passed and ready:
        heading = "### ✅ All Checks Passed — Ready for Review"
    elif all_passed:
        heading = "### ✅ All Policy Checks Passed"
    else:
        heading = "### ❌ PR Check — Action Required"
    rows = []
    for r in results:
        if r.wip:
            status = "🚧 WIP"
        elif r.tbe:
            status = "🔜 To Be Enabled"
        elif r.pending:
            status = "⏳ Pending"
        elif r.passed:
            status = "✅ Pass"
        else:
            status = "❌ Fail"
        if r.passed or r.wip or r.tbe or not r.details:
            detail = "—"
        else:
            blocks: List[str] = []
            for part in r.details:
                lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
                if lines:
                    blocks.append("<br>".join(lines))
            detail = "<br>───<br>".join(blocks)
            # A literal '|' ends a markdown table cell early, which makes the
            # column show only "half" the text. Escape it (and other table-
            # breaking chars) so the FULL details always render in the cell.
            detail = detail.replace("|", "&#124;")
        rows.append(f"| {r.icon} **{r.name}** | {status} | {detail} |")

    table = "| Check | Status | Details |\n" "|---|:---:|---|\n" + "\n".join(rows)
    # WIP and TBE rows are neither pass nor fail — exclude from both counts.
    failing_count = sum(
        1 for r in results if not r.passed and not r.pending and not r.wip and not r.tbe
    )
    if not all_passed:
        failing_names = [
            r.name
            for r in results
            if not r.passed and not r.pending and not r.wip and not r.tbe
        ]
        failing_list = "\n".join(f"> - ❌ {n}" for n in failing_names)
        footer = (
            f"\n\n> ⚠️ **{failing_count} policy check(s) failed.** "
            "Please address the issues above before this PR can be Reviewed.\n>\n"
            "> 🚫 **Please fix the failed policies**\n"
            f"{failing_list}\n>\n"
            f"> The **`{NOT_READY_LABEL}`** label was added to this PR. Once all "
            "policies pass, the label is removed automatically."
        )
    elif ready:
        footer = "\n\n> 🎉 All checks passed! This PR is ready for review."
    else:
        footer = "\n\n> 🎉 All policy checks passed!"

    faq_url = "https://github.com/ROCm/TheRock/tree/main/skills/therock_pr_bot/FAQ.md"

    faq_link = (
        "\n\n📖 **Need help?** See the "
        f"[Policy FAQ]({faq_url}) "
        "for details on every check and how to fix failures."
    )

    return f"{marker}\n{heading}\n\n{table}{footer}{faq_link}"


def build_check_results(
    policy: Policy,
    check_runs: List[Dict[str, Any]],
    code_scanning_alerts: Optional[List[Dict[str, Any]]] = None,
    include_self: bool = False,
) -> List[CheckResult]:
    """Turn required check-runs into table rows (so they appear in one table)."""
    ok = {"success", "neutral", "skipped"}
    by_name = {
        r.get("name"): r
        for r in check_runs
        if isinstance(r, dict) and isinstance(r.get("name"), str)
    }

    # Count error/critical/high-level alerts per language from Code Scanning API.
    lang_alert_counts: Dict[str, int] = {}
    if code_scanning_alerts:
        error_levels = {"error", "critical", "high"}
        print(
            f"ℹ️  build_check_results: processing {len(code_scanning_alerts)} alert(s)."
        )
        for alert in code_scanning_alerts:
            rule = alert.get("rule", {}) or {}
            sev = str(rule.get("severity") or "").lower()
            sec_sev = str(rule.get("security_severity_level") or "").lower()
            rule_id = str(rule.get("id") or "")
            print(f"    → rule_id={rule_id!r} sev={sev!r} sec_sev={sec_sev!r}")
            if sev in error_levels or sec_sev in error_levels:
                if rule_id.startswith("py/"):
                    lang_alert_counts["python"] = lang_alert_counts.get("python", 0) + 1
                elif rule_id.startswith("actions/"):
                    lang_alert_counts["actions"] = (
                        lang_alert_counts.get("actions", 0) + 1
                    )
                elif rule_id.startswith(("js/", "ts/")):
                    lang_alert_counts["javascript"] = (
                        lang_alert_counts.get("javascript", 0) + 1
                    )
                else:
                    # Catch-all so no error-level alert is ever missed.
                    lang_alert_counts["other"] = lang_alert_counts.get("other", 0) + 1
        print(f"ℹ️  lang_alert_counts after processing: {lang_alert_counts}")

    # CodeQL job names that get folded into a single "CodeQL" row.
    codeql_names = {"Analyze (python)", "Analyze (actions)"}

    def status_of(name: str) -> Tuple[bool, bool, Optional[str]]:
        """Return (passed, pending, conclusion) for one check-run."""
        r = by_name.get(name)
        if not r:
            return False, True, None  # not reported yet -> pending
        conc = r.get("conclusion")
        if conc is None:
            return False, True, None  # still running -> pending
        return str(conc) in ok, False, str(conc)

    results: List[CheckResult] = []
    codeql_done = False

    for name in policy.required_checks:
        if name in codeql_names:
            if codeql_done:
                continue
            codeql_done = True

            sub_pending = False
            details: List[str] = []  # type: ignore[no-redef]
            for n in [x for x in policy.required_checks if x in codeql_names]:
                p, pend, conc = status_of(n)
                if pend:
                    sub_pending = True
                elif not p:
                    details.append(f"{n}: conclusion={conc}")

            # A concrete job failure takes precedence over a still-running
            # sibling: show Fail (not Pending) as soon as any job has failed.
            if details:
                passed, pending = False, False
            elif sub_pending:
                passed, pending = False, True
            else:
                passed, pending = True, False

            # Even if jobs succeeded, fail if CodeQL reported error-level alerts.
            if code_scanning_alerts is not None:
                total = sum(lang_alert_counts.values())
                if total > 0:
                    passed, pending = False, False
                    details = [
                        f"**Error:** {total} error-level CodeQL alert(s) found.\n"
                        "**Expected:** no error / critical / high severity alerts.\n"
                        "**Current:** see Security → Code scanning alerts"
                    ]

            if pending and not details:
                details = ["⏳ CodeQL analysis still running…"]
            results.append(CheckResult("CodeQL", "🔎", passed, details, pending))
        else:
            passed, pending, conc = status_of(name)
            if pending:
                details = ["⏳ Still running…"]
            elif passed:
                details = []
            else:
                details = [f"**Error:** Check concluded with `{conc}`."]
            results.append(CheckResult(name, "🔎", passed, details, pending))

    if include_self:
        results.append(CheckResult("therock-pr-bot", "🤖", True, []))
    return results


def maybe_comment_precommit_failure(
    owner: str,
    repo: str,
    pr_number: int,
    token: str,
    policy: Policy,
    check_runs: List[Dict[str, Any]],
) -> None:
    if not policy.precommit_failure_comment:
        return

    precommit_run = None
    for r in check_runs:
        if r.get("name") == "pre-commit":
            precommit_run = r
            break
    if not precommit_run:
        return

    conc = precommit_run.get("conclusion")
    if conc not in ("failure", "cancelled", "timed_out", "action_required"):
        return

    marker = "<!-- therock-pr-bot-precommit-failed -->"
    msg = (
        f"{marker}\n"
        f"### {policy.precommit_failure_comment.title}\n\n"
        f"{policy.precommit_failure_comment.body}"
    )
    upsert_comment(owner, repo, pr_number, token, marker, msg)


LABEL_STYLES = {
    "Not ready to Review": (
        "e11d48",
        "PR has unresolved policy failures — reviews blocked",
    ),
}


def ensure_label_exists(owner: str, repo: str, token: str, label: str) -> None:
    """Create the label in the repo if it does not already exist."""
    color, description = LABEL_STYLES.get(label, ("ededed", ""))
    encoded = urllib.parse.quote(label, safe="")
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/labels/{encoded}",
        headers=gh_headers(token),
        timeout=30,
    )
    if r.status_code == 200:
        return
    try:
        gh_post(
            f"https://api.github.com/repos/{owner}/{repo}/labels",
            token,
            {"name": label, "color": color, "description": description},
        )
    except RuntimeError:
        pass  # already exists (race condition) — safe to ignore


def add_label(owner: str, repo: str, pr_number: int, token: str, label: str) -> None:
    ensure_label_exists(owner, repo, token, label)
    try:
        gh_post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels",
            token,
            {"labels": [label]},
        )
    except RuntimeError as exc:
        print(f"⚠️  Could not add label '{label}': {exc}", file=sys.stderr)


def remove_label(owner: str, repo: str, pr_number: int, token: str, label: str) -> None:
    encoded = urllib.parse.quote(label, safe="")
    r = requests.delete(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels/{encoded}",
        headers=gh_headers(token),
        timeout=30,
    )
    if r.status_code not in (200, 204, 404):
        print(
            f"⚠️  Could not remove label '{label}': {r.status_code}: {r.text}",
            file=sys.stderr,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="TheRock PR Bot policy check (pre-review gate)"
    )
    parser.add_argument(
        "--policy",
        default="skills/therock_pr_bot/policy.yml",
        help="Path to policy.yml",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Max time to wait for required checks",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Polling interval while waiting for checks",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("OWNER")
    repo = os.environ.get("REPO")
    pr_number_s = os.environ.get("PR_NUMBER")
    sha = os.environ.get("SHA")

    missing_env = [
        k
        for k, v in [
            ("GH_TOKEN/GITHUB_TOKEN", token),
            ("OWNER", owner),
            ("REPO", repo),
            ("PR_NUMBER", pr_number_s),
            ("SHA", sha),
        ]
        if not v
    ]
    if missing_env:
        raise RuntimeError(f"Missing required environment: {', '.join(missing_env)}")

    pr_number = int(pr_number_s)  # type: ignore[arg-type]

    repo_root = find_repo_root(Path.cwd())
    policy_path = (repo_root / args.policy).resolve()
    policy = load_policy(policy_path)

    # GITHUB_TOKEN has `security-events: read` from workflow permissions.
    # The App token does not — use GITHUB_TOKEN specifically for Code Scanning.
    cs_token = os.environ.get("GITHUB_TOKEN") or token
    if cs_token == token:
        print(
            "⚠️  GITHUB_TOKEN not set — Code Scanning API will use the App token "
            "(may 403 if the App lacks security-events permission).",
            file=sys.stderr,
        )
    else:
        print("ℹ️  Using GITHUB_TOKEN for Code Scanning API calls.")

    errors: List[str] = []

    pr = get_pr(owner=owner, repo=repo, pr_number=pr_number, token=token)  # type: ignore[arg-type]
    branch_name = str((pr.get("head") or {}).get("ref") or "")
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    pr_files = list(iter_pr_files(owner, repo, pr_number, token))  # type: ignore[arg-type]

    results: List[CheckResult] = []

    _e: List[str] = []
    ensure_branch_name(policy, branch_name, _e)
    results.append(CheckResult("Branch Name", "🌿", not _e, _e))

    _e = []
    ensure_pr_title(policy, title, _e)
    ensure_pr_description(policy, body, _e)
    results.append(CheckResult("PR Title/Description", "📝", not _e, _e))

    # Draft PR check is "Enabled soon" — logic kept in ensure_pr_not_draft but
    # not enforced yet (no check is performed).
    results.append(CheckResult("Draft PR", "🚫", passed=True, details=[], tbe=True))

    _e = []
    ensure_no_forbidden_files(policy, pr_files, _e)
    results.append(CheckResult("Forbidden Files", "⛔", not _e, _e))

    _e = []
    ensure_unit_tests(policy, pr_files, _e)
    results.append(CheckResult("Unit Test", "🧪", not _e, _e))

    # "Enabled soon" placeholders — logic to be implemented later.
    results.append(CheckResult("Feature Flag", "🚩", passed=True, details=[], tbe=True))
    results.append(
        CheckResult("Code Coverage", "📊", passed=True, details=[], tbe=True)
    )

    # Build the policy table; on failure we ALSO append the current
    # pre-commit / CodeQL rows so the table is always complete.
    errors = [d for r in results for d in r.details]
    marker = "<!-- therock-pr-bot-policy-check -->"

    if errors:
        current_runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
        current_alerts = get_code_scanning_alerts(
            owner, repo, pr_number, sha, cs_token, retries=1, delay=0  # type: ignore[arg-type]
        )
        combined = results + build_check_results(
            policy, current_runs, code_scanning_alerts=current_alerts
        )
        upsert_comment(
            owner,
            repo,
            pr_number,
            token,  # type: ignore[arg-type]
            marker,
            build_policy_table_comment(combined, marker),
        )

        # Add "Not ready to Review" ONLY when one of the key policy checks
        # (Branch Name, PR Title/Description, Unit Test, Forbidden Files) fails.
        if any(not r.passed and r.name in LABEL_TRIGGER_CHECKS for r in results):
            add_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
        else:
            remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]

        # Post/update a dedicated "fix policies" comment.
        failing_names = [r.name for r in results if not r.passed]
        fix_marker = "<!-- therock-pr-bot-fix-policies -->"
        fix_body = (
            f"{fix_marker}\n"
            "🚫 **Please fix the failed policies before requesting reviews.**\n\n"
            "The following policy checks failed:\n"
            + "\n".join(f"- ❌ {n}" for n in failing_names)
            + "\n\n"
            f"The **`{NOT_READY_LABEL}`** label has been added to this PR.\n"
            "Once all policies pass, the label will be removed automatically."
        )
        upsert_comment(owner, repo, pr_number, token, fix_marker, fix_body)  # type: ignore[arg-type]

        # --- Poll until pre-commit / CodeQL have a final conclusion so the
        # table updates from ⏳ Pending to a real ✅ Pass or ❌ Fail. ---
        ci_start = time.time()
        ok_set = {"success", "neutral", "skipped"}
        while True:
            poll_runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
            by_name = {
                r.get("name"): r
                for r in poll_runs
                if isinstance(r, dict) and isinstance(r.get("name"), str)
            }
            # Check whether every required CI check has a conclusion yet.
            all_concluded = all(
                by_name.get(n) is not None and by_name[n].get("conclusion") is not None
                for n in policy.required_checks
            )
            if all_concluded:
                poll_alerts = get_code_scanning_alerts(
                    owner, repo, pr_number, sha, cs_token  # type: ignore[arg-type]
                )
                final_combined = results + build_check_results(
                    policy, poll_runs, code_scanning_alerts=poll_alerts
                )
                upsert_comment(
                    owner,
                    repo,
                    pr_number,
                    token,  # type: ignore[arg-type]
                    marker,
                    build_policy_table_comment(final_combined, marker),
                )
                break

            if time.time() - ci_start > args.timeout_seconds:
                print("⚠️  Timed out waiting for CI checks to conclude.")
                break

            time.sleep(args.poll_seconds)

        print("❌ Policy errors:\n")
        for e in errors:
            print(f"- {e}")
        return 1

    # No policy errors — show policy rows now; the required check rows
    # (pre-commit / CodeQL) are appended during the polling loop below.
    upsert_comment(
        owner,
        repo,
        pr_number,
        token,  # type: ignore[arg-type]
        marker,
        build_policy_table_comment(results, marker),
    )

    start = time.time()
    last: Dict[str, str] = {}

    while True:
        runs = get_check_runs(owner=owner, repo=repo, sha=sha, token=token)  # type: ignore[arg-type]
        missing, failing, conc_by_name = summarize_required_checks(policy, runs)
        last = conc_by_name

        if failing:
            final_results = results + build_check_results(policy, runs)
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,
                marker,
                build_policy_table_comment(final_results, marker),
            )
            maybe_comment_precommit_failure(owner, repo, pr_number, token, policy, runs)  # type: ignore[arg-type]
            print("❌ Required checks failing:")
            for f in failing:
                print(f"- {f}")
            return 1

        # If any required checks are missing or still running, keep waiting.
        all_present = not missing
        all_ok = True
        ok = {"success", "neutral", "skipped"}
        by_name = {
            r.get("name"): r
            for r in runs
            if isinstance(r, dict) and isinstance(r.get("name"), str)
        }
        for name in policy.required_checks:
            r = by_name.get(name)
            if not r:
                all_ok = False
                continue
            conc = r.get("conclusion")
            if conc is None:
                all_ok = False
            elif str(conc) not in ok:
                all_ok = False

        if all_present and all_ok:
            # Query Code Scanning API — CodeQL job exits 0 even when it finds
            # vulnerabilities, so we check the alerts API ourselves.
            code_alerts = get_code_scanning_alerts(
                owner, repo, pr_number, sha, cs_token  # type: ignore[arg-type]
            )

            final_results = results + build_check_results(
                policy,
                runs,
                code_scanning_alerts=code_alerts,
                include_self=True,
            )
            has_alert_failures = any(not r.passed for r in final_results)
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,
                marker,
                build_policy_table_comment(
                    final_results, marker, ready=not has_alert_failures
                ),
            )
            if has_alert_failures:
                # CodeQL found vulnerabilities — keep the not-ready label.
                add_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
                print("❌ CodeQL found error-level vulnerabilities.")
                return 1

            # Update the "fix policies" comment to reflect success.
            fix_marker = "<!-- therock-pr-bot-fix-policies -->"
            upsert_comment(
                owner,
                repo,
                pr_number,
                token,  # type: ignore[arg-type]
                fix_marker,
                f"{fix_marker}\n🎉 All checks passed! This PR is ready for review.",
            )

            # Remove the "blocked reviewer/assignee" gate comment if present.
            delete_comment_by_marker(
                owner,
                repo,
                pr_number,
                token,  # type: ignore[arg-type]
                "<!-- therock-pr-bot-review-gate -->",
            )

            # All clean — remove the "Not ready to Review" label.
            remove_label(owner, repo, pr_number, token, NOT_READY_LABEL)  # type: ignore[arg-type]
            print("✅ All required checks passed.")
            return 0

        if time.time() - start > args.timeout_seconds:
            print("❌ Timed out waiting for required checks to complete.")
            print(json.dumps(last, indent=2))
            return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    "Main block"
    raise SystemExit(main())
