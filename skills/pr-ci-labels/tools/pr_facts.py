#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gather the facts the pr-ci-labels skill needs to suggest labels, in one call.

For a PR (or a local branch) it reports: draft state and current CI labels, the changed files
with their auto-skip classification, the touched projects and the test jobs default CI would
select for them, and the CI labels the repository has created. The agent then applies the
selection rules in reference.md without making a dozen separate lookups.

Usage:
    python pr_facts.py --pr <number|url> [--repo owner/name] [--repo-path <clone>] [--json]
    python pr_facts.py --local [--base origin/main] [--repo owner/name] [--json]

Requires: gh (authenticated) for --pr and for the repository's labels.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import ci_labels as cl  # noqa: E402

# This skill lives in TheRock, so TheRock's test-selection tool is next door.
THEROCK_ROOT = THIS_DIR.parents[2]
TEST_SELECTOR = THEROCK_ROOT / "test_tools" / "determine_rocm_test_dependencies.py"
# Top-level directories whose second path component names a project in repositories that
# call TheRock's CI (rocm-libraries, rocm-systems).
PROJECT_ROOTS = ("projects", "shared", "dnn-providers", "emulation")
# Projects registered as nested subtrees, which CI selects separately from their parent.
NESTED_PROJECTS = ("projects/hipblaslt/tensilelite",)
FILES_LIMIT = 3000  # the pulls files API stops listing here
TEXT_FILE_LIMIT = 40


def touched_projects(paths: list[str]) -> list[str]:
    """Project subtrees (for example `projects/hipblaslt`) that the paths fall under."""
    found = set()
    for path in paths:
        nested = next((n for n in NESTED_PROJECTS if path.startswith(n + "/")), None)
        if nested:
            found.add(nested)
            continue
        parts = path.split("/")
        if len(parts) > 2 and parts[0] in PROJECT_ROOTS:
            found.add("/".join(parts[:2]))
    return sorted(found)


def default_test_selection(projects: list[str]) -> tuple[list[str] | None, str]:
    """(test jobs, note): what TheRock's CI would select for these changed projects."""
    if not projects:
        return None, "no project paths changed"
    if not TEST_SELECTOR.is_file():
        return None, f"test selector not found at {TEST_SELECTOR}"
    # The selector applies ci:no-downstream-test from a GitHub Actions event payload; drop it
    # so a CI job's own labels never leak into the answer.
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_EVENT_PATH"}
    proc = subprocess.run(
        [sys.executable, str(TEST_SELECTOR), "--changed-projects", *projects],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        return None, f"test selector failed: {proc.stderr.strip()[-300:]}"
    try:
        return (
            sorted(json.loads(proc.stdout)),
            "from test_tools/determine_rocm_test_dependencies.py",
        )
    except json.JSONDecodeError:
        return None, "test selector output was not JSON"


def presubmit_families() -> dict[str, list[str]] | None:
    """GPU families default PR CI builds and tests, from this TheRock checkout's family matrix.

    Returns {"build_and_test": [...], "build_only": [...]} with entries like "gfx94x (linux)",
    or None if the matrix cannot be loaded.
    """
    sys.path.insert(0, str(THEROCK_ROOT / "build_tools" / "github_actions"))
    try:
        import amdgpu_family_matrix
    except Exception:  # the matrix is TheRock code, not ours; any failure means unknown
        return None
    result: dict[str, list[str]] = {"build_and_test": [], "build_only": []}
    for family, platforms in amdgpu_family_matrix.amdgpu_family_info_matrix.items():
        for platform, info in platforms.items():
            if not isinstance(info, dict):
                continue
            if "presubmit" not in info.get("builds_on_trigger", []):
                continue
            tested = "presubmit" in info.get("tests_on_trigger", [])
            key = "build_and_test" if tested else "build_only"
            result[key].append(f"{family} ({platform})")
    return result


def repo_ci_labels(repo: str) -> list[str] | None:
    """The CI-related labels the repository has created, or None if they can't be read."""
    code, out, _ = cl.run(
        ["gh", "label", "list", "-R", repo, "--limit", "1000", "--json", "name"]
    )
    if code != 0:
        return None
    try:
        names = [item["name"] for item in json.loads(out)]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return sorted(n for n in names if cl.is_ci_label(n))


def pr_changed_files(repo: str, number) -> list[str] | None:
    items = cl.gh_api_pages(f"repos/{repo}/pulls/{number}/files")
    if items is None:
        return None
    return [item.get("filename", "") for item in items]


def local_changed_files(base: str) -> list[str] | None:
    code, out, _ = cl.run(["git", "diff", "--name-only", f"{base}...HEAD"])
    return [line for line in out.splitlines() if line] if code == 0 else None


def build_facts(
    repo: str,
    files: list[str] | None,
    patterns: list[str] | None,
    skip_method: str,
    pr: dict | None = None,
    labels_in_repo: list[str] | None = None,
    families: dict[str, list[str]] | None = None,
) -> dict:
    """Assemble the report from already-fetched inputs (kept separate so it can be tested)."""
    pr = pr or {}
    classified = [
        {"path": f, "skippable": cl.is_ci_skippable(f, patterns)} for f in files or []
    ]
    is_therock = repo.lower() == cl.THEROCK_REPO.lower()
    projects = [] if is_therock else touched_projects(files or [])
    if is_therock:
        tests, tests_note = None, "TheRock PR: CI scope comes from its own path filters"
    else:
        tests, tests_note = default_test_selection(projects)
    return {
        "repo": repo,
        "number": pr.get("number"),
        "url": pr.get("url"),
        "draft": pr.get("isDraft"),
        "base": pr.get("baseRefName"),
        "labels": sorted(
            lbl.get("name", "")
            for lbl in pr.get("labels") or []
            if cl.is_ci_label(lbl.get("name", ""))
        ),
        "files": classified,
        "files_known": files is not None and len(files) < FILES_LIMIT,
        "all_skippable": bool(classified) and all(f["skippable"] for f in classified),
        "skip_method": skip_method,
        "projects": projects,
        "default_tests": tests,
        "default_tests_note": tests_note,
        "pr_ci_families": families,
        "repo_ci_labels": labels_in_repo,
    }


def print_facts(facts: dict) -> None:
    where = (
        f"{facts['repo']}#{facts['number']}" if facts.get("number") else facts["repo"]
    )
    draft = {True: "yes", False: "no"}.get(facts.get("draft"), "unknown")
    print(f"PR {where} (draft: {draft}) base {facts.get('base') or 'n/a'}")
    print(f"CI labels on the PR: {', '.join(facts['labels']) or 'none'}")
    files = facts["files"]
    skippable = sum(f["skippable"] for f in files)
    known = "" if facts["files_known"] else " (incomplete: treat as code)"
    print(
        f"Changed files: {len(files)}, {skippable} CI-skippable{known} "
        f"[{facts['skip_method']}]"
    )
    for f in files[:TEXT_FILE_LIMIT]:
        print(f"  {'skip' if f['skippable'] else 'code'}  {f['path']}")
    if len(files) > TEXT_FILE_LIMIT:
        print(f"  ... {len(files) - TEXT_FILE_LIMIT} more")
    print(
        f"All changed files auto-skipped: {'yes' if facts['all_skippable'] else 'no'}"
    )
    print(f"Touched projects: {', '.join(facts['projects']) or 'none'}")
    tests = facts["default_tests"]
    shown = ", ".join(tests) if tests is not None else "unknown"
    print(f"Default CI tests: {shown} ({facts['default_tests_note']})")
    families = facts.get("pr_ci_families")
    if families is None:
        print("PR CI GPU families: unknown (family matrix not found)")
    else:
        print("PR CI GPU families by default (from this TheRock checkout):")
        print(f"  build and test: {', '.join(families['build_and_test']) or 'none'}")
        print(f"  build only: {', '.join(families['build_only']) or 'none'}")
    labels = facts["repo_ci_labels"]
    if labels is None:
        print("CI labels in the repo: unknown (gh label list failed)")
        return
    print("CI labels in the repo:")
    for name, members in group_repo_labels(labels).items():
        if members:
            print(f"  {name}: {', '.join(members)}")


def group_repo_labels(labels: list[str]) -> dict[str, list[str]]:
    """Group label names so the text report stays short; prefixes are dropped where implied."""
    groups: dict[str, list[str]] = {
        "control": [],
        "ci:gfx...": [],
        "ci:build:...": [],
        "ci:test:...": [],
        "test:...": [],
        "test_filter:...": [],
    }
    for lbl in labels:
        if lbl.startswith("ci:build:"):
            groups["ci:build:..."].append(lbl.removeprefix("ci:build:"))
        elif lbl.startswith("ci:test:"):
            groups["ci:test:..."].append(lbl.removeprefix("ci:test:"))
        elif lbl.startswith("ci:gfx"):
            groups["ci:gfx..."].append(lbl)
        elif lbl.startswith("test_filter:"):
            groups["test_filter:..."].append(lbl.removeprefix("test_filter:"))
        elif lbl.startswith("test:"):
            groups["test:..."].append(lbl.removeprefix("test:"))
        else:
            groups["control"].append(lbl)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Facts for choosing a PR's CI labels (pr-ci-labels skill)."
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pr", help="PR number or URL")
    source.add_argument(
        "--local", action="store_true", help="use the local branch's diff"
    )
    ap.add_argument(
        "--base",
        default="origin/main",
        help="base ref for --local (default: origin/main)",
    )
    ap.add_argument(
        "--repo", help="owner/name (default: from the PR URL, then the checkout)"
    )
    ap.add_argument(
        "--repo-path", help="local clone to read skip-ci configs from (default: GitHub)"
    )
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    pr = None
    if args.pr:
        code, out, err = cl.run(
            [
                "gh",
                "pr",
                "view",
                args.pr,
                "--json",
                "number,url,isDraft,labels,baseRefName",
            ]
            + (["--repo", args.repo] if args.repo else [])
        )
        if code != 0:
            sys.stderr.write(f"ERROR: gh pr view failed: {err.strip()}\n")
            sys.exit(3)
        pr = json.loads(out)
    repo = (
        args.repo
        or cl.repo_from_pr_url((pr or {}).get("url", ""))
        or cl.resolve_repo(None)
    )
    if not repo:
        sys.stderr.write("ERROR: could not determine the repository; pass --repo\n")
        sys.exit(3)
    files = (
        pr_changed_files(repo, pr["number"]) if pr else local_changed_files(args.base)
    )
    patterns, method = cl.load_skip_patterns(
        repo, (pr or {}).get("baseRefName"), args.repo_path
    )
    facts = build_facts(
        repo, files, patterns, method, pr, repo_ci_labels(repo), presubmit_families()
    )
    if args.json:
        print(json.dumps(facts, indent=2))
    else:
        print_facts(facts)


if __name__ == "__main__":
    main()
