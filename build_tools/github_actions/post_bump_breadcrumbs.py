#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Notifies upstream rocm-systems/rocm-libraries/rocgdb PRs once their
commits land in (or get reverted out of) TheRock via a submodule bump.

Posts a single sticky comment per upstream PR (see BREADCRUMB_MARKER) with a
newest-first history list of every land/revert event, and a separate sticky
summary comment on the TheRock bump PR itself (see UNMAPPED_MARKER) for any
commits that couldn't be resolved to an upstream PR -- one section per
submodule, since a single push can bump more than one.

Example usage:
  python build_tools/github_actions/post_bump_breadcrumbs.py \\
      --before <sha> --after <sha> \\
      --systems_token $TOKEN --libraries_token $TOKEN --rocgdb_token $TOKEN

  # Preview without posting any comments:
  python build_tools/github_actions/post_bump_breadcrumbs.py \\
      --before <sha> --after <sha> \\
      --systems_token $TOKEN --libraries_token $TOKEN --rocgdb_token $TOKEN \\
      --dry_run
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_TOOLS_DIR = THIS_SCRIPT_DIR.parent
if str(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_TOOLS_DIR))

from generate_manifest_diff_report import (
    fetch_commits_in_range,
    get_api_base_from_url,
    is_revert,
)
from github_actions_api import (
    GitHubAPI,
    gha_query_prs_for_commit,
    gha_update_pr_comment,
)
from bump_automation import (
    SUBMODULE_CONFIG,
    THEROCK_REPO,
    get_submodule_sha,
    run,
    submodule_changed,
)


# Marker for the single sticky "breadcrumb" comment posted on an upstream
# rocm-systems/rocm-libraries PR once its commits land in TheRock via a
# submodule bump.
BREADCRUMB_MARKER = "<!-- therock-bump-breadcrumb -->"
HISTORY_HEADER = "### TheRock Submodule Bump Activity\n_Newest first_"

# Marker for the (single-use, no history needed) summary comment posted on
# the TheRock bump PR itself for commits with no resolvable upstream PR.
UNMAPPED_MARKER = "<!-- therock-bump-breadcrumb-unmapped -->"
UNMAPPED_HEADER = (
    "### TheRock Submodule Bump Activity\n_No upstream PR found for these commits_"
)

_COMMENT_SEARCH_MAX_PAGES = 10
_COMMENT_SEARCH_PER_PAGE = 100


def detect_changed_submodules(before, after):
    """Finds every monitored submodule that changed between two commits.

    Returns a list of dicts (possibly empty), one per changed submodule, each
    with keys "name", "repo", "old_sha", "new_sha" (see SUBMODULE_CONFIG).
    """
    changed = []
    for name, config in SUBMODULE_CONFIG.items():
        if submodule_changed(before, after, name):
            changed.append(
                {
                    "name": name,
                    "repo": config["repo"],
                    "old_sha": get_submodule_sha(before, name),
                    "new_sha": get_submodule_sha(after, name),
                }
            )
    return changed


def resolve_therock_pr_number(therock_after_sha, github_api):
    """Resolves the TheRock PR whose merge produced `therock_after_sha`."""
    prs = gha_query_prs_for_commit(
        THEROCK_REPO, therock_after_sha, github_api=github_api
    )
    if not prs:
        print(f"[WARN] No TheRock PR found for commit {therock_after_sha[:7]}")
        return None
    if len(prs) > 1:
        numbers = ", ".join(str(p["number"]) for p in prs)
        print(
            f"[WARN] Commit {therock_after_sha[:7]} is associated with multiple "
            f"TheRock PRs ({numbers}); using the first"
        )
    return prs[0]["number"]


def resolve_prs_for_commits(repo, commits, github_api):
    """Resolves each commit to its upstream PR(s) using the given superrepo client.

    Returns (deduped PR numbers, SHAs with no associated PR).
    """
    pr_numbers = set()
    unmapped_shas = []
    for commit in commits:
        sha = commit["sha"]
        prs = gha_query_prs_for_commit(repo, sha, github_api=github_api)
        if prs:
            numbers = [pr["number"] for pr in prs]
            pr_numbers.update(numbers)
            print(f"[INFO]   {sha[:7]} -> PR(s) {', '.join(f'#{n}' for n in numbers)}")
        else:
            unmapped_shas.append(sha)
            print(f"[INFO]   {sha[:7]} -> no associated PR")
    return pr_numbers, unmapped_shas


def find_existing_comment_body(pr_number, marker, github_repository, github_api):
    """Reads back the body of the existing sticky comment for `marker`, if any."""
    comments_url = (
        f"https://api.github.com/repos/{github_repository}/issues/{pr_number}/comments"
    )
    page = 1
    while page <= _COMMENT_SEARCH_MAX_PAGES:
        page_url = f"{comments_url}?per_page={_COMMENT_SEARCH_PER_PAGE}&page={page}"
        comments = github_api.send_request(page_url)
        if not isinstance(comments, list):
            return None
        for comment in comments:
            body = comment.get("body", "")
            if marker in body:
                return body
        if len(comments) < _COMMENT_SEARCH_PER_PAGE:
            return None
        page += 1
    return None


def build_event_key(therock_after_sha, submodule, reverted):
    """Builds a stable identity for one bump event, used to skip re-posting
    on job reruns: two events are the "same" iff they share the triggering
    TheRock push SHA, the submodule, and the land/revert direction."""
    return f"{therock_after_sha}_{submodule}_{reverted}"


def event_key_marker(event_key):
    """HTML-comment marker embedding an event key invisibly in a history line."""
    return f"<!-- event-id: {event_key} -->"


def history_has_event(existing_body, event_key):
    """Whether `existing_body` already contains an entry for `event_key`."""
    return bool(existing_body) and event_key_marker(event_key) in existing_body


def build_timeline_entry(event_date, reverted, therock_pr_number, submodule, event_key):
    """Builds a single newest-first history-list line for one bump event."""
    if therock_pr_number is not None:
        ref = (
            f"[{THEROCK_REPO}#{therock_pr_number}]"
            f"(https://github.com/{THEROCK_REPO}/pull/{therock_pr_number})"
        )
    else:
        ref = f"a `{submodule}` submodule bump in {THEROCK_REPO}"
    action = "Reverted out of TheRock via" if reverted else "Included in TheRock via"
    return f"- **{event_date}** — {action} {ref}. {event_key_marker(event_key)}"


def build_breadcrumb_body(
    existing_body, reverted, therock_pr_number, submodule, event_date, event_key
):
    """Builds the single sticky comment body posted on an upstream PR."""
    new_entry = build_timeline_entry(
        event_date, reverted, therock_pr_number, submodule, event_key
    )

    prior_entries = ""
    if existing_body and HISTORY_HEADER in existing_body:
        prior_entries = existing_body.split(HISTORY_HEADER, 1)[1].strip()

    entries = f"{new_entry}\n{prior_entries}" if prior_entries else new_entry

    return f"{BREADCRUMB_MARKER}\n{HISTORY_HEADER}\n\n{entries}\n"


def build_unmapped_summary_entry(reverted, repo, unmapped_shas, event_key):
    """Builds a single submodule's section of the unmapped-commit summary."""
    verb = "removed from" if reverted else "included in"
    lines = [
        f"The following {len(unmapped_shas)} commit(s) were {verb} this bump "
        f"but have no associated pull request on `{repo}`:"
    ]
    lines.extend(
        f"- [{sha[:7]}](https://github.com/{repo}/commit/{sha})"
        for sha in unmapped_shas
    )
    lines.append(event_key_marker(event_key))
    return "\n".join(lines)


def build_unmapped_summary_body(
    existing_body, reverted, repo, unmapped_shas, event_key
):
    """Builds/updates the sticky unmapped-commit summary comment on the
    TheRock bump PR, with one section per submodule-bump event, keyed by
    `event_key`.
    """
    if history_has_event(existing_body, event_key):
        return existing_body

    new_section = build_unmapped_summary_entry(reverted, repo, unmapped_shas, event_key)

    prior_sections = ""
    if existing_body and UNMAPPED_HEADER in existing_body:
        prior_sections = existing_body.split(UNMAPPED_HEADER, 1)[1].strip()

    sections = f"{new_section}\n\n{prior_sections}" if prior_sections else new_section

    return f"{UNMAPPED_MARKER}\n{UNMAPPED_HEADER}\n\n{sections}\n"


def get_submodule_url(path):
    """Resolves a submodule's URL from .gitmodules by its `path`, since the
    section name can differ from the path (e.g. rocgdb)."""
    path_entries = run(
        [
            "git",
            "config",
            "--file",
            ".gitmodules",
            "--get-regexp",
            r"^submodule\..*\.path$",
        ]
    )
    for line in path_entries.splitlines():
        key, _, value = line.partition(" ")
        if value == path:
            section = key[len("submodule.") : -len(".path")]
            return run(
                [
                    "git",
                    "config",
                    "--file",
                    ".gitmodules",
                    "--get",
                    f"submodule.{section}.url",
                ]
            )
    raise ValueError(f"No .gitmodules entry found with path '{path}'")


def process_bump(changed, therock_after_sha, tokens, dry_run=False):
    """Posts breadcrumb comments for one detected submodule bump.

    `therock_after_sha` is TheRock's own push SHA (for resolving the TheRock
    bump PR) -- distinct from `changed`'s submodule-repo SHAs (for resolving
    the upstream commit range). When `dry_run` is set, prints the comment(s)
    that would be posted instead of calling gha_update_pr_comment().
    """
    name = changed["name"]
    repo = changed["repo"]
    old_sha = changed["old_sha"]
    new_sha = changed["new_sha"]

    print(f"[INFO] Detected {name} change: {old_sha[:7]} -> {new_sha[:7]}")

    app_token = tokens[SUBMODULE_CONFIG[name]["token_key"]]
    github_api = GitHubAPI(github_token=app_token)

    submodule_url = get_submodule_url(name)
    api_base = get_api_base_from_url(submodule_url, name)

    therock_pr_number = resolve_therock_pr_number(therock_after_sha, github_api)

    reverted = is_revert(old_sha, new_sha, api_base, github_api=github_api)
    range_start, range_end = (new_sha, old_sha) if reverted else (old_sha, new_sha)

    commits = fetch_commits_in_range(
        repo_name=repo,
        start_sha=range_start,
        end_sha=range_end,
        api_base=api_base,
        github_api=github_api,
    )
    if not commits:
        print(f"[INFO] No commits found in range for {name}; nothing to post.")
        return

    pr_numbers, unmapped_shas = resolve_prs_for_commits(repo, commits, github_api)

    event_key = build_event_key(therock_after_sha, name, reverted)
    event_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for pr_number in sorted(pr_numbers):
        existing_body = find_existing_comment_body(
            pr_number, BREADCRUMB_MARKER, repo, github_api
        )
        if history_has_event(existing_body, event_key):
            print(
                f"[INFO] {repo}#{pr_number} already has an entry for this event "
                f"({event_key}); skipping (idempotent rerun)."
            )
            continue
        body = build_breadcrumb_body(
            existing_body, reverted, therock_pr_number, name, event_date, event_key
        )
        if dry_run:
            print(f"[DRY RUN] Would post breadcrumb to {repo}#{pr_number}:\n{body}")
            continue
        gha_update_pr_comment(
            pr_number=pr_number,
            marker=BREADCRUMB_MARKER,
            body=body,
            github_repository=repo,
            github_api=github_api,
        )
        print(f"[INFO] Posted breadcrumb to {repo}#{pr_number}")

    if unmapped_shas:
        if therock_pr_number is None:
            print(
                f"[WARN] {len(unmapped_shas)} unmapped commit(s) for {name} but "
                "no TheRock PR to summarize them on"
            )
        else:
            existing_unmapped_body = find_existing_comment_body(
                therock_pr_number, UNMAPPED_MARKER, THEROCK_REPO, github_api
            )
            if history_has_event(existing_unmapped_body, event_key):
                print(
                    f"[INFO] {THEROCK_REPO}#{therock_pr_number} already has an "
                    f"unmapped-commit entry for this event ({event_key}); "
                    "skipping (idempotent rerun)."
                )
            else:
                body = build_unmapped_summary_body(
                    existing_unmapped_body, reverted, repo, unmapped_shas, event_key
                )
                if dry_run:
                    print(
                        f"[DRY RUN] Would post unmapped-commit summary "
                        f"({len(unmapped_shas)} commits) to "
                        f"{THEROCK_REPO}#{therock_pr_number}:\n{body}"
                    )
                else:
                    gha_update_pr_comment(
                        pr_number=therock_pr_number,
                        marker=UNMAPPED_MARKER,
                        body=body,
                        github_repository=THEROCK_REPO,
                        github_api=github_api,
                    )
                    print(
                        f"[INFO] Posted unmapped-commit summary "
                        f"({len(unmapped_shas)} commits) to "
                        f"{THEROCK_REPO}#{therock_pr_number}"
                    )


def handle_post_breadcrumbs(before, after, tokens, dry_run=False):
    """Notifies upstream PRs for every monitored submodule that changed
    between `before` and `after`. Each submodule is attempted
    independently; if any fail, raises once at the end.
    """
    changed_list = detect_changed_submodules(before, after)
    if not changed_list:
        print(
            "[INFO] No monitored submodule changed between "
            f"{before[:7]} and {after[:7]}; nothing to do."
        )
        return

    failures = []
    for changed in changed_list:
        try:
            process_bump(changed, after, tokens, dry_run=dry_run)
        except Exception as e:
            print(f"[WARN] Failed to process bump for {changed['name']}: {e}")
            failures.append(f"{changed['name']}: {e}")

    if failures:
        raise RuntimeError(
            f"Failed to process {len(failures)}/{len(changed_list)} submodule "
            "bump(s): " + "; ".join(failures)
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--systems_token", required=True)
    parser.add_argument("--libraries_token", required=True)
    parser.add_argument("--rocgdb_token", required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the full detection/routing/comment-building pipeline "
        "against live GitHub data, but print the comment(s) that would be "
        "posted instead of actually posting them.",
    )
    args = parser.parse_args(argv)

    tokens = {
        "systems": args.systems_token,
        "libraries": args.libraries_token,
        "rocgdb": args.rocgdb_token,
    }

    handle_post_breadcrumbs(args.before, args.after, tokens, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
