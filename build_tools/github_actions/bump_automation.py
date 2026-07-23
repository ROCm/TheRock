#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import subprocess
import sys
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

# Add build_tools/ to the path so generate_manifest_diff_report is importable
# (this script lives in build_tools/github_actions/; github_actions_api is a
# sibling module in that same directory and needs no path setup).
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

THEROCK_REPO = "ROCm/TheRock"
THEROCK_MAIN_BRANCH = "main"

BOT_NAME = "therockbot"
BOT_EMAIL = "therockbot@amd.com"

CI_LABEL = "ci:run-all-archs"

# Marker for the single sticky "breadcrumb" comment posted on an upstream
# rocm-systems/rocm-libraries PR once its commits land in TheRock via a
# submodule bump. GitHub always displays comments in creation order, even
# after a later edit -- so rather than using a separate marker/comment per
# event (land vs. revert vs. re-land), which would visually sort a later
# revert *above* an earlier inclusion and read backwards, this keeps exactly
# one comment per PR and maintains a newest-first history list inside it
# (see build_breadcrumb_body()).
BREADCRUMB_MARKER = "<!-- therock-bump-breadcrumb -->"
HISTORY_HEADER = "### TheRock submodule-bump history (newest first)"

# Separate marker/comment on the TheRock bump PR itself, for commits that
# could not be resolved to an upstream PR (e.g. pushed directly to the
# superrepo's default branch). Each TheRock bump PR is single-use (a fresh
# `bump-{submodule}-{sha}` branch per bump event), so no cross-event history
# is needed here.
UNMAPPED_MARKER = "<!-- therock-bump-breadcrumb-unmapped -->"

_COMMENT_SEARCH_MAX_PAGES = 10
_COMMENT_SEARCH_PER_PAGE = 100

ROCM_SYSTEMS_FILES = [
    ".github/workflows/therock-ci-linux.yml",
    ".github/workflows/therock-ci-windows.yml",
    ".github/workflows/therock-rccl-ci-linux.yml",
    ".github/workflows/therock-rccl-test-packages-multi-node.yml",
    ".github/workflows/therock-rccl-test-packages-single-node.yml",
    ".github/workflows/therock-test-component.yml",
    ".github/workflows/therock-test-packages.yml",
]

ROCM_LIBRARIES_FILES = [
    ".github/actions/ci-env/action.yml",
]

SUBMODULE_CONFIG = {
    "rocm-systems": {
        "repo": "ROCm/rocm-systems",
        "files": ROCM_SYSTEMS_FILES,
        "updater": "ref",
        "token_key": "systems",
    },
    "rocm-libraries": {
        "repo": "ROCm/rocm-libraries",
        "files": ROCM_LIBRARIES_FILES,
        "updater": "ci-env",
        "token_key": "libraries",
    },
    "debug-tools/rocgdb/source": {
        "repo": "ROCm/rocgdb",
        "files": [],
        "updater": "submodule-only",
        # We will reuse the rocm-systems token for now.
        "token_key": "systems",
        "branch": "amd-staging-rocgdb-16",
    },
}


def _clone_url(repo: str, token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def run(cmd: list[str]) -> str:
    """Run a shell command and return its stdout, raising on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def get_submodule_sha(commit: str, path: str) -> str:
    """Return SHA of submodule at path in given commit."""
    out = run(["git", "ls-tree", commit, path])
    return out.split()[2]


def submodule_changed(before: str, after: str, path: str) -> bool:
    """Return True if the submodule at path differs between two commits."""
    diff = run(["git", "diff", before, after, "--", path])
    return bool(diff.strip())


def detect_changed_submodule(before, after):
    """Finds which monitored submodule (if any) advanced between two commits.

    Returns a dict with keys "name", "repo", "old_sha", "new_sha", or None if
    none of the monitored submodules (see SUBMODULE_CONFIG) changed.
    """
    for name, config in SUBMODULE_CONFIG.items():
        if submodule_changed(before, after, name):
            return {
                "name": name,
                "repo": config["repo"],
                "old_sha": get_submodule_sha(before, name),
                "new_sha": get_submodule_sha(after, name),
            }
    return None


def get_submodule_url(submodule):
    """Reads a submodule's remote URL from .gitmodules in the current checkout."""
    return run(
        [
            "git",
            "config",
            "--file",
            ".gitmodules",
            "--get",
            f"submodule.{submodule}.url",
        ]
    )


def resolve_app_token(name, tokens):
    """Picks whichever App token matches the given submodule's configured token_key."""
    return tokens[SUBMODULE_CONFIG[name]["token_key"]]


def resolve_therock_pr_number(after_sha, github_api):
    """Resolves the TheRock PR whose merge produced `after_sha`."""
    prs = gha_query_prs_for_commit(THEROCK_REPO, after_sha, github_api=github_api)
    if not prs:
        print(f"[WARN] No TheRock PR found for commit {after_sha[:7]}")
        return None
    if len(prs) > 1:
        numbers = ", ".join(str(p["number"]) for p in prs)
        print(
            f"[WARN] Commit {after_sha[:7]} is associated with multiple TheRock "
            f"PRs ({numbers}); using the first"
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
            pr_numbers.update(pr["number"] for pr in prs)
        else:
            unmapped_shas.append(sha)
    return pr_numbers, unmapped_shas


def find_existing_comment_body(pr_number, marker, github_repository, github_api):
    """Reads back the body of the existing sticky comment for `marker`, if any.

    Mirrors the marker-search pagination gha_update_pr_comment()
    (github_actions_api.py) already does internally, so a prior comment's
    history section can be read and extended here rather than being clobbered
    outright by a fresh gha_update_pr_comment() call.
    """
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


def build_timeline_entry(event_date, reverted, therock_pr_number, submodule):
    """Builds a single newest-first history-list line for one bump event."""
    if therock_pr_number is not None:
        ref = (
            f"[{THEROCK_REPO}#{therock_pr_number}]"
            f"(https://github.com/{THEROCK_REPO}/pull/{therock_pr_number})"
        )
    else:
        ref = f"a `{submodule}` submodule bump in {THEROCK_REPO}"
    action = "Reverted out of TheRock via" if reverted else "Included in TheRock via"
    return f"- **{event_date}** — {action} {ref}."


def build_breadcrumb_body(
    existing_body, reverted, therock_pr_number, submodule, event_date
):
    """Builds the single sticky comment body posted on an upstream PR.

    Rather than posting a separate comment per event -- which GitHub always
    displays in creation order, even after a later PATCH edits an earlier
    comment's content, so a commit that lands/reverts/lands-again would
    visually sort backwards -- this keeps exactly one comment per PR and
    maintains a newest-first history list inside it. Prior entries are read
    back from `existing_body` (if any, via find_existing_comment_body) and
    preserved verbatim below the new entry.
    """
    new_entry = build_timeline_entry(event_date, reverted, therock_pr_number, submodule)

    prior_entries = ""
    if existing_body and HISTORY_HEADER in existing_body:
        prior_entries = existing_body.split(HISTORY_HEADER, 1)[1].strip()

    entries = f"{new_entry}\n{prior_entries}" if prior_entries else new_entry

    return f"{BREADCRUMB_MARKER}\n{HISTORY_HEADER}\n\n{entries}\n"


def build_unmapped_summary_body(reverted, submodule, repo, unmapped_shas):
    """Builds the summary comment posted on the TheRock bump PR for commits
    that could not be resolved to an upstream PR (e.g. pushed directly to the
    default branch)."""
    verb = "removed from" if reverted else "included in"
    lines = [
        UNMAPPED_MARKER,
        f"### Unmapped `{submodule}` commits",
        (
            f"The following {len(unmapped_shas)} commit(s) were {verb} this bump "
            f"but have no associated pull request on `{repo}`:"
        ),
    ]
    lines.extend(
        f"- [{sha[:7]}](https://github.com/{repo}/commit/{sha})"
        for sha in unmapped_shas
    )
    return "\n".join(lines) + "\n"


def process_bump(changed, tokens):
    """Posts breadcrumb comments for a single detected submodule bump."""
    name = changed["name"]
    repo = changed["repo"]
    old_sha = changed["old_sha"]
    new_sha = changed["new_sha"]

    print(f"[INFO] Detected {name} change: {old_sha[:7]} -> {new_sha[:7]}")

    app_token = resolve_app_token(name, tokens)
    # The App token generation steps in bump_submodules.yml scope this token
    # to both the changed submodule's repo AND ROCm/TheRock, so a single
    # client can post comments to PRs in either repo below.
    github_api = GitHubAPI(github_token=app_token)

    api_base = get_api_base_from_url(get_submodule_url(name), name)

    therock_pr_number = resolve_therock_pr_number(new_sha, github_api)

    reverted = is_revert(old_sha, new_sha, api_base)
    # On a revert, determine_status()-style logic in
    # generate_manifest_diff_report.py swaps the fetch range so the walked
    # commits are the ones being undone (new_sha -> old_sha), not the ones
    # (re-)introduced.
    range_start, range_end = (new_sha, old_sha) if reverted else (old_sha, new_sha)

    commits = fetch_commits_in_range(
        repo_name=repo, start_sha=range_start, end_sha=range_end, api_base=api_base
    )
    if not commits:
        print(f"[INFO] No commits found in range for {name}; nothing to post.")
        return

    pr_numbers, unmapped_shas = resolve_prs_for_commits(repo, commits, github_api)

    event_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for pr_number in sorted(pr_numbers):
        existing_body = find_existing_comment_body(
            pr_number, BREADCRUMB_MARKER, repo, github_api
        )
        body = build_breadcrumb_body(
            existing_body, reverted, therock_pr_number, name, event_date
        )
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
            gha_update_pr_comment(
                pr_number=therock_pr_number,
                marker=UNMAPPED_MARKER,
                body=build_unmapped_summary_body(reverted, name, repo, unmapped_shas),
                github_repository=THEROCK_REPO,
                github_api=github_api,
            )
            print(
                f"[INFO] Posted unmapped-commit summary ({len(unmapped_shas)} commits) "
                f"to {THEROCK_REPO}#{therock_pr_number}"
            )


def handle_post_breadcrumbs(before, after, tokens):
    """Post-bump-breadcrumbs event: notifies upstream rocm-systems/rocm-libraries
    (and rocgdb) PRs once their commits land in (or get reverted out of)
    TheRock via a submodule bump. Run as its own step in the `push` handler of
    bump_submodules.yml, alongside handle_push(), reusing the same App tokens
    minted there."""
    changed = detect_changed_submodule(before, after)
    if changed is None:
        print(
            "[INFO] No monitored submodule changed between "
            f"{before[:7]} and {after[:7]}; nothing to do."
        )
        return

    process_bump(changed, tokens)


def gh_api(
    token: str, endpoint: str, method: str = "GET", data: dict | None = None
) -> Any:
    """Make a GitHub API request and return the parsed JSON response."""
    url = f"https://api.github.com/{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.request(method, url, headers=headers, json=data)

    if not response.ok:
        raise RuntimeError(f"GitHub API failed: {response.status_code} {response.text}")

    return response.json()


def latest_commit(repo: str, token: str, branch: str | None = None) -> str:
    """Return the SHA of the latest commit on the given branch, or the default branch."""
    url = f"repos/{repo}/commits"
    if branch:
        url += f"?sha={branch}"
    data = gh_api(token, url)
    return data[0]["sha"]


def generate_pr_body(repo: str, base: str, head: str) -> str:
    base_url = f"https://github.com/{repo}/commit/{base}"
    head_url = f"https://github.com/{repo}/commit/{head}"
    compare_url = f"https://github.com/{repo}/compare/{base}...{head}"
    return f"""
Bumps [{repo}](https://github.com/{repo}) from {base_url} to {head_url}.

See full comparison here: {compare_url}
"""


def update_ref_in_file(file_path: str, new_sha: str) -> None:
    """
    Update all ROCm/TheRock refs in a YAML file.
    Replaces existing 'ref:' after 'repository: "ROCm/TheRock"'.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        updated_lines.append(line)

        if line.strip() == 'repository: "ROCm/TheRock"':
            # Determine the indentation level of the 'repository:' line
            repo_indent = len(line) - len(line.lstrip())
            j = i + 1
            ref_line_index = None
            while j < len(lines):
                next_line = lines[j]

                # Skip empty lines
                if next_line.strip() == "":
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent < repo_indent:
                    break

                if next_line.strip().startswith("ref:"):
                    ref_line_index = j
                    break

                j += 1

            if ref_line_index is not None:
                # Copy lines between repository and ref as-is (e.g., path: "TheRock")
                for k in range(i + 1, ref_line_index):
                    updated_lines.append(lines[k])

                # Replace the existing ref line, preserving indentation and removing old comment
                indent = lines[ref_line_index][: lines[ref_line_index].find("ref:")]
                date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                updated_lines.append(f"{indent}ref: {new_sha} # {date} commit\n")

                # Skip past all lines we've already handled
                i = ref_line_index
        i += 1

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"[INFO] Updated {file_path}")


def update_ci_env_file(file_path: str, new_sha: str) -> None:
    """Update the therock-ref value in a ci-env composite action file.

    Matches:
      therock-ref:
        description: ...
        value: "<old_sha>" # <date> commit
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    in_therock_ref = False
    for line in lines:
        stripped = line.strip()
        if stripped == "therock-ref:":
            in_therock_ref = True
            updated_lines.append(line)
            continue

        if in_therock_ref and stripped.startswith("value:"):
            indent = line[: line.find("value:")]
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            updated_lines.append(f'{indent}value: "{new_sha}" # {date} commit\n')
            in_therock_ref = False
            continue

        if in_therock_ref and stripped and not stripped.startswith("description:"):
            in_therock_ref = False

        updated_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"[INFO] Updated {file_path}")


def close_stale_prs(submodule: str, old_sha: str, token: str) -> None:
    """Close all open PRs on TheRock that originated from old submodule SHA."""
    old_short = old_sha[:7]
    prs = gh_api(token, f"repos/{THEROCK_REPO}/pulls?state=open")
    for pr in prs:
        title = pr["title"].lower()
        if f"bump {submodule}" in title and f"from {old_short}" in title:
            number = pr["number"]
            print(f"[INFO] Closing stale PR #{number}")

            # Add a comment to the PR being closed
            gh_api(
                token,
                f"repos/{THEROCK_REPO}/issues/{number}/comments",
                method="POST",
                data={"body": "Closing stale PR."},
            )

            # Close the PR
            gh_api(
                token,
                f"repos/{THEROCK_REPO}/pulls/{number}",
                method="PATCH",
                data={"state": "closed"},
            )


def _git_commit(title: str) -> None:
    """Create a git commit as the bot identity with the given title."""
    run(
        [
            "git",
            "-c",
            f"user.name={BOT_NAME}",
            "-c",
            f"user.email={BOT_EMAIL}",
            "commit",
            "-m",
            title,
        ]
    )


def create_therock_bump(submodule: str, token: str) -> None:
    """Create a bump PR for the given submodule in TheRock."""
    config = SUBMODULE_CONFIG[submodule]
    repo = config["repo"]
    branch = config.get("branch")

    original_cwd = os.getcwd()
    # Get latest SHA from upstream submodule repo
    latest = latest_commit(repo, token, branch)

    # The submodule path may contain slashes (e.g. debug-tools/rocgdb/source);
    # flatten it so the bump branch name is a single ref component.
    branch_name = f"bump-{submodule.replace('/', '-')}-{latest[:7]}"

    # Skip if a PR for this exact commit is already open.
    open_prs = gh_api(
        token,
        f"repos/{THEROCK_REPO}/pulls?state=open&head=ROCm:{branch_name}",
    )
    if open_prs:
        print(
            f"[INFO] Bump PR for {branch_name} already open"
            f" (#{open_prs[0]['number']}), skipping"
        )
        return

    # Use a temp directory for safe cloning
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "TheRock")
        print(f"[INFO] Cloning TheRock into {clone_dir}")
        run(
            ["git", "clone", "--depth", "1", _clone_url(THEROCK_REPO, token), clone_dir]
        )
        os.chdir(clone_dir)

        run(["git", "checkout", "-b", branch_name])

        # Initialize the submodule if needed
        if not os.path.exists(os.path.join(submodule, ".git")):
            run(["git", "submodule", "update", "--init", "--depth", "1", submodule])
        else:
            print(f"[INFO] Submodule {submodule} already initialized")

        current_sha = get_submodule_sha("HEAD", submodule)

        if current_sha == latest:
            print(f"[INFO] {submodule} is already at {latest[:7]}, nothing to bump")
            os.chdir(original_cwd)
            return

        # Fetch the exact target commit in the submodule. A plain depth-1 fetch
        # only retrieves the default branch tip, which misses commits that live
        # on a non-default branch (e.g. rocgdb's amd-staging-rocgdb-16).
        print(f"[INFO] Fetching {latest[:7]} for {submodule}")
        run(["git", "-C", submodule, "fetch", "--depth=1", "origin", latest])
        run(["git", "-C", submodule, "checkout", latest])

        # Stage the submodule change
        run(["git", "add", submodule])

        # Commit and push
        title = f"Bump {submodule} from {current_sha[:7]} to {latest[:7]}"
        body = generate_pr_body(repo, current_sha, latest)
        _git_commit(title)
        run(["git", "push", "origin", branch_name])

        # Create PR
        pr = gh_api(
            token,
            f"repos/{THEROCK_REPO}/pulls",
            method="POST",
            data={
                "title": title,
                "head": branch_name,
                "base": THEROCK_MAIN_BRANCH,
                "body": body,
            },
        )

        try:
            # Add ci:run-all-archs label to the PR
            gh_api(
                token,
                f"repos/{THEROCK_REPO}/issues/{pr['number']}/labels",
                method="POST",
                data={"labels": [CI_LABEL]},
            )
        except RuntimeError as e:
            print(f"[WARN] Failed to apply ci:run-all-archs to PR #{pr['number']}: {e}")
        print(f"[INFO] Created bump PR for {submodule}")
        os.chdir(original_cwd)


def handle_schedule(tokens: dict[str, str], submodule: str = "all") -> None:
    """Create bump PRs for the specified submodule(s)."""
    if submodule in ("all", "rocm-systems"):
        create_therock_bump("rocm-systems", tokens["systems"])
    if submodule in ("all", "rocm-libraries"):
        create_therock_bump("rocm-libraries", tokens["libraries"])
    if submodule in ("all", "rocgdb"):
        create_therock_bump("debug-tools/rocgdb/source", tokens["rocgdb"])


def handle_push(before: str, after: str, tokens: dict[str, str]) -> None:
    """Push event: update TheRock refs, close stale PRs, create next bump PR."""
    changed = None
    for path in SUBMODULE_CONFIG:
        if submodule_changed(before, after, path):
            changed = path
            break
    if not changed:
        print("[INFO] No monitored submodule changed")
        return

    config = SUBMODULE_CONFIG[changed]
    token = tokens[config["token_key"]]
    old_sha = get_submodule_sha(before, changed)

    print(f"[INFO] Detected {changed} change: {old_sha[:7]} -> {after[:7]}")

    close_stale_prs(changed, old_sha, token)

    # submodule-only entries (e.g. rocgdb) have no back-ref files to update in
    # the upstream repo; closing stale bump PRs above is all the push handler
    # needs to do for them.
    if config.get("updater") == "submodule-only":
        print(f"[INFO] {changed} uses submodule-only bumping, skipping ref update")
        return

    # Update workflow YAML
    repo_name = config["repo"]
    branch = f"update-therock-{changed}-{after[:7]}"

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        run(["git", "clone", "--depth", "1", _clone_url(repo_name, token), tmp])
        os.chdir(tmp)  # Change working directory to the cloned repo

        # Verify that the file exists before accessing
        for f in config["files"]:
            if not os.path.exists(f):
                print(f"[ERROR] File not found: {f}")
                os.chdir(original_cwd)
                return

        run(["git", "checkout", "-b", branch])

        updater = (
            update_ci_env_file
            if config.get("updater") == "ci-env"
            else update_ref_in_file
        )
        for f in config["files"]:
            updater(f, after)

        run(["git", "add"] + config["files"])
        _git_commit(f"Update TheRock ref to {after[:7]}")
        run(["git", "push", "origin", branch])
        gh_api(
            token,
            f"repos/{repo_name}/pulls",
            method="POST",
            data={
                "title": f"Update TheRock reference to ({after[:7]})",
                "head": branch,
                "base": "develop",
                "body": f"Updated TheRock ref to `{after[:7]}` due to submodule bump",
            },
        )

    os.chdir(original_cwd)

    # Immediately queue the next bump PR so the cycle continues without
    # waiting for the next scheduled run.
    print(f"[INFO] Creating next bump PR for {changed} after merge")
    try:
        create_therock_bump(changed, token)
    except Exception as e:
        print(f"[WARN] create_therock_bump failed: {e}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_type",
        required=True,
        choices=["schedule", "push", "post_breadcrumbs"],
    )
    parser.add_argument(
        "--submodule",
        default="all",
        choices=["all", "rocm-systems", "rocm-libraries", "rocgdb"],
    )
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--systems_token", required=True)
    parser.add_argument("--libraries_token", required=True)
    parser.add_argument("--rocgdb_token", required=True)
    args = parser.parse_args(argv)

    run(["git", "config", "--global", "user.name", BOT_NAME])
    run(["git", "config", "--global", "user.email", BOT_EMAIL])

    tokens = {
        "systems": args.systems_token,
        "libraries": args.libraries_token,
        "rocgdb": args.rocgdb_token,
    }

    if args.event_type == "schedule":
        handle_schedule(tokens, args.submodule)
    elif args.event_type == "push":
        handle_push(args.before, args.after, tokens)
    elif args.event_type == "post_breadcrumbs":
        handle_post_breadcrumbs(args.before, args.after, tokens)


if __name__ == "__main__":
    main()
