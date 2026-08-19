#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import subprocess
import tempfile
import os
from datetime import datetime, timezone
from typing import Any
import requests

THEROCK_REPO = "ROCm/TheRock"
THEROCK_MAIN_BRANCH = "main"

BOT_NAME = "therockbot"
BOT_EMAIL = "therockbot@amd.com"

COMMON_CI_LABELS = ["ci:run-all-archs"]

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
        # Changes to rocm-systems should run the full matrix of CI jobs:
        #   * Build for all gfx archs
        #   * Build for all variants (asan)
        #   * All builds and tests (including downstream rocm-libraries jobs)
        "labels": [*COMMON_CI_LABELS, "ci:asan"],
    },
    "rocm-libraries": {
        "repo": "ROCm/rocm-libraries",
        "files": ROCM_LIBRARIES_FILES,
        "updater": "ci-env",
        "token_key": "libraries",
        # Changes to rocm-libraries should run the full matrix of CI jobs:
        #   * Build for all gfx archs
        #   * Build for all variants (asan)
        #   * All rocm-libraries tests
        "labels": [*COMMON_CI_LABELS, "ci:asan"],
    },
    "debug-tools/rocgdb/source": {
        "repo": "ROCm/rocgdb",
        "files": [],
        "updater": "submodule-only",
        # We will reuse the rocm-systems token for now.
        "token_key": "systems",
        "branch": "amd-staging-rocgdb-16",
        # Changes to rocgdb can run a limited matrix of CI jobs:
        #   * Build for all gfx archs
        #   * rocgdb tests only (no impact on other project builds/tests)
        "labels": [*COMMON_CI_LABELS, "test:rocgdb"],
    },
    "third-party/sysdeps/linux/amd-mesa/mesa-fork": {
        "repo": "ROCm/mesa-fork",
        "files": [],
        "updater": "submodule-only",
        # We will reuse the rocm-systems token for now.
        "token_key": "systems",
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


def get_baseline_run_id_from_merged_pr(
    repo: str, token: str, merge_commit_sha: str, workflow_name: str = "Multi-Arch CI"
) -> str | None:
    """Get the baseline run ID from the PR that was just merged."""
    # Find the PR that produced this merge commit
    prs = gh_api(token, f"repos/{repo}/commits/{merge_commit_sha}/pulls")
    pr = next(
        (
            p
            for p in prs
            if p.get("merged_at") and p.get("merge_commit_sha") == merge_commit_sha
        ),
        None,
    )
    if not pr:
        print(f"[WARN] No merged PR found for commit {merge_commit_sha[:7]}")
        return None

    # Get the completed workflow run for this PR's head commit
    pr_head_sha = pr["head"]["sha"]
    runs = gh_api(
        token, f"repos/{repo}/actions/runs?head_sha={pr_head_sha}&status=completed"
    )
    for run in runs.get("workflow_runs", []):
        if run["name"] == workflow_name:
            print(
                f"[INFO] Found {workflow_name} run {run['id']} for PR #{pr['number']}"
            )
            return str(run["id"])

    print(f"[WARN] No {workflow_name} run found for PR #{pr['number']}")
    return None


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


def update_ci_env_file(
    file_path: str, new_sha: str, baseline_run_id: str | None = None
) -> None:
    """Update therock-ref and baseline-run-id in a ci-env composite action file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    in_therock_ref = False
    in_baseline_run_id = False

    for line in lines:
        stripped = line.strip()

        # Handle therock-ref updates
        if stripped == "therock-ref:":
            in_therock_ref = True
            updated_lines.append(line)
            continue

        if in_therock_ref and stripped.startswith("value:"):
            indent = line[: line.find("value:")]
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            updated_lines.append(f'{indent}value: "{new_sha}" # {date}\n')
            in_therock_ref = False
            continue

        if in_therock_ref and stripped and not stripped.startswith("description:"):
            in_therock_ref = False

        # Handle baseline-run-id updates
        if stripped == "baseline-run-id:":
            in_baseline_run_id = True
            updated_lines.append(line)
            continue

        if in_baseline_run_id and stripped.startswith("value:"):
            indent = line[: line.find("value:")]
            if baseline_run_id:
                updated_lines.append(
                    f'{indent}value: "{baseline_run_id}" # Updated by assistant-librarian[bot] on submodule bumps\n'
                )
            else:
                # Keep empty if no successful run found
                updated_lines.append(
                    f'{indent}value: "" # Updated by assistant-librarian[bot] on submodule bumps\n'
                )
            in_baseline_run_id = False
            continue

        if in_baseline_run_id and stripped and not stripped.startswith("description:"):
            in_baseline_run_id = False

        updated_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"[INFO] Updated {file_path}")
    if baseline_run_id:
        print(f"[INFO] Set baseline-run-id to {baseline_run_id}")


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
            # Add CI labels to the PR (run-all-archs + asan for full coverage)
            gh_api(
                token,
                f"repos/{THEROCK_REPO}/issues/{pr['number']}/labels",
                method="POST",
                data={"labels": config["labels"]},
            )
        except RuntimeError as e:
            print(f"[WARN] Failed to apply CI labels to PR #{pr['number']}: {e}")
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
    if submodule in ("all", "mesa-fork"):
        create_therock_bump(
            "third-party/sysdeps/linux/amd-mesa/mesa-fork", tokens["mesa-fork"]
        )


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

    # For ci-env updater, get baseline run ID from the merged PR
    baseline_run_id = None
    if config.get("updater") == "ci-env":
        baseline_run_id = get_baseline_run_id_from_merged_pr(
            THEROCK_REPO, token, after, workflow_name="Multi-Arch CI"
        )
        if baseline_run_id:
            print(f"[INFO] Using baseline run ID {baseline_run_id} from merged PR")
        else:
            print(
                f"[WARN] Could not get baseline run ID from merged PR, proceeding without it"
            )

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

        updater = config.get("updater")
        for f in config["files"]:
            if updater == "ci-env":
                update_ci_env_file(f, after, baseline_run_id)
            else:
                update_ref_in_file(f, after)

        run(["git", "add"] + config["files"])

        commit_msg = f"Update TheRock ref to {after[:7]}"
        if baseline_run_id:
            commit_msg += f" (baseline: {baseline_run_id})"
        _git_commit(commit_msg)

        run(["git", "push", "origin", branch])

        pr_body = f"Updated TheRock ref to `{after[:7]}` due to submodule bump"
        if baseline_run_id:
            pr_body += f"\n\nBaseline run ID: [{baseline_run_id}](https://github.com/{THEROCK_REPO}/actions/runs/{baseline_run_id})"

        gh_api(
            token,
            f"repos/{repo_name}/pulls",
            method="POST",
            data={
                "title": f"Update TheRock reference to ({after[:7]})",
                "head": branch,
                "base": "develop",
                "body": pr_body,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_type", required=True, choices=["schedule", "push"])
    parser.add_argument(
        "--submodule",
        default="all",
        choices=["all", "rocm-systems", "rocm-libraries", "rocgdb", "mesa-fork"],
    )
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--systems_token", required=True)
    parser.add_argument("--libraries_token", required=True)
    parser.add_argument("--rocgdb_token", required=True)
    parser.add_argument("--mesa_token", required=True)
    args = parser.parse_args()

    run(["git", "config", "--global", "user.name", BOT_NAME])
    run(["git", "config", "--global", "user.email", BOT_EMAIL])

    tokens = {
        "systems": args.systems_token,
        "libraries": args.libraries_token,
        "rocgdb": args.rocgdb_token,
        "mesa-fork": args.mesa_token,
    }

    if args.event_type == "schedule":
        handle_schedule(tokens, args.submodule)
    elif args.event_type == "push":
        handle_push(args.before, args.after, tokens)


if __name__ == "__main__":
    main()
