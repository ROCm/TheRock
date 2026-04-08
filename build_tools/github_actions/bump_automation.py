#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import subprocess
import tempfile
import os
from datetime import datetime
import requests

THEROCK_REPO = "ROCm/TheRock"

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
    ".github/workflows/therock-ci-linux.yml",
    ".github/workflows/therock-ci-nightly.yml",
    ".github/workflows/therock-ci-windows.yml",
    ".github/workflows/therock-ci.yml",
    ".github/workflows/therock-test-component.yml",
    ".github/workflows/therock-test-packages.yml",
]

SUBMODULE_CONFIG = {
    "rocm-systems": {
        "repo": "ROCm/rocm-systems",
        "files": ROCM_SYSTEMS_FILES,
    },
    "rocm-libraries": {
        "repo": "ROCm/rocm-libraries",
        "files": ROCM_LIBRARIES_FILES,
    },
}


def run(cmd):
    """Run a shell command"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def get_submodule_sha(commit, path):
    """Return SHA of submodule at path in given commit"""
    out = run(["git", "ls-tree", commit, path])
    return out.split()[2]


def submodule_changed(before, after, path):
    diff = run(["git", "diff", before, after, "--", path])
    return bool(diff.strip())


def gh_api(token, endpoint, method="GET", data=None):
    url = f"https://api.github.com/{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.request(method, url, headers=headers, json=data)

    if not response.ok:
        raise RuntimeError(f"GitHub API failed: {response.status_code} {response.text}")

    return response.json()


def latest_commit(repo, token):
    data = gh_api(token, f"repos/{repo}/commits")
    return data[0]["sha"]


def generate_pr_body(repo, base, head):
    compare = f"https://github.com/{repo}/compare/{base}...{head}"
    return f"""
Bumps [{repo}](https://github.com/{repo}) from `{base[:7]}` to `{head[:7]}`.

<details>
<summary>Commits</summary>

See full comparison here:

{compare}

</details>
<br />
"""


def update_ref_in_file(file_path, new_sha):
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
                date = datetime.utcnow().strftime("%Y-%m-%d")
                updated_lines.append(f"{indent}ref: {new_sha} # {date} commit\n")

                # Skip past all lines we've already handled
                i = ref_line_index
        i += 1

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"[INFO] Updated {file_path}")


def close_stale_prs(submodule, old_sha, systems_token):
    """Close all open PRs on TheRock that originated from old submodule SHA"""
    old_short = old_sha[:7]
    prs = gh_api(systems_token, f"repos/{THEROCK_REPO}/pulls?state=open")
    for pr in prs:
        title = pr["title"].lower()
        if f"bump {submodule}" in title and f"from {old_short}" in title:
            number = pr["number"]
            print(f"[INFO] Closing stale PR #{number}")

            # Add a comment to the PR being closed
            gh_api(
                systems_token,
                f"repos/{THEROCK_REPO}/issues/{number}/comments",
                method="POST",
                data={"body": "Closing stale PR."},
            )

            # Close the PR
            gh_api(
                systems_token,
                f"repos/{THEROCK_REPO}/pulls/{number}",
                method="PATCH",
                data={"state": "closed"},
            )


def create_therock_bump(submodule, token):
    """Create a bump PR for the given submodule in TheRock."""
    repo = SUBMODULE_CONFIG[submodule]["repo"]

    original_cwd = os.getcwd()
    # Get latest SHA from upstream submodule repo
    latest = latest_commit(repo, token)

    # Use a temp directory for safe cloning
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "TheRock")
        print(f"[INFO] Cloning TheRock into {clone_dir}")
        clone_url = f"https://x-access-token:{token}@github.com/ROCm/TheRock.git"
        run(["git", "clone", "--depth", "1", clone_url, clone_dir])

        os.chdir(clone_dir)

        branch_name = f"bump-{submodule}-{latest[:7]}"
        run(["git", "checkout", "-b", branch_name])

        # Initialize the submodule if needed
        sub_path = submodule
        if not os.path.exists(os.path.join(sub_path, ".git")):
            run(["git", "submodule", "update", "--init", "--depth", "1", sub_path])
        else:
            print(f"[INFO] Submodule {sub_path} already initialized")

        current_sha = get_submodule_sha("HEAD", sub_path)

        # Fetch latest commit in submodule
        print(f"[INFO] Fetching latest commit for {submodule}")
        run(["git", "-C", sub_path, "fetch", "--depth=1", "origin"])
        run(["git", "-C", sub_path, "checkout", latest])

        # Stage the submodule change
        run(["git", "add", sub_path])

        # Commit and push
        title = f"Bump {submodule} from {current_sha[:7]} to {latest[:7]}"
        body = generate_pr_body(repo, current_sha, latest)
        run(
            [
                "git",
                "-c",
                "user.name=therockbot",
                "-c",
                "user.email=therockbot@amd.com",
                "commit",
                "-m",
                title,
            ]
        )
        run(["git", "push", "origin", branch_name])

        # Create PR
        gh_api(
            token,
            f"repos/{THEROCK_REPO}/pulls",
            method="POST",
            data={"title": title, "head": branch_name, "base": "main", "body": body},
        )
        print(f"[INFO] Created bump PR for {submodule}")
        os.chdir(original_cwd)


def handle_schedule(systems_token, libraries_token):
    """Create bump PRs for both submodules"""
    create_therock_bump("rocm-systems", systems_token)
    create_therock_bump("rocm-libraries", libraries_token)


def handle_push(before, after, systems_token, libraries_token):
    """Push event: update TheRock refs, close stale PRs, create next bump PR"""
    changed = None
    for m in SUBMODULE_CONFIG:
        if submodule_changed(before, after, m):
            changed = m
            break
    if not changed:
        print("[INFO] No monitored submodule changed")
        return

    old_sha = get_submodule_sha(before, changed)
    new_sha = get_submodule_sha(after, changed)

    print(f"[INFO] Detected {changed} change: {old_sha[:7]} → {new_sha[:7]}")

    close_stale_prs(changed, old_sha, systems_token)

    # update workflow YAML
    config = SUBMODULE_CONFIG[changed]
    repo_name = config["repo"]
    token = systems_token if "rocm-systems" in repo_name else libraries_token
    branch = f"update-therock-{changed}-{after[:7]}"
    clone_url = f"https://x-access-token:{token}@github.com/{repo_name}.git"

    with tempfile.TemporaryDirectory() as tmp:
        run(["git", "clone", "--depth", "1", clone_url, tmp])
        os.chdir(tmp)  # Change working directory to the cloned repo

        # Verify that the file exists before accessing
        for f in config["files"]:
            if not os.path.exists(f):
                print(f"[ERROR] File not found: {f}")
                return

        run(["git", "checkout", "-b", branch])

        for f in config["files"]:
            update_ref_in_file(f, after)

        run(["git", "add"] + config["files"])
        run(["git", "commit", "-m", f"Update TheRock ref to {after[:7]}"])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_type", required=True, choices=["schedule", "push"])
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--systems_token", required=True)
    parser.add_argument("--libraries_token", required=True)
    args = parser.parse_args()
    run(
        [
            "git",
            "-c",
            "user.name=therockbot",
            "-c",
            "user.email=therockbot@amd.com",
        ]
    )
    if args.event_type == "schedule":
        handle_schedule(args.systems_token, args.libraries_token)
    elif args.event_type == "push":
        handle_push(
            args.before,
            args.after,
            args.systems_token,
            args.libraries_token,
        )


if __name__ == "__main__":
    main()
