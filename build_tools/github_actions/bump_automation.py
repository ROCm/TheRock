#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Automated submodule bump and upstream ref update for TheRock.

On schedule: each bumper checks for newer versions of its submodule(s)
and creates/updates a rolling PR on TheRock.

On push: when a monitored submodule changes on main, bumpers that have
upstream workflow files update the TheRock ref in those files and
create/update a rolling PR on the upstream repo.

To add a new submodule, create a bumper class with handle_schedule()
and handle_push() methods, then add it to the BUMPERS list.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import requests

THEROCK_REPO = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")

FUSILLI_VERSION_JSON_URL = (
    "https://raw.githubusercontent.com/iree-org/fusilli/main/version.json"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def write_step_summary(line: str) -> None:
    """Append a line to the GitHub Actions job summary."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(line + "\n")


def run(cmd: list[str]) -> str:
    """Run a shell command and return stripped stdout."""
    log(f"++ {shlex.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        log(result.stdout.strip())
    if result.returncode != 0:
        log(f"STDERR: {result.stderr.strip()}")
        raise RuntimeError(f"Command failed (exit code {result.returncode})")
    return result.stdout.strip()


def get_submodule_sha(commit: str, path: str) -> str:
    """Return the SHA a submodule points to in a given commit."""
    out = run(["git", "ls-tree", commit, path])
    return out.split()[2]


def submodule_changed(before: str, after: str, path: str) -> bool:
    """Return True if a submodule changed between two commits."""
    diff = run(["git", "diff", before, after, "--", path])
    return bool(diff.strip())


def gh_api(
    token: str,
    endpoint: str,
    method: str = "GET",
    data: dict | None = None,
) -> dict | list:
    """Call the GitHub REST API and return parsed JSON."""
    url = f"https://api.github.com/{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    response = requests.request(method, url, headers=headers, json=data)
    if not response.ok:
        raise RuntimeError(f"GitHub API failed: {response.status_code} {response.text}")
    return response.json()


def latest_commit(repo: str, token: str) -> str:
    """Return the latest commit SHA on the default branch of a repo."""
    data = gh_api(token, f"repos/{repo}/commits")
    return data[0]["sha"]


def fetch_fusilli_version_json() -> dict[str, str]:
    """Fetch and return version.json from fusilli main branch."""
    resp = requests.get(FUSILLI_VERSION_JSON_URL, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Failed to fetch fusilli version.json: {resp.status_code}")
    data = resp.json()
    if "iree-version" not in data:
        raise ValueError("Missing 'iree-version' in fusilli version.json")
    return data


def create_or_update_pr(
    token: str,
    repo: str,
    branch: str,
    base: str,
    title: str,
    body: str,
) -> None:
    """Create a new PR or update an existing open one on the given branch.

    Uses a static branch name so repeated runs update the same PR
    rather than creating duplicates. Appends PR info to GITHUB_STEP_SUMMARY.
    """
    owner = repo.split("/")[0]
    prs = gh_api(
        token,
        f"repos/{repo}/pulls?head={owner}:{branch}&state=open",
    )
    if prs:
        pr_number = prs[0]["number"]
        pr_url = prs[0]["html_url"]
        log(f"[INFO] Updating existing PR #{pr_number} on {repo} branch {branch}")
        gh_api(
            token,
            f"repos/{repo}/pulls/{pr_number}",
            method="PATCH",
            data={"title": title, "body": body},
        )
        write_step_summary(f"- Updated [{title}]({pr_url})")
    else:
        log(f"[INFO] Creating new PR on {repo} branch {branch}")
        result = gh_api(
            token,
            f"repos/{repo}/pulls",
            method="POST",
            data={"title": title, "head": branch, "base": base, "body": body},
        )
        pr_url = result["html_url"]
        write_step_summary(f"- Created [{title}]({pr_url})")


def update_ref_in_file(file_path: Path, new_sha: str) -> None:
    """Update all ROCm/TheRock refs in an upstream workflow YAML file.

    Finds lines matching 'repository: "ROCm/TheRock"' and replaces the
    subsequent 'ref:' value with new_sha.
    """
    lines = file_path.read_text().splitlines(keepends=True)

    updated_lines: list[str] = []
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

    file_path.write_text("".join(updated_lines))
    log(f"[INFO] Updated {file_path}")


def run_schedule_bump(
    token: str,
    branch: str,
    stage_fn: Callable[[], tuple[str, str] | None],
) -> None:
    """Clone TheRock, run a stage function, and create/update a rolling PR.

    This is the shared infrastructure for all schedule bumps. Each bumper
    provides its own stage_fn that stages git changes and returns
    (title, body) or None if no changes are needed.
    """
    original_cwd = os.getcwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "TheRock"
        log(f"[INFO] Cloning TheRock into {clone_dir}")
        clone_url = f"https://x-access-token:{token}@github.com/{THEROCK_REPO}.git"
        run(["git", "clone", "--depth", "1", clone_url, str(clone_dir)])

        os.chdir(clone_dir)

        result = stage_fn()
        if result is None:
            log(f"[INFO] No changes needed for {branch}")
            os.chdir(original_cwd)
            return

        title, body = result

        run(["git", "checkout", "-b", branch])
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
        run(["git", "push", "--force", "origin", branch])
        create_or_update_pr(token, THEROCK_REPO, branch, "main", title, body)

        log(f"[INFO] Done with {branch}")
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Bumper classes
# ---------------------------------------------------------------------------


class ROCmBumper:
    """Bumps a single submodule to its upstream HEAD.

    On schedule: creates/updates a PR on TheRock bumping the submodule.
    On push: when the submodule changes on TheRock main, updates the
    TheRock ref in the upstream repo's workflow files.
    """

    def __init__(
        self,
        submodule: str,
        upstream_repo: str,
        upstream_files: list[str],
        token_key: str,
    ):
        self.submodule = submodule
        self.upstream_repo = upstream_repo
        self.upstream_files = upstream_files
        self.token_key = token_key
        self.branch_name = f"bump-{submodule}"

    def handle_schedule(self, token: str) -> None:
        run_schedule_bump(
            token,
            self.branch_name,
            lambda: self._stage_changes(token),
        )

    def _stage_changes(self, token: str) -> tuple[str, str] | None:
        """Stage a submodule bump to latest upstream HEAD.

        Returns (title, body) or None if already up-to-date.
        """
        latest = latest_commit(self.upstream_repo, token)

        sub_path = Path(self.submodule)
        if not (sub_path / ".git").exists():
            run(
                ["git", "submodule", "update", "--init", "--depth", "1", self.submodule]
            )

        current_sha = get_submodule_sha("HEAD", self.submodule)

        if current_sha == latest:
            return None

        log(f"[INFO] Bumping {self.submodule}: {current_sha[:7]} -> {latest[:7]}")
        run(["git", "-C", self.submodule, "fetch", "--depth=1", "origin"])
        run(["git", "-C", self.submodule, "checkout", latest])
        run(["git", "add", self.submodule])

        title = f"Bump {self.submodule} from {current_sha[:7]} to {latest[:7]}"
        compare = (
            f"https://github.com/{self.upstream_repo}"
            f"/compare/{current_sha}...{latest}"
        )
        body = f"""\
Bumps [{self.upstream_repo}](https://github.com/{self.upstream_repo}) \
from `{current_sha[:7]}` to `{latest[:7]}`.

**Diff**: {compare}
"""
        return title, body

    def handle_push(self, before: str, after: str, token: str) -> None:
        """Update TheRock refs in upstream workflow files after a bump merges."""
        if not submodule_changed(before, after, self.submodule):
            return

        log(
            f"[INFO] Detected {self.submodule} change, "
            f"updating refs in {self.upstream_repo}"
        )

        original_cwd = os.getcwd()
        branch = f"update-therock-ref-{self.submodule}"
        clone_url = (
            f"https://x-access-token:{token}@github.com" f"/{self.upstream_repo}.git"
        )

        with tempfile.TemporaryDirectory() as tmp:
            run(["git", "clone", "--depth", "1", clone_url, tmp])
            os.chdir(tmp)

            missing = [f for f in self.upstream_files if not Path(f).exists()]
            if missing:
                log(f"[ERROR] Files not found in {self.upstream_repo}: {missing}")
                os.chdir(original_cwd)
                return

            run(["git", "checkout", "-b", branch])

            for f in self.upstream_files:
                update_ref_in_file(Path(f), after)

            run(["git", "add"] + self.upstream_files)
            run(["git", "commit", "-m", f"Update TheRock ref to {after[:7]}"])
            run(["git", "push", "--force", "origin", branch])
            create_or_update_pr(
                token=token,
                repo=self.upstream_repo,
                branch=branch,
                base="develop",
                title=f"Update TheRock reference to ({after[:7]})",
                body=f"Updated TheRock ref to `{after[:7]}` due to submodule bump",
            )

            os.chdir(original_cwd)


class IreeLibsBumper:
    """Bumps iree-libs/iree and iree-libs/fusilli submodules together.

    IREE is pinned to the version tag declared in fusilli's version.json
    (ensuring TheRock uses a validated IREE version). Fusilli is bumped
    to HEAD of main.
    """

    def __init__(self, token_key: str):
        self.token_key = token_key
        self.branch_name = "bump-iree-libs"

    def handle_schedule(self, token: str) -> None:
        run_schedule_bump(token, self.branch_name, self._stage_changes)

    def _stage_changes(self) -> tuple[str, str] | None:
        """Stage IREE and fusilli submodule bumps.

        Returns (title, body) or None if already up-to-date.
        """
        for sub in ["iree-libs/iree", "iree-libs/fusilli"]:
            if not (Path(sub) / ".git").exists():
                run(["git", "submodule", "update", "--init", "--depth", "1", sub])

        current_iree = get_submodule_sha("HEAD", "iree-libs/iree")
        current_fusilli = get_submodule_sha("HEAD", "iree-libs/fusilli")

        # Get target IREE version from fusilli's version.json
        version_data = fetch_fusilli_version_json()
        iree_version = version_data["iree-version"]
        tag = f"iree-{iree_version}"

        # Resolve IREE tag to commit SHA using git peel syntax.
        # tag^{commit} dereferences the tag to the underlying commit,
        # handling both lightweight tags (no-op) and annotated tags
        # (unwraps the tag object).
        run(["git", "-C", "iree-libs/iree", "fetch", "--depth=1", "origin", "tag", tag])
        latest_iree = run(
            ["git", "-C", "iree-libs/iree", "rev-parse", f"{tag}^{{commit}}"]
        )

        # Get fusilli HEAD
        run(["git", "-C", "iree-libs/fusilli", "fetch", "--depth=1", "origin", "main"])
        latest_fusilli = run(
            ["git", "-C", "iree-libs/fusilli", "rev-parse", "origin/main"]
        )

        if current_iree == latest_iree and current_fusilli == latest_fusilli:
            return None

        if current_iree != latest_iree:
            run(["git", "-C", "iree-libs/iree", "checkout", latest_iree])
            run(["git", "add", "iree-libs/iree"])

        if current_fusilli != latest_fusilli:
            run(["git", "-C", "iree-libs/fusilli", "checkout", latest_fusilli])
            run(["git", "add", "iree-libs/fusilli"])

        title = (
            f"Bump IREE libs: IREE to iree-{iree_version},"
            f" fusilli to {latest_fusilli[:7]}"
        )
        body = f"""\
## Summary
Automated bump of IREE libs submodules.

IREE version is pinned to the tag declared in
[fusilli's `version.json`]\
(https://github.com/iree-org/fusilli/blob/main/version.json).

| Submodule | Old | New |
|-----------|-----|-----|
| iree-libs/iree | `{current_iree}` | `{latest_iree}` (tag `{tag}`) |
| iree-libs/fusilli | `{current_fusilli}` | `{latest_fusilli}` |

**IREE diff**: \
https://github.com/iree-org/iree/compare/{current_iree}...{latest_iree}
**Fusilli diff**: \
https://github.com/iree-org/fusilli/compare/{current_fusilli}...{latest_fusilli}
"""
        return title, body

    def handle_push(self, before: str, after: str, token: str) -> None:
        """No-op: IREE/fusilli repos don't reference TheRock in workflows."""


# ---------------------------------------------------------------------------
# Bumper registry — add new submodule bumpers here.
# ---------------------------------------------------------------------------

BUMPERS = [
    ROCmBumper(
        submodule="rocm-systems",
        upstream_repo="ROCm/rocm-systems",
        upstream_files=[
            ".github/workflows/therock-ci-linux.yml",
            ".github/workflows/therock-ci-windows.yml",
            ".github/workflows/therock-rccl-ci-linux.yml",
            ".github/workflows/therock-rccl-test-packages-multi-node.yml",
            ".github/workflows/therock-test-component.yml",
            ".github/workflows/therock-test-packages.yml",
        ],
        token_key="systems_token",
    ),
    ROCmBumper(
        submodule="rocm-libraries",
        upstream_repo="ROCm/rocm-libraries",
        upstream_files=[
            ".github/workflows/therock-ci-linux.yml",
            ".github/workflows/therock-ci-nightly.yml",
            ".github/workflows/therock-ci-windows.yml",
            ".github/workflows/therock-ci.yml",
            ".github/workflows/therock-test-component.yml",
            ".github/workflows/therock-test-packages.yml",
        ],
        token_key="libraries_token",
    ),
    IreeLibsBumper(token_key="systems_token"),
]


# ---------------------------------------------------------------------------
# Event dispatch
# ---------------------------------------------------------------------------


def handle_schedule(tokens: dict[str, str]) -> None:
    """Run schedule bumps for all registered bumpers."""
    for bumper in BUMPERS:
        bumper.handle_schedule(tokens[bumper.token_key])


def handle_push(before: str, after: str, tokens: dict[str, str]) -> None:
    """Run push handlers for all registered bumpers."""
    for bumper in BUMPERS:
        bumper.handle_push(before, after, tokens[bumper.token_key])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Automated submodule bump automation for TheRock."
    )
    parser.add_argument("--event_type", required=True, choices=["schedule", "push"])
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--systems_token", required=True)
    parser.add_argument("--libraries_token", required=True)
    args = parser.parse_args(argv)

    tokens = {
        "systems_token": args.systems_token,
        "libraries_token": args.libraries_token,
    }

    if args.event_type == "schedule":
        handle_schedule(tokens)
    elif args.event_type == "push":
        handle_push(args.before, args.after, tokens)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
