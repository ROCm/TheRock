#!/usr/bin/env python3

"""
Sync Patches to Subrepositories
-------------------------------

This script is part of the monorepo synchronization system. It runs after a monorepo pull request
is merged and applies relevant changes to the corresponding sub-repositories using Git patches.

- Uses the merge commit of the monorepo PR to extract subtree changes.
- Generates patch files per changed subtree.
- Applies each patch to its respective sub-repository, adjusting for subtree prefix.
- Uses the repos-config.json file to map subtrees to sub-repos.
- Assumes this script is run from the root of the monorepo.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --subtrees  : A newline-separated list of subtree paths in category/name format (e.g., projects/rocBLAS)
    --config    : OPTIONAL, path to the repos-config.json file
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Example Usage:
    python pr_merge_sync_patches.py --repo ROCm/rocm-libraries --pr 123 --subtrees "$(printf 'projects/rocBLAS\nprojects/hipBLASLt\nshared/rocSPARSE')" --dry-run --debug
"""

import argparse
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional, List
from pathlib import Path
from github_cli_client import GitHubCLIClient
from config_loader import load_repo_config
from repo_config_model import RepoEntry

logger = logging.getLogger(__name__)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply subtree patches to sub-repositories."
    )
    parser.add_argument(
        "--repo", required=True, help="Full repository name (e.g., org/repo)"
    )
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument(
        "--subtrees",
        required=True,
        help="Newline-separated list of changed subtrees (category/name)",
    )
    parser.add_argument(
        "--config",
        required=False,
        default=".github/repos-config.json",
        help="Path to the repos-config.json file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only logs actions without making changes.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="If set, enables detailed debug logging."
    )
    return parser.parse_args(argv)


def get_subtree_info(config: List[RepoEntry], subtrees: List[str]) -> List[RepoEntry]:
    """Return config entries matching the given subtrees in category/name format."""
    requested = set(subtrees)
    matched = [
        entry for entry in config if f"{entry.category}/{entry.name}" in requested
    ]
    missing = requested - {f"{e.category}/{e.name}" for e in matched}
    if missing:
        logger.warning(
            f"Some subtrees not found in config: {', '.join(sorted(missing))}"
        )
    return matched


def _run_git(args: List[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return stdout."""
    cmd = ["git"] + args
    logger.debug(f"Running git command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def _clone_subrepo(repo_url: str, branch: str, destination: Path) -> None:
    """Clone a specific branch from the given GitHub repository into the destination path."""
    _run_git(
        [
            "clone",
            "--branch",
            branch,
            "--single-branch",
            f"https://github.com/{repo_url}",
            str(destination),
        ]
    )
    logger.debug(f"Cloned {repo_url} into {destination}")


def _configure_git_user(repo_path: Path) -> None:
    """Configure git user.name and user.email for the given repository directory."""
    _run_git(["config", "user.name", "assistant-librarian[bot]"], cwd=repo_path)
    _run_git(
        ["config", "user.email", "assistant-librarian[bot]@users.noreply.github.com"],
        cwd=repo_path,
    )


def _apply_patch(repo_path: Path, patch_path: Path) -> None:
    """Apply a patch file to the working tree."""
    _run_git(["apply", str(patch_path)], cwd=repo_path)
    logger.info(f"Applied patch to working tree at {repo_path}")


def _stage_changes(repo_path: Path) -> None:
    """Stage all changes in the repository."""
    _run_git(["add", "."], cwd=repo_path)
    logger.debug(f"Staged all changes in {repo_path}")


def _extract_commit_message_from_patch(patch_path: Path) -> str:
    """Extract and clean the original commit message from the patch file,
    removing '[PATCH]' and trailing PR references like (#NN) from the title."""
    with open(patch_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    commit_msg_lines = []
    in_msg = False
    for line in lines:
        if line.startswith("Subject: "):
            subject = line[len("Subject: ") :].strip()
            # Remove leading "[PATCH]" if present
            if subject.startswith("[PATCH]"):
                subject = subject[len("[PATCH]") :].strip()
            # Remove trailing PR refs like (#NN)
            subject = re.sub(r"\s*\(#\d+\)$", "", subject)
            commit_msg_lines.append(subject + "\n")
            in_msg = True
        elif in_msg:
            if line.startswith("---"):
                break
            commit_msg_lines.append(line)
    return "".join(commit_msg_lines).strip()


def _format_commit_message(
    monorepo_url: str, pr_number: int, merge_sha: str, original_msg: str
) -> str:
    """Prepend a sync annotation to the original commit message."""
    annotation = (
        f"[rocm-libraries] {monorepo_url}#{pr_number} (commit {merge_sha[:7]})\n\n"
    )
    return annotation + original_msg


def _commit_changes(
    repo_path: Path, message: str, author_name: str, author_email: str
) -> None:
    """Commit staged changes with the specified author and message."""
    _run_git(
        ["commit", "--author", f"{author_name} <{author_email}>", "-m", message],
        cwd=repo_path,
    )
    logger.debug(f"Committed changes with author {author_name} <{author_email}>")


def _set_authenticated_remote(repo_path: Path, repo_url: str) -> None:
    """Set the push URL to use the GitHub App token from GH_TOKEN env."""
    token = os.environ["GH_TOKEN"]
    if not token:
        raise RuntimeError("GH_TOKEN environment variable is not set")
    remote_url = f"https://x-access-token:{token}@github.com/{repo_url}.git"
    _run_git(["remote", "set-url", "origin", remote_url], cwd=repo_path)


def _push_changes(repo_path: Path, branch: str) -> None:
    """Push the commit to origin of branch."""
    _run_git(["push", "origin", branch], cwd=repo_path)
    logger.debug(f"Pushed changes from {repo_path} to origin")


def generate_patch(prefix: str, merge_sha: str, patch_path: Path) -> None:
    """Generate a patch file for a given subtree prefix from a merge commit."""
    args = [
        "format-patch",
        "-1",
        merge_sha,
        f"--relative={prefix}",
        "--output",
        str(patch_path),
    ]
    _run_git(args)
    logger.debug(f"Generated patch for prefix '{prefix}' at {patch_path}")


def resolve_patch_author(
    client: GitHubCLIClient, repo: str, pr: int
) -> tuple[str, str]:
    """Determine the appropriate author for the patch
    Returns: (author_name, author_email)"""
    pr_data = client.get_pr_by_number(repo, pr)
    body = pr_data.get("body", "") or ""
    match = re.search(r"Originally authored by @([A-Za-z0-9_-]+)", body)
    if match:
        username = match.group(1)
        logger.debug(f"Found originally authored username in PR body: @{username}")
    else:
        username = pr_data["user"]["login"]
        logger.debug(f"No explicit original author, using PR author: @{username}")
    name, email = client.get_user(username)
    return name or username, email


def apply_patch_to_subrepo(
    entry: RepoEntry,
    monorepo_url: str,
    monorepo_pr: int,
    patch_path: Path,
    author_name: str,
    author_email: str,
    merge_sha: str,
    dry_run: bool = False,
) -> None:
    """Clone the subrepo, apply the patch, and attribute to the original author with commit message annotations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subrepo_path = Path(tmpdir) / entry.name
        _clone_subrepo(entry.url, entry.branch, subrepo_path)
        if dry_run:
            logger.info(
                f"[Dry-run] Would apply patch to {entry.url} as {author_name} <{author_email}>"
            )
            return
        _configure_git_user(subrepo_path)
        _apply_patch(subrepo_path, patch_path)
        _stage_changes(subrepo_path)
        original_commit_msg = _extract_commit_message_from_patch(patch_path)
        commit_msg = _format_commit_message(
            monorepo_url, monorepo_pr, merge_sha, original_commit_msg
        )
        _commit_changes(subrepo_path, commit_msg, author_name, author_email)
        _set_authenticated_remote(subrepo_path, entry.url)
        _push_changes(subrepo_path, entry.branch)
        logger.info(
            f"Patch applied, committed, and pushed to {entry.url} as {author_name} <{author_email}>"
        )


def main(argv: Optional[List[str]] = None) -> None:
    """Main function to apply patches to sub-repositories."""
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    subtrees = [line.strip() for line in args.subtrees.splitlines() if line.strip()]
    relevant_subtrees = get_subtree_info(config, subtrees)
    merge_sha = client.get_squash_merge_commit(args.repo, args.pr)
    logger.debug(f"Merge commit for PR #{args.pr} in {args.repo}: {merge_sha}")
    for entry in relevant_subtrees:
        prefix = f"{entry.category}/{entry.name}"
        logger.debug(f"Processing subtree {prefix}")
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_file = Path(tmpdir) / f"{entry.name}.patch"
            generate_patch(prefix, merge_sha, patch_file)
            author_name, author_email = resolve_patch_author(client, args.repo, args.pr)
            apply_patch_to_subrepo(
                entry,
                args.repo,
                args.pr,
                patch_file,
                author_name,
                author_email,
                merge_sha,
                args.dry_run,
            )


if __name__ == "__main__":
    main()
