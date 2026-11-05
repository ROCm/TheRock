# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Construct clean cherry-pick commits without a working tree.

A conventional `git cherry-pick` workflow starts from a checked-out branch. A
full checkout materializes every tracked file even when the selected commit
changes only a small number of them.

This module instead works directly with Git's stored objects. It clones the
full commit history without checking out any files. For each source commit,
`git merge-tree` calculates the directory tree that results from applying its
change to the current target head. `git commit-tree` records that tree as a new
commit whose parent is the current target head. The new commit then becomes
the target head for the next pick.

Repeating this process produces the same resulting trees and linear history
shape as clean `git cherry-pick` operations without creating a working tree.
Conflicts and empty picks fail the operation; the module does not attempt to
resolve them or publish a partial result.
"""

import base64
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess


MINIMUM_GIT_VERSION = (2, 41)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CherryPickError(RuntimeError):
    """An expected validation or operational failure."""


class CherryPickConflictError(CherryPickError):
    """A cherry-pick that could not be merged cleanly."""

    def __init__(self, commit: str, paths: list[str]) -> None:
        self.commit = commit
        self.paths = paths
        details = ", ".join(paths) if paths else "unknown paths"
        super().__init__(f"Commit {commit} conflicts in: {details}")


@dataclass(frozen=True)
class CommitIdentity:
    """Name and email used for a generated commit."""

    name: str
    email: str


@dataclass(frozen=True)
class CommitInfo:
    """Metadata read from one source commit."""

    sha: str
    parent: str
    subject: str
    message: str
    author_name: str
    author_email: str
    author_date: str


@dataclass(frozen=True)
class PreparedCommit:
    """A source commit paired with its final generated message."""

    source: CommitInfo
    message: str


@dataclass(frozen=True)
class BuiltCommit:
    """A generated commit and the source commit from which it was built."""

    source: CommitInfo
    sha: str


def _run_git(
    repo: Path | None,
    args: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if repo is not None:
        command.extend(["-C", os.fspath(repo)])
    command.extend(args)
    env = os.environ.copy()
    if extra_env is not None:
        env.update(extra_env)
    try:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except OSError as exc:
        operation = args[0] if args else "command"
        raise CherryPickError(f"Could not execute git {operation}: {exc}") from exc
    if check and result.returncode != 0:
        operation = args[0] if args else "command"
        detail = result.stderr.strip() or result.stdout.strip() or "no details"
        raise CherryPickError(f"git {operation} failed: {detail}")
    return result


def check_git_version() -> tuple[int, int]:
    """Require Git support for `merge-tree` and `GIT_ATTR_SOURCE`."""
    result = _run_git(None, ["--version"])
    match = re.search(r"git version ([0-9]+)\.([0-9]+)", result.stdout)
    if match is None:
        raise CherryPickError(f"Could not parse Git version from {result.stdout!r}")
    version = (int(match.group(1)), int(match.group(2)))
    if version < MINIMUM_GIT_VERSION:
        required = ".".join(str(part) for part in MINIMUM_GIT_VERSION)
        actual = ".".join(str(part) for part in version)
        raise CherryPickError(f"Git {required} or newer is required; found {actual}")
    return version


def validate_branch_name(branch: str, description: str) -> None:
    """Reject a branch name that Git cannot represent."""
    result = _run_git(None, ["check-ref-format", "--branch", branch], check=False)
    if result.returncode != 0:
        raise CherryPickError(f"Invalid {description} branch name {branch!r}")


def clone_repository(clone_url: str, target_path: Path) -> None:
    """Create a full-history, commit-only clone without a working tree."""
    _run_git(
        None,
        [
            "clone",
            "--filter=tree:0",
            "--no-checkout",
            clone_url,
            os.fspath(target_path),
        ],
    )
    if not (target_path / ".git").is_dir():
        raise CherryPickError(f"Target clone was not created at {target_path}")


def _remote_ref(branch: str) -> str:
    return f"refs/remotes/origin/{branch}"


def require_remote_branch(repo: Path, branch: str, description: str) -> str:
    """Require a remote branch and return its remote-tracking ref."""
    ref = _remote_ref(branch)
    result = _run_git(
        repo,
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        check=False,
    )
    if result.returncode != 0:
        raise CherryPickError(
            f"The {description} branch {branch!r} does not exist in the fresh "
            "target clone"
        )
    sha = result.stdout.strip()
    if not FULL_SHA_RE.fullmatch(sha):
        raise CherryPickError(
            f"The {description} branch {branch!r} resolved to {sha!r}"
        )
    return ref


def resolve_commits(repo: Path, commit_specs: Sequence[str]) -> list[str]:
    """Resolve input SHAs uniquely and deduplicate them in first-seen order."""
    resolved: list[str] = []
    seen: set[str] = set()
    for spec in commit_specs:
        result = _run_git(
            repo,
            ["rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}"],
            check=False,
        )
        sha = result.stdout.strip()
        if result.returncode != 0 or not FULL_SHA_RE.fullmatch(sha):
            raise CherryPickError(
                f"Commit {spec!r} does not resolve to exactly one target commit"
            )
        if sha not in seen:
            resolved.append(sha)
            seen.add(sha)
    return resolved


def branch_exists(repo: Path, branch: str) -> bool:
    """Return whether a remote-tracking branch exists."""
    result = _run_git(
        repo,
        ["show-ref", "--verify", "--quiet", _remote_ref(branch)],
        check=False,
    )
    if result.returncode not in (0, 1):
        raise CherryPickError(
            f"Could not check whether target branch {branch!r} exists"
        )
    return result.returncode == 0


def _commit_message(repo: Path, sha: str) -> str:
    result = _run_git(repo, ["cat-file", "commit", sha])
    _, separator, message = result.stdout.partition("\n\n")
    if not separator:
        raise CherryPickError(f"Commit {sha} has no parseable commit message")
    return message


def _commit_metadata(repo: Path, sha: str, parent: str) -> CommitInfo:
    result = _run_git(
        repo,
        ["log", "-1", "--format=%an%x00%ae%x00%aI%x00%s", sha],
    )
    metadata = result.stdout.removesuffix("\n").split("\0")
    if len(metadata) != 4 or not all(metadata[:3]):
        raise CherryPickError(f"Commit {sha} has incomplete author metadata")
    author_name, author_email, author_date, subject = metadata
    return CommitInfo(
        sha=sha,
        parent=parent,
        subject=subject,
        message=_commit_message(repo, sha),
        author_name=author_name,
        author_email=author_email,
        author_date=author_date,
    )


def validate_commits(
    repo: Path,
    resolved_commits: Sequence[str],
    default_branch: str,
    target_branch: str,
    target_description: str,
    previous_pick_message_prefix: str,
) -> list[CommitInfo]:
    """Validate all commits before constructing any cherry-pick commit."""
    default_ref = require_remote_branch(repo, default_branch, "default")
    target_ref = require_remote_branch(repo, target_branch, target_description)
    commits: list[CommitInfo] = []

    for sha in resolved_commits:
        reachability = _run_git(
            repo,
            ["merge-base", "--is-ancestor", sha, default_ref],
            check=False,
        )
        if reachability.returncode == 1:
            raise CherryPickError(
                f"Commit {sha} is not reachable from target default branch "
                f"origin/{default_branch}"
            )
        if reachability.returncode != 0:
            raise CherryPickError(
                f"Could not check whether commit {sha} is on "
                f"origin/{default_branch}"
            )

        parent_result = _run_git(repo, ["rev-list", "--parents", "-n", "1", sha])
        commit_and_parents = parent_result.stdout.split()
        if len(commit_and_parents) != 2:
            parent_count = max(0, len(commit_and_parents) - 1)
            raise CherryPickError(
                f"Commit {sha} has {parent_count} parents; exactly one is required"
            )
        parent = commit_and_parents[1]

        ancestry = _run_git(
            repo,
            ["merge-base", "--is-ancestor", sha, target_ref],
            check=False,
        )
        if ancestry.returncode == 0:
            raise CherryPickError(
                f"Commit {sha} is already an ancestor of {target_description} branch "
                f"{target_branch}"
            )
        if ancestry.returncode != 1:
            raise CherryPickError(
                f"Could not inspect {target_description} ancestry for commit {sha}"
            )

        previous_pick = _run_git(
            repo,
            [
                "log",
                "--format=%H",
                "--fixed-strings",
                f"--grep={previous_pick_message_prefix} {sha}",
                "-n",
                "1",
                target_ref,
            ],
        )
        if previous_pick.stdout.strip():
            raise CherryPickError(
                f"Commit {sha} was already cherry-picked onto {target_branch}"
            )

        commits.append(_commit_metadata(repo, sha, parent))

    return commits


def _conflict_paths(merge_output: str) -> list[str]:
    paths: list[str] = []
    # With --name-only, the tree hash and conflict paths precede a blank line.
    # Everything after that separator is diagnostic text, not a path.
    for line in merge_output.splitlines()[1:]:
        if not line:
            break
        if line not in paths:
            paths.append(line)
    return paths


def build_cherry_pick_commits(
    repo: Path,
    target_branch: str,
    target_description: str,
    commits: Sequence[PreparedCommit],
    committer: CommitIdentity,
) -> list[BuiltCommit]:
    """Construct picks using attributes from each generated target head."""
    head = _run_git(
        repo,
        ["rev-parse", "--verify", f"{_remote_ref(target_branch)}^{{commit}}"],
    ).stdout.strip()
    built: list[BuiltCommit] = []

    for prepared_commit in commits:
        source = prepared_commit.source
        merge = _run_git(
            repo,
            [
                "merge-tree",
                "--write-tree",
                "--name-only",
                f"--merge-base={source.parent}",
                head,
                source.sha,
            ],
            check=False,
            extra_env={"GIT_ATTR_SOURCE": head},
        )
        if merge.returncode == 1:
            raise CherryPickConflictError(source.sha, _conflict_paths(merge.stdout))
        if merge.returncode != 0:
            details = merge.stderr.strip() or "no details"
            raise CherryPickError(
                f"git merge-tree failed for commit {source.sha}: {details}"
            )
        lines = merge.stdout.splitlines()
        tree = lines[0].strip() if lines else ""
        if not FULL_SHA_RE.fullmatch(tree):
            raise CherryPickError(
                f"git merge-tree returned invalid tree {tree!r} for commit "
                f"{source.sha}"
            )

        head_tree = _run_git(
            repo, ["rev-parse", "--verify", f"{head}^{{tree}}"]
        ).stdout.strip()
        if tree == head_tree:
            raise CherryPickError(
                f"Cherry-picking commit {source.sha} onto {target_description} branch "
                f"{target_branch} is empty"
            )

        commit_result = _run_git(
            repo,
            [
                "-c",
                "commit.gpgsign=false",
                "commit-tree",
                tree,
                "-p",
                head,
                "-F",
                "-",
            ],
            input_text=prepared_commit.message,
            extra_env={
                "GIT_AUTHOR_NAME": source.author_name,
                "GIT_AUTHOR_EMAIL": source.author_email,
                "GIT_AUTHOR_DATE": source.author_date,
                "GIT_COMMITTER_NAME": committer.name,
                "GIT_COMMITTER_EMAIL": committer.email,
            },
        )
        head = commit_result.stdout.strip()
        if not FULL_SHA_RE.fullmatch(head):
            raise CherryPickError(
                f"git commit-tree returned invalid commit {head!r} for {source.sha}"
            )
        built.append(BuiltCommit(source=source, sha=head))

    return built


def push_branch(
    repo: Path,
    remote_url: str,
    remote_description: str,
    head: str,
    branch: str,
    *,
    token: str | None = None,
) -> None:
    """Create an absent branch and verify creation without exposing credentials."""
    destination = f"refs/heads/{branch}"
    refspec = f"{head}:{destination}"
    command = [
        "git",
        "-C",
        os.fspath(repo),
        "push",
        "--porcelain",
        f"--force-with-lease={destination}:",
        remote_url,
        refspec,
    ]
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    encoded_credentials: str | None = None
    if token is not None:
        encoded_credentials = base64.b64encode(
            f"x-access-token:{token}".encode("utf-8")
        ).decode("ascii")
        config_index = int(env.get("GIT_CONFIG_COUNT", "0"))
        env["GIT_CONFIG_COUNT"] = str(config_index + 1)
        env[f"GIT_CONFIG_KEY_{config_index}"] = "http.https://github.com/.extraheader"
        env[f"GIT_CONFIG_VALUE_{config_index}"] = (
            f"AUTHORIZATION: basic {encoded_credentials}"
        )
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except OSError as exc:
        raise CherryPickError(
            f"Could not execute git push for {remote_description} branch {branch!r}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no details"
        for secret in (token, encoded_credentials):
            if secret is not None:
                detail = detail.replace(secret, "***")
        raise CherryPickError(
            f"Failed to push {remote_description} branch {branch!r} "
            f"(git exited with status {result.returncode}): {detail}"
        )

    # An existing branch at the same commit yields a successful "up to date"
    # result even with the lease. Require confirmation of a newly created ref.
    statuses: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[1] == refspec:
            statuses.append(fields[0])
    if statuses == ["="]:
        raise CherryPickError(
            f"Branch {branch!r} already exists in {remote_description}"
        )
    if statuses != ["*"]:
        raise CherryPickError(
            f"Could not confirm creation of {remote_description} branch {branch!r}; "
            "inspect the destination before retrying"
        )
