# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Construct clean cherry-pick commits without a working tree.

This module is an importable Git library and does not provide a command-line
entry point. The release workflow coordinator in
`build_tools/github_actions/cherry_pick_to_release.py` uses it without coupling
the Git operations to GitHub Actions policy.

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
import shlex
import subprocess
import sys


class CherryPickError(RuntimeError):
    """An expected validation or operational failure."""


class CherryPickConflictError(CherryPickError):
    """A cherry-pick that could not be merged cleanly."""

    def __init__(self, commit: str, diagnostics: str) -> None:
        self.commit = commit
        self.diagnostics = diagnostics
        super().__init__(f"Commit {commit} did not apply cleanly")


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
    capture_stderr: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if repo is not None:
        command.extend(["-C", os.fspath(repo)])
    command.extend(args)
    env = os.environ.copy()
    if extra_env is not None:
        env.update(extra_env)
    print(f"+ {shlex.join(command)}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(
            command,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else None,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except OSError as exc:
        raise CherryPickError(
            f"Could not execute {shlex.join(command)}: {exc}"
        ) from exc
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="", flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="", flush=True)
        raise CherryPickError(
            f"{shlex.join(command)} failed with status {result.returncode}"
        )
    return result


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


def _remote_ref(branch: str) -> str:
    return f"refs/remotes/origin/{branch}"


def resolve_commits(repo: Path, commit_specs: Sequence[str]) -> list[str]:
    """Resolve input SHAs in order, preserving repeated commits."""
    resolved: list[str] = []
    for spec in commit_specs:
        result = _run_git(
            repo,
            ["rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}"],
            check=False,
        )
        sha = result.stdout.strip()
        if result.returncode != 0:
            raise CherryPickError(
                f"Commit {spec!r} does not resolve to exactly one target commit"
            )
        resolved.append(sha)
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
    return result.stdout.partition("\n\n")[2]


def _commit_metadata(repo: Path, sha: str, parent: str) -> CommitInfo:
    result = _run_git(
        repo,
        ["log", "-1", "--format=%an%x00%ae%x00%aI%x00%s", sha],
    )
    author_name, author_email, author_date, subject = result.stdout.removesuffix(
        "\n"
    ).split("\0", maxsplit=3)
    return CommitInfo(
        sha=sha,
        parent=parent,
        subject=subject,
        message=_commit_message(repo, sha),
        author_name=author_name,
        author_email=author_email,
        author_date=author_date,
    )


def inspect_commits(
    repo: Path,
    resolved_commits: Sequence[str],
    default_branch: str,
) -> list[CommitInfo]:
    """Inspect source commits and enforce source-selection policy."""
    default_ref = _remote_ref(default_branch)
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
        commits.append(_commit_metadata(repo, sha, parent))

    return commits


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
                f"--attr-source={head}",
                "merge-tree",
                "--write-tree",
                "--name-only",
                f"--merge-base={source.parent}",
                head,
                source.sha,
            ],
            check=False,
        )
        if merge.returncode == 1:
            # The first line is the merge result tree OID, not a commit.
            # Strip it because the tree exists only in the temporary clone.
            _, _, diagnostics = merge.stdout.partition("\n")
            raise CherryPickConflictError(source.sha, diagnostics.lstrip("\n"))
        if merge.returncode != 0:
            raise CherryPickError(
                f"git merge-tree failed for commit {source.sha} with status "
                f"{merge.returncode}"
            )
        tree = merge.stdout.splitlines()[0].strip()

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
    args = [
        "push",
        "--porcelain",
        f"--force-with-lease={destination}:",
        remote_url,
        refspec,
    ]
    extra_env = {"GIT_TERMINAL_PROMPT": "0"}
    encoded_credentials: str | None = None
    if token is not None:
        encoded_credentials = base64.b64encode(
            f"x-access-token:{token}".encode("utf-8")
        ).decode("ascii")
        config_index = int(os.getenv("GIT_CONFIG_COUNT", "0"))
        extra_env["GIT_CONFIG_COUNT"] = str(config_index + 1)
        extra_env[f"GIT_CONFIG_KEY_{config_index}"] = (
            "http.https://github.com/.extraheader"
        )
        extra_env[f"GIT_CONFIG_VALUE_{config_index}"] = (
            f"AUTHORIZATION: basic {encoded_credentials}"
        )
    result = _run_git(
        repo,
        args,
        check=False,
        capture_stderr=True,
        extra_env=extra_env,
    )
    redacted_outputs: list[str] = []
    for output in (result.stdout, result.stderr):
        detail = output or ""
        for secret in (token, encoded_credentials):
            if secret is not None:
                detail = detail.replace(secret, "***")
        redacted_outputs.append(detail)
    stdout_detail, stderr_detail = redacted_outputs
    if result.returncode != 0:
        if stdout_detail:
            print(stdout_detail, file=sys.stderr, end="", flush=True)
        if stderr_detail:
            print(stderr_detail, file=sys.stderr, end="", flush=True)
        raise CherryPickError(
            f"Failed to push {remote_description} branch {branch!r} "
            f"(git exited with status {result.returncode})"
        )
    if stderr_detail:
        print(stderr_detail, file=sys.stderr, end="", flush=True)

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
