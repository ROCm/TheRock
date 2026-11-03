# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Construct clean cherry-pick commits in a conventional working tree.

The target repository is cloned with its complete history and default-branch
working tree. After validation, the release branch is checked out in detached
mode. Each requested commit is applied with `git cherry-pick --no-commit` and
then committed with its prepared release message, original author identity,
and the configured committer identity.

Advancing `HEAD` after every pick preserves the requested commit sequence and
makes attributes introduced by an earlier pick apply to later picks. Conflicts
and empty picks fail the operation; the module does not attempt to resolve them
or publish a partial result.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess


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


def validate_branch_name(branch: str, description: str) -> None:
    """Reject a branch name that Git cannot represent."""
    result = _run_git(None, ["check-ref-format", "--branch", branch], check=False)
    if result.returncode != 0:
        raise CherryPickError(f"Invalid {description} branch name {branch!r}")


def clone_repository(clone_url: str, target_path: Path) -> None:
    """Create a full-history clone with its default branch checked out."""
    _run_git(None, ["clone", clone_url, os.fspath(target_path)])
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


def _unmerged_paths(repo: Path) -> list[str]:
    result = _run_git(
        repo,
        ["diff", "--name-only", "--diff-filter=U", "-z"],
    )
    return [path for path in result.stdout.split("\0") if path]


def _index_has_changes(repo: Path) -> bool:
    result = _run_git(repo, ["diff", "--cached", "--quiet"], check=False)
    if result.returncode not in (0, 1):
        raise CherryPickError("Could not inspect the staged cherry-pick changes")
    return result.returncode == 1


def build_cherry_pick_commits(
    repo: Path,
    target_branch: str,
    target_description: str,
    commits: Sequence[PreparedCommit],
    committer: CommitIdentity,
) -> list[BuiltCommit]:
    """Apply and commit each pick in a checked-out release working tree."""
    _run_git(repo, ["checkout", "--detach", _remote_ref(target_branch)])
    built: list[BuiltCommit] = []

    for prepared_commit in commits:
        source = prepared_commit.source
        cherry_pick = _run_git(
            repo,
            [
                "cherry-pick",
                "--no-commit",
                source.sha,
            ],
            check=False,
        )
        if cherry_pick.returncode != 0:
            paths = _unmerged_paths(repo)
            if paths:
                raise CherryPickConflictError(source.sha, paths)
            details = (
                cherry_pick.stderr.strip() or cherry_pick.stdout.strip() or "no details"
            )
            raise CherryPickError(
                f"git cherry-pick failed for commit {source.sha}: {details}"
            )
        if not _index_has_changes(repo):
            raise CherryPickError(
                f"Cherry-picking commit {source.sha} onto {target_description} branch "
                f"{target_branch} is empty"
            )

        _run_git(
            repo,
            [
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--no-verify",
                "--cleanup=verbatim",
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
        head = _run_git(repo, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip()
        if not FULL_SHA_RE.fullmatch(head):
            raise CherryPickError(
                f"git commit returned invalid commit {head!r} for {source.sha}"
            )
        built.append(BuiltCommit(source=source, sha=head))

    return built


def push_branch(
    repo: Path,
    remote_url: str,
    remote_description: str,
    head: str,
    branch: str,
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
        raise CherryPickError(
            f"Failed to push {remote_description} branch {branch!r} "
            f"(git exited with status {result.returncode})"
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
