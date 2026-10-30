# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cherry-pick default-branch commits into a ROCm release branch.

A conventional `git cherry-pick` workflow starts from a checked-out release
branch. A full checkout materializes every tracked file even when the selected
commit changes only a small number of them.

This tool instead works directly with Git's stored objects. It clones the full
commit history without checking out any files. For each source commit,
`git merge-tree` calculates the directory tree that results from applying its
change to the current release head. `git commit-tree` records that tree as a
new commit whose parent is the current release head. The new commit then
becomes the release head for the next pick.

Repeating this process produces the same resulting trees and linear history
shape as clean `git cherry-pick` operations without creating a working tree.
Conflicts and empty picks fail the operation; the tool does not attempt to
resolve them or publish a partial result.
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Protocol
from urllib.parse import quote, urlsplit

from github_actions_api import (
    gha_append_step_summary,
    GitHubAPI,
    GitHubAPIError,
)


TARGET_OWNER = "ROCm"
HOST_REPOSITORY = "ROCm/TheRock"
MINIMUM_GIT_VERSION = (2, 41)
MINIMUM_SHORT_SHA_LENGTH = 7
FULL_SHA_LENGTH = 40
SUPPORTED_REPOSITORIES = {
    "TheRock": "systems-assistant",
    "rocm-systems": "systems-assistant",
    "rocm-libraries": "assistant-librarian",
}

COMMIT_RE = re.compile(
    rf"^[0-9a-fA-F]{{{MINIMUM_SHORT_SHA_LENGTH},{FULL_SHA_LENGTH}}}$"
)
JIRA_RE = re.compile(r"^[A-Z][A-Z0-9_]*-[1-9][0-9]*$")
REGULAR_RELEASE_RE = re.compile(r"^release/therock-([0-9]+\.[0-9]+)$")
BKC_RELEASE_RE = re.compile(r"^release/bkc/therock-([0-9]+\.[0-9]+)-([0-9]{8})$")
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


class GitHubClient(Protocol):
    """The subset of :class:`GitHubAPI` used by this script."""

    def send_request(
        self,
        url: str,
        timeout_seconds: int = 300,
        *,
        method: str = "GET",
        body: object | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class ReleaseInfo:
    branch: str
    label: str
    version: str
    bkc_date: str | None

    @property
    def is_bkc(self) -> bool:
        return self.bkc_date is not None


@dataclass(frozen=True)
class BotIdentity:
    name: str
    email: str


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    parent: str
    subject: str
    message: str
    author_name: str
    author_email: str
    author_date: str


@dataclass(frozen=True)
class BuiltCommit:
    source: CommitInfo
    sha: str


@dataclass(frozen=True)
class RunResult:
    repository: str
    release_branch: str
    pr_branch: str
    built_commits: list[BuiltCommit]
    pr_title: str
    pr_body: str
    pr_url: str | None
    dry_run: bool


def parse_repository(value: str) -> str:
    """Validate and return a supported target repository name."""
    if value not in SUPPORTED_REPOSITORIES:
        choices = ", ".join(sorted(SUPPORTED_REPOSITORIES))
        raise CherryPickError(
            f"Unsupported repository {value!r}; expected one of: {choices}"
        )
    return value


def parse_jira_id(value: str) -> str | None:
    """Parse an optional JIRA issue key."""
    jira_id = value.strip()
    if not jira_id:
        return None
    if not JIRA_RE.fullmatch(jira_id):
        raise CherryPickError(
            f"Invalid JIRA ID {jira_id!r}; expected a key such as AIRDEL-1234"
        )
    return jira_id


def parse_commit_list(value: str) -> list[str]:
    """Parse comma or whitespace separated abbreviated or full commit SHAs."""
    commits = [item for item in re.split(r"[,\s]+", value.strip()) if item]
    if not commits:
        raise CherryPickError("At least one commit SHA is required")
    for commit in commits:
        if not COMMIT_RE.fullmatch(commit):
            raise CherryPickError(
                f"Invalid commit {commit!r}; expected 7 to 40 hexadecimal characters"
            )
    return commits


def parse_release_branch(branch: str) -> ReleaseInfo:
    """Parse a supported regular or BKC release branch."""
    regular_match = REGULAR_RELEASE_RE.fullmatch(branch)
    if regular_match:
        version = regular_match.group(1)
        return ReleaseInfo(
            branch=branch,
            label=version,
            version=version,
            bkc_date=None,
        )

    bkc_match = BKC_RELEASE_RE.fullmatch(branch)
    if bkc_match:
        version, bkc_date = bkc_match.groups()
        return ReleaseInfo(
            branch=branch,
            label=f"{version} BKC {bkc_date}",
            version=version,
            bkc_date=bkc_date,
        )

    raise CherryPickError(
        f"Unsupported release branch {branch!r}; expected "
        "release/therock-X.Y or release/bkc/therock-X.Y-YYYYMMDD"
    )


def parse_dry_run(value: str) -> bool:
    """Parse the workflow's boolean input without truthy fallbacks."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise CherryPickError(
        f"Invalid dry-run value {value!r}; expected exactly true or false"
    )


def validate_ticket_requirement(release: ReleaseInfo, jira_id: str | None) -> None:
    """Require a ticket for regular releases while permitting ticketless BKC."""
    if not release.is_bkc and jira_id is None:
        raise CherryPickError(
            f"A JIRA ID is required for regular release branch {release.branch!r}"
        )


def derive_pr_branch(
    release: ReleaseInfo, jira_id: str | None, first_commit: str
) -> str:
    """Derive the new target branch from the ticket or ticketless BKC inputs."""
    if jira_id is not None:
        release_name = f"therock-{release.version}"
        if release.bkc_date is not None:
            release_name = f"bkc-therock-{release.version}-{release.bkc_date}"
        return f"shared/cherry-pick/{release_name}-{jira_id.lower()}"
    if release.bkc_date is None:
        raise CherryPickError("A ticketless PR branch is only valid for a BKC release")
    return (
        f"shared/cherry-pick/bkc-therock-{release.version}-{release.bkc_date}-"
        f"{first_commit[:7]}"
    )


def build_commit_message(
    commit: str,
    release_label: str,
    jira_id: str | None,
    original_message: str,
) -> str:
    """Build the exact message for one generated cherry-pick commit."""
    header = f"Cherry-picks commit {commit} into the ROCm {release_label} branch.\n\n"
    if jira_id is not None:
        header += f"JIRA ID: {jira_id}\n\n"
    return header + original_message


def build_pr_title(
    release_label: str, jira_id: str | None, commits: Sequence[CommitInfo]
) -> str:
    """Build the release pull request title."""
    if len(commits) == 1:
        return f"[ROCm {release_label}] {commits[0].subject}"
    ticket_prefix = f"{jira_id}: " if jira_id is not None else ""
    return (
        f"[ROCm {release_label}] {ticket_prefix}cherry-pick " f"{len(commits)} commits"
    )


def build_pr_body(
    release: ReleaseInfo,
    jira_id: str | None,
    commits: Sequence[CommitInfo],
    workflow_run_url: str,
    workflow_actor: str,
) -> str:
    """Build the pull request body, including TheRock run provenance."""
    lines = [
        f"Cherry-picks the following commits into the ROCm {release.label} "
        f"branch (`{release.branch}`):",
        "",
    ]
    lines.extend(f"- `{commit.sha}` {commit.subject}" for commit in commits)
    if jira_id is not None:
        lines.extend(["", f"JIRA ID: {jira_id}"])
    actor_url = f"https://github.com/{quote(workflow_actor, safe='')}"
    lines.extend(
        [
            "",
            f"Triggered by [@{workflow_actor}]({actor_url}) in "
            f"[the TheRock workflow run]({workflow_run_url}).",
        ]
    )
    return "\n".join(lines)


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


def _validate_branch_name(branch: str, description: str) -> None:
    result = _run_git(None, ["check-ref-format", "--branch", branch], check=False)
    if result.returncode != 0:
        raise CherryPickError(f"Invalid {description} branch name {branch!r}")


def target_metadata(api: GitHubClient, repository: str) -> str:
    """Return and validate the selected target repository's default branch."""
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{repository}"
    response = api.send_request(url)
    if not isinstance(response, dict):
        raise CherryPickError(
            f"Malformed metadata response for {TARGET_OWNER}/{repository}: "
            f"expected an object, got {type(response).__name__}"
        )
    default_branch = response.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch:
        raise CherryPickError(
            f"Metadata for {TARGET_OWNER}/{repository} has no valid default_branch"
        )
    _validate_branch_name(default_branch, "default")
    return default_branch


def bot_identity(api: GitHubClient, app_slug: str) -> BotIdentity:
    """Resolve the GitHub App bot account's commit identity."""
    login = f"{app_slug}[bot]"
    url = f"https://api.github.com/users/{quote(login, safe='')}"
    response = api.send_request(url)
    if not isinstance(response, dict):
        raise CherryPickError(
            f"Malformed bot response for {login}: expected an object, "
            f"got {type(response).__name__}"
        )
    response_login = response.get("login")
    if not isinstance(response_login, str) or not response_login:
        raise CherryPickError(f"Bot response for {login} has no valid login")
    if response_login.lower() != login.lower():
        raise CherryPickError(
            f"Bot lookup for {login} returned unexpected login {response_login!r}"
        )
    email = response.get("email")
    if email is not None:
        if not isinstance(email, str) or not email.strip():
            raise CherryPickError(f"Bot response for {login} has a malformed email")
    else:
        account_id = response.get("id")
        if not isinstance(account_id, int) or isinstance(account_id, bool):
            raise CherryPickError(
                f"Bot response for {login} has neither a usable email nor numeric ID"
            )
        email = f"{account_id}+{response_login}@users.noreply.github.com"
    return BotIdentity(name=login, email=email)


def create_pull_request(
    api: GitHubClient,
    repository: str,
    release_branch: str,
    pr_branch: str,
    title: str,
    body: str,
) -> str:
    """Create a target pull request and return its validated browser URL."""
    url = f"https://api.github.com/repos/{TARGET_OWNER}/{repository}/pulls"
    response = api.send_request(
        url,
        method="POST",
        body={
            "title": title,
            "body": body,
            "head": pr_branch,
            "base": release_branch,
        },
    )
    if not isinstance(response, dict):
        raise CherryPickError(
            f"Malformed pull request response for {TARGET_OWNER}/{repository}: "
            f"expected an object, got {type(response).__name__}"
        )
    html_url = response.get("html_url")
    if not isinstance(html_url, str):
        raise CherryPickError("Created pull request response has no valid html_url")
    parsed_url = urlsplit(html_url)
    expected_path = f"/{TARGET_OWNER}/{repository}/pull/"
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "github.com"
        or not parsed_url.path.startswith(expected_path)
    ):
        raise CherryPickError(
            f"Created pull request response has malformed html_url {html_url!r}"
        )
    return html_url


def clone_target(
    repository: str, target_path: Path, clone_url: str | None = None
) -> None:
    """Create a full-history, commit-only clone without a working tree."""
    public_url = clone_url or f"https://github.com/{TARGET_OWNER}/{repository}.git"
    _run_git(
        None,
        [
            "clone",
            "--filter=tree:0",
            "--no-checkout",
            public_url,
            os.fspath(target_path),
        ],
    )
    if not (target_path / ".git").is_dir():
        raise CherryPickError(f"Target clone was not created at {target_path}")


def _remote_ref(branch: str) -> str:
    return f"refs/remotes/origin/{branch}"


def _require_remote_branch(repo: Path, branch: str, description: str) -> str:
    ref = _remote_ref(branch)
    result = _run_git(
        repo,
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        check=False,
    )
    if result.returncode != 0:
        raise CherryPickError(
            f"The {description} branch {branch!r} does not exist in the fresh target clone"
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


def _branch_exists(repo: Path, branch: str) -> bool:
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
    release_branch: str,
) -> list[CommitInfo]:
    """Validate all commits before any cherry-pick commit is constructed."""
    default_ref = _require_remote_branch(repo, default_branch, "default")
    release_ref = _require_remote_branch(repo, release_branch, "release")
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
                f"Could not check whether commit {sha} is on origin/{default_branch}"
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
            ["merge-base", "--is-ancestor", sha, release_ref],
            check=False,
        )
        if ancestry.returncode == 0:
            raise CherryPickError(
                f"Commit {sha} is already an ancestor of release branch {release_branch}"
            )
        if ancestry.returncode != 1:
            raise CherryPickError(
                f"Could not inspect release ancestry for commit {sha}"
            )

        previous_pick = _run_git(
            repo,
            [
                "log",
                "--format=%H",
                "--fixed-strings",
                f"--grep=Cherry-picks commit {sha}",
                "-n",
                "1",
                release_ref,
            ],
        )
        if previous_pick.stdout.strip():
            raise CherryPickError(
                f"Commit {sha} was already cherry-picked onto {release_branch}"
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
    release_branch: str,
    release_label: str,
    jira_id: str | None,
    commits: Sequence[CommitInfo],
    committer: BotIdentity,
) -> list[BuiltCommit]:
    """Construct picks using attributes from each generated release head."""
    head = _run_git(
        repo,
        ["rev-parse", "--verify", f"{_remote_ref(release_branch)}^{{commit}}"],
    ).stdout.strip()
    built: list[BuiltCommit] = []

    for commit in commits:
        merge = _run_git(
            repo,
            [
                "merge-tree",
                "--write-tree",
                "--name-only",
                f"--merge-base={commit.parent}",
                head,
                commit.sha,
            ],
            check=False,
            extra_env={"GIT_ATTR_SOURCE": head},
        )
        if merge.returncode == 1:
            raise CherryPickConflictError(commit.sha, _conflict_paths(merge.stdout))
        if merge.returncode != 0:
            details = merge.stderr.strip() or "no details"
            raise CherryPickError(
                f"git merge-tree failed for commit {commit.sha}: {details}"
            )
        lines = merge.stdout.splitlines()
        tree = lines[0].strip() if lines else ""
        if not FULL_SHA_RE.fullmatch(tree):
            raise CherryPickError(
                f"git merge-tree returned invalid tree {tree!r} for commit {commit.sha}"
            )

        head_tree = _run_git(
            repo, ["rev-parse", "--verify", f"{head}^{{tree}}"]
        ).stdout.strip()
        if tree == head_tree:
            raise CherryPickError(
                f"Cherry-picking commit {commit.sha} onto {release_branch} is empty"
            )

        message = build_commit_message(
            commit=commit.sha,
            release_label=release_label,
            jira_id=jira_id,
            original_message=commit.message,
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
            input_text=message,
            extra_env={
                "GIT_AUTHOR_NAME": commit.author_name,
                "GIT_AUTHOR_EMAIL": commit.author_email,
                "GIT_AUTHOR_DATE": commit.author_date,
                "GIT_COMMITTER_NAME": committer.name,
                "GIT_COMMITTER_EMAIL": committer.email,
            },
        )
        head = commit_result.stdout.strip()
        if not FULL_SHA_RE.fullmatch(head):
            raise CherryPickError(
                f"git commit-tree returned invalid commit {head!r} for {commit.sha}"
            )
        built.append(BuiltCommit(source=commit, sha=head))

    return built


def push_branch(
    repo: Path,
    repository: str,
    token: str,
    head: str,
    pr_branch: str,
    push_url: str | None = None,
) -> None:
    """Create an absent branch and verify creation without exposing credentials."""
    authenticated_url = push_url or (
        f"https://x-access-token:{quote(token, safe='')}@github.com/"
        f"{TARGET_OWNER}/{repository}.git"
    )
    destination = f"refs/heads/{pr_branch}"
    refspec = f"{head}:{destination}"
    command = [
        "git",
        "-C",
        os.fspath(repo),
        "push",
        "--porcelain",
        f"--force-with-lease={destination}:",
        authenticated_url,
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
            f"Could not execute git push for {TARGET_OWNER}/{repository} branch "
            f"{pr_branch!r}"
        ) from exc
    if result.returncode != 0:
        raise CherryPickError(
            f"Failed to push {TARGET_OWNER}/{repository} branch {pr_branch!r} "
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
            f"Target PR branch {pr_branch!r} already exists in "
            f"{TARGET_OWNER}/{repository}; no pull request was created"
        )
    if statuses != ["*"]:
        raise CherryPickError(
            f"Could not confirm creation of {TARGET_OWNER}/{repository} branch "
            f"{pr_branch!r}; inspect the destination before retrying. "
            "No pull request was created"
        )


def _validate_workflow_provenance(workflow_run_url: str, workflow_actor: str) -> None:
    parsed_url = urlsplit(workflow_run_url)
    expected_path = f"/{HOST_REPOSITORY}/actions/runs/"
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "github.com"
        or not parsed_url.path.startswith(expected_path)
    ):
        raise CherryPickError(
            f"Workflow run URL must point to a {HOST_REPOSITORY} Actions run"
        )
    if not workflow_actor.strip():
        raise CherryPickError("The triggering workflow actor is required")


def _conflict_summary(
    repository: str, release_branch: str, error: CherryPickConflictError
) -> str:
    paths = "\n".join(f"- `{path}`" for path in error.paths)
    if not paths:
        paths = "- Conflict paths were not reported by Git"
    return "\n".join(
        [
            "# Cherry-pick conflict",
            "",
            f"- Target repository: `{TARGET_OWNER}/{repository}`",
            f"- Release branch: `{release_branch}`",
            "",
            f"Commit `{error.commit}` did not apply cleanly.",
            "",
            "Conflicting paths:",
            paths,
            "",
            "Nothing was pushed and no pull request was created.",
        ]
    )


def _dry_run_summary(result: RunResult) -> str:
    generated = "\n".join(
        f"- `{item.sha}` from `{item.source.sha}`" for item in result.built_commits
    )
    return "\n".join(
        [
            "# Dry run: cherry-pick to release",
            "",
            "**No branch was pushed and no pull request was created.**",
            "",
            f"- Target repository: `{TARGET_OWNER}/{result.repository}`",
            f"- Release branch: `{result.release_branch}`",
            f"- Prospective branch: `{result.pr_branch}`",
            "- Generated commits:",
            generated,
            "",
            "## Prospective pull request",
            "",
            f"**Title:** {result.pr_title}",
            "",
            result.pr_body,
        ]
    )


def _success_summary(result: RunResult) -> str:
    if result.pr_url is None:
        raise AssertionError("A live-run summary requires a pull request URL")
    return "\n".join(
        [
            "# Cherry-pick pull request created",
            "",
            f"- Target repository: `{TARGET_OWNER}/{result.repository}`",
            f"- Release branch: `{result.release_branch}`",
            f"- Pull request: {result.pr_url}",
        ]
    )


def execute_cherry_pick(
    *,
    repository_value: str,
    commits_value: str,
    release_branch_value: str,
    jira_id_value: str,
    dry_run: bool,
    app_token: str,
    app_slug: str,
    workflow_run_url: str,
    workflow_actor: str,
    api: GitHubClient,
    summary_writer: Callable[[str], None] = gha_append_step_summary,
    clone_url: str | None = None,
    push_url: str | None = None,
) -> RunResult:
    """Validate, construct and optionally publish a release cherry-pick."""
    repository = parse_repository(repository_value)
    commit_specs = parse_commit_list(commits_value)
    release = parse_release_branch(release_branch_value)
    jira_id = parse_jira_id(jira_id_value)
    validate_ticket_requirement(release, jira_id)
    if not app_token.strip():
        raise CherryPickError("The selected GitHub App token is missing or empty")
    expected_app_slug = SUPPORTED_REPOSITORIES[repository]
    if app_slug != expected_app_slug:
        raise CherryPickError(
            f"Repository {repository!r} requires App {expected_app_slug!r}, "
            f"not {app_slug!r}"
        )
    _validate_workflow_provenance(workflow_run_url, workflow_actor)
    check_git_version()

    default_branch = target_metadata(api, repository)
    with tempfile.TemporaryDirectory(prefix="therock-cherry-pick-") as temp_dir:
        target_path = Path(temp_dir) / repository
        clone_target(repository, target_path, clone_url=clone_url)
        _require_remote_branch(target_path, release.branch, "release")
        resolved_commits = resolve_commits(target_path, commit_specs)
        pr_branch = derive_pr_branch(
            release=release, jira_id=jira_id, first_commit=resolved_commits[0]
        )
        if _branch_exists(target_path, pr_branch):
            raise CherryPickError(
                f"Target PR branch {pr_branch!r} already exists in "
                f"{TARGET_OWNER}/{repository}"
            )

        commits = validate_commits(
            repo=target_path,
            resolved_commits=resolved_commits,
            default_branch=default_branch,
            release_branch=release.branch,
        )
        committer = bot_identity(api, app_slug)
        try:
            built_commits = build_cherry_pick_commits(
                repo=target_path,
                release_branch=release.branch,
                release_label=release.label,
                jira_id=jira_id,
                commits=commits,
                committer=committer,
            )
        except CherryPickConflictError as exc:
            summary_writer(
                _conflict_summary(
                    repository=repository, release_branch=release.branch, error=exc
                )
            )
            raise

        pr_title = build_pr_title(
            release_label=release.label, jira_id=jira_id, commits=commits
        )
        pr_body = build_pr_body(
            release=release,
            jira_id=jira_id,
            commits=commits,
            workflow_run_url=workflow_run_url,
            workflow_actor=workflow_actor,
        )
        result = RunResult(
            repository=repository,
            release_branch=release.branch,
            pr_branch=pr_branch,
            built_commits=built_commits,
            pr_title=pr_title,
            pr_body=pr_body,
            pr_url=None,
            dry_run=dry_run,
        )
        if dry_run:
            summary_writer(_dry_run_summary(result))
            return result

        push_branch(
            repo=target_path,
            repository=repository,
            token=app_token,
            head=built_commits[-1].sha,
            pr_branch=pr_branch,
            push_url=push_url,
        )
        try:
            pr_url = create_pull_request(
                api=api,
                repository=repository,
                release_branch=release.branch,
                pr_branch=pr_branch,
                title=pr_title,
                body=pr_body,
            )
        except (GitHubAPIError, CherryPickError) as exc:
            raise CherryPickError(
                f"Branch {pr_branch!r} was pushed to {TARGET_OWNER}/{repository}, "
                f"but pull request creation failed: {exc}"
            ) from exc
        except Exception as exc:
            exc.add_note(
                f"Branch {pr_branch!r} was pushed to {TARGET_OWNER}/{repository}, "
                "but pull request creation failed. Inspect the destination "
                "before retrying."
            )
            raise

        result = replace(result, pr_url=pr_url, dry_run=False)
        summary_writer(_success_summary(result))
        return result


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cherry-pick target default-branch commits into a release branch"
    )
    parser.add_argument(
        "--repository",
        default=os.getenv("CHERRY_PICK_REPOSITORY", ""),
        help="Target repository: TheRock, rocm-systems, or rocm-libraries "
        "(env: CHERRY_PICK_REPOSITORY)",
    )
    parser.add_argument(
        "--commits",
        default=os.getenv("CHERRY_PICK_COMMITS", ""),
        help="Comma or whitespace separated commit SHAs in cherry-pick order "
        "(env: CHERRY_PICK_COMMITS)",
    )
    parser.add_argument(
        "--release-branch",
        default=os.getenv("CHERRY_PICK_RELEASE_BRANCH", ""),
        help="Target release/therock-X.Y or release/bkc/therock-X.Y-YYYYMMDD "
        "branch (env: CHERRY_PICK_RELEASE_BRANCH)",
    )
    parser.add_argument(
        "--jira-id",
        default=os.getenv("CHERRY_PICK_JIRA_ID", ""),
        help="JIRA key, required for regular releases and optional for BKC "
        "(env: CHERRY_PICK_JIRA_ID)",
    )
    parser.add_argument(
        "--app-token",
        default=os.getenv("GITHUB_APP_TOKEN", ""),
        help="GitHub App installation token (env: GITHUB_APP_TOKEN)",
    )
    parser.add_argument(
        "--app-slug",
        default=os.getenv("GITHUB_APP_SLUG", ""),
        help="GitHub App slug matching the target repository " "(env: GITHUB_APP_SLUG)",
    )
    parser.add_argument(
        "--workflow-run-url",
        default=os.getenv("WORKFLOW_RUN_URL", ""),
        help="TheRock Actions run URL for pull request provenance "
        "(env: WORKFLOW_RUN_URL)",
    )
    parser.add_argument(
        "--workflow-actor",
        default=os.getenv("WORKFLOW_ACTOR", ""),
        help="GitHub login of the triggering operator (env: WORKFLOW_ACTOR)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=parse_dry_run(os.getenv("CHERRY_PICK_DRY_RUN", "true")),
        help="Build and report without publishing (default: true; "
        "env: CHERRY_PICK_DRY_RUN). --no-dry-run publishes the branch and PR; "
        "either flag overrides the environment setting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _create_argument_parser().parse_args(argv)
        api = GitHubAPI(github_token=args.app_token)
        execute_cherry_pick(
            repository_value=args.repository,
            commits_value=args.commits,
            release_branch_value=args.release_branch,
            jira_id_value=args.jira_id,
            dry_run=args.dry_run,
            app_token=args.app_token,
            app_slug=args.app_slug,
            workflow_run_url=args.workflow_run_url,
            workflow_actor=args.workflow_actor,
            api=api,
        )
    except (CherryPickError, GitHubAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
