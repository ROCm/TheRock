# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Coordinate release cherry-picks and publish their pull requests.

Examples:

    python3 build_tools/github_actions/cherry_pick_to_release.py --help

    python3 build_tools/github_actions/cherry_pick_to_release.py \
        --repository rocm-systems \
        --commits abcdef0,1234567 \
        --release-branch release/therock-10.2 \
        --jira-id AIRDEL-1234 \
        --dry-run

The GitHub Actions workflow supplies the same inputs through environment
variables.
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import cast
from urllib.parse import quote, urlsplit


_BUILD_TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_BUILD_TOOLS_DIR))

from git_cherry_pick import (
    branch_exists,
    BuiltCommit,
    build_cherry_pick_commits,
    CherryPickConflictError,
    CherryPickError,
    clone_repository,
    CommitIdentity,
    CommitInfo,
    inspect_commits,
    PreparedCommit,
    push_branch,
    resolve_commits,
)
from github_actions_api import (
    gha_append_step_summary,
    GitHubAPI,
    GitHubAPIError,
)


TARGET_OWNER = "ROCm"
HOST_REPOSITORY = "ROCm/TheRock"
MINIMUM_SHORT_SHA_LENGTH = 7
# Target repository to GitHub App slug. The repository selects the bot
# committer identity instead of accepting it as an independent operator input,
# matching the repository-specific App token routing in the workflow.
REPOSITORY_APP_SLUGS = {
    "TheRock": "systems-assistant",
    "rocm-systems": "systems-assistant",
    "rocm-libraries": "assistant-librarian",
}

COMMIT_RE = re.compile(rf"^[0-9a-fA-F]{{{MINIMUM_SHORT_SHA_LENGTH},}}$")
JIRA_RE = re.compile(r"^[A-Z][A-Z0-9_]*-[1-9][0-9]*$")
REGULAR_RELEASE_RE = re.compile(r"^release/therock-([0-9]+\.[0-9]+)$")
BKC_RELEASE_RE = re.compile(r"^release/bkc/therock-([0-9]+\.[0-9]+)-([0-9]{8})$")
CHERRY_PICK_MESSAGE_PREFIX = "Cherry-picks commit"


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
    if value not in REPOSITORY_APP_SLUGS:
        choices = ", ".join(sorted(REPOSITORY_APP_SLUGS))
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
    """Parse comma or whitespace separated hexadecimal commit IDs."""
    commits = [item for item in re.split(r"[,\s]+", value.strip()) if item]
    if not commits:
        raise CherryPickError("At least one commit SHA is required")
    for commit in commits:
        if not COMMIT_RE.fullmatch(commit):
            raise CherryPickError(
                f"Invalid commit {commit!r}; expected at least 7 hexadecimal characters"
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
    release: ReleaseInfo, jira_id: str | None, first_commit_sha: str
) -> str:
    """Derive the destination branch name.

    For a ticketless BKC release, first_commit_sha is the full resolved SHA
    of the first commit in cherry-pick order and supplies the branch suffix.
    """
    if jira_id is not None:
        release_name = f"therock-{release.version}"
        if release.bkc_date is not None:
            release_name = f"bkc-therock-{release.version}-{release.bkc_date}"
        return f"shared/cherry-pick/{release_name}-{jira_id.lower()}"
    if release.bkc_date is None:
        raise CherryPickError("A ticketless PR branch is only valid for a BKC release")
    return (
        f"shared/cherry-pick/bkc-therock-{release.version}-{release.bkc_date}-"
        f"{first_commit_sha[:7]}"
    )


def build_commit_message(
    commit: str,
    release_label: str,
    jira_id: str | None,
    original_message: str,
) -> str:
    """Build the exact message for one generated cherry-pick commit."""
    header = (
        f"{CHERRY_PICK_MESSAGE_PREFIX} {commit} into the ROCm "
        f"{release_label} branch.\n\n"
    )
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
    """Build the pull request body, including available workflow provenance."""
    lines = [
        f"Cherry-picks the following commits into the ROCm {release.label} "
        f"branch (`{release.branch}`):",
        "",
    ]
    lines.extend(f"- `{commit.sha}` {commit.subject}" for commit in commits)
    if jira_id is not None:
        lines.extend(["", f"JIRA ID: {jira_id}"])
    if workflow_run_url and workflow_actor:
        actor_url = f"https://github.com/{quote(workflow_actor, safe='')}"
        lines.extend(
            [
                "",
                f"Triggered by [@{workflow_actor}]({actor_url}) in "
                f"[the TheRock workflow run]({workflow_run_url}).",
            ]
        )
    return "\n".join(lines)


def target_metadata(api: GitHubAPI, repository: str) -> str:
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
    return default_branch


def bot_identity(api: GitHubAPI, app_slug: str) -> CommitIdentity:
    """Resolve the GitHub App bot account's commit identity."""
    login = f"{app_slug}[bot]"
    url = f"https://api.github.com/users/{quote(login, safe='')}"
    response = api.send_request(url)
    if not isinstance(response, dict):
        raise CherryPickError(
            f"Malformed bot response for {login}: expected an object, "
            f"got {type(response).__name__}"
        )
    email = cast(str | None, response.get("email"))
    if email is None:
        account_id = response.get("id")
        if not isinstance(account_id, int):
            raise CherryPickError(
                f"Bot response for {login} has no numeric ID for its fallback email"
            )
        email = f"{account_id}+{login}@users.noreply.github.com"
    return CommitIdentity(name=login, email=email)


def create_pull_request(
    api: GitHubAPI,
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


def _validate_workflow_provenance(
    workflow_run_url: str, workflow_actor: str, *, required: bool
) -> None:
    has_run_url = bool(workflow_run_url)
    has_actor = bool(workflow_actor.strip())
    if not has_run_url and not has_actor:
        if required:
            raise CherryPickError(
                "The workflow run URL and triggering actor are required for live runs"
            )
        return
    if not has_run_url or not has_actor:
        raise CherryPickError(
            "The workflow run URL and triggering actor must be provided together"
        )
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


def _failure_summary(
    repository: str,
    commits_value: str,
    release_branch: str,
    jira_id_value: str,
    error: Exception,
) -> str:
    commits = [item for item in re.split(r"[,\s]+", commits_value.strip()) if item]
    lines = [
        "# Cherry-pick to release failed",
        "",
        f"- Target repository: `{TARGET_OWNER}/{repository}`",
        f"- Release branch: `{release_branch}`",
        "- Requested commits:",
    ]
    lines.extend(f"  - `{commit}`" for commit in commits)
    if not commits:
        lines.append("  - not provided")
    jira_id = jira_id_value.strip()
    lines.append(f"- JIRA ID: `{jira_id}`" if jira_id else "- JIRA ID: not provided")
    lines.extend(["", "## Failure", ""])
    if isinstance(error, CherryPickConflictError):
        lines.extend(
            [
                f"Commit `{error.commit}` did not apply cleanly.",
                "",
                "## Git diagnostics",
                "",
                "```text",
                error.diagnostics.rstrip(),
                "```",
            ]
        )
    else:
        lines.extend(["```text", str(error), "```"])
    return "\n".join(lines)


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


def _success_summary(result: RunResult, pr_url: str) -> str:
    return "\n".join(
        [
            "# Cherry-pick pull request created",
            "",
            f"- Target repository: `{TARGET_OWNER}/{result.repository}`",
            f"- Release branch: `{result.release_branch}`",
            f"- Pull request: {pr_url}",
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
    workflow_run_url: str,
    workflow_actor: str,
    api: GitHubAPI,
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
    if not dry_run and not app_token.strip():
        raise CherryPickError("The selected GitHub App token is missing or empty")
    app_slug = REPOSITORY_APP_SLUGS[repository]
    _validate_workflow_provenance(
        workflow_run_url, workflow_actor, required=not dry_run
    )

    default_branch = target_metadata(api, repository)
    with tempfile.TemporaryDirectory(prefix="therock-cherry-pick-") as temp_dir:
        target_path = Path(temp_dir) / repository
        public_url = clone_url or f"https://github.com/{TARGET_OWNER}/{repository}.git"
        clone_repository(public_url, target_path)
        resolved_commits = resolve_commits(target_path, commit_specs)
        pr_branch = derive_pr_branch(
            release=release, jira_id=jira_id, first_commit_sha=resolved_commits[0]
        )
        if branch_exists(target_path, pr_branch):
            raise CherryPickError(
                f"Target PR branch {pr_branch!r} already exists in "
                f"{TARGET_OWNER}/{repository}"
            )

        commits = inspect_commits(
            repo=target_path,
            resolved_commits=resolved_commits,
            default_branch=default_branch,
        )
        committer = bot_identity(api, app_slug)
        prepared_commits = [
            PreparedCommit(
                source=commit,
                message=build_commit_message(
                    commit=commit.sha,
                    release_label=release.label,
                    jira_id=jira_id,
                    original_message=commit.message,
                ),
            )
            for commit in commits
        ]
        try:
            built_commits = build_cherry_pick_commits(
                repo=target_path,
                target_branch=release.branch,
                target_description="release",
                commits=prepared_commits,
                committer=committer,
            )
        except CherryPickConflictError as exc:
            summary_writer(
                _failure_summary(
                    repository=repository,
                    commits_value=commits_value,
                    release_branch=release.branch,
                    jira_id_value=jira_id_value,
                    error=exc,
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

        if push_url is None:
            remote_url = f"https://github.com/{TARGET_OWNER}/{repository}.git"
            push_token = app_token
        else:
            remote_url = push_url
            push_token = None
        try:
            push_branch(
                repo=target_path,
                remote_url=remote_url,
                remote_description=f"{TARGET_OWNER}/{repository}",
                head=built_commits[-1].sha,
                branch=pr_branch,
                token=push_token,
            )
        except CherryPickError as exc:
            raise CherryPickError(f"{exc}; no pull request was created") from exc
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
        summary_writer(_success_summary(result, pr_url))
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
        help="GitHub App installation token, required for live runs and optional "
        "for dry runs (env: GITHUB_APP_TOKEN)",
    )
    parser.add_argument(
        "--workflow-run-url",
        default=os.getenv("WORKFLOW_RUN_URL", ""),
        help="Link to the ROCm/TheRock Actions run that invoked this script, "
        "added to the pull request body. Must be provided together with "
        "--workflow-actor. Set automatically by the cherry-pick-to-release "
        "workflow; required for live runs, optional for dry runs "
        "(env: WORKFLOW_RUN_URL)",
    )
    parser.add_argument(
        "--workflow-actor",
        default=os.getenv("WORKFLOW_ACTOR", ""),
        help="GitHub login of the user who triggered the workflow, added to the "
        "pull request body with the workflow run link. Must be provided together "
        "with --workflow-run-url. Set automatically by the "
        "cherry-pick-to-release workflow; required for live runs, optional for "
        "dry runs (env: WORKFLOW_ACTOR)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build and report without publishing (default: true; "
        "env: CHERRY_PICK_DRY_RUN). --no-dry-run publishes the branch and PR; "
        "either flag overrides the environment setting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _create_argument_parser().parse_args(argv)
        dry_run = (
            args.dry_run
            if args.dry_run is not None
            else parse_dry_run(os.getenv("CHERRY_PICK_DRY_RUN", "true"))
        )
        api = GitHubAPI(github_token=args.app_token or None)
        execute_cherry_pick(
            repository_value=args.repository,
            commits_value=args.commits,
            release_branch_value=args.release_branch,
            jira_id_value=args.jira_id,
            dry_run=dry_run,
            app_token=args.app_token,
            workflow_run_url=args.workflow_run_url,
            workflow_actor=args.workflow_actor,
            api=api,
        )
    except (CherryPickError, GitHubAPIError) as exc:
        if not isinstance(exc, CherryPickConflictError):
            gha_append_step_summary(
                _failure_summary(
                    repository=args.repository,
                    commits_value=args.commits,
                    release_branch=args.release_branch,
                    jira_id_value=args.jira_id,
                    error=exc,
                )
            )
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
