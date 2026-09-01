#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Resolve the ROCm/TheRock commit an external repo CI run should build against.

Policy (see the "Merge-base-time TheRock ref" design):

- On a ``pull_request`` event, pin TheRock to the tip of ``main`` as it existed
  at the time of the PR branch's merge-base with its base branch. The pin stays
  frozen while the author pushes new commits and only advances when they merge
  or rebase the base branch back in. This keeps the ROCm build underneath a PR
  stable during authoring and debugging.
- A caller-supplied override short-circuits everything (manual pin).
- On any other event (push / workflow_dispatch / schedule) there is no
  merge-base, so use the live tip of ``main`` (equivalent to the prior
  behavior).

The merge-base is computed server-side by GitHub's compare API, so no deep
clone of the calling repository is required. Staleness is warn-only: if the
resolved TheRock commit is older than the threshold we surface a warning but
still use the commit.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Protocol, Tuple, Type, TypeVar

import requests

F = TypeVar("F", bound=Callable[..., object])

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_THEROCK_REPO = "ROCm/TheRock"
DEFAULT_THEROCK_BRANCH = "main"
DEFAULT_STALENESS_DAYS = 14

MODE_OVERRIDE = "override"
MODE_MERGE_BASE = "pull-request merge-base"
MODE_MERGE_BASE_FALLBACK = "pull-request merge-base (fell back to live tip of main)"
MODE_LIVE_TIP = "live-tip (push/dispatch/schedule)"


def retry(
    max_attempts: int,
    delay_seconds: float,
    exceptions: Tuple[Type[BaseException], ...],
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                    attempt += 1
                    if attempt < max_attempts:
                        backoff = delay_seconds * (2 ** (attempt - 1))
                        time.sleep(backoff)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def set_github_output(outputs: Mapping[str, str]) -> None:
    """Write key=value pairs to GITHUB_OUTPUT (or print if not in CI)."""
    logger.info(f"Setting github output: {dict(outputs)}")
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        for k, v in outputs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")


def append_step_summary(summary: str) -> None:
    """Append markdown to GITHUB_STEP_SUMMARY (or print if not in CI)."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not summary_file:
        print(summary)
        return
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(summary)


def parse_github_time(value: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp (``...Z``) into an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_utc(dt: datetime) -> str:
    """Render an aware datetime as a compact UTC ISO-8601 string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def humanize_delta(seconds: float) -> str:
    """Render a duration in seconds as a coarse human string (e.g. '3 days')."""
    seconds = abs(seconds)
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    if days >= 1:
        value = int(round(days))
        return f"{value} day{'s' if value != 1 else ''}"
    if hours >= 1:
        value = int(round(hours))
        return f"{value} hour{'s' if value != 1 else ''}"
    if minutes >= 1:
        value = int(round(minutes))
        return f"{value} minute{'s' if value != 1 else ''}"
    value = int(round(seconds))
    return f"{value} second{'s' if value != 1 else ''}"


def humanize_age(dt: datetime, now: datetime) -> str:
    """Render how long ago ``dt`` was relative to ``now``."""
    return f"{humanize_delta((now - dt).total_seconds())} ago"


@dataclass
class Commit:
    """A resolved commit: its SHA and committer timestamp."""

    sha: str
    committed_at: datetime


@dataclass
class Resolution:
    """The outcome of resolving which TheRock commit to build against."""

    therock_ref: str
    therock_repo: str
    mode: str
    therock_commit: Optional[Commit] = None
    merge_base: Optional[Commit] = None
    source_repo: str = ""
    warnings: list[str] = field(default_factory=list)
    staleness_days: int = DEFAULT_STALENESS_DAYS


class GitHubClient(Protocol):
    """Minimal GitHub read surface the resolver depends on (for testability)."""

    def get_merge_base(self, repo: str, base_sha: str, head_sha: str) -> Commit: ...

    def get_commit_at_or_before(
        self, repo: str, branch: str, until: datetime
    ) -> Optional[Commit]: ...

    def get_live_tip(self, repo: str, branch: str) -> Commit: ...

    def get_commit(self, repo: str, ref: str) -> Commit: ...


class RestGitHubClient:
    """GitHub REST client used at runtime. Read-only; retries transient errors."""

    def __init__(self, token: str, api_url: str = GITHUB_API):
        self._api_url = api_url
        self._session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._session.headers.update(headers)

    @retry(max_attempts=3, delay_seconds=2, exceptions=(requests.RequestException,))
    def _get(self, path: str, params: Optional[dict] = None):
        response = self._session.get(
            f"{self._api_url}{path}", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _commit_from_payload(payload: dict) -> Commit:
        return Commit(
            sha=payload["sha"],
            committed_at=parse_github_time(payload["commit"]["committer"]["date"]),
        )

    def get_merge_base(self, repo: str, base_sha: str, head_sha: str) -> Commit:
        payload = self._get(f"/repos/{repo}/compare/{base_sha}...{head_sha}")
        return self._commit_from_payload(payload["merge_base_commit"])

    def get_commit_at_or_before(
        self, repo: str, branch: str, until: datetime
    ) -> Optional[Commit]:
        payload = self._get(
            f"/repos/{repo}/commits",
            params={"sha": branch, "until": iso_utc(until), "per_page": 1},
        )
        if not payload:
            return None
        return self._commit_from_payload(payload[0])

    def get_live_tip(self, repo: str, branch: str) -> Commit:
        payload = self._get(
            f"/repos/{repo}/commits", params={"sha": branch, "per_page": 1}
        )
        if not payload:
            raise ValueError(f"No commits found on {repo}@{branch}")
        return self._commit_from_payload(payload[0])

    def get_commit(self, repo: str, ref: str) -> Commit:
        payload = self._get(f"/repos/{repo}/commits/{ref}")
        return self._commit_from_payload(payload)


def resolve_ref(
    client: GitHubClient,
    *,
    event_name: str,
    source_repo: str,
    base_sha: str,
    head_sha: str,
    override: str,
    therock_repo: str = DEFAULT_THEROCK_REPO,
    therock_branch: str = DEFAULT_THEROCK_BRANCH,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
    now: Optional[datetime] = None,
) -> Resolution:
    """Resolve which TheRock commit to build against for this run."""
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []

    if override:
        commit = client.get_commit(therock_repo, override)
        resolution = Resolution(
            therock_ref=commit.sha,
            therock_repo=therock_repo,
            mode=MODE_OVERRIDE,
            therock_commit=commit,
            warnings=warnings,
            staleness_days=staleness_days,
        )
    elif event_name == "pull_request" and base_sha and head_sha:
        merge_base = client.get_merge_base(source_repo, base_sha, head_sha)
        commit = client.get_commit_at_or_before(
            therock_repo, therock_branch, merge_base.committed_at
        )
        if commit is None:
            commit = client.get_live_tip(therock_repo, therock_branch)
            warnings.append(
                f"No {therock_repo}@{therock_branch} commit found at or before the "
                "merge-base time; fell back to the live tip of the branch."
            )
            mode = MODE_MERGE_BASE_FALLBACK
        else:
            mode = MODE_MERGE_BASE
        resolution = Resolution(
            therock_ref=commit.sha,
            therock_repo=therock_repo,
            mode=mode,
            therock_commit=commit,
            merge_base=merge_base,
            warnings=warnings,
            staleness_days=staleness_days,
        )
    else:
        commit = client.get_live_tip(therock_repo, therock_branch)
        resolution = Resolution(
            therock_ref=commit.sha,
            therock_repo=therock_repo,
            mode=MODE_LIVE_TIP,
            therock_commit=commit,
            warnings=warnings,
            staleness_days=staleness_days,
        )

    resolution.source_repo = source_repo
    _check_staleness(resolution, now)
    return resolution


def _is_merge_base_mode(resolution: Resolution) -> bool:
    """True when the ref was pinned from a PR merge-base (vs. override/live-tip)."""
    return resolution.merge_base is not None


def _check_staleness(resolution: Resolution, now: datetime) -> None:
    commit = resolution.therock_commit
    if commit is None:
        return
    age_days = (now - commit.committed_at).total_seconds() / 86400
    if age_days > resolution.staleness_days:
        if _is_merge_base_mode(resolution):
            hint = (
                "consider syncing your base branch (merge or rebase) to pick up "
                "a newer TheRock."
            )
        else:
            hint = (
                "consider building against a newer TheRock (for example via the "
                "`therock_ref_override` input)."
            )
        resolution.warnings.append(
            f"This TheRock commit is {int(age_days)} days old "
            f"(threshold {resolution.staleness_days} days); {hint}"
        )


def build_summary(resolution: Resolution, now: Optional[datetime] = None) -> str:
    """Render the verbose, self-explaining GitHub step-summary card."""
    now = now or datetime.now(timezone.utc)
    repo = resolution.therock_repo
    commit = resolution.therock_commit

    lines: list[str] = []
    lines.append("### TheRock version locked for this run")
    lines.append("")
    if _is_merge_base_mode(resolution):
        intro = (
            "This PR is built against a fixed TheRock commit chosen from the "
            "point where your branch last synced with its base branch (the "
            "merge-base). It stays frozen while you push new commits, and only "
            "moves when you merge or rebase the base branch back into your "
            "branch. This keeps the ROCm build underneath you stable while you "
            "author and debug."
        )
    elif resolution.mode == MODE_OVERRIDE:
        intro = (
            "This run is built against an explicitly pinned TheRock ref supplied "
            "via the `therock_ref_override` input."
        )
    else:
        intro = (
            "This run is built against the live tip of "
            f"{repo}@{DEFAULT_THEROCK_BRANCH}. Merge-base pinning only applies to "
            "`pull_request` events; other events (push, workflow_dispatch, "
            "schedule) use the current tip."
        )
    lines.append(intro)
    lines.append("")

    if commit is not None:
        commit_url = f"https://github.com/{repo}/commit/{commit.sha}"
        lines.append(
            f"- **Resolved TheRock commit:** [`{commit.sha[:12]}`]({commit_url}) "
            f"(committed {iso_utc(commit.committed_at)}, "
            f"{humanize_age(commit.committed_at, now)})"
        )

    merge_base = resolution.merge_base
    if merge_base is not None:
        if resolution.source_repo:
            mb_url = (
                f"https://github.com/{resolution.source_repo}/commit/{merge_base.sha}"
            )
            mb_label = f"[`{merge_base.sha[:12]}`]({mb_url})"
        else:
            mb_label = f"`{merge_base.sha[:12]}`"
        lines.append(
            f"- **Merge-base commit (this repo):** {mb_label} "
            f"(committed {iso_utc(merge_base.committed_at)}, "
            f"{humanize_age(merge_base.committed_at, now)}) - the base-branch "
            "state your branch is built on."
        )
        lines.append(
            "- **Mapping rule:** chose the newest "
            f"{repo}@{DEFAULT_THEROCK_BRANCH} commit at or before the merge-base "
            "time."
        )
        if commit is not None:
            skew = (merge_base.committed_at - commit.committed_at).total_seconds()
            direction = "older than" if skew >= 0 else "newer than"
            lines.append(
                f"- **Repo skew:** the TheRock commit is {humanize_delta(skew)} "
                f"{direction} the merge-base, a rough gauge of how well-aligned "
                "the two repos were at that moment."
            )

    lines.append(f"- **Resolution mode:** {resolution.mode}")
    if resolution.merge_base is None and resolution.mode != MODE_OVERRIDE:
        lines.append(
            "  - Merge-base pinning only applies to `pull_request` events; this "
            "run used the live tip instead."
        )
    lines.append("")
    if _is_merge_base_mode(resolution):
        how_to_change = (
            "**How to change this:** merge or rebase your base branch into this "
            "PR to advance the TheRock version, or set the `therock_ref_override` "
            "input on a manual run to pin an explicit ref."
        )
    else:
        how_to_change = (
            "**How to change this:** set the `therock_ref_override` input on a "
            "manual run to pin an explicit ref."
        )
    lines.append(how_to_change)

    for warning in resolution.warnings:
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append(f"> {warning}")

    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    source_repo = os.environ.get("GITHUB_REPOSITORY", "")
    base_sha = os.environ.get("BASE_SHA", "")
    head_sha = os.environ.get("HEAD_SHA", "")
    override = os.environ.get("THEROCK_REF_OVERRIDE", "").strip()
    therock_repo = os.environ.get("THEROCK_REPO", DEFAULT_THEROCK_REPO)
    therock_branch = os.environ.get("THEROCK_BRANCH", DEFAULT_THEROCK_BRANCH)
    staleness_days = int(
        os.environ.get("STALENESS_THRESHOLD_DAYS", str(DEFAULT_STALENESS_DAYS))
    )

    client = RestGitHubClient(token)
    resolution = resolve_ref(
        client,
        event_name=event_name,
        source_repo=source_repo,
        base_sha=base_sha,
        head_sha=head_sha,
        override=override,
        therock_repo=therock_repo,
        therock_branch=therock_branch,
        staleness_days=staleness_days,
    )

    set_github_output({"therock_ref": resolution.therock_ref})
    append_step_summary(build_summary(resolution))


if __name__ == "__main__":
    main()
