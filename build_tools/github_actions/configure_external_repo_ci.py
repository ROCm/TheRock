#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure CI for external repos (rocm-systems, rocm-libraries).

This script determines which projects changed based on git diff.

Note: Skip-CI logic (for docs-only changes) has been moved to
configure_multi_arch_ci.py, which uses TOML config files for skip patterns.
This allows external repos to define their own skip patterns.

Stage Reuse:
    TheRock's stage_reuse_decision.py handles stage impact analysis and
    baseline run selection using commit compatibility. The external repo
    workflow should:
    1. Resolve therock_ref using resolve_therock_ref.py (merge-base pinning)
    2. Pass that ref to setup_multi_arch.yml
    3. TheRock's stage_reuse_decision.py finds commit-compatible baseline runs

Usage:
    python configure_external_repo_ci.py \
        --event-name pull_request \
        --github-repo ROCm/rocm-libraries \
        --base-sha abc123 \
        --head-sha def456 \
        --config-path .github/repos-config.json

Outputs (to $GITHUB_OUTPUT):
    changed_projects: Comma-separated list of changed project paths. Empty when
                the run cannot be narrowed safely -- for schedule/dispatch,
                shared CI changes, workflow changes without a configured
                project scope, or an unclassifiable diff -- which downstream
                means "test every project".

Test depth (test_type) is not decided here: TheRock's
configure_multi_arch_ci.py derives it from the same diff, using the CI-plumbing
patterns in configure_ci_path_filters.py.
"""

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configure_ci_path_filters import determine_changed_projects

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])

# Patterns for files that don't require tests (docs, etc.)
SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "*CODEOWNERS",
    "docs/*",
    "projects/*/docs/*",
    "shared/*/docs/*",
]

# Full-test triggers, caller-workflow scoping and the non-subtree prefix list
# all live in configure_ci_path_filters.determine_changed_projects(), which
# configure_multi_arch_ci.py uses for the same decision.


@dataclass
class RepoEntry:
    """Repository entry from repos-config.json."""

    name: str
    url: str
    branch: str
    category: str
    auto_subtree_pull: bool = False
    auto_subtree_push: bool = False
    monorepo_source_of_truth: bool = False


def retry(
    max_attempts: int,
    delay_seconds: float,
    exceptions: Tuple[Type[BaseException], ...],
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""

    def decorator(func: F) -> F:
        def wrapper(*args: object, **kwargs: object) -> object:
            last_exception: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay_seconds * (2**attempt))
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


@retry(
    max_attempts=3,
    delay_seconds=2,
    exceptions=(subprocess.TimeoutExpired, subprocess.CalledProcessError),
)
def get_modified_paths_api(
    github_repo: str, base_sha: str, head_sha: str
) -> Optional[Set[str]]:
    """Get paths of files changed using GitHub API (compare endpoint).

    Returns None if the result is truncated (>300 files) to signal that the
    diff is incomplete and cannot be used to narrow the run.
    """
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{github_repo}/compare/{base_sha}...{head_sha}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    data = json.loads(result.stdout)
    files = data.get("files", [])
    # GitHub compare API returns max 300 files; a truncated diff is unusable.
    if len(files) >= 300:
        logger.warning("Compare API returned 300+ files, result may be truncated")
        return None
    return {f["filename"] for f in files}


def load_repo_config(config_path: str) -> List[RepoEntry]:
    """Load repository config from JSON."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter to known fields to handle future additions gracefully
        known_fields = {f.name for f in fields(RepoEntry)}
        entries = []
        for entry in data.get("repositories", []):
            filtered = {k: v for k, v in entry.items() if k in known_fields}
            entries.append(RepoEntry(**filtered))
        return entries
    except FileNotFoundError:
        logger.warning(f"Config not found: {config_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return []


def get_valid_prefixes(config: List[RepoEntry]) -> Set[str]:
    """Extract valid subtree prefixes from config."""
    return {f"{entry.category}/{entry.name}" for entry in config}


def set_github_output(outputs: Mapping[str, str]) -> None:
    """Write outputs to $GITHUB_OUTPUT or print if not in CI."""
    logger.info(f"Outputs: {dict(outputs)}")
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        for k, v in outputs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")


def configure(
    event_name: str,
    github_repo: str,
    base_sha: Optional[str],
    head_sha: Optional[str],
    config_path: str,
) -> str:
    """Main configuration logic.

    Returns the comma-separated project paths this run is scoped to. An empty
    string means the run could not be narrowed safely, which downstream means
    "test every project".

    Note: Skip-CI logic (for docs-only changes) is now handled by
    configure_multi_arch_ci.py using TOML config files. This function
    focuses on determining changed_projects.
    """

    # Schedule/workflow_dispatch events have no diff to narrow against.
    if event_name in ("schedule", "workflow_dispatch"):
        logger.info(f"{event_name} event has no diff - not scoping to projects")
        return ""

    # Get modified paths via GitHub API
    if event_name == "pull_request" and base_sha and head_sha:
        logger.info(f"Getting PR diff via API: {base_sha}...{head_sha}")
        modified_paths = get_modified_paths_api(github_repo, base_sha, head_sha)
    elif event_name == "push" and base_sha and head_sha:
        # For push, the caller passes github.event.before as base_sha;
        # the compare API does not understand git "^" ancestry syntax.
        logger.info(f"Getting push diff via API: {base_sha}...{head_sha}")
        modified_paths = get_modified_paths_api(github_repo, base_sha, head_sha)
    else:
        logger.warning("No SHAs provided - not scoping to projects")
        return ""

    # A truncated diff is an incomplete diff, so it cannot narrow the run.
    if modified_paths is None:
        logger.info("Truncated API response - not scoping to projects")
        return ""

    logger.info(f"Modified paths: {len(modified_paths)} files")

    # Attribution is TheRock's, shared with configure_multi_arch_ci.py so both
    # ends of the handoff classify a diff identically. This script only supplies
    # the inputs: the caller's diff and its subtree registry.
    return ",".join(
        determine_changed_projects(
            modified_paths,
            get_valid_prefixes(load_repo_config(config_path)),
            SKIPPABLE_PATH_PATTERNS,
        )
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure CI for external repos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--event-name",
        required=True,
        choices=["pull_request", "push", "schedule", "workflow_dispatch"],
        help="GitHub event name",
    )
    parser.add_argument(
        "--github-repo",
        required=True,
        help="GitHub repository (e.g., ROCm/rocm-libraries)",
    )
    parser.add_argument(
        "--base-sha",
        default="",
        help="Base SHA for PR (github.event.pull_request.base.sha)",
    )
    parser.add_argument(
        "--head-sha",
        default="",
        help="Head SHA for PR (github.event.pull_request.head.sha)",
    )
    parser.add_argument(
        "--config-path",
        default=".github/repos-config.json",
        help="Path to repos-config.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    logger.info(f"Configuring CI for {args.github_repo}")

    changed_projects = configure(
        event_name=args.event_name,
        github_repo=args.github_repo,
        base_sha=args.base_sha or None,
        head_sha=args.head_sha or None,
        config_path=args.config_path,
    )

    set_github_output({"changed_projects": changed_projects})

    return 0


if __name__ == "__main__":
    sys.exit(main())
