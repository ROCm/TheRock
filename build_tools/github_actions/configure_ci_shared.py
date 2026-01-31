#!/usr/bin/env python3

"""Shared utilities for CI configuration.

This module contains utilities shared by both TheRock CI (configure_ci.py) and
external repository CI (configure_ci_external_repos.py).

Shared utilities include:
- Path modification detection (git diff)
- Path pattern matching for skippable files
- Common environment variable names
"""

import fnmatch
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from detect_external_repo_config import get_external_repo_path

# Get TheRock directory
THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

# Environment variable names (centralized as they may change or be used across modules)
ENV_EXTERNAL_SOURCE_PATH = "EXTERNAL_SOURCE_PATH"
ENV_PROJECTS = "PROJECTS"
ENV_GITHUB_REPOSITORY = "GITHUB_REPOSITORY"
ENV_GITHUB_REPOSITORY_OVERRIDE = "GITHUB_REPOSITORY_OVERRIDE"

# TheRock's default skippable path patterns
# Paths matching any of these patterns are considered to have no influence over
# build or test workflows so any related jobs can be skipped if all paths
# modified by a commit/PR match a pattern in this list.
THEROCK_SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*.pre-commit-config.*",
    ".github/dependabot.yml",
    "*CODEOWNERS",
    "*LICENSE",
    "external-builds/*",
    "dockerfiles/*",
    "experimental/*",
]


# --------------------------------------------------------------------------- #
# Path utility functions
# --------------------------------------------------------------------------- #


def get_modified_paths(
    base_ref: str, repo_name: Optional[str] = None
) -> Optional[Iterable[str]]:
    """Returns the paths of modified files relative to the base reference.

    Args:
        base_ref: Base git ref to diff against
        repo_name: Optional external repository name. If provided, attempts to get
                   the external repo path for running git diff.

    Returns:
        List of modified file paths, or None if determination fails
    """
    git_cwd = None

    # For external repos, use proper path resolution
    if repo_name:
        try:
            git_cwd = get_external_repo_path(repo_name)
            print(f"Running git diff from external repo at: {git_cwd}", file=sys.stderr)
        except ValueError as e:
            print(f"Could not determine external repo path: {e}", file=sys.stderr)
            # Fall back to EXTERNAL_SOURCE_PATH or current directory
            external_source_path = os.environ.get(ENV_EXTERNAL_SOURCE_PATH, "")
            if external_source_path:
                # Handle both absolute and relative paths properly
                external_path = Path(external_source_path)
                if external_path.is_absolute():
                    git_cwd = external_path
                else:
                    git_cwd = THEROCK_DIR / external_path
                print(
                    f"Falling back to EXTERNAL_SOURCE_PATH: {git_cwd}", file=sys.stderr
                )

    try:
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
            cwd=git_cwd,
        ).stdout.splitlines()
    except TimeoutError:
        print(
            "Computing modified files timed out. Not using PR diff to determine"
            " jobs to run.",
            file=sys.stderr,
        )
        return None


def has_non_skippable_paths(paths: Iterable[str], skip_patterns: list[str]) -> bool:
    """Check if any paths don't match the skip patterns.

    Args:
        paths: List of file paths to check
        skip_patterns: Custom patterns to check against

    Returns:
        True if at least one path doesn't match any skip pattern
    """
    return any(
        not any(fnmatch.fnmatch(path, pattern) for pattern in skip_patterns)
        for path in paths
    )
