"""CI path filtering logic for determining whether to run CI based on modified files.

This module provides utilities to:
- Get modified file paths from git
- Filter paths based on skippable patterns (docs, markdown, etc.)
- Identify CI-related workflow files
- Decide whether CI should run based on the modified paths

Public API:
    get_git_modified_paths() - Get modified files from git diff compared to worktree
    get_git_submodule_paths() - Get list of git submodule paths in the repository
    is_ci_run_required() - Check if CI run is required based on modified paths
"""

import fnmatch
import subprocess
import sys
from typing import Iterable, Optional


# ============================================================================
# Public API
# ============================================================================


def get_git_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of files modified since the base reference commit.

    Uses `git diff --name-only` to find files that have changed between the
    base reference and the current working tree (including any uncommitted changes).

    Args:
        base_ref: Git reference (commit SHA, branch name, or HEAD^1) to compare against

    Returns:
        List of relative file paths that were modified, or None if the operation times out
    """
    try:
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except TimeoutError:
        print(
            "Computing modified files timed out. Not using PR diff to determine"
            " jobs to run.",
            file=sys.stderr,
        )
        return None


def get_git_submodule_paths(repo_root: Optional[str] = None) -> Optional[Iterable[str]]:
    """Returns the paths of git submodules in the repository.

    Uses `git submodule status` to list all submodules and extracts their paths.

    Args:
        repo_root: Path to the repository root directory. If None, uses current directory.

    Returns:
        List of relative paths to submodules, or empty list if the operation times out
    """
    try:
        response = subprocess.run(
            ["git", "submodule", "status"],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
            cwd=repo_root,
        ).stdout.splitlines()

        submodule_paths = []
        for line in response:
            submodule_data_array = line.split()
            # The line will be "{commit-hash} {path} {branch}". We will retrieve the path.
            if len(submodule_data_array) >= 2:
                submodule_paths.append(submodule_data_array[1])
        return submodule_paths
    except TimeoutError:
        print(
            "Computing submodule paths timed out.",
            file=sys.stderr,
        )
        return []


def is_ci_run_required(paths: Optional[Iterable[str]]) -> bool:
    """Checks if a CI run is required based on modified file paths.

    CI will run if:
    - At least one CI-related workflow file was modified, OR
    - At least one non-skippable file was modified

    CI will be skipped if:
    - No files were modified, OR
    - Only skippable files were modified (docs, markdown, etc.), OR
    - Only non-CI workflow files were modified that are skippable.

    Args:
        paths: Iterable of file paths to evaluate, or None if no files modified

    Returns:
        True if CI run is required, False if CI can be skipped
    """
    if paths is None:
        print("No files were modified, skipping build jobs")
        return False

    paths_set = set(paths)
    github_workflows_paths = set(
        [p for p in paths if p.startswith(".github/workflows")]
    )
    other_paths = paths_set - github_workflows_paths

    related_to_ci = _check_for_workflow_file_related_to_ci(github_workflows_paths)
    contains_other_non_skippable_files = _check_for_non_skippable_path(other_paths)

    print("is_ci_run_required findings:")
    print(f"  related_to_ci: {related_to_ci}")
    print(f"  contains_other_non_skippable_files: {contains_other_non_skippable_files}")

    if related_to_ci:
        print("Enabling build jobs since a related workflow file was modified")
        return True
    elif contains_other_non_skippable_files:
        print("Enabling build jobs since a non-skippable path was modified")
        return True
    else:
        print(
            "Only unrelated and/or skippable paths were modified, skipping build jobs"
        )
        return False


# ============================================================================
# Private Constants
# ============================================================================

# File path patterns that don't trigger CI runs.
# Changes to files matching these patterns are considered documentation/configuration
# that don't affect build or test workflows.
_SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*.pre-commit-config.*",
    ".github/dependabot.yml",
    "*CODEOWNERS",
    "*LICENSE",
    # Changes to dockerfiles do not currently affect CI workflows directly.
    # Docker images are built and published after commits are pushed, then
    # workflows can be updated to use the new image sha256 values.
    "dockerfiles/*",
    # Changes to experimental code do not run standard build/test workflows.
    "experimental/*",
]

# GitHub workflow file patterns that are considered CI-related.
# Changes to workflow files matching these patterns will trigger CI runs,
# as they may affect the CI pipeline itself.
_GITHUB_WORKFLOWS_CI_PATTERNS = [
    "setup.yml",
    "ci*.yml",
    "multi_arch*.yml",
    "build*artifact*.yml",
    "build*ci.yml",
    "build*python_packages.yml",
    "test*artifacts.yml",
    "test_rocm_wheels.yml",
    "test_sanity_check.yml",
    "test_component.yml",
]


# ============================================================================
# Private Helper Functions
# ============================================================================


def _is_path_skippable(path: str) -> bool:
    """Checks if a single file path matches any skippable pattern."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in _SKIPPABLE_PATH_PATTERNS)


def _check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Checks if any path in the collection is non-skippable.

    Returns True if at least one path doesn't match any skippable pattern.
    """
    if paths is None:
        return False
    return any(not _is_path_skippable(p) for p in paths)


def _is_path_workflow_file_related_to_ci(path: str) -> bool:
    """Checks if a single path is a CI-related workflow file."""
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in _GITHUB_WORKFLOWS_CI_PATTERNS
    )


def _check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    """Checks if any path in the collection is a CI-related workflow file.

    Returns True if at least one path matches a CI workflow pattern.
    """
    if paths is None:
        return False
    return any(_is_path_workflow_file_related_to_ci(p) for p in paths)
