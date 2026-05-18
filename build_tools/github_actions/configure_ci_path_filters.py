# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CI path filtering logic for determining whether to run CI based on modified files.

This module provides utilities to:
- Get modified file paths from git
- Filter paths based on skippable patterns (docs, markdown, etc.)
- Identify CI-related workflow files
- Decide whether CI should run based on the modified paths

Public API:
    get_git_commit_hash() - Resolve a git ref to a commit hash
    get_git_modified_paths() - Get modified files from git diff compared to worktree
    get_git_submodule_paths() - Get list of git submodule paths in the repository
    is_ci_run_required() - Check if CI run is required based on modified paths
"""

import fnmatch
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Optional


_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _ensure_git_commit_available(ref: str) -> None:
    """Fetch a full SHA when it is missing from the shallow checkout."""
    if _FULL_GIT_SHA_RE.fullmatch(ref) is None:
        return

    is_commit_available_locally = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
        ).returncode
        == 0
    )
    if is_commit_available_locally:
        return

    print(f"Commit {ref} is not available locally. Fetching it...")
    subprocess.run(
        [
            "git",
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--depth=1",
            "origin",
            ref,
        ],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    )


# ============================================================================
# Public API
# ============================================================================


def get_git_commit_hash(ref: str) -> str:
    """Resolve a git ref to its full commit hash."""
    _ensure_git_commit_available(ref)
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def get_git_modified_paths(base_ref: str) -> Optional[Iterable[str]]:
    """Returns the paths of files modified since the base reference commit.

    Uses `git diff --name-only` to find files that have changed between the
    base reference and the current working tree (including any uncommitted changes).

    Args:
        base_ref: Git reference (commit SHA, branch name, or HEAD^1) to compare against

    Returns:
        List of relative file paths that were modified, or None if the operation times out
    """
    print(f"Computing modified paths with: 'git diff --name-only {base_ref}'")
    try:
        # Push events can advance a branch by multiple commits. The setup
        # checkout is intentionally shallow, so event.before may be older than
        # the fetched history even though it is a valid reachable commit.
        _ensure_git_commit_available(base_ref)

        # We have the commit, now run the diff.
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
        ).stdout.splitlines()
    except subprocess.TimeoutExpired:
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
# Changes matching these patterns shouldn't affect CI build/test workflows.
_SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*.mdc",
    "*.pre-commit-config.*",
    ".github/dependabot.yml",
    "*CODEOWNERS",
    "*LICENSE",
    # Files used by gitleaks, no impact on CI.
    "gitleaks.toml",
    "build_tools/scan_tools/*",
    # Changes to dockerfiles do not currently affect CI workflows directly.
    # Docker images are built and published after commits are pushed, then
    # workflows can be updated to use the new image sha256 values.
    "dockerfiles/*",
    # Changes to experimental code do not run standard build/test workflows.
    "experimental/*",
    # This directory contains a collection of AI skills and other developer
    # utilities. It includes some Python scripts, none of which affect CI.
    "skills/*",
]

# GitHub workflow files that are used by CI workflows. Changes to any of
# these trigger CI runs. This is an explicit list (not glob patterns) so that
# unrelated workflows are never accidentally included.
#
# This list is maintained manually and a unit test verifies that every
# workflow transitively called by multi_arch_ci.yml is listed here.
#
# TODO: Compute this set dynamically by scanning workflow files for
# ``uses: ./.github/workflows/...`` references instead of maintaining it
# by hand.
_GITHUB_WORKFLOWS_CI_FILENAMES = {
    "build_portable_linux_python_packages.yml",
    "build_windows_python_packages.yml",
    "manifest-diff.yml",
    "multi_arch_build_native_linux_packages.yml",
    "multi_arch_build_portable_linux_artifacts.yml",
    "multi_arch_build_portable_linux_pytorch_wheels_ci.yml",
    "multi_arch_build_linux_jax_wheels_ci.yml",
    "multi_arch_build_portable_linux.yml",
    "multi_arch_build_windows_artifacts.yml",
    "multi_arch_build_windows_pytorch_wheels_ci.yml",
    "multi_arch_build_windows.yml",
    "multi_arch_build_wsl_rocdxg_artifacts.yml",
    "multi_arch_ci_linux.yml",
    "build_opencl_cts.yml",
    "multi_arch_ci_windows.yml",
    "multi_arch_ci.yml",
    "setup_multi_arch.yml",
    "test_artifacts_structure.yml",
    "test_artifacts.yml",
    "test_component.yml",
    "test_native_linux_packages_install.yml",
    "test_rocm_wheels.yml",
}


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
    return Path(path).name in _GITHUB_WORKFLOWS_CI_FILENAMES


def _check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    """Checks if any path in the collection is a CI-related workflow file.

    Returns True if at least one path matches a CI workflow pattern.
    """
    if paths is None:
        return False
    return any(_is_path_workflow_file_related_to_ci(p) for p in paths)
