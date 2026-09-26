# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CI path filtering logic for determining whether to run CI based on modified files.

This module provides utilities to:
- Get modified file paths from git
- Filter paths based on skippable patterns (docs, markdown, etc.)
- Identify CI-related workflow files
- Decide whether CI should run based on the modified paths

Skip-CI patterns are loaded from TOML config files:
- skip-ci-base.toml: Universal patterns that apply to all repos
- skip-ci-config.toml: Repo-specific extension patterns

Public API:
    get_git_commit_hash() - Resolve a git ref to a commit hash
    get_git_modified_paths() - Get modified files from git diff compared to worktree
    get_git_submodule_paths() - Get list of git submodule paths in the repository
    is_ci_run_required() - Check if CI run is required based on modified paths
    load_skip_ci_config() - Load skip-CI patterns from TOML config files
"""

import fnmatch
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Optional

# Try tomllib (Python 3.11+), fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore


_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# Directory containing this script (and the TOML configs)
_SCRIPT_DIR = Path(__file__).resolve().parent

# Base config with universal patterns (applies to all repos)
_BASE_CONFIG_PATH = _SCRIPT_DIR / "skip-ci-base.toml"

# TheRock extension config (repo-specific patterns)
_THEROCK_CONFIG_PATH = _SCRIPT_DIR / "skip-ci-config.toml"


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
# TOML Config Loading
# ============================================================================


def _load_toml_file(path: Path) -> Optional[dict]:
    """Load a TOML file and return its contents as a dict."""
    if not path.exists():
        print(f"  Config file not found: {path}")
        return None

    if tomllib is None:
        print("  Warning: tomllib not available, cannot read TOML config")
        return None

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"  Warning: Failed to parse TOML config {path}: {e}")
        return None


def _extract_skip_patterns_from_config(config: dict) -> list[str]:
    """Extract skip patterns from a parsed TOML config dict."""
    skip_ci = config.get("skip_ci", {})
    patterns: list[str] = []

    # Common patterns apply to all platforms
    common = skip_ci.get("common", [])
    if isinstance(common, list):
        patterns.extend(common)

    # Platform-specific patterns (based on RUNNER_OS)
    runner_os = os.environ.get("RUNNER_OS", "").lower()
    if runner_os == "linux":
        linux_patterns = skip_ci.get("linux", [])
        if isinstance(linux_patterns, list):
            patterns.extend(linux_patterns)
    elif runner_os == "windows":
        windows_patterns = skip_ci.get("windows", [])
        if isinstance(windows_patterns, list):
            patterns.extend(windows_patterns)

    return patterns


def _extract_ci_workflow_filenames(config: dict) -> set[str]:
    """Extract CI workflow filenames from a parsed TOML config dict."""
    ci_workflows = config.get("ci_workflows", {})
    filenames = ci_workflows.get("filenames", [])
    if isinstance(filenames, list):
        return set(filenames)
    return set()


def load_skip_ci_config(
    extension_config_path: Optional[Path] = None,
) -> tuple[list[str], set[str]]:
    """Load skip-CI patterns and CI workflow filenames from TOML configs.

    Loads patterns from:
    1. Base config (skip-ci-base.toml) - universal patterns for all repos
    2. Extension config - repo-specific patterns

    Args:
        extension_config_path: Path to the extension config file.
            If None, uses TheRock's skip-ci-config.toml.

    Returns:
        Tuple of (skip_patterns, ci_workflow_filenames):
        - skip_patterns: Combined list of glob patterns for skippable files
        - ci_workflow_filenames: Set of CI workflow filenames that trigger CI
    """
    patterns: list[str] = []
    ci_filenames: set[str] = set()

    # Load base config (universal patterns)
    base_config = _load_toml_file(_BASE_CONFIG_PATH)
    if base_config:
        base_patterns = _extract_skip_patterns_from_config(base_config)
        patterns.extend(base_patterns)
        print(f"  Loaded {len(base_patterns)} base skip-CI patterns")

    # Load extension config (repo-specific patterns)
    ext_path = extension_config_path or _THEROCK_CONFIG_PATH
    ext_config = _load_toml_file(ext_path)
    if ext_config:
        ext_patterns = _extract_skip_patterns_from_config(ext_config)
        patterns.extend(ext_patterns)
        ci_filenames = _extract_ci_workflow_filenames(ext_config)
        print(f"  Loaded {len(ext_patterns)} extension skip-CI patterns")
        if ci_filenames:
            print(f"  Loaded {len(ci_filenames)} CI workflow filenames")

    return patterns, ci_filenames


# Cached config loaded at module import time for TheRock
_cached_skip_patterns: Optional[list[str]] = None
_cached_ci_workflow_filenames: Optional[set[str]] = None


def _get_therock_config() -> tuple[list[str], set[str]]:
    """Get TheRock's skip-CI config, loading and caching on first access."""
    global _cached_skip_patterns, _cached_ci_workflow_filenames
    if _cached_skip_patterns is None:
        _cached_skip_patterns, _cached_ci_workflow_filenames = load_skip_ci_config()
    return _cached_skip_patterns, _cached_ci_workflow_filenames or set()


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


def get_git_modified_paths(
    base_ref: str, cwd: Optional[str] = None
) -> Optional[Iterable[str]]:
    """Returns the paths of files modified since the base reference commit.

    Uses `git diff --name-only` to find files that have changed between the
    base reference and the current working tree (including any uncommitted changes).

    Args:
        base_ref: Git reference (commit SHA, branch name, or HEAD^1) to compare against
        cwd: Working directory for git commands. If None, uses current directory.

    Returns:
        List of relative file paths that were modified, or None if the operation times out
    """
    cwd_msg = f" (in {cwd})" if cwd else ""
    print(f"Computing modified paths with: 'git diff --name-only {base_ref}'{cwd_msg}")
    try:
        # Push events can advance a branch by multiple commits. The setup
        # checkout is intentionally shallow, so event.before may be older than
        # the fetched history even though it is a valid reachable commit.
        # Note: _ensure_git_commit_available only works in the current repo,
        # for external repos we rely on the checkout having sufficient depth.
        if cwd is None:
            _ensure_git_commit_available(base_ref)

        # We have the commit, now run the diff.
        return subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
            cwd=cwd,
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


def is_ci_run_required(
    paths: Optional[Iterable[str]],
    skip_patterns: Optional[list[str]] = None,
    repo_name: str = "",
) -> bool:
    """Checks if a CI run is required based on modified file paths.

    CI will run if:
    - At least one CI-related workflow file was modified (TheRock only), OR
    - At least one non-skippable file was modified

    CI will be skipped if:
    - No files were modified, OR
    - Only skippable files were modified (docs, markdown, etc.)

    Args:
        paths: Iterable of file paths to evaluate, or None if no files modified
        skip_patterns: Optional list of glob patterns for skippable files.
            If provided (external repo case), uses these patterns.
            If None (TheRock case), loads patterns from TOML configs.
        repo_name: Optional repo name for logging (e.g., "rocm-libraries")

    Returns:
        True if CI run is required, False if CI can be skipped
    """
    label = f"[{repo_name}] " if repo_name else ""

    if paths is None:
        print(f"{label}No changed files provided, CI required (conservative)")
        return True

    paths_list = list(paths)
    if not paths_list:
        print(f"{label}No files changed, skipping CI")
        return False

    # Determine which patterns to use
    use_external_patterns = skip_patterns is not None

    if use_external_patterns:
        # External repo: use provided patterns (already includes base patterns)
        effective_patterns = skip_patterns
        ci_workflow_filenames: set[str] = set()
    else:
        # TheRock: load from TOML configs
        effective_patterns, ci_workflow_filenames = _get_therock_config()

    def is_skippable(path: str) -> bool:
        return _is_path_skippable_for_patterns(path, effective_patterns)

    # For TheRock, also check CI workflow files
    if ci_workflow_filenames:
        github_workflows_paths = [
            p for p in paths_list if p.startswith(".github/workflows")
        ]
        other_paths = [p for p in paths_list if not p.startswith(".github/workflows")]

        related_to_ci = _check_for_workflow_file_related_to_ci(
            github_workflows_paths, ci_workflow_filenames
        )
        contains_non_skippable = any(not is_skippable(p) for p in other_paths)

        print(f"{label}is_ci_run_required findings:")
        print(f"  related_to_ci: {related_to_ci}")
        print(f"  contains_non_skippable: {contains_non_skippable}")

        if related_to_ci:
            print(f"{label}CI required: CI-related workflow file modified")
            return True
        elif contains_non_skippable:
            print(f"{label}CI required: non-skippable file modified")
            return True
        else:
            print(f"{label}All files skippable, CI can be skipped")
            return False

    # External repo case (or TheRock without ci_workflows): just check against skip_patterns
    non_skippable = [f for f in paths_list if not is_skippable(f)]

    print(f"{label}Evaluating {len(paths_list)} changed file(s):")
    for f in paths_list[:10]:
        skippable = is_skippable(f)
        print(f"  {'[skip]' if skippable else '[ci]  '} {f}")
    if len(paths_list) > 10:
        print(f"  ... and {len(paths_list) - 10} more")

    if non_skippable:
        print(f"{label}CI required: {len(non_skippable)} non-skippable file(s)")
        return True
    else:
        print(f"{label}All files skippable, CI can be skipped")
        return False


# ============================================================================
# Private Helper Functions
# ============================================================================


def _is_path_skippable_for_patterns(path: str, patterns: list[str]) -> bool:
    """Checks if a file path matches any of the provided skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _check_for_workflow_file_related_to_ci(
    paths: Optional[Iterable[str]], ci_workflow_filenames: set[str]
) -> bool:
    """Checks if any path in the collection is a CI-related workflow file.

    Returns True if at least one path matches a CI workflow pattern.
    """
    if paths is None:
        return False
    return any(Path(p).name in ci_workflow_filenames for p in paths)


def get_ci_workflow_filenames() -> set[str]:
    """Get the set of CI workflow filenames from TheRock's config.

    This is primarily for testing to verify the workflow list is complete.
    """
    _, ci_filenames = _get_therock_config()
    return ci_filenames
