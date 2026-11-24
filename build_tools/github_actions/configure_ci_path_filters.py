# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CI path filtering logic for determining whether to run CI based on modified files.

This module provides utilities to:
- Get modified file paths from git
- Filter paths based on skippable patterns (docs, markdown, etc.)
- Identify CI-related workflow files
- Decide whether CI should run based on the modified paths
- Identify diffs confined to an external repo's TheRock CI plumbing
- Attribute an external repo's diff to the projects it changed

Skip-CI patterns are loaded from TOML config files:
- skip-ci-base.toml: Universal patterns that apply to all repos
- skip-ci-config.toml: Repo-specific extension patterns

Public API:
    get_git_commit_hash() - Resolve a git ref to a commit hash
    get_git_modified_paths() - Get modified files from git diff compared to worktree
    get_git_submodule_paths() - Get list of git submodule paths in the repository
    get_changed_files_for_external_repo() - Get changed files for external repos via API/git
    is_ci_run_required() - Check if CI run is required based on modified paths
    get_jobs_for_changes() - Identify job groups directly affected by changed paths
    is_external_repo_ci_only_change() - Check if an external repo diff is confined to TheRock CI plumbing
    determine_changed_projects() - Scope an external repo run to the projects its diff changed
    load_skip_ci_config() - Load skip-CI patterns from TOML config files
    load_skip_ci_patterns_for_external_repo() - Load skip-CI patterns for external repos
"""

import fnmatch
from functools import cache
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


@cache
def _get_therock_config() -> tuple[list[str], set[str]]:
    """Get TheRock's skip-CI config, cached on first access."""
    return load_skip_ci_config()


@cache
def _get_job_paths() -> dict[str, list[str]]:
    """Get TheRock's job ownership patterns, cached on first access."""
    config = _load_toml_file(_THEROCK_CONFIG_PATH)
    return config.get("job_paths", {}) if config else {}


# Default path where external repo config is checked out in setup_multi_arch.yml
EXTERNAL_REPO_CONFIG_DIR = "external-repo-config"


def load_skip_ci_patterns_for_external_repo(config_path: str) -> list[str] | None:
    """Load skip CI patterns from base config + external repo's TOML config file.

    This function is used by configure_multi_arch_ci.py to load skip-CI patterns
    for external repositories (like rocm-libraries, rocm-systems) that call into
    TheRock's CI.

    Loads patterns from:
    1. Base config (skip-ci-base.toml in TheRock) - universal patterns for all repos
    2. External repo's extension config - repo-specific patterns

    The external repo config file is expected to be checked out to the
    external-repo-config/ directory by setup_multi_arch.yml.

    Args:
        config_path: Relative path to the skip-CI config file within the external
            repo (e.g., ".github/skip-ci-config.toml")

    Returns:
        Combined list of skip patterns (base + extension), or None if no patterns
        could be loaded.
    """
    # The external repo config is checked out to external-repo-config/
    full_path = Path(EXTERNAL_REPO_CONFIG_DIR) / config_path
    if not full_path.exists():
        print(f"  Skip CI config not found: {full_path}")
        # Still load base patterns even if extension config is missing
        base_patterns, _ = load_skip_ci_config(extension_config_path=full_path)
        if base_patterns:
            print(f"  Using {len(base_patterns)} base skip-CI patterns only")
            return base_patterns
        return None

    # Load base + extension patterns using the shared loader
    patterns, _ = load_skip_ci_config(extension_config_path=full_path)
    print(f"  Loaded {len(patterns)} total skip CI patterns (base + {config_path})")
    return patterns


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

    def is_relevant(path: str) -> bool:
        return not _is_path_unrelated_to_ci(
            path, effective_patterns, ci_workflow_filenames
        )

    relevant_paths = [path for path in paths_list if is_relevant(path)]

    print(f"{label}Evaluating {len(paths_list)} changed file(s):")
    for f in paths_list[:10]:
        print(f"  {'[ci]  ' if is_relevant(f) else '[skip]'} {f}")
    if len(paths_list) > 10:
        print(f"  ... and {len(paths_list) - 10} more")

    if relevant_paths:
        print(f"{label}CI required: {len(relevant_paths)} CI-relevant file(s)")
        return True
    else:
        print(f"{label}All files skippable, CI can be skipped")
        return False


def get_jobs_for_changes(paths: Iterable[str] | None) -> set[str]:
    """Return job groups directly affected by CI-relevant changes.

    Job names come from [job_paths] in skip-ci-config.toml: pytorch, jax,
    rocm_python, rocm_native_linux, and rocm_native_windows. Unmapped,
    CI-relevant paths add rocm. Matches accumulate across paths, for example:

    * {"pytorch"}: only PyTorch-owned paths are relevant to CI.
    * {"pytorch", "jax"}: both frameworks are directly affected.
    * {"pytorch", "rocm"}: PyTorch and shared ROCm paths changed.
    * {"rocm_python"}: only ROCm Python packaging paths changed.
    * set(): no paths changed, or all changed paths are unrelated to this CI.

    If paths is None (diff unavailable), return all configured groups plus rocm
    conservatively. These are direct impacts, not the full set of jobs to run;
    dependency requirements, downstream coverage, and overrides are handled by
    job selection.
    """
    job_paths = _get_job_paths()
    if paths is None:
        return {"rocm", *job_paths}

    skip_patterns, ci_workflow_filenames = _get_therock_config()
    jobs: set[str] = set()
    for path in paths:
        if _is_path_unrelated_to_ci(path, skip_patterns, ci_workflow_filenames):
            continue

        affected = {
            job
            for job, patterns in job_paths.items()
            if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
        }
        if affected:
            jobs.update(affected)
        else:
            jobs.add("rocm")
    return jobs


def get_ci_workflow_filenames() -> set[str]:
    """Get the set of CI workflow filenames from TheRock's config.

    This is primarily for testing to verify the workflow list is complete.
    """
    _, ci_filenames = _get_therock_config()
    return ci_filenames


def get_modified_paths_via_api(
    github_repo: str,
    base_sha: str,
    head_sha: str,
    pr_number: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[list[str]]:
    """Get paths of files changed using GitHub API.

    For pull requests (when pr_number is provided), uses the PR files endpoint
    which correctly computes the diff against the merge-base. This avoids issues
    with stale base SHAs when the target branch has been updated.

    For non-PR events, falls back to the compare endpoint using base_sha/head_sha.

    Args:
        github_repo: Repository in "owner/repo" format (e.g., "ROCm/rocm-libraries")
        base_sha: Base commit SHA to compare from (used for non-PR events)
        head_sha: Head commit SHA to compare to (used for non-PR events)
        pr_number: Pull request number (if this is a PR event)
        max_retries: Maximum number of retry attempts for transient failures
        retry_delay: Initial delay between retries (doubles with each retry)

    Returns:
        List of changed file paths, or None if:
        - The result is truncated (>300 files for compare, >3000 for PR files)
        - API call fails after retries
    """
    import json
    import time

    # Use PR files endpoint for pull requests - it correctly handles merge-base
    if pr_number:
        return _get_pr_files_via_api(github_repo, pr_number, max_retries, retry_delay)

    # Fall back to compare endpoint for non-PR events (push, etc.)
    for attempt in range(max_retries):
        try:
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

            # GitHub compare API returns max 300 files; if truncated, signal caller
            if len(files) >= 300:
                print(
                    f"  GitHub compare API returned 300+ files, result may be truncated"
                )
                return None

            return [f["filename"] for f in files]

        except subprocess.TimeoutExpired:
            print(
                f"  GitHub API request timed out (attempt {attempt + 1}/{max_retries})"
            )
        except subprocess.CalledProcessError as e:
            print(
                f"  GitHub API request failed (attempt {attempt + 1}/{max_retries}): "
                f"{e.stderr or e}"
            )
        except json.JSONDecodeError as e:
            print(f"  Failed to parse GitHub API response: {e}")
            return None

        if attempt < max_retries - 1:
            delay = retry_delay * (2**attempt)
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)

    print("  GitHub API request failed after all retries")
    return None


def _get_pr_files_via_api(
    github_repo: str,
    pr_number: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[list[str]]:
    """Get changed files for a PR using the PR files endpoint.

    The PR files endpoint (/pulls/{number}/files) correctly computes the diff
    against the merge-base, unlike the compare endpoint which can include
    unrelated commits when the base branch has been updated.

    Args:
        github_repo: Repository in "owner/repo" format
        pr_number: Pull request number
        max_retries: Maximum retry attempts
        retry_delay: Initial delay between retries

    Returns:
        List of changed file paths, or None if truncated/failed.
    """
    import json
    import time

    all_files: list[str] = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API

    for attempt in range(max_retries):
        try:
            # Paginate through all files (PR files endpoint supports pagination)
            while True:
                # Use query parameters in URL for GET request (not -f which is for POST)
                result = subprocess.run(
                    [
                        "gh",
                        "api",
                        f"repos/{github_repo}/pulls/{pr_number}/files?per_page={per_page}&page={page}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                files = json.loads(result.stdout)

                if not files:
                    break

                all_files.extend(f["filename"] for f in files)

                # PR files endpoint returns max 3000 files total
                if len(all_files) >= 3000:
                    print(
                        f"  GitHub PR files API returned 3000+ files, result may be truncated"
                    )
                    return None

                # If we got fewer than per_page, we've reached the end
                if len(files) < per_page:
                    break

                page += 1

            return all_files

        except subprocess.TimeoutExpired:
            print(
                f"  GitHub API request timed out (attempt {attempt + 1}/{max_retries})"
            )
        except subprocess.CalledProcessError as e:
            print(
                f"  GitHub API request failed (attempt {attempt + 1}/{max_retries}): "
                f"{e.stderr or e}"
            )
        except json.JSONDecodeError as e:
            print(f"  Failed to parse GitHub API response: {e}")
            return None

        if attempt < max_retries - 1:
            delay = retry_delay * (2**attempt)
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)
            # Reset pagination for retry
            all_files = []
            page = 1

    print("  GitHub API request failed after all retries")
    return None


# ============================================================================
# Private Helper Functions
# ============================================================================


def _matches_pattern(path: str, patterns: Iterable[str]) -> bool:
    """Whether a path matches any of the given glob patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_path_unrelated_to_ci(
    path: str, skip_patterns: list[str], ci_workflow_filenames: set[str]
) -> bool:
    """Check skip patterns and exclude workflows outside the CI allowlist.

    An empty allowlist disables this exclusion, as used by external repositories
    or when the allowlist is unavailable.
    """
    return any(fnmatch.fnmatch(path, pattern) for pattern in skip_patterns) or (
        bool(ci_workflow_filenames)
        and path.startswith(".github/workflows/")
        and Path(path).name not in ci_workflow_filenames
    )


def get_changed_files_for_external_repo(
    external_repo: dict,
) -> Optional[list[str]]:
    """Get list of changed files for an external repository.

    Determines changed files via GitHub API (preferred) or git diff (fallback).
    The method used depends on the information available in external_repo:

    1. For pull requests (pr_number available): Uses PR files endpoint
    2. For push events (base_sha + head_sha): Uses compare endpoint
    3. Fallback: Uses local git diff if checkout exists

    Args:
        external_repo: Dict containing repository info with optional keys:
            - repository: Full repo name (e.g., "ROCm/rocm-libraries")
            - event_name: GitHub event type (push, pull_request, etc.)
            - pr_number: Pull request number (for PR events)
            - base_sha: Base commit SHA
            - head_sha: Head commit SHA
            - base_ref: Base ref for git diff fallback

    Returns:
        List of changed file paths, or None if:
        - Event type is schedule/workflow_dispatch (no meaningful diff)
        - API/git diff fails or returns truncated results
    """
    repo_full_name = external_repo.get("repository", "")
    repo_name = repo_full_name.split("/")[-1]
    event_name = external_repo.get("event_name", "")
    base_sha = external_repo.get("base_sha")
    head_sha = external_repo.get("head_sha")
    pr_number = external_repo.get("pr_number")

    # Schedule/workflow_dispatch events don't have a meaningful diff
    if event_name in ("schedule", "workflow_dispatch"):
        print(f"  External repo {repo_name}: {event_name} event, no diff available")
        return None

    # Try PR files endpoint first (best for PRs - handles merge-base correctly)
    if pr_number and repo_full_name:
        print(f"  External repo {repo_name}: fetching changed files via GitHub API...")
        print(f"    Using PR #{pr_number} files endpoint")
        changed_files = get_modified_paths_via_api(
            repo_full_name, base_sha or "", head_sha or "", pr_number=pr_number
        )
        if changed_files is not None:
            print(f"  External repo {repo_name}: {len(changed_files)} file(s) changed")
        else:
            print(f"  External repo {repo_name}: API returned truncated/failed result")
        return changed_files

    # Fall back to compare API for non-PR events (push, etc.)
    if base_sha and head_sha and repo_full_name:
        print(f"  External repo {repo_name}: fetching changed files via GitHub API...")
        print(f"    Comparing {base_sha[:12]}...{head_sha[:12]}")
        changed_files = get_modified_paths_via_api(repo_full_name, base_sha, head_sha)
        if changed_files is not None:
            print(f"  External repo {repo_name}: {len(changed_files)} file(s) changed")
        else:
            print(f"  External repo {repo_name}: API returned truncated/failed result")
        return changed_files

    # Fallback to git diff if SHAs not provided (legacy path)
    external_repo_path = Path(EXTERNAL_REPO_CONFIG_DIR)
    base_ref = external_repo.get("base_ref")
    if external_repo_path.exists() and external_repo_path.is_dir():
        if not base_ref:
            base_ref = "HEAD^"
        print(f"  External repo {repo_name}: computing changed files via git diff...")
        changed_files = list(
            get_git_modified_paths(base_ref, cwd=str(external_repo_path)) or []
        )
        print(f"  External repo {repo_name}: {len(changed_files)} file(s) changed")
        return changed_files

    print(f"  External repo {repo_name}: no SHAs provided and checkout not found")
    return None


# ============================================================================
# External repo CI-plumbing detection
# ============================================================================

# Workflows in a calling repository (rocm-libraries, rocm-systems) that invoke
# TheRock's reusable workflows. They configure *how* a run happens rather than
# changing any library source, and they belong to no single project.
#
# They do not, however, belong to any project *outside* the calling repo either,
# so a diff confined to them is scoped to that repo's own configured subtrees
# rather than to every project TheRock builds. See determine_changed_projects().
EXTERNAL_REPO_CALLER_WORKFLOW_PATTERNS = [
    ".github/workflows/therock*",
    # Pins the TheRock ref, Docker image and runner labels.
    ".github/actions/ci-env/*",
]

# CI plumbing that decides *which* tests run: project discovery, the subtree
# registry, and the CTest categorization every project's tests consume. A change
# here can alter test selection for every project at once, so unlike a caller
# workflow it keeps full coverage -- the scope it would narrow to is itself what
# the change rewrites.
EXTERNAL_REPO_TEST_SELECTION_PATTERNS = [
    ".github/scripts/therock*",
    ".github/scripts/get_changed_projects.py",
    ".github/scripts/ci_utils.py",
    ".github/scripts/config_loader.py",
    ".github/scripts/repo_config_model.py",
    ".github/scripts/pr_detect_changed_subtrees.py",
    ".github/repos-config.json",
    "shared/ctest/*",
]

# Every path that is TheRock CI plumbing rather than library source. Test
# *depth* reads this whole set via is_external_repo_ci_only_change(); test
# *breadth* distinguishes the two halves in determine_changed_projects().
EXTERNAL_REPO_CI_PATTERNS = (
    EXTERNAL_REPO_CALLER_WORKFLOW_PATTERNS + EXTERNAL_REPO_TEST_SELECTION_PATTERNS
)


def is_external_repo_ci_only_change(
    changed_files: Optional[Iterable[str]],
    skip_patterns: Optional[list[str]] = None,
) -> bool:
    """Whether an external repo's diff touches *only* TheRock CI plumbing.

    CI-plumbing changes build no new library code, so a diff confined to them
    can be tested at quick depth, mirroring TheRock's own CI-workflow changes.
    The test is deliberately strict: a diff that also edits library source
    returns False, so that source keeps its normal (standard) depth. Such a
    change is CI-*affecting* but not CI-*only*. Breadth for that case is decided
    by determine_changed_projects(); widening breadth is not a reason to shrink
    depth.

    Args:
        changed_files: Paths changed in the calling repository, or None when
            the diff could not be determined.
        skip_patterns: Skip-CI patterns from the calling repo's TOML config.
            Paths matching these (docs, markdown) are ignored, so a workflow
            edit that also touches a README still qualifies as CI-only.

    Returns:
        True only when at least one CI-relevant path changed and every such
        path matches EXTERNAL_REPO_CI_PATTERNS.
    """
    if not changed_files:
        return False
    # An empty workflow allowlist disables the TheRock-only workflow exclusion,
    # matching how is_ci_run_required() treats external repositories.
    relevant_paths = [
        path
        for path in changed_files
        if not _is_path_unrelated_to_ci(path, skip_patterns or [], set())
    ]
    if not relevant_paths:
        return False
    return all(
        _matches_pattern(path, EXTERNAL_REPO_CI_PATTERNS) for path in relevant_paths
    )


# ============================================================================
# External repo project scoping
# ============================================================================

# CI-relevant monorepo directories that are NOT subtree-synced repos and so are
# absent from a calling repo's repos-config.json. Without these, a diff confined
# to one of them matches no subtree, which would fall back to testing every
# project. Each entry MUST have a corresponding mapping in TheRock (a
# BUILD_TOPOLOGY.toml alias plus _EXTERNAL_SUBTREE_ALIASES in
# determine_rocm_test_dependencies.py), so keep this list in lock-step when
# adding directories.
CI_RELEVANT_NON_SUBTREE_PREFIXES = {
    # hipCCL is monorepo-native (projects/hipccl/hipccl3, the unified
    # rocPRIM+hipCUB+rocThrust superbuild); it has no upstream repo to
    # subtree-sync. Resolves to the hipccl graph node -> prim test jobs.
    "projects/hipccl",
    "dnn-providers/cmake",
    "shared/amdgpu-windows-interop",
    "shared/kpack",
    "shared/machine-readable-isa",
    "shared/primbench",
    "emulation/mirage",
    "emulation/rocjitsu",
}


def find_matched_subtrees(paths: Iterable[str], prefixes: set[str]) -> list[str]:
    """Attribute each path to the longest registered prefix containing it.

    Longest-prefix-first, rather than a fixed 2-segment truncation, so a nested
    subtree in repos-config.json (`hipblaslt/tensilelite`) stays distinct from
    its parent (`hipblaslt`) instead of both collapsing to the same prefix.
    """
    # Longest prefixes first, so a nested subtree wins over its parent.
    by_specificity = sorted(prefixes, key=lambda p: p.count("/"), reverse=True)
    matched: set[str] = set()
    for path in paths:
        segments = path.split("/")
        for prefix in by_specificity:
            prefix_segments = prefix.split("/")
            if segments[: len(prefix_segments)] == prefix_segments:
                matched.add(prefix)
                break
    return sorted(matched)


def determine_changed_projects(
    changed_files: Optional[Iterable[str]],
    subtree_prefixes: set[str],
    skip_patterns: Optional[list[str]] = None,
) -> list[str]:
    """Projects an external repo's diff changed, scoping which tests run.

    Decides test *breadth*, the companion to is_external_repo_ci_only_change()'s
    *depth*; both read the same diff and skip patterns, so they agree on which
    paths are docs-only noise. subtree_prefixes comes from the caller's
    repos-config.json via external_repo_config.subtree_prefixes().

    Returns the prefixes to scope the run to. Empty means the diff could not be
    narrowed safely, which downstream means "test every project".
    """
    if not changed_files:
        print("  Project scope: no diff available, testing every project")
        return []

    relevant_paths = [
        path
        for path in changed_files
        if not _is_path_unrelated_to_ci(path, skip_patterns or [], set())
    ]
    if not relevant_paths:
        print("  Project scope: no CI-relevant paths changed")
        return []

    # A change to test-selection plumbing rewrites the project scope itself, so
    # the scope it would narrow to cannot be trusted.
    if any(
        _matches_pattern(path, EXTERNAL_REPO_TEST_SELECTION_PATTERNS)
        for path in relevant_paths
    ):
        print("  Project scope: test-selection plumbing changed, testing every project")
        return []

    if not subtree_prefixes:
        print("  Project scope: no configured subtrees, testing every project")
        return []

    known_prefixes = subtree_prefixes | CI_RELEVANT_NON_SUBTREE_PREFIXES

    # A caller workflow belongs to no single project, but it also belongs to no
    # project outside the calling repo. Scoping it to the caller's own subtrees
    # keeps every project it could plausibly affect while dropping the
    # downstream suites of other repos -- a workflow edit in rocm-libraries has
    # no way to break rocm-systems' RCCL tests. Riding along with component
    # changes it adds nothing: those changes already select a narrower scope
    # within the same repo, so the workflow contributes no project of its own.
    component_paths = [
        path
        for path in relevant_paths
        if not _matches_pattern(path, EXTERNAL_REPO_CALLER_WORKFLOW_PATTERNS)
    ]
    if not component_paths:
        scope = sorted(subtree_prefixes)
        print(
            f"  Project scope: caller workflow changed, testing {len(scope)} own projects"
        )
        return scope

    # find_matched_subtrees() silently drops paths it cannot attribute, so
    # narrowing on the rest would miss their impact. Run everything instead.
    unclassified = [
        path
        for path in component_paths
        if "/".join(path.split("/", 2)[:2]) not in known_prefixes
    ]
    if unclassified:
        print(f"  Project scope: unclassified {sorted(unclassified)[:5]}, testing all")
        return []

    matched = find_matched_subtrees(component_paths, known_prefixes)
    print(f"  Project scope: {matched}")
    return matched
