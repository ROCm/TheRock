#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure CI for external repos (rocm-systems, rocm-libraries).

This script determines which projects changed, whether to run/skip tests,
and which TheRock build stages can be reused vs must be rebuilt.

Usage:
    python configure_external_repo_ci.py \
        --event-name pull_request \
        --github-repo ROCm/rocm-libraries \
        --base-sha abc123 \
        --head-sha def456 \
        --config-path .github/repos-config.json

Outputs (to $GITHUB_OUTPUT):
    changed_projects: Comma-separated list of changed project paths
    run_all_tests: "true" if CI files changed (run full test suite)
    skip_tests: "true" if only docs/skippable files changed
    reusable_stages: Comma-separated list of TheRock stages that can be reused
    rebuild_stages: Comma-separated list of TheRock stages that must be rebuilt
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
from pathlib import Path
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])

# Patterns for files that don't require tests (docs, etc.)
SKIPPABLE_PATH_PATTERNS = [
    "*.md",
    "*.rst",
    "docs/*",
    "projects/*/docs/*",
    "shared/*/docs/*",
]

# Patterns that trigger a full test run when changed (CI infrastructure)
FULL_TEST_TRIGGER_PATTERNS = [
    ".github/workflows/therock*",
    ".github/scripts/therock*",
    ".github/scripts/get_changed_projects.py",
    ".github/scripts/ci_utils.py",
    ".github/scripts/config_loader.py",
    ".github/scripts/repo_config_model.py",
    ".github/scripts/pr_detect_changed_subtrees.py",
    ".github/repos-config.json",
]

# =============================================================================
# TEST OVERRIDE: Force specific changed_projects for testing cross-repo CI
# Set to non-empty string to override detected changes (e.g., "projects/rocprim")
# TODO(geomin12): Remove this after testing cross-repo artifact reuse
# =============================================================================
_TEST_OVERRIDE_CHANGED_PROJECTS = os.environ.get(
    "TEST_OVERRIDE_CHANGED_PROJECTS", ""
)


def compute_stage_impact(
    external_repo_name: str,
    modified_paths: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Compute which TheRock stages are affected by external repo changes.

    Maps external repo paths to TheRock source sets and runs stage impact
    analysis to determine which stages can be reused vs must be rebuilt.

    Args:
        external_repo_name: Name of external repo (e.g., "rocm-libraries")
        modified_paths: Paths changed in the external repo

    Returns:
        Tuple of (reusable_stages, rebuild_stages)
    """
    try:
        # Import stage_impact here to avoid circular imports and make it optional
        # for environments that don't have the full TheRock build tools
        from stage_impact import analyze_stage_impact, StageImpactRuleSet
        from _therock_utils.build_topology import get_topology
    except ImportError as e:
        logger.warning(f"Stage impact analysis unavailable: {e}")
        return ([], [])

    # Map external repo paths to TheRock paths by prefixing with repo name
    # e.g., "projects/rocprim/foo.cpp" -> "rocm-libraries/projects/rocprim/foo.cpp"
    therock_paths = [f"{external_repo_name}/{p}" for p in modified_paths]

    # For external repos, we need custom rules that don't trigger full CI
    # for .github/ changes (since those are external repo's own CI files)
    external_rules = StageImpactRuleSet(
        full_ci_prefixes=(
            # Only TheRock's own build tooling triggers full CI
            # External repo .github/ changes don't affect TheRock stages
            "build_tools/",
            "scripts/",
        ),
        full_ci_exact_paths=(
            "BUILD_TOPOLOGY.toml",
            "CMakeLists.txt",
        ),
    )

    try:
        topology = get_topology()
        result = analyze_stage_impact(
            changed_inputs=therock_paths,
            topology=topology,
            rules=external_rules,
        )

        logger.info(f"Stage impact: rebuild={result.rebuild_stages}, copy={result.copy_stages}")
        return (list(result.copy_stages), list(result.rebuild_stages))
    except Exception as e:
        logger.warning(f"Stage impact analysis failed: {e}")
        return ([], [])


@dataclass
class ConfigureResult:
    """Result of CI configuration analysis."""

    changed_projects: str  # Comma-separated list
    run_all_tests: bool
    skip_tests: bool
    reusable_stages: str = ""  # Comma-separated list of stages that can be reused
    rebuild_stages: str = ""  # Comma-separated list of stages that must be rebuilt


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

    Returns None if the result is truncated (>300 files) to signal caller
    should fall back to run_all_tests.
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
    # GitHub compare API returns max 300 files; if truncated, fall back to run-all
    if len(files) >= 300:
        logger.warning("Compare API returned 300+ files, result may be truncated")
        return None
    return {f["filename"] for f in files}


def matches_patterns(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    """Check if any path matches any pattern."""
    for path in paths:
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
    return False


def is_skippable(path: str) -> bool:
    """Check if path is skippable (docs, etc.)."""
    return any(fnmatch.fnmatch(path, p) for p in SKIPPABLE_PATH_PATTERNS)


def has_non_skippable(paths: Iterable[str]) -> bool:
    """Check if any path is non-skippable."""
    return any(not is_skippable(p) for p in paths)


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


def find_matched_subtrees(
    changed_files: Iterable[str], valid_prefixes: Set[str]
) -> List[str]:
    """Find subtrees matching changed files."""
    changed_subtrees = {
        "/".join(path.split("/", 2)[:2])
        for path in changed_files
        if len(path.split("/")) >= 2
    }
    return sorted(changed_subtrees & valid_prefixes)


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
    enable_stage_reuse: bool = False,
) -> ConfigureResult:
    """Main configuration logic."""

    # Extract external repo name from github_repo (e.g., "ROCm/rocm-libraries" -> "rocm-libraries")
    external_repo_name = github_repo.split("/")[-1] if "/" in github_repo else github_repo

    # TEST OVERRIDE: If set, bypass all detection and return the override value
    if _TEST_OVERRIDE_CHANGED_PROJECTS:
        logger.info(
            f"TEST OVERRIDE: Using forced changed_projects='{_TEST_OVERRIDE_CHANGED_PROJECTS}'"
        )
        return ConfigureResult(
            changed_projects=_TEST_OVERRIDE_CHANGED_PROJECTS,
            run_all_tests=False,
            skip_tests=False,
        )

    # Schedule/workflow_dispatch events run all tests
    if event_name in ("schedule", "workflow_dispatch"):
        logger.info(f"{event_name} event - running all tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

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
        logger.warning("No SHAs provided - running all tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

    # If API returned None (truncated results), fall back to run-all
    if modified_paths is None:
        logger.info("Truncated API response - running all tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

    if not modified_paths:
        logger.info("No modified paths - skipping tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=False, skip_tests=True
        )

    logger.info(f"Modified paths: {len(modified_paths)} files")

    # Check if CI files changed (run all tests)
    if matches_patterns(modified_paths, FULL_TEST_TRIGGER_PATTERNS):
        logger.info("CI files changed - running all tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

    # Check if only skippable files changed
    if not has_non_skippable(modified_paths):
        logger.info("Only skippable files changed - skipping tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=False, skip_tests=True
        )

    # Find changed projects from config
    config = load_repo_config(config_path)
    if not config:
        logger.warning("No config loaded - running all tests")
        return ConfigureResult(
            changed_projects="", run_all_tests=True, skip_tests=False
        )

    valid_prefixes = get_valid_prefixes(config)
    matched = find_matched_subtrees(modified_paths, valid_prefixes)
    logger.info(f"Matched projects: {matched}")

    # Compute stage impact for automatic stage reuse
    reusable_stages: List[str] = []
    rebuild_stages: List[str] = []
    if enable_stage_reuse:
        reusable_stages, rebuild_stages = compute_stage_impact(
            external_repo_name, modified_paths
        )

    return ConfigureResult(
        changed_projects=",".join(matched),
        run_all_tests=False,
        skip_tests=False,
        reusable_stages=",".join(reusable_stages),
        rebuild_stages=",".join(rebuild_stages),
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
    parser.add_argument(
        "--enable-stage-reuse",
        action="store_true",
        help="Enable automatic stage reuse analysis (outputs reusable_stages, rebuild_stages)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    logger.info(f"Configuring CI for {args.github_repo}")

    result = configure(
        event_name=args.event_name,
        github_repo=args.github_repo,
        base_sha=args.base_sha or None,
        head_sha=args.head_sha or None,
        config_path=args.config_path,
        enable_stage_reuse=args.enable_stage_reuse,
    )

    outputs = {
        "changed_projects": result.changed_projects,
        "run_all_tests": str(result.run_all_tests).lower(),
        "skip_tests": str(result.skip_tests).lower(),
    }

    # Add stage reuse outputs if enabled
    if args.enable_stage_reuse:
        outputs["reusable_stages"] = result.reusable_stages
        outputs["rebuild_stages"] = result.rebuild_stages

    set_github_output(outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
