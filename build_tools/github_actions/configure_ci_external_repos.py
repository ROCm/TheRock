#!/usr/bin/env python3

"""External repository CI configuration helpers.

This module contains helper functions for external repositories (rocm-libraries,
rocm-systems) that call TheRock's workflows.

These are pure helper functions - main CI orchestration logic remains in configure_ci.py.
"""

import json
import os
import sys
from typing import Optional

import configure_ci_shared as shared
from detect_external_repo_config import (
    detect_repo_name,
    get_repo_config,
    get_skip_patterns,
    get_test_list,
)


# --------------------------------------------------------------------------- #
# External repository detection and configuration
# --------------------------------------------------------------------------- #


def _should_skip_build_due_to_paths(
    repo_name: str,
    base_ref: str,
    github_event_name: str,
) -> bool:
    """Check if external repo build should be skipped based on modified paths.

    Args:
        repo_name: External repository name
        base_ref: Base git ref to diff against
        github_event_name: GitHub event name (for logging)

    Returns:
        True if build should be skipped (only skippable paths changed), False otherwise

    Raises:
        RuntimeError: If modified paths cannot be determined
    """
    print(f"Detecting changed files for event: {github_event_name}")
    modified_paths = shared.get_modified_paths(base_ref, repo_name=repo_name)

    if modified_paths is None:
        raise RuntimeError(
            "ERROR: Could not determine modified paths. Cannot safely determine if build should run."
        )

    if not modified_paths:
        print("No files modified - skipping builds")
        return True

    print(f"Found {len(modified_paths)} modified files")

    # Get skip patterns (external repo custom patterns if available, otherwise use TheRock's)
    external_skip_patterns = get_skip_patterns(repo_name)

    if external_skip_patterns:
        print(f"Using custom skip patterns from {repo_name}")
        skip_patterns = external_skip_patterns
    else:
        print(
            f"No custom skip patterns from {repo_name}, using TheRock's default patterns"
        )
        skip_patterns = shared.THEROCK_SKIPPABLE_PATH_PATTERNS

    has_non_skippable = shared.has_non_skippable_paths(modified_paths, skip_patterns)

    if not has_non_skippable:
        print("Only skippable paths modified (docs, etc) - skipping builds")
        return True

    return False


def should_build_external_repo(
    repo_name: str,
    base_ref: str,
    github_event_name: str,
    projects_input: str,
    specific_projects: list[str],
) -> bool:
    """Determine if external repo build should run.

    Args:
        repo_name: External repository name
        base_ref: Base git ref to diff against
        github_event_name: GitHub event that triggered this
        projects_input: Raw projects input string
        specific_projects: Parsed list of specific projects

    Returns:
        True if build should run, False otherwise
    """
    # Case 1: Scheduled builds always run
    if github_event_name == "schedule":
        print("Schedule event detected - building all")
        return True

    # Case 2: Explicit "all" request always runs
    if projects_input and projects_input.strip().lower() == "all":
        print("Projects override: building all")
        return True

    # Case 3: Specific projects requested
    if specific_projects:
        print(f"Building specific projects: {specific_projects}")
        return True

    # Case 4: Check if non-skippable files changed
    return not _should_skip_build_due_to_paths(
        repo_name,
        base_ref,
        github_event_name,
    )


def detect_external_repo(github_repository: str) -> tuple[bool, Optional[str]]:
    """Detect if we're running for an external repository.

    Uses the detect_repo_name() and get_repo_config() functions from
    detect_external_repo_config.py to determine if the given
    repository is a known external repo.

    Args:
        github_repository: Value from GITHUB_REPOSITORY or GITHUB_REPOSITORY_OVERRIDE env var
                          Format: "ROCm/rocm-libraries" or "rocm-libraries" or "ROCm/TheRock"

    Returns:
        Tuple of (is_external_repo, repo_name)
        - (True, "rocm-libraries") if it's a known external repo
        - (False, None) if it's TheRock or unknown repo

    Examples:
        >>> detect_external_repo("ROCm/TheRock")
        (False, None)

        >>> detect_external_repo("ROCm/rocm-libraries")
        (True, "rocm-libraries")
    """
    if not github_repository:
        return False, None

    try:
        # Extract repo name (handles both "ROCm/repo" and "repo" formats)
        repo_name = detect_repo_name(github_repository)

        # Try to get config - this validates it's a known external repo
        get_repo_config(repo_name)

        # If we got here, it's a known external repo
        print(f"Detected external repository: {repo_name} (from GITHUB_REPOSITORY)")
        return True, repo_name

    except ValueError:
        # Not a known external repo (likely TheRock or other)
        return False, None


def parse_projects_input(projects_input: str) -> list[str]:
    """Parse comma-separated projects input, stripping 'projects/' prefix.

    Args:
        projects_input: Comma-separated list like "projects/rocprim,projects/rocrand"
                        or "all" or empty string

    Returns:
        List of project names like ["rocprim", "rocrand"]
        Empty list if input is "all", empty, or whitespace
    """
    if not projects_input or projects_input.strip().lower() in ["all", ""]:
        return []

    projects = [
        p.strip().replace("projects/", "")
        for p in projects_input.split(",")
        if p.strip()
    ]

    if projects:
        print(f"Specific projects requested: {projects}")

    return projects


def get_test_list_for_build(specific_projects: list[str], repo_name: str) -> list[str]:
    """Determine which tests to run for a build.

    Args:
        specific_projects: Specific projects requested by user (if any)
        repo_name: External repository name

    Returns:
        List of test names to run
    """
    if specific_projects:
        return specific_projects

    # Get test list from external repo
    # NOTE: We do FULL BUILDS (no selective cmake options), but we can
    # still use their test list for test selection
    test_list = get_test_list(repo_name)
    if not test_list:
        print("Using default test list: ['all']")
        return ["all"]

    return test_list


def apply_external_repo_cross_product(
    linux_configs: list[dict],
    windows_configs: list[dict],
    linux_variants: list[dict],
    windows_variants: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Apply cross-product of external repo configs with GPU variants.

    Args:
        linux_configs: External repo Linux project configs
        windows_configs: External repo Windows project configs
        linux_variants: GPU variant matrix for Linux
        windows_variants: GPU variant matrix for Windows

    Returns:
        Tuple of (updated_linux_variants, updated_windows_variants)
    """
    if not linux_configs and not windows_configs:
        return linux_variants, windows_variants

    print(f"\n=== External repo detected: applying cross-product ===")
    print(
        f"Linux configs: {len(linux_configs)}, Windows configs: {len(windows_configs)}"
    )

    updated_linux = linux_variants
    updated_windows = windows_variants

    if linux_configs:
        updated_linux = cross_product_projects_with_gpu_variants(
            linux_configs, linux_variants
        )
    if windows_configs:
        updated_windows = cross_product_projects_with_gpu_variants(
            windows_configs, windows_variants
        )

    print(f"Final Linux matrix: {len(updated_linux)} entries")
    print(f"Final Windows matrix: {len(updated_windows)} entries")
    print("")

    return updated_linux, updated_windows


def cross_product_projects_with_gpu_variants(
    project_configs: list[dict], gpu_variants: list[dict]
) -> list[dict]:
    """Cross-products external repo project configs with GPU family variants.

    Args:
        project_configs: List of project configs, each with "projects_to_test" key
        gpu_variants: List of GPU variant dicts from matrix_generator

    Returns:
        List of combined dicts (GPU variant + projects_to_test)
    """
    final_variants: list[dict] = []
    for project_config in project_configs:
        for gpu_variant in gpu_variants:
            final_variants.append(
                {
                    **gpu_variant,
                    "projects_to_test": project_config["projects_to_test"],
                    # Note: cmake_options removed - external repos do full builds
                }
            )
    return final_variants


def output_empty_matrix_and_exit(gha_set_output_func):
    """Output empty CI matrix when no projects detected for external repo.

    Args:
        gha_set_output_func: Function to call for setting GitHub Actions output
    """
    print("No projects to build - outputting empty matrix")
    output = {
        "linux_variants": json.dumps([]),
        "linux_test_labels": json.dumps([]),
        "windows_variants": json.dumps([]),
        "windows_test_labels": json.dumps([]),
        "enable_build_jobs": json.dumps(False),
        "test_type": "smoke",
    }
    gha_set_output_func(output)
    sys.exit(0)


def detect_external_repo_projects_to_build(
    *,
    repo_name: str,
    base_ref: str,
    github_event_name: str,
    projects_input: str = "",
) -> dict[str, list[dict]]:
    """Determine which projects to build for an external repository.

    This orchestration function centralizes the "when to build" logic for external repos.

    Args:
        repo_name: Repository name (e.g., "rocm-libraries")
        base_ref: Base git ref to diff against
        github_event_name: GitHub event that triggered this (e.g., "pull_request", "schedule")
        projects_input: Optional manual projects override (comma-separated)

    Returns:
        Dict with keys:
          - linux_projects: list[dict] - Empty list or single test config
          - windows_projects: list[dict] - Empty list or single test config

        Each config dict contains:
          - projects_to_test: str - Comma-separated list of tests to run
    """
    # Parse projects input (e.g., "projects/rocprim,projects/rocrand" -> ["rocprim", "rocrand"])
    specific_projects = parse_projects_input(projects_input)

    # Determine if we should build
    should_build = should_build_external_repo(
        repo_name,
        base_ref,
        github_event_name,
        projects_input,
        specific_projects,
    )

    # If we shouldn't build, return empty configs
    if not should_build:
        return {"linux_projects": [], "windows_projects": []}

    # Determine test list
    test_list = get_test_list_for_build(specific_projects, repo_name)

    # Convert test list to comma-separated string (consumed by fetch_test_configurations.py)
    test_config = {"projects_to_test": ",".join(test_list)}
    return {
        "linux_projects": [test_config],
        "windows_projects": [test_config],
    }


def setup_external_repo_configs(
    base_args: dict,
    output_empty_matrix_and_exit_func,
) -> Optional[dict]:
    """Detect and configure external repository settings.

    Args:
        base_args: Dictionary containing base_ref, github_event_name, etc.
        output_empty_matrix_and_exit_func: Function to output empty matrix (wraps configure_ci func)

    Returns:
        Dict with linux_external_project_configs and windows_external_project_configs,
        or None if not an external repo. Exits early if no projects detected.
    """
    github_repository = os.environ.get(
        shared.ENV_GITHUB_REPOSITORY_OVERRIDE
    ) or os.environ.get(shared.ENV_GITHUB_REPOSITORY, "")
    is_external_repo, repo_name = detect_external_repo(github_repository)

    if not is_external_repo or not repo_name:
        return None

    print(f"\n=== Detected external repository: {repo_name} ===")

    # Determine which projects to build for this external repo
    project_detection = detect_external_repo_projects_to_build(
        repo_name=repo_name,
        base_ref=base_args["base_ref"],
        github_event_name=base_args.get("github_event_name", ""),
        projects_input=os.environ.get(shared.ENV_PROJECTS, ""),
    )

    linux_configs = project_detection.get("linux_projects", [])
    windows_configs = project_detection.get("windows_projects", [])

    print(
        f"Project detection result: Linux={len(linux_configs)}, Windows={len(windows_configs)}"
    )

    # If no projects detected, skip builds entirely
    if not linux_configs and not windows_configs:
        output_empty_matrix_and_exit_func()

    return {
        "linux_external_project_configs": linux_configs,
        "windows_external_project_configs": windows_configs,
    }
