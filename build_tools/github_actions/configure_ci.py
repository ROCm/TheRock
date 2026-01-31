#!/usr/bin/env python3

"""Configures metadata for a CI workflow run.

----------
| Inputs |
----------

  Environment variables (for all triggers):
  * GITHUB_EVENT_NAME    : GitHub event name, e.g. pull_request.
  * GITHUB_OUTPUT        : path to write workflow output variables.
  * GITHUB_STEP_SUMMARY  : path to write workflow summary output.
  * INPUT_LINUX_AMDGPU_FAMILIES (optional): Comma-separated string of Linux AMD GPU families
  * LINUX_TEST_LABELS (optional): Comma-separated list of test labels to test
  * LINUX_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Linux tests
  * INPUT_WINDOWS_AMDGPU_FAMILIES (optional): Comma-separated string of Windows AMD GPU families
  * WINDOWS_TEST_LABELS (optional): Comma-separated list of test labels to test
  * WINDOWS_USE_PREBUILT_ARTIFACTS (optional): If enabled, CI will only run Windows tests
  * BRANCH_NAME (optional): The branch name
  * BUILD_VARIANT (optional): The build variant to run (ex: release, asan)
  * ROCM_THEROCK_TEST_RUNNERS (optional): Test runner JSON object, coming from ROCm organization
  * LOAD_TEST_RUNNERS_FROM_VAR (optional): boolean env variable that loads in ROCm org data if enabled

  Environment variables (for pull requests):
  * PR_LABELS (optional) : JSON list of PR label names.
  * BASE_REF  (required) : base commit SHA of the PR.

  Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

  Written to GITHUB_OUTPUT:
  * linux_amdgpu_families : List of valid Linux AMD GPU families to execute build and test jobs
  * linux_test_labels : List of test names to run on Linux, optionally filtered by PR labels.
  * windows_amdgpu_families : List of valid Windows AMD GPU families to execute build and test jobs
  * windows_test_labels : List of test names to run on Windows, optionally filtered by PR labels.
  * enable_build_jobs: If true, builds will be enabled
  * test_type: The type of test that component tests will run (i.e. smoke, full)

  Written to GITHUB_STEP_SUMMARY:
  * Human-readable summary for most contributors

  Written to stdout/stderr:
  * Detailed information for CI maintainers
"""

import argparse
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List, Optional

from amdgpu_family_matrix import (
    all_build_variants,
    get_all_families_for_trigger_types,
)
import configure_ci_shared as shared
import configure_ci_external_repos as external_repos
from detect_external_repo_config import get_skip_patterns

from fetch_test_configurations import test_matrix

from github_actions_utils import *

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_gpu_family_overrides(value: str, lookup_matrix: dict) -> list[str]:
    """Parses an amdgpu_families override string into lookup-matrix keys.

    Accepts comma/whitespace-separated tokens. Each token may be either:
    - A matrix key (e.g. "gfx94x", "gfx110x")
    - A canonical family string from the matrix (e.g. "gfx94X-dcgpu", "gfx110X-all")
    """
    # Build a reverse map from canonical family strings -> lookup keys.
    family_to_key: dict[str, str] = {}
    for key, platform_set in lookup_matrix.items():
        if not isinstance(platform_set, dict):
            continue
        for platform_info in platform_set.values():
            if not isinstance(platform_info, dict):
                continue
            family = platform_info.get("family")
            if family:
                family_to_key[str(family).lower()] = key

    tokens = [t for t in value.replace(",", " ").split() if t]
    resolved: list[str] = []
    for token in tokens:
        tl = token.lower()
        if tl in lookup_matrix:
            resolved.append(tl)
        elif tl in family_to_key:
            resolved.append(family_to_key[tl])
        else:
            raise Exception(
                f"Unknown target name '{token}' not found in matrix keys or family strings"
            )
    return resolved


# --------------------------------------------------------------------------- #
# Main orchestration helpers
# --------------------------------------------------------------------------- #


def _add_gpu_families_from_input(
    families: dict,
    lookup_matrix: dict,
    selected_target_names: list,
    fallback_targets: list = None,
) -> bool:
    """Add GPU families from input to selected_target_names.

    Args:
        families: Dictionary containing 'amdgpu_families' key with comma-separated values
        lookup_matrix: Family info matrix from amdgpu_family_matrix.py
        selected_target_names: List to append selected target names to (modified in place)
        fallback_targets: Optional list of fallback targets if no explicit input provided

    Returns:
        True if explicit input was used, False if fallback was used
    """
    input_gpu_targets = families.get("amdgpu_families", "")
    if input_gpu_targets:
        print(f"Using explicitly provided GPU families: {input_gpu_targets}")
        requested_target_names = parse_gpu_family_overrides(
            input_gpu_targets, lookup_matrix
        )
        selected_target_names.extend(
            filter_known_names(requested_target_names, "target", lookup_matrix)
        )
        return True
    elif fallback_targets is not None:
        selected_target_names.extend(fallback_targets)
        return False
    return False


def _format_variants_for_summary(variants: list[dict]) -> list:
    result = []
    for item in variants:
        if "family" in item:
            result.append(item["family"])
        elif "matrix_per_family_json" in item:
            # Multi-arch mode: show the families from the JSON
            families = json.loads(item["matrix_per_family_json"])
            result.append([f["amdgpu_family"] for f in families])
    return result


def _cross_product_projects_with_gpu_variants(
    project_configs: list[dict], gpu_variants: list[dict]
) -> list[dict]:
    """Cross-products external repo project configs with GPU family variants."""
    final_variants: list[dict] = []
    for project_config in project_configs:
        for gpu_variant in gpu_variants:
            final_variants.append(
                {
                    **gpu_variant,
                    "project_to_test": project_config["project_to_test"],
                    # Note: cmake_options removed - external repos do full builds
                }
            )
    return final_variants


def _determine_enable_build_jobs_and_test_type(
    *,
    github_event_name: str,
    is_schedule: bool,
    base_ref: str,
    linux_test_output: list,
    windows_test_output: list,
    has_external_project_configs: bool,
) -> tuple[bool, str]:
    test_type = "smoke"

    # In the case of a scheduled run, we always want to build and we want to run full tests
    if is_schedule:
        enable_build_jobs = True
        test_type = "full"
        return enable_build_jobs, test_type

    # When external repos explicitly call TheRock's CI (workflow_dispatch or workflow_call),
    # they're requesting specific project builds. Honor this request without checking modified paths.
    if has_external_project_configs:
        print("External repo requesting builds - enabling without path checks")
        enable_build_jobs = True
        return enable_build_jobs, (
            "full" if (linux_test_output or windows_test_output) else test_type
        )

    modified_paths = shared.get_modified_paths(base_ref)
    print("modified_paths (max 200):", modified_paths[:200])
    print(f"Checking modified files since this had a {github_event_name} trigger")
    # TODO(#199): other behavior changes
    #     * workflow_dispatch or workflow_call with inputs controlling enabled jobs?
    enable_build_jobs = should_ci_run_given_modified_paths(modified_paths)

    # If the modified path contains any git submodules, we want to run a full test suite.
    # Otherwise, we just run smoke tests
    submodule_paths = get_therock_submodule_paths()
    matching_submodule_paths = list(set(submodule_paths) & set(modified_paths))
    if matching_submodule_paths:
        print(
            f"Found changed submodules: {str(matching_submodule_paths)}. Running full tests."
        )
        test_type = "full"

    # If any test label is included, run full test suite for specified tests
    if linux_test_output or windows_test_output:
        test_type = "full"

    return enable_build_jobs, test_type


def _emit_summary_and_outputs(
    *,
    linux_variants_output: list[dict],
    linux_test_output: list,
    windows_variants_output: list[dict],
    windows_test_output: list,
    enable_build_jobs: bool,
    test_type: str,
    base_args: dict,
) -> None:
    gha_append_step_summary(
        f"""## Workflow configure results

* `linux_variants`: {str(_format_variants_for_summary(linux_variants_output))}
* `linux_test_labels`: {str([test for test in linux_test_output])}
* `linux_use_prebuilt_artifacts`: {json.dumps(base_args.get("linux_use_prebuilt_artifacts"))}
* `windows_variants`: {str(_format_variants_for_summary(windows_variants_output))}
* `windows_test_labels`: {str([test for test in windows_test_output])}
* `windows_use_prebuilt_artifacts`: {json.dumps(base_args.get("windows_use_prebuilt_artifacts"))}
* `enable_build_jobs`: {json.dumps(enable_build_jobs)}
* `test_type`: {test_type}
    """
    )

    output = {
        "linux_variants": json.dumps(linux_variants_output),
        "linux_test_labels": json.dumps(linux_test_output),
        "windows_variants": json.dumps(windows_variants_output),
        "windows_test_labels": json.dumps(windows_test_output),
        "enable_build_jobs": json.dumps(enable_build_jobs),
        "test_type": test_type,
    }
    gha_set_output(output)


def _detect_external_repo(cwd: Path, repo_override: str) -> tuple[bool, Optional[str]]:
    # List of supported external repositories - derived from REPO_CONFIGS
    from detect_external_repo_config import REPO_CONFIGS

    external_repos = list(REPO_CONFIGS.keys())

    if repo_override:
        print(f"Using repository override: {repo_override}")
        repo_name_from_override = (
            repo_override.split("/")[-1] if "/" in repo_override else repo_override
        )
        for external_repo in external_repos:
            if external_repo in repo_name_from_override.lower():
                print(f"Detected external repository from override: {external_repo}")
                return True, external_repo
        return False, None

    # Check if any part of the path matches an external repo name
    # Using path parts prevents false matches like "/my-rocm-libraries-backup/other-project"
    cwd_parts = [p.lower() for p in cwd.parts]
    for external_repo in external_repos:
        if external_repo in cwd_parts:
            print(f"Detected external repository from path: {external_repo}")
            return True, external_repo

    return False, None


def _collect_inputs_from_env() -> tuple[dict, dict, dict]:
    base_args: dict = {}
    linux_families: dict = {}
    windows_families: dict = {}

    linux_families["amdgpu_families"] = os.environ.get(
        "INPUT_LINUX_AMDGPU_FAMILIES", ""
    )
    windows_families["amdgpu_families"] = os.environ.get(
        "INPUT_WINDOWS_AMDGPU_FAMILIES", ""
    )

    base_args["pr_labels"] = os.environ.get("PR_LABELS", '{"labels": []}')
    base_args["branch_name"] = os.environ.get("GITHUB_REF_NAME", "")
    if base_args["branch_name"] == "":
        print(
            "[ERROR] GITHUB_REF_NAME is not set! No branch name detected. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    base_args["github_event_name"] = os.environ.get("GITHUB_EVENT_NAME", "")
    base_args["base_ref"] = os.environ.get("BASE_REF", "HEAD^1")
    base_args["linux_use_prebuilt_artifacts"] = (
        os.environ.get("LINUX_USE_PREBUILT_ARTIFACTS") == "true"
    )
    base_args["windows_use_prebuilt_artifacts"] = (
        os.environ.get("WINDOWS_USE_PREBUILT_ARTIFACTS") == "true"
    )
    base_args["workflow_dispatch_linux_test_labels"] = os.getenv(
        "LINUX_TEST_LABELS", ""
    )
    base_args["workflow_dispatch_windows_test_labels"] = os.getenv(
        "WINDOWS_TEST_LABELS", ""
    )
    base_args["workflow_dispatch_additional_label_options"] = os.getenv(
        "ADDITIONAL_LABEL_OPTIONS", ""
    )
    base_args["build_variant"] = os.getenv("BUILD_VARIANT", "release")
    base_args["multi_arch"] = os.environ.get("MULTI_ARCH", "false") == "true"

    return base_args, linux_families, windows_families


# =============================================================================
# External Repository CI Detection Logic
# =============================================================================


def _detect_external_repo_projects(
    *,
    repo_name: str,
    base_ref: str,
    github_event_name: str,
    projects_input: str = "",
) -> dict:
    """Detects whether to run CI for external repos based on changes (pure function).

    This is a pure detection function that:
    - Imports skip patterns and test lists from external repos
    - Determines if CI should run based on file changes
    - Returns project configuration for full builds

    NOTE: Always performs FULL BUILDS (ignores selective cmake options from external repos).

    Args:
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")
        base_ref: Base git ref to diff against
        github_event_name: GitHub event that triggered this (e.g., "pull_request", "schedule")
        projects_input: Optional manual projects override

    Returns:
        Dict with keys:
          - linux_projects: list[dict] - Empty list or single test config
          - windows_projects: list[dict] - Empty list or single test config

        Each config dict contains:
          - project_to_test: [list of tests] - Tests to run (full builds always happen)
    """
    from detect_external_repo_config import get_skip_patterns, get_test_list

    # Check if specific projects were requested
    specific_projects = []
    if projects_input and projects_input.strip().lower() not in ["all", ""]:
        # Parse comma-separated project list (e.g., "projects/rocprim,projects/rocrand")
        # Strip "projects/" prefix to get just the project names
        specific_projects = [
            p.strip().replace("projects/", "")
            for p in projects_input.split(",")
            if p.strip()
        ]
        if specific_projects:
            print(f"Specific projects requested: {specific_projects}")

    # For scheduled builds or explicit "all" request, always build
    if github_event_name == "schedule":
        print("Schedule event detected - building all")
        should_build = True
    elif projects_input and projects_input.strip().lower() == "all":
        print("Projects override: building all")
        should_build = True
    elif specific_projects:
        # Specific projects requested - skip change detection
        print(f"Building specific projects: {specific_projects}")
        should_build = True
    else:
        # Get skip patterns from external repo
        skip_patterns = get_skip_patterns(repo_name)

        # Fallback to defaults if external repo doesn't provide patterns
        if not skip_patterns:
            print("Using default skip patterns")
            skip_patterns = [
                "docs/*",
                "*.gitignore",
                "*.md",
                "*.pre-commit-config.*",
                "*CODEOWNERS",
                "*LICENSE",
            ]

        # Check if only skippable paths changed
        print(f"Detecting changed files for event: {github_event_name}")
        modified_paths = shared.get_modified_paths(base_ref)

        if modified_paths is None:
            print("ERROR: Could not determine modified paths")
            return {"linux_projects": [], "windows_projects": []}
        if not modified_paths:
            print("No files modified - skipping builds")
            return {"linux_projects": [], "windows_projects": []}

        print(f"Found {len(modified_paths)} modified files")

        # Check if any non-skippable paths changed
        has_non_skippable = any(
            not any(fnmatch.fnmatch(path, pattern) for pattern in skip_patterns)
            for path in modified_paths
        )

        if not has_non_skippable:
            print("Only skippable paths modified (docs, etc) - skipping builds")
            return {"linux_projects": [], "windows_projects": []}

        should_build = True

    if should_build:
        # Determine which tests to run
        if specific_projects:
            # Use the specific projects requested
            test_list = specific_projects
        else:
            # Get test list from external repo
            # NOTE: We do FULL BUILDS (no selective cmake options), but we can
            # still use their test list for test selection
            test_list = get_test_list(repo_name)

            # Fallback to "all" if external repo doesn't provide test list
            if not test_list:
                print("Using default test list: ['all']")
                test_list = ["all"]

        # Convert test list to comma-separated string (consumed by fetch_test_configurations.py)
        test_config = {"project_to_test": ",".join(test_list)}
        return {
            "linux_projects": [test_config],
            "windows_projects": [test_config],
        }
    else:
        return {"linux_projects": [], "windows_projects": []}


def _handle_external_repo_detection_or_exit(base_args: dict, repo_name: str) -> None:
    """Orchestrates external repo CI detection and handles early exit (has side effects).

    This orchestrator function:
    1. Calls _detect_external_repo_projects() for pure detection logic
    2. Prints detection results
    3. Exits early with empty matrix if nothing to build
    4. Stores project configs in base_args for downstream processing

    This is separate from _detect_external_repo_projects() to maintain separation
    between pure logic (detection) and side effects (I/O, exit, state mutation).

    Args:
        base_args: Configuration dictionary to store results in
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")

    Side Effects:
        - Prints to stdout
        - Calls sys.exit(0) if no projects detected
        - Mutates base_args by adding linux/windows_external_project_configs
    """
    print(f"\n=== Detecting projects for {repo_name} ===")
    project_detection = _detect_external_repo_projects(
        repo_name=repo_name,
        base_ref=base_args["base_ref"],
        github_event_name=base_args.get("github_event_name", ""),
        projects_input=os.environ.get("PROJECTS", ""),
    )

    linux_project_configs = project_detection.get("linux_projects", [])
    windows_project_configs = project_detection.get("windows_projects", [])

    print(
        f"\nLinux: {len(linux_project_configs)} config(s), Windows: {len(windows_project_configs)} config(s)"
    )

    # If no projects detected for either platform, skip builds
    if not linux_project_configs and not windows_project_configs:
        print("No projects to build - outputting empty matrix")
        output = {
            "linux_variants": json.dumps([]),
            "linux_test_labels": json.dumps([]),
            "windows_variants": json.dumps([]),
            "windows_test_labels": json.dumps([]),
            "enable_build_jobs": json.dumps(False),
            "test_type": "smoke",
        }
        gha_set_output(output)
        sys.exit(0)

    # Store platform-specific project configs
    base_args["linux_external_project_configs"] = linux_project_configs
    base_args["windows_external_project_configs"] = windows_project_configs


def run_from_env() -> None:
    # Auto-detect if we're running for an external repository
    # When running from setup.yml with external_source_checkout=true, the working directory
    # will be 'source-repo' which contains the external repo
    cwd = Path.cwd()

    # Check for repository override (for testing from TheRock)
    repo_override = os.environ.get("GITHUB_REPOSITORY_OVERRIDE", "")
    is_external_repo, repo_name = _detect_external_repo(cwd, repo_override)

    if is_external_repo:
        print("Using TheRock's matrix configuration for external repository")
        # External repos use TheRock's GPU family matrix (no custom matrices needed)
        # The project maps are centralized in external_repo_project_maps.py
    else:
        # We're running for TheRock itself - already imported at top of file
        print("Using TheRock's own matrix configuration")

    base_args, linux_families, windows_families = _collect_inputs_from_env()

    # For external repos, call detect_external_projects first to get project-based matrix
    if is_external_repo and repo_name:
        _handle_external_repo_detection_or_exit(base_args, repo_name)

    main(base_args, linux_families, windows_families)


# --------------------------------------------------------------------------- #
# Filtering by modified paths
# --------------------------------------------------------------------------- #


def get_therock_submodule_paths() -> Optional[Iterable[str]]:
    """Returns TheRock submodules paths."""
    try:
        response = subprocess.run(
            ["git", "submodule", "status"],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=60,
            cwd=THEROCK_DIR,
        ).stdout.splitlines()

        submodule_paths = []
        for line in response:
            submodule_data_array = line.split()
            # The line will be "{commit-hash} {path} {branch}". We will retrieve the path.
            submodule_paths.append(submodule_data_array[1])
        return submodule_paths
    except TimeoutError:
        print(
            "Computing modified files timed out. Not using PR diff to determine"
            " jobs to run.",
            file=sys.stderr,
        )
        return []


# Use TheRock's skippable path patterns from shared module
SKIPPABLE_PATH_PATTERNS = shared.THEROCK_SKIPPABLE_PATH_PATTERNS


def is_path_skippable(path: str) -> bool:
    """Determines if a given relative path to a file matches any skippable patterns."""
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if at least one path is not in the skippable set."""
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


GITHUB_WORKFLOWS_CI_PATTERNS = [
    "setup.yml",
    "ci*.yml",
    "multi_arch*.yml",
    "build*artifact*.yml",
    "test*artifacts.yml",
    "test_sanity_check.yml",
    "test_component.yml",
]


def is_path_workflow_file_related_to_ci(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, ".github/workflows/" + pattern)
        for pattern in GITHUB_WORKFLOWS_CI_PATTERNS
    )


def check_for_workflow_file_related_to_ci(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(is_path_workflow_file_related_to_ci(p) for p in paths)


def should_ci_run_given_modified_paths(paths: Optional[Iterable[str]]) -> bool:
    """Returns true if CI workflows should run given a list of modified paths."""

    if paths is None:
        print("No files were modified, skipping build jobs")
        return False

    paths_set = set(paths)
    github_workflows_paths = set(
        [p for p in paths if p.startswith(".github/workflows")]
    )
    other_paths = paths_set - github_workflows_paths

    related_to_ci = check_for_workflow_file_related_to_ci(github_workflows_paths)
    contains_other_non_skippable_files = check_for_non_skippable_path(other_paths)

    print("should_ci_run_given_modified_paths findings:")
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


# --------------------------------------------------------------------------- #
# Matrix creation logic based on PR, push, or workflow_dispatch
# --------------------------------------------------------------------------- #


def get_pr_labels(args) -> List[str]:
    """Gets a list of labels applied to a pull request."""
    pr_labels_str = args.get("pr_labels", "")
    if not pr_labels_str:
        return []
    data = json.loads(pr_labels_str)
    labels = []
    for label in data.get("labels", []):
        labels.append(label["name"])
    return labels


def get_workflow_dispatch_additional_label_options(args) -> List[str]:
    """Gets a list of additional label options from workflow_dispatch."""
    additional_label_options = args.get(
        "workflow_dispatch_additional_label_options", ""
    )
    if additional_label_options:
        return [
            label.strip()
            for label in additional_label_options.split(",")
            if label.strip()
        ]
    return []


def filter_known_names(
    requested_names: List[str], name_type: str, target_matrix=None
) -> List[str]:
    """Filters a requested names list down to known names.

    Args:
        requested_names: List of names to filter
        name_type: Type of name ('target' or 'test')
        target_matrix: For 'target' type, the specific family matrix to use. Required for 'target' type.
    """
    if name_type == "target":
        assert (
            target_matrix is not None
        ), "target_matrix must be provided for 'target' name_type"
        known_references = {"target": target_matrix}
    else:
        known_references = {"test": test_matrix}

    filtered_names = []
    if name_type not in known_references:
        print(f"WARNING: unknown name_type '{name_type}'")
        return filtered_names
    for name in requested_names:
        # Standardize on lowercase names.
        # This helps prevent potential user-input errors.
        name = name.lower()

        if name in known_references[name_type]:
            filtered_names.append(name)
        else:
            print(
                f"WARNING: unknown {name_type} name '{name}' not found in matrix:\n{known_references[name_type]}"
            )

    return filtered_names


def generate_multi_arch_matrix(
    target_names: List[str],
    lookup_matrix: dict,
    platform: str,
    platform_build_variants: dict,
    base_args: dict,
) -> List[dict]:
    """Generate matrix grouped by build_variant with structured per-family data.

    In multi-arch mode, instead of creating one entry per (family × build_variant),
    we create one entry per build_variant containing all families that support it.
    This allows multi_arch_build_portable_linux.yml to run generic stages once
    and matrix over families only for per-arch stages.

    Args:
        target_names: List of target family names (e.g., ["gfx94X", "gfx1201"])
        lookup_matrix: Family info matrix from amdgpu_family_matrix.py
        platform: Platform name ("linux" or "windows")
        platform_build_variants: Dict of build variant configs for this platform
        base_args: Base arguments including 'build_variant' to filter by

    Returns:
        List of matrix entries, each containing:
        - matrix_per_family_json: JSON array of {amdgpu_family, test-runs-on, sanity_check_only_for_family} objects
          for per-architecture job matrix expansion
        - dist_amdgpu_families: Semicolon-separated family names for THEROCK_DIST_AMDGPU_TARGETS
        - build_variant_label: Human-readable label (e.g., "Release", "ASAN")
        - build_variant_suffix: Suffix for artifact naming (e.g., "", "asan"). Empty string
          for release builds, short identifier for other variants.
        - build_variant_cmake_preset: CMake preset name (e.g., "release", "asan")
        - expect_failure: If True, job failure is non-blocking (continue-on-error)
        - artifact_group: Unique identifier for artifact grouping, formatted as
          "multi-arch-{suffix}" where suffix defaults to "release" if empty
    """
    # Collect per-family info for each build_variant
    variant_to_family_info: dict[str, List[dict]] = {}
    variant_info: dict[str, dict] = {}

    for target_name in target_names:
        platform_set = lookup_matrix.get(target_name)
        if not platform_set or platform not in platform_set:
            continue
        platform_info = platform_set.get(platform)
        family_name = platform_info["family"]
        test_runs_on = platform_info.get("test-runs-on", "")

        for build_variant_name in platform_info.get("build_variants", []):
            if build_variant_name != base_args.get("build_variant"):
                continue

            if build_variant_name not in variant_to_family_info:
                variant_to_family_info[build_variant_name] = []
                variant_info[build_variant_name] = platform_build_variants.get(
                    build_variant_name
                )

            # Check for duplicates by family name
            existing_families = [
                f["amdgpu_family"] for f in variant_to_family_info[build_variant_name]
            ]
            if family_name not in existing_families:
                variant_to_family_info[build_variant_name].append(
                    {
                        "amdgpu_family": family_name,
                        "test-runs-on": test_runs_on,
                        "sanity_check_only_for_family": platform_info.get(
                            "sanity_check_only_for_family", False
                        ),
                    }
                )

    # Create one matrix entry per build_variant
    matrix_output = []
    for variant_name, family_info_list in variant_to_family_info.items():
        info = variant_info[variant_name]
        if not info:
            continue

        # Extract family names for dist_amdgpu_families
        family_names = [f["amdgpu_family"] for f in family_info_list]

        matrix_row = {
            "matrix_per_family_json": json.dumps(family_info_list),
            "dist_amdgpu_families": ";".join(family_names),
            "artifact_group": f"multi-arch-{info.get('build_variant_suffix') or 'release'}",
            "build_variant_label": info["build_variant_label"],
            "build_variant_suffix": info["build_variant_suffix"],
            "build_variant_cmake_preset": info["build_variant_cmake_preset"],
            "expect_failure": info.get("expect_failure", False),
        }
        matrix_output.append(matrix_row)

    return matrix_output


def determine_long_lived_branch(branch_name: str) -> bool:
    # For long-lived branches (main, releases) we want to run both presubmit and postsubmit jobs on push,
    # instead of just presubmit jobs (as for other branches)
    is_long_lived_branch = False
    # Let's differentiate between full/complete matches and prefix matches for long-lived branches
    long_lived_full_match = ["main"]
    long_lived_prefix_match = ["release/therock-"]
    if branch_name in long_lived_full_match or any(
        branch_name.startswith(prefix) for prefix in long_lived_prefix_match
    ):
        is_long_lived_branch = True

    return is_long_lived_branch


def matrix_generator(
    is_pull_request=False,
    is_workflow_dispatch=False,
    is_push=False,
    is_schedule=False,
    base_args={},
    families={},
    platform="linux",
    multi_arch=False,
):
    """
    Generates a matrix of "family" and "test-runs-on" parameters based on the workflow inputs.
    Second return value is a list of test names to run, if any.
    """

    # Select target names based on inputs. Targets will be filtered by platform afterwards.
    selected_target_names = []
    # Select only test names based on label inputs, if applied. If no test labels apply, use default logic.
    selected_test_names = []

    branch_name = base_args.get("branch_name", "")
    # For long-lived branches (main, releases) we want to run both presubmit and postsubmit jobs on push,
    # instead of just presubmit jobs (as for other branches)
    is_long_lived_branch = determine_long_lived_branch(branch_name)

    print(f"* {branch_name} is considered a long-lived branch: {is_long_lived_branch}")

    # Determine which trigger types are active for proper matrix lookup
    active_trigger_types = []
    if is_pull_request:
        active_trigger_types.append("presubmit")
    if is_push:
        if is_long_lived_branch:
            active_trigger_types.extend(["presubmit", "postsubmit"])
        else:
            # Non-long-lived branch pushes (e.g., multi_arch/bringup1) use presubmit defaults
            active_trigger_types.append("presubmit")
    if is_schedule:
        active_trigger_types.extend(["presubmit", "postsubmit", "nightly"])

    # Get the appropriate family matrix based on active triggers
    # For workflow_dispatch and PR labels, we need to check all matrices
    presubmit_matrix = None
    if is_workflow_dispatch or is_pull_request:
        # For workflow_dispatch, check all possible matrices
        lookup_trigger_types = ["presubmit", "postsubmit", "nightly"]
        lookup_matrix = get_all_families_for_trigger_types(lookup_trigger_types)
        print(f"Using family matrix for trigger types: {lookup_trigger_types}")
        # For PR defaults we still need the presubmit set (subset of lookup_matrix keys).
        if is_pull_request:
            presubmit_matrix = get_all_families_for_trigger_types(["presubmit"])
    elif active_trigger_types:
        lookup_matrix = get_all_families_for_trigger_types(active_trigger_types)
        print(f"Using family matrix for trigger types: {active_trigger_types}")
    else:
        # This code path should never be reached in production workflows
        # as they only trigger on main branch pushes, PRs, workflow_dispatch, or schedule.
        # If this error is raised, it indicates an unexpected trigger combination.
        raise AssertionError(
            f"Unreachable code: no trigger types determined. "
            f"is_pull_request={is_pull_request}, is_workflow_dispatch={is_workflow_dispatch}, "
            f"is_push={is_push}, is_schedule={is_schedule}, "
            f"branch_name={branch_name}"
        )

    if is_workflow_dispatch:
        print(f"[WORKFLOW_DISPATCH] Generating build matrix with {str(base_args)}")

        # Parse GPU family inputs
        _add_gpu_families_from_input(families, lookup_matrix, selected_target_names)

        # If any workflow dispatch test labels are specified, we run full tests for those specific tests
        workflow_dispatch_test_labels_str = (
            base_args.get("workflow_dispatch_linux_test_labels", "")
            if platform == "linux"
            else base_args.get("workflow_dispatch_windows_test_labels", "")
        )
        # (ex: "test:rocprim, test:hipcub" -> ["test:rocprim", "test:hipcub"])
        workflow_dispatch_test_labels = [
            test_label.strip()
            for test_label in workflow_dispatch_test_labels_str.split(",")
        ]

        requested_test_names = []
        for label in workflow_dispatch_test_labels:
            if "test:" in label:
                _, test_name = label.split(":")
                requested_test_names.append(test_name)
        selected_test_names.extend(filter_known_names(requested_test_names, "test"))

    if is_pull_request:
        print(f"[PULL_REQUEST] Generating build matrix with {str(base_args)}")

        # Check if GPU families were explicitly provided (e.g., via workflow_call inputs)
        # If not, fall back to presubmit defaults
        assert (
            presubmit_matrix is not None
        ), "presubmit_matrix should be set for pull_request runs"
        _add_gpu_families_from_input(
            families,
            lookup_matrix,
            selected_target_names,
            fallback_targets=list(presubmit_matrix.keys()),
        )

        # Extend with any additional targets that PR labels opt-in to running.
        # TODO(#1097): This (or the code below) should handle opting in for
        #     a GPU family for only one platform (e.g. Windows but not Linux)
        requested_target_names = []
        requested_test_names = []
        pr_labels = get_pr_labels(base_args)
        for label in pr_labels:
            # if a GPU target label was added, we add the GPU target to the build and test matrix
            if "gfx" in label:
                target = label.split("-")[0]
                requested_target_names.append(target)
            # If a test label was added, we run the full test for the specified test
            if "test:" in label:
                _, test_name = label.split(":")
                requested_test_names.append(test_name)
            # If the "skip-ci" label was added, we skip all builds and tests
            # We don't want to check for anymore labels
            if "skip-ci" == label:
                selected_target_names = []
                selected_test_names = []
                break
            if "run-all-archs-ci" == label:
                # lookup_matrix already contains presubmit+postsubmit+nightly for PR runs.
                selected_target_names = list(lookup_matrix.keys())

            selected_target_names.extend(
                filter_known_names(requested_target_names, "target", lookup_matrix)
            )
            selected_test_names.extend(filter_known_names(requested_test_names, "test"))

    if is_push:
        if is_long_lived_branch:
            print(
                f"[PUSH - {branch_name.upper()}] Generating build matrix with {str(base_args)}"
            )

            # Add presubmit and postsubmit targets.
            for target in get_all_families_for_trigger_types(
                ["presubmit", "postsubmit"]
            ):
                selected_target_names.append(target)
        else:
            print(
                f"[PUSH - {branch_name}] Generating build matrix with {str(base_args)}"
            )

            # Non-long-lived branch pushes use presubmit targets
            for target in get_all_families_for_trigger_types(["presubmit"]):
                selected_target_names.append(target)

    if is_schedule:
        print(f"[SCHEDULE] Generating build matrix with {str(base_args)}")

        # For nightly runs, we run all builds and full tests
        amdgpu_family_info_matrix_all = get_all_families_for_trigger_types(
            ["presubmit", "postsubmit", "nightly"]
        )
        for key in amdgpu_family_info_matrix_all:
            selected_target_names.append(key)

    # Ensure the lists are unique
    unique_target_names = list(set(selected_target_names))
    unique_test_names = list(set(selected_test_names))

    platform_build_variants = all_build_variants.get(platform)
    assert isinstance(
        platform_build_variants, dict
    ), f"Expected build variant {platform} in {all_build_variants}"

    # In multi-arch mode, group all families into one entry per build_variant
    if multi_arch:
        matrix_output = generate_multi_arch_matrix(
            unique_target_names,
            lookup_matrix,
            platform,
            platform_build_variants,
            base_args,
        )
        print(f"Generated multi-arch build matrix: {str(matrix_output)}")
        print(f"Generated test list: {str(unique_test_names)}")
        return matrix_output, unique_test_names

    # Expand selected target names back to a matrix (cross-product of families × variants).
    matrix_output = []
    for target_name in unique_target_names:
        # Filter targets to only those matching the requested platform.
        # Use the trigger-appropriate lookup matrix
        platform_set = lookup_matrix.get(target_name)
        if platform in platform_set:
            platform_info = platform_set.get(platform)
            assert isinstance(platform_info, dict)

            # Further expand it based on build_variant.
            build_variant_names = platform_info.get("build_variants")
            assert isinstance(
                build_variant_names, list
            ), f"Expected 'build_variant' in platform: {platform_info}"
            for build_variant_name in build_variant_names:
                # We have custom build variants for specific CI flows.
                # For CI, we use the release build variant (for PRs, pushes to main, nightlies)
                # For CI ASAN, we use the ASAN build variant (for pushes to main)
                # In the case that the build variant is not requested, we skip it
                if build_variant_name != base_args.get("build_variant"):
                    continue

                # Merge platform_info and build_variant_info into a matrix_row.
                matrix_row = dict(platform_info)

                build_variant_info = platform_build_variants.get(build_variant_name)
                assert isinstance(
                    build_variant_info, dict
                ), f"Expected {build_variant_name} in {platform_build_variants} for {platform_info}"

                # If the build variant level notes expect_failure, set it on the overall row.
                # But if not, honor what is already there.
                if build_variant_info.get("expect_failure", False):
                    matrix_row["expect_failure"] = True
                del matrix_row["build_variants"]
                matrix_row.update(build_variant_info)

                # Assign a computed "artifact_group" combining the family and variant.
                artifact_group = platform_info["family"]
                build_variant_suffix = build_variant_info["build_variant_suffix"]
                if build_variant_suffix:
                    artifact_group += f"-{build_variant_suffix}"
                matrix_row["artifact_group"] = artifact_group

                # We retrieve labels from both PR and workflow_dispatch to customize the build and test jobs
                label_options = []
                label_options.extend(get_pr_labels(base_args))
                label_options.extend(
                    get_workflow_dispatch_additional_label_options(base_args)
                )
                for label in label_options:
                    # If a specific test kernel type was specified, we use that kernel-enabled test runners
                    # We disable the other machines that do not have the specified kernel type
                    # If a kernel test label was added, we set the test-runs-on accordingly to kernel-specific test machines
                    if "test_runner" in label:
                        _, kernel_type = label.split(":")
                        # If the architecture has a valid kernel machine, we set it here
                        if (
                            "test-runs-on-kernel" in platform_info
                            and kernel_type in platform_info["test-runs-on-kernel"]
                        ):
                            matrix_row["test-runs-on"] = platform_info[
                                "test-runs-on-kernel"
                            ][kernel_type]
                        # Otherwise, we disable the test runner for this architecture
                        else:
                            matrix_row["test-runs-on"] = ""
                            if "test-runs-on-multi-gpu" in platform_info:
                                matrix_row["test-runs-on-multi-gpu"] = ""
                        break

                matrix_output.append(matrix_row)

    print(f"Generated build matrix: {str(matrix_output)}")
    print(f"Generated test list: {str(unique_test_names)}")
    return matrix_output, unique_test_names


# --------------------------------------------------------------------------- #
# Core script logic
# --------------------------------------------------------------------------- #


def _extract_event_flags(base_args: dict) -> dict:
    """Extract and log event type flags.

    Args:
        base_args: Dictionary containing 'github_event_name'

    Returns:
        Dictionary with event name and boolean flags for each event type
    """
    github_event_name = base_args.get("github_event_name")
    flags = {
        "github_event_name": github_event_name,
        "is_push": github_event_name == "push",
        "is_workflow_dispatch": github_event_name == "workflow_dispatch",
        "is_pull_request": github_event_name == "pull_request",
        "is_schedule": github_event_name == "schedule",
    }
    print("Found metadata:")
    print(f"  github_event_name: {flags['github_event_name']}")
    print(f"  is_push: {flags['is_push']}")
    print(f"  is_workflow_dispatch: {flags['is_workflow_dispatch']}")
    print(f"  is_pull_request: {flags['is_pull_request']}")
    print("")
    return flags


def _generate_base_matrices(
    base_args: dict,
    linux_families: dict,
    windows_families: dict,
    event_flags: dict,
) -> tuple[list[dict], list, list[dict], list]:
    """Generate base GPU family matrices for both platforms.

    Args:
        base_args: Base arguments including 'multi_arch'
        linux_families: Linux GPU family input
        windows_families: Windows GPU family input
        event_flags: Dictionary with event type flags

    Returns:
        Tuple of (linux_variants, linux_tests, windows_variants, windows_tests)
    """
    multi_arch = base_args.get("multi_arch", False)

    print(
        f"Generating build matrix for Linux (multi_arch={multi_arch}): {str(linux_families)}"
    )
    linux_variants, linux_tests = matrix_generator(
        event_flags["is_pull_request"],
        event_flags["is_workflow_dispatch"],
        event_flags["is_push"],
        event_flags["is_schedule"],
        base_args,
        linux_families,
        platform="linux",
        multi_arch=multi_arch,
    )
    print("")

    print(
        f"Generating build matrix for Windows (multi_arch={multi_arch}): {str(windows_families)}"
    )
    windows_variants, windows_tests = matrix_generator(
        event_flags["is_pull_request"],
        event_flags["is_workflow_dispatch"],
        event_flags["is_push"],
        event_flags["is_schedule"],
        base_args,
        windows_families,
        platform="windows",
        multi_arch=multi_arch,
    )
    print("")

    print(
        f"Generated matrix sizes: Linux={len(linux_variants)} variants, Windows={len(windows_variants)} variants"
    )
    return linux_variants, linux_tests, windows_variants, windows_tests


def _apply_external_project_cross_product(
    base_args: dict,
    linux_variants: list[dict],
    windows_variants: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Cross-product external project configs with GPU families if applicable.

    Args:
        base_args: Dictionary containing optional external project configs
        linux_variants: Base Linux GPU family variants
        windows_variants: Base Windows GPU family variants

    Returns:
        Tuple of (linux_variants, windows_variants) after cross-product
    """
    linux_configs = base_args.get("linux_external_project_configs")
    windows_configs = base_args.get("windows_external_project_configs")

    if not (linux_configs or windows_configs):
        return linux_variants, windows_variants

    print("\n=== Cross-producting projects with GPU families ===")

    if linux_configs:
        linux_variants = _cross_product_projects_with_gpu_variants(
            linux_configs, linux_variants
        )

    if windows_configs:
        windows_variants = _cross_product_projects_with_gpu_variants(
            windows_configs, windows_variants
        )

    print(f"Final Linux matrix: {len(linux_variants)} entries")
    print(f"Final Windows matrix: {len(windows_variants)} entries")

    return linux_variants, windows_variants


def main(base_args, linux_families, windows_families):
    """Main orchestration function for CI configuration.

    Args:
        base_args: Dictionary with CI metadata (event name, base ref, etc.)
        linux_families: Linux GPU family input
        windows_families: Windows GPU family input
    """
    # Extract event flags
    event_flags = _extract_event_flags(base_args)

    # Check for external project configs
    has_external_configs = bool(
        base_args.get("linux_external_project_configs")
        or base_args.get("windows_external_project_configs")
    )
    if has_external_configs:
        linux_count = len(base_args.get("linux_external_project_configs", []))
        windows_count = len(base_args.get("windows_external_project_configs", []))
        print(
            f"Using external project configurations: Linux={linux_count}, Windows={windows_count}"
        )

    # Generate base matrices
    linux_variants, linux_tests, windows_variants, windows_tests = (
        _generate_base_matrices(
            base_args, linux_families, windows_families, event_flags
        )
    )

    # Apply external project cross-product if needed
    linux_variants, windows_variants = _apply_external_project_cross_product(
        base_args, linux_variants, windows_variants
    )

    # Determine build configuration and emit outputs
    enable_build_jobs, test_type = _determine_enable_build_jobs_and_test_type(
        github_event_name=event_flags["github_event_name"],
        is_schedule=event_flags["is_schedule"],
        base_ref=base_args.get("base_ref"),
        linux_test_output=linux_tests,
        windows_test_output=windows_tests,
        has_external_project_configs=has_external_configs,
    )

    _emit_summary_and_outputs(
        linux_variants_output=linux_variants,
        linux_test_output=linux_tests,
        windows_variants_output=windows_variants,
        windows_test_output=windows_tests,
        enable_build_jobs=enable_build_jobs,
        test_type=test_type,
        base_args=base_args,
    )


if __name__ == "__main__":
    run_from_env()
