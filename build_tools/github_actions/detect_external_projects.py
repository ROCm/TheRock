#!/usr/bin/env python3

"""Detects which projects to build/test in external repos based on changed files.

This script implements project-based change detection for external repositories
(rocm-libraries, rocm-systems) that use a subtree-based project structure.

Based on therock_configure_ci.py and therock_matrix.py from:
- https://github.com/ROCm/rocm-libraries
- https://github.com/ROCm/rocm-systems

----------
| Inputs |
----------

Environment variables:
  * GITHUB_EVENT_NAME    : GitHub event name (e.g. pull_request, push, schedule, workflow_dispatch)
  * PROJECTS (optional)  : Space-separated list of projects or 'all' (for workflow_dispatch)
  * PLATFORM             : Target platform ('linux' or 'windows')

Arguments:
  * --base-ref           : Git reference to diff against (e.g., HEAD^)
  * --repo-name          : Repository name (rocm-libraries or rocm-systems)

Local git history with at least fetch-depth of 2 for file diffing.

-----------
| Outputs |
-----------

Returns (via stdout as JSON):
  * projects: List of project configurations, each containing:
    - project_to_test: Comma-separated list of projects to test
    - cmake_options: Space-separated CMake configuration flags
    - artifact_group: Identifier for artifact grouping (same as project key)

-----------------
| How It Works |
-----------------

1. Determines changed files via git diff
2. Maps changed paths to projects using subtree_to_project_map from external_repo_project_maps.py
3. Resolves project dependencies and combines overlapping builds
4. Returns project-specific CMake options and test targets

For workflow_dispatch with 'all' or for schedule events, returns all projects.
For empty changes (e.g., doc-only PRs), returns empty list to skip builds.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

# Import common utilities from configure_ci
from configure_ci import (
    SKIPPABLE_PATH_PATTERNS,
    check_for_non_skippable_path,
    get_modified_paths,
    is_path_skippable,
)


def get_changed_subtrees(
    modified_paths: List[str], subtree_to_project_map: dict
) -> Set[str]:
    """Extracts subtree paths from modified files that match known project directories.

    Args:
        modified_paths: List of file paths changed in the commit/PR
        subtree_to_project_map: Mapping of subtree paths to project names

    Returns:
        Set of subtree paths that were changed and are in the project map
    """
    changed_subtrees = set()

    for path in modified_paths:
        # Check if this path matches any known subtree prefix
        for subtree in subtree_to_project_map.keys():
            if path.startswith(subtree + "/") or path == subtree:
                changed_subtrees.add(subtree)
                break

    return changed_subtrees


def detect_projects_from_changes(
    base_ref: str,
    repo_config: dict,
    platform: str,
    github_event_name: str,
    projects_input: str = "",
) -> List[dict]:
    """Main logic to detect which projects need to be built/tested.

    Args:
        base_ref: Git reference to diff against
        repo_config: Dictionary containing subtree_to_project_map, project_map, etc.
        platform: Target platform ('linux' or 'windows')
        github_event_name: GitHub event type
        projects_input: Optional project override from workflow_dispatch

    Returns:
        List of project configurations with cmake_options and project_to_test
    """
    subtree_to_project_map = repo_config["subtree_to_project_map"]

    # For scheduled builds, always build all projects
    if github_event_name == "schedule":
        print("Schedule event detected - building all projects")
        subtrees_to_build = set(subtree_to_project_map.keys())
    # For workflow_dispatch with explicit project list (override for testing)
    elif (
        github_event_name == "workflow_dispatch"
        and projects_input
        and projects_input.strip()
    ):
        projects_input = projects_input.strip()
        print(f"workflow_dispatch with projects override: '{projects_input}'")

        if projects_input.lower() == "all":
            print("Building all projects (override: 'all')")
            subtrees_to_build = set(subtree_to_project_map.keys())
        else:
            # Parse comma-separated project list (e.g., "projects/rocprim,projects/hipcub")
            requested_subtrees = [
                p.strip() for p in projects_input.split(",") if p.strip()
            ]
            subtrees_to_build = set()

            for subtree in requested_subtrees:
                # Normalize path separators
                subtree = subtree.replace("\\", "/")
                if subtree in subtree_to_project_map:
                    subtrees_to_build.add(subtree)
                else:
                    print(f"WARNING: Unknown project '{subtree}' - skipping")

            if not subtrees_to_build:
                print("No valid projects found in override - skipping all builds")
                return []
    # For PRs, pushes, and workflow_dispatch without project override - detect based on changed files
    else:
        print(f"Detecting changed files for event: {github_event_name}")
        modified_paths = get_modified_paths(base_ref)

        if modified_paths is None:
            print("ERROR: Could not determine modified paths")
            return []

        if not modified_paths:
            print("No files modified - skipping all builds")
            return []

        print(f"Found {len(modified_paths)} modified files")
        print(f"Modified paths (first 20): {modified_paths[:20]}")

        # Check if all changes are skippable (docs, markdown, etc.)
        if not check_for_non_skippable_path(modified_paths):
            print(
                "Only skippable paths modified (docs, markdown, etc.) - skipping all builds"
            )
            return []

        # Find which project subtrees were modified
        subtrees_to_build = get_changed_subtrees(modified_paths, subtree_to_project_map)

        if not subtrees_to_build:
            print("No project-related files changed - skipping builds")
            print(
                f"Note: Modified paths don't match any known projects in subtree_to_project_map"
            )
            print(f"Known project paths: {sorted(subtree_to_project_map.keys())[:10]}")
            return []

        print(f"Changed subtrees: {sorted(subtrees_to_build)}")

    # Import the project collection logic from centralized module
    from external_repo_project_maps import collect_projects_to_run

    # Call the collect_projects_to_run function with repo-specific config
    project_configs = collect_projects_to_run(
        subtrees=list(subtrees_to_build),
        platform=platform,
        subtree_to_project_map=repo_config["subtree_to_project_map"],
        project_map=repo_config["project_map"],
        additional_options=repo_config["additional_options"],
        dependency_graph=repo_config["dependency_graph"],
    )

    print(
        f"Generated {len(project_configs)} project configuration(s) for platform {platform}"
    )

    # Add artifact_group to each config (use the project key as the identifier)
    # The project configs from collect_projects_to_run already have cmake_options and project_to_test
    for config in project_configs:
        # For artifact_group, we'll use the first project in project_to_test
        # This groups artifacts by the primary project
        if isinstance(config.get("project_to_test"), str):
            first_project = config["project_to_test"].split(",")[0].strip()
        else:
            first_project = "unknown"

        config["artifact_group"] = first_project

    return project_configs


def main():
    parser = argparse.ArgumentParser(
        description="Detect which projects to build/test in external repos based on changed files"
    )
    parser.add_argument(
        "--base-ref",
        type=str,
        default="HEAD^",
        help="Git reference to diff against (default: HEAD^)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name (rocm-libraries or rocm-systems)",
    )

    args = parser.parse_args()

    # Get repo configuration from centralized module
    try:
        from external_repo_project_maps import get_repo_config

        repo_config = get_repo_config(args.repo_name)
        print(f"Loaded project configuration for: {args.repo_name}")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(
            f"ERROR: Failed to import external_repo_project_maps: {e}", file=sys.stderr
        )
        sys.exit(1)

    # Get inputs from environment
    github_event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    platform = os.environ.get("PLATFORM", "linux")
    projects_input = os.environ.get("PROJECTS", "")

    print(f"Event: {github_event_name}, Platform: {platform}")

    # Detect projects
    project_configs = detect_projects_from_changes(
        base_ref=args.base_ref,
        repo_config=repo_config,
        platform=platform,
        github_event_name=github_event_name,
        projects_input=projects_input,
    )

    # Output as JSON
    output = {
        "projects": project_configs,
    }

    print("\n=== Project Detection Results ===")
    print(json.dumps(output, indent=2))

    # Write to stdout for capture by calling script
    sys.stdout.flush()


if __name__ == "__main__":
    main()
