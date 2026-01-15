#!/usr/bin/env python3

"""Project mapping configurations for external repositories.

This module defines how file changes in external repos (rocm-libraries, rocm-systems)
map to build configurations. These mappings determine:
- Which projects to build based on changed files
- What CMake options to use for each project
- What tests to run for each project

Based on configuration originally in:
- ROCm/rocm-libraries/.github/scripts/therock_matrix.py
- ROCm/rocm-systems/.github/scripts/therock_matrix.py

These maps should be kept in sync with the actual project structure in those repos.
Unit tests verify that the paths referenced here actually exist.
"""

import fnmatch
import subprocess
import sys
from typing import Iterable, Optional, Set

# =============================================================================
# ROCm Libraries Project Maps
# =============================================================================

ROCM_LIBRARIES_SUBTREE_TO_PROJECT_MAP = {
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hipdnn": "hipdnn",
    "projects/hipfft": "fft",
    "projects/hiprand": "rand",
    "projects/hipsolver": "solver",
    "projects/hipsparse": "sparse",
    "projects/hipsparselt": "sparse",
    "projects/miopen": "miopen",
    "projects/rocblas": "blas",
    "projects/rocfft": "fft",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocsolver": "solver",
    "projects/rocsparse": "sparse",
    "projects/rocthrust": "prim",
    "projects/rocwmma": "rocwmma",
    "shared/mxdatagenerator": "blas",
    "shared/origami": "blas",
    "shared/rocroller": "blas",
    "shared/tensile": "blas",
}

ROCM_LIBRARIES_PROJECT_MAP = {
    "prim": {
        "cmake_options": ["-DTHEROCK_ENABLE_PRIM=ON"],
        "project_to_test": ["rocprim", "rocthrust", "hipcub"],
    },
    "rand": {
        "cmake_options": ["-DTHEROCK_ENABLE_RAND=ON"],
        "project_to_test": ["rocrand", "hiprand"],
    },
    "blas": {
        "cmake_options": ["-DTHEROCK_ENABLE_BLAS=ON"],
        "project_to_test": ["hipblaslt", "rocblas", "hipblas", "rocroller"],
    },
    "miopen": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPEN=ON",
            "-DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../source-repo/projects/composablekernel",
        ],
        "project_to_test": ["miopen", "miopen_plugin"],
    },
    "fft": {
        "cmake_options": ["-DTHEROCK_ENABLE_FFT=ON", "-DTHEROCK_ENABLE_RAND=ON"],
        "project_to_test": ["hipfft", "rocfft"],
    },
    "hipdnn": {  # due to MIOpen plugin project being inside the hipDNN directory
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../source-repo/projects/composablekernel",
        ],
        "project_to_test": ["hipdnn", "miopen_plugin"],
    },
    "rocwmma": {
        "cmake_options": ["-DTHEROCK_ENABLE_ROCWMMA=ON"],
        "project_to_test": ["rocwmma"],
    },
}

# For certain math components, they are optional during building and testing.
# As they are optional, we do not want to include them as default as this takes more time in the CI.
# However, if we run a separate build for optional components, those files will be overridden as
# these components share the same umbrella as other projects.
# Example: SPARSE is included in BLAS, but a separate build would cause overwriting of the
# blas_lib.tar.xz and blas_test.tar.xz and be missing libraries and tests
ROCM_LIBRARIES_ADDITIONAL_OPTIONS = {
    "sparse": {
        "cmake_options": ["-DTHEROCK_ENABLE_SPARSE=ON"],
        "project_to_test": ["rocsparse", "hipsparse", "hipsparselt"],
        "project_to_add": "blas",
    },
    "solver": {
        "cmake_options": ["-DTHEROCK_ENABLE_SOLVER=ON"],
        "project_to_test": ["rocsolver", "hipsolver"],
        "project_to_add": "blas",
    },
}

# If a project has dependencies that are also being built, we combine build options and test options
# This way, there will be no S3 upload overlap and we save redundant builds
ROCM_LIBRARIES_DEPENDENCY_GRAPH = {
    "miopen": ["blas", "rand"],
}


# =============================================================================
# ROCm Systems Project Maps
# =============================================================================

ROCM_SYSTEMS_SUBTREE_TO_PROJECT_MAP = {
    "projects/aqlprofile": "profiler",
    "projects/clr": "core",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/hipother": "core",
    "projects/rdc": "rdc",
    "projects/rocm-core": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocminfo": "core",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocprofiler": "profiler",
    "projects/rocr-runtime": "core",
    "projects/roctracer": "profiler",
}

ROCM_SYSTEMS_PROJECT_MAP = {
    "core": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
        ],
        "project_to_test": ["hip-tests"],
    },
    "profiler": {
        "cmake_options": ["-DTHEROCK_ENABLE_PROFILER=ON"],
        "project_to_test": ["rocprofiler-tests"],
    },
    "rdc": {
        "cmake_options": ["-DTHEROCK_ENABLE_RDC=ON"],
        "project_to_test": ["rdc-tests"],
    },
    "all": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_PROFILER=ON",
        ],
        "project_to_test": ["hip-tests", "rocprofiler-tests"],
    },
}

ROCM_SYSTEMS_ADDITIONAL_OPTIONS = {}
ROCM_SYSTEMS_DEPENDENCY_GRAPH = {}


# =============================================================================
# Project Collection Logic (shared by both repos)
# =============================================================================


def collect_projects_to_run(
    subtrees: list,
    platform: str,
    repo_name: str,
) -> list:
    """Collects projects to run based on changed subtrees.

    This function implements the core logic from the external repos' therock_matrix.py
    collect_projects_to_run() function.

    Args:
        subtrees: List of changed subtree paths (e.g., ["projects/rocprim"])
        platform: Target platform ("linux" or "windows")
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")

    Returns:
        List of project configurations with cmake_options and project_to_test
    """
    import copy

    # Get repository configuration
    repo_config = get_repo_config(repo_name)
    subtree_to_project_map = repo_config["subtree_to_project_map"]
    project_map = repo_config["project_map"]
    additional_options = repo_config["additional_options"]
    dependency_graph = repo_config["dependency_graph"]

    # Create a deep copy to avoid modifying the original
    project_map = copy.deepcopy(project_map)

    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))

    for project in list(projects):
        # Check if an optional math component was included.
        if project in additional_options:
            project_options_to_add = additional_options[project]

            project_to_add = project_options_to_add["project_to_add"]
            # If `project_to_add` is in included, add options to the existing `project_map` entry
            if project_to_add in projects:
                project_map[project_to_add]["cmake_options"].extend(
                    project_options_to_add["cmake_options"]
                )
                project_map[project_to_add]["project_to_test"].extend(
                    project_options_to_add["project_to_test"]
                )
            # If `project_to_add` is not included, only run build and tests for the optional project
            else:
                projects.add(project_to_add)
                project_map[project_to_add] = {
                    "cmake_options": project_options_to_add["cmake_options"][:],
                    "project_to_test": project_options_to_add["project_to_test"][:],
                }

    # Check for potential dependencies
    to_remove_from_project_map = []
    for project in list(projects):
        # Check if project has a dependency combine
        if project in dependency_graph:
            for dependency in dependency_graph[project]:
                # If the dependency is also included, let's combine to avoid overlap
                if dependency in projects:
                    project_map[project]["cmake_options"].extend(
                        project_map[dependency]["cmake_options"]
                    )
                    project_map[project]["project_to_test"].extend(
                        project_map[dependency]["project_to_test"]
                    )
                    to_remove_from_project_map.append(dependency)

    # if dependency is included in projects and parent is found, we delete the dependency as the parent will build and test
    for to_remove_item in to_remove_from_project_map:
        projects.remove(to_remove_item)
        del project_map[to_remove_item]

    # retrieve the subtrees to checkout, cmake options to build, and projects to test
    project_to_run = []
    for project in projects:
        if project in project_map:
            project_map_data = project_map.get(project)

            # Check if platform-based additional flags are needed
            if (
                "additional_flags" in project_map_data
                and platform in project_map_data["additional_flags"]
            ):
                project_map_data["cmake_options"].extend(
                    project_map_data["additional_flags"][platform]
                )

            # To save time, only build what is needed
            project_map_data["cmake_options"].append("-DTHEROCK_ENABLE_ALL=OFF")

            cmake_flag_options = " ".join(project_map_data["cmake_options"])
            project_to_test_options = ",".join(project_map_data["project_to_test"])

            project_to_run.append(
                {
                    "cmake_options": cmake_flag_options,
                    "project_to_test": project_to_test_options,
                }
            )

    return project_to_run


def get_repo_config(repo_name: str) -> dict:
    """Returns the project map configuration for a given repository.

    Args:
        repo_name: Repository name ("rocm-libraries" or "rocm-systems")

    Returns:
        Dictionary containing subtree_to_project_map, project_map,
        additional_options, and dependency_graph
        Example:
            {
                "subtree_to_project_map": {"projects/rocprim": "prim", ...},
                "project_map": {"prim": {"cmake_options": [...], "project_to_test": [...]}, ...},
                "additional_options": {"solver": {...}, ...},
                "dependency_graph": {"miopen": ["blas", "rand"], ...},
            }

    Raises:
        ValueError: If repo_name is not recognized
    """
    if "rocm-libraries" in repo_name.lower():
        return {
            "subtree_to_project_map": ROCM_LIBRARIES_SUBTREE_TO_PROJECT_MAP,
            "project_map": ROCM_LIBRARIES_PROJECT_MAP,
            "additional_options": ROCM_LIBRARIES_ADDITIONAL_OPTIONS,
            "dependency_graph": ROCM_LIBRARIES_DEPENDENCY_GRAPH,
        }
    elif "rocm-systems" in repo_name.lower():
        return {
            "subtree_to_project_map": ROCM_SYSTEMS_SUBTREE_TO_PROJECT_MAP,
            "project_map": ROCM_SYSTEMS_PROJECT_MAP,
            "additional_options": ROCM_SYSTEMS_ADDITIONAL_OPTIONS,
            "dependency_graph": ROCM_SYSTEMS_DEPENDENCY_GRAPH,
        }
    else:
        raise ValueError(f"Unknown repository: {repo_name}")


# =============================================================================
# Change detection (shared logic)
# =============================================================================


def get_modified_paths(base_ref: str) -> Optional[list[str]]:
    """Returns modified paths relative to `base_ref` (via `git diff --name-only`)."""
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
            "Computing modified files timed out. Not using PR diff to determine projects.",
            file=sys.stderr,
        )
        return None


# Paths matching any of these patterns are considered to have no influence over
# external repo project selection (docs, markdown, etc).
SKIPPABLE_PATH_PATTERNS = [
    "docs/*",
    "*.gitignore",
    "*.md",
    "*.pre-commit-config.*",
    "*CODEOWNERS",
    "*LICENSE",
]


def is_path_skippable(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in SKIPPABLE_PATH_PATTERNS)


def check_for_non_skippable_path(paths: Optional[Iterable[str]]) -> bool:
    if paths is None:
        return False
    return any(not is_path_skippable(p) for p in paths)


def get_changed_subtrees(
    modified_paths: list[str], subtree_to_project_map: dict
) -> Set[str]:
    """Returns subtree roots that were touched by modified paths."""
    changed_subtrees: Set[str] = set()
    for path in modified_paths:
        for subtree in subtree_to_project_map.keys():
            if path.startswith(subtree + "/") or path == subtree:
                changed_subtrees.add(subtree)
                break
    return changed_subtrees


def detect_projects_from_changes(
    *,
    repo_name: str,
    base_ref: str,
    github_event_name: str,
    projects_input: str = "",
) -> dict:
    """Detects per-platform project configs for external repos based on changes.

    Returns:
        Dict with keys:
          - linux_projects: list[dict]
          - windows_projects: list[dict]
    """
    repo_config = get_repo_config(repo_name)
    subtree_to_project_map = repo_config["subtree_to_project_map"]

    # For scheduled builds, always build all projects
    if github_event_name == "schedule":
        print("Schedule event detected - building all projects")
        subtrees_to_build = set(subtree_to_project_map.keys())
    # For workflow_dispatch or when PROJECTS is explicitly set (e.g., via workflow_call)
    elif projects_input and projects_input.strip():
        projects_input = projects_input.strip()
        print(f"Projects override specified: '{projects_input}'")

        if projects_input.lower() == "all":
            print("Building all projects (override: 'all')")
            subtrees_to_build = set(subtree_to_project_map.keys())
        else:
            requested_subtrees = [
                p.strip() for p in projects_input.split(",") if p.strip()
            ]
            subtrees_to_build = set()
            for subtree in requested_subtrees:
                subtree = subtree.replace("\\", "/")
                if subtree in subtree_to_project_map:
                    subtrees_to_build.add(subtree)
                else:
                    print(f"WARNING: Unknown project '{subtree}' - skipping")

            if not subtrees_to_build:
                print("No valid projects found in override - skipping all builds")
                return {"linux_projects": [], "windows_projects": []}
    else:
        print(f"Detecting changed files for event: {github_event_name}")
        modified_paths = get_modified_paths(base_ref)

        if modified_paths is None:
            print("ERROR: Could not determine modified paths")
            return {"linux_projects": [], "windows_projects": []}
        if not modified_paths:
            print("No files modified - skipping all builds")
            return {"linux_projects": [], "windows_projects": []}

        print(f"Found {len(modified_paths)} modified files")
        print(f"Modified paths (first 20): {modified_paths[:20]}")

        if not check_for_non_skippable_path(modified_paths):
            print("Only skippable paths modified - skipping all builds")
            return {"linux_projects": [], "windows_projects": []}

        subtrees_to_build = get_changed_subtrees(modified_paths, subtree_to_project_map)
        if not subtrees_to_build:
            print("No project-related files changed - skipping builds")
            return {"linux_projects": [], "windows_projects": []}

        print(f"Changed subtrees: {sorted(subtrees_to_build)}")

    linux_configs = collect_projects_to_run(
        subtrees=list(subtrees_to_build),
        platform="linux",
        repo_name=repo_name,
    )

    windows_configs = collect_projects_to_run(
        subtrees=list(subtrees_to_build),
        platform="windows",
        repo_name=repo_name,
    )

    # Add artifact_group to each config
    for configs in (linux_configs, windows_configs):
        for cfg in configs:
            first_project = str(cfg.get("project_to_test", "")).split(",")[0].strip()
            cfg["artifact_group"] = first_project or "unknown"

    return {"linux_projects": linux_configs, "windows_projects": windows_configs}
