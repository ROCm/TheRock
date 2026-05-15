#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Determine build and test configuration for specific projects.

Given a list of project names, this script outputs:
1. CMake flags to enable only those projects (and dependencies)
2. Test labels to run for those projects

The project-to-feature mapping is parsed from CMakeLists.txt files by looking
for therock_cmake_subproject_declare() calls inside if(THEROCK_ENABLE_*) blocks.

Example:
    # Get build config for rocprim
    python build_tools/determine_build_config.py --projects rocprim

    # Get config for multiple projects
    python build_tools/determine_build_config.py --projects rocprim rocblas

    # Output for GitHub Actions
    python build_tools/determine_build_config.py --projects rocprim --gha-output

    # List all known projects and their features
    python build_tools/determine_build_config.py --list-projects
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "test_tools"))

from _therock_utils.build_topology import BuildTopology
from github_actions.github_actions_api import gha_set_output
from determine_rocm_test_dependencies import get_subprojects_to_test


def parse_project_to_feature_map(therock_dir: Path) -> Dict[str, str]:
    """Parse CMakeLists.txt files to build project -> feature mapping.

    Looks for patterns like:
        if(THEROCK_ENABLE_PRIM)
          ...
          therock_cmake_subproject_declare(rocPRIM ...)
          ...
        endif()

    Returns dict mapping lowercase project name -> feature name (e.g., "rocprim" -> "PRIM")
    """
    project_to_feature: Dict[str, str] = {}
    cmake_files = list(therock_dir.rglob("CMakeLists.txt"))

    for cmake_file in cmake_files:
        content = cmake_file.read_text()

        # Find all if(THEROCK_ENABLE_*) blocks and track current feature context
        # We use a simple state machine approach
        lines = content.split("\n")
        current_feature = None
        depth = 0

        for line in lines:
            # Check for if(THEROCK_ENABLE_*)
            enable_match = re.match(
                r"\s*if\s*\(\s*THEROCK_ENABLE_(\w+)\s*\)", line, re.IGNORECASE
            )
            if enable_match:
                if depth == 0:
                    current_feature = enable_match.group(1)
                depth += 1
                continue

            # Check for endif
            if re.match(r"\s*endif\s*\(", line, re.IGNORECASE):
                depth -= 1
                if depth == 0:
                    current_feature = None
                continue

            # Check for therock_cmake_subproject_declare
            if current_feature:
                declare_match = re.match(
                    r"\s*therock_cmake_subproject_declare\s*\(\s*(\w+)", line
                )
                if declare_match:
                    project_name = declare_match.group(1).lower()
                    project_to_feature[project_name] = current_feature

    return project_to_feature


# Fallback mapping for projects in subdirectories (add_subdirectory)
# These are not directly inside if(THEROCK_ENABLE_*) blocks
SUBDIRECTORY_PROJECT_TO_FEATURE: Dict[str, str] = {
    # BLAS subdirectory projects
    "rocblas": "BLAS",
    "rocblas_tests": "BLAS",
    "hipblas": "BLAS",
    "hipblas_tests": "BLAS",
    "hipblaslt": "BLAS",
    "hipblaslt_tests": "BLAS",
    "hipsparselt": "BLAS",
    "rocroller": "BLAS",
    # SPARSE subdirectory projects
    "rocsparse": "SPARSE",
    "rocsparse_tests": "SPARSE",
    "hipsparse": "SPARSE",
    "hipsparse_tests": "SPARSE",
    # SOLVER subdirectory projects
    "rocsolver": "SOLVER",
    "rocsolver_tests": "SOLVER",
    "hipsolver": "SOLVER",
    "hipsolver_tests": "SOLVER",
    # PRIM projects (rocthrust, hipcub are in prim block)
    "rocthrust": "PRIM",
    "rocthrust_tests": "PRIM",
    "hipcub": "PRIM",
    "hipcub_tests": "PRIM",
    # FFT projects
    "rocfft": "FFT",
    "rocfft_tests": "FFT",
    "hipfft": "FFT",
    "hipfft_tests": "FFT",
    # ML libs
    "composable_kernel": "COMPOSABLE_KERNEL",
    "miopen": "MIOPEN",
    "miopen_tests": "MIOPEN",
    "hipdnn": "HIPDNN",
    "miopenprovider": "MIOPENPROVIDER",
    # Comm libs
    "rccl": "RCCL",
    "rccl_tests": "RCCL",
    "rocshmem": "ROCSHMEM",
    # Other
    "rocwmma": "ROCWMMA",
    "rocwmma_tests": "ROCWMMA",
    "libhipcxx": "LIBHIPCXX",
}

# Cache for parsed mapping
_PROJECT_TO_FEATURE_CACHE: Dict[str, str] | None = None


def get_project_to_feature_map(therock_dir: Path = None) -> Dict[str, str]:
    """Get project -> feature mapping, combining parsed and fallback mappings."""
    global _PROJECT_TO_FEATURE_CACHE
    if _PROJECT_TO_FEATURE_CACHE is None:
        if therock_dir is None:
            therock_dir = Path(__file__).parent.parent
        # Start with fallback, then override with parsed (parsed takes precedence)
        _PROJECT_TO_FEATURE_CACHE = dict(SUBDIRECTORY_PROJECT_TO_FEATURE)
        _PROJECT_TO_FEATURE_CACHE.update(parse_project_to_feature_map(therock_dir))
    return _PROJECT_TO_FEATURE_CACHE


# Feature dependencies - if you enable X, you also need Y
# These are based on artifact_deps in BUILD_TOPOLOGY.toml
FEATURE_DEPS: Dict[str, List[str]] = {
    "SPARSE": ["BLAS", "PRIM"],
    "SOLVER": ["BLAS", "PRIM", "SPARSE"],
    "MIOPEN": ["BLAS", "COMPOSABLE_KERNEL", "RAND"],
    "MIOPENPROVIDER": ["MIOPEN", "HIPDNN"],
    "FFT": ["RAND"],
}


def get_topology() -> BuildTopology:
    """Load BUILD_TOPOLOGY.toml."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    topology_path = repo_root / "BUILD_TOPOLOGY.toml"
    return BuildTopology(str(topology_path))


def resolve_feature_deps(features: Set[str]) -> Set[str]:
    """Resolve feature dependencies recursively."""
    result = set(features)
    changed = True
    while changed:
        changed = False
        for feature in list(result):
            if feature in FEATURE_DEPS:
                for dep in FEATURE_DEPS[feature]:
                    if dep not in result:
                        result.add(dep)
                        changed = True
    return result


def get_features_for_projects(
    projects: List[str], therock_dir: Path = None
) -> Set[str]:
    """Get the set of features needed to build the given projects."""
    project_to_feature = get_project_to_feature_map(therock_dir)
    features = set()
    for project in projects:
        project_lower = project.lower()
        if project_lower in project_to_feature:
            features.add(project_to_feature[project_lower])
        else:
            print(f"Warning: Unknown project '{project}'", file=sys.stderr)
    return resolve_feature_deps(features)


def generate_cmake_args(features: Set[str]) -> List[str]:
    """Generate CMake arguments to enable specific features."""
    args = ["-DTHEROCK_ENABLE_ALL=OFF"]
    for feature in sorted(features):
        args.append(f"-DTHEROCK_ENABLE_{feature}=ON")
    return args


def get_build_and_test_config(
    projects: List[str], therock_dir: Path = None
) -> Dict[str, any]:
    """Get complete build and test configuration for projects.

    Returns dict with:
        - projects: input project list
        - features: list of features to enable
        - cmake_args: list of cmake arg strings
        - cmake_args_str: space-separated cmake args
        - test_projects: list of projects to test
        - test_labels: comma-separated test labels
    """
    if therock_dir is None:
        therock_dir = Path(__file__).parent.parent

    features = get_features_for_projects(projects, therock_dir)
    cmake_args = generate_cmake_args(features)
    test_projects = get_subprojects_to_test(projects, therock_dir)

    return {
        "projects": projects,
        "features": sorted(features),
        "cmake_args": cmake_args,
        "cmake_args_str": " ".join(cmake_args),
        "test_projects": sorted(test_projects),
        "test_labels": ",".join(f"test:{p}" for p in sorted(test_projects)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Determine build and test configuration for specific projects"
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        help="Project names (e.g., rocprim, rocblas)",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all known projects",
    )
    parser.add_argument(
        "--gha-output",
        action="store_true",
        help="Output for GitHub Actions",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    therock_dir = Path(__file__).parent.parent

    if args.list_projects:
        project_to_feature = get_project_to_feature_map(therock_dir)
        print("Known projects (parsed from CMakeLists.txt):")
        for project in sorted(project_to_feature.keys()):
            print(f"  {project} -> THEROCK_ENABLE_{project_to_feature[project]}")
        return 0

    if not args.projects:
        parser.error("--projects is required unless --list-projects is specified")

    result = get_build_and_test_config(args.projects, therock_dir)

    if args.gha_output:
        gha_set_output(
            {
                "cmake_args": result["cmake_args_str"],
                "test_labels": result["test_labels"],
                "features": ",".join(result["features"]),
            }
        )
    elif args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Projects: {', '.join(args.projects)}")
        print(f"Features: {', '.join(result['features'])}")
        print(f"CMake args: {result['cmake_args_str']}")
        print(f"Test projects: {', '.join(result['test_projects'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
