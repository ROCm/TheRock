#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Determine build and test configuration for specific projects.

Given a list of project names, this script outputs:
1. CMake flags to enable only those projects (and dependencies)
2. Test labels to run for those projects

Example:
    # Get build config for rocprim
    python build_tools/determine_build_config.py --projects rocprim

    # Get config for multiple projects
    python build_tools/determine_build_config.py --projects rocprim rocblas

    # Output for GitHub Actions
    python build_tools/determine_build_config.py --projects rocprim --gha-output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "test_tools"))

from _therock_utils.build_topology import BuildTopology
from github_actions.github_actions_api import gha_set_output
from determine_rocm_test_dependencies import get_subprojects_to_test

# Map from project/subproject names to their THEROCK_ENABLE_* feature names
# This maps the lowercase project names used in CMakeLists.txt to feature flags
PROJECT_TO_FEATURE: Dict[str, str] = {
    # Math libs - prim group
    "rocprim": "PRIM",
    "rocprim_tests": "PRIM",
    "rocthrust": "PRIM",
    "rocthrust_tests": "PRIM",
    "hipcub": "PRIM",
    "hipcub_tests": "PRIM",
    # Math libs - rand group
    "rocrand": "RAND",
    "rocrand_tests": "RAND",
    "hiprand": "RAND",
    "hiprand_tests": "RAND",
    # Math libs - fft group
    "rocfft": "FFT",
    "rocfft_tests": "FFT",
    "hipfft": "FFT",
    "hipfft_tests": "FFT",
    # Math libs - blas group
    "rocblas": "BLAS",
    "rocblas_tests": "BLAS",
    "hipblas": "BLAS",
    "hipblas_tests": "BLAS",
    "hipblaslt": "BLAS",
    "hipblaslt_tests": "BLAS",
    "hipsparselt": "BLAS",
    "rocroller": "BLAS",
    # Math libs - sparse group
    "rocsparse": "SPARSE",
    "rocsparse_tests": "SPARSE",
    "hipsparse": "SPARSE",
    "hipsparse_tests": "SPARSE",
    # Math libs - solver group
    "rocsolver": "SOLVER",
    "rocsolver_tests": "SOLVER",
    "hipsolver": "SOLVER",
    "hipsolver_tests": "SOLVER",
    # Math libs - other
    "rocwmma": "ROCWMMA",
    "rocwmma_tests": "ROCWMMA",
    "libhipcxx": "LIBHIPCXX",
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
}

# Feature dependencies - if you enable X, you also need Y
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


def get_features_for_projects(projects: List[str]) -> Set[str]:
    """Get the set of features needed to build the given projects."""
    features = set()
    for project in projects:
        project_lower = project.lower()
        if project_lower in PROJECT_TO_FEATURE:
            features.add(PROJECT_TO_FEATURE[project_lower])
        else:
            print(f"Warning: Unknown project '{project}'", file=sys.stderr)
    return resolve_feature_deps(features)


def generate_cmake_args(features: Set[str]) -> List[str]:
    """Generate CMake arguments to enable specific features."""
    args = ["-DTHEROCK_ENABLE_ALL=OFF"]
    for feature in sorted(features):
        args.append(f"-DTHEROCK_ENABLE_{feature}=ON")
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Determine build and test configuration for specific projects"
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        required=True,
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

    if args.list_projects:
        print("Known projects:")
        for project in sorted(PROJECT_TO_FEATURE.keys()):
            print(f"  {project} -> THEROCK_ENABLE_{PROJECT_TO_FEATURE[project]}")
        return 0

    # Get features and cmake args
    features = get_features_for_projects(args.projects)
    cmake_args = generate_cmake_args(features)

    # Get test dependencies
    therock_dir = Path(__file__).parent.parent
    test_projects = get_subprojects_to_test(args.projects, therock_dir)

    result = {
        "projects": args.projects,
        "features": sorted(features),
        "cmake_args": cmake_args,
        "cmake_args_str": " ".join(cmake_args),
        "test_projects": sorted(test_projects),
        "test_labels": ",".join(f"test:{p}" for p in sorted(test_projects)),
    }

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
        print(f"Features: {', '.join(sorted(features))}")
        print(f"CMake args: {' '.join(cmake_args)}")
        print(f"Test projects: {', '.join(sorted(test_projects))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
