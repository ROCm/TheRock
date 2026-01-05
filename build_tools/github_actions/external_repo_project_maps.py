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
        ],
        "additional_flags": {
            # As composable_kernel is not enabled for Windows, we only enable these flags during Linux builds
            "linux": [
                "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
            ]
        },
        "project_to_test": ["miopen", "miopen_plugin"],
    },
    "fft": {
        "cmake_options": ["-DTHEROCK_ENABLE_FFT=ON", "-DTHEROCK_ENABLE_RAND=ON"],
        "project_to_test": ["hipfft", "rocfft"],
    },
    "hipdnn": {  # due to MIOpen plugin project being inside the hipDNN directory
        "cmake_options": ["-DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON"],
        "additional_flags": {
            # As composable_kernel is not enabled for Windows, we only enable these flags during Linux builds
            "linux": [
                "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
            ]
        },
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
        "cmake_options": ["-DTHEROCK_ENABLE_CORE=ON", "-DTHEROCK_ENABLE_HIP_RUNTIME=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "project_to_test": ["hip-tests"],
    },
    "profiler": {
        "cmake_options": ["-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "project_to_test": ["rocprofiler-tests"],
    },
    "rdc": {
        "cmake_options": ["-DTHEROCK_ENABLE_RDC=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "project_to_test": ["rdc-tests"],
    },
    "all": {
        "cmake_options": ["-DTHEROCK_ENABLE_CORE=ON", "-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "project_to_test": ["hip-tests", "rocprofiler-tests"],
    },
}

ROCM_SYSTEMS_ADDITIONAL_OPTIONS = {}
ROCM_SYSTEMS_DEPENDENCY_GRAPH = {}


# =============================================================================
# GPU Family Matrices (all_build_variants)
# =============================================================================
# These are imported by configure_ci.py instead of the deleted therock_matrix.py files

# Note: External repos typically use simpler GPU family matrices than TheRock
# For now, they can use TheRock's defaults from amdgpu_family_matrix.py
# If they need custom GPU family matrices, add them here

ROCM_LIBRARIES_ALL_BUILD_VARIANTS = None  # Use TheRock defaults
ROCM_SYSTEMS_ALL_BUILD_VARIANTS = None  # Use TheRock defaults


# =============================================================================
# Project Collection Logic (shared by both repos)
# =============================================================================

def collect_projects_to_run(
    subtrees: list,
    platform: str,
    subtree_to_project_map: dict,
    project_map: dict,
    additional_options: dict,
    dependency_graph: dict,
) -> list:
    """Collects projects to run based on changed subtrees.
    
    This function implements the core logic from the external repos' therock_matrix.py
    collect_projects_to_run() function.
    
    Args:
        subtrees: List of changed subtree paths (e.g., ["projects/rocprim"])
        platform: Target platform ("linux" or "windows")
        subtree_to_project_map: Mapping of subtree paths to project names
        project_map: Mapping of project names to build configurations
        additional_options: Optional components that get merged into other projects
        dependency_graph: Project dependencies that should be combined
        
    Returns:
        List of project configurations with cmake_options and project_to_test
    """
    import copy
    
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
            
            project_to_run.append({
                "cmake_options": cmake_flag_options,
                "project_to_test": project_to_test_options,
            })

    return project_to_run


def get_repo_config(repo_name: str) -> dict:
    """Returns the project map configuration for a given repository.
    
    Args:
        repo_name: Repository name ("rocm-libraries" or "rocm-systems")
        
    Returns:
        Dictionary containing subtree_to_project_map, project_map, 
        additional_options, and dependency_graph
        
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
