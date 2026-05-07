"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""

import os

subtree_to_project_map = {
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

project_map = {
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
            ],
            "windows": [
                "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
            ],
        },
        "project_to_test": ["miopen", "miopen_plugin"],
    },
    "fft": {
        "cmake_options": ["-DTHEROCK_ENABLE_FFT=ON", "-DTHEROCK_ENABLE_RAND=ON"],
        "project_to_test": ["hipfft", "rocfft"],
    },
    "hipdnn": {  # due to MIOpen plugin project being inside the hipDNN directory, we cannot have the MIOpen plugin project as a separate project for now https://github.com/ROCm/rocm-libraries/issues/2316
        "cmake_options": ["-DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON"],
        "additional_flags": {
            # As composable_kernel is not enabled for Windows, we only enable these flags during Linux builds
            "linux": [
                "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
            ],
            "windows": [
                "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON",
                "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
            ],
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
# However, if we run a separate build for optional components, those files will be overriden as these components share the same umbrella as other projects
# Example: SPARSE is included in BLAS, but a separate build would cause overwriting of the blas_lib.tar.xz and blas_test.tar.xz and be missing libraries and tests
additional_options = {
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
dependency_graph = {
    "miopen": ["blas", "rand"],
}


def collect_projects_to_run(subtrees):
    platform = os.getenv("PLATFORM")
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
                project_map[project_to_add]["cmake_options"] = project_options_to_add[
                    "cmake_options"
                ]
                project_map[project_to_add]["project_to_test"] = project_options_to_add[
                    "project_to_test"
                ]

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
            project_map_data["cmake_options"].extend(["-DTHEROCK_ENABLE_ALL=OFF"])

            cmake_flag_options = " ".join(project_map_data["cmake_options"])
            project_to_test_options = ",".join(project_map_data["project_to_test"])
            project_map_data["cmake_options"] = cmake_flag_options
            project_map_data["project_to_test"] = project_to_test_options
            project_to_run.append(project_map_data)

    return project_to_run
