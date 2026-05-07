#!/usr/bin/env python3

"""
Azure Pipeline Resolver Script
------------------------------
This script determines which Azure pipelines to run based on changed subtrees.
Using a predefined dependency map, the script resolves which projects need to be processed,
skipping those that will be covered by their dependencies.

Steps:
    1. Load a list of changed projects from a file.
    2. Consult a dependency map to determine transitive and direct dependencies.
    3. Identify projects that should be processed, excluding those handled by dependencies.
    4. Output the list of projects to be run, along with their Azure pipeline IDs.

Arguments:
    --subtree-file : Path to the file containing a newline-separated list of changed subtrees.

Outputs:
    Prints a newline-separated list of "project_name=definition_id" for the projects that need
    to be processed, where `definition_id` is the Azure pipeline ID associated with the project.

Example Usage:
    To determine which pipelines to run given the changed subtrees listed in a file:
        python azure_pipeline_resolver.py --subtree-file changed_subtrees.txt
"""

import argparse
from typing import List, Optional


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Given a list of changed subtrees, determine which Azure pipelines to run."
    )
    parser.add_argument(
        "--subtree-file",
        required=True,
        help="Path to the file containing changed subtrees",
    )
    return parser.parse_args(argv)


def read_file_into_set(file_path):
    """Reads the project names from the file into a set."""
    with open(file_path, "r") as file:
        return {line.strip() for line in file}


def resolve_dependencies(projects, dependencies):
    """Resolves projects to be run by checking all levels of dependencies."""

    def has_dependency(project, projects_set):
        """Recursively checks if a project has any dependencies in the projects_set."""
        if project not in dependencies:
            return False
        for dependency in dependencies[project]:
            if dependency in projects_set or has_dependency(dependency, projects_set):
                return True
        return False

    projects_to_run = set(projects)

    for project in projects:
        if has_dependency(project, projects_to_run):
            projects_to_run.discard(project)

    return projects_to_run


def main(argv=None) -> None:
    """Main function to process the projects and output those to be run."""
    # Mathlib build+test dependency tree as defined in Azure CI and TheRock
    math_dependencies = {
        "shared/tensile": {},
        "shared/origami": {},
        "projects/rocrand": {},
        "projects/hiprand": {"projects/rocrand"},
        "projects/rocfft": {"projects/hiprand"},
        "projects/hipfft": {"projects/rocfft"},
        "projects/rocprim": {},
        "projects/hipcub": {"projects/rocprim"},
        "projects/rocthrust": {"projects/rocprim"},
        "projects/hipblas-common": {},
        "projects/hipblaslt": {"projects/hipblas-common"},
        "projects/rocblas": {"projects/hipblaslt"},
        "projects/rocsolver": {"projects/rocprim", "projects/rocblas"},
        "projects/rocsparse": {"projects/rocprim", "projects/rocblas"},
        "projects/hipblas": {"projects/rocsolver"},
        "projects/hipsolver": {"projects/rocsolver", "projects/rocsparse"},
        "projects/hipsparse": {"projects/rocsparse"},
        "projects/hipsparselt": {"projects/hipsparse"},
        "projects/hiptensor": {},
    }
    # Azure pipeline IDs for each project, to be populated as projects are enabled
    definition_ids = {
        "shared/tensile": 305,
        "projects/rocrand": 274,
        "projects/hiprand": 275,
        "projects/rocfft": 282,
        "projects/hipfft": 283,
        "projects/rocprim": 273,
        "projects/hipcub": 277,
        "projects/rocthrust": 276,
        "projects/hipblas-common": 300,
        "projects/hipblaslt": 301,
        "projects/rocblas": 302,
        "projects/rocsolver": 303,
        "projects/rocsparse": 314,
        "projects/hipblas": 317,
        "projects/hipsolver": 322,
        "projects/hipsparse": 315,
        "projects/hipsparselt": 309,
        "shared/origami": 364,
        "projects/rocwmma": 370,
        "projects/hiptensor": 374,
    }

    args = parse_arguments(argv)
    projects = read_file_into_set(args.subtree_file)
    projects_to_run = resolve_dependencies(projects, math_dependencies)

    for project in projects_to_run:
        if project in definition_ids:
            print(f"{project}={definition_ids[project]}")


if __name__ == "__main__":
    main()
