# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the job matrix for multi_arch_ci_coverage_linux.yml.

COVERAGE_PROJECTS lists the projects coverage CI runs for, with what the
workflows need to know about each one.

Environment variables:
  - PROJECTS_TO_TEST: comma-separated project names, case-insensitive. Empty
    selects every project in COVERAGE_PROJECTS.
  - AMDGPU_FAMILIES: comma-separated GPU families to collect coverage on.
  - NIGHTLY: "true" adds NIGHTLY_CMAKE_OPTIONS, so the build instruments every
    project in TheRock's coverage group lists, not just the selected ones.

Outputs (written to GITHUB_OUTPUT):
  - coverage_matrix: JSON array with one report job per (project, GPU family).
  - dist_amdgpu_families: the families, semicolon-separated.
  - families_matrix_json: JSON array of {"amdgpu_family": ...} objects, one
    math-libs build per family.
  - coverage_cmake_options: space-separated CMake options that instrument the
    selected projects (plus NIGHTLY_CMAKE_OPTIONS when NIGHTLY is set), for the
    math-libs build's extra_cmake_options input.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

from github_actions_api import gha_set_output, str2bool

DEFAULT_AMDGPU_FAMILIES = "gfx94X-dcgpu"

# The group options declared in the top-level CMakeLists.txt. RFC0014's nightly
# phase 1 instruments the whole stack in one build.
NIGHTLY_CMAKE_OPTIONS = (
    "-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON",
    "-DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON",
    "-DTHEROCK_COVERAGE_ALL=ON",
)


@dataclass(frozen=True)
class CoverageProject:
    """What the coverage workflows need to know about one project.

    Attributes:
        cmake_options: CMake options that instrument the project. The build
            splits extra_cmake_options on whitespace, so no option may contain
            a space.
        test_component: Key of the project's tests in
            fetch_test_configurations.test_matrix.
        object_globs: Globs, relative to the install directory, matching the
            instrumented binaries the report covers.
        fetch_artifact_args: install_rocm_from_artifacts.py arguments that
            install those binaries in the report job.
        build_stage: Stage whose sources the report job fetches.
        codecov_flag: Codecov flag the report is uploaded under.
    """

    cmake_options: tuple[str, ...]
    test_component: str
    object_globs: tuple[str, ...]
    fetch_artifact_args: str
    build_stage: str
    codecov_flag: str


# multi_arch_ci_coverage_linux.yml only builds compiler-runtime and math-libs,
# so every project here must be in the math-libs stage.
COVERAGE_PROJECTS: dict[str, CoverageProject] = {
    "hipdnn": CoverageProject(
        cmake_options=("-DHIPDNN_ENABLE_COVERAGE=ON",),
        test_component="hipdnn",
        object_globs=("lib/libhipdnn_backend.so*",),
        fetch_artifact_args="--hipdnn",
        build_stage="math-libs",
        codecov_flag="hipDNN",
    ),
}


def parse_projects(projects_to_test: str) -> list[str]:
    """Returns the selected project names, in the order given."""
    names = [name.strip().lower() for name in projects_to_test.split(",")]
    names = list(dict.fromkeys(name for name in names if name))
    if not names:
        return list(COVERAGE_PROJECTS)
    unknown = [name for name in names if name not in COVERAGE_PROJECTS]
    if unknown:
        raise ValueError(
            f"Unknown coverage project(s): {', '.join(unknown)}. "
            f"Available: {', '.join(COVERAGE_PROJECTS)}."
        )
    return names


def parse_amdgpu_families(amdgpu_families: str) -> list[str]:
    families = [family.strip() for family in amdgpu_families.split(",")]
    return [family for family in families if family] or [DEFAULT_AMDGPU_FAMILIES]


def build_outputs(
    projects: list[str], amdgpu_families: list[str], nightly: bool = False
) -> dict[str, str]:
    matrix = []
    cmake_options = {}
    for name in projects:
        project = COVERAGE_PROJECTS[name]
        cmake_options.update(dict.fromkeys(project.cmake_options))
        for family in amdgpu_families:
            matrix.append(
                {
                    "project_name": name,
                    "test_component": project.test_component,
                    "object_globs": ",".join(project.object_globs),
                    "fetch_artifact_args": project.fetch_artifact_args,
                    "build_stage": project.build_stage,
                    "codecov_flag": project.codecov_flag,
                    "amdgpu_families": family,
                }
            )
    if nightly:
        cmake_options.update(dict.fromkeys(NIGHTLY_CMAKE_OPTIONS))
    return {
        "coverage_matrix": json.dumps(matrix),
        "dist_amdgpu_families": ";".join(amdgpu_families),
        "families_matrix_json": json.dumps(
            [{"amdgpu_family": family} for family in amdgpu_families]
        ),
        "coverage_cmake_options": " ".join(cmake_options),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_outputs",
        help="Print the outputs instead of writing them to GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

    try:
        projects = parse_projects(os.getenv("PROJECTS_TO_TEST", ""))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    outputs = build_outputs(
        projects,
        parse_amdgpu_families(os.getenv("AMDGPU_FAMILIES", "")),
        nightly=str2bool(os.getenv("NIGHTLY")),
    )
    if args.print_outputs:
        print(json.dumps(outputs, indent=2))
    else:
        gha_set_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
