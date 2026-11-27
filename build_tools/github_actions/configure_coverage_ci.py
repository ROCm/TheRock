# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the job matrix for multi_arch_ci_coverage_linux.yml.

COVERAGE_PROJECTS lists the projects coverage CI runs for, by TheRock
subproject name. The rest follows from the name: the <PROJECT>_ENABLE_COVERAGE
option that instruments the project, its tests in
fetch_test_configurations.test_matrix, and its Codecov flag.

Environment variables:
  - PROJECTS_TO_TEST: comma-separated project names, case-insensitive. Empty
    selects every project in COVERAGE_PROJECTS.
  - AMDGPU_FAMILIES: comma-separated GPU families to collect coverage on.
  - COVERAGE_ROCM_LIBRARIES_ALL, COVERAGE_ROCM_SYSTEMS_ALL, COVERAGE_ALL: when
    true, add the matching option from GROUP_CMAKE_OPTIONS, and turn hipCCL
    back off for the groups that include it. Report jobs are still generated
    only for the selected projects.

Outputs (written to GITHUB_OUTPUT):
  - coverage_matrix: JSON array with one report job per (project, GPU family).
  - dist_amdgpu_families: the families, semicolon-separated.
  - families_matrix_json: JSON array of {"amdgpu_family": ...} objects, one
    math-libs build per family.
  - coverage_cmake_options: space-separated CMake options that instrument the
    selected projects and any selected groups, for the math-libs build's
    extra_cmake_options input.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

import fetch_test_configurations
from github_actions_api import gha_set_output, str2bool

DEFAULT_AMDGPU_FAMILIES = "gfx94X-dcgpu"

# The group options declared in the top-level CMakeLists.txt, keyed by the
# environment variable that turns each one on.
GROUP_CMAKE_OPTIONS = {
    "COVERAGE_ROCM_LIBRARIES_ALL": "-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON",
    "COVERAGE_ROCM_SYSTEMS_ALL": "-DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON",
    "COVERAGE_ALL": "-DTHEROCK_COVERAGE_ALL=ON",
}

# hipCCL is in THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS, but its instrumented
# build doesn't link on the CI build image yet. The groups that include it turn
# it back off; an explicit OFF wins over a group option.
HIPCCL_GROUPS = ("COVERAGE_ROCM_LIBRARIES_ALL", "COVERAGE_ALL")
HIPCCL_COVERAGE_OFF = "-DHIPCCL_ENABLE_COVERAGE=OFF"


# multi_arch_ci_coverage_linux.yml only builds compiler-runtime and math-libs,
# so every project here must be built by the math-libs stage.
BUILD_STAGE = "math-libs"

# Keyed by TheRock subproject name, which is also the Codecov flag the report is
# uploaded under (rocm-libraries' codecov.yml uses the same names). Values are
# globs, relative to the install directory, matching the instrumented binaries
# the report covers.
COVERAGE_PROJECTS: dict[str, tuple[str, ...]] = {
    "hipDNN": ("lib/libhipdnn_backend.so*",),
}


def parse_projects(projects_to_test: str) -> list[str]:
    """Returns the selected COVERAGE_PROJECTS keys, in the order given."""
    keys = {name.lower(): name for name in COVERAGE_PROJECTS}
    names = [name.strip().lower() for name in projects_to_test.split(",")]
    names = list(dict.fromkeys(name for name in names if name))
    if not names:
        return list(COVERAGE_PROJECTS)
    unknown = [name for name in names if name not in keys]
    if unknown:
        raise ValueError(
            f"Unknown coverage project(s): {', '.join(unknown)}. "
            f"Available: {', '.join(COVERAGE_PROJECTS)}."
        )
    return [keys[name] for name in names]


def parse_amdgpu_families(amdgpu_families: str) -> list[str]:
    families = [family.strip() for family in amdgpu_families.split(",")]
    return [family for family in families if family] or [DEFAULT_AMDGPU_FAMILIES]


def parse_group_cmake_options(env: Mapping[str, str]) -> list[str]:
    """Returns the group options whose environment variable is true."""
    options = [
        option
        for env_var, option in GROUP_CMAKE_OPTIONS.items()
        if str2bool(env.get(env_var))
    ]
    if any(str2bool(env.get(env_var)) for env_var in HIPCCL_GROUPS):
        options.append(HIPCCL_COVERAGE_OFF)
    return options


def build_outputs(
    projects: list[str],
    amdgpu_families: list[str],
    group_cmake_options: Sequence[str] = (),
) -> dict[str, str]:
    matrix = []
    cmake_options = []
    for name in projects:
        test_component = name.lower()
        cmake_options.append(f"-D{name.upper()}_ENABLE_COVERAGE=ON")
        test_install_args = fetch_test_configurations.test_matrix[test_component][
            "fetch_artifact_args"
        ].split()
        # The report job needs the instrumented binaries, not the tests.
        fetch_artifact_args = " ".join(
            arg for arg in test_install_args if arg != "--tests"
        )
        for family in amdgpu_families:
            matrix.append(
                {
                    "project_name": test_component,
                    "test_component": test_component,
                    "object_globs": ",".join(COVERAGE_PROJECTS[name]),
                    "fetch_artifact_args": fetch_artifact_args,
                    "build_stage": BUILD_STAGE,
                    "codecov_flag": name,
                    "amdgpu_families": family,
                }
            )
    cmake_options.extend(group_cmake_options)
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
        parse_group_cmake_options(os.environ),
    )
    if args.print_outputs:
        print(json.dumps(outputs, indent=2))
    else:
        gha_set_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
