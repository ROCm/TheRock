# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the job matrix for TheRock's code coverage CI workflows.

Coverage is opt-in per project: instrumenting a library slows it down
substantially and only pays off for projects whose test suites are good enough
to produce a meaningful signal. `COVERAGE_PROJECTS` below is the allowlist, and
everything the coverage workflows need to know about a project lives there.

Environment variables:
  - PROJECTS_TO_TEST: comma-separated project keys; empty selects every
    project in the allowlist.
  - AMDGPU_FAMILIES: comma-separated GPU families to build coverage for.
  - COVERAGE_CONFIG_SOURCE: "<owner>/<repo>@<ref>" holding the per-project
    coverage metadata files (currently ROCm/rocm-libraries).

Outputs (GITHUB_OUTPUT):
  - coverage_matrix: JSON array of per-project job configurations.
  - dist_amdgpu_families: semicolon-separated families, as CMake expects them.
  - families_matrix_json: JSON array of {amdgpu_family} objects for the
    per-arch stages of multi_arch_build_portable_linux.yml.
  - coverage_cmake_options: every selected project's coverage flag, for the
    nightly full-stack instrumented build.
  - build_stages: comma-separated stages the selected projects live in.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

from github_actions_api import gha_set_output

DEFAULT_AMDGPU_FAMILIES = "gfx94X-dcgpu"
DEFAULT_COVERAGE_CONFIG_SOURCE = "ROCm/rocm-libraries@main"

# Values for CoverageProject.source_repo: the repo a project ships from. Drives
# the group aliases (rocm_libraries_all / rocm_systems_all) below.
ROCM_LIBRARIES = "rocm-libraries"
ROCM_SYSTEMS = "rocm-systems"


@dataclass(frozen=True)
class CoverageProject:
    """Everything the coverage pipeline needs to know about one project.

    Attributes:
        cmake_target: TheRock subproject name. Upper cased to form the
            `<PROJECT>_ENABLE_COVERAGE` flag that therock_subproject.cmake
            forwards to the subproject.
        stage: Build stage that produces the project.
        stage_project: Name passed to configure_stage.py --projects, which
            narrows the stage down so only this project is rebuilt.
        test_component: Key in fetch_test_configurations.py's test matrix.
        coverage_config: Per-project coverage metadata file, relative to the
            root of the repository named by COVERAGE_CONFIG_SOURCE.
        object_globs: Globs, relative to the extracted artifact directory,
            matching the instrumented binaries handed to `llvm-cov`.
        fetch_artifact_args: Arguments to install_rocm_from_artifacts.py that
            pull the instrumented libraries into the report generation job.
        codecov_flag: Flag the report is filed under in Codecov.
        source_repo: The repo this project ships from (ROCM_LIBRARIES or
            ROCM_SYSTEMS); drives the group aliases.
    """

    cmake_target: str
    stage: str
    stage_project: str
    test_component: str
    coverage_config: str
    object_globs: list[str] = field(default_factory=list)
    fetch_artifact_args: str = ""
    codecov_flag: str = ""
    source_repo: str = ROCM_LIBRARIES


# Projects that participate in coverage CI. Start small (RFC0014 phase 1) and
# grow the list as component test suites become good enough to report on.
COVERAGE_PROJECTS: dict[str, CoverageProject] = {
    "hiprand": CoverageProject(
        cmake_target="hipRAND",
        stage="math-libs",
        stage_project="hiprand",
        test_component="hiprand",
        coverage_config="projects/hiprand/test_categories_coverage.yaml",
        object_globs=["lib/libhiprand.so*"],
        fetch_artifact_args="--rand",
        codecov_flag="hipRAND",
    ),
}

_VALID_SOURCE_REPOS = frozenset({ROCM_LIBRARIES, ROCM_SYSTEMS})
for _key, _proj in COVERAGE_PROJECTS.items():
    assert (
        _proj.source_repo in _VALID_SOURCE_REPOS
    ), f"{_key}: source_repo must be one of {sorted(_VALID_SOURCE_REPOS)}"

ROCM_LIBRARIES_PROJECTS: frozenset[str] = frozenset(
    k for k, v in COVERAGE_PROJECTS.items() if v.source_repo == ROCM_LIBRARIES
)
ROCM_SYSTEMS_PROJECTS: frozenset[str] = frozenset(
    k for k, v in COVERAGE_PROJECTS.items() if v.source_repo == ROCM_SYSTEMS
)

# Group-alias tokens accepted by PROJECTS_TO_TEST (and the projects_to_test CI
# input) that expand to a whole component group. Matched case-insensitively.
_GROUP_ALIASES: dict[str, frozenset[str]] = {
    "rocm_libraries_all": ROCM_LIBRARIES_PROJECTS,
    "rocm_systems_all": ROCM_SYSTEMS_PROJECTS,
    "all": frozenset(COVERAGE_PROJECTS),
}


def parse_projects(raw_projects: str) -> list[str]:
    """Resolves PROJECTS_TO_TEST into an ordered list of known project keys.

    Recognizes the group aliases 'rocm_libraries_all', 'rocm_systems_all', and
    'all', which expand to a whole component group and may be mixed with
    explicit project names. Matching is case-insensitive.
    """
    requested = [p.strip().lower() for p in raw_projects.split(",") if p.strip()]
    if not requested:
        return sorted(COVERAGE_PROJECTS)

    expanded: list[str] = []
    unknown: list[str] = []
    for token in requested:
        if token in _GROUP_ALIASES:
            group = _GROUP_ALIASES[token]
            if not group:
                raise ValueError(
                    f"Group alias '{token}' selected no projects: no such "
                    "projects are onboarded to coverage yet."
                )
            expanded.extend(sorted(group))
        elif token in COVERAGE_PROJECTS:
            expanded.append(token)
        else:
            unknown.append(token)

    if unknown:
        raise ValueError(
            f"Unknown coverage project(s): {', '.join(sorted(unknown))}. "
            f"Coverage-enabled projects are: {', '.join(sorted(COVERAGE_PROJECTS))}. "
            f"Group aliases: {', '.join(sorted(_GROUP_ALIASES))}."
        )
    # Preserve first-seen order, drop duplicates (explicit + alias overlap).
    return list(dict.fromkeys(expanded))


def parse_amdgpu_families(raw_families: str) -> list[str]:
    families = [f.strip() for f in raw_families.split(",") if f.strip()]
    return families or [DEFAULT_AMDGPU_FAMILIES]


def parse_config_source(raw_source: str) -> tuple[str, str]:
    """Splits "<owner>/<repo>@<ref>" into its repository and ref."""
    source = raw_source.strip() or DEFAULT_COVERAGE_CONFIG_SOURCE
    repository, separator, ref = source.partition("@")
    if not separator or not repository or not ref:
        raise ValueError(
            f"Invalid coverage config source '{raw_source}'. "
            "Expected the form '<owner>/<repo>@<ref>'."
        )
    return repository, ref


def build_coverage_matrix(
    project_keys: list[str],
    amdgpu_families: list[str],
    config_repository: str,
    config_ref: str,
) -> list[dict]:
    """Produces one job configuration per (project, GPU family) pair."""
    matrix = []
    for project_key in project_keys:
        project = COVERAGE_PROJECTS[project_key]
        for family in amdgpu_families:
            matrix.append(
                {
                    "project_name": project_key,
                    # therock_subproject.cmake only honors the upper case
                    # spelling of the per-project coverage flag.
                    "coverage_flag": f"{project.cmake_target.upper()}_ENABLE_COVERAGE",
                    "cmake_target": project.cmake_target,
                    "build_stage": project.stage,
                    "stage_project": project.stage_project,
                    "test_component": project.test_component,
                    "coverage_config": project.coverage_config,
                    "coverage_config_repository": config_repository,
                    "coverage_config_ref": config_ref,
                    "object_globs": ",".join(project.object_globs),
                    "fetch_artifact_args": project.fetch_artifact_args,
                    "codecov_flag": project.codecov_flag or project_key,
                    "amdgpu_families": family,
                }
            )
    return matrix


def emit_cmake(output_path: Path) -> None:
    """Writes the coverage group-membership lists as a CMake include.

    Groups COVERAGE_PROJECTS by source_repo and emits one set() per group, using
    each project's cmake_target (upper cased by CMake to form the
    <PROJECT>_ENABLE_COVERAGE flag). CMakeLists.txt reads this so it does not
    hardcode the project names.
    """

    def targets(repo: str) -> list[str]:
        return sorted(
            (
                p.cmake_target
                for p in COVERAGE_PROJECTS.values()
                if p.source_repo == repo
            ),
            key=str.lower,
        )

    output_path.write_text(
        "# Generated by configure_coverage_ci.py --emit-cmake; do not edit.\n"
        "# Edit COVERAGE_PROJECTS in\n"
        "# build_tools/github_actions/configure_coverage_ci.py instead.\n"
        f"set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS {' '.join(targets(ROCM_LIBRARIES))})\n"
        f"set(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS {' '.join(targets(ROCM_SYSTEMS))})\n"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_matrix",
        help="Print the matrix to stdout instead of writing GITHUB_OUTPUT",
    )
    parser.add_argument(
        "--emit-cmake",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the coverage group lists as a CMake include to PATH, then exit.",
    )
    args = parser.parse_args(argv)

    if args.emit_cmake is not None:
        emit_cmake(args.emit_cmake)
        return 0

    project_keys = parse_projects(os.getenv("PROJECTS_TO_TEST", ""))
    amdgpu_families = parse_amdgpu_families(os.getenv("AMDGPU_FAMILIES", ""))
    config_repository, config_ref = parse_config_source(
        os.getenv("COVERAGE_CONFIG_SOURCE", "")
    )

    matrix = build_coverage_matrix(
        project_keys, amdgpu_families, config_repository, config_ref
    )
    coverage_flags = [
        f"-D{COVERAGE_PROJECTS[key].cmake_target.upper()}_ENABLE_COVERAGE=ON"
        for key in project_keys
    ]
    build_stages = sorted({COVERAGE_PROJECTS[key].stage for key in project_keys})
    outputs = {
        "coverage_matrix": json.dumps(matrix),
        "dist_amdgpu_families": ";".join(amdgpu_families),
        "families_matrix_json": json.dumps(
            [{"amdgpu_family": family} for family in amdgpu_families]
        ),
        "coverage_cmake_options": " ".join(coverage_flags),
        "build_stages": ",".join(["compiler-runtime"] + build_stages),
    }

    if args.print_matrix:
        print(json.dumps(outputs, indent=2))
    else:
        gha_set_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
