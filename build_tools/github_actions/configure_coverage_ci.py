# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Registry of the projects TheRock can build with code coverage.

Coverage is opt-in per project: instrumenting a library slows it down
substantially and only pays off for projects whose test suites are good enough
to produce a meaningful signal. `COVERAGE_PROJECTS` below is the allowlist, and
everything the coverage build and workflows need to know about a project lives
there.

`--emit-cmake PATH` writes the coverage group lists and the
THEROCK_COVERAGE_OPTION_<PROJECT> map as a CMake include. CMakeLists.txt runs
this on every configure, so this script must not depend on anything outside the
Python standard library.

Without it, builds the job matrix for the coverage CI workflow
(multi_arch_ci_coverage_linux.yml).

Environment variables:
  - PROJECTS_TO_TEST: comma-separated project keys or group aliases; empty
    selects every project in the allowlist.
  - AMDGPU_FAMILIES: comma-separated GPU families to build coverage for.
  - COVERAGE_CONFIG_SOURCE: "<owner>/<repo>@<ref>" holding the per-project
    coverage metadata files (currently ROCm/rocm-libraries).

Outputs (GITHUB_OUTPUT):
  - coverage_matrix: JSON array of per-project job configurations.
  - dist_amdgpu_families: semicolon-separated families, as CMake expects them.
  - families_matrix_json: JSON array of {amdgpu_family} objects for the
    per-arch math-libs build in multi_arch_ci_coverage_linux.yml.
  - coverage_cmake_options: the coverage flags for the instrumented build,
    collapsed to a THEROCK_COVERAGE_*_ALL group option whenever the selection
    covers a whole group.
  - needs_math_libs: "true" when a selected project is built in math-libs.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from github_actions_api import gha_set_output

DEFAULT_AMDGPU_FAMILIES = "gfx94X-dcgpu"
DEFAULT_COVERAGE_CONFIG_SOURCE = "ROCm/rocm-libraries@main"

# Values for CoverageProject.source_repo: the repo a project ships from. Drives
# the THEROCK_COVERAGE_ROCM_LIBRARIES_ALL / _ROCM_SYSTEMS_ALL group lists.
ROCM_LIBRARIES = "rocm-libraries"
ROCM_SYSTEMS = "rocm-systems"

# Build stages multi_arch_ci_coverage_linux.yml has build jobs for.
# compiler-runtime is always built because every other stage takes its inbound
# artifacts from it.
STAGE_COMPILER_RUNTIME = "compiler-runtime"
STAGE_MATH_LIBS = "math-libs"
BUILDABLE_STAGES = frozenset({STAGE_COMPILER_RUNTIME, STAGE_MATH_LIBS})


@dataclass(frozen=True)
class CoverageProject:
    """Everything the coverage pipeline needs to know about one project.

    Attributes:
        cmake_target: TheRock subproject name. Upper cased to form the
            `<PROJECT>_ENABLE_COVERAGE` flag, which is TheRock's own knob and
            is not what the subproject sees; see coverage_option.
        coverage_option: CMake option the upstream project implements to turn
            its instrumentation on. These names are not standardised -- most
            projects use `BUILD_CODE_COVERAGE` or `CODE_COVERAGE`, a few use
            `<PROJECT>_ENABLE_COVERAGE`, RCCL uses `ENABLE_CODE_COVERAGE` --
            so therock_subproject.cmake translates TheRock's flag into this
            name when configuring the subproject. Being per-subproject is what
            keeps the generic names from instrumenting the whole build. The
            option must select LLVM source-based coverage: gcov writes `.gcda`
            files, not the `.profraw` files this pipeline merges.
        stage: Build stage that produces the project. The coverage workflow
            needs a build job for this stage, so onboarding a project from a
            new stage means adding one.
        test_component: Key in fetch_test_configurations.py's test matrix.
        coverage_config: Per-project coverage metadata file, relative to the
            root of the repository named by COVERAGE_CONFIG_SOURCE.
        artifact_names: BUILD_TOPOLOGY artifact(s) the instrumented project
            ships in. These are grouped (`rand` holds both rocRAND and
            hipRAND), which is why artifact_relpaths exists.
        artifact_relpaths: Subproject stage directories inside those artifacts
            that belong to this project alone.
        object_globs: Globs, relative to the extracted artifact directory,
            matching the instrumented binaries handed to `llvm-cov`.
        fetch_artifact_args: Arguments to install_rocm_from_artifacts.py that
            pull the instrumented libraries into the report generation job.
        codecov_flag: Flag the report is filed under in Codecov.
        source_repo: The repo this project ships from (ROCM_LIBRARIES or
            ROCM_SYSTEMS); picks its THEROCK_COVERAGE_*_ALL group.
    """

    cmake_target: str
    coverage_option: str
    stage: str
    test_component: str
    coverage_config: str
    artifact_names: list[str] = field(default_factory=list)
    artifact_relpaths: list[str] = field(default_factory=list)
    object_globs: list[str] = field(default_factory=list)
    fetch_artifact_args: str = ""
    codecov_flag: str = ""
    source_repo: str = ROCM_LIBRARIES


# Projects that can be built with coverage instrumentation.
#
# A run never instruments this whole list unless it is asked to. `PROJECTS_TO_TEST`
# selects a subset by name, or by one of the group aliases below, and only the
# stages that subset needs get built. Selecting everything is possible and
# expensive: instrumented builds are substantially slower than normal ones.
COVERAGE_PROJECTS: dict[str, CoverageProject] = {
    #
    # rocm-libraries -- math-libs stage
    #
    "hiprand": CoverageProject(
        cmake_target="hipRAND",
        artifact_names=["rand"],
        artifact_relpaths=["math-libs/hipRAND/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hiprand",
        coverage_config="projects/hiprand/test_categories_coverage.yaml",
        object_globs=["lib/libhiprand.so*"],
        fetch_artifact_args="--rand",
        codecov_flag="hipRAND",
    ),
    "hipdnn": CoverageProject(
        cmake_target="hipDNN",
        artifact_names=["hipdnn"],
        artifact_relpaths=["ml-libs/hipDNN/stage"],
        coverage_option="HIPDNN_ENABLE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipdnn",
        coverage_config="projects/hipdnn/test_categories_coverage.yaml",
        # Not libhipdnn.so: hipDNN's shared library is the backend, which is
        # also the name TheRock's own packaging metadata records for it.
        object_globs=["lib/libhipdnn_backend.so*"],
        fetch_artifact_args="--hipdnn",
        codecov_flag="hipDNN",
    ),
}

_VALID_SOURCE_REPOS = frozenset({ROCM_LIBRARIES, ROCM_SYSTEMS})
for _key, _proj in COVERAGE_PROJECTS.items():
    assert (
        _proj.source_repo in _VALID_SOURCE_REPOS
    ), f"{_key}: source_repo must be one of {sorted(_VALID_SOURCE_REPOS)}"
    # An entry without one would be accepted and instrument nothing.
    assert _proj.coverage_option, f"{_key}: coverage_option must be set"
    # A project in a stage the workflow cannot build would be selectable and
    # then tested against binaries nothing instrumented.
    assert (
        _proj.stage in BUILDABLE_STAGES
    ), f"{_key}: stage must be one of {sorted(BUILDABLE_STAGES)}"
    assert _proj.artifact_names, f"{_key}: artifact_names must be set"
    assert _proj.artifact_relpaths, f"{_key}: artifact_relpaths must be set"

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
    explicit project names. Matching is case-insensitive. An empty selection
    means every registered project.
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
                    "cmake_target": project.cmake_target,
                    "test_component": project.test_component,
                    "coverage_config": project.coverage_config,
                    "coverage_config_repository": config_repository,
                    "coverage_config_ref": config_ref,
                    "object_globs": ",".join(project.object_globs),
                    "fetch_artifact_args": project.fetch_artifact_args,
                    # The report job fetches this stage's sources so the HTML
                    # rendering has something to annotate.
                    "build_stage": project.stage,
                    "codecov_flag": project.codecov_flag or project_key,
                    "amdgpu_families": family,
                    "artifact_names": ",".join(project.artifact_names),
                    "artifact_relpaths": ",".join(project.artifact_relpaths),
                }
            )
    return matrix


def build_coverage_cmake_options(project_keys: list[str]) -> list[str]:
    """Picks the CMake coverage flags that instrument exactly this selection.

    A selection covering a whole group is what the THEROCK_COVERAGE_*_ALL group
    options say directly; a narrower selection falls back to naming each
    project. Both spellings reach the same <PROJECT>_ENABLE_COVERAGE flags, so
    this only affects how the configure line reads.
    """
    selected = set(project_keys)
    covers_libraries = bool(ROCM_LIBRARIES_PROJECTS) and ROCM_LIBRARIES_PROJECTS <= (
        selected
    )
    covers_systems = bool(ROCM_SYSTEMS_PROJECTS) and ROCM_SYSTEMS_PROJECTS <= selected

    options: list[str] = []
    grouped: set[str] = set()
    if covers_libraries and covers_systems:
        options.append("-DTHEROCK_COVERAGE_ALL=ON")
        grouped |= ROCM_LIBRARIES_PROJECTS | ROCM_SYSTEMS_PROJECTS
    else:
        if covers_libraries:
            options.append("-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON")
            grouped |= ROCM_LIBRARIES_PROJECTS
        if covers_systems:
            options.append("-DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON")
            grouped |= ROCM_SYSTEMS_PROJECTS

    for key in project_keys:
        if key in grouped:
            continue
        target = COVERAGE_PROJECTS[key].cmake_target
        options.append(f"-D{target.upper()}_ENABLE_COVERAGE=ON")
    return options


def resolve_build_stages(project_keys: list[str]) -> set[str]:
    """Returns the build stages the selected projects are built in."""
    return {COVERAGE_PROJECTS[key].stage for key in project_keys}


def emit_cmake(output_path: Path) -> None:
    """Writes the coverage group lists and option-name map as a CMake include.

    The group lists drive the THEROCK_COVERAGE_*_ALL options. The option map
    tells therock_subproject.cmake which flag each subproject actually
    implements, since the names are not standardised upstream. CMakeLists.txt
    reads both so it does not hardcode either.
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

    option_lines = [
        f"set(THEROCK_COVERAGE_OPTION_{p.cmake_target.upper()} {p.coverage_option})\n"
        for p in sorted(
            COVERAGE_PROJECTS.values(), key=lambda v: v.cmake_target.lower()
        )
    ]

    output_path.write_text(
        "# Generated by configure_coverage_ci.py --emit-cmake; do not edit.\n"
        "# Edit COVERAGE_PROJECTS in\n"
        "# build_tools/github_actions/configure_coverage_ci.py instead.\n"
        f"set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS {' '.join(targets(ROCM_LIBRARIES))})\n"
        f"set(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS {' '.join(targets(ROCM_SYSTEMS))})\n"
        + "".join(option_lines)
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
    coverage_flags = build_coverage_cmake_options(project_keys)
    needs_math_libs = STAGE_MATH_LIBS in resolve_build_stages(project_keys)
    outputs = {
        "coverage_matrix": json.dumps(matrix),
        "dist_amdgpu_families": ";".join(amdgpu_families),
        "families_matrix_json": json.dumps(
            [{"amdgpu_family": family} for family in amdgpu_families]
        ),
        "coverage_cmake_options": " ".join(coverage_flags),
        "needs_math_libs": "true" if needs_math_libs else "false",
    }

    if args.print_matrix:
        print(json.dumps(outputs, indent=2))
    else:
        gha_set_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
