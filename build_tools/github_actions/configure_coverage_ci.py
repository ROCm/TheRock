# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the job matrix for TheRock's nightly code coverage CI workflow.

Coverage is opt-in per project: instrumenting a library slows it down
substantially and only pays off for projects whose test suites are good enough
to produce a meaningful signal. The allowlist, `COVERAGE_PROJECTS`, lives in
build_tools/_therock_utils/coverage_registry.py and is shared with the build.

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
  - coverage_cmake_options: the coverage flags for the nightly full-stack
    instrumented build, collapsed to a THEROCK_COVERAGE_*_ALL group option
    whenever the selection covers a whole group.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from github_actions_api import gha_set_output
from _therock_utils.coverage_registry import (
    COVERAGE_PROJECTS,
    DEFAULT_PROJECTS,
    ROCM_LIBRARIES_PROJECTS,
    ROCM_SYSTEMS_PROJECTS,
    STAGE_MATH_LIBS,
    SUPPORTED_PROJECTS,
    BLOCKED_PROJECTS,
    BUILDABLE_STAGES,
)

DEFAULT_AMDGPU_FAMILIES = "gfx94X-dcgpu"
DEFAULT_COVERAGE_CONFIG_SOURCE = "ROCm/rocm-libraries@main"

# Group-alias tokens accepted by PROJECTS_TO_TEST (and the projects_to_test CI
# input) that expand to a whole component group. Matched case-insensitively.
_GROUP_ALIASES: dict[str, frozenset[str]] = {
    "rocm_libraries_all": ROCM_LIBRARIES_PROJECTS,
    "rocm_systems_all": ROCM_SYSTEMS_PROJECTS,
    "all": SUPPORTED_PROJECTS - BLOCKED_PROJECTS,
}


def parse_projects(raw_projects: str) -> list[str]:
    """Resolves PROJECTS_TO_TEST into an ordered list of known project keys.

    Recognizes the group aliases 'rocm_libraries_all', 'rocm_systems_all', and
    'all', which expand to a whole component group and may be mixed with
    explicit project names. Matching is case-insensitive.

    An empty selection means every project this workflow can build, which is
    narrower than 'all' while comm-libs has no build job. Asking for those
    projects by name or via a group alias still fails in
    resolve_build_stages(), so nothing is dropped from an explicit request.

    Projects with a known-broken instrumented build are likewise left out of
    the default and the aliases, but naming one is honoured with a warning
    rather than an error: retesting a block is the only way to notice it has
    been fixed.
    """
    requested = [p.strip().lower() for p in raw_projects.split(",") if p.strip()]
    if not requested:
        return sorted(DEFAULT_PROJECTS)

    expanded: list[str] = []
    unknown: list[str] = []
    unsupported: list[str] = []
    for token in requested:
        if token in _GROUP_ALIASES:
            group = _GROUP_ALIASES[token]
            if not group:
                raise ValueError(
                    f"Group alias '{token}' selected no projects: no such "
                    "projects are onboarded to coverage yet."
                )
            expanded.extend(sorted(group))
        elif token in SUPPORTED_PROJECTS:
            expanded.append(token)
        elif token in COVERAGE_PROJECTS:
            unsupported.append(token)
        else:
            unknown.append(token)

    if unknown:
        raise ValueError(
            f"Unknown coverage project(s): {', '.join(sorted(unknown))}. "
            f"Coverage-enabled projects are: {', '.join(sorted(SUPPORTED_PROJECTS))}. "
            f"Group aliases: {', '.join(sorted(_GROUP_ALIASES))}."
        )

    # Failing here beats scheduling a build and several hours of tests that
    # cannot produce a report.
    if unsupported:
        detail = "; ".join(
            f"{name} ({COVERAGE_PROJECTS[name].unsupported_reason})"
            for name in sorted(unsupported)
        )
        raise ValueError(
            f"Coverage is not supported for: {detail}. These projects are "
            "registered so the gap is tracked, but cannot be measured until "
            "they emit LLVM profraw data."
        )
    # Preserve first-seen order, drop duplicates (explicit + alias overlap).
    selection = list(dict.fromkeys(expanded))

    # Not fatal: the caller went out of their way to name these, and the only
    # way to find out a block has been lifted is to run into it again.
    for name in selection:
        blocked_reason = COVERAGE_PROJECTS[name].blocked_reason
        if blocked_reason:
            print(
                f"warning: {name} is expected to fail: {blocked_reason}",
                file=sys.stderr,
            )
    return selection


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
                    "codecov_flag": project.codecov_flag,
                    "amdgpu_families": family,
                    "artifact_names": ",".join(project.artifact_names),
                    "artifact_relpaths": ",".join(project.artifact_relpaths),
                }
            )
    return matrix


def build_coverage_cmake_options(project_keys: list[str]) -> list[str]:
    """Picks the CMake coverage flags that instrument exactly this selection.

    A nightly run instruments every onboarded project, which is what the
    THEROCK_COVERAGE_*_ALL group options say directly; a narrowed selection
    falls back to naming each project. Both spellings reach the same
    <PROJECT>_ENABLE_COVERAGE flags, so this only affects how the configure
    line reads.
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

    # The group options already cover extra_cmake_targets, because the registry
    # writes them into the group lists CMakeLists.txt expands.
    for key in project_keys:
        if key in grouped:
            continue
        project = COVERAGE_PROJECTS[key]
        for target in [project.cmake_target, *project.extra_cmake_targets]:
            options.append(f"-D{target.upper()}_ENABLE_COVERAGE=ON")
    return options


def resolve_build_stages(project_keys: list[str]) -> set[str]:
    """Returns the stages the selection needs, rejecting stages with no build job.

    Only compiler-runtime and math-libs have build jobs. A project from any
    other stage would have nothing built for it, and that would not surface
    until the test job found no instrumented files to overlay, hours in, so it
    is an error here instead.
    """
    stages = {COVERAGE_PROJECTS[key].stage for key in project_keys}
    unbuildable = sorted(stages - BUILDABLE_STAGES)
    if unbuildable:
        blocked = sorted(
            key for key in project_keys if COVERAGE_PROJECTS[key].stage in unbuildable
        )
        raise ValueError(
            f"cannot build coverage for {', '.join(blocked)}: this workflow has "
            f"build jobs for {', '.join(sorted(BUILDABLE_STAGES))} only, not "
            f"{', '.join(unbuildable)}. Restore the staged build jobs in "
            "multi_arch_ci_coverage_linux.yml to select these projects."
        )
    return stages


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_matrix",
        help="Print the matrix to stdout instead of writing GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

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
