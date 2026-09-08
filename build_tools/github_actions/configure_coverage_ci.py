# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the job matrix for TheRock's nightly code coverage CI workflow.

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
  - coverage_cmake_options: the coverage flags for the nightly full-stack
    instrumented build, collapsed to a THEROCK_COVERAGE_*_ALL group option
    whenever the selection covers a whole group.
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
# the group aliases (rocm_libraries_all / rocm_systems_all) below.
ROCM_LIBRARIES = "rocm-libraries"
ROCM_SYSTEMS = "rocm-systems"

# Build stages that can carry an instrumented project. compiler-runtime is
# always built because every other stage takes its inbound artifacts from it.
STAGE_COMPILER_RUNTIME = "compiler-runtime"
STAGE_MATH_LIBS = "math-libs"
STAGE_COMM_LIBS = "comm-libs"
STAGE_PROFILER_APPS = "profiler-apps"

# Stages a project may be registered against, validated below.
KNOWN_STAGES = frozenset(
    {
        STAGE_COMPILER_RUNTIME,
        STAGE_MATH_LIBS,
        STAGE_COMM_LIBS,
        STAGE_PROFILER_APPS,
    }
)

# Phase 1 has build jobs for these two only. The jobs that built any other
# stage were removed while their sole measurable consumers -- rccl and rocshmem
# in comm-libs -- are blocked upstream, so a selection reaching further is
# rejected by resolve_build_stages() rather than failing hours later on a
# missing inbound artifact.
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
            keeps the generic names from instrumenting the whole build.
        unsupported_reason: Set instead of coverage_option when the project
            cannot be measured by this pipeline. Such a project stays in the
            registry so the gap is recorded, but is left out of the group
            aliases and rejected if named explicitly.
        blocked_reason: Set alongside coverage_option when the project is
            measurable in principle but its instrumented build is currently
            broken. Unlike unsupported_reason this is expected to be
            temporary, so the project only drops out of the default selection
            and the group aliases; naming it still works, which is how the
            block gets retested once the fix lands.
        artifact_names: BUILD_TOPOLOGY artifact(s) the instrumented project
            ships in. These are grouped (`rand` holds both rocRAND and
            hipRAND), which is why artifact_relpaths exists.
        artifact_relpaths: Subproject stage directories inside those artifacts
            that belong to this project alone. The nightly hybrid fetch copies
            only these over the baseline install tree, so a run measuring
            hipRAND does not also pick up an instrumented rocRAND.
        stage: Build stage that produces the project. The nightly needs a build
            job for this stage, so onboarding a project from a new stage means
            adding one.
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
        extra_cmake_targets: Additional subprojects to instrument alongside
            cmake_target. Header-only projects report against their own test
            binaries, and some of those are built by a sibling subproject
            (rocPRIM's live in rocPRIM_tests), which therefore has to be
            instrumented too or the binaries carry no coverage mapping.
    """

    cmake_target: str
    stage: str
    test_component: str
    coverage_config: str
    coverage_option: str = ""
    unsupported_reason: str = ""
    blocked_reason: str = ""
    artifact_names: list[str] = field(default_factory=list)
    artifact_relpaths: list[str] = field(default_factory=list)
    object_globs: list[str] = field(default_factory=list)
    fetch_artifact_args: str = ""
    codecov_flag: str = ""
    source_repo: str = ROCM_LIBRARIES
    extra_cmake_targets: list[str] = field(default_factory=list)


# Projects that participate in coverage CI.
#
# A run never instruments this whole list unless it is asked to. `PROJECTS_TO_TEST`
# selects a subset by name, or by one of the group aliases below, and only the
# stages that subset needs get built. Selecting everything is possible and
# expensive: instrumented builds are substantially slower than normal ones.
#
# A selection covering several projects instruments all of them in one build and
# still reports on each separately, because `object_globs` is what scopes a
# report: llvm-cov only reports functions found in the objects it is handed, so
# counters emitted by a sibling project are never looked up.
#
# Header-only projects (rocPRIM, hipCUB, rocThrust, rocWMMA) ship no library to
# report against, so their object_globs name their installed test binaries
# instead, and their fetch args include --tests to bring those binaries into the
# report job.
COVERAGE_PROJECTS: dict[str, CoverageProject] = {
    #
    # rocm-libraries -- math-libs stage
    #
    "rocrand": CoverageProject(
        cmake_target="rocRAND",
        artifact_names=["rand"],
        artifact_relpaths=["math-libs/rocRAND/stage"],
        coverage_option="CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocrand",
        coverage_config="projects/rocrand/test_categories_coverage.yaml",
        object_globs=["lib/librocrand.so*"],
        fetch_artifact_args="--rand",
        codecov_flag="rocRAND",
    ),
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
    "rocfft": CoverageProject(
        cmake_target="rocFFT",
        artifact_names=["fft"],
        artifact_relpaths=["math-libs/rocFFT/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocfft",
        coverage_config="projects/rocfft/test_categories_coverage.yaml",
        object_globs=["lib/librocfft.so*"],
        fetch_artifact_args="--fft",
        codecov_flag="rocFFT",
    ),
    "hipfft": CoverageProject(
        cmake_target="hipFFT",
        artifact_names=["fft"],
        artifact_relpaths=["math-libs/hipFFT/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipfft",
        coverage_config="projects/hipfft/test_categories_coverage.yaml",
        object_globs=["lib/libhipfft.so*"],
        fetch_artifact_args="--fft",
        codecov_flag="hipFFT",
    ),
    "rocblas": CoverageProject(
        cmake_target="rocBLAS",
        artifact_names=["blas"],
        artifact_relpaths=["math-libs/BLAS/rocBLAS/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocblas",
        coverage_config="projects/rocblas/test_categories_coverage.yaml",
        object_globs=["lib/librocblas.so*"],
        fetch_artifact_args="--blas",
        codecov_flag="rocBLAS",
    ),
    "hipblas": CoverageProject(
        cmake_target="hipBLAS",
        artifact_names=["blas"],
        artifact_relpaths=["math-libs/BLAS/hipBLAS/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipblas",
        coverage_config="projects/hipblas/test_categories_coverage.yaml",
        object_globs=["lib/libhipblas.so*"],
        fetch_artifact_args="--blas",
        codecov_flag="hipBLAS",
    ),
    "hipblaslt": CoverageProject(
        cmake_target="hipBLASLt",
        artifact_names=["blas"],
        artifact_relpaths=["math-libs/BLAS/hipBLASLt/stage"],
        coverage_option="HIPBLASLT_ENABLE_COVERAGE",
        # The coverage-only branch of clients/CMakeLists.txt links hipblaslt-test
        # against a bare `rocroller` while its neighbours use the imported
        # targets. rocRoller exports as roc::rocroller, and under TheRock it is a
        # separate subproject found via its package config, so the bare name is
        # not a target here and reaches the linker as -lrocroller with no -L to
        # resolve it. hipBLASLt's own build gets away with it by having rocRoller
        # in-tree. Fix belongs upstream in ROCm/rocm-libraries.
        blocked_reason=(
            "hipblaslt-test fails to link with 'unable to find library "
            "-lrocroller'; its coverage build needs roc::rocroller upstream"
        ),
        stage=STAGE_MATH_LIBS,
        test_component="hipblaslt",
        coverage_config="projects/hipblaslt/test_categories_coverage.yaml",
        object_globs=["lib/libhipblaslt.so*"],
        fetch_artifact_args="--blas",
        codecov_flag="hipBLASLt",
    ),
    "rocsparse": CoverageProject(
        cmake_target="rocSPARSE",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/rocSPARSE/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocsparse",
        coverage_config="projects/rocsparse/test_categories_coverage.yaml",
        object_globs=["lib/librocsparse.so*"],
        fetch_artifact_args="--sparse",
        codecov_flag="rocSPARSE",
    ),
    "hipsparse": CoverageProject(
        cmake_target="hipSPARSE",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/hipSPARSE/stage"],
        unsupported_reason=(
            "HIPSPARSE_ENABLE_COVERAGE selects gcov instrumentation, which writes .gcda files rather than the .profraw this pipeline merges"
        ),
        stage=STAGE_MATH_LIBS,
        test_component="hipsparse",
        coverage_config="projects/hipsparse/test_categories_coverage.yaml",
        object_globs=["lib/libhipsparse.so*"],
        fetch_artifact_args="--sparse",
        codecov_flag="hipSPARSE",
    ),
    "hipsparselt": CoverageProject(
        cmake_target="hipSPARSELt",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/hipSPARSELt/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipsparselt",
        coverage_config="projects/hipsparselt/test_categories_coverage.yaml",
        object_globs=["lib/libhipsparselt.so*"],
        fetch_artifact_args="--sparse",
        codecov_flag="hipSPARSELt",
    ),
    "rocsolver": CoverageProject(
        cmake_target="rocSOLVER",
        artifact_names=["solver"],
        artifact_relpaths=["math-libs/BLAS/rocSOLVER/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocsolver",
        coverage_config="projects/rocsolver/test_categories_coverage.yaml",
        object_globs=["lib/librocsolver.so*"],
        fetch_artifact_args="--solver",
        codecov_flag="rocSOLVER",
    ),
    "hipsolver": CoverageProject(
        cmake_target="hipSOLVER",
        artifact_names=["solver"],
        artifact_relpaths=["math-libs/BLAS/hipSOLVER/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipsolver",
        coverage_config="projects/hipsolver/test_categories_coverage.yaml",
        object_globs=["lib/libhipsolver.so*"],
        fetch_artifact_args="--solver",
        codecov_flag="hipSOLVER",
    ),
    "rocalution": CoverageProject(
        cmake_target="rocALUTION",
        artifact_names=["rocalution"],
        artifact_relpaths=["math-libs/rocALUTION/stage"],
        unsupported_reason=(
            "BUILD_CODE_COVERAGE selects gcov instrumentation, which writes .gcda files rather than the .profraw this pipeline merges"
        ),
        stage=STAGE_MATH_LIBS,
        test_component="rocalution",
        coverage_config="projects/rocalution/test_categories_coverage.yaml",
        object_globs=["lib/librocalution.so*"],
        fetch_artifact_args="--rocalution",
        codecov_flag="rocALUTION",
    ),
    "hiptensor": CoverageProject(
        cmake_target="hipTensor",
        artifact_names=["hiptensor"],
        artifact_relpaths=["math-libs/hipTensor/stage"],
        # Prefixed, unlike rocALUTION's bare CODE_COVERAGE just above. Getting
        # this wrong is silent: the option is simply never read, the library
        # builds uninstrumented, and the tests pass having written no profile.
        coverage_option="HIPTENSOR_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hiptensor",
        coverage_config="projects/hiptensor/test_categories_coverage.yaml",
        object_globs=["lib/libhiptensor.so*"],
        fetch_artifact_args="--hiptensor",
        codecov_flag="hipTensor",
    ),
    "miopen": CoverageProject(
        cmake_target="MIOpen",
        artifact_names=["miopen"],
        artifact_relpaths=["ml-libs/MIOpen/stage"],
        unsupported_reason="no coverage option in its CMake",
        stage=STAGE_MATH_LIBS,
        test_component="miopen",
        coverage_config="projects/miopen/test_categories_coverage.yaml",
        object_globs=["lib/libMIOpen.so*"],
        fetch_artifact_args="--miopen",
        codecov_flag="MIOpen",
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
    #
    # rocm-libraries -- math-libs stage, header-only
    #
    "rocprim": CoverageProject(
        cmake_target="rocPRIM",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/rocPRIM/stage", "math-libs/rocPRIM_tests/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocprim",
        coverage_config="projects/rocprim/test_categories_coverage.yaml",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
        codecov_flag="rocPRIM",
        # rocPRIM's tests are a sibling subproject, so both stage dirs are
        # overlaid and both are instrumented.
        extra_cmake_targets=["rocPRIM_tests"],
    ),
    "hipcub": CoverageProject(
        cmake_target="hipCUB",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/hipCUB/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="hipcub",
        coverage_config="projects/hipcub/test_categories_coverage.yaml",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
        codecov_flag="hipCUB",
    ),
    "rocthrust": CoverageProject(
        cmake_target="rocThrust",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/rocThrust/stage"],
        coverage_option="CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocthrust",
        coverage_config="projects/rocthrust/test_categories_coverage.yaml",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
        codecov_flag="rocThrust",
    ),
    "rocwmma": CoverageProject(
        cmake_target="rocWMMA",
        artifact_names=["rocwmma"],
        artifact_relpaths=["math-libs/rocWMMA/stage"],
        coverage_option="CODE_COVERAGE",
        stage=STAGE_MATH_LIBS,
        test_component="rocwmma",
        coverage_config="projects/rocwmma/test_categories_coverage.yaml",
        object_globs=["bin/*_test*"],
        fetch_artifact_args="--rocwmma --tests",
        codecov_flag="rocWMMA",
    ),
    #
    # rocm-systems -- comm-libs stage
    #
    "rccl": CoverageProject(
        cmake_target="rccl",
        artifact_names=["rccl"],
        artifact_relpaths=["comm-libs/rccl/stage"],
        coverage_option="ENABLE_CODE_COVERAGE",
        stage=STAGE_COMM_LIBS,
        test_component="rccl",
        coverage_config="projects/rccl/test_categories_coverage.yaml",
        object_globs=["lib/librccl.so*"],
        fetch_artifact_args="--rccl",
        codecov_flag="rccl",
        source_repo=ROCM_SYSTEMS,
    ),
    "rocshmem": CoverageProject(
        cmake_target="rocshmem",
        artifact_names=["rocshmem"],
        artifact_relpaths=["comm-libs/rocshmem/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        stage=STAGE_COMM_LIBS,
        test_component="rocshmem",
        coverage_config="projects/rocshmem/test_categories_coverage.yaml",
        object_globs=["lib/librocshmem.so*"],
        fetch_artifact_args="--rocshmem",
        codecov_flag="rocSHMEM",
        source_repo=ROCM_SYSTEMS,
    ),
    #
    # rocm-systems -- compiler-runtime stage
    #
    "rocprofiler-sdk": CoverageProject(
        cmake_target="rocprofiler-sdk",
        artifact_names=["rocprofiler-sdk"],
        artifact_relpaths=["profiler/rocprofiler-sdk/stage"],
        unsupported_reason=(
            "ROCPROFILER_BUILD_CODECOV selects gcov instrumentation, which writes .gcda files rather than the .profraw this pipeline merges"
        ),
        stage=STAGE_COMPILER_RUNTIME,
        test_component="rocprofiler-sdk",
        coverage_config="projects/rocprofiler-sdk/test_categories_coverage.yaml",
        object_globs=["lib/librocprofiler-sdk.so*"],
        fetch_artifact_args="--rocprofiler-sdk",
        codecov_flag="rocprofiler-sdk",
        source_repo=ROCM_SYSTEMS,
    ),
    "aqlprofile": CoverageProject(
        cmake_target="aqlprofile",
        artifact_names=["aqlprofile"],
        artifact_relpaths=["profiler/aqlprofile/stage"],
        unsupported_reason="no coverage option in its CMake",
        stage=STAGE_COMPILER_RUNTIME,
        test_component="aqlprofile",
        coverage_config="projects/aqlprofile/test_categories_coverage.yaml",
        object_globs=["lib/libhsa-amd-aqlprofile*.so*"],
        fetch_artifact_args="--aqlprofile",
        codecov_flag="aqlprofile",
        source_repo=ROCM_SYSTEMS,
    ),
    "amdsmi": CoverageProject(
        cmake_target="amdsmi",
        artifact_names=["core-amdsmi"],
        artifact_relpaths=["core/amdsmi/stage"],
        unsupported_reason="no coverage option in its CMake",
        stage=STAGE_COMPILER_RUNTIME,
        test_component="amdsmi",
        coverage_config="projects/amdsmi/test_categories_coverage.yaml",
        object_globs=["lib/libamd_smi.so*"],
        fetch_artifact_args="--base-only",
        codecov_flag="amdsmi",
        source_repo=ROCM_SYSTEMS,
    ),
    "rocprofiler-compute": CoverageProject(
        cmake_target="rocprofiler-compute",
        artifact_names=["rocprofiler-compute"],
        artifact_relpaths=["profiler/rocprofiler-compute/stage"],
        unsupported_reason="a Python tool with no native instrumentation option",
        stage=STAGE_COMPILER_RUNTIME,
        test_component="rocprofiler-compute",
        coverage_config="projects/rocprofiler-compute/test_categories_coverage.yaml",
        object_globs=["lib/librocprofiler-compute*.so*"],
        fetch_artifact_args="--rocprofiler-compute",
        codecov_flag="rocprofiler-compute",
        source_repo=ROCM_SYSTEMS,
    ),
    #
    # rocm-systems -- profiler-apps stage
    #
    "rocprofiler-systems": CoverageProject(
        cmake_target="rocprofiler-systems",
        artifact_names=["rocprofiler-systems"],
        artifact_relpaths=["profiler/rocprofiler-systems/stage"],
        unsupported_reason=(
            "no build-time coverage option; its coverage feature instruments other programs at runtime"
        ),
        stage=STAGE_PROFILER_APPS,
        test_component="rocprofiler-systems",
        coverage_config="projects/rocprofiler-systems/test_categories_coverage.yaml",
        object_globs=["lib/librocprofiler-systems*.so*"],
        fetch_artifact_args="--rocprofiler-systems",
        codecov_flag="rocprofiler-systems",
        source_repo=ROCM_SYSTEMS,
    ),
}

_VALID_SOURCE_REPOS = frozenset({ROCM_LIBRARIES, ROCM_SYSTEMS})
for _key, _proj in COVERAGE_PROJECTS.items():
    assert (
        _proj.source_repo in _VALID_SOURCE_REPOS
    ), f"{_key}: source_repo must be one of {sorted(_VALID_SOURCE_REPOS)}"
    # A project in a stage the workflow cannot build would be selectable and
    # then silently tested against uninstrumented binaries.
    assert (
        _proj.stage in KNOWN_STAGES
    ), f"{_key}: stage must be one of {sorted(KNOWN_STAGES)}"
    # An entry with neither would be selectable and instrument nothing; one
    # with both leaves it ambiguous whether the project can be measured.
    assert bool(_proj.coverage_option) != bool(
        _proj.unsupported_reason
    ), f"{_key}: set exactly one of coverage_option and unsupported_reason"
    # Without these the nightly test job cannot tell which files belong to the
    # project, and would measure an entirely non-instrumented install.
    if _proj.coverage_option:
        assert _proj.artifact_names, f"{_key}: needs artifacts to overlay"
        assert _proj.artifact_relpaths, f"{_key}: needs relpaths to overlay"

SUPPORTED_PROJECTS: frozenset[str] = frozenset(
    k for k, v in COVERAGE_PROJECTS.items() if v.coverage_option
)
# Measurable, but the instrumented build is broken today. Kept selectable by
# name so the block can be retested, and kept out of everything that picks
# projects on the caller's behalf.
BLOCKED_PROJECTS: frozenset[str] = frozenset(
    k for k, v in COVERAGE_PROJECTS.items() if v.blocked_reason
)
# What an empty selection defaults to: everything measurable that this workflow
# also has a build job for. Naming a project outside BUILDABLE_STAGES stays an
# error, so the narrowing applies only to the default, never to an explicit ask.
DEFAULT_PROJECTS: frozenset[str] = frozenset(
    k
    for k in SUPPORTED_PROJECTS - BLOCKED_PROJECTS
    if COVERAGE_PROJECTS[k].stage in BUILDABLE_STAGES
)
# Group membership is limited to what can actually be measured, so a group
# alias never schedules a job that is guaranteed to fail for want of profiles.
ROCM_LIBRARIES_PROJECTS: frozenset[str] = frozenset(
    k
    for k, v in COVERAGE_PROJECTS.items()
    if v.source_repo == ROCM_LIBRARIES and v.coverage_option and not v.blocked_reason
)
ROCM_SYSTEMS_PROJECTS: frozenset[str] = frozenset(
    k
    for k, v in COVERAGE_PROJECTS.items()
    if v.source_repo == ROCM_SYSTEMS and v.coverage_option and not v.blocked_reason
)

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
                    "codecov_flag": project.codecov_flag or project_key,
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

    # The group options already cover extra_cmake_targets, because emit_cmake
    # writes them into the group lists CMakeLists.txt expands.
    for key in project_keys:
        if key in grouped:
            continue
        project = COVERAGE_PROJECTS[key]
        for target in [project.cmake_target, *project.extra_cmake_targets]:
            options.append(f"-D{target.upper()}_ENABLE_COVERAGE=ON")
    return options


def resolve_build_stages(project_keys: list[str]) -> set[str]:
    """Returns the stages the selection needs, rejecting what phase 1 cannot build.

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
            "multi_arch_ci_coverage_nightly.yml to select these projects."
        )
    return stages


def emit_cmake(output_path: Path) -> None:
    """Writes the coverage group lists and option-name map as a CMake include.

    Two things come out of here. The group lists drive the
    THEROCK_COVERAGE_*_ALL options, and are limited to projects that can be
    measured and are not currently blocked. The option map is not: a blocked
    project stays in it so that naming it explicitly still instruments it.
    The option map tells therock_subproject.cmake which flag each
    subproject actually implements, since the names are not standardised
    upstream. CMakeLists.txt reads both so it does not hardcode either.
    """

    def targets(repo: str) -> list[str]:
        names: set[str] = set()
        for p in COVERAGE_PROJECTS.values():
            if p.source_repo != repo or not p.coverage_option:
                continue
            # Kept in step with the Python group sets: a group flag that
            # instrumented a project whose instrumented build is known to fail
            # would take the whole stage down with it.
            if p.blocked_reason:
                continue
            # extra_cmake_targets are part of the group: a group flag that
            # instrumented rocPRIM but not rocPRIM_tests would leave rocPRIM's
            # only reportable binaries without a coverage mapping.
            names.update([p.cmake_target, *p.extra_cmake_targets])
        return sorted(names, key=str.lower)

    # A sibling subproject is built from the same upstream source as its parent,
    # so it takes the same option.
    option_lines = []
    for p in sorted(COVERAGE_PROJECTS.values(), key=lambda v: v.cmake_target.lower()):
        if not p.coverage_option:
            continue
        for target in sorted([p.cmake_target, *p.extra_cmake_targets], key=str.lower):
            option_lines.append(
                f"set(THEROCK_COVERAGE_OPTION_{target.upper()} {p.coverage_option})\n"
            )

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
