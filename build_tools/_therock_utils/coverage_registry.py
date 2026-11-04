# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The coverage project registry, and the CMake include generated from it.

`COVERAGE_PROJECTS` is the allowlist of projects that can be built with LLVM
source-based coverage, with everything the build and the coverage CI need to
know about each one. CMake reads it at configure time through `--emit-cmake`,
so this module depends only on the build topology and never on CI code.

Usage:
  python build_tools/_therock_utils/coverage_registry.py --emit-cmake PATH
"""

import argparse
import functools
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from _therock_utils.build_topology import get_topology

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

# Only compiler-runtime and math-libs have build jobs. rccl and rocshmem
# (comm-libs) are blocked upstream, so selecting them is rejected here rather
# than failing hours later on a missing inbound artifact.
BUILDABLE_STAGES = frozenset({STAGE_COMPILER_RUNTIME, STAGE_MATH_LIBS})


@functools.lru_cache(maxsize=1)
def _topology():
    """BUILD_TOPOLOGY.toml, parsed once per process.

    CMake runs this script on every configure to emit the coverage CMake
    include, so the parse is cached rather than repeated per project.
    """
    return get_topology()


@functools.lru_cache(maxsize=None)
def _stage_for_artifact(artifact_name: str) -> str:
    """Resolves the build stage that produces an artifact."""
    stage = _topology().get_stage_for_artifact(artifact_name)
    if stage is None:
        raise ValueError(
            f"no build stage in BUILD_TOPOLOGY.toml produces artifact "
            f"'{artifact_name}'"
        )
    return stage


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
            so therock_coverage.cmake translates TheRock's flag into this
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
        object_globs: Globs, relative to the extracted artifact directory,
            matching the instrumented binaries handed to `llvm-cov`.
        fetch_artifact_args: Arguments to install_rocm_from_artifacts.py that
            pull the instrumented libraries into the report generation job.
        source_repo: The repo this project ships from (ROCM_LIBRARIES or
            ROCM_SYSTEMS); drives the group aliases.
        extra_cmake_targets: Additional subprojects to instrument alongside
            cmake_target. Header-only projects report against their own test
            binaries, and some of those are built by a sibling subproject
            (rocPRIM's live in rocPRIM_tests), which therefore has to be
            instrumented too or the binaries carry no coverage mapping.
        codecov_flag_override: Set only when the Codecov flag differs from
            cmake_target, which it does for rocshmem (reported as rocSHMEM).
        stage_override: Set only when the build stage cannot be derived from
            artifact_names, which no project needs today.

    The `key`, `stage`, `test_component`, `coverage_config` and `codecov_flag`
    properties below are derived rather than stored: repeating them per entry
    gave three dozen chances to disagree with the build topology and with the
    registry key itself.
    """

    cmake_target: str
    coverage_option: str = ""
    unsupported_reason: str = ""
    blocked_reason: str = ""
    artifact_names: list[str] = field(default_factory=list)
    artifact_relpaths: list[str] = field(default_factory=list)
    object_globs: list[str] = field(default_factory=list)
    fetch_artifact_args: str = ""
    source_repo: str = ROCM_LIBRARIES
    extra_cmake_targets: list[str] = field(default_factory=list)
    codecov_flag_override: str = ""
    stage_override: str = ""

    @property
    def key(self) -> str:
        """The project's key in COVERAGE_PROJECTS."""
        return self.cmake_target.lower()

    @property
    def stage(self) -> str:
        """Build stage that produces the project.

        Read out of the build topology via the project's first artifact, so a
        project moving between stages upstream cannot leave a stale stage here.
        The nightly needs a build job for this stage, so onboarding a project
        from a new stage still means adding one.
        """
        if self.stage_override:
            return self.stage_override
        if not self.artifact_names:
            raise ValueError(
                f"{self.key}: stage is derived from artifact_names, so set "
                "either that or stage_override"
            )
        # Every artifact of a project has to come from one stage for this to be
        # well defined; configure_coverage_ci_test.py checks that for all of
        # them, not just the first.
        return _stage_for_artifact(self.artifact_names[0])

    @property
    def test_component(self) -> str:
        """Key in fetch_test_configurations.py's test matrix."""
        return self.key

    @property
    def coverage_config(self) -> str:
        """Per-project coverage metadata file.

        Relative to the root of the repository named by COVERAGE_CONFIG_SOURCE.
        """
        return f"projects/{self.key}/test_categories_coverage.yaml"

    @property
    def codecov_flag(self) -> str:
        """Flag the report is filed under in Codecov."""
        return self.codecov_flag_override or self.cmake_target


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
        object_globs=["lib/librocrand.so*"],
        fetch_artifact_args="--rand",
    ),
    "hiprand": CoverageProject(
        cmake_target="hipRAND",
        artifact_names=["rand"],
        artifact_relpaths=["math-libs/hipRAND/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/libhiprand.so*"],
        fetch_artifact_args="--rand",
    ),
    "rocfft": CoverageProject(
        cmake_target="rocFFT",
        artifact_names=["fft"],
        artifact_relpaths=["math-libs/rocFFT/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/librocfft.so*"],
        fetch_artifact_args="--fft",
    ),
    "hipfft": CoverageProject(
        cmake_target="hipFFT",
        artifact_names=["fft"],
        artifact_relpaths=["math-libs/hipFFT/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/libhipfft.so*"],
        fetch_artifact_args="--fft",
    ),
    "rocblas": CoverageProject(
        cmake_target="rocBLAS",
        artifact_names=["blas"],
        artifact_relpaths=["math-libs/BLAS/rocBLAS/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/librocblas.so*"],
        fetch_artifact_args="--blas",
    ),
    "hipblas": CoverageProject(
        cmake_target="hipBLAS",
        artifact_names=["blas"],
        artifact_relpaths=["math-libs/BLAS/hipBLAS/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/libhipblas.so*"],
        fetch_artifact_args="--blas",
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
        object_globs=["lib/libhipblaslt.so*"],
        fetch_artifact_args="--blas",
    ),
    "rocsparse": CoverageProject(
        cmake_target="rocSPARSE",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/rocSPARSE/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/librocsparse.so*"],
        fetch_artifact_args="--sparse",
    ),
    "hipsparse": CoverageProject(
        cmake_target="hipSPARSE",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/hipSPARSE/stage"],
        unsupported_reason=(
            "HIPSPARSE_ENABLE_COVERAGE selects gcov instrumentation, which writes .gcda files rather than the .profraw this pipeline merges"
        ),
        object_globs=["lib/libhipsparse.so*"],
        fetch_artifact_args="--sparse",
    ),
    "hipsparselt": CoverageProject(
        cmake_target="hipSPARSELt",
        artifact_names=["sparse"],
        artifact_relpaths=["math-libs/BLAS/hipSPARSELt/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/libhipsparselt.so*"],
        fetch_artifact_args="--sparse",
    ),
    "rocsolver": CoverageProject(
        cmake_target="rocSOLVER",
        artifact_names=["solver"],
        artifact_relpaths=["math-libs/BLAS/rocSOLVER/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/librocsolver.so*"],
        fetch_artifact_args="--solver",
    ),
    "hipsolver": CoverageProject(
        cmake_target="hipSOLVER",
        artifact_names=["solver"],
        artifact_relpaths=["math-libs/BLAS/hipSOLVER/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/libhipsolver.so*"],
        fetch_artifact_args="--solver",
    ),
    "rocalution": CoverageProject(
        cmake_target="rocALUTION",
        artifact_names=["rocalution"],
        artifact_relpaths=["math-libs/rocALUTION/stage"],
        unsupported_reason=(
            "BUILD_CODE_COVERAGE selects gcov instrumentation, which writes .gcda files rather than the .profraw this pipeline merges"
        ),
        object_globs=["lib/librocalution.so*"],
        fetch_artifact_args="--rocalution",
    ),
    "hiptensor": CoverageProject(
        cmake_target="hipTensor",
        artifact_names=["hiptensor"],
        artifact_relpaths=["math-libs/hipTensor/stage"],
        # Prefixed, unlike rocALUTION's bare CODE_COVERAGE just above. Getting
        # this wrong is silent: the option is simply never read, the library
        # builds uninstrumented, and the tests pass having written no profile.
        coverage_option="HIPTENSOR_CODE_COVERAGE",
        # Instrumenting hipTensor pushes libhiptensor.so past the 2GB that the
        # small code model allows, and the link dies on out-of-range
        # R_X86_64_PC32 relocations from .rodata into .text. Upstream applies
        # coverage as a blanket add_compile_options(-fprofile-instr-generate
        # -fcoverage-mapping), and hipTensor is built on composable_kernel, so
        # the coverage mapping and name data scale with an already enormous
        # amount of instantiated template code. Building it with a larger code
        # model is the likely fix and belongs upstream. Blocked meanwhile
        # because the failure takes the whole math-libs stage with it, and
        # every other project in the stage along with it.
        blocked_reason=(
            "libhiptensor.so exceeds the small code model when instrumented; "
            "the link fails on out-of-range R_X86_64_PC32 relocations"
        ),
        object_globs=["lib/libhiptensor.so*"],
        fetch_artifact_args="--hiptensor",
    ),
    "miopen": CoverageProject(
        cmake_target="MIOpen",
        artifact_names=["miopen"],
        artifact_relpaths=["ml-libs/MIOpen/stage"],
        unsupported_reason="no coverage option in its CMake",
        object_globs=["lib/libMIOpen.so*"],
        fetch_artifact_args="--miopen",
    ),
    "hipdnn": CoverageProject(
        cmake_target="hipDNN",
        artifact_names=["hipdnn"],
        artifact_relpaths=["ml-libs/hipDNN/stage"],
        coverage_option="HIPDNN_ENABLE_COVERAGE",
        # Not libhipdnn.so: hipDNN's shared library is the backend, which is
        # also the name TheRock's own packaging metadata records for it.
        object_globs=["lib/libhipdnn_backend.so*"],
        fetch_artifact_args="--hipdnn",
    ),
    #
    # rocm-libraries -- math-libs stage, header-only
    #
    "rocprim": CoverageProject(
        cmake_target="rocPRIM",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/rocPRIM/stage", "math-libs/rocPRIM_tests/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
        # rocPRIM's tests are a sibling subproject, so both stage dirs are
        # overlaid and both are instrumented.
        extra_cmake_targets=["rocPRIM_tests"],
    ),
    "hipcub": CoverageProject(
        cmake_target="hipCUB",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/hipCUB/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
    ),
    "rocthrust": CoverageProject(
        cmake_target="rocThrust",
        artifact_names=["prim"],
        artifact_relpaths=["math-libs/rocThrust/stage"],
        coverage_option="CODE_COVERAGE",
        object_globs=["bin/test_*"],
        fetch_artifact_args="--prim --tests",
    ),
    "rocwmma": CoverageProject(
        cmake_target="rocWMMA",
        artifact_names=["rocwmma"],
        artifact_relpaths=["math-libs/rocWMMA/stage"],
        coverage_option="CODE_COVERAGE",
        object_globs=["bin/*_test*"],
        fetch_artifact_args="--rocwmma --tests",
    ),
    #
    # rocm-systems -- comm-libs stage
    #
    "rccl": CoverageProject(
        cmake_target="rccl",
        artifact_names=["rccl"],
        artifact_relpaths=["comm-libs/rccl/stage"],
        coverage_option="ENABLE_CODE_COVERAGE",
        object_globs=["lib/librccl.so*"],
        fetch_artifact_args="--rccl",
        source_repo=ROCM_SYSTEMS,
    ),
    "rocshmem": CoverageProject(
        cmake_target="rocshmem",
        artifact_names=["rocshmem"],
        artifact_relpaths=["comm-libs/rocshmem/stage"],
        coverage_option="BUILD_CODE_COVERAGE",
        object_globs=["lib/librocshmem.so*"],
        fetch_artifact_args="--rocshmem",
        source_repo=ROCM_SYSTEMS,
        # The only project whose Codecov flag is not its cmake_target.
        codecov_flag_override="rocSHMEM",
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
        object_globs=["lib/librocprofiler-sdk.so*"],
        fetch_artifact_args="--rocprofiler-sdk",
        source_repo=ROCM_SYSTEMS,
    ),
    "aqlprofile": CoverageProject(
        cmake_target="aqlprofile",
        artifact_names=["aqlprofile"],
        artifact_relpaths=["profiler/aqlprofile/stage"],
        unsupported_reason="no coverage option in its CMake",
        object_globs=["lib/libhsa-amd-aqlprofile*.so*"],
        fetch_artifact_args="--aqlprofile",
        source_repo=ROCM_SYSTEMS,
    ),
    "amdsmi": CoverageProject(
        cmake_target="amdsmi",
        artifact_names=["core-amdsmi"],
        artifact_relpaths=["core/amdsmi/stage"],
        unsupported_reason="no coverage option in its CMake",
        object_globs=["lib/libamd_smi.so*"],
        fetch_artifact_args="--base-only",
        source_repo=ROCM_SYSTEMS,
    ),
    "rocprofiler-compute": CoverageProject(
        cmake_target="rocprofiler-compute",
        artifact_names=["rocprofiler-compute"],
        artifact_relpaths=["profiler/rocprofiler-compute/stage"],
        unsupported_reason="a Python tool with no native instrumentation option",
        object_globs=["lib/librocprofiler-compute*.so*"],
        fetch_artifact_args="--rocprofiler-compute",
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
        object_globs=["lib/librocprofiler-systems*.so*"],
        fetch_artifact_args="--rocprofiler-systems",
        source_repo=ROCM_SYSTEMS,
    ),
}

_VALID_SOURCE_REPOS = frozenset({ROCM_LIBRARIES, ROCM_SYSTEMS})


class CoverageRegistryError(ValueError):
    """Raised when COVERAGE_PROJECTS is internally inconsistent."""


def validate_registry(projects: dict[str, CoverageProject]) -> None:
    """Checks the hand-written parts of the registry.

    Raises rather than asserts: `assert` is compiled out under `python -O` or
    PYTHONOPTIMIZE, and CMake runs this script on every configure with
    whatever environment the build inherits. A registry mistake has to fail
    the configure, not emit a quietly wrong therock_coverage_projects.cmake.

    Every problem is collected before raising, so adding a project reports all
    of its mistakes at once rather than one per run.
    """
    problems: list[str] = []
    for key, proj in projects.items():
        if proj.source_repo not in _VALID_SOURCE_REPOS:
            problems.append(
                f"{key}: source_repo must be one of {sorted(_VALID_SOURCE_REPOS)}, "
                f"not '{proj.source_repo}'"
            )
        # test_component and coverage_config are derived from cmake_target, so
        # a key that disagrees with it would point the test matrix and the
        # metadata lookup at a project other than the one being selected.
        if proj.key != key:
            problems.append(
                f"{key}: key must be cmake_target lower cased, not '{proj.key}'"
            )
        # An entry with neither would be selectable and instrument nothing; one
        # with both leaves it ambiguous whether the project can be measured.
        if proj.coverage_option and proj.unsupported_reason:
            problems.append(
                f"{key}: set only one of coverage_option and unsupported_reason"
            )
        elif not proj.coverage_option and not proj.unsupported_reason:
            problems.append(
                f"{key}: set either coverage_option (if it can be measured) or "
                "unsupported_reason (if it cannot)"
            )
        # Without these the nightly test job cannot tell which files belong to
        # the project, and would measure an entirely non-instrumented install.
        if proj.coverage_option and not proj.artifact_names:
            problems.append(f"{key}: needs artifact_names to overlay")
        if proj.coverage_option and not proj.artifact_relpaths:
            problems.append(f"{key}: needs artifact_relpaths to overlay")
        # The stage comes from the build topology, so this catches an artifact
        # that was renamed or moved to a stage the coverage pipeline does not
        # know about. Such a project would be selectable and then silently
        # tested against uninstrumented binaries.
        try:
            stage = proj.stage
        except ValueError as e:
            problems.append(f"{key}: {e}")
        else:
            if stage not in KNOWN_STAGES:
                problems.append(
                    f"{key}: stage must be one of {sorted(KNOWN_STAGES)}, "
                    f"not '{stage}'"
                )
    if problems:
        raise CoverageRegistryError(
            "COVERAGE_PROJECTS in "
            f"{Path(__file__).name} is inconsistent:\n  " + "\n  ".join(problems)
        )


validate_registry(COVERAGE_PROJECTS)

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


def emit_cmake(output_path: Path) -> None:
    """Writes the coverage group lists and option-name map as a CMake include.

    Two things come out of here. The group lists drive the
    THEROCK_COVERAGE_*_ALL options, and are limited to projects that can be
    measured and are not currently blocked. The option map is not: a blocked
    project stays in it so that naming it explicitly still instruments it.
    The option map tells therock_coverage.cmake which flag each
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
        "# Generated by coverage_registry.py --emit-cmake; do not edit.\n"
        "# Edit COVERAGE_PROJECTS in\n"
        "# build_tools/_therock_utils/coverage_registry.py instead.\n"
        f"set(THEROCK_COVERAGE_ROCM_LIBRARIES_PROJECTS {' '.join(targets(ROCM_LIBRARIES))})\n"
        f"set(THEROCK_COVERAGE_ROCM_SYSTEMS_PROJECTS {' '.join(targets(ROCM_SYSTEMS))})\n"
        + "".join(option_lines)
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-cmake",
        type=Path,
        required=True,
        metavar="PATH",
        help="Write the coverage group lists and option map as a CMake include to PATH.",
    )
    args = parser.parse_args(argv)
    emit_cmake(args.emit_cmake)
    return 0


if __name__ == "__main__":
    sys.exit(main())
