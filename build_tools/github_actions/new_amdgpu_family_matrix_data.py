"""
AMD GPU Family Matrix — source of truth for GitHub CI workflows.

Each MatrixEntry defines the build, test, and release configuration for a
specific GPU target (e.g. gfx942, gfx1151). Family membership is auto-populated
from cmake/therock_amdgpu_targets.cmake.
CMake targets are defined in: cmake/therock_amdgpu_targets.cmake

Dataclass types and field documentation live in new_amdgpu_family_matrix_types.py.

-------------------------------------------------------------------------------
Exported variables
-------------------------------------------------------------------------------

all_build_variants (AllBuildVariants)
    CMake preset and artifact naming per platform and build variant
    (e.g. "release", "asan"). Consumed by the CI build job configuration.

amdgpu_family_predefined_groups (dict)
    Named groups of matrix keys selected by CI workflow triggers:
    amdgpu_presubmit  — runs on pull_request (all PRs)
    amdgpu_postsubmit — runs on push to main and release branches
    amdgpu_nightly    — runs on schedule

amdgpu_family_info_matrix_all (AmdGpuFamilyMatrix)
    All MatrixEntry rows, auto-populated from every GFX* attribute defined in
    _MatrixEntries, sorted alphabetically by target name. Each entry's family
    list is auto-populated from cmake/therock_amdgpu_targets.cmake. To add a
    new GPU, define a new GFX* attribute in _MatrixEntries.

-------------------------------------------------------------------------------
Key lookup and auto-resolution
-------------------------------------------------------------------------------

Entries are looked up by canonical key via:
    amdgpu_family_info_matrix_all.get_entry(key)

Accepted key formats:
    "gfx942"        — specific GPU target name
    "gfx1151"       — specific GPU (registered individually)
    "gfx94X-dcgpu"  — family name, resolves to the entry marked
                      is_family_default=True for that family, otherwise
                      returns None

Families without a default (e.g. gfx115X-all) return None for family lookup.
"""

##########################################################################################
# NOTE: when doing changes here, also check that they are done in amdgpu_family_matrix.py
##########################################################################################

from new_amdgpu_family_matrix_types import (
    AllBuildVariants,
    AmdGpuFamilyMatrix,
    BuildConfig,
    BuildVariantInfo,
    GpuRunners,
    MatrixEntry,
    PlatformConfig,
    ReleaseConfig,
    TestConfig,
)

from pathlib import Path


def _parse_amdgpu_targets() -> dict[str, list[str]]:
    """Parse cmake/therock_amdgpu_targets.cmake, return {gfx_target: [family_names]}.

    Reads FAMILY args from each therock_add_amdgpu_target() call. The target itself
    is not included in the returned family list (it is stored as MatrixEntry.target).
    """
    cmake_path = Path(__file__).parents[2] / "cmake" / "therock_amdgpu_targets.cmake"
    text = cmake_path.read_text()

    result: dict[str, list[str]] = {}
    for block in text.split("therock_add_amdgpu_target(")[1:]:
        target = block.split()[0]
        if "FAMILY" not in block:
            raise
        family_line = block.split("FAMILY")[1].split("\n")[0]
        result[target] = family_line.rstrip(")").split()
    return result


##########################################################################################
# Build variants: CMake preset and artifact naming per platform
##########################################################################################

all_build_variants = AllBuildVariants(
    linux={
        "release": BuildVariantInfo(
            label="release",
            suffix="",
            # TODO: Enable linux-release-package once capacity and rccl link
            # issues are resolved. https://github.com/ROCm/TheRock/issues/1781
            # cmake_preset="linux-release-package",
            cmake_preset="",
        ),
        "asan": BuildVariantInfo(
            label="asan",
            suffix="asan",
            cmake_preset="linux-release-asan",
        ),
        "tsan": BuildVariantInfo(
            label="tsan",
            suffix="tsan",
            cmake_preset="linux-release-tsan",
            expect_failure=True,
        ),
    },
    windows={
        "release": BuildVariantInfo(
            label="release",
            suffix="",
            cmake_preset="windows-release",
        ),
    },
)

##########################################################################################
# Predefined groups: named sets of matrix keys used by CI workflow triggers
##########################################################################################


def _build_groups() -> dict[str, list[str]]:
    """Build named groups of GPU family keys for use in CI workflow triggers.

    Each group is a named subset of matrix keys (e.g. for pull_request, push, schedule,
    or any custom trigger like a weekly run). New groups can be added freely here.
    Defined as a function to avoid polluting the module namespace with intermediate variables.
    """
    # Each tier is built by unioning the previous tier with new keys (set for dedup),
    # then sorted alphabetically to produce a stable, ordered list[str].
    presubmit = sorted(["gfx94X-dcgpu", "gfx110X-all", "gfx1151", "gfx120X-all"])
    postsubmit = sorted(set(presubmit) | {"gfx950-dcgpu"})
    nightly = sorted(
        set(postsubmit)
        | {
            "gfx906",
            "gfx908",
            "gfx90a",
            "gfx101X-dgpu",
            "gfx103X-dgpu",
            "gfx1150",
            "gfx1152",
            "gfx1153",
        }
    )
    return {
        "amdgpu_presubmit": presubmit,
        "amdgpu_postsubmit": postsubmit,
        "amdgpu_nightly": nightly,
    }


amdgpu_family_predefined_groups = _build_groups()

##########################################################################################
# GPU matrix entries: one MatrixEntry per GPU target.
#
# Add new GPUs by defining a new GFX* attribute in _MatrixEntries below.
# Field defaults are defined in new_amdgpu_family_matrix_types.py — only specify
# what differs from the default.
##########################################################################################


class _MatrixEntries:
    """All MatrixEntry definitions, scoped to avoid polluting the module namespace.

    To add a new GPU, define a new GFX* attribute here — no other change needed.
    Entries are auto-populated into amdgpu_family_info_matrix_all, sorted alphabetically.

    Generic scope entries (e.g. gfx110X-all, gfx101X-dgpu) are expanded to individual
    GPU targets as listed in cmake/therock_amdgpu_targets.cmake. The is_family_default
    entry carries the CI runner config; siblings get minimal config.
    """

    # gfx906/908/90a split into separate families - each has different instruction
    # support (e.g., fp8 variants, WMMA) so CK/MIOpen need to build/test individually.
    GFX906 = MatrixEntry(
        target="gfx906",
        linux=PlatformConfig(
            # Test disabled due to hardware availability
            test=TestConfig(
                sanity_check_only_for_family=True,
            ),
        ),
        # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`,
        # to enable PyTorch builds
        windows=PlatformConfig(
            test=TestConfig(expect_pytorch_failure=True),
        ),
    )

    GFX908 = MatrixEntry(
        target="gfx908",
        linux=PlatformConfig(
            # Test disabled due to hardware availability
            test=TestConfig(
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(
            test=TestConfig(expect_pytorch_failure=True),
        ),
    )

    GFX90A = MatrixEntry(
        target="gfx90a",
        linux=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(test="linux-gfx90a-gpu-rocm"),
                fetch_gfx_targets=["gfx90a"],
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(
            test=TestConfig(expect_pytorch_failure=True),
        ),
    )

    # gfx94X family — gfx942 is the only member of gfx94X-dcgpu
    GFX942 = MatrixEntry(
        target="gfx942",
        is_family_default=True,
        linux=PlatformConfig(
            build=BuildConfig(build_variants=["release", "asan", "tsan"]),
            test=TestConfig(
                runs_on=GpuRunners(
                    test="linux-mi325-1gpu-ossci-rocm-frac",
                    test_multi_gpu="linux-mi325-8gpu-ossci-rocm",
                    # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                    benchmark="linux-mi325-8gpu-ossci-rocm",
                    # TODO(#3433): Remove sandbox label once ASAN tests are passing
                    extra={"test-sandbox": "linux-mi325-8gpu-ossci-rocm-sandbox"},
                ),
                fetch_gfx_targets=["gfx942"],
            ),
        ),
    )

    # gfx950 family — gfx950 is the only member of gfx950-dcgpu
    GFX950 = MatrixEntry(
        target="gfx950",
        is_family_default=True,
        linux=PlatformConfig(
            build=BuildConfig(build_variants=["release", "asan", "tsan"]),
            test=TestConfig(
                runs_on=GpuRunners(test="linux-mi355-1gpu-ossci-rocm"),
                fetch_gfx_targets=["gfx950"],
            ),
        ),
    )

    # gfx101X family (dgpu members only) — no runner yet, gfx1010 as default
    GFX1010 = MatrixEntry(
        target="gfx1010",
        is_family_default=True,
        linux=PlatformConfig(
            # TODO(#1926): Resolve bgemm kernel hip file generation error,
            # to enable PyTorch builds
            test=TestConfig(expect_pytorch_failure=True),
        ),
        windows=PlatformConfig(),
    )
    GFX1011 = MatrixEntry(
        target="gfx1011", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1012 = MatrixEntry(
        target="gfx1012", linux=PlatformConfig(), windows=PlatformConfig()
    )

    # gfx103X family (dgpu members only) — gfx1030 confirmed by runner linux-gfx1030-gpu-rocm
    GFX1030 = MatrixEntry(
        target="gfx1030",
        is_family_default=True,
        linux=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(test="linux-gfx1030-gpu-rocm"),
                sanity_check_only_for_family=True,
                fetch_gfx_targets=["gfx1030"],
            ),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                # TODO(#3200): Re-enable machine once it is stable
                run_tests=False,
                runs_on=GpuRunners(test="windows-gfx1030-gpu-rocm"),
                sanity_check_only_for_family=True,
            ),
        ),
    )
    GFX1031 = MatrixEntry(
        target="gfx1031", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1032 = MatrixEntry(
        target="gfx1032", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1034 = MatrixEntry(
        target="gfx1034", linux=PlatformConfig(), windows=PlatformConfig()
    )

    # gfx110X family (all members, including igpu gfx1103) — gfx1101 as default
    GFX1100 = MatrixEntry(
        target="gfx1100", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1101 = MatrixEntry(
        target="gfx1101",
        is_family_default=True,
        linux=PlatformConfig(
            test=TestConfig(
                # TODO(#3298): Re-enable machine once HSA_STATUS_ERROR_OUT_OF_RESOURCES
                # issues are resolved.
                run_tests=False,
                runs_on=GpuRunners(test="linux-gfx110X-gpu-rocm"),
                fetch_gfx_targets=["gfx1101"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(test="windows-gfx110X-gpu-rocm"),
                fetch_gfx_targets=["gfx1101"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
    )
    GFX1102 = MatrixEntry(
        target="gfx1102", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1103 = MatrixEntry(target="gfx1103", linux=PlatformConfig())  # igpu, no windows

    # gfx115X GPUs registered individually — no family-level default
    GFX1150 = MatrixEntry(
        target="gfx1150",
        linux=PlatformConfig(
            test=TestConfig(
                # TODO(#3199): Re-enable machine once it is stable
                run_tests=False,
                runs_on=GpuRunners(test="linux-gfx1150-gpu-rocm"),
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(),
    )

    GFX1151 = MatrixEntry(
        target="gfx1151",
        linux=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(
                    test="linux-gfx1151-gpu-rocm",
                    extra={"oem": "linux-strix-halo-gpu-rocm-oem"},
                ),
                fetch_gfx_targets=["gfx1151"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(
                    test="windows-gfx1151-gpu-rocm",
                    # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                    benchmark="windows-gfx1151-gpu-rocm",
                ),
                fetch_gfx_targets=["gfx1151"],
                # TODO(#3299): Re-enable smoke tests once capacity is available
                test_scope="full",
            ),
        ),
    )

    GFX1152 = MatrixEntry(
        target="gfx1152",
        linux=PlatformConfig(),
        windows=PlatformConfig(),
    )

    GFX1153 = MatrixEntry(
        target="gfx1153",
        linux=PlatformConfig(
            test=TestConfig(
                # TODO(#2682): Re-enable machine once it is stable
                run_tests=False,
                runs_on=GpuRunners(test="linux-gfx1153-gpu-rocm"),
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(),
    )

    # gfx120X family (all members) — gfx1201 as default
    GFX1200 = MatrixEntry(
        target="gfx1200", linux=PlatformConfig(), windows=PlatformConfig()
    )
    GFX1201 = MatrixEntry(
        target="gfx1201",
        is_family_default=True,
        linux=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(test="linux-gfx120X-gpu-rocm"),
                fetch_gfx_targets=["gfx1200", "gfx1201"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                # TODO(#2962): Re-enable machine once sanity checks work with this architecture
                run_tests=False,
                runs_on=GpuRunners(test="windows-gfx120X-gpu-rocm"),
                fetch_gfx_targets=["gfx1200", "gfx1201"],
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
    )


##########################################################################################
# Auto-populated matrix: collects all MatrixEntry attributes from _MatrixEntries above,
# sorted alphabetically by target name. family is auto-populated from cmake.
# To add a new GPU, define a new GFX* attribute in _MatrixEntries — no other change needed.
##########################################################################################

amdgpu_family_info_matrix_all = AmdGpuFamilyMatrix(
    entries=sorted(
        (v for v in vars(_MatrixEntries).values() if isinstance(v, MatrixEntry)),
        key=lambda e: e.target,
    ),
    cmake_families=_parse_amdgpu_targets(),
)
