"""
AMD GPU Family Matrix — source of truth for GitHub CI workflows.

Each MatrixEntry defines the build, test, and release configuration for a
specific GPU family + scope combination (e.g. gfx94X-dcgpu, gfx1151).
Generic scopes within a family are "all", "dcgpu", "dgpu".
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
    All MatrixEntry rows, auto-populated from every GFX* attribute defined
    in _MatrixEntries, sorted alphabetically by canonical key. To add a new
    GPU, define a new GFX* attribute there — no other registration step needed.

-------------------------------------------------------------------------------
Key lookup and auto-resolution
-------------------------------------------------------------------------------

Entries are looked up by canonical key via:
    amdgpu_family_info_matrix_all.get_entry(key)

Accepted key formats:
    "gfx94X-dcgpu"  — explicit family + scope
    "gfx1151"       — specific GPU (scope == family member)
    "gfx950"        — family name only, resolves to the entry marked
                      is_family_default=True (e.g. gfx950 → gfx950-dcgpu)

Families without a default (e.g. gfx115X) return None for family-only lookup.
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
# GPU matrix entries: one MatrixEntry per (family, scope) combination.
#
# Add new GPUs by defining a new GFX* attribute in _MatrixEntries below.
# Field defaults are defined in new_amdgpu_family_matrix_types.py — only specify
# what differs from the default.
#
# family = gfx94X, gfx115X, ...
# scope = dcgpu, dgpu, all, ... or specific GPU name like gfx1151
##########################################################################################


class _MatrixEntries:
    """All MatrixEntry definitions, scoped to avoid polluting the module namespace.

    To add a new GPU, define a new GFX* attribute here — no other change needed.
    Entries are collected in definition order into amdgpu_family_info_matrix_all.
    """

    # gfx906/908/90a split into separate families - each has different instruction
    # support (e.g., fp8 variants, WMMA) so CK/MIOpen need to build/test individually.
    GFX906 = MatrixEntry(
        family="gfx90X",
        scope="gfx906",
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
        family="gfx90X",
        scope="gfx908",
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
        family="gfx90X",
        scope="gfx90a",
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

    GFX94X_DCGPU = MatrixEntry(
        family="gfx94X",
        scope="dcgpu",
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

    GFX950_DCGPU = MatrixEntry(
        family="gfx950",
        scope="dcgpu",
        is_family_default=True,
        linux=PlatformConfig(
            build=BuildConfig(build_variants=["release", "asan", "tsan"]),
            test=TestConfig(
                runs_on=GpuRunners(test="linux-mi355-1gpu-ossci-rocm"),
                fetch_gfx_targets=["gfx950"],
            ),
        ),
    )

    GFX101X_DGPU = MatrixEntry(
        family="gfx101X",
        scope="dgpu",
        is_family_default=True,
        linux=PlatformConfig(
            # TODO(#1926): Resolve bgemm kernel hip file generation error,
            # to enable PyTorch builds
            test=TestConfig(expect_pytorch_failure=True),
        ),
        windows=PlatformConfig(),
    )

    GFX103X_DGPU = MatrixEntry(
        family="gfx103X",
        scope="dgpu",
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

    GFX110X_ALL = MatrixEntry(
        family="gfx110X",
        scope="all",
        is_family_default=True,
        linux=PlatformConfig(
            test=TestConfig(
                # TODO(#3298): Re-enable machine once HSA_STATUS_ERROR_OUT_OF_RESOURCES
                # issues are resolved.
                run_tests=False,
                runs_on=GpuRunners(test="linux-gfx110X-gpu-rocm"),
                fetch_gfx_targets=["gfx1100"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                runs_on=GpuRunners(test="windows-gfx110X-gpu-rocm"),
                fetch_gfx_targets=["gfx1100"],
                sanity_check_only_for_family=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
    )

    # gfx115X GPUs are registered individually without a family-level default.

    GFX1150 = MatrixEntry(
        family="gfx115X",
        scope="gfx1150",
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
        family="gfx115X",
        scope="gfx1151",
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
        family="gfx115X",
        scope="gfx1152",
        linux=PlatformConfig(),
        windows=PlatformConfig(),
    )

    GFX1153 = MatrixEntry(
        family="gfx115X",
        scope="gfx1153",
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

    GFX120X_ALL = MatrixEntry(
        family="gfx120X",
        scope="all",
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
# Auto-populated matrix: collects all MatrixEntry attributes from _MatrixEntries above
# in alphabetical order by canonical key. To add a new GPU, define a new GFX* attribute there.
##########################################################################################

amdgpu_family_info_matrix_all = AmdGpuFamilyMatrix(
    entries=sorted(
        (v for v in vars(_MatrixEntries).values() if isinstance(v, MatrixEntry)),
        key=lambda e: e.key,
    )
)
