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
    All MatrixEntry rows, auto-populated from every _GFX* variable defined
    below in definition order. To add a new GPU, define a new _GFX* variable
    — no other registration step is needed.

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

amdgpu_family_predefined_groups = {
    # The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
    "amdgpu_presubmit": ["gfx94X-dcgpu", "gfx110X-all", "gfx1151", "gfx120X-all"],
    # The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
    "amdgpu_postsubmit": ["gfx950-dcgpu"],
    # The 'nightly' matrix runs on 'schedule' triggers.
    "amdgpu_nightly": [
        "gfx90X-dcgpu",
        "gfx101X-dgpu",
        "gfx103X-dgpu",
        "gfx1150",
        "gfx1152",
        "gfx1153",
    ],
}

##########################################################################################
# GPU matrix entries: one MatrixEntry per (family, scope) combination.
# Any variable starting with _GFX is automatically added to amdgpu_family_info_matrix_all.
#
# Field defaults are defined in new_amdgpu_family_matrix_types.py — only specify
# what differs from the default.
#
# family = gfx94X, gfx115X, ...
# scope = dcgpu, dgpu, all, ... or specific GPU name like gfx1151
##########################################################################################

_GFX90X_DCGPU = MatrixEntry(
    family="gfx90X",
    scope="dcgpu",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            runs_on=GpuRunners(test="linux-gfx90X-gpu-rocm"),
            fetch_gfx_targets=["gfx90a"],
            sanity_check_only_for_family=True,
        ),
    ),
    # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`,
    # to enable PyTorch builds
    windows=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(expect_pytorch_failure=True),
    ),
)

_GFX94X_DCGPU = MatrixEntry(
    family="gfx94X",
    scope="dcgpu",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(build_variants=["release", "asan"]),
        test=TestConfig(
            runs_on=GpuRunners(
                test="linux-mi325-1gpu-ossci-rocm-frac",
                test_multi_gpu="linux-mi325-8gpu-ossci-rocm",
                # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                benchmark="linux-mi325-8gpu-ossci-rocm",
            ),
            fetch_gfx_targets=["gfx942"],
        ),
        release=ReleaseConfig(push_on_success=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
    ),
)

_GFX950_DCGPU = MatrixEntry(
    family="gfx950",
    scope="dcgpu",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(build_variants=["release", "asan"]),
        test=TestConfig(
            runs_on=GpuRunners(test="linux-mi355-1gpu-ossci-rocm"),
            fetch_gfx_targets=["gfx950"],
        ),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
    ),
)

_GFX101X_DGPU = MatrixEntry(
    family="gfx101X",
    scope="dgpu",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(),
        # TODO(#1926): Resolve bgemm kernel hip file generation error,
        # to enable PyTorch builds
        test=TestConfig(expect_pytorch_failure=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
    ),
)

_GFX103X_DGPU = MatrixEntry(
    family="gfx103X",
    scope="dgpu",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            # TODO(#2740): Re-enable machine once `amdsmi` test is fixed
            # fetch_gfx_targets should be ["gfx1030"] when re-enabled
            runs_on=GpuRunners(test="linux-gfx1030-gpu-rocm"),
            run_tests=False,
            sanity_check_only_for_family=True,
        ),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            # TODO(#3200): Re-enable machine once it is stable
            runs_on=GpuRunners(test="windows-gfx1030-gpu-rocm"),
            run_tests=False,
            sanity_check_only_for_family=True,
            expect_pytorch_failure=True,
        ),
    ),
)

_GFX110X_ALL = MatrixEntry(
    family="gfx110X",
    scope="all",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            # TODO(#2740): Re-enable machine once `amdsmi` test is fixed
            # fetch_gfx_targets should be ["gfx1100"] when re-enabled
            runs_on=GpuRunners(test="linux-gfx110X-gpu-rocm"),
            run_tests=False,
            sanity_check_only_for_family=True,
        ),
        release=ReleaseConfig(push_on_success=True, bypass_tests_for_releases=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            runs_on=GpuRunners(test="windows-gfx110X-gpu-rocm"),
            fetch_gfx_targets=["gfx1100"],
            sanity_check_only_for_family=True,
        ),
        release=ReleaseConfig(push_on_success=True, bypass_tests_for_releases=True),
    ),
)

# gfx115X GPUs are registered individually without a family-level default.

_GFX1150 = MatrixEntry(
    family="gfx115X",
    scope="gfx1150",
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            # TODO(#3199): Re-enable machine once it is stable
            runs_on=GpuRunners(test="linux-gfx1150-gpu-rocm"),
            run_tests=False,
            sanity_check_only_for_family=True,
        ),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
    ),
)

_GFX1151 = MatrixEntry(
    family="gfx115X",
    scope="gfx1151",
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            runs_on=GpuRunners(
                test="linux-gfx1151-gpu-rocm",
                oem="linux-strix-halo-gpu-rocm-oem",
            ),
            fetch_gfx_targets=["gfx1151"],
            sanity_check_only_for_family=True,
        ),
        release=ReleaseConfig(push_on_success=True, bypass_tests_for_releases=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            runs_on=GpuRunners(
                test="windows-gfx1151-gpu-rocm",
                # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                benchmark="windows-gfx1151-gpu-rocm",
            ),
            fetch_gfx_targets=["gfx1151"],
        ),
    ),
)

_GFX1152 = MatrixEntry(
    family="gfx115X",
    scope="gfx1152",
    linux=PlatformConfig(
        build=BuildConfig(expect_failure=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(expect_failure=True),
    ),
)

_GFX1153 = MatrixEntry(
    family="gfx115X",
    scope="gfx1153",
    linux=PlatformConfig(
        build=BuildConfig(expect_failure=True),
        test=TestConfig(
            # TODO(#2682): Re-enable machine once it is stable
            runs_on=GpuRunners(test="linux-gfx1153-gpu-rocm"),
            run_tests=False,
            sanity_check_only_for_family=True,
        ),
    ),
    windows=PlatformConfig(
        build=BuildConfig(expect_failure=True),
    ),
)

_GFX120X_ALL = MatrixEntry(
    family="gfx120X",
    scope="all",
    is_family_default=True,
    linux=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            runs_on=GpuRunners(test="linux-gfx120X-gpu-rocm"),
            fetch_gfx_targets=["gfx1200", "gfx1201"],
            sanity_check_only_for_family=True,
        ),
        release=ReleaseConfig(push_on_success=True, bypass_tests_for_releases=True),
    ),
    windows=PlatformConfig(
        build=BuildConfig(),
        test=TestConfig(
            # TODO(#2962): Re-enable machine once sanity checks work with this architecture
            # fetch_gfx_targets should be ["gfx1200", "gfx1201"] when re-enabled
            runs_on=GpuRunners(test="windows-gfx120X-gpu-rocm"),
            run_tests=False,
        ),
        release=ReleaseConfig(push_on_success=True, bypass_tests_for_releases=True),
    ),
)

##########################################################################################
# Auto-populated matrix: collects all _GFX* entries above in definition order.
# To add a new GPU, define a new _GFX* variable above — no other change needed here.
##########################################################################################

amdgpu_family_info_matrix_all = AmdGpuFamilyMatrix(
    entries=[
        v
        for k, v in globals().items()
        if k.startswith("_GFX") and isinstance(v, MatrixEntry)
    ]
)
