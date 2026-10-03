# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AMD GPU Family Matrix — source of truth for GitHub CI workflows.

Each MatrixEntry defines build / test / release config for a GPU target
(e.g. gfx942, gfx1151). Family membership is auto-populated from
cmake/therock_amdgpu_targets.cmake. Dataclass types and field documentation
live in new_amdgpu_family_matrix_types.py.

Exported:
    all_build_variants               — CMake preset / artifact-naming per
                                       platform and build variant.
    amdgpu_family_predefined_groups  — named subsets used by CI triggers
                                       (presubmit / postsubmit / nightly).
    amdgpu_family_info_matrix_all    — the complete AmdGpuFamilyMatrix,
                                       auto-collected from _MatrixEntries.

To add a GPU: define a new GFX* attribute in _MatrixEntries below — no other
change needed. Lookup keys: exact target ("gfx942") or family name
("gfx94X-dcgpu", which resolves via is_family_default). See
new_amdgpu_family_matrix_types.py for the full lookup API.
"""

##########################################################################################
# NOTE: when doing changes here, also check that they are done in amdgpu_family_matrix.py
##########################################################################################

import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.cmake_amdgpu_targets import parse_amdgpu_targets_cmake

from new_amdgpu_family_matrix_types import (
    AllBuildVariants,
    AmdGpuFamilyMatrix,
    BuildConfig,
    BuildVariantInfo,
    MatrixEntry,
    PlatformConfig,
    ReleaseConfig,
    TestConfig,
)


def _parse_amdgpu_targets() -> dict[str, list[str]]:
    """Parse cmake/therock_amdgpu_targets.cmake, return {gfx_target: [family_names]}.

    Returns the explicit FAMILY arguments per target. The target itself is not
    included in the returned family list (it is stored as MatrixEntry.target).
    Delegates parsing to _therock_utils.cmake_amdgpu_targets so the tokenizer
    handles quoted product names, inline comments, and EXCLUDE_TARGET_PROJECTS.
    """
    cmake_path = Path(__file__).parents[2] / "cmake" / "therock_amdgpu_targets.cmake"
    return {
        info.gfx_target: info.families
        for info in parse_amdgpu_targets_cmake(cmake_path)
    }


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
    """Named groups of matrix keys for CI triggers. Each tier is a superset of
    the previous one (presubmit ⊂ postsubmit ⊂ nightly)."""
    presubmit = sorted(["gfx94X-dcgpu", "gfx110X-all", "gfx1151", "gfx120X-all"])
    postsubmit = sorted(set(presubmit) | {"gfx950-dcgpu"})
    nightly = sorted(
        set(postsubmit)
        | {
            "gfx900",
            "gfx906",
            "gfx908",
            "gfx90a",
            "gfx101X-dgpu",
            "gfx103X-all",
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
# Field defaults live in new_amdgpu_family_matrix_types.py — only set what differs.
# Runner labels are auto-filled from the inventories in
# new_amdgpu_family_matrix_runners.py; do NOT pass `runs_on=` here unless overriding.
##########################################################################################


class _MatrixEntries:
    """All MatrixEntry definitions, namespaced to keep module globals clean.
    Auto-collected into amdgpu_family_info_matrix_all alphabetically. Within
    a family, the is_family_default entry carries the full CI config; siblings
    get minimal config and inherit on family-name lookups."""

    GFX900 = MatrixEntry(
        target="gfx900",
        linux=PlatformConfig(
            # Test disabled due to hardware availability
            test=TestConfig(
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(),
    )

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
        windows=PlatformConfig(),
    )

    GFX908 = MatrixEntry(
        target="gfx908",
        linux=PlatformConfig(
            # Test disabled due to hardware availability
            test=TestConfig(
                sanity_check_only_for_family=True,
            ),
        ),
        windows=PlatformConfig(),
    )

    GFX90A = MatrixEntry(
        target="gfx90a",
        linux=PlatformConfig(
            test=TestConfig(
                fetch_gfx_targets=["gfx90a"],
                bypass_tests_for_unscheduled=True,
            ),
        ),
        windows=PlatformConfig(),
    )

    # gfx94X family — gfx942 is the only member of gfx94X-dcgpu
    GFX942 = MatrixEntry(
        target="gfx942",
        is_family_default=True,
        linux=PlatformConfig(
            build=BuildConfig(build_variants=["release", "asan", "tsan"]),
            test=TestConfig(
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
                fetch_gfx_targets=["gfx950"],
            ),
        ),
    )

    # gfx101X family (dgpu members only) — no runner yet, gfx1010 as default
    GFX1010 = MatrixEntry(
        target="gfx1010",
        is_family_default=True,
        linux=PlatformConfig(),
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
                fetch_gfx_targets=["gfx1030"],
                bypass_tests_for_unscheduled=True,
            ),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                bypass_tests_for_unscheduled=True,
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
                fetch_gfx_targets=[],
                bypass_tests_for_unscheduled=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                fetch_gfx_targets=["gfx1100", "gfx1101"],
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
                bypass_tests_for_unscheduled=True,
            ),
        ),
        windows=PlatformConfig(),
    )

    GFX1151 = MatrixEntry(
        target="gfx1151",
        linux=PlatformConfig(
            test=TestConfig(
                fetch_gfx_targets=["gfx1151"],
                bypass_tests_for_unscheduled=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                fetch_gfx_targets=["gfx1151"],
                bypass_tests_for_unscheduled=True,
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
                bypass_tests_for_unscheduled=True,
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
                fetch_gfx_targets=["gfx1200", "gfx1201"],
                bypass_tests_for_unscheduled=True,
            ),
            release=ReleaseConfig(bypass_tests_for_releases=True),
        ),
        windows=PlatformConfig(
            test=TestConfig(
                fetch_gfx_targets=[],
                bypass_tests_for_unscheduled=True,
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
