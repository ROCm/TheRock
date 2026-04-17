# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows.

* Each entry determines which families and test runners are available to use
* Each group determines which entries run by default on workflow triggers

For presubmit, postsubmit and nightly family selection:

- presubmit runs the targets from presubmit dictionary on pull requests
- postsubmit runs the targets from presubmit and postsubmit dictionaries on pushes to main branch
- nightly runs targets from presubmit, postsubmit and nightly dictionaries

TODO(#2200): clarify AMD GPU family selection
"""

#############################################################################################
# NOTE: when doing changes here, also check that they are done in new_amdgpu_family_matrix.py
#############################################################################################

all_build_variants = {
    "linux": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            # TODO: Enable linux-release-package once capacity and rccl link
            # issues are resolved. https://github.com/ROCm/TheRock/issues/1781
            # "build_variant_cmake_preset": "linux-release-package",
            "build_variant_cmake_preset": "",
        },
        "asan": {
            "build_variant_label": "asan",
            "build_variant_suffix": "asan",
            "build_variant_cmake_preset": "linux-release-asan",
        },
        "tsan": {
            "build_variant_label": "tsan",
            "build_variant_suffix": "tsan",
            "build_variant_cmake_preset": "linux-release-tsan",
        },
    },
    "windows": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            "build_variant_cmake_preset": "windows-release",
        },
    },
}

"""
amdgpu_family_info_matrix dictionary fields:
- test-runs-on: (required) GitHub runner label for this architecture
- test-runs-on-alternate: (optional) Alternate runner label for load balancing across runner pools
- test-runs-on-alternate-weight: (optional) Probability (0.0-1.0) of selecting the alternate runner.
- test-runs-on-multi-gpu: (optional) GitHub runner label for multi-GPU tests for this architecture
- benchmark-runs-on: (optional) GitHub runner label for benchmarks for this architecture
- test-runs-on-kernel: (optional) dict of kernel-specific runner labels, keyed by kernel type (e.g. "oem")
- family: (required) AMD GPU family name, used for test selection and artifact fetching
- fetch-gfx-targets: (required) list of gfx targets to fetch split test artifacts for (e.g. ["gfx942", "gfx942:xnack+"])
- build_variants: (optional) list of build variants to test for this architecture (e.g. ["release", "asan"])
- bypass_tests_for_releases: (optional) if enabled, bypass tests for release builds (e.g. by skipping test steps in the workflow, or by not running tests on release builds in test scripts)
- sanity_check_only_for_family: (optional) if enabled, only run sanity check tests for this architecture
- run-full-tests-only: (optional) if enabled, only run full tests for this architecture
- nightly_check_only_for_family (optional): if enabled, only run CI nightly tests for this architecture
"""
# The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
amdgpu_family_info_matrix_presubmit = {
    "gfx94x": {
        "linux": {
            # TODO: Remove alternative weight once we get dedicated set of machines
            # As we are bringing back up mi325, we are using a dual-label configuration to distribute load
            "test-runs-on": "linux-gfx942-1gpu-ossci-rocm",
            "test-runs-on-alternate": "linux-gfx942-1gpu-ccs-ossci-rocm",
            "test-runs-on-alternate-weight": 0.35,  # 35% chance of using alternate
            # TODO(#3433): Remove sandbox label once ASAN tests are passing
            "test-runs-on-sandbox": "rocm-asan-mi325-sandbox",
            "test-runs-on-multi-gpu": "linux-gfx942-8gpu-ossci-rocm",
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "linux-gfx942-8gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            # Individual GPU target(s) on the test runner, for fetching split artifacts.
            # TODO(#3444): ASAN variants may need xnack suffix expansion (e.g. gfx942:xnack+).
            "fetch-gfx-targets": ["gfx942"],
            "build_variants": ["release", "asan", "tsan"],
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "linux-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "fetch-gfx-targets": [],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "fetch-gfx-targets": ["gfx1100", "gfx1101"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
        },
    },
    "gfx1151": {
        "linux": {
            "test-runs-on": "linux-gfx1151-gpu-rocm",
            "test-runs-on-kernel": {
                "oem": "linux-strix-halo-gpu-rocm-oem",
            },
            "family": "gfx1151",
            "fetch-gfx-targets": ["gfx1151"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx1151-gpu-rocm",
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "windows-gfx1151-gpu-rocm",
            "family": "gfx1151",
            "fetch-gfx-targets": ["gfx1151"],
            "build_variants": ["release"],
            # TODO(#3299): Re-enable quick tests once capacity is available for Windows gfx1151
            "nightly_check_only_for_family": True,
        },
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-gfx120X-gpu-rocm",
            "family": "gfx120X-all",
            "fetch-gfx-targets": ["gfx1200", "gfx1201"],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx120X-gpu-rocm",
            "family": "gfx120X-all",
            "fetch-gfx-targets": [],
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
    },
}

# The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
amdgpu_family_info_matrix_postsubmit = {
    "gfx950": {
        "linux": {
            "test-runs-on": "linux-mi355-1gpu-ossci-rocm",
            "family": "gfx950-dcgpu",
            "fetch-gfx-targets": ["gfx950"],
            "build_variants": ["release", "asan", "tsan"],
        }
    },
}

# The 'nightly' matrix runs on 'schedule' triggers.
amdgpu_family_info_matrix_nightly = {
    "gfx900": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx900",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx900",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    # gfx906/908/90a split into separate families - each has different instruction
    # support (e.g., fp8 variants, WMMA) so CK/MIOpen need to build/test individually.
    "gfx906": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx906",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`, to enable PyTorch builds
        "windows": {
            "test-runs-on": "",
            "family": "gfx906",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx908": {
        "linux": {
            # Disabled due to hardware availability
            "test-runs-on": "",
            "family": "gfx908",
            "fetch-gfx-targets": [],
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx908",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx90a": {
        "linux": {
            "test-runs-on": "linux-gfx90a-gpu-rocm",
            "family": "gfx90a",
            "fetch-gfx-targets": ["gfx90a"],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx90a",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "linux-gfx1030-gpu-rocm",
            "family": "gfx103X-dgpu",
            "fetch-gfx-targets": ["gfx1030"],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx1030-gpu-rocm",
            "family": "gfx103X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
    },
    "gfx1150": {
        "linux": {
            "test-runs-on": "linux-gfx1150-gpu-rocm",
            "family": "gfx1150",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1150",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx1152": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1152",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1152",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx1153": {
        "linux": {
            "test-runs-on": "linux-gfx1153-gpu-rocm",
            "family": "gfx1153",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "nightly_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1153",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
}


def get_all_families_for_trigger_types(trigger_types):
    """
    Returns a combined family matrix for the specified trigger types.
    trigger_types: list of strings, e.g. ['presubmit', 'postsubmit', 'nightly']
    """
    result = {}
    matrix_map = {
        "presubmit": amdgpu_family_info_matrix_presubmit,
        "postsubmit": amdgpu_family_info_matrix_postsubmit,
        "nightly": amdgpu_family_info_matrix_nightly,
    }

    for trigger_type in trigger_types:
        if trigger_type in matrix_map:
            for family_name, family_config in matrix_map[trigger_type].items():
                result[family_name] = family_config

    return result
