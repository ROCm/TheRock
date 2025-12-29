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

from github_actions_utils import str2bool

import json
import os

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
            "expect_failure": True,
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

# The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
amdgpu_family_info_matrix_presubmit = {
    "gfx94x": {
        "linux": {
            "test-runs-on": "linux-mi325-1gpu-ossci-rocm-frac",
            # TODO: Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "linux-mi325-1gpu-ossci-rocm-frac",
            "family": "gfx94X-dcgpu",
            "build_variants": ["release", "asan"],
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "linux-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
    },
    "gfx1151": {
        "linux": {
            "test-runs-on": "linux-gfx1151-gpu-rocm",
            "family": "gfx1151",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-gfx1151-gpu-rocm",
            "family": "gfx1151",
            "build_variants": ["release"],
        },
    },
}

# The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
amdgpu_family_info_matrix_postsubmit = {
    "gfx950": {
        "linux": {
            "test-runs-on": "linux-mi355-1gpu-ossci-rocm",
            "family": "gfx950-dcgpu",
            "build_variants": ["release", "asan"],
        }
    },
    "gfx120x": {
        "linux": {
            # TODO(#2683): Re-enable machine once it is stable
            # Label is "linux-gfx120X-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx120X-all",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx120X-all",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
        },
    },
}

# The 'nightly' matrix runs on 'schedule' triggers.
amdgpu_family_info_matrix_nightly = {
    "gfx90x": {
        "linux": {
            # label is linux-gfx90X-gpu-rocm
            # Disabled due to inconsistent up-time
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "sanity_check_only_for_family": True,
            "build_variants": ["release"],
        },
        # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`, to enable PyTorch builds
        "windows": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx101x": {
        # TODO(#1926): Resolve bgemm kernel hip file generation error, to enable PyTorch builds
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": True,
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
        # TODO(#1925): Enable arch for aotriton to enable PyTorch builds
        "windows": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "linux-gfx1030-gpu-rocm",
            "family": "gfx103X-dgpu",
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        # TODO(#1925): Enable arch for aotriton to enable PyTorch builds
        "windows": {
            "test-runs-on": "windows-gfx1030-gpu-rocm",
            "family": "gfx103X-dgpu",
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
            "sanity_check_only_for_family": True,
        },
    },
    "gfx1150": {
        "linux": {
            # TODO(#2614): Re-enable machine once it is stable
            # Label is "linux-gfx1150-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx1150",
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1150",
            "build_variants": ["release"],
        },
    },
    "gfx1152": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1152",
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1152",
            "build_variants": ["release"],
        },
    },
    "gfx1153": {
        "linux": {
            # TODO(#2682): Re-enable machine once it is stable
            # Label is "linux-gfx1153-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx1153",
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1153",
            "build_variants": ["release"],
        },
    },
}


def load_test_runner_from_gh_variables():
    """
    As test runner names are frequently updated, we are pulling the runner label data from the ROCm organization variable called "ROCM_THEROCK_TEST_RUNNERS"

    For more info, go to 'docs/development/test_runner_info.md'
    """
    test_runner_json_str = os.getenv("ROCM_THEROCK_TEST_RUNNERS", "{}")
    test_runner_dict = json.loads(test_runner_json_str)
    for key in test_runner_dict.keys():
        for platform in test_runner_dict[key].keys():
            # Checking in presubmit dictionary
            if (
                key in amdgpu_family_info_matrix_presubmit
                and platform in amdgpu_family_info_matrix_presubmit[key]
            ):
                amdgpu_family_info_matrix_presubmit[key][platform]["test-runs-on"] = (
                    test_runner_dict[key][platform]
                )
            # Checking in postsubmit dictionary
            if (
                key in amdgpu_family_info_matrix_postsubmit
                and platform in amdgpu_family_info_matrix_postsubmit[key]
            ):
                amdgpu_family_info_matrix_postsubmit[key][platform]["test-runs-on"] = (
                    test_runner_dict[key][platform]
                )
            # Checking in nightly dictionary
            if (
                key in amdgpu_family_info_matrix_nightly
                and platform in amdgpu_family_info_matrix_nightly[key]
            ):
                amdgpu_family_info_matrix_nightly[key][platform]["test-runs-on"] = (
                    test_runner_dict[key][platform]
                )


def get_all_families_for_trigger_types(trigger_types):
    """
    Returns a combined family matrix for the specified trigger types.
    trigger_types: list of strings, e.g. ['presubmit', 'postsubmit', 'nightly']
    """
    # Load in test runners from ROCm organization variable "ROCM_THEROCK_TEST_RUNNERS"
    load_test_runners_from_var = str2bool(
        os.getenv("LOAD_TEST_RUNNERS_FROM_VAR", "true")
    )
    if load_test_runners_from_var:
        load_test_runner_from_gh_variables()
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
