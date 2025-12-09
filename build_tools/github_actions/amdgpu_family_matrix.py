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
            "test-runs-on": "linux-strix-halo-gpu-rocm",
            "family": "gfx1151",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "windows-strix-halo-gpu-rocm",
            "family": "gfx1151",
            "build_variants": ["release"],
        },
    },
}

# The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
amdgpu_family_info_matrix_postsubmit = {
    "gfx950": {
        "linux": {
            # Networking issue: https://github.com/ROCm/TheRock/issues/1660
            # Label is "linux-mi355-1gpu-ossci-rocm"
            "test-runs-on": "",
            "family": "gfx950-dcgpu",
            "build_variants": ["release", "asan"],
        }
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-rx9070-gpu-rocm",
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
            "test-runs-on": "linux-rx6950-gpu-rocm",
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
        },
    },
    "gfx1150": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1150",
            "build_variants": ["release"],
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
            "expect_failure": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1152",
            "expect_failure": True,
            "build_variants": ["release"],
        },
    },
    "gfx1153": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1153",
            "expect_failure": True,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1153",
            "expect_failure": True,
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
