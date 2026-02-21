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
            "test-runs-on": "linux-mi325-1gpu-ossci-rocm",
            # TODO(#3433): Remove sandbox label once ASAN tests are passing
            "test-runs-on-sandbox": "linux-mi325-8gpu-ossci-rocm-sandbox",
            "test-runs-on-multi-gpu": "linux-mi325-8gpu-ossci-rocm",
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "linux-mi325-8gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            # Individual GPU target(s) on the test runner, for fetching split artifacts.
            # TODO(#3444): ASAN variants may need xnack suffix expansion (e.g. gfx942:xnack+).
            "fetch-gfx-targets": ["gfx942"],
            "build_variants": ["release", "asan", "tsan"],
        }
    },
    "gfx110x": {
        "linux": {
            # TODO(#3298): Re-enable machine once HSA_STATUS_ERROR_OUT_OF_RESOURCES issues are resolved
            # Label is linux-gfx110X-gpu-rocm, fetch-gfx-targets should be ["gfx1100"]
            "test-runs-on": "",
            "family": "gfx110X-all",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "windows-gfx110X-gpu-rocm",
            "family": "gfx110X-all",
            "fetch-gfx-targets": ["gfx1100"],
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
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "windows-gfx1151-gpu-rocm",
            # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
            "benchmark-runs-on": "windows-gfx1151-gpu-rocm",
            "family": "gfx1151",
            "fetch-gfx-targets": ["gfx1151"],
            "build_variants": ["release"],
        },
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-gfx120X-gpu-rocm",
            "family": "gfx120X-all",
            "fetch-gfx-targets": ["gfx1200", "gfx1201"],
            "build_variants": ["release"],
        },
        "windows": {
            # TODO(#2962): Re-enable machine once sanity checks work with this architecture
            # Label is windows-gfx120X-gpu-rocm, fetch-gfx-targets should be ["gfx1200", "gfx1201"]
            "test-runs-on": "",
            "family": "gfx120X-all",
            "fetch-gfx-targets": [],
            "bypass_tests_for_releases": True,
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
            "fetch-gfx-targets": ["gfx950"],
            "build_variants": ["release", "asan", "tsan"],
        }
    },
}

# The 'nightly' matrix runs on 'schedule' triggers.
amdgpu_family_info_matrix_nightly = {
    "gfx90x": {
        "linux": {
            "test-runs-on": "linux-gfx90X-gpu-rocm",
            "family": "gfx90X-dcgpu",
            "fetch-gfx-targets": ["gfx90a"],
            "build_variants": ["release"],
        },
        # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`, to enable PyTorch builds
        "windows": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx101x": {
        # TODO(#1926): Resolve bgemm kernel hip file generation error to enable PyTorch builds
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
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
        },
        "windows": {
            # TODO(#3200): Re-enable machine once it is stable
            # Label is "windows-gfx1030-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
        },
    },
    "gfx1150": {
        "linux": {
            # TODO(#3199): Re-enable machine once it is stable
            # Label is "linux-gfx1150-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx1150",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
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
            # TODO(#2682): Re-enable machine once it is stable
            # Label is "linux-gfx1153-gpu-rocm"
            "test-runs-on": "",
            "family": "gfx1153",
            "fetch-gfx-targets": [],
            "build_variants": ["release"],
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
