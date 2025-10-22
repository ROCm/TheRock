"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows.

* Each entry determines which families and test runners are available to use
* Each group determines which entries run by default on workflow triggers
"""

all_build_variants = {
    "linux": {
        "release": {
            "build_variant_label": "Release",
            "build_variant_suffix": "",
            # TODO: Enable linux-release-package once capacity and rccl link
            # issues are resolved. https://github.com/ROCm/TheRock/issues/1781
            # "build_variant_cmake_preset": "linux-release-package",
            "build_variant_cmake_preset": "",
        },
        "asan": {
            "build_variant_label": "ASAN",
            "build_variant_suffix": "asan",
            "build_variant_cmake_preset": "linux-release-asan",
            "expect_failure": True,
            "skip_presubmit_build": True,
        },
    },
    "windows": {
        "release": {
            "build_variant_label": "Release",
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
            "family": "gfx94X-dcgpu",
            "build_variants": ["release", "asan"],
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "linux-gfx110X-gpu-rocm",
            "family": "gfx110X-dgpu",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
            "sanity_check_only_for_family": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
            "bypass_tests_for_releases": True,
            "build_variants": ["release"],
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
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "expect_failure": False,
            "build_variants": ["release"],
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "expect_failure": False,
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": True,
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": False,
            "build_variants": ["release"],
            "expect_pytorch_failure": True,
        },
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "linux-rx6950-gpu-rocm",
            "family": "gfx103X-dgpu",
            "build_variants": ["release"],
            "expect_failure": False,
            "sanity_check_only_for_family": True,
            "expect_pytorch_failure": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "build_variants": ["release"],
            "expect_failure": False,
            "expect_pytorch_failure": True,
        },
    },
    "gfx1150": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1150",
            "build_variants": ["release"],
            "expect_failure": False,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx1150",
            "build_variants": ["release"],
            "expect_failure": False,
        },
    },
}

amdgpu_family_info_matrix_all = (
    amdgpu_family_info_matrix_presubmit
    | amdgpu_family_info_matrix_postsubmit
    | amdgpu_family_info_matrix_nightly
)
