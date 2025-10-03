"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows.

* Each entry determines which families and test runners are available to use
* Each group determines which entries run by default on workflow triggers
"""

# The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
amdgpu_family_info_matrix_presubmit = {
    "gfx94x": {
        "linux": {
            "test-runs-on": "linux-mi325-1gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
            "bypass_tests_for_releases": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
            "bypass_tests_for_releases": True,
        },
    },
    "gfx115x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1151",
            "bypass_tests_for_releases": True,
        },
        "windows": {
            "test-runs-on": "windows-strix-halo-gpu-rocm",
            "family": "gfx1151",
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
        }
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "",  # removed due to machine issues, label is "linux-rx9070-gpu-rocm"
            "family": "gfx120X-all",
            "bypass_tests_for_releases": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx120X-all",
            "bypass_tests_for_releases": True,
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
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "expect_failure": False,
        },
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": False,
        },
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "expect_failure": True,
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "expect_failure": True,
        },
    },
}

amdgpu_family_info_matrix_all = (
    amdgpu_family_info_matrix_presubmit
    | amdgpu_family_info_matrix_postsubmit
    | amdgpu_family_info_matrix_nightly
)
