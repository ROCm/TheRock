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
}

# The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
amdgpu_family_info_matrix_postsubmit = {
    "gfx950": {
        "linux": {
            "test-runs-on": "linux-mi355-1gpu-ossci-rocm",
            "family": "gfx950-dcgpu",
        }
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
        }
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": False,
        }
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "expect_failure": True,
        }
    },
}


def merge_matrices(matrices):
    merged = {}
    for matrix in matrices:
        for key in matrix:
            info_for_key = matrix[key]
            if not key in merged:
                merged[key] = info_for_key
            else:
                # The key, e.g. "gfx115x", already exists.
                # This probably means that presubmit had one platform and
                # postsubmit had a different platform for that key.
                for platform in info_for_key:
                    if platform in merged[key]:
                        raise LookupError(
                            f"Duplicate platform {platform} for key {key}"
                        )
                    merged[key][platform] = info_for_key[platform]
    return merged


amdgpu_family_info_matrix_all = merge_matrices(
    [
        amdgpu_family_info_matrix_presubmit,
        amdgpu_family_info_matrix_postsubmit,
        amdgpu_family_info_matrix_nightly,
    ]
)
