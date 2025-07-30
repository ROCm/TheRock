"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows, indicating which families and test runners are available to use
"""

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
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
        },
    },
}

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
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx120X-all",
        },
    },
}

amdgpu_family_matrix_xfail = {
    "gfx90x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "expect_failure": True,
        }
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "expect_failure": True,
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
