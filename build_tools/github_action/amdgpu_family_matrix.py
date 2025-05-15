"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows, indicating which families and test runners are available to use
"""

amdgpu_family_info_matrix = {
    "gfx90x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "pytorch-target": "gfx90a",
        }
    },
    "gfx94x": {
        "linux": {
            "test-runs-on": "linux-mi300-1gpu-ossci-rocm-test",
            "family": "gfx94X-dcgpu",
            "pytorch-target": "gfx942",
        }
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "pytorch-target": "gfx1010",
        }
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "pytorch-target": "gfx1030",
        }
    },
    "gfx110x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
            "pytorch-target": "gfx1100",
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx110X-dgpu",
        },
    },
    "gfx115x": {
        "linux": {"test-runs-on": "", "family": "gfx1151", "pytorch-target": "gfx1151"}
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-rx9070-gpu-rocm",
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        }
    },
}
