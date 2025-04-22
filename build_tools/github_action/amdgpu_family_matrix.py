"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows, indicating which families and test runners are available to use
"""

amdgpu_family_info_matrix = {
    "gfx94x": {
        "linux": {
            "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            "pytorch-target": "gfx942",
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
        "linux": {
            "test-runs-on": "",
            "family": "gfx1151",
        }
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        }
    },
}
