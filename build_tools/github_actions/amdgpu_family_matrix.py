"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows, indicating which families and test runners are available to use
"""

# DEFAULT_LINUX_CONFIGURATIONS = ["gfx94x", "gfx110x"]
# DEFAULT_WINDOWS_CONFIGURATIONS = ["gfx110x"]
DEFAULT_LINUX_CONFIGURATIONS = ["gfx94x", "gfx110x"]
DEFAULT_WINDOWS_CONFIGURATIONS = []

amdgpu_family_info_matrix_presubmit = {
    "gfx94x": {
        "linux": {
            "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            "pytorch-target": "gfx942",
        }
    },
    "gfx950": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx950-dcgpu",
            "pytorch-target": "gfx950",
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
            "pytorch-target": "gfx1100",
        },
    },
    "gfx115x": {
        "linux": {"test-runs-on": "", "family": "gfx1151", "pytorch-target": "gfx1151"},
        "windows": {
            "test-runs-on": "windows-strix-halo-gpu-rocm",
            "family": "gfx1151",
            "pytorch-target": "gfx1151",
        },
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "",  # removed due to machine issues, label is "linux-rx9070-gpu-rocm"
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        },
        "windows": {
            "test-runs-on": "",
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        },
    },
}
