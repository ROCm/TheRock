"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows, indicating which families and test runners are available to use
"""

DEFAULT_LINUX_CONFIGURATIONS = ["gfx94x", "gfx110x"]
DEFAULT_WINDOWS_CONFIGURATIONS = ["gfx110x"]

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
            "test-runs-on": "linux-strix-halo-gpu-rocm",
            # Specifying gfx1151, since rocBLAS doesn't have gfx1150 (https://github.com/ROCm/rocBLAS/blob/d93e919f945e76ee89457c83d72aac4cb473f492/CMakeLists.txt#L84-L88)
            # Using gfx115X-igpu causes build failure because rocBLAS cannot find gfx1150
            "family": "gfx1151",
            "pytorch-target": "gfx1151",
        },
        "windows": {
            "test-runs-on": "windows-strix-halo-gpu-rocm",
            "family": "gfx1151",
            "pytorch-target": "gfx1151",
        },
    },
    "gfx120x": {
        "linux": {
            "test-runs-on": "linux-rx9070-gpu-rocm",
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        }
    },
}
