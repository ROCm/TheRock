amdgpu_family_info_matrix = {
    "gfx94X": {
        "linux": {
            "test-runs-on": "linux-mi300-1gpu-ossci-rocm",
            "family": "gfx94X-dcgpu",
            "pytorch-target": "gfx942",
        }
    },
    "gfx110X": {
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
    "gfx115X": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx1151",
        }
    },
    "gfx120X": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx120X-all",
            "pytorch-target": "gfx1201",
        }
    },
}
