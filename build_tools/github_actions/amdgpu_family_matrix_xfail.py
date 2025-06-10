"""
This AMD GPU Family Matrix is used to build and test for families that expect failure
"""

amdgpu_family_matrix_xfail = {
    "gfx90x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx90X-dcgpu",
            "pytorch-target": "gfx90a",
            "expect_failure": True,
        }
    },
    "gfx101x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx101X-dgpu",
            "pytorch-target": "gfx1010",
            "expect_failure": True,
        }
    },
    "gfx103x": {
        "linux": {
            "test-runs-on": "",
            "family": "gfx103X-dgpu",
            "pytorch-target": "gfx1030",
            "expect_failure": True,
        }
    },
}
