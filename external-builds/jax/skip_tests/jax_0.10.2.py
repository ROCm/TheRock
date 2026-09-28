# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests skipped on JAX 0.10.2. See README.md for the entry format."""

skip_tests = {
    "gfx94": {
        # Same nondeterministic MIOpen kernel selection as 0.10.1; see that file.
        "keywords": [
            {"deny": "conv", "unless": ["convert", "conversion"]},
            {"deny": "sumpool"},
            {"deny": "minmaxpool"},
            {"deny": "toeplitz"},
            {"deny": "hankel"},
            {"deny": "polymul"},
        ],
    },
}
