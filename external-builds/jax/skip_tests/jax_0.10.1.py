# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests skipped on JAX 0.10.1. See README.md for the entry format."""

skip_tests = {
    "gfx94": {
        # MIOpen selects convolution kernels nondeterministically here, so the
        # float32 convolution tests flake. The other keywords are convolutions
        # under another name: pooling, Toeplitz and Hankel build on the same
        # kernels, and polymul convolves fixed inputs and returned a different
        # wrong answer in each of the five jobs it has failed. convert and
        # conversion only match "conv" by accident and are healthy.
        #
        # Remove once these pass across repeated runs.
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
