# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known nightly transformer failures found by the AIPYTORCH-1609
# cross-version validation with PyTorch 2.15 wheels.
skip_tests = {
    "windows": {
        "transformers": [
            # Fails fake-tensor execution on each validated Windows family.
            "test_transformer_encoder_layer_fwd_fake_cuda",
        ],
    },
}
