# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "nn": [
            # MIOpen and native gradients diverge substantially on gfx942.
            "test_ctc_loss_cudnn_tensor_cuda_cuda",
        ],
    },
}
