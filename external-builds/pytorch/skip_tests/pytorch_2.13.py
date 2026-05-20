# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "gfx94": {
        "distributed": [
            # TestFullyShardMixedPrecisionTraining - process timeout (300s)
            "(TestFullyShardMixedPrecisionTraining and test_grad_acc_with_reduce_dtype)",
            # TestFullyShard1DTrainingCompose - process timeout (300s)
            "(TestFullyShard1DTrainingCompose and test_double_forward_with_nested_fsdp_and_checkpoint)",
            # TestJoin - rank exits with error code 10 (scalar mismatch)
            "(TestJoin and test_single_joinable)",
            # TestDataParallel - Fatal Python error: segmentation fault
            "(TestDataParallel and test_strided_grad_layout)",
        ],
    },
}
