# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch 2.13 wheels. These are already tracked in the
# other version skip files (pytorch_2.10.py - pytorch_2.12.py) and/or generic.py,
# but those exclusions are not picked up for the 2.13 version, so they are
# mirrored here. See
# https://github.com/ROCm/TheRock/issues/5596

skip_tests = {
    "common": {
        "cuda": [
            # TestCuda - conflicts with how our test script and runners are
            # configured.
            "test_hip_device_count",
            # TestCudaAllocator - passes on single run, crashes if run in a
            # group. TypeError: 'CustomDecompTable' object is not a mapping
            "test_memory_compile_regions",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator'. The hipblas.h include error persists in the
            # ROCm SDK environment:
            #   fatal error: 'hipblas/hipblas.h' file not found
            "test_mempool_empty_cache_inactive",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator_v1' (same hipblas.h include error)
            "test_mempool_limited_memory_with_allocator",
        ],
        "nn": [
            # TestNNDeviceTypeCUDA - AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
            # TestNNDeviceTypeCUDA - cudnn CTC loss numerical mismatch
            "test_ctc_loss_cudnn_tensor_cuda_cuda",
            # TestNNDeviceTypeCUDA - per-call dropout randomness mismatch
            "test_LSTM_dropout_per_call_randomness_dropout_p_0_5_training_True_cuda",
            # TestNNDeviceTypeCUDA - upsampling launch failure on gfx950
            # Separately tracked in https://github.com/ROCm/TheRock/issues/5270
            "test_upsamplingNearest2d_launch_rocm_cuda",
        ],
    },
}
