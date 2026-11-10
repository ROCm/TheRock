# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Failures from the ROCm 10.1.0rc3 / release/2.15 full suite on gfx94X, where
# rocm[devel] is installed. The per-family hipblas.h mempool failure is not
# listed: that test passes once devel and `rocm-sdk init` are present.

skip_tests = {
    "common": {
        "utils": [
            # Standalone JIT load path gained a _v2 suffix the assertion
            # still compares against the old name.
            "test_load_standalone",
        ],
        "multiprocessing": [
            # torch_shm_manager does not answer; pytest -k matches test_fs_*.
            "test_fs",
        ],
        "dynamo": [
            # Log text no longer matches the expected warning/info sequence.
            "test_logs_out",
        ],
        "serialization": [
            # Asserts a CI debug flag this runner does not set.
            "test_debug_set_in_ci",
        ],
        "distributed": [
            # Child rank aborts: HIP_VISIBLE_DEVICES conflicts with
            # CUDA_VISIBLE_DEVICES on the single-GPU full-suite runner.
            "test_watchdog_uses_group_device",
            "test_eager_init_with_per_rank_visible_device",
        ],
    },
    "gfx1151": {
        "torch": [
            # Tries to allocate 56 GiB for int64 upsample3d and OOMs.
            "test_int64_upsample3d_cuda_bfloat16",
        ],
    },
}
