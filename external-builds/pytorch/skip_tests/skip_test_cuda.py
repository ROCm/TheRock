skip_tests = {
    "always": [
        "test_device_count_not_cached_pre_init",
        "test_host_memory_stats",
        # In file included from /home/tester/.cache/torch_extensions/py312_cpu/dummy_allocator/main_hip.cpp:5:
        # /home/tester/TheRock/.venv/lib/python3.12/site-packages/torch/include/ATen/hip/Exceptions.h:4:10: fatal error: hipblas/hipblas.h: No such file or directory
        #     4 | #include <hipblas/hipblas.h>
        #     |          ^~~~~~~~~~~~~~~~~~~
        # compilation terminated.
        "test_mempool_with_allocator",
    ],
    "pytorch_version": {
        "2.10": [
            "test_cpp_memory_snapshot_pickle",
            "test_memory_compile_regions",
            "test_memory_plots",
            "test_memory_plots_free_segment_stack",
            "test_mempool_ctx_multithread",
            "test_mempool_empty_cache_inactive",
            "test_mempool_limited_memory_with_allocator",
            "test_compile_kernel",
            "test_compile_kernel_advanced",
            "test_compile_kernel_as_custom_op",
            "test_compile_kernel_cuda_headers",
            "test_compile_kernel_custom_op_validation",
            "test_compile_kernel_dlpack",
            "test_compile_kernel_double_precision",
            "test_compile_kernel_large_shared_memory",
            "test_compile_kernel_template",
            "test_cudnn_rnn_dropout_states_device",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_float32_matmul_precision",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_tf32",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_invalid_status_for_legacy_api",
        ]
    },
    "amdgpu_family": {
        "gfx950": [
            "test_preferred_blas_library_settings",
            "test_autocast_torch_bf16",
            "test_autocast_torch_fp16",
        ]
    },
}

# maybe failing
# general
# "test_hip_device_count"
# "test_nvtx"

# gfx942
# TestCuda under test_cuda.py, failing on gfx942 (#1143) --> not on sharkmi300x-4
#    "test_float32_matmul_precision_get_set ",
# TestCude under test_cuda.py, failing on gfx942 (#1151) --> not on sharkmi300x-4
#    "test_graph_concurrent_replay",

# Explicitly deselected since givind segfault
#    "test_unused_output_device_cuda",
#    "test_pinned_memory_empty_cache",
