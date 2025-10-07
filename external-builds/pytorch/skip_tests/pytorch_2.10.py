skip_tests = {
    "common": {},
    "gfx942": {
        "autograd": [
            # fixed or just good with no caching?
            # "test_reentrant_parent_error_on_cpu_cuda",
            # "test_multi_grad_all_hooks",
            # "test_side_stream_backward_overlap",
            # "test_side_stream_backward_overlap",
            #
            # AttributeError: 'torch._C._autograd.SavedTensor' object has no attribute 'data'
            "test_get_data_and_hooks_from_raw_saved_variable ",  # new?
            # AssertionError: tensor(1., grad_fn=<AsStridedBackward0>) is not None -- weakref not working?
            "test_custom_function_saving_mutated_view_no_leak",  # new?
        ],
        "cuda": [
            # fixed or just good with no caching?
            # "test_cpp_memory_snapshot_pickle",
            # "test_memory_plots",
            # "test_memory_compile_regions",
            # "test_memory_plots_free_segment_stack",
            # "test_mempool_ctx_multithread",
            # "test_mempool_empty_cache_inactive",
            #
            # Error building extension 'dummy_allocator'
            "test_mempool_limited_memory_with_allocator",
            # OSError: libhiprtc.so: cannot open shared object file: No such file or directory
            # File "/home/tester/TheRock/.venv/lib/python3.12/site-packages/torch/cuda/_utils.py", line 57, in _get_hiprtc_library
            # lib = ctypes.CDLL("libhiprtc.so")
            "test_compile_kernel",
            "test_compile_kernel_advanced",
            "test_compile_kernel_as_custom_op",
            "test_compile_kernel_cuda_headers",
            "test_compile_kernel_custom_op_validation",
            "test_compile_kernel_dlpack",
            "test_compile_kernel_double_precision",
            "test_compile_kernel_large_shared_memory",
            "test_compile_kernel_template",
            #
            # working now?
            # # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            # "test_fp32_precision_with_float32_matmul_precision",
            # # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            # "test_fp32_precision_with_tf32",
            # # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            # "test_invalid_status_for_legacy_api",
        ],
        "nn": [
            # RuntimeError: miopenStatusUnknownError
            # MIOpen(HIP): Warning [BuildHip] In file included from /tmp/comgr-f75870/input/MIOpenDropoutHIP.cpp:32:
            # /tmp/comgr-f75870/include/miopen_rocrand.hpp:45:10: fatal error: 'rocrand/rocrand_xorwow.h' file not found
            # 45 | #include <rocrand/rocrand_xorwow.h>
            #     |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
            "test_cudnn_rnn_dropout_states_device",
        ],
        "torch": [
            "test_terminate_handler_on_crash",  # flaky !! hangs forever or works... can need up to 30 sec to pass
        ],
    },
}
