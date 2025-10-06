skip_tests = {
    "common": {},
    "gfx942": {
        "autograd": [
            "test_reentrant_parent_error_on_cpu_cuda",  # flaky? or new?
            "test_get_data_and_hooks_from_raw_saved_variable ",  # flaky? or new?
            "test_custom_function_saving_mutated_view_no_leak",  # flaky? or new?
            "test_multi_grad_all_hooks",  # flaky?
            "test_side_stream_backward_overlap",
        ],
        "cuda": [
            "test_cpp_memory_snapshot_pickle",
            "test_memory_compile_regions",
            # "test_memory_plots", --> not failing anymore?
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
        ],
        "nn": [
            # RuntimeError: miopenStatusUnknownError
            "test_side_stream_backward_overlap"
        ],
        "torch": [
            "test_terminate_handler_on_crash",  # hangs forever
        ],
    },
}
