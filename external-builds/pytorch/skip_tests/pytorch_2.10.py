skip_tests = {
    "common": {
        "autograd": [
            # AssertionError: Booleans mismatch: False is not True
            "test_warn_on_accumulate_grad_stream_mismatch_flag_cuda",
        ],
        "cuda": [
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_float32_matmul_precision",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_tf32",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_invalid_status_for_legacy_api",
            # passes on single run, crashes if run in a group
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_memory_compile_regions",
            # AssertionError: False is not true
            "test_memory_plots",
            # AssertionError: Booleans mismatch: False is not True
            "test_memory_plots_free_segment_stack",
            # FileNotFoundError: [Errno 2] No such file or directory: '/github/home/.cache//flamegraph.pl'
            "test_memory_snapshot",
            # AssertionError: String comparison failed: 'test_memory_snapshot' != 'foo'
            "test_memory_snapshot_script",
            # AssertionError: False is not true
            "test_memory_snapshot_with_cpp",
            # AssertionError: Scalars are not equal!
            "test_mempool_ctx_multithread",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_empty_cache_inactive",
            # RuntimeError: Error building extension 'dummy_allocator_v1'
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
            # torch._dynamo.exc.BackendCompilerFailed: backend='aot_eager' raised:
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_record_stream_on_shifted_view",
        ],
        "torch": [
            "test_index_add_correctness",
        ],
    },
    "gfx942": {
        "autograd": [
            # fixed or just good with no caching?
            # "test_reentrant_parent_error_on_cpu_cuda",
            # "test_multi_grad_all_hooks",
            # flaky, was good some itme
            "test_side_stream_backward_overlap",
            "test_side_stream_backward_overlap_cuda"
            #
            #  Test run says they are good????
            # # AttributeError: 'torch._C._autograd.SavedTensor' object has no attribute 'data'
            # "test_get_data_and_hooks_from_raw_saved_variable ",  # new?
            # # AssertionError: tensor(1., grad_fn=<AsStridedBackward0>) is not None -- weakref not working?
            # "test_custom_function_saving_mutated_view_no_leak",  # new?
            # #
            # # RuntimeError: Output 0 of IdOneOutputBackward is a view and is being modified inplace. This view was created inside a custom
            # # Function (or because an input was returned as-is) and the autograd logic to handle view+inplace would override the custom backward
            # # associated with the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by cloning the output
            # # of the custom Function.
            # "test_autograd_simple_views_python",
            "test_grad_dtype",
        ],
        "cuda": [
            # "test_cpp_memory_snapshot_pickle",
            #
            # what():  HIP error: operation not permitted when stream is capturing
            # Search for `hipErrorStreamCaptureUnsupported' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html for more information.
            # HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
            # For debugging consider passing AMD_SERIALIZE_KERNEL=3
            # Compile with `TORCH_USE_HIP_DSA` to enable device-side assertions.
            #
            # Exception raised from ~CUDAGraph at /__w/TheRock/TheRock/external-builds/pytorch/pytorch/aten/src/ATen/hip/HIPGraph.cpp:320 (most recent call first):
            # frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x80 (0x7f2316f1bdf0 in /home/tester/TheRock/.venv/lib/python3.12/site-packages/torch/lib/libc10.so)
            "test_graph_make_graphed_callables_parameterless_nograd_module_without_amp_allow_unused_input",
            "test_graph_make_graphed_callables_parameterless_nograd_module_without_amp_not_allow_unused_input",
            "test_graph_concurrent_replay ",
            #
            #
            # for whatever reason these are also flaky: if run standalone they pass?
            # AttributeError: Unknown attribute allow_bf16_reduced_precision_reduction_split_k
            "test_cublas_allow_bf16_reduced_precision_reduction_get_set",
            # AttributeError: Unknown attribute allow_fp16_reduced_precision_reduction_split_k
            "test_cublas_allow_fp16_reduced_precision_reduction_get_set",
            # AssertionError: Scalars are not close!
            "test_allocator_settings",
            # AttributeError: Unknown attribute allow_bf16_reduced_precision_reduction_split_k
            "test_cublas_allow_bf16_reduced_precision_reduction_get_set",
            # AttributeError: Unknown attribute allow_fp16_reduced_precision_reduction_split_k
            "test_cublas_allow_fp16_reduced_precision_reduction_get_set",
            "test_allocator_settings",
        ],
        "nn": [
            # Is now skipped.. on pytorch side
            # RuntimeError: miopenStatusUnknownError
            # MIOpen(HIP): Warning [BuildHip] In file included from /tmp/comgr-f75870/input/MIOpenDropoutHIP.cpp:32:
            # /tmp/comgr-f75870/include/miopen_rocrand.hpp:45:10: fatal error: 'rocrand/rocrand_xorwow.h' file not found
            # 45 | #include <rocrand/rocrand_xorwow.h>
            #     |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
            "test_cudnn_rnn_dropout_states_device",
        ],
        "torch": [
            "test_terminate_handler_on_crash",  # flaky !! hangs forever or works... can need up to 30 sec to pass
            "test_cpp_warnings_have_python_context_cuda",
            # torch._dynamo.exc.BackendCompilerFailed: backend='aot_eager' raised:
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_fx_memory_profiler_augmentation",
        ],
    },
    "gfx950": {
        "cuda": [
            "test_cpp_warnings_have_python_context_cuda",
        ],
        "binary_ufuncs": [
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other___rpow___cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other__refs_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig___rpow___cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig__refs_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig_pow_cuda_complex64",
        ],
        "torch": [
            # SEGMENTATION FAULT!!!!!
            # Kernel Name: _ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIbNS0_14func_wrapper_tIbZZZNS0_15and_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE10_clEvEUlbbE_EEjbLi4ELi4EEEEEvT1_
            # :0:rocdevice.cpp            :3603: 1544877169487 us:  Callback: Queue 0x7f599b800000 Aborting with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events. Code: 0x1008 Available Free mem : 17592186044276 MB
            # see pytorch_2.9.py for more details
            "test_masked_scatter_cuda_uint8 "
        ],
    },
}
