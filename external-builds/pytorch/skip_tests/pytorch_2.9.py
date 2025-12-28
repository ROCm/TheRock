# PyTorch 2.9 specific skip tests
# Tests moved to generic.py have been removed to avoid duplication
# NOTE: Originally combined pytorch_2.7.py and pytorch_2.10.py to resolve OOM errors
skip_tests = {
    "common": {
        "autograd": [
            "test_side_stream_backward_overlap",
        ],
        "cuda": [
            # for whatever reason these are also flaky: if run standalone they pass?
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_float32_matmul_precision",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_tf32",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_invalid_status_for_legacy_api",
        ],
    },
    "gfx1151": {
        # Consumer GPU-specific failures for PyTorch 2.9
        "cuda": [
            # AttributeError: Unknown attribute allow_bf16_reduced_precision_reduction_split_k
            "test_cublas_allow_bf16_reduced_precision_reduction_get_set",
            # AttributeError: Unknown attribute allow_fp16_reduced_precision_reduction_split_k
            "test_cublas_allow_fp16_reduced_precision_reduction_get_set",
            # OSError: libhiprtc.so: cannot open shared object file: No such file or directory
            "test_compile_kernel",
            "test_compile_kernel_advanced",
            "test_compile_kernel_as_custom_op",
            "test_compile_kernel_cuda_headers",
            "test_compile_kernel_custom_op_validation",
            "test_compile_kernel_dlpack",
            "test_compile_kernel_double_precision",
            "test_compile_kernel_large_shared_memory",
            "test_compile_kernel_template",
        ],
        "windows": {
            # Windows + gfx1151 + PyTorch 2.9 specific failures
            "cuda": [
                # This test uses subprocess.run, so it hangs on Windows
                "test_pinned_memory_use_background_threads",
                # Windows fatal exception: access violation in serialization
                "test_serialization_array_with_empty",
                "test_serialization_array_with_storage",
            ],
        },
    },
    "gfx1152": {
        # Same failures as gfx1151 - consumer GPU with PyTorch 2.9
        "cuda": [
            "test_cublas_allow_bf16_reduced_precision_reduction_get_set",
            "test_cublas_allow_fp16_reduced_precision_reduction_get_set",
            "test_compile_kernel",
            "test_compile_kernel_advanced",
            "test_compile_kernel_as_custom_op",
            "test_compile_kernel_cuda_headers",
            "test_compile_kernel_custom_op_validation",
            "test_compile_kernel_dlpack",
            "test_compile_kernel_double_precision",
            "test_compile_kernel_large_shared_memory",
            "test_compile_kernel_template",
        ],
        "windows": {
            "cuda": [
                "test_pinned_memory_use_background_threads",
                "test_serialization_array_with_empty",
                "test_serialization_array_with_storage",
            ],
        },
    },
    "gfx1153": {
        # Same failures as gfx1151/gfx1152 - consumer GPU with PyTorch 2.9
        "cuda": [
            "test_cublas_allow_bf16_reduced_precision_reduction_get_set",
            "test_cublas_allow_fp16_reduced_precision_reduction_get_set",
            "test_compile_kernel",
            "test_compile_kernel_advanced",
            "test_compile_kernel_as_custom_op",
            "test_compile_kernel_cuda_headers",
            "test_compile_kernel_custom_op_validation",
            "test_compile_kernel_dlpack",
            "test_compile_kernel_double_precision",
            "test_compile_kernel_large_shared_memory",
            "test_compile_kernel_template",
        ],
        "windows": {
            "cuda": [
                "test_pinned_memory_use_background_threads",
                "test_serialization_array_with_empty",
                "test_serialization_array_with_storage",
            ],
        },
    },
    "gfx110X-all": {
        # Datacenter GPU-specific failures for PyTorch 2.9
        "nn": [
            # Segmentation fault (core dump) on Linux
            # Exit code 134 (SIGABRT) during test execution
            # https://github.com/ROCm/TheRock/actions/runs/20554976437/job/59039577384
            # Using broad pattern to catch all convolution tests as precaution
            "test_Conv",  # Matches test_Conv1d, test_Conv2d, test_Conv3d, etc.
        ],
    },
    "gfx942": {
        "autograd": [
            # fixed or just good with no caching?
            # "test_reentrant_parent_error_on_cpu_cuda",
            # "test_multi_grad_all_hooks",
            # "test_side_stream_backward_overlap",
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
            # Skip entire TestAutogradMultipleDispatchCUDA class - all tests in this class fail
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
            "test_record_stream_on_shifted_view",
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
        ],
    },
    "gfx950": {
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
    },
}
