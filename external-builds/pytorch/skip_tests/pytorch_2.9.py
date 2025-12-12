# NOTE: not tested. just combining pytorch_2.7.py and pytorch_2.10.py to see if that resolves the OOM errors
skip_tests = {
    "common": {
        "autograd": [
            "test_side_stream_backward_overlap",
        ],
        "cuda": [
            # Explicitly deselected since giving segfault
            "test_unused_output_device_cuda",  # this test does not exist in nightly anymore
            "test_pinned_memory_empty_cache",
            "test_float32_matmul_precision_get_set",
            # AssertionError: Tensor-likes are not close!
            # Mismatched elements: 1 / 327680 (0.0%)
            # Greatest absolute difference: 0.03125 at index (3, 114, 184) (up to 0.01 allowed)
            # Greatest relative difference: 0.01495361328125 at index (3, 114, 184) (up to 0.01 allowed)
            "test_index_add_correctness",
            "test_graph_concurrent_replay",
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
            # for whatever reason these are also flaky: if run standalone they pass?
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_float32_matmul_precision",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_fp32_precision_with_tf32",
            # AttributeError: module 'torch.backends.cudnn.rnn' has no attribute 'fp32_precision'
            "test_invalid_status_for_legacy_api",
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
        "torch": [
            # SEGMENTATION FAULT!!!!!
            # external-builds/pytorch/pytorch/test/test_torch.py::TestTorchDeviceTypeCUDA::test_masked_scatter_cuda_uint8 Kernel Name: _ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIbNS0_14func_wrapper_tIbZZZNS0_15and_kernel_cudaERNS_14TensorIteratorEENKUlvE_clEvENKUlvE10_clEvEUlbbE_EEjbLi4ELi4EEEEEvT1_
            # :0:rocdevice.cpp            :3603: 1545695326084 us:  Callback: Queue 0x7f3dd8000000 Aborting with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events. Code: 0x1008 Available Free mem : 17592186044276 MB
            # Thread 0x00007f4b4fd88b80 (most recent call first):
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1087 in _compare_regular_values_close
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_comparison.py", line 905 in _compare_values
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_comparison.py", line 747 in compare
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1298 in not_close_error_metas
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py", line 4248 in assertEqual
            # File "/__w/TheRock/TheRock/external-builds/pytorch/pytorch/test/test_torch.py", line 3683 in test_masked_scatter
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_device_type.py", line 1473 in only_fn
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_device_type.py", line 428 in instantiated_test
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py", line 3332 in wrapper
            # File "/__w/_tool/Python/3.11.14/x64/lib/python3.11/unittest/case.py", line 579 in _callTestMethod
            # File "/__w/_tool/Python/3.11.14/x64/lib/python3.11/unittest/case.py", line 623 in run
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py", line 3484 in _run_custom
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_utils.py", line 3514 in run
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/torch/testing/_internal/common_device_type.py", line 519 in run
            # File "/__w/_tool/Python/3.11.14/x64/lib/python3.11/unittest/case.py", line 678 in __call__
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/_pytest/unittest.py", line 351 in runtest
            # File "/__w/TheRock/TheRock/.venv/lib/python3.11/site-packages/_pytest/runner.py", line 174 in pytest_runtest_call
            "test_masked_scatter_cuda_uint8 "
        ],
    },
}
