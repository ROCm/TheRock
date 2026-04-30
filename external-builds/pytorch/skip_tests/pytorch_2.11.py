# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "autograd": [
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_logging",
            "test_checkpoint_compile_no_recompile",
            "test_checkpoint_detects_non_determinism",
            "test_checkpoint_device_context_fn",
            "test_checkpoint_graph_execution_group",
            "test_checkpoint_valid_reset_on_error",
            "test_checkpointing_non_reentrant_autocast_cpu",
            "test_checkpointing_non_reentrant_autocast_gpu",
            "test_checkpointing_without_reentrant_arbitrary_input_output",
            "test_checkpointing_without_reentrant_correct_grad",
            "test_checkpointing_without_reentrant_custom_function_works",
            "test_checkpointing_without_reentrant_dataparallel",
            "test_checkpointing_without_reentrant_detached_tensor_use_reentrant_True",
            "test_checkpointing_without_reentrant_parameter_used_in_an_out",
            "test_checkpointing_without_reentrant_saved_object_identity",
            "test_checkpointing_without_reentrant_with_context_fn",
            "test_clear_saved_tensors_on_access",
            "test_clear_saved_tensors_on_access_double_access_error",
            "test_create_graph_and_full_backward_hook_cycle",
            "test_current_graph_task_execution_order",
            "test_custom_autograd_ac_early_stop",
            "test_custom_autograd_no_early_free",
            "test_custom_autograd_repeated_grad_grad",
        ],
        "cuda": [
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
            # new for pytorch 2.11
            # RuntimeError: Error building extension 'dummy_allocator_v3'
            "test_tensor_delete_after_allocator_delete",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_deleted_mempool_not_used_on_oom",
            # Same hipblas.h compilation error as test_mempool_with_allocator.
            # See https://github.com/pytorch/pytorch/pull/173330
            "test_mempool_expandable",
            # ModuleNotFoundError: No module named 'torchvision'
            "test_resnet",
            # RuntimeError: miopenStatusUnknownError
            "test_graph_cudnn_dropout",
            # Fatal Python error: Segmentation fault - https://github.com/ROCm/TheRock/issues/4745
            "test_snapshot_include_traces",
        ],
        "nn": [
            # new in 2.11
            # AssertionError: Scalars are not close!
            "test_CTCLoss_cudnn_cuda",
        ],
        "torch": [
            "test_cpp_warnings_have_python_context_cuda",
        ],
    },
    "gfx1151": {
        "nn": [
            # AssertionError: Tensor-likes are not close! - https://github.com/ROCm/TheRock/issues/4744
            "test_Embedding_discontiguous_cuda",
        ],
    },
    # "gfx120": {
    #     "unary_ufuncs": [
    #         # this failed only once. maybe python version dependent? probably the run was python 3.13
    #         # AssertionError: Tensor-likes are not close!
    #         "test_batch_vs_slicing_polygamma_polygamma_n_2_cuda_float16",
    #     ],
    # },
    # "windows": {
    #     empty for the moment
    # },
}
