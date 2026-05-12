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
            # AssertionError: Tensor-likes are not close! - https://github.com/ROCm/TheRock/issues/4744
            # Failed on gfx1151 and gfx942 (only with python 3.13)
            "test_Embedding_discontiguous_cuda",
        ],
        "torch": [
            "test_cpp_warnings_have_python_context_cuda",
        ],
        "distributed": [
            # Error while creating shared memory segment /dev/shm/nccl-VPyhzw (size 21823872), error: No space left on device (28)
            "test_3d_mlp_with_nd_mesh",
            # AssertionError: False is not true : cuda:0 used 2615148544.0 bytes after collective, 70% more than the status before (1495269376.0 bytes). Extra CUDA context may have been created.
            "test_extra_cuda_context",
            # timeouts:
            "(TestClipGradNormWorldSize2 and test_clip_grad_norm_1d)",
            "(TestClipGradNormWorldSize4 and test_clip_grad_norm_2d)",
            "(TestFullyShardAllGatherExtensionsMultiProcess and test_all_gather_extensions_train_parity)",
            "(TestFullyShardFrozen and test_multi_forward_mixed_requires_grad)",
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_across_groups)",
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_per_group)",
            "(TestFullyShardGradientScaler and test_gradient_scaler)",
            "(TestFullyShardIgnoreParams and test_ddp_A_fsdp_B_ddp_C)",
            "(TestFullyShardMemory and test_fully_shard_training_memory)",
            "(TestFullyShardOverlap and test_fully_shard_training_overlap)",
            "(TestFullyShardCommunication and test_set_reduce_scatter_divide_factor)",
            "(TestFullyShardBackwardPrefetch and test_backward_misprefetch)",
            "(TestFullyShardBackwardPrefetch and test_fully_shard_backward_prefetch)",
            "(TestFullyShardBackwardPrefetch and test_fully_shard_multi_module_backward_prefetch)",
            "(TestFullyShardBackwardPrefetch and test_set_modules_to_backward_prefetch)",
            "(TestFullyShardBackwardPrefetch and test_set_modules_to_backward_prefetch_inside_ac)",
            "(TestFullyShardBackwardPrefetch and test_set_modules_to_forward_prefetch)",
            "(TestFullyShardBackwardPrefetch and test_unshard_async)",
            "(TestFullyShardBackwardPrefetch and test_fully_shard_force_sum_reduce_scatter)",
            "(TestFullyShardBackwardPrefetch and test_delayed_optim_step_offload_true_none_cuda)",
            "(TestStateDict and test_fsdp)",
            "(TestTrackerFullyShard1DTrainingCompose and test_tracker_with_activation_checkpointing)",
            "(TestBackwardPrefetch and test_backward_prefetch)",
            "(TestHooksCUDA and test_pre_backward_hook_registration_after_state_dict_cuda)",
            "(TestHooksCUDA and test_pre_backward_hook_registration_cuda_first_True_cuda)",
            "(TestParityWithDDPCUDA and test_delayed_optim_step_offload_false_none_cuda)",
            "(TestParityWithDDPCUDA and test_nested_always_wrap_model_offload_true_none_cuda)",
            "(TestParityWithDDPCUDA and test_nested_wrapped_model_single_iteration_mixed_precision_offload_false_none_cuda)", or
            "(TestNoGradCUDA and test_transformer_no_grad_mixed_precision_True_cuda)",
            "test_nested_wrapped_model_single_iteration_mixed_precision_offload_true_none_cuda",
            "test_transformer_offload_false_none_cuda",
            "test_nested_fully_shard_backend_aot_eager",
            "test_nested_fully_shard_backend_aot_eager_decomp_partition",
            "test_nested_fully_shard_backend_inductor_fullgraph_True",
            "test_nested_fully_shard_backend_inductor_fullgraph_True_graph_partition",
            "test_transformer_backend_aot_eager",
            "test_transformer_backend_aot_eager_decomp_partition",
            "test_compute_dtype",
            "test_grad_acc_with_reduce_dtype",
            "test_reduce_dtype",
            "test_explicit_prefetching",
            "test_post_optim_event",
            "test_train_parity_multi_group",
            "test_train_parity_multi_group_cpu_offload_eager",
            "test_train_parity_multi_group_unshard_async_op",
            "test_double_forward_with_nested_fsdp_and_checkpoint",
            "test_train_parity_shard_placement_fn_shard_largest_dim",
            "test_train_parity_with_shared_params",
            "test_gradient_accumulation",
            "test_2d_mlp_with_nd_mesh",
            "test_train_parity_hsdp",
            "test_pre_backward_hook_registration_cuda_first_False_cuda",
            # AssertionError: Scalars are not close (allreduce_total != expected_total); survivors hung in Join._get_num_nonjoined_procs
            "test_single_joinable",
            "test_single_joinable_throw",
        ],
    },
    "gfx942": {
        "cuda": [
            # new test
            # AssertionError: Scalars are not equal!
            "test_graph_capture_reclaim_shared_pool",
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
