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
            # --- Host / CI limits (/dev/shm, GPU memory context) ---
            # Error while creating shared memory segment /dev/shm/nccl-VPyhzw (size 21823872), error: No space left on device (28)
            "test_3d_mlp_with_nd_mesh",
            # AssertionError: False is not true : cuda:0 used 2615148544.0 bytes after collective, 70% more than the status before (1495269376.0 bytes). Extra CUDA context may have been created.
            "test_extra_cuda_context",

            # --- Timeouts (pytest-timeout, multiprocessing join, undifferentiated CI hangs / ROCm slowness) ---
            # Former umbrella: "Timout errors (fsdp)". Additional cases from gfx94X-dcgpu —
            # https://github.com/ROCm/TheRock/actions/runs/25875103215
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
            "(TestFullyShardCollectiveOps and test_all_gather_fp32)",
            "(TestFullyShardCommunication and test_set_reduce_scatter_divide_factor)",
            "(TestFullyShardPrefetch and test_backward_misprefetch)",
            "(TestFullyShardPrefetch and test_fully_shard_backward_prefetch)",
            "(TestFullyShardPrefetch and test_fully_shard_multi_module_backward_prefetch)",
            "(TestFullyShardPrefetch and test_set_modules_to_backward_prefetch_inside_ac)",
            "(TestFullyShardPrefetch and test_set_modules_to_backward_prefetch and not inside_ac)",
            "(TestFullyShardPrefetch and test_set_modules_to_forward_prefetch)",
            "(TestFullyShardUnshardMultiProcess and test_unshard_async)",
            "(TestFullyShardForceSumReduction and test_fully_shard_force_sum_both_reductions)",
            "(TestFullyShardForceSumReduction and test_fully_shard_force_sum_reduce_scatter)",
            "(TestStateDict and test_fsdp)",
            "(TestTrackerFullyShard1DTrainingCompose and test_tracker_with_activation_checkpointing)",
            "(TestBackwardPrefetch and test_backward_prefetch)",
            "(TestClipGradNormCUDA and test_ddp_parity_cuda)",
            "(TestClipGradNormCUDA and test_low_precision_grads_cuda)",
            "(TestFSDPCheckpoint and cpu_offload0 and offload_activations_False and use_orig_params_False)",
            "(TestFSDPCheckpoint and cpu_offload0 and offload_activations_False and use_orig_params_True)",
            "(TestFSDPCheckpoint and cpu_offload0 and offload_activations_True and use_orig_params_False)",
            "(TestFSDPCheckpoint and cpu_offload1 and offload_activations_False and use_orig_params_False)",
            "(TestFSDPCheckpoint and cpu_offload1 and offload_activations_False and use_orig_params_True)",
            "(TestFSDPCheckpoint and cpu_offload1 and offload_activations_True and use_orig_params_False)",
            "(TestFSDPCheckpoint and cpu_offload1 and offload_activations_True and use_orig_params_True)",
            "(TestHooksCUDA and test_pre_backward_hook_registration_after_state_dict_cuda)",
            "(TestHooksCUDA and test_pre_backward_hook_registration_cuda_first_True_cuda)",
            "(TestHooksCUDA and test_pre_backward_hook_registration_cuda_first_False_cuda)",
            "(TestParityWithDDPCUDA and test_delayed_optim_step_offload_false_none_cuda)",
            "(TestParityWithDDPCUDA and test_delayed_optim_step_offload_true_none_cuda)",
            "(TestParityWithDDPCUDA and test_nested_always_wrap_model_offload_true_none_cuda)",
            "(TestParityWithDDPCUDA and test_nested_wrapped_model_single_iteration_mixed_precision_offload_false_none_cuda)",
            "(TestParityWithDDPCUDA and test_nested_wrapped_model_single_iteration_mixed_precision_offload_true_none_cuda)",
            "(TestParityWithDDPCUDA and test_transformer_offload_false_none_cuda)",
            "(TestNoGradCUDA and test_transformer_no_grad_mixed_precision_True_cuda)",
            "(TestFullyShardCompile and test_nested_fully_shard_backend_aot_eager and not decomp_partition)",
            "(TestFullyShardCompile and test_nested_fully_shard_backend_aot_eager_decomp_partition)",
            "(TestFullyShardCompile and test_nested_fully_shard_backend_inductor_fullgraph_True and not graph_partition)",
            "(TestFullyShardCompile and test_nested_fully_shard_backend_inductor_fullgraph_True_graph_partition)",
            "(TestFullyShardCompile and test_transformer_backend_aot_eager and not decomp_partition)",
            "(TestFullyShardCompile and test_transformer_backend_aot_eager_decomp_partition)",
            "(TestFullyShardMixedPrecisionTraining and test_compute_dtype)",
            "(TestFullyShardMixedPrecisionTraining and test_grad_acc_with_reduce_dtype)",
            "(TestFullyShardMixedPrecisionTraining and test_reduce_dtype)",
            "(TestFullyShard1DTrainingCore and test_explicit_prefetching)",
            "(TestFullyShard1DTrainingCore and test_non_root_forward_backward)",
            "(TestFullyShard1DTrainingCore and test_post_optim_event)",
            "(TestFullyShard1DTrainingCore and test_train_parity_multi_group and not cpu_offload and not unshard_async_op)",
            "(TestFullyShard1DTrainingCore and test_train_parity_multi_group_cpu_offload_eager)",
            "(TestFullyShard1DTrainingCore and test_train_parity_multi_group_unshard_async_op)",
            "(TestFullyShard1DTrainingCompose and test_double_forward_with_nested_fsdp_and_checkpoint)",
            "(TestFullyShardShardPlacementFnMultiProcess and test_train_parity_shard_placement_fn_shard_largest_dim)",
            "(TestFullyShardSharedParams and test_train_parity_with_shared_params)",
            "(TestFullyShardGradientAccumulation and test_gradient_accumulation)",
            "(TestFullyShardNDTraining and test_2d_mlp_with_nd_mesh)",
            "(TestFullyShardHSDPTraining and test_train_parity_hsdp)",
            "(TestFSDPFineTuneCUDA and test_backward_reshard_hooks_cuda)",
            "(TestFreezingWeights and nested_trunk_True and GradToNone and forward_prefetch_False)",
            "(TestFSDPOptimState and test_flatten_sharded_optim_state_dict_nested)",
            "(TestFSDPOptimState and test_flatten_sharded_optim_state_dict_transformer)",
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and local_state_dict and cpu_offload1 and fp16_True)",
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and local_state_dict and cpu_offload0 and fp16_True)",
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and cpu_offload1 and fp16_False and not local_state_dict)",
            "(TestFSDPUseOrigParamsMultipleParamGroups and test_fsdp_compile)",
            "(TestFSDPUseOrigParamsMultipleParamGroups and test_diff_hyperparams_cpu_offload_sharding_strategy_str_full_shard)",
            "(DeviceMeshCollectiveTest and test_reduce_scatter_uneven)",
            "(TestExplicitUnshardCUDA and test_unshard_async_use_orig_params_False_cuda)",
            "(TestCommunicationHooks and test_bf16_hook_has_wrapping_True_sharding_strategy1)",
            "(TestParityWithDDPCUDA and test_delayed_reduce_scatter_offload_true_none_cuda)",
            "(TestFSDPMixedPrecisionSharded and test_full_precision_in_eval and not comm)",
            "(TestFSDPMixedPrecisionSharded and test_full_precision_in_eval_comm)",
            "(TestFSDPWrap and test_main_wrap_api and cpu_offload0 and backward_prefetch0)",
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_False)",
            "(TestJoin and test_single_joinable)",
            "(TestFullyShardWithDistributedStateDict and test_save_with_fsdp1_and_load_with_fsdp2)",
            "(TestFSDPCheckpoint and cpu_offload0 and offload_activations_True and use_orig_params_True)",
            "(TestGradAcc and test_grad_acc_configs0_use_orig_params_False)",
            "(TestFSDPHybridShard and test_fsdp_hybrid_shard_basic_setup)",
            "(TestFSDPIgnoredModules and test_ignored_modules_not_under_wrapped_root_ignore_modules_True)",
            "(TestFSDPMemory and test_fsdp_memory_ckpt_ckpt)",
            "(TestFSDPWithMetaDevice and test_nested_model_with_meta_device_default_init_auto_wrap_True)",
            "(TestPureFP16CUDA and test_pure_fp16_training_cuda)",
            "(TestShardedGradScalerParityWithDDP and test_fsdp_ddp_parity_with_grad_scaler_offload_false_none_mixed_precision_none)",
            "(TestUnshardParams and test_with_grads_core)",
            "(TestCommModeFeatures and test_MLPStacked_distributed_sharding_display)",
            "(DistributedDataParallelTest and test_ddp_weight_sharing)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs_stop_iteration_sync_bn)",
            "(TestFSDPFineTuneCUDA and test_parity_with_ddp_cuda)",
            "(TestFreezingWeights and nested_trunk_True and GradToNone and forward_prefetch_True)",
            "(TestFSDPOptimState and test_full_optim_state_dict_keys)",
            "(TestFSDPWrap and test_main_wrap_api and cpu_offload0 and backward_prefetch1 and forward_prefetch_False and device_init_mode0)",
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and local_state_dict and cpu_offload1 and fp16_False and use_orig_params_False)",
            "(TestFSDPMixedPrecisionSharded and test_mp_batchnorm_convert_sync_bn_True)",
            "(TestFSDPOptimState and test_optim_state_dict_nested and FULL_STATE_DICT and use_multiple_param_groups_False and rank0_only_False and use_diff_optim_inputs_False)",
            "(TestParityWithDDPCUDA and test_nested_always_wrap_model_offload_false_none_cuda)",
            "(TestCommunicationHooks and test_fp16_hook_has_wrapping_True_sharding_strategy1)",
            "(TestExplicitUnshardCUDA and test_unshard_async_use_orig_params_True_cuda)",
            "(TestFSDPUseOrigParamsMultipleParamGroups and test_diff_trainability)",
            "(TestFullyShardWithDistributedStateDict and test_save_with_fsdp2_tp_and_load_with_tp)",
            "(TestFSDPIgnoredModules and test_ignored_modules_not_under_wrapped_root_ignore_modules_False)",
            "(TestFSDPHybridShard and test_fsdp_hybrid_shard_parity)",
            "(TestFSDPMemory and test_fsdp_memory_ckpt_no_ckpt)",
            "(TestFSDPMiscMultiProcess and test_fsdp_device_id_use_index_False)",
            "(TestShardedGradScalerParityWithDDP and test_fsdp_ddp_parity_with_grad_scaler_offload_false_none_none_none)",
            "(TestJoin and test_join_kwargs)",
            "(TestGradAcc and test_grad_acc_configs0_use_orig_params_True)",

            # --- Numerical / parity (tensor mismatch, meta init, FSDP vs ref loss) ---
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and cpu_offload1 and fp16_True and use_orig_params_False and not local_state_dict)",
            "(TestFSDPStateDict and test_basic_save_and_load_state_dict and sharded_state_dict and cpu_offload0 and fp16_False and use_orig_params_False)",
            "(TestFSDPWithMetaDevice and test_nested_model_with_meta_device_reset_params_auto_wrap_True)",
            "(TestParityWithDDPCUDA and test_transformer_offload_true_no_shard_cuda)",

            # --- Spawned child abnormal exit (non-zero / signal from fork/spawn worker) ---
            "(TestDistBackendWithSpawn and test_ddp_buffer_hook_allreduce_return_future)",

            # --- Worker exception (distributed autograd / pipeline) ---
            "(ScheduleTest and test_grad_with_manual_ScheduleClass1_shape_inference_False)",

            # --- Context-parallel flex attention (scalar mismatch) ---
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",
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
