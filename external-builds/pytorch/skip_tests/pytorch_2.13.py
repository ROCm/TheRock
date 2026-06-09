# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch nightly (2.13) wheels. These are already
# tracked in the stable-version skip files (pytorch_2.9.py - pytorch_2.12.py)
# and/or generic.py, but those exclusions are not picked up for the nightly
# version, so they are mirrored here. See
# https://github.com/ROCm/TheRock/issues/5596

skip_tests = {
    "common": {
        "cuda": [
            # TestCuda - conflicts with how our test script and runners are
            # configured.
            "test_hip_device_count",
            # TestCudaAllocator - passes on single run, crashes if run in a
            # group. TypeError: 'CustomDecompTable' object is not a mapping
            "test_memory_compile_regions",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator'. The hipblas.h include error persists in the
            # ROCm SDK environment:
            #   fatal error: 'hipblas/hipblas.h' file not found
            "test_mempool_empty_cache_inactive",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator_v1' (same hipblas.h include error)
            "test_mempool_limited_memory_with_allocator",
        ],
        "nn": [
            # TestNNDeviceTypeCUDA - AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
            # TestNNDeviceTypeCUDA - cudnn CTC loss numerical mismatch
            "test_ctc_loss_cudnn_tensor_cuda_cuda",
            # TestNNDeviceTypeCUDA - per-call dropout randomness mismatch
            "test_LSTM_dropout_per_call_randomness_dropout_p_0_5_training_True_cuda",
            # TestNNDeviceTypeCUDA - upsampling launch failure on gfx950
            # Separately tracked in https://github.com/ROCm/TheRock/issues/5270
            "test_upsamplingNearest2d_launch_rocm_cuda",
        ],
        "dynamo": [
            # Run 27182415475 default shards: exact Dynamo failures from raw logs.
            "(TestAfterAot and test_get_compile_args_e2e_symbolic_compile)",
            "(ListTests and test_remove_matches_identity_before_richcompare)",
            "(ListTests and test_remove_uses_item_richcompare)",
            "(LoggingTests and test_logs_out)",
            "(AOTAutogradCacheTests and test_region_activation_memory_budget_causes_cache_miss)",
            "(AOTAutogradCacheTests and test_region_activation_memory_budget_graph_break_cache)",
            "(AOTAutogradCacheBundledTests and test_region_activation_memory_budget_causes_cache_miss)",
            "(AOTAutogradCacheBundledTests and test_region_activation_memory_budget_graph_break_cache)",
            "(DictTests and test_defaultdict_inplace_union_preserves_factory)",
            "(DictTests and test_dict_update_from_mapping_like)",
            "(DictTests and test_dict_update_no_args)",
            "(DictTests and test_dict_update_rejects_bad_sequence_element_length)",
            "(DictTests and test_dict_update_rejects_too_many_args)",
            "(GetItemTests and test_tuple_constructor_preserves_exact_tuple_identity)",
            "(GetItemTests and test_tuple_constructor_rejects_keywords)",
            "(MiscTests and test_param_autograd_grad_in_forward)",
            "(MiscTests and test_param_grad_in_forward)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_conflict_raises_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_covers_invoke_subgraph_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_distinct_per_invoke_subgraph_raises_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_eager_raises_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_partial_annotation_raises_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_per_region_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_reduces_act_mem_cuda)",
            "(ActivationCheckpointingViaTagsTestsCUDA and test_region_activation_memory_budget_validation_cuda)",
            "(FrozensetTests and test_copy_preserves_identity)",
            "(RunDiffGuardTests and test_skip_all_guards_single_cache_entry)",
            "(TpRichcompareTests and test_contains_nan_identity)",
            "(TpRichcompareTests and test_dict_nan_identity)",
            "(TpRichcompareTests and test_dispatch_key_set_eq)",
            "(TpRichcompareTests and test_immutable_list_cmp)",
            "(TpRichcompareTests and test_list_nan_identity)",
            "(TpRichcompareTests and test_list_subclass_cmp)",
            "(TpRichcompareTests and test_set_subclass_custom_eq)",
            "(TpRichcompareTests and test_set_subclass_custom_eq_reversed)",
            "(TpRichcompareTests and test_tensor_eq_user_defined_object)",
            "(TpRichcompareTests and test_tensor_ne_user_defined_object)",
            "(TpRichcompareTests and test_tuple_nan_identity)",
            "(TpRichcompareTests and test_tuple_subclass_custom_eq)",
        ],
        "export": [
            # Run 27182415475 default shards: exact export failures from raw logs.
            "(CppSerdesTestExport and test_aot_export_buffer_assignment_hook_cleanup_after_failed_export_cpp_serdes)",
            "(CppSerdesTestExport and test_buffer_assignment_hook_cleanup_after_failed_export_cpp_serdes)",
            "(CppSerdesTestExport and test_distributed_all_gather_into_tensor_cpp_serdes)",
            "(CppSerdesTestExport and test_distributed_reduce_scatter_tensor_cpp_serdes)",
            "(CppSerdesTestExport and test_export_lstm_hidden_state_shapes_cpp_serdes)",
            "(CppSerdesTestExport and test_export_lstm_where_hidden_state_shape_cpp_serdes)",
            "(CppSerdesTestExport and test_export_min_scaled_dynamic_dim_cpp_serdes)",
            "(StrictExportTestExport and test_aot_export_buffer_assignment_hook_cleanup_after_failed_export_strict)",
            "(StrictExportTestExport and test_buffer_assignment_hook_cleanup_after_failed_export_strict)",
            "(StrictExportTestExport and test_distributed_all_gather_into_tensor_strict)",
            "(StrictExportTestExport and test_distributed_reduce_scatter_tensor_strict)",
            "(StrictExportTestExport and test_export_lstm_hidden_state_shapes_strict)",
            "(StrictExportTestExport and test_export_lstm_where_hidden_state_shape_strict)",
            "(StrictExportTestExport and test_export_min_scaled_dynamic_dim_strict)",
            "(TestExportOnFakeCudaCUDA and test_fake_export___getitem___cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_batch_norm_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_batch_norm_without_cudnn_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_conv2d_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_instance_norm_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_multi_margin_loss_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nn_functional_scaled_dot_product_attention_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_fake_export_nonzero_cuda_float32)",
            "(TestExportOnFakeCudaCUDA and test_preserve_original_behavior_cuda)",
        ],
        "inductor": [
            # Run 27182415475 default shards: exact Inductor failures from raw logs.
            "(TestGpuWrapper and test_aoti_debug_printer_works_on_constants)",
            "(TestGpuWrapper and test_bernoulli1_combo_kernels_False_cuda_gpu_wrapper)",
            "(TestGpuWrapper and test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code)",
            "(TestGpuWrapper and test_map_fullgraph_cpp_wrapper)",
            "(TestGpuWrapper and test_randint_cuda_gpu_wrapper)",
            "(TestGpuWrapper and test_sort_cuda_gpu_wrapper)",
            "(DynamicShapesGpuWrapperGpuTests and test_annotation_training)",
            "(DynamicShapesGpuWrapperGpuTests and test_layer_norm_cuda_dynamic_shapes_gpu_wrapper)",
            "(DynamicShapesGpuWrapperGpuTests and test_randint_cuda_dynamic_shapes_gpu_wrapper)",
            "(DynamicShapesGpuWrapperGpuTests and test_repeat_interleave_2_cuda_dynamic_shapes_gpu_wrapper)",
            "(DynamicShapesGpuWrapperGpuTests and test_scaled_dot_product_attention_cuda_dynamic_shapes_gpu_wrapper)",
            "(TestOperatorReorderForPeakMemory and test_fusion_acc_large_reads)",
            "(TestSelectAlgorithmCleanup and test_benchmark_only_clears_matching_precompile_cache_entry)",
            "(TestSelectAlgorithmCleanup and test_release_benchmark_artifacts_closes_and_clears_state)",
        ],
        "modules": [
            # Run 27182415475 default/inductor: exact CTCLoss failure.
            "test_forward_nn_CTCLoss_cuda_float32",
        ],
        "distributed": [
            # torch.linalg.eig has no non-MAGMA backend; MAGMA not linked in this build
            "test_linalg_ops",
            # Native SIGSEGV during bf16 forward in the mixed-precision cast path
            "(TestFullyShardMixedPrecisionCasts and test_norm_modules_bf16)",
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_bf16)",
            # Child process exits with SIGABRT inside torchelastic launcher (test_run.py)
            "(ElasticLaunchTest and test_virtual_local_rank)",
            # torch.compile + bf16 + DDP composable failure
            "(ReplicateTest and test_compile_bf16)",
            # Kineto/NCCL annotation metadata missing on recorded GPU kernels
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_True)",
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_False)",
            # NCCL symmetric memory rendezvous not supported; host communicator not found
            "test_ce_allgather",
            "test_ce_alltoall",
            # AssertionError: Scalars are not close! fp32 numerical drift in FSDP post-optimizer event
            "(TestFullyShard1DTrainingCore and test_post_optim_event)",
            # /dev/shm exhausted by 8-rank 3D mesh tensor allocs; NCCL shared memory OOM
            "(TestFullyShardHSDP3DTraining and test_3d_mlp_with_nd_mesh)",
            # fp32 gradient mismatch in FSDP activation checkpointing; 1/128 elements exceed tolerance
            "(TestFullyShard1DTrainingCompose and test_train_parity_with_activation_checkpointing)",
            # Run 27182402748 distributed shards: exact failures from raw logs.
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_per_group)",
            "(TestFullyShardMixedPrecisionTraining and test_compute_dtype)",
            "(TestFullyShardMixedPrecisionTraining and test_grad_acc_with_reduce_dtype)",
            "(TestFullyShardMixedPrecisionTraining and test_reduce_dtype)",
            "(TestFullyShardMixedPrecisionCasts and test_clamp_reduce_dtype)",
            "(TestCommunicationCUDA and test_communication_nested_model_False_use_no_sync_False_sharding_strategy0_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_False_use_no_sync_False_sharding_strategy1_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_False_use_no_sync_True_sharding_strategy0_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_False_use_no_sync_True_sharding_strategy1_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_True_use_no_sync_False_sharding_strategy0_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_True_use_no_sync_False_sharding_strategy1_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_True_use_no_sync_True_sharding_strategy0_cuda)",
            "(TestCommunicationCUDA and test_communication_nested_model_True_use_no_sync_True_sharding_strategy1_cuda)",
            "(TestFSDPMixedPrecisionSharded and test_full_precision_in_eval_comm)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_false_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_false_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_false_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_false_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_true_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_true_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_true_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_diff_buffer_reduce_offload_true_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_false_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_false_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_false_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_false_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_true_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_true_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_true_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_fp16_offload_true_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_false_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_false_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_false_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_false_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_true_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_true_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_true_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_no_mp_offload_true_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_false_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_false_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_false_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_false_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_true_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_true_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_true_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_param_and_buf_offload_true_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_false_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_false_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_false_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_false_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_true_fp32_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_true_fp32_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_true_fp64_enable_sharded_grad_scaler)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_e2e_full_shard_mp_only_reduce_offload_true_fp64_none)",
            "(TestFSDPMixedPrecisionSharded and test_mixed_precision_no_reshard_after_forward)",
            "(TestFSDPMixedPrecisionSharded and test_mp_embedding_default)",
            "(TestFSDPMixedPrecisionSharded and test_mp_embedding_only_params_and_bufs)",
            "(TestFSDPMixedPrecisionSharded and test_mp_embedding_params_and_reduce_diff)",
            "(TestFSDPMixedPrecisionSharded and test_mp_embedding_reduce)",
            "(TestFSDPMixedPrecisionUnsharded and test_mixed_precision_e2e_full_shard)",
            "(TestFSDPMixedPrecisionUnsharded and test_mixed_precision_no_reshard_after_forward)",
            "(TestExample02CollectiveOperations and test_reduce_scatter_tensors)",
            "(CommTest and test_coalesced_manager_op_integrity)",
            "(CommTest and test_reduce_scatter_base_k)",
            "(NCCLTraceTest and test_coalescing_manager_collective_timing_enabled_True)",
            "(NCCLTraceTest and test_coalescing_manager_collective_timing_enabled_False)",
            "(TestFakePG and test_allgather_into_tensor_requires_grad)",
            "(TestCollectivesInductor and test_dynamo_graphbreaks_unsupported_async_op)",
            "(TestCollectivesInductor and test_dynamo_rewrite_dist_all_gather)",
            "(TestCollectivesInductor and test_dynamo_rewrite_dist_all_gather_args_match)",
            "(TestCollectivesInductor and test_dynamo_rewrite_dist_reduce_scatter)",
            "(TestCollectivesInductor and test_dynamo_support_collective_op_with_async_op_False)",
            "(TestDTensorDebugModeNCCLBackend and test_allgather_base)",
            "(TestDTensorDebugModeNCCLBackend and test_allgather_base_async_op)",
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",
            "(CPFlexAttentionTest and test_cp_flex_attention_document_mask)",
            "(DistTensorOpsTest and test_scatter)",
            "(DistTensorOpsTestWithLocalTensor and test_scatter)",
            "(ProcessGroupGlooLazyInitTest and test_reduce_scatter_tensor)",
            "(ProcessGroupGlooTest and test_reduce_scatter_tensor)",
            "(ProcessGroupGlooFRTest and test_reduce_scatter_tensor)",
            "(CommTest and test_reduce_scatter_tensor_coalesced)",
            "(ProcessGroupNCCLOneRankTest and test_reduce_scatter)",
            "(TestSingleProc and test_compiled_all_gather_into_tensor_returns_none)",
            "(TestSingleProc and test_compiled_reduce_scatter_tensor_returns_none)",
            "(TestLocalTensorWorld3 and test_all_gather_into_tensor_collective)",
            "(TestLocalTensorWorld3 and test_reduce_scatter_tensor_collective)",
            "(TestFullyShardNonFloatParam and test_non_float_param)",
            "(TestFullyShardOverlap and test_fully_shard_training_overlap)",
            "(TestFakeCollectives and test_collectives)",
            "(TestFSDPHybridShard and test_fsdp_hybrid_shard_basic_setup)",
            "(TestForwardOverlapWorldSizeOneCUDA and test_forward_overlap_cuda)",
            "(TestTPFSDPIntegration and test_fsdp_tp_integration)",
            "(TestCommMode and test_comm_mode_with_c10d)",
            "(DTensorTest and test_to_local_preserves_parameter)",
            "(DTensorTestWithLocalTensor and test_to_local_preserves_parameter)",
            "(CommTest and test_all_gather_into_tensor_coalesced_manager_nccl)",
            "(NcclProcessGroupWithDispatchedCollectivesTests and test_allgather_base)",
            "(ProcessGroupNCCLOpTest and test_reduce_scatter_bfloat16)",
        ],
    },
}
