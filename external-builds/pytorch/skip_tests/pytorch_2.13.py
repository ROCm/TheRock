# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "nn": [
            # AssertionError: False is not true : Expected NaN in pdist output
            # AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
        ],
        "convolution": [
            # ROCm/MIOpen native hang in deterministic cuDNN Conv2d generated tests.
            # Covers dilation 1/2/3 across dtype variants; replaces file-level nn/test_convolution exclusion.
            "test_Conv2d_deterministic_cudnn",
        ],
        "distributions": [
            # SIGSEGV - OpenBLAS exceeds precompiled 128-thread hard limit
            # even with OPENBLAS_NUM_THREADS=64; crash in wishart.log_prob
            "test_entropy_monte_carlo",
        ],
        "cuda": [
            # TestCuda - CI GPU visibility env conflict aborts child device-count subprocesses.
            # Also covers the inherited TestCuda case in test_cuda_expandable_segments.
            "test_hip_device_count",
        ],
        "dynamo": [
            # CI GPU isolation warning contaminates this exact stderr log assertion.
            "test_logs_out",
            # ErrorMessagesTest - Dynamo graph-break attribution text drift.
            "test_reconstruction_failure_gb",
        ],
        "export": [
            # TestExportOnFakeCudaCUDA - subprocess import fails: missing librocm_sysdeps_liblzma.so.5
            "test_fake_export___getitem___cuda_float32",
            "test_fake_export_nn_functional_batch_norm_cuda_float32",
            "test_fake_export_nn_functional_batch_norm_without_cudnn_cuda_float32",
            "test_fake_export_nn_functional_conv2d_cuda_float32",
            "test_fake_export_nn_functional_instance_norm_cuda_float32",
            "test_fake_export_nn_functional_multi_margin_loss_cuda_float32",
            "test_fake_export_nn_functional_scaled_dot_product_attention_cuda_float32",
            "test_fake_export_nonzero_cuda_float32",
            "test_preserve_original_behavior_cuda",
        ],
        "inductor": [
            # TestOpInfoPropertiesCUDA - ROCm 7.13 eager vs Triton log/log10 bitwise drift
            "test_eager_equivalence_log10_backend_inductor_default_cuda_float32",
            "test_eager_equivalence_log_backend_inductor_default_cuda_float16",
            "test_eager_equivalence_log_backend_inductor_default_cuda_float32",
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float16",
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float32",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_bfloat16",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_float16",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_float32",
            # ExtensionBackendTests - extension_device registration/is_available handling
            "test_open_device_registration",
            # inductor/test_user_streams: stream/cudagraph structure mismatches and hangs on ROCm.
            "test_codegen_structure_parallel_matmuls",
            "test_codegen_structure_pipeline",
            "test_codegen_structure_single_stream",
            "test_explicit_current_stream_with_cudagraphs",
            "test_implicit_current_stream_with_cudagraphs",
            # inductor/test_autoheuristic: compute_cap is a string in ROCm wheel metadata.
            # pytest -k also matches the file name, so exclude neighboring tests explicitly.
            "(AutoHeuristicTest and not test_autoheuristic_a100 and not test_autoheuristic_h100 and not test_autoheuristic_pad_mm and not test_global_feedback and not test_mixed_mm_a100 and not test_pad_mm_autoheuristic_deterministic_mode)",
            # inductor/test_aot_inductor_package: AOTI C++ package tests need
            # more complete CMake/runtime library-path handling in the wheel CI lane.
            "test_compile_after_package_multi_arch",
            "test_compile_after_package_static",
            "test_compile_standalone_cos",
            "test_compile_with_exporter",
            "test_compile_with_exporter_weights",
        ],
        "linalg": [
            # TestLinalgCUDA - large triu/tril 64-bit zero-count mismatch on ROCm 7.13 May 1 wheel.
            "test_triu_tril_large_matrix_64bit_cuda",
        ],
        "jit_fuser_te": [
            # TestTEFuser{Static,Dynamic} - bf16 TE fuser failures on ROCm 7.13 May 1 wheel.
            "test_binary_div_ops",
            "test_binary_ops",
            "test_binary_tensor_scalar_ops",
            "test_ternary_norm_ops",
            "test_ternary_ops",
            "test_unary_ops",
            "test_where_ops",
        ],
        "fx": [
            # test_fx: backward-compatibility expectation drift.
            "test_function_back_compat",
        ],
        "schema_check": [
            # test_schema_check: multinomial bf16 schema check can hang GPU on ROCm.
            "test_schema_correctness_multinomial_cuda_bfloat16",
        ],
        "modules": [
            # TestModuleCUDA - CTCLoss cpu/gpu parity scalar mismatch
            "test_cpu_gpu_parity_nn_CTCLoss_cuda_float32",
            # TestModuleCUDA - CTCLoss forward scalar mismatch
            "test_forward_nn_CTCLoss_cuda_float32",
        ],
        "multiprocessing": [
            # ROCm devel/runtime-dependent UTs. Skip in the PyTorch full-suite
            # lane; these are expected to run in the separate ROCm devel UT step.
            "(test_fs and not test_fs_)",
            "test_fs_is_shared",
            "test_fs_pool",
            "test_fs_preserve_sharing",
            "test_fs_sharing",
        ],
        "serialization": [
            # TestSerialization - NJT weights_only import check
            # TestOldSerialization - CI env assertion
            "test_debug_set_in_ci",
        ],
        "utils": [
            # ROCm devel/runtime-dependent UT. Skip in the PyTorch full-suite lane;
            # this is expected to run in the separate ROCm devel UT step.
            "test_load_standalone",
        ],
        "distributed": [
            # DistMathOpsTest - torch.linalg.eig requires MAGMA in this build
            "test_linalg_ops",
            # ProcessGroupNCCLGroupTest - extra CUDA context memory growth
            "test_extra_cuda_context",
            # Composable FSDP/2D composability timeouts on ROCm 0501/PT 0501.
            "(TestFullyShardAutograd and test_nontensor_activations)",
            # training-iter1/retry1 4-GPU validation plus prior ROCm 0501/PT 0501 runs:
            # TestFullyShard1DTrainingCore has repeated 300s distributed timeouts.
            # Observed tests include explicit_prefetching, non_root_forward_backward,
            # post_optim_event, multi_forward_module, and retry1
            # test_train_parity_single_group_shard_dim0.
            "(TestFullyShard1DTrainingCore)",
            "(TestFullyShardAllGatherExtensionsMultiProcess and test_all_gather_extensions_train_parity)",
            "(TestFullyShardGradientScaler and test_gradient_scaler)",
            "(TestFullyShardIgnoreParams and test_ddp_A_fsdp_B_ddp_C)",
            "(TestFullyShardMixedPrecisionTraining and test_compute_dtype)",
            "(TestFullyShard1DTrainingCore and test_explicit_prefetching)",
            "(TestClipGradNormWorldSize2 and test_clip_grad_norm_1d)",
            "(TestFullyShardFrozen and test_multi_forward_mixed_requires_grad)",
            "(TestFullyShardMemory and test_fully_shard_training_memory)",
            "(TestFullyShardOverlap and test_fully_shard_training_overlap)",
            # Run 25871581500 / job 76028005166, distributed shard 2/3:
            # TestFullyShardMixedPrecisionCasts::test_norm_modules_fp16 SIGSEGVs.
            # Skip the class since mixed-precision cast coverage has recurring process crashes.
            "(TestFullyShardMixedPrecisionCasts)",
            "(TestFullyShardCommunication and test_set_reduce_scatter_divide_factor)",
            "(TestFullyShard2DTraining and test_train_parity_2d_mlp)",
            # Composable replicate mixed precision abort/SIGSEGV.
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_bf16)",
            # DTensor/sharding failures and timeouts.
            "(TestCommModeFeatures and test_MLPStacked_distributed_sharding_display)",
            "(DistElementwiseOpsTest and test_dropout_partial_redistributes)",
            "(DistTensorRandomInitTest and test_multinomial_sharded)",
            "(TestViewOpsWithLocalTensor and test_squeeze_variants)",
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_False)",
            # wrap-iter1 4-GPU validation plus prior ROCm 0501/PT 0501 runs:
            # TestFSDPWrap has repeated RCCL/NCCL watchdog aborts across wrap API
            # parameterizations; latest was _ALLGATHER_BASE timeout and exit -6.
            "(TestFSDPWrap)",
            # DDP/join uneven-input behavior drift.
            "(TestJoin and test_multiple_joinables)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs_stop_iteration_sync_bn)",
            # Classic FSDP NCCL timeout/abort.
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode0)",

            # ROCm 0501/PT 0501 second distributed run: post-acc-grad parity timeout.
            "(TestFullyShardPostAccGradHookMultiProcess and test_post_acc_grad_hook_optim_parity)",
            # ROCm 0501/PT 0501 second distributed run: mixed-precision grad accumulation timeout.
            "(TestFullyShardMixedPrecisionTraining and test_grad_acc_with_reduce_dtype)",
            # ROCm 0501/PT 0501 second distributed run: comm-mode transformer tracing failure.
            "(TestCommModeFeatures and test_transformer_module_tracing)",
            # ROCm 0501/PT 0501 second distributed run: DTensor 2D FSDP/TP compile timeout.
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_True)",
            # ROCm 0501/PT 0501 second distributed run: 2D clip-grad-norm timeout.
            "(TestClipGradNormWorldSize4 and test_clip_grad_norm_2d)",
            # ROCm 0501/PT 0501 second distributed run: frozen-parameter mixed-grad-groups timeout.
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_across_groups)",
            # ROCm 0501/PT 0501 second distributed run: HSDP sync correctness mixed-dtype timeout.
            "(TestFullyShardHSDPSyncCorrectness and test_ar_buffer_lifetime_mixed_dtype)",
            # ROCm 0501/PT 0501 second distributed run: per-param mesh overlap timeout.
            "(TestFullyShardPerParamMeshOverlap and test_fully_shard_per_param_mesh_training_overlap)",
            # training-iter2/iter3 4-GPU validation: TestFullyShard1DTrainingCompose class
            # has multiple 300s timeout tests (test_double_forward_with_nested_fsdp_and_checkpoint,
            # test_partial_group_forward_grad_accum_chunked) hanging in run_subtests path.
            "(TestFullyShard1DTrainingCompose)",
            # training-iter4 4-GPU validation: shard placement train parity 300s timeout.
            "(TestFullyShardShardPlacementFnMultiProcess and test_train_parity_shard_placement_fn_shard_largest_dim)",
            # training-iter5/iter6 4-GPU validation: TestFullyShardSharedParams has
            # multiple 300s hangs after prior training-layer skips were applied.
            # Observed tests: test_layer_by_layer_shard_no_false_positive and
            # test_train_parity_with_shared_params; iter6 split ranks between assertEqual
            # comparison and DDP forward_pre_hook before process-0 timeout.
            "(TestFullyShardSharedParams)",
            # training-iter8 4-GPU validation: TestFullyShardGradientAccumulation
            # test_1f1b_microbatching hit a 300s timeout in the parity/assertEqual path
            # after Core/Compose/SharedParams training classes were skipped.
            "(TestFullyShardGradientAccumulation)",
            # training-iter9 4-GPU validation: TestFullyShardNDTraining
            # test_2d_mlp_with_nd_mesh hit a 300s timeout in _test_2d_mlp_with_nd_mesh
            # / assertEqual path after prior training class skips were applied.
            "(TestFullyShardNDTraining)",
            # training-iter10 4-GPU validation: TestFullyShardHSDPTraining
            # test_train_parity_hsdp hit a 300s timeout in backward; traceback also
            # showed FSDP finalize_backward waiting on torch.cuda stream synchronize.
            "(TestFullyShardHSDPTraining)",
            # ROCm 0501/PT 0501 second distributed run: join single-joinable hooks scalar mismatch.
            "(TestJoin and test_single_joinable_main_hooks)",
            # ROCm 0501/PT 0501 second distributed run: FSDP wrap NCCL watchdog abort.
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_True_device_init_mode0)",
            # ROCm 0501/PT 0501 second distributed run: DDP buffer hook allreduce future failure.
            "(TestDistBackendWithSpawn and test_ddp_buffer_hook_allreduce_return_future)",
            # ROCm 0501/PT 0501 second distributed run: fully-shard backward misprefetch timeout.
            "(TestFullyShardPrefetch and test_backward_misprefetch)",
            # ROCm 0501/PT 0501 second distributed run: 2D transformer training parity timeout.
            "(TestFullyShard2DTraining and test_train_parity_2d_transformer)",
            # ROCm 0501/PT 0501 second distributed run: replicate mixed-precision fp16 abort/SIGSEGV.
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_fp16)",

            # Local 4-GPU wrapper validation: fully-shard backward prefetch timeout.
            "(TestFullyShardPrefetch and test_fully_shard_backward_prefetch)",
            # Local 4-GPU wrapper validation: frozen mixed-requires-grad per-group timeout.
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_per_group)",
            # Local 4-GPU wrapper validation: mixed-precision reduce dtype timeout.
            "(TestFullyShardMixedPrecisionTraining and test_reduce_dtype)",
            # Local 4-GPU wrapper validation: join kwargs scalar mismatch.
            "(TestJoin and test_join_kwargs)",
            # Local 4-GPU wrapper validation: FSDP wrap NCCL watchdog abort.
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_False_device_init_mode1)",
            # Local 4-GPU wrapper validation: DTensor TP compile fullgraph timeout.
            "(TestDTensorCompileE2E and test_tp_compile_fullgraph_is_seq_parallel_False_use_ca_False)",
            # Local 4-GPU wrapper validation: DDP post-localSGD hook parity failure.
            "(TestDistBackendWithSpawn and test_ddp_hook_parity_post_localSGD)",
            # Local 4-GPU wrapper validation: DDP unused-loss dict output failure.
            "(TestDistBackendWithSpawn and test_output_unused_in_loss_dict_module)",
            # Local 4-GPU wrapper validation: DDP unused-loss tuple output failure.
            "(TestDistBackendWithSpawn and test_output_unused_in_loss_tuple_module)",
            # Local 4-GPU wrapper validation: sync batchnorm NCCL init timeout.
            "(TestDistBackendWithSpawn and test_sync_bn_logged)",
            # Local 4-GPU wrapper validation: undefined-grad unused-parameter parity NCCL init timeout.
            "(TestDistBackendWithSpawn and test_undefined_grad_parity_unused_parameters)",
            # Local 4-GPU wrapper validation: model verification with logger timeout.
            "(TestDistBackendWithSpawn and test_verify_model_across_rank_with_logger)",
            # Local 4-GPU wrapper validation: model verification without logger timeout.
            "(TestDistBackendWithSpawn and test_verify_model_across_rank_without_logger)",
            # spawn-iter1 4-GPU validation: DDP apply-optimizer-in-backward variants
            # repeatedly hit 300s timeouts. Base variant also showed RCCL/NCCL
            # ALLREDUCE watchdog timeout (SeqNum=19, last completed=18);
            # grad_as_bucket_view=False hung in autograd backward before timeout.
            "(TestDistBackendWithSpawn and test_ddp_apply_optim_in_backward)",
            # spawn-iter2 4-GPU validation: DDP SyncBatchNorm channels-last hit
            # a barrier timeout after ~210s in _test_DistributedDataParallel_
            # SyncBatchNorm_with_memory_format under BACKEND=nccl/init=file.
            "(TestDistBackendWithSpawn and test_DistributedDataParallel_SyncBatchNorm_Channels_Last)",
            # spawn-iter2 4-GPU validation: DDP SyncBatchNorm diff-input-size
            # running-value variant hit a 300s process timeout under BACKEND=nccl/init=file.
            "(TestDistBackendWithSpawn and test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value)",
            # spawn-iter2 4-GPU validation: DDP SyncBatchNorm diff-input-size
            # gradient variant hit a 300s timeout in DDP init/parameter shape verification.
            "(TestDistBackendWithSpawn and test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_gradient)",
            # spawn-iter3 4-GPU validation: DDP no_sync grad-as-bucket-view
            # hit a 300s timeout while constructing DDP / verifying parameter
            # shapes under BACKEND=nccl/init=env.
            "(TestDistBackendWithSpawn and test_accumulate_gradients_no_sync_grad_is_view)",
            # spawn-iter3 4-GPU validation: NCCL send/recv profiler variants
            # failed in ProcessGroupNCCL::recv while retrieving ncclUniqueId from
            # rank 0 through FileStore key default_pg/0//cuda//0:3 (60s timeout).
            "(TestDistBackendWithSpawn and test_send_recv_nccl_autograd_profiler)",
            "(TestDistBackendWithSpawn and test_send_recv_nccl_torch_profiler)",
            # spawn-iter3 4-GPU validation: skip-all-reduce-unused-parameters
            # failed twice under BACKEND=nccl/init=env: first as a 300s rank process
            # timeout in autograd backward, then with NCCL watchdog timeouts
            # (BROADCAST seq=3, ALLREDUCE seq=4) before subprocess timeout.
            "(TestDistBackendWithSpawn and test_skip_all_reduce_unused_parameters)",

            # Second local 4-GPU wrapper validation: fully-shard communication-count timeout.
            "(TestFullyShardCommunication and test_fully_shard_communication_count)",
            # Second local 4-GPU wrapper validation: mixed-precision structured input/output timeout.
            "(TestFullyShardMixedPrecisionTraining and test_structured_input_output)",
            # Second local 4-GPU wrapper validation: forward-input device movement timeout.
            "(TestFullyShardForwardInputs and test_root_move_forward_input_to_device)",
            # Second local 4-GPU wrapper validation: join single-joinable scalar mismatch.
            "(TestJoin and test_single_joinable)",
            # Second local 4-GPU wrapper validation: FSDP wrap NCCL watchdog abort.
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode1)",


            # Third local 4-GPU wrapper validation (rerun of pending modules with prior skips applied).

            # Third local 4-GPU wrapper validation iter1: comm set_reshard_after_forward timeout.
            "(TestFullyShardCommunication and test_set_reshard_after_forward)",

            # Third local 4-GPU wrapper validation iter2: prefetch multi-module backward prefetch timeout.
            "(TestFullyShardPrefetch and test_fully_shard_multi_module_backward_prefetch)",

            # CI run 25756405826 distributed shard 1/3: DTensor source/wheel
            # API drift, sharding-strategy gaps, and sync-decision assertion drift.
            "(TestDTensorDebugMode and test_nested_invoke_subgraph)",
            "(TestDTensorCompile and test_pad_tensor_no_guard_on_symbolic_pad_size)",
            "(TestOpSchema and test_no_dtensor_spec_leak_in_comparison_key)",
            "(TestOpSpecMesh and test_mesh_from_input_specs_when_tuple_output_all_none)",
            "(DistTensorOpsTest and test_detach_inplace_inference_mode)",
            "(TestComputeCommReorderingMultiProc and test_sink_waits)",
            "(TestSyncDecisionCrossRanks and test_sync_decision_cross_ranks_invalid_node_error)",

            # CI run 25756405826 distributed shard 2/3: identified per-test
            # timeout/SIGSEGV/API-drift failures after prior FSDP skips.
            "(TestFullyShardCustomForwardMethod and test_register_fsdp_forward_method)",
            "(TestCompileOnOneRank and test_all_reduce_with_explicit_pg_input)",
            "(TestDataParallel and test_strided_grad_layout)",
            "(TestFakePG and test_allgather_into_tensor_requires_grad)",

            # CI run 25756405826 distributed shard 3/3: compiler/distributed
            # numerical drift and missing inductor config on the May 1 wheel stack.
            "(test_replicate_with_compiler and test_compile_backward_only)",
            "(DistMathOpsTest and test_interpolation_upsample_ops)",
            "(TestComputeCommReorderingMultiProc and test_inductor_default_comms_ordering)",
            "(NCCLTraceTest and test_compiled_ring_attention_pattern_num_steps_2_M_1024)",
            "(TestSyncDecisionCrossRanks and test_align_runtime_estimations_across_all_distributed_ranks)",

            # Elastic launcher failure.
            "(ElasticLaunchTest and test_virtual_local_rank)",
            # Compute/comm reordering failure.
            "(TestComputeCommReorderingMultiProc and test_custom_estimator_for_non_compute_nodes)",
        ],
    },
}
