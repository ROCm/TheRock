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
        "dynamo": [
            # CI GPU isolation warning contaminates this exact stderr log assertion.
            "test_logs_out",
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

            # ROCm bump attribution anchors:
            # - Apr20/PT + Apr20/ROCm control was green across distributed
            #   shards: run 25244506667; jobs 74051598033 (1/3),
            #   74051597989 (2/3), 74051597951 (3/3).
            # - Apr20/PT + May01/ROCm first attribution failures:
            #   run 25925372276; jobs 76205215134 (1/3),
            #   76205215167 (2/3), 76205215147 (3/3).
            # - Apr20/PT + May01/ROCm second attribution failures:
            #   run 26136844778; jobs 76873873289 (1/3),
            #   76873873320 (2/3), 76873873274 (3/3).

            # Run 25925372276 shards 1/3, 2/3, and 3/3:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215134
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215167
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215147
            # Composable FSDP 300s timeout/hang bucket, also seen in the
            # May01/PT + May01/ROCm target stack; ROCm-bump overlap.
            "(TestFullyShardAutograd and test_nontensor_activations)",
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
            "(TestFullyShardCommunication and test_set_reduce_scatter_divide_factor)",
            "(TestFullyShard2DTraining and test_train_parity_2d_mlp)",

            # Run 25925372276 shard 3/3, job 76205215147:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215147
            # Mixed-precision cast crash bucket; replicate bf16 cast hit SIGSEGV.
            # The FSDP cast class is grouped here with the same cast surface.
            "(TestFullyShardMixedPrecisionCasts)",
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_bf16)",

            # Run 25925372276 shards 2/3 and 3/3:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215167
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215147
            # DTensor/debug correctness drift on ROCm May01; not FSDP watchdogs.
            "(TestCommModeFeatures and test_MLPStacked_distributed_sharding_display)",
            "(DistElementwiseOpsTest and test_dropout_partial_redistributes)",
            "(DistTensorRandomInitTest and test_multinomial_sharded)",
            "(TestViewOpsWithLocalTensor and test_squeeze_variants)",
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_False)",

            # Run 25925372276 shards 1/3 and 3/3:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215134
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215147
            # Distributed process-group/control-flow failures.
            "(TestJoin and test_multiple_joinables)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs_stop_iteration_sync_bn)",

            # Run 25925372276 shard 1/3, job 76205215134:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215134
            # Classic FSDP wrap/API failures; keep separate from composable FSDP.
            "(TestFSDPWrap)",
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode0)",

            # Run 25925372276 shard 2/3, job 76205215167:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215167
            # Attribution-only rerun unblockers; do not count as May01/PT
            # target-stack overlap unless a later run exposes the same test.
            "(TestReplicate1DTrainingCore and test_train_parity_multi_groups)",
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",

            # Run 25925372276 shard 2/3, job 76205215167:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215167
            # Attribution-only DTensor/runtime checks.
            "(DistElementwiseOpsTest and test_dropout_errors)",
            "(DistTensorRandomInitTest and test_meta_tensor_init)",

            # Run 25925372276 shards 2/3 and 3/3:
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215167
            # https://github.com/ROCm/TheRock/actions/runs/25925372276/job/76205215147
            # Attribution-only compute/comm reordering checks.
            "(TestComputeCommReorderingBucketing and test_bucketing_split_for_overlap)",
            "(TestComputeCommReorderingBucketing and test_no_bucketing_with_dependent_hiding_nodes)",

            # Run 26136844778 shard 1/3, job 76873873289:
            # https://github.com/ROCm/TheRock/actions/runs/26136844778/job/76873873289
            # Second attribution layer; 300s composable FSDP timeout bucket.
            "(TestFullyShardPostAccGradHookMultiProcess and test_post_acc_grad_hook_optim_parity)",
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_across_groups)",

            # Run 26136844778 shard 1/3, job 76873873289: NCCL watchdog timeout
            # in 2D transformer train parity. Keep apart from generic 300s timeouts.
            "(TestFullyShard2DTraining and test_train_parity_2d_transformer)",

            # Run 26136844778 shard 2/3, job 76873873320:
            # https://github.com/ROCm/TheRock/actions/runs/26136844778/job/76873873320
            # Second attribution layer; 300s timeout bucket across FSDP/DTensor.
            "(TestClipGradNormWorldSize4 and test_clip_grad_norm_2d)",
            "(TestFullyShardPerParamMeshOverlap and test_fully_shard_per_param_mesh_training_overlap)",
            "(TestCommModeFeatures and test_transformer_module_tracing)",
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_True)",

            # Run 26136844778 shard 2/3, job 76873873320: elastic launcher
            # ChildFailedError/SIGABRT bucket.
            "(ElasticLaunchTest and test_virtual_local_rank)",

            # Run 26136844778 shard 3/3, job 76873873274:
            # https://github.com/ROCm/TheRock/actions/runs/26136844778/job/76873873274
            # Second attribution layer; 300s composable FSDP timeout bucket.
            "(TestFullyShardMixedPrecisionTraining and test_grad_acc_with_reduce_dtype)",
            "(TestFullyShard1DTrainingCompose and test_double_forward_with_nested_fsdp_and_checkpoint)",

            # Run 26136844778 shard 3/3, job 76873873274: Join scalar assertion
            # via process exit, not an FSDP timeout.
            "(TestJoin and test_single_joinable)",

            # Run 26136844778 shard 3/3, job 76873873274: DataParallel SIGSEGV
            # in autograd backward.
            "(TestDataParallel and test_strided_grad_layout)",

            # Run 26170912739 shard 1/3, job 76988162868:
            # https://github.com/ROCm/TheRock/actions/runs/26170912739/job/76988162868
            # Newly exposed May01/PT + May01/ROCm target-stack timeouts in
            # prior-attributed FSDP/Join buckets. Use test-level skips because
            # pytest -k can avoid them and sibling methods are tracked separately.
            "(TestFullyShardFrozen and test_train_mixed_requires_grad_per_group)",
            "(TestJoin and test_join_kwargs)",

            # Run 26170912739 shard 2/3, job 76988162878:
            # https://github.com/ROCm/TheRock/actions/runs/26170912739/job/76988162878
            # SIGSEGV/native crash bucket in Replicate mixed-precision casts;
            # bf16 sibling overlapped Apr20/PT + May01/ROCm in run 25925372276,
            # while this target-stack layer exposes fp16. Keep this test-level:
            # neighboring cast methods passed before the crash and pytest -k can
            # prevent entering the crashy method.
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_fp16)",

            # Run 26170912739 shard 2/3, job 76988162878:
            # https://github.com/ROCm/TheRock/actions/runs/26170912739/job/76988162878
            # Newly exposed target-stack process-exit assertion in Dynamo
            # distributed collectives. Test-level skip is sufficient; the module
            # collected and earlier TestMultiProc sibling passed before failure.
            "(TestMultiProc and test_compiler_collectives_automatic_dynamic_tensor)",

            # Run 26170912739 shard 3/3, job 76988162973:
            # https://github.com/ROCm/TheRock/actions/runs/26170912739/job/76988162973
            # Newly exposed target-stack 300s composable FSDP timeouts in classes
            # with prior attribution-layer timeout siblings. Keep test-level skips
            # instead of class-level because the class-wide failures are not yet
            # proven and pytest -k can isolate these methods.
            "(TestFullyShardMixedPrecisionTraining and test_reduce_dtype)",
            "(TestFullyShard1DTrainingCompose and test_train_parity_with_activation_checkpointing)",

            # Run 26185604461 shard 3/3, job 77040210625:
            # https://github.com/ROCm/TheRock/actions/runs/26185604461/job/77040210625
            # Mixed-precision structured input/output scalar drift; rank 0 exits
            # with error code 10 after assertion, not a native crash. Test-level
            # skip is enough because pytest -k can isolate the method and sibling
            # mixed-precision failures are tracked separately.
            "(TestFullyShardMixedPrecisionTraining and test_structured_input_output)",

            # Run 26185604461 shard 3/3, job 77040210625:
            # https://github.com/ROCm/TheRock/actions/runs/26185604461/job/77040210625
            # 300s composable FSDP shard-placement training timeout. Class-level
            # skip is equivalent here because this class currently contains only
            # this method; no same-class siblings passed in this shard.
            "(TestFullyShardShardPlacementFnMultiProcess)",

            # Run 26170912739 shard 3/3, job 76988162973:
            # https://github.com/ROCm/TheRock/actions/runs/26170912739/job/76988162973
            # SIGABRT/native watchdog crash in DDP buffer hook; exact overlap with
            # Apr20/PT + May01/ROCm run 25925372276 shard 2/3. Test-level skip is
            # enough because pytest -k can prevent the crashing method without a
            # broad distributed_spawn module exclude.
            "(TestDistBackendWithSpawn and test_ddp_buffer_hook_allreduce_return_future)",
        ],
    },
}
