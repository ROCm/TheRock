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
            # Composable replicate mixed precision abort/SIGSEGV.
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_bf16)",
            # DTensor/sharding failures and timeouts.
            "(TestCommModeFeatures and test_MLPStacked_distributed_sharding_display)",
            "(DistElementwiseOpsTest and test_dropout_partial_redistributes)",
            "(DistTensorRandomInitTest and test_multinomial_sharded)",
            "(TestViewOpsWithLocalTensor and test_squeeze_variants)",
            "(TestDTensorCompileE2E and test_2d_fsdp_tp_compile_use_ca_False)",
            # DDP/join uneven-input behavior drift.
            "(TestJoin and test_multiple_joinables)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs)",
            "(TestDistBackendWithSpawn and test_ddp_uneven_inputs_stop_iteration_sync_bn)",
            # Classic FSDP NCCL timeout/abort.
            "(TestFSDPWrap and test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode0)",
            # Elastic launcher failure.
            "(ElasticLaunchTest and test_virtual_local_rank)",
            # Compute/comm reordering failure.
            "(TestComputeCommReorderingMultiProc and test_custom_estimator_for_non_compute_nodes)",
        ],
    },
}
