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
            # Run 27246343570 default shard 1/10, job 80461205629:
            # CrossEntropyLoss 2d out-of-bounds device assert reports HIP
            # visibility warning / GPU memory fault text instead of the
            # expected device-assert stderr signature; FAILED CONSISTENTLY.
            "(TestNNDeviceTypeCUDA and test_cross_entropy_loss_2d_out_of_bounds_class_index_cuda_float16)",
            "(TestNNDeviceTypeCUDA and test_cross_entropy_loss_2d_out_of_bounds_class_index_cuda_float32)",
            # Run 27228539427 default shard 8/10:
            # Conv2d deterministic cudnn dilation=2 bf16 hangs GPU in backward
            # after MIOpen GemmFwdRest workspace warning.
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_2_cuda_bfloat16)",
            # Run 27246343570 default shard 8/10, job 80461205597:
            # Conv2d deterministic cudnn dilation=2 complex64 exceeds the
            # TheRock command watchdog; dilation=3 bf16 hits a MIOpen GPU hang.
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_2_cuda_complex64)",
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_3_cuda_bfloat16)",
            # Run 27228539427 default shard 10/10:
            # Conv2d deterministic cudnn dilation=1 variants exceed TheRock's
            # command watchdog on gfx94X; float16 was active when GitHub
            # cancelled at 6h.
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_1_cuda_bfloat16)",
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_1_cuda_complex64)",
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_1_cuda_float16)",
            # Run 27246343570 default shard 10/10, job 80461205617:
            # Additional deterministic convolution variants repeatedly hit
            # TheRock 30-minute watchdogs before the job's 6h cancellation.
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_1_cuda_float32)",
            "(TestConvolutionNNDeviceTypeCUDA and test_Conv2d_deterministic_cudnn_dilation_2_cuda_float16)",
        ],
        "custom_operator": [
            # Run 27228539427 default shard 7/10:
            # TestInferSchemaWithAnnotation::test_name_error_hint failed
            # consistently due to an error-message regex mismatch around
            # `from __future__ import annotations`.
            "(TestInferSchemaWithAnnotation and test_name_error_hint)",
        ],
        "dynamo": [
            # Run 27228539427 default shard 3/10:
            # LoggingTests::test_logs_out fails consistently because a ROCm
            # HIP visibility warning is injected into Dynamo's exact
            # log-output comparison.
            "(LoggingTests and test_logs_out)",
            # Run 27228539427 default shard 6/10:
            # linalg_inv_ex AOT dynamic-shapes graph does not preserve the
            # expected error behavior on ROCm.
            "(DynamicShapesReproTests and test_linalg_inv_check_errors_preserved_in_aot_graph_dynamic_shapes)",
            # Run 27228539427 default shard 6/10:
            # singular linalg.inv under aot_eager dynamic shapes does not
            # raise _LinAlgError on ROCm.
            "(DynamicShapesReproTests and test_linalg_inv_singular_aot_eager_raises_dynamic_shapes)",
            # Run 27228539427 default shard 1/10:
            # linalg_inv_ex AOT graph does not preserve expected error behavior
            # on ROCm.
            "(ReproTests and test_linalg_inv_check_errors_preserved_in_aot_graph)",
            # Run 27228539427 default shard 1/10:
            # singular linalg.inv under aot_eager does not raise _LinAlgError
            # on ROCm.
            "(ReproTests and test_linalg_inv_singular_aot_eager_raises)",
        ],
        "export": [
            # Run 27246343570 default shard 10/10, job 80461205617:
            # TestExportOnFakeCudaCUDA subprocesses exit 127 because
            # libpython3.12.so.1.0 is missing. Expressions mirrored from 2.12.
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
        "functorch": [
            # Run 27228539427 default shard 7/10:
            # TestOperatorsCUDA::test_grad_unbind_copy_cuda_float32 hit a GPU
            # hang followed by Fatal Python error: Aborted.
            "(TestOperatorsCUDA and test_grad_unbind_copy_cuda_float32)",
        ],
        "modules": [
            # Run 27228539427 inductor shard 1/4:
            # CTCLoss CPU/GPU parity scalar mismatch under --inductor.
            "test_cpu_gpu_parity_nn_CTCLoss_cuda_float32",
            # Run 27228539427 inductor shard 3/4:
            # CTCLoss forward returns a huge scalar under --inductor.
            "test_forward_nn_CTCLoss_cuda_float32",
        ],
        "inductor": [
            # Run 27228539427 inductor shard 1/4 and default shard 6/10:
            # diagonal_scatter backward grad mismatch on ROCm Inductor.
            "(GPUTests and test_diagonal_scatter_backward_cuda)",
            # Run 27228539427 default shard 6/10:
            # PadMM exclude-padding cache assertion fails: expected 2 local-cache
            # entries but got 0.
            "(PadMMTest and test_exclude_padding)",
            # Run 27228539427 default shard 2/10:
            # diagonal_scatter backward dynamic-shapes mismatch on ROCm Inductor.
            "(DynamicShapesGPUTests and test_diagonal_scatter_backward_dynamic_shapes_cuda)",
            # Run 27228539427 default shard 1/10:
            # diagonal_scatter backward dynamic-shapes codegen grad mismatch on ROCm Inductor.
            "(DynamicShapesCodegenGPUTests and test_diagonal_scatter_backward_dynamic_shapes_cuda)",
            # Run 27228539427 default shard 9/10:
            # diagonal_scatter backward dynamic-shapes CPU grad mismatch on ROCm Inductor.
            "(DynamicShapesCodegenCpuTests and test_diagonal_scatter_backward_dynamic_shapes_cpu)",
            # Run 27246343570 default shard 7/10, job 80461205591:
            # diagonal_scatter backward dynamic-shapes CPU mismatch on ROCm
            # Inductor in the non-codegen CPU class; FAILED CONSISTENTLY.
            "(DynamicShapesCpuTests and test_diagonal_scatter_backward_dynamic_shapes_cpu)",
            # Run 27228539427 default shard 8/10:
            # test_triton_kernels expects max helper reuse removal, but ROCm
            # generated source still contains triton_helpers.max2.
            "(KernelTests and test_dim_max_min_reuse_argreduce_value)",
            # Run 27228539427 default shard 4/10:
            # log10 inductor_numerics float16 XPASSes on ROCm, tripping xfail metadata.
            "test_unary_ufunc_numerical_log10_backend_inductor_numerics_cuda_float16",
            # Run 27228539427 default shard 1/10:
            # log10 inductor_numerics float32 XPASSes on ROCm, tripping xfail metadata.
            "test_unary_ufunc_numerical_log10_backend_inductor_numerics_cuda_float32",
            # Run 27228539427 default shard 6/10:
            # log10 inductor_default float32 differs from eager under exact equality.
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float32",
            # Run 27228539427 default shard 6/10:
            # log10 inductor_default float16 differs from eager under exact equality.
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float16",
            # Run 27246343570 default shard 4/10, job 80461205582:
            # OpInfo determinism minimum under ROCm Inductor hung the GPU and
            # aborted Python before rerun/classification.
            "(TestOpInfoPropertiesCUDA and test_determinism_minimum_backend_inductor_numerics_cuda_float32)",
            # Run 27246343570 default shard 10/10, job 80461205617:
            # SaveGpuKernelSchemaTest reads disagree with saved ROCm kernel
            # schema metadata such as name, num_warps, and shared_mem.
            "(SaveGpuKernelSchemaTest and test_schema_path_reads_entry_name)",
            "(SaveGpuKernelSchemaTest and test_schema_path_reads_num_warps)",
            "(SaveGpuKernelSchemaTest and test_schema_path_reads_shared_mem)",
        ],
        "jit_fuser_te": [
            # Run 27246343570 default shard 3/10, job 80461205622:
            # TE fuser static/dynamic op tests fail consistently on ROCm,
            # mostly with bfloat16 CUDA runtime failures; one dynamic norm
            # rerun also ended in a GPU hang and Fatal Python error: Aborted.
            "(TestTEFuserStatic and test_binary_div_ops)",
            "(TestTEFuserStatic and test_binary_ops)",
            "(TestTEFuserStatic and test_binary_tensor_scalar_ops)",
            "(TestTEFuserStatic and test_ternary_norm_ops)",
            "(TestTEFuserStatic and test_ternary_ops)",
            "(TestTEFuserStatic and test_unary_ops)",
            "(TestTEFuserStatic and test_where_ops)",
            "(TestTEFuserDynamic and test_binary_div_ops)",
            "(TestTEFuserDynamic and test_binary_ops)",
            "(TestTEFuserDynamic and test_binary_tensor_scalar_ops)",
            "(TestTEFuserDynamic and test_ternary_norm_ops)",
        ],
        "ops_gradients": [
            # Run 27228539427 inductor shard 3/4:
            # test_ops_gradients aborted with ROCm GPU hang while running
            # TestBwdGradientsCUDA::test_fn_gradgrad_svd_lowrank_cuda_complex128.
            # Provisional: keep only if repeat evidence confirms this is not runner-only.
            "(TestBwdGradientsCUDA and test_fn_gradgrad_svd_lowrank_cuda_complex128)",
            # Run 27228539427 inductor shard 4/4:
            # test_ops_gradients aborted with ROCm GPU hang while running
            # TestBwdGradientsCUDA::test_fn_gradgrad_ormqr_cuda_complex128.
            # Provisional: keep only if repeat evidence confirms this is not runner-only.
            "(TestBwdGradientsCUDA and test_fn_gradgrad_ormqr_cuda_complex128)",
        ],
        "torch": [
            # Run 27238010876 inductor shard 3/4:
            # test_lognormal_kstest float16 aborts after ROCm GPU hang under
            # --inductor.
            "(TestTorchDeviceTypeCUDA and test_lognormal_kstest_cuda_float16)",
        ],
        "ops": [
            # Run 27238010876 inductor shard 4/4:
            # test_out_nn_functional_hardshrink float32 aborts after ROCm GPU
            # hang under --inductor.
            "(TestCommonCUDA and test_out_nn_functional_hardshrink_cuda_float32)",
        ],
        "privateuseone_python_backend": [
            # Run 27228539427 default shard 2/10:
            # PrivateUse1 ldexp hits missing npy DispatchStub kernel.
            "(PrivateUse1BackendTest and test_ldexp)",
        ],
        "cpp_extensions": [
            # Run 27228539427 default shard 4/10:
            # libtorch AGN 2.10 version-compatibility tests fail before
            # compile because g++ is absent.
            "(FunctionVersionCompatibilityTest and requires_2_10)",
        ],
        "utils": [
            # Run 27228539427 default shard 9/10:
            # TestStandaloneCPPJIT::test_load_standalone sees versioned
            # extension paths (`_v1`/`_v2`) while the test expects the base path.
            "(TestStandaloneCPPJIT and test_load_standalone)",
        ],
        "multiprocessing": [
            # Run 27228539427 default shard 1/10:
            # torch_shm_manager cannot load librocprofiler-sdk.so.1 in CI, so
            # file-system sharing tests fail consistently.
            "test_fs",
            "test_fs_is_shared",
            "test_fs_pool",
            "test_fs_preserve_sharing",
            "test_fs_sharing",
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
        ],
    },
}
