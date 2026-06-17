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
            # Run 27473608564 default shard 5/10, job 81208818459 and shard 2/10, job 81208818462:
            # TestCudaAllocator::test_allocator_backend subprocess fails to launch venv python
            # (libpython3.12.so.1.0 missing) in test/test_cuda.py and test_cuda_expandable_segments.py.
            "(TestCudaAllocator and test_allocator_backend)",
            # Run 27662748552 default shard 6/10, job 81810307258:
            # test_cuda_expandable_segments.py::TestCuda::test_cudnn_multiple_threads_same_device
            # crashes consistently with pure virtual method called / SIGSEGV
            # while expandable_segments is enabled.
            "(test_cuda_expandable_segments and TestCuda and test_cudnn_multiple_threads_same_device)",
        ],
        "nn": [
            # TestNNDeviceTypeCUDA - AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
            # Run 27473608564 default shard 9/10, job 81208818474:
            # TestNNDeviceTypeCUDA::test_linear_cross_entropy_loss_default_bias_False_cuda_float32
            # input-grad ULP worst case 952 > 854 on ROCm June 12 wheel.
            "(TestNNDeviceTypeCUDA and test_linear_cross_entropy_loss_default_bias_False_cuda_float32)",
        ],
        "compiled_autograd": [
            # Run 27361388921 default shard 6/10, job 80849478677:
            # Compiled-autograd bytecode/graph text mismatches and accumulate_grad
            # signature errors on ROCm.
            "(TestCompiledAutograd and test_checkpointing_simple_reentrant)",
            "(TestCompiledAutograd and test_inputs_aliasing_bytecode_attr_mutations)",
            "(TestCompiledAutograd and test_tensor_subclass_basic)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # FuncTorch HOP compiled-autograd expecttests differ from ROCm graphs.
            "(FuncTorchHigherOrderOpTestsWithCompiledAutograd and test_hessian)",
            "(FuncTorchHigherOrderOpTestsWithCompiledAutograd and test_jacfwd)",
            "(FuncTorchHigherOrderOpTestsWithCompiledAutograd and test_jacrev)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # register_hook HOP is unsupported under compiled autograd on ROCm.
            "(TestCompiledAutogradOpInfoCUDA and test_hops_in_bwd_register_hook_simple_cuda_float32)",
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
            # Run 27361388921 default shard 3/10, job 80849478629:
            # TestControlFlow::test_scan_* parametrized variants fail consistently
            # with transposed-shape assertEqual mismatches (e.g. [6, 1] vs [1, 6]),
            # Dynamo fake-tensor size mismatches, and scan dim expand errors on ROCm.
            "(TestControlFlow and test_scan)",
        ],
        "linalg": [
            # Run 27361388921 default shard 7/10, job 80849478660:
            # TestLinalgCUDA::test_cholesky_solve_batched_many_batches_* dtypes fail
            # consistently with 6/3276800 element mismatches at batch boundary indices.
            "(TestLinalgCUDA and test_cholesky_solve_batched_many_batches)",
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
            # Run 27361388921 default shard 5/10, job 80849478569:
            # TestInductorOpInfoCUDA::test_comprehensive_new_zeros_cuda_float32 hit
            # HW Exception GPU Hang and Fatal Python error: Aborted.
            "(TestInductorOpInfoCUDA and test_comprehensive_new_zeros_cuda_float32)",
            # Run 27361388921 default shard 10/10, job 80849478637:
            # TestOpInfoPropertiesCUDA::test_batch_invariance_log1p_backend_inductor_default_cuda_float16
            # hit HW Exception GPU Hang and Fatal Python error: Aborted.
            "(TestOpInfoPropertiesCUDA and test_batch_invariance_log1p_backend_inductor_default_cuda_float16)",
            # Run 27361388921 default shard 1/10, job 80849478791:
            # TestMaxAutotuneAsyncPipelined::test_triton_error_precompilation_and_autotuning
            # failed consistently after reruns with NoValidChoicesError instead of the
            # expected ATen fallback when all simulated Triton choices fail on ROCm.
            "(TestMaxAutotuneAsyncPipelined and test_triton_error_precompilation_and_autotuning)",
            # Run 27361388921 default shard 1/10, job 80849478791:
            # DynamicShapesCodegenGPUTests::test_scalar_cpu_tensor_arg_dynamic_shapes_cuda
            # hit HW Exception GPU Hang and Fatal Python error: Aborted.
            "(DynamicShapesCodegenGPUTests and test_scalar_cpu_tensor_arg_dynamic_shapes_cuda)",
            # Run 27361388921 default shard 8/10, job 80849478727:
            # TestInductorOpInfoCUDA::test_comprehensive_nn_functional_cosine_similarity_cuda_float32
            # hit HW Exception GPU Hang and Fatal Python error: Aborted.
            "(TestInductorOpInfoCUDA and test_comprehensive_nn_functional_cosine_similarity_cuda_float32)",
            # Run 27361388921 default shard 7/10, job 80849478660:
            # TestInductorDynamicCUDA::test_embedding_backward_dynamic_shapes_large_grid_cuda
            # exceeds ROCm total-threads grid limit (2400000000 > 2147483647).
            "(TestInductorDynamicCUDA and test_embedding_backward_dynamic_shapes_large_grid_cuda)",
            # Run 27361388921 default shard 4/10, job 80849478614:
            # GPUTests::test_dropout2_cuda fails with Scalars are not equal (Expected 1 but got 0).
            "(GPUTests and test_dropout2_cuda)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # GPUTests::test_dropout3_cuda fails with Scalars are not equal (Expected 2 but got 0).
            "(GPUTests and test_dropout3_cuda)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # DynamicShapesGPUTests::test_dropout2_dynamic_shapes_cuda fails like static dropout2.
            "(DynamicShapesGPUTests and test_dropout2_dynamic_shapes_cuda)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # DynamicShapesCpuTests::test_tmp_not_defined_issue3_dynamic_shapes_cpu grad mismatch.
            "(DynamicShapesCpuTests and test_tmp_not_defined_issue3_dynamic_shapes_cpu)",
            # Run 27373187888 default shard 4/10, job 80890658592:
            # CompiledOptimizerTests::test_adam_tensor_lr_amsgrad_capturable_cuda_steplr hit
            # HW Exception GPU Hang and Fatal Python error: Aborted in
            # inductor/test_compiled_optimizers.py (shard 2/2). Intermittent on gfx942
            # (same test passed in run 27361388921 shard 4/10); provisional skip pending ROCm issue.
            "(CompiledOptimizerTests and test_adam_tensor_lr_amsgrad_capturable_cuda_steplr)",
            # Run 27420816170 default shard 2/10, job 81045685954:
            # GPUTests::test_graph_partition_refcount_cuda hit HW Exception GPU Hang in
            # inductor/test_compile_subprocess.py (~55% through module).
            "(GPUTests and test_graph_partition_refcount)",
            # Run 27373187888 default shard 3/10, job 80890658562:
            # TestInductorOpInfoCUDA::test_comprehensive_masked_mean_cuda_float32 hit
            # HW Exception GPU Hang and Fatal Python error: Aborted in
            # inductor/test_torchinductor_opinfo.py (shard 6/9). Provisional skip pending ROCm issue.
            "(TestInductorOpInfoCUDA and test_comprehensive_masked_mean_cuda_float32)",
            # Run 27473608564 default shard 1/10, job 81208818457:
            # TestGpuWrapper::test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code
            # expects at::vec:: in generated cpp wrapper; ROCm codegen omits CPU vec ISA.
            "(TestGpuWrapper and test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code)",
        ],
        "extension_backend": [
            # Run 27361388921 default shard 4/10, job 80849478614:
            # ExtensionBackendTests::test_open_device_registration expects CPU-style
            # inductor_cpp_wrapper source but ROCm generates privateuse1 AOTI glue.
            "(ExtensionBackendTests and test_open_device_registration)",
        ],
        "invoke_subgraph": [
            # Run 27361388921 default shard 4/10, job 80849478614:
            # TestInvokeSubgraphCompile::test_input_mutation_mutiple_times_fake_tensor_cahche_hit
            # fails with AttributeError: 'AutoFunctionalizedV2' object has no attribute '_schema'.
            "(TestInvokeSubgraphCompile and test_input_mutation_mutiple_times_fake_tensor_cahche_hit)",
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
            # Run 27390088455 default shard 5/10, job 80945585188:
            # Remaining TestTEFuserDynamic bfloat16 CUDA failures (static variants
            # already skipped above): lerp (ternary), lgamma (unary), where.
            "(TestTEFuserDynamic and test_ternary_ops)",
            "(TestTEFuserDynamic and test_unary_ops)",
            "(TestTEFuserDynamic and test_where_ops)",
        ],
        "control_flow": [
            # Run 27420816170 default shard 5/10, job 81045685919:
            # ScanTests::test_cond_in_scan_* parametrized variants hit HW Exception GPU
            # Hang in inductor/test_control_flow.py (~23% through module).
            "(ScanTests and test_cond_in_scan)",
        ],
        "serialization": [
            # Mirrored from pytorch_2.12.py — TestSerialization/TestOldSerialization
            # expect debug env flags set in CI; TheRock wheel CI does not enable them.
            # Run 27390088455 default shard 1/10, job 80945585178: FAILED CONSISTENTLY.
            "test_debug_set_in_ci",
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
            # Run 27433143875 default shard 2/10, job 81088448462:
            # TestTorchDeviceTypeCUDA::test_masked_scatter_large_tensor_cuda hit
            # HW Exception GPU Hang in test/test_torch.py (~63% through module).
            # Prior shard-2 hang (graph_partition_refcount) was already skipped and
            # compile_subprocess finished green; this is a separate default-lane hang.
            "(TestTorchDeviceTypeCUDA and test_masked_scatter_large_tensor_cuda)",
        ],
        "ops": [
            # Run 27238010876 inductor shard 4/4:
            # test_out_nn_functional_hardshrink float32 aborts after ROCm GPU
            # hang under --inductor.
            "(TestCommonCUDA and test_out_nn_functional_hardshrink_cuda_float32)",
            # Run 27361388921 default shard 9/10, job 80849478632:
            # TestCommonCUDA::test_dtypes_baddbmm_cuda hit HW Exception GPU Hang and
            # Fatal Python error: Aborted.
            "(TestCommonCUDA and test_dtypes_baddbmm_cuda)",
            # Run 27373187888 default shard 9/10, job 80890658618:
            # TestCompositeComplianceCUDA::test_forward_ad_nn_functional_group_norm_cuda_float32
            # hit HW Exception GPU Hang after baddbmm skip; provisional ROCm hang.
            "(TestCompositeComplianceCUDA and test_forward_ad_nn_functional_group_norm_cuda_float32)",
            # Run 27361388921 default shard 6/10, job 80849478677:
            # TestCommonCUDA::test_reduction_ops_reduce_std_cuda hits ZeroDivisionError
            # during std reduction on ROCm.
            "(TestCommonCUDA and test_reduction_ops_reduce_std_cuda)",
            # Run 27390088455 default shard 4/10, job 80945585179:
            # TestCommonCUDA::test_out__refs_atan_cuda_float32 hit HW Exception GPU Hang
            # in test/test_ops.py (~16% through ops shard 4.5); intermittent on gfx942.
            "(TestCommonCUDA and test_out__refs_atan_cuda_float32)",
        ],
        "pattern_matcher": [
            # Run 27361388921 default shard 6/10, job 80849478677:
            # TestPatternMatcher pointless-convert and unfuse-bias tests fail scalar
            # equality / pattern-match expectations on ROCm.
            "(TestPatternMatcher and test_pointless_convert_float32_float16_emulate_precision_casts_False_expected_calls_1)",
            "(TestPatternMatcher and test_unfuse_bias_addmm_half_dtypes_narrowing_cast)",
        ],
        "public_bindings": [
            # Run 27361388921 default shard 2/10, job 80849478676:
            # TestPublicBindings::test_modules_can_be_imported fails because optional
            # CUDA-only native modules nvmath and cutlass are absent on ROCm wheels.
            "(TestPublicBindings and test_modules_can_be_imported)",
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
            # Run 27480021236 distributed shard 1/3, job 81226065498:
            # ComposabilityTest::test_replicate_pp_ScheduleClass3_bfloat16 hits NCCL
            # /dev/shm exhaustion (ncclSystemError: No space left on device) under
            # 8-rank pipeline-parallel reduce-grad; recovered on subprocess retry in
            # the same job but blocks shard when /dev/shm is already warm.
            "(ComposabilityTest and test_replicate_pp_ScheduleClass3_bfloat16)",
            # Run 27480021236 distributed shard 1/3, job 81226065498:
            # TestStateDict::test_shared_weight fails consistently after reruns with
            # AssertionError: Tensor-likes are not close! on optimizer state_dict after
            # save/load with shared-weight DDP/FSDP models on June 12 ROCm wheel.
            "(TestStateDict and test_shared_weight)",
            # Run 27480021236 distributed shard 1/3, job 81226065498:
            # TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward fails with
            # Tensor-likes are not close! (10/1048576 elements; rel diff ~0.045 vs
            # 1.3e-06) at iteration 2; also failed on June 1 stack (FLAKY_TESTS.md).
            "(TestDistBackendWithSpawn and test_ddp_apply_optim_in_backward)",
            "(TestDistBackendWithSpawn and test_ddp_apply_optim_in_backward_grad_as_bucket_view_false)",
        ],
    },
}
