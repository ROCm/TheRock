# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "cuda": [
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_empty_cache_inactive",
            # RuntimeError: Error building extension 'dummy_allocator_v1'
            "test_mempool_limited_memory_with_allocator",
            # RuntimeError: Error building extension 'dummy_allocator_v3'
            "test_tensor_delete_after_allocator_delete",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_deleted_mempool_not_used_on_oom",
            # AssertionError: Scalars are not equal!
            "test_mempool_ctx_multithread",
            # Same hipblas.h compilation error as test_mempool_with_allocator.
            # See https://github.com/pytorch/pytorch/pull/173330
            "test_mempool_expandable",
            # torch.AcceleratorError: HIP error: operation not permitted when
            # stream is capturing
            "test_cuda_graph_tensor_item_not_allowed",
            # AssertionError: CalledProcessError not raised
            "test_allocator_memory_fraction_setting",
        ],
        "nn": [
            # AssertionError: False is not true : Expected NaN in pdist output
            "test_pdist_inf_nan_propagation",
            # AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
        ],
        "distributions": [
            # SIGSEGV - OpenBLAS exceeds precompiled 128-thread hard limit
            # even with OPENBLAS_NUM_THREADS=64; crash in wishart.log_prob
            "test_entropy_monte_carlo",
        ],
        "dynamo": [
            # ROCm does not support inline asm instructions
            "test_hops_compile_backend_aot_eager_inline_asm_elementwise_simple_cuda_float32",
            # CallbackTests - fails across all GPU families
            "test_triggers",
            # LoggingTests - string comparison mismatch in log output
            "test_logs_out",
        ],
        "export": [
            # ROCm does not support inline asm instructions
            "test_aot_export_inline_asm_elementwise_simple_cuda_float32",
            # subprocess exit status 127 (missing binary)
            "test_fake_export___getitem___cuda_float32",
        ],
        "inductor": [
            # BenchmarkMultiTemplateFusionGpuTest - extern code mismatch
            "test_equivalent_extern_code",
            # ComboKernelTestsPerSubkernelBlocks
            "test_activation_functions",
            # CPUReproTests - stride ordering
            "test_require_stride_order_non_owning",
            # CudaReproTests - index_add fallback path
            "test_index_add_fallback",
            # TestBlockMaskCUDA - recompilation check
            "test_compiling_create_block_mask_no_recompile_cuda",
            # InplacePaddingTest - max_autotune
            "test_linear_and_cel_max_autotune",
            # TestLookupTableE2E - addmm lookup table
            "test_bias_addmm_lookup_table_entry",
            # TestMaxAutotune - gemm choice validation
            "test_autotune_gemm_choice_validation_op_addmm_max_autotune_True",
            # TestProvenanceTracingStackTraces - deferred triton kernels
            "test_deferred_triton_kernels",
            # TestSelectAlgorithm - addmm fp16
            "test_addmm_fp16",
            # TestOpInfoPropertiesCUDA - numerical fmod
            "test_binary_ufunc_numerical_fmod_backend_inductor_default_cuda_bfloat16",
            # TestMaxAutotuneSubproc - benchmark choice assertion
            "test_benchmark_choice_in_subproc",
            # TestLearnableBiasesCUDA - LoweringException in dynamic max_autotune
            "test_flex_attention_with_dynamic_max_autotune_graph_partition_cuda",
            # ComboKernelTests - KeyError: 'grid'
            "test_combo_kernel_dynamic_shapes_grid_changes",
            # TestPatternMatcher - mm_plus_mm count mismatch
            "test_mm_plus_mm",
            # TestTemplateRender - triton_helpers.maximum not found
            "test_external_template_prologue_epilogue_fusion",
            # TestAOTInductorPackageCpp_cpu - exit status 127
            "test_compile_with_exporter",
            # TestFlexAttentionCUDA - deprecation warnings
            "test_return_aux_deprecation_warnings_cuda_float16",
            # HigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_concat_unbacked_shape_tensor",
            # AOTInductorTestABICompatibleCpuWithStackAllocation - XPASS
            "test_while_loop_with_mixed_device_dynamic_False_cpu_with_stack_allocation",
        ],
        "profiler": [
            # TestProfiler - backward compat filter
            "test_activity_filter_backward_compat",
            # TestProfiler - dict syntax filter assertion
            "test_activity_filter_dict_syntax",
        ],
        "ci_sanity_check": [
            # TestCISanityCheck - TheRock CI env differs from upstream
            "test_env_vars_exist",
        ],
        "dataloader": [
            # TestDataLoader - large sampler indices
            "test_large_sampler_indices",
            # TestDataLoader - HIP invalid device pointer in multiprocessing
            "test_multiprocessing_contexts",
        ],
        "multiprocessing": [
            # TestMultiprocessing - file system test assertion
            "test_fs",
        ],
        "multiprocessing_spawn": [
            # SpawnTest - exception handling across processes
            "test_exception_all",
        ],
        "serialization": [
            # TestSerialization - NJT weights_only import check
            "test_load_njt_weights_only_should_import_False",
        ],
        "stateless": [
            # TestStatelessDeprecation - deprecation warning mismatch
            "test_private_stateless_warns",
        ],
        "utils": [
            # TestStandaloneCPPJIT - error building extension
            "test_load_standalone",
        ],
        "distributed": [
            # ComposabilityTest - pipeline parallel (8-GPU)
            "test_replicate_pp_ScheduleClass3_bfloat16",
            # TestFSDPMemory - memory accounting mismatch
            "test_fsdp_memory_ckpt_ckpt",
            # ProcessGroupNCCLGroupTest - error code 10
            "test_resume",
            "test_get_memory_stats",
            "test_suspend",
            # TestDistBackendWithSpawn - monitored barrier hang
            "test_monitored_barrier_allreduce_hang_wait_all_ranks",
            # FSDP2 tests - 300s per-process timeout on 8-GPU runner
            "test_clip_grad_norm_1d",
            "test_compiled_autograd_fsdp2_backward",
            "test_all_gather_extensions_train_parity",
            "test_gradient_scaler",
            "test_ddp_A_fsdp_B_ddp_C",
            "test_compute_dtype",
            "test_cached_state_dict",
            "test_explicit_prefetching",
            # TestParityWithDDPCUDA - error code 10
            "test_mixture_of_experts_offload_true_shard_grad_op_cuda",
            # TestFullyShardHSDP3DTraining - error code 10
            "test_3d_mlp_with_nd_mesh",
            # DistMathOpsTest - error code 10
            "test_linalg_ops",
        ],
    },
    "gfx94": {
        "inductor": [
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd
            "test_grad_freevar_python_scalar",
            # TestCompiledAutogradOpInfoCUDA - inline asm
            "test_hops_in_bwd_inline_asm_elementwise_simple_cuda_float32",
            # TestOpInfoPropertiesCUDA - numerical XPASS failures
            "test_binary_ufunc_numerical_fmod_backend_inductor_default_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_float16",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_bfloat16",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_numerics_cuda_float32",
        ],
        "linalg": [
            # TestLinalgCUDA - tunableop_rocm addmm relu
            "test_addmm_relu_tunableop_rocm_cuda_float32",
        ],
        "scaled_matmul": [
            # TestFP8MatmulCUDA - deepseek error messages
            "test_scaled_mm_deepseek_error_messages_bfloat16_lhs_block_128_rhs_block_1_M_256_N_256_K_256_cuda",
        ],
    },
}
