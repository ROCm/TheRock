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
        ],
        "export": [
            # ROCm does not support inline asm instructions
            "test_aot_export_inline_asm_elementwise_simple_cuda_float32",
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
        ],
        "profiler": [
            # TestProfiler - backward compat filter
            "test_activity_filter_backward_compat",
        ],
        "ci_sanity_check": [
            # TestCISanityCheck - TheRock CI env differs from upstream
            "test_env_vars_exist",
        ],
        "dataloader": [
            # TestDataLoader - large sampler indices
            "test_large_sampler_indices",
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
    },
    "gfx94": {
        "inductor": [
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd
            "test_grad_freevar_python_scalar",
            # TestCompiledAutogradOpInfoCUDA - inline asm
            "test_hops_in_bwd_inline_asm_elementwise_simple_cuda_float32",
            # TestOpInfoPropertiesCUDA - additional numerical failures
            "test_binary_ufunc_numerical_fmod_backend_inductor_default_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_float16",
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
    "gfx90": {
        "inductor": [
            # TestFlexAttentionCUDA - deprecation warnings (gfx90a only)
            "test_return_aux_deprecation_warnings_cuda_float16",
            # HigherOrderOpTestsWithCompiledAutograd
            "test_concat_unbacked_shape_tensor",
        ],
    },
}
