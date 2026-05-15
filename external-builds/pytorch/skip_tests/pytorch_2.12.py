# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "cuda": [
            # RuntimeError: Error building extension 'dummy_allocator_v1'
            "test_mempool_limited_memory_with_allocator",
            # AssertionError: Scalars are not equal!
            "test_mempool_ctx_multithread",
            # torch.AcceleratorError: HIP error: operation not permitted when
            # stream is capturing
            "test_cuda_graph_tensor_item_not_allowed",
            # AssertionError: CalledProcessError not raised
            "test_allocator_memory_fraction_setting",
            # TestBlockStateAbsorption - ModuleNotFoundError: torchvision
            "test_resnet",
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
            # subprocess.CalledProcessError in batch_norm export
            "test_fake_export_nn_functional_batch_norm_cuda_float32",
            # subprocess.CalledProcessError (batch_norm without cudnn path)
            "test_fake_export_nn_functional_batch_norm_without_cudnn_cuda_float32",
            # subprocess.CalledProcessError in conv2d export
            "test_fake_export_nn_functional_conv2d_cuda_float32",
            # subprocess.CalledProcessError in instance_norm export
            "test_fake_export_nn_functional_instance_norm_cuda_float32",
            # subprocess.CalledProcessError in multi_margin_loss export
            "test_fake_export_nn_functional_multi_margin_loss_cuda_float32",
            # subprocess.CalledProcessError in scaled_dot_product_attention export
            "test_fake_export_nn_functional_scaled_dot_product_attention_cuda_float32",
            # subprocess.CalledProcessError in nonzero export
            "test_fake_export_nonzero_cuda_float32",
            # TestExportOnFakeCudaCUDA - preserve original behavior
            "test_preserve_original_behavior_cuda",
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
            # TestPrologueFusion - expected fused kernel name substring missing
            "test_lazy_template_fusion_multiple_candidates_use_async_compile_False",
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
            "test_combo_kernel_yz_overflow",
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
            "test_hints_wrapper",
            # AOTInductorTestABICompatibleCpuWithStackAllocation - XPASS
            "test_while_loop_with_mixed_device_dynamic_False_cpu_with_stack_allocation",
            "test_while_loop_with_mixed_device_dynamic_True_cpu_with_stack_allocation",
            "test_while_loop_with_pytree_inputs_cpu_with_stack_allocation",
            # TestMaxAutotunePrecompile - no kernel image available
            "test_filled_cache_precompile",
            # TestOpInfoPropertiesCUDA - eager equivalence mismatch
            "test_eager_equivalence_exp_backend_inductor_default_cuda_float32",
            "test_eager_equivalence_rsqrt_backend_inductor_numerics_cuda_float32",
            "test_eager_equivalence_log_backend_inductor_default_cuda_float16",
            # TestFxGraphCache - remote cache stats mismatch
            "test_remote_cache_load_function_device_cuda_float32_dynamic_False_bundle_triton_False_use_static_triton_launcher_False",
            # TestOpInfoPropertiesCUDA - determinism mismatch
            "test_determinism_log1p_backend_inductor_default_cuda_float32",
            # ComboKernelTests - KeyError: 'grid'
            "test_combo_kernel_per_config_subkernel_block_size",
            # HigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_cond_branches_no_arguments_no_closure",
            # TestLearnableBiasesCUDA - flex attention log file not created
            "test_flex_attention_logging_cuda",
            # TestTemplateConfigPruning - NoValidChoicesError / no kernel image
            "test_shared_memory_pruning_mm_float32_mat1_transposed_False_mat2_transposed_True_use_tma_False",
            "test_shared_memory_pruning_mm_float32_mat1_transposed_True_mat2_transposed_False_use_tma_False",
            "test_shared_memory_pruning_mm_float32_mat1_transposed_True_mat2_transposed_True_use_tma_False",
            # TestEpilogueFusionStaticAnalysis - expected fused kernel name not found
            "test_template_epilogue_fusion_static_analysis_test_case_timing_reject_use_async_compile_False",
            "test_template_epilogue_fusion_static_analysis_test_case_timing_reject_use_async_compile_True",
            # TestOpInfoPropertiesCUDA - numerical cos mismatch
            "test_unary_ufunc_numerical_cos_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - XPASS reciprocal
            "test_eager_equivalence_reciprocal_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - numerical log10 mismatch
            "test_eager_equivalence_log10_backend_inductor_default_cuda_float32",
            # HigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_tensor_and_unbacked_symbol_closure",
            "test_tensor_to_list_closure",
            # TestOpInfoPropertiesCUDA - numerical log1p mismatch
            "test_eager_equivalence_log1p_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - XPASS remainder
            "test_eager_equivalence_remainder_backend_inductor_numerics_cuda_bfloat16",
            # TestOpInfoPropertiesCUDA - XPASS exp2
            "test_unary_ufunc_numerical_exp2_backend_inductor_default_cuda_bfloat16",
            # TestOpInfoPropertiesCUDA - eager equivalence log mismatch
            "test_eager_equivalence_log_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - XPASS remainder float16
            "test_eager_equivalence_remainder_backend_inductor_numerics_cuda_float16",
            # TestOpInfoPropertiesCUDA - numerical expm1 mismatch
            "test_unary_ufunc_numerical_expm1_backend_inductor_default_cuda_float32",
            # TestMaxAutotuneAsyncPipelined - cache same inputs assertion
            "test_async_autotuner_cache_same_inputs",
            # TestMaxAutotuneAsyncPipelined - compilation after inactivity
            "test_compilation_after_inactivity",
            # HigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_tensor_with_unbacked_shape_closure",
            # TestMaxAutotuneAsyncPipelined - NoValidChoicesError bmm
            "test_bmm_out_dtype",
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_functional_call",
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_grad_has_aux",
            # TestOpInfoPropertiesCUDA - XPASS remainder bfloat16
            "test_eager_equivalence_remainder_backend_inductor_default_cuda_bfloat16",
            # TestOpInfoPropertiesCUDA - XPASS remainder float32
            "test_eager_equivalence_remainder_backend_inductor_numerics_cuda_float32",
            # TestOpInfoPropertiesCUDA - numerical log10 mismatch
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float32",
            # TestMaxAutotuneAsyncPipelined - cat extern code assertion
            "test_cat_max_autotune_extern",
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd - _GeneratorContextManager
            "test_grad_capture_tensor",
            # TestOpInfoPropertiesCUDA - XPASS remainder float16
            "test_eager_equivalence_remainder_backend_inductor_default_cuda_float16",
            # TestOpInfoPropertiesCUDA - rsqrt numerical mismatch
            "test_eager_equivalence_rsqrt_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - log numerical mismatch bfloat16
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_bfloat16",
            # TestOpInfoPropertiesCUDA - numerical exp2 float32 XPASS
            "test_unary_ufunc_numerical_exp2_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - eager equivalence remainder float32
            "test_eager_equivalence_remainder_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - numerical log float16 mismatch
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_float16",
            # TestOpInfoPropertiesCUDA - numerical sqrt bfloat16 mismatch
            "test_unary_ufunc_numerical_sqrt_backend_inductor_default_cuda_bfloat16",
            # TestOpInfoPropertiesCUDA - numerical exp float32 mismatch
            "test_unary_ufunc_numerical_exp_backend_inductor_default_cuda_float32",
            # TestOpInfoPropertiesCUDA - eager equivalence tanh float32 mismatch
            "test_eager_equivalence_tanh_backend_inductor_default_cuda_float32",
            # TestPrologueFusion - async compile variant
            "test_lazy_template_fusion_multiple_candidates_use_async_compile_True",
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd - grad closure scalar
            "test_grad_closure_scalar",
            # FuncTorchHigherOrderOpTestsWithCompiledAutograd
            "test_grad_freevar_python_scalar",
            # TestCompiledAutogradOpInfoCUDA - inline asm
            "test_hops_in_bwd_inline_asm_elementwise_simple_cuda_float32",
            # TestOpInfoPropertiesCUDA - numerical fmod/remainder XPASS
            "test_binary_ufunc_numerical_fmod_backend_inductor_default_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_float16",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_bfloat16",
            "test_binary_ufunc_numerical_remainder_backend_inductor_default_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_numerics_cuda_float32",
            "test_binary_ufunc_numerical_remainder_backend_inductor_numerics_cuda_bfloat16",
            "test_binary_ufunc_numerical_remainder_backend_inductor_numerics_cuda_float16",
        ],
        "linalg": [
            # TestLinalgCUDA - tunableop_rocm addmm relu
            "test_addmm_relu_tunableop_rocm_cuda_float32",
        ],
        "modules": [
            # TestModuleCUDA - CTCLoss cpu/gpu parity scalar mismatch
            "test_cpu_gpu_parity_nn_CTCLoss_cuda_float32",
            # TestModuleCUDA - CTCLoss forward scalar mismatch
            "test_forward_nn_CTCLoss_cuda_float32",
        ],
        "profiler": [
            # TestProfiler - backward compat filter
            "test_activity_filter_backward_compat",
            # TestProfiler - dict syntax filter assertion
            "test_activity_filter_dict_syntax",
            # TestProfiler - kineto kernel metadata missing 'grid'
            "test_kineto_kernel_metadata_in_trace",
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
            # TestDataLoaderDeviceTypeCUDA - worker exited unexpectedly
            "test_nested_tensor_multiprocessing_context_forkserver_cuda",
            "test_nested_tensor_multiprocessing_context_spawn_cuda",
            # TestDataLoaderDeviceTypeCUDA - sparse tensor worker exited unexpectedly
            "test_sparse_tensor_multiprocessing_context_forkserver_cuda",
            "test_sparse_tensor_multiprocessing_context_spawn_cuda",
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
            # TestOldSerialization - CI env assertion
            "test_debug_set_in_ci",
        ],
        "stateless": [
            # TestStatelessDeprecation - deprecation warning mismatch
            "test_private_stateless_warns",
        ],
        "functorch": [
            # TestOperatorsCUDA - conv3d numerical mismatch
            "test_grad_nn_functional_conv3d_cuda_float32",
        ],
        "utils": [
            # TestStandaloneCPPJIT - error building extension
            "test_load_standalone",
        ],
        "distributed": [
            # === Timeouts, hangs, NCCL / spawn watchdogs (FSDP2, fully_shard, compile+dist,
            #     pytest >900s, monitored barrier, Join stalls, classic FSDP wrap / NCCL abort) ===
            # FSDP2 — ~300s per-process watchdog
            "test_clip_grad_norm_1d",
            "test_clip_grad_norm_2d",
            "test_compiled_autograd_fsdp2_backward",
            "test_all_gather_extensions_train_parity",
            "test_gradient_scaler",
            "test_ddp_A_fsdp_B_ddp_C",
            "test_compute_dtype",
            "test_cached_state_dict",
            "test_explicit_prefetching",
            # Fully Shard — ~300s per-process watchdog (autograd / DTensor / memory / overlap)
            "test_nontensor_activations",
            "test_dtensor_train_parity",
            "test_multi_forward_mixed_requires_grad",
            "test_fully_shard_training_memory",
            "test_fully_shard_training_overlap",
            "test_multi_forward_module",
            "test_manual_reshard_with_reshard_after_forward_false",
            "test_train_parity_with_shared_params",
            "test_unused_forward_module",
            # Fully Shard / comm / prefetch (distributed CI) — ~300s per-process watchdog
            "test_2d_mlp_with_nd_mesh",
            "test_MLPStacked_distributed_sharding_display",
            "test_backward_misprefetch",
            "test_double_forward_with_nested_fsdp_and_checkpoint",
            "test_fully_shard_backward_prefetch",
            "test_fully_shard_force_sum_both_reductions",
            "test_fully_shard_force_sum_reduce_scatter",
            "test_fully_shard_multi_module_backward_prefetch",
            "test_fully_shard_per_param_mesh_training_overlap",
            "test_gradient_accumulation",
            "test_layer_by_layer_shard_no_false_positive",
            "test_post_optim_event",
            "test_reduce_dtype",
            "test_set_modules_to_backward_prefetch",
            "test_set_modules_to_backward_prefetch_inside_ac",
            "test_set_modules_to_forward_prefetch",
            "test_set_reduce_scatter_divide_factor",
            "test_shard_placement_fn_tp_ep",
            "test_train_mixed_requires_grad_across_groups",
            "test_train_mixed_requires_grad_per_group",
            "test_train_parity_hsdp",
            "test_train_parity_shard_placement_fn_shard_largest_dim",
            "test_train_parity_with_activation_checkpointing",
            "test_unshard_async",
            # Fully Shard mixed precision — ~300s watchdog (CI also reported exit code 10)
            "test_structured_input_output",
            # Pytest-timeout (>900s) — replicate, composability, 2D offload
            "test_replicate_device_id",
            "test_tp_with_fsdp_offloading",
            "test_3d_with_tp_dp_pp_ScheduleClass0_bfloat16",
            "test_replicate_fully_shard_init",
            "test_grad_acc_with_reduce_dtype",
            "test_replicate_ignore_module",
            "test_train_parity_2d_mlp",
            "test_replicate_move_args_kwargs_to_device",
            "test_replicate_single_module",
            # torch.compile + 2D FSDP/TP — ~300s per-process watchdog
            "test_2d_fsdp_tp_compile_use_ca_False",
            # Monitored barrier — ranks hang (no completion)
            "test_monitored_barrier_allreduce_hang_wait_all_ranks",
            "test_monitored_barrier_allreduce_hang",
            # Join — spawn timeouts (200–300s) and/or exit code 10
            "test_join_kwargs",
            "test_multiple_joinables",
            # FSDP1 wrap — ProcessGroupNCCL _ALLGATHER_BASE watchdog; child Exited with -6
            "test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode0",
            "test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_False_device_init_mode1",
            "test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_True_device_init_mode0",
            "test_main_wrap_api_cpu_offload0_backward_prefetch0_forward_prefetch_True_device_init_mode1",
            "test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_False_device_init_mode0",
            "test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_False_device_init_mode1",
            "test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_True_device_init_mode1",
            "test_main_wrap_api_cpu_offload1_backward_prefetch0_forward_prefetch_False_device_init_mode1",
            "test_main_wrap_api_cpu_offload1_backward_prefetch0_forward_prefetch_True_device_init_mode1",
            "test_main_wrap_api_cpu_offload1_backward_prefetch1_forward_prefetch_False_device_init_mode1",
            "test_main_wrap_api_cpu_offload1_backward_prefetch1_forward_prefetch_True_device_init_mode1",
            # === Subprocess exit code 10 (spawned rank crash) ===
            "test_resume",
            "test_get_memory_stats",
            "test_suspend",
            "test_mixture_of_experts_offload_true_shard_grad_op_cuda",
            "test_3d_mlp_with_nd_mesh",
            "test_linalg_ops",
            "test_ddp_uneven_inputs",
            "test_non_root_forward_backward",
            "test_reduce_scatter_uneven",
            "test_single_joinable",
            # === Assertions / numeric mismatch ===
            "test_depthwise_convolution",
            "test_ddp_buffer_hook_allreduce_return_future",
            # === Communication count, parity, optimizer reload, compiler collectives ===
            "test_post_localSGD_optimizer_step_reload",
            "test_fully_shard_communication_count",
            "test_train_parity_single_group_shard_largest_dim",
            "test_compiler_collectives_automatic_dynamic_tensor",
            # === Pipeline parallel composability (8-GPU or PP RuntimeError) ===
            "test_replicate_pp_ScheduleClass3_bfloat16",
            "test_replicate_pp_ScheduleClass3_float32",
            "test_replicate_pp_ScheduleClass4_bfloat16",
            "test_replicate_pp_grads_ScheduleClass1",
            "test_replicate_pp_grads_ScheduleClass2",
            "test_replicate_pp_grads_ScheduleClass3",
            "test_replicate_pp_grads_ScheduleClass4",
            # === Memory accounting, init failure, elastic, misc ===
            "test_fsdp_memory_ckpt_ckpt",
            "test_replicate_multi_module",
            "test_virtual_local_rank",
            "test_view_ops",
        ],

    },
}
