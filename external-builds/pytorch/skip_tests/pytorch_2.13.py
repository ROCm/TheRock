# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "cuda": [
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
            # --- Added from CI run 24859184370 ---
            # TestNNDeviceTypeCUDA: AssertionError: Tensor-likes are not close! Mismatched elements: 6474 / 48480 (13.4%) Greatest ab...
            "test_ctc_loss_cudnn_tensor_cuda_cuda",
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
            # --- Added from CI run 24893492849 ---
            # MiscTests: AssertionError: Unsupported not raised (DataPtrVariable constant comparison)
            "test_data_ptr_constant_comparison_graph_break",
            # MiscTests: Unsupported: Builtin operator.* comparison with constant `self` failed on DataPtrVariable
            "test_data_ptr_detach_equality_fullgraph",
            # --- Added from CI run 24859184370 ---
            # AutogradFunctionTests: AssertionError: RuntimeError not raised
            "test_aliased_intermediate_captured_by_side_effect",
            # DictTests: ImportError: cannot import name '_canonical_node_names' from 'torch._functorch.partitioners' (/__...
            "test_canonical_names_different_models",
            "test_canonical_names_invariant_to_dict_order",
            # EnumTests: torch._dynamo.exc.Unsupported: User-defined object with overridden __hash__ Explanation: Found a ...
            "test_dispatch_key_as_dict_key",
            # TestStreams: AssertionError graph module text mismatch
            "test_epilogue_copy_stream_tracking",
            # TestStreams: AssertionError graph module text mismatch
            "test_epilogue_copy_streams_inference",
            # TestStreams: AssertionError graph module text mismatch
            "test_event_synchronize_tracing",
            # TestStreams: AssertionError graph module text mismatch
            "test_event_tracing",
            # StructuredTraceTest: AssertionError graph module text mismatch
            "test_example_training_fn",
            # TestStreams: AssertionError graph module text mismatch
            "test_external_event_synchronize_threads_inputs",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_grad",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_grad_non_tensor_input",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_grad_over_grad",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_grad_two_tensor_all_grad_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_grad_two_tensor_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_hessian",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_hessian_argnums",
            # AutogradFunctionTests: AssertionError: Tensor-likes are not close! Mismatched elements: 8 / 8 (100.0%) Greatest absolute...
            "test_inplace_op_with_side_effect_wrong_grad",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacfwd",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacfwd_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacfwd_randomness",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacfwd_two_tensors_argnums",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacrev",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacrev_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jacrev_two_tensors_argnums",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_jvp",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_simple",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_two_tensors_disable_enable_disable_grad",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_two_tensors_disable_grad",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_jvp_two_tensors_has_aux",
            # TestStreams: AssertionError graph module text mismatch
            "test_local_stream_enter_exit",
            # TestStreams: AssertionError graph module text mismatch
            "test_local_stream_nested_enter_exit",
            # TestStreams: AssertionError graph module text mismatch
            "test_nested_stream_enter_exit",
            # ActivationCheckpointingNonStrictTracerTests: AttributeError: module 'torch.compiler' has no attribute '_patch_engine_backward'
            "test_patch_engine_backward_does_not_leak_backward_tag",
            "test_patch_engine_backward_requires_non_strict_tracing",
            # EnumTests: torch._dynamo.exc.Unsupported: User-defined object with overridden __hash__ Explanation: Found a ...
            "test_pybind11_enum_as_dict_key",
            # EnumTests: torch._dynamo.exc.Unsupported: Builtin `operator.*` comparison with constant `self` failed Explan...
            "test_pybind11_enum_equality",
            # TestStreams: AssertionError graph module text mismatch
            "test_record_stream_problem_basic",
            # TestStreams: AssertionError graph module text mismatch
            "test_record_stream_problem_interleaved",
            # ContextlibContextManagerTests: AssertionError graph module text mismatch
            "test_retrace_grad",
            # TestDefaultBackend: AttributeError: module 'torch.compiler' has no attribute 'set_default_backend'
            "test_set_default_backend",
            "test_set_default_backend_explicit_override",
            # TestStreams: AssertionError graph module text mismatch
            "test_stream_backward_simple",
            # TestStreams: AssertionError graph module text mismatch
            "test_stream_backward_sync",
            # TestStreams: AssertionError graph module text mismatch
            "test_stream_enter_exit",
            # TestStreams: AssertionError graph module text mismatch
            "test_stream_with_mutation",
            # SubclassTests: RuntimeError: prims::convert_element_type() Expected a value of type 'Tensor' for argument 'a' bu...
            "test_tensorify_under_disabled_torch_function",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_vjp",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_vjp_has_aux",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_vjp_multiple_outputs",
            # FuncTorchHigherOrderOpTests: AssertionError graph module text mismatch
            "test_vjp_multiple_outputs_python_struct",
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
            # --- Added from CI run 24859184370 ---
            # TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors",
            # CppSerdesTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_cpp_serdes",
            # RetraceExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_retraceability_nonstrict",
            # RetraceExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_retraceability_strict",
            # SerDesExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_serdes_nonstrict",
            # SerDesExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_serdes_strict",
            # StrictExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_strict",
            # StrictExportV2TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_strict_export_v2",
            # TrainingIRToRunDecompExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_training_ir_to_decomp_nonstrict",
            # TrainingIRToRunDecompExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_gradient_tracking_tensors_training_ir_to_decomp_strict",
            # TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype",
            # CppSerdesTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_cpp_serdes",
            # RetraceExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_retraceability_nonstrict",
            # RetraceExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_retraceability_strict",
            # SerDesExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_serdes_nonstrict",
            # SerDesExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_serdes_strict",
            # StrictExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_strict",
            # StrictExportV2TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_strict_export_v2",
            # TrainingIRToRunDecompExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_training_ir_to_decomp_nonstrict",
            # TrainingIRToRunDecompExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_complex_dtype_training_ir_to_decomp_strict",
            # TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops",
            # CppSerdesTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_cpp_serdes",
            # RetraceExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_retraceability_nonstrict",
            # RetraceExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_retraceability_strict",
            # SerDesExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_serdes_nonstrict",
            # SerDesExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_serdes_strict",
            # StrictExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_strict",
            # StrictExportV2TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_strict_export_v2",
            # TrainingIRToRunDecompExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_training_ir_to_decomp_nonstrict",
            # TrainingIRToRunDecompExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_inplace_ops_training_ir_to_decomp_strict",
            # TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs",
            # CppSerdesTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_cpp_serdes",
            # RetraceExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_retraceability_nonstrict",
            # RetraceExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_retraceability_strict",
            # SerDesExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_serdes_nonstrict",
            # SerDesExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_serdes_strict",
            # StrictExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_strict",
            # StrictExportV2TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_strict_export_v2",
            # TrainingIRToRunDecompExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_training_ir_to_decomp_nonstrict",
            # TrainingIRToRunDecompExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_multiple_outputs_training_ir_to_decomp_strict",
            # TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested",
            # CppSerdesTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_cpp_serdes",
            # RetraceExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_retraceability_nonstrict",
            # RetraceExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_retraceability_strict",
            # SerDesExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_serdes_nonstrict",
            # SerDesExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_serdes_strict",
            # StrictExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_strict",
            # StrictExportV2TestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_strict_export_v2",
            # TrainingIRToRunDecompExportNonStrictTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_training_ir_to_decomp_nonstrict",
            # TrainingIRToRunDecompExportTestExport: AssertionError: Expected FakeTensor, got <class 'torch.Tensor'>
            "test_jvp_export_nested_training_ir_to_decomp_strict",
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
            # --- Added from CI run 24859184370 ---
            # GPUTests: AssertionError: Tensor-likes are not close! Mismatched elements: 1 / 2 (50.0%) Greatest absolute ...
            "test_adaptive_avg_pool2d_flatten_sum_comprehensive_padding_True_cuda",
            # TestCustomOpOutLowering: AttributeError: type object 'torch._C.Tag' has no attribute 'out'
            "test_add_one_lowered_to_out_device_cpu",
            "test_add_one_lowered_to_out_device_cuda",
            # AOTInductorTestABICompatibleCpu: torch._inductor.exc.InductorError: CppCompileError: C++ compile error Command: /__w/TheRock/TheRo...
            "test_aoti_fp8_cpu",
            # ComboKernelMetadataTests: AssertionError: codegen output missing expected substring
            "test_combo_inductor_meta_has_optimize_mem",
            # ComboKernelMetadataTests: AssertionError: codegen output missing expected substring
            "test_combo_inductor_meta_optimize_mem_false_in_training_forward",
            # ComboKernelMetadataTests: AssertionError: codegen output missing expected substring
            "test_combo_triton_meta_has_disable_ftz_disable_ftz_False",
            # ComboKernelMetadataTests: AssertionError: codegen output missing expected substring
            "test_combo_triton_meta_has_disable_ftz_disable_ftz_True",
            # TestInductorOpInfoCUDA: torch._inductor.exc.InductorError: RuntimeError: CUDA driver error: 209 Set TORCHDYNAMO_VERBOSE=1...
            "test_comprehensive_randint_like_cuda_float64",
            # PaddingTest: AssertionError: Scalars are not equal! Expected 1 but got 2. Absolute difference: 1 Relative diff...
            "test_concat_output_no_redundant_copy_with_padding",
            # DynamicShapesCpuTests: torch._inductor.exc.InductorError: AssertionError: ((s77, 7, 7), (7*s77, 7)) Set TORCHDYNAMO_VERB...
            "test_group_norm_sdpa_bmm_cpu_cpp_fusion_dynamic_shapes_cpu",
            # OverFusionTest: AttributeError: torch._inductor.config.triton.mix_order_reduction_max_reads does not exist
            "test_max_reads_limits_fusion",
            # TestCustomOpOutLowering: AttributeError: type object 'torch._C.Tag' has no attribute 'out'
            "test_multi_output_lowered_to_out_device_cpu",
            "test_multi_output_lowered_to_out_device_cuda",
            # TestUnbackedSymintsCUDA: torch._dynamo.exc.Unsupported: Observed exception Explanation: Dynamo found no exception handler ...
            "test_override_optimization_hint_compiled_cuda",
            "test_override_optimization_hint_compiled_tolist_cuda",
            # TestUnbackedSymintsCUDA: AttributeError: module 'torch._dynamo' has no attribute 'override_optimization_hint'
            "test_override_optimization_hint_concrete_int_noop_cuda",
            "test_override_optimization_hint_eager_cuda",
            # TestUnbackedSymintsCUDA: torch._dynamo.exc.Unsupported: Observed exception Explanation: Dynamo found no exception handler ...
            "test_override_optimization_hint_multiple_items_cuda",
            "test_override_optimization_hint_rejects_backed_symbol_cuda",
            # TestUnbackedSymintsCUDA: AttributeError: module 'torch._dynamo' has no attribute 'override_optimization_hint'
            "test_override_optimization_hint_rejects_derived_expression_cuda",
            "test_override_optimization_hint_rejects_non_int_val_cuda",
            "test_override_optimization_hint_rejects_wrong_type_cuda",
            # DeterministicTest: AttributeError: torch._inductor.config.batch_invariant does not exist
            "test_persistent_reduction_batch_invariance",
            # PaddingTest: AssertionError: Tensor-likes are not close! Mismatched elements: 3 / 4 (75.0%) Greatest absolute ...
            "test_reduction_comprehensive_padding_stride",
            # AOTInductorTestABICompatibleGpu: AssertionError: Tensor-likes are not close! Mismatched elements: 2 / 4 (50.0%) Greatest absolute ...
            "test_update_inactive_constant_buffer_with_interleaved_folded_constants_cuda",
            # --- Added from CI run 24871076312 ---
            # AOTInductorTestABICompatibleCpu: AssertionError: Tensor-likes are not close! Mismatched elements: 2 / 4 (50.0%) (CPU variant)
            "test_update_inactive_constant_buffer_with_interleaved_folded_constants_cpu",
            # --- Added from CI run 24893492849 ---
            # TestFastCudaLauncherCompileResult: AttributeError: torch._inductor.config.use_fast_triton_launcher does not exist
            # (missing upstream config attribute in installed torch wheel)
            "test_basic_compile",
            "test_disable_fast_launcher",
            # ExtensionBackendTests: torch._dynamo.exc.InternalTorchDynamoError: AttributeError: module 'extension_device' has no attribute 'is_available'.
            "test_open_device_registration",
        ],
        "linalg": [
            # TestLinalgCUDA - tunableop_rocm addmm relu
            "test_addmm_relu_tunableop_rocm_cuda_float32",
        ],
        "scaled_matmul": [
            # TestFP8MatmulCUDA - deepseek error messages
            "test_scaled_mm_deepseek_error_messages_bfloat16_lhs_block_128_rhs_block_1_M_256_N_256_K_256_cuda",
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
            # --- Added from CI run 24859184370 ---
            # TestCodegenOutputAlias: AssertionError: Scalars are not equal! Expected 1 but got 0. Absolute difference: 1 Relative diff...
            "test_alias_of_intermediate_save_as_output",
            "test_codegen_source_contains_gen_alias",
            "test_codegen_source_noop_handler",
            # TestAcLogging: TypeError: create_activation_checkpointing_logging_structure_payload() got an unexpected keyword ...
            "test_create_activation_checkpointing_logging_structure_payload",
            # TestAcLogging: TypeError: create_structured_trace_for_min_cut_info() got an unexpected keyword argument 'memory_...
            "test_create_structured_trace_for_min_cut_info",
            # TestCodegenOutputAlias: AssertionError: Scalars are not equal! Expected 1 but got 0. Absolute difference: 1 Relative diff...
            "test_cross_dtype_view_alias",
            "test_mixed_alias_and_non_alias_outputs",
            "test_multiple_views_of_same_input",
            # TestVmapOperatorsOpInfoCUDA: RuntimeError: Batching rule not implemented for aten::unbind_copy.int. We could not generate a fa...
            "test_op_has_batch_rule_unbind_copy_cuda_float32",
            # TestCodegenOutputAlias: AssertionError: Scalars are not equal! Expected 1 but got 0. Absolute difference: 1 Relative diff...
            "test_output_alias_with_mutation",
            "test_output_aliases_intermediate",
            "test_output_is_input",
            "test_output_is_view_of_input",
            "test_training_path_alias_of_intermediate_detach",
            "test_training_path_is_input",
            "test_training_path_mixed_requires_grad",
            "test_training_path_mutation_and_alias",
            "test_training_path_view_of_input",
            "test_view_replay_config_false",
            "test_view_replay_config_true",
            # TestOperatorsCUDA: RuntimeError: Batching rule not implemented for aten::unbind_copy.int. We could not generate a fa...
            "test_vjpvmap_unbind_copy_cuda_float32",
            # TestVmapOperatorsOpInfoCUDA: RuntimeError: Batching rule not implemented for aten::unbind_copy.int. We could not generate a fa...
            "test_vmap_exhaustive_unbind_copy_cuda_float32",
            # TestOperatorsCUDA: RuntimeError: Batching rule not implemented for aten::unbind_copy.int. We could not generate a fa...
            "test_vmapjvpall_has_batch_rule_unbind_copy_cuda_float32",
            "test_vmapjvpall_unbind_copy_cuda_float32",
            "test_vmapjvpvjp_unbind_copy_cuda_float32",
            "test_vmapvjp_has_batch_rule_unbind_copy_cuda_float32",
            "test_vmapvjp_unbind_copy_cuda_float32",
            "test_vmapvjpvjp_unbind_copy_cuda_float32",
            # TestCodegenOutputAlias: AssertionError: Scalars are not equal! Expected 1 but got 0. Absolute difference: 1 Relative diff...
            "test_xform_aliased_output_tensoralias_wrapping",
            "test_xform_metadata_only_mutation",
            "test_xform_unsafe_view_output",
        ],
        "utils": [
            # TestStandaloneCPPJIT - error building extension
            "test_load_standalone",
        ],
        "distributed": [
            # ComposabilityTest - pipeline parallel (8-GPU)
            "test_replicate_pp_ScheduleClass3_bfloat16",
            "test_replicate_pp_ScheduleClass3_float32",
            "test_replicate_pp_ScheduleClass4_bfloat16",
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_device_id",
            # ProcessGroupNCCLGroupTest - error code 10
            "test_resume",
            "test_get_memory_stats",
            "test_suspend",
            # TestDistBackendWithSpawn - monitored barrier hang
            "test_monitored_barrier_allreduce_hang_wait_all_ranks",
            "test_monitored_barrier_allreduce_hang",
            # FSDP2 tests - 300s per-process timeout on 8-GPU runner
            "test_clip_grad_norm_1d",
            "test_clip_grad_norm_2d",
            "test_compiled_autograd_fsdp2_backward",
            "test_all_gather_extensions_train_parity",
            "test_gradient_scaler",
            "test_ddp_A_fsdp_B_ddp_C",
            "test_compute_dtype",
            "test_cached_state_dict",
            "test_explicit_prefetching",
            # DistMathOpsTest - RuntimeError: Calling torch.linalg.eig with MAGMA requires
            # compiling PyTorch with MAGMA. TODO: Revisit once MAGMA is supported in TheRock
            "test_linalg_ops",
            # TestFullyShardCommunication - communication count mismatch
            "test_fully_shard_communication_count",
            # TestFullyShardAutograd - 300s per-process timeout
            "test_nontensor_activations",
            # TestFullyShardDTensor - 300s per-process timeout
            "test_dtensor_train_parity",
            # TestFullyShardFrozen - 300s per-process timeout
            "test_multi_forward_mixed_requires_grad",
            # TestFullyShardMemory - 300s per-process timeout
            "test_fully_shard_training_memory",
            # TestFullyShardOverlap - 300s per-process timeout
            "test_fully_shard_training_overlap",
            # TestFullyShard2DTraining - pytest-timeout (>900s)
            "test_tp_with_fsdp_offloading",
            # ComposabilityTest - pytest-timeout (>900s)
            "test_3d_with_tp_dp_pp_ScheduleClass0_bfloat16",
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_fully_shard_init",
            # TestReplicateMixedPrecisionTraining - pytest-timeout (>900s)
            "test_grad_acc_with_reduce_dtype",
            # TestReplicate1DTrainingCore - 300s per-process timeout
            "test_multi_forward_module",
            # ComposabilityTest - pipeline parallel RuntimeError
            "test_replicate_pp_grads_ScheduleClass1",
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_ignore_module",
            # TestFullyShardCommunication - 300s per-process timeout
            "test_manual_reshard_with_reshard_after_forward_false",
            # TestFullyShardSharedParams - 300s per-process timeout
            "test_train_parity_with_shared_params",
            # ReplicateTest - pytest-timeout (>900s)
            "test_train_parity_2d_mlp",
            # ComposabilityTest - pipeline parallel RuntimeError
            "test_replicate_pp_grads_ScheduleClass2",
            # ComposabilityTest - pipeline parallel RuntimeError
            "test_replicate_pp_grads_ScheduleClass3",
            # ComposabilityTest - pipeline parallel RuntimeError
            "test_replicate_pp_grads_ScheduleClass4",
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_move_args_kwargs_to_device",
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_single_module",
            # TestFullyShardAutograd - 300s per-process timeout
            "test_unused_forward_module",
            # --- Added from CI run 24859184370 ---
            # TestNewEmptyStridedUneven: RuntimeError: Process 2 exited with error code 10 and exception: Traceback (most recent call last...
            "test_backward_partial_grad_with_transpose",
            # TestZeroRedundancyOptimizerDistributed: RuntimeError: Process 1 exited with error code 10 and exception: Traceback (most recent call last...
            "test_ddp_zero_overlap_use_gpu_True_use_interleaved_hook_False_gradient_as_bucket_view_True_static_graph_True_shard_buckets_True",
            # ProcessGroupNCCLGroupTest: RuntimeError: Process 0 exited with error code 10 and exception: Traceback (most recent call last...
            "test_extra_cuda_context",
            # DistMathOpsTest: RuntimeError: Process 3 exited with error code 10 and exception: Traceback (most recent call last...
            "test_interpolation_upsample_ops",
            # TestNewEmptyStridedUneven: RuntimeError: Process 0 exited with error code 10 and exception: Traceback (most recent call last...
            "test_new_empty_propagates_partial",
            # TestDTensorCompile: AssertionError: True is not false : pad_tensor created a guard that concretized the symbolic dim:...
            "test_pad_tensor_no_guard_on_symbolic_pad_size",
            # TestSyncDecisionCrossRanks: RuntimeError: Process 0 exited with error code 10 and exception: Traceback (most recent call last...
            "test_sync_decision_cross_ranks_different_node_order",
            # TestSyncDecisionCrossRanks: RuntimeError: Process 1 exited with error code 10 and exception: Traceback (most recent call last...
            "test_sync_decision_cross_ranks_invalid_node_error",
            # --- Added from CI run 24871076312 ---
            # DistTensorRandomInitTest: RuntimeError: Process exited with error code 10
            "test_meta_tensor_init",
            # TestFullyShardCustomForwardMethod: RuntimeError: Process exited with error code 10
            "test_register_fsdp_forward_method",
            # TestFullyShardHSDP3DTraining: all 8 worker processes exit with error
            # code 10.
            "test_3d_mlp_with_nd_mesh",
            # TestMultiProc: worker processes exit with error code 10.
            "test_compiler_collectives_automatic_dynamic_tensor",
        ],
        "autograd": [
            # --- Added from CI run 24871076312 ---
            # TestAutogradDeviceTypeCUDA: AssertionError: "Simulate error" does not match
            # "grad can be implicitly created only for scalar outputs"
            # (was previously only in windows section of generic.py; now also fails on gfx94X Linux)
            "test_reentrant_parent_error_on_cpu_cuda",
            # TestSelectiveActivationCheckpoint: AssertionError: Scalars are not equal! Expected 1 but got 2. Absolute difference: 1 Relative diff...
            "test_auto_naming_mode_names",
            # TestAutogradDeviceTypeCUDA: AttributeError: module 'torch.autograd' has no attribute 'enforce_grad_layout_policy'
            "test_enforce_grad_layout_policy_cuda",
            # TestSelectiveActivationCheckpoint: AssertionError: Scalars are not equal! Expected 1 but got 2. Absolute difference: 1 Relative diff...
            "test_function_with_more_than_one_output",
            # TestSelectiveActivationCheckpoint: AssertionError: 'invocation index .* not found in storage' does not match 'Trying to backward an ...
            "test_mismatch_extra_invocation_during_recompute",
            # TestSelectiveActivationCheckpoint: AssertionError: 'not found in storage' does not match 'torch.utils.checkpoint: trying to save mor...
            "test_mismatch_new_op_during_recompute",
            # TestSelectiveActivationCheckpoint: AssertionError: The length of the sequences mismatch: 5 != 3
            "test_policy_with_state",
        ],
        "cpp_extensions": [
            # FunctionVersionCompatibilityTest: FileNotFoundError: [Errno 2] No such file or directory: 'g++'
            "test_get_any_data_ptr_requires_2_10",
            "test_get_template_any_data_ptr_requires_2_10",
            "test_make_tensor_clones_and_call_foreach_requires_2_10",
            "test_my__foreach_mul__requires_2_10",
            "test_my__foreach_mul_requires_2_10",
            "test_my__foreach_mul_vec_requires_2_10",
            "test_my_contiguous_requires_2_10",
            "test_my_empty_requires_2_10",
            "test_my_from_blob_requires_2_10",
            "test_my_full_requires_2_10",
            "test_my_new_empty_zeros_requires_2_10",
            "test_my_reshape_requires_2_10",
            "test_my_set_requires_grad_requires_2_10",
            "test_my_shape_requires_2_10",
            "test_my_string_op_requires_2_10",
            "test_my_string_op_variants_requires_2_10",
            "test_my_subtract_requires_2_10",
            "test_my_sum_out_requires_2_10",
            "test_my_sum_requires_2_10",
            "test_my_to_requires_2_10",
            "test_my_view_requires_2_10",
            "test_test_device_constructor_requires_2_10",
            "test_test_device_equality_requires_2_10",
            "test_test_device_index_requires_2_10",
            "test_test_device_is_cpu_requires_2_10",
            "test_test_device_is_cuda_requires_2_10",
            "test_test_device_set_index_requires_2_10",
            "test_test_get_num_threads_requires_2_10",
            "test_test_parallel_for_requires_2_10",
            "test_test_tensor_device_requires_2_10",
        ],
        "custom_ops": [
            # TestCustomOpAPI: IndexError: tuple index out of range
            "test_mutated_optional_arg_default_none",
        ],
        "dynamic_shapes": [
            # TestMaybeFastEvalComparison: torch._dynamo.exc.UserError: Could not guard on data-dependent expression Eq(u4, u2) (unhinted: E...
            "test_unbacked_slice_assignment_same_bounds",
        ],
        "fake_tensor": [
            # FakeTensorConverterTest: AssertionError: Object comparison failed: torch.bfloat16 != torch.float32
            "test_grad_dtype_functional_tensor_no_crash",
            # PropagateRealTensorsFakeTensorConverterTest: AssertionError: Object comparison failed: torch.bfloat16 != torch.float32
            "test_grad_dtype_functional_tensor_no_crash_propagate_real_tensors",
            # FakeTensorConverterTest: AssertionError: Object comparison failed: torch.float32 != torch.bfloat16
            "test_grad_dtype_make_fx",
            # PropagateRealTensorsFakeTensorConverterTest: AssertionError: Object comparison failed: torch.float32 != torch.bfloat16
            "test_grad_dtype_make_fx_propagate_real_tensors",
            # FakeTensorConverterTest: AssertionError: torch.bfloat16 is not None
            "test_grad_dtype_none_preserved",
            # PropagateRealTensorsFakeTensorConverterTest: AssertionError: torch.bfloat16 is not None
            "test_grad_dtype_none_preserved_propagate_real_tensors",
            # FakeTensorConverterTest: AssertionError: Object comparison failed: torch.bfloat16 != torch.float32
            "test_grad_dtype_preserved",
            # PropagateRealTensorsFakeTensorConverterTest: AssertionError: Object comparison failed: torch.bfloat16 != torch.float32
            "test_grad_dtype_preserved_propagate_real_tensors",
        ],
        "higher_order_ops": [
            # TestInvokeSubgraphCompile: AssertionError: 'aliases an input or output.*clone' does not match 'RuntimeError: Argument 'view'...
            "test_side_effect_with_aliased_intermediate",
        ],
        "testing": [
            # --- Added from CI run 24871076312 ---
            # TestImports: RuntimeError: Failed to import torch._inductor.mkldnn_lowerings:
            # partially initialized module has no attribute 'register_onednn_fusion_ops'
            # (circular import)
            "test_circular_dependencies",
        ],
    },
}
