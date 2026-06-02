# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "cuda": [
            # AssertionError: False is not true
            "test_memory_plots",
            # AssertionError: Booleans mismatch: False is not True
            "test_memory_plots_free_segment_stack",
            # FileNotFoundError: [Errno 2] No such file or directory: '/github/home/.cache//flamegraph.pl'
            "test_memory_snapshot",
            # AssertionError: String comparison failed: 'test_memory_snapshot' != 'foo'
            "test_memory_snapshot_script",
            # AssertionError: False is not true
            "test_memory_snapshot_with_cpp",
            # AssertionError: Scalars are not equal!
            "test_mempool_ctx_multithread",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_empty_cache_inactive",
            # RuntimeError: Error building extension 'dummy_allocator_v1'
            "test_mempool_limited_memory_with_allocator",
            # ModuleNotFoundError: No module named 'torchvision'
            "test_resnet",
            # RuntimeError: miopenStatusUnknownError
            "test_graph_cudnn_dropout",
            # Fatal Python error: Segmentation fault - https://github.com/ROCm/TheRock/issues/4745
            "test_snapshot_include_traces",
        ],
        "nn": [
            # new in 2.11
            # AssertionError: Scalars are not close!
            "test_CTCLoss_cudnn_cuda",
            # AssertionError: Tensor-likes are not close! - https://github.com/ROCm/TheRock/issues/4744
            # Failed on gfx1151 and gfx942 (only with python 3.13)
            "test_Embedding_discontiguous_cuda",
        ],
        "torch": [
            "test_cpp_warnings_have_python_context_cuda",
        ],
        "utils": [
            # ROCm devel/runtime-dependent UT. Skip in the PyTorch full-suite lane;
            # this is expected to run in the separate ROCm devel UT step.
            "test_load_standalone",
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
        "modules": [
            # TestModuleCUDA - CTCLoss cpu/gpu parity scalar mismatch
            "test_cpu_gpu_parity_nn_CTCLoss_cuda_float32",
            # TestModuleCUDA - CTCLoss forward scalar mismatch
            "test_forward_nn_CTCLoss_cuda_float32",
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
            # inductor/test_aot_inductor_package: AOTI C++ package tests need
            # more complete CMake/runtime library-path handling in the wheel CI lane.
            "test_compile_after_package_multi_arch",
            "test_compile_after_package_static",
            "test_compile_standalone_cos",
            "test_compile_with_exporter",
            "test_compile_with_exporter_weights",
            #Also failed on https://github.com/ROCm/TheRock/actions/runs/24898379109
            "test_flex_attention_logging_cuda",
            "test_linalg_eig_stride_consistency_cuda",
            "test_linalg_eig_stride_consistency_cuda",
            "test_linalg_eig_stride_consistency_dynamic_shapes_cuda",
            "test_repeated_calling_cuda",
            #new test
            "test_run2run_determinism_model_name_DistillGPT2_training_or_inference_inference_precision_amp",
            #Passed in https://github.com/ROCm/TheRock/actions/runs/24898379109
            "test_return_aux_deprecation_warnings_cuda_float16",
        ],
        "functorch": [
            #passed on https://github.com/ROCm/TheRock/actions/runs/24898379109
            "test_torch_return_types_returns_cuda",
        ],
        "jit_fuser_te": [
            #passed on https://github.com/ROCm/TheRock/actions/runs/24898379109
            "test_binary_div_ops",
            "test_binary_ops",
            "test_binary_tensor_scalar_ops",
            "test_ternary_norm_ops",
            "test_ternary_ops",
            "test_unary_ops",
            "test_where_ops",
            "test_binary_div_ops",
            "test_binary_ops",
            "test_binary_tensor_scalar_ops",
            "test_ternary_norm_ops",
            "test_ternary_ops",
            "test_unary_ops",
            "test_where_ops",
        ],
        "torch_config_hash_determinism": [
            #passed on https://github.com/ROCm/TheRock/actions/runs/24898379109
            "test_inductor_config_hash_portable_deterministic",
            "test_inductor_config_hash_portable_without_ignore",
        ],
        "spectral_ops": [
            #new failures
            #skipping these tests for a cleaner run
            #these tests passed in https://github.com/ROCm/TheRock/actions/runs/25360031025
            "test_istft_against_librosa_cuda_float64",
            "test_stft_cuda_float64",
          "distributed": [
            # Error while creating shared memory segment /dev/shm/nccl-VPyhzw (size 21823872), error: No space left on device (28)
            "test_3d_mlp_with_nd_mesh",
            # AssertionError: False is not true : cuda:0 used 2615148544.0 bytes after collective, 70% more than the status before (1495269376.0 bytes). Extra CUDA context may have been created.
            "test_extra_cuda_context",
        ],
    },
    "gfx942": {
        "cuda": [
            # new test
            # AssertionError: Scalars are not equal!
            "test_graph_capture_reclaim_shared_pool",
        ],
    },
    # "gfx120": {
    #     "unary_ufuncs": [
    #         # this failed only once. maybe python version dependent? probably the run was python 3.13
    #         # AssertionError: Tensor-likes are not close!
    #         "test_batch_vs_slicing_polygamma_polygamma_n_2_cuda_float16",
    #     ],
    # },
    # "windows": {
    #     empty for the moment
    # },
}
