# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch nightly (2.14) wheels. create_skip_tests.py
# loads only generic.py + pytorch_{version}.py with no inheritance between
# version files, so this is a full mirror of pytorch_2.13.py plus any
# 2.14-specific additions. Module-level hangs are handled separately via
# EXCLUDED_TEST_MODULES in run_pytorch_tests_full.py, not here. See
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
            # Run 27662748552 default shard 6/10, job 81810307258 (expandable segments):
            # Run 27771426668 default shard 6/10, job 82172287851 (plain test_cuda.py):
            # TestCuda::test_cudnn_multiple_threads_same_device crashes consistently with
            # pure virtual method called / SIGSEGV in both test_cuda.py and
            # test_cuda_expandable_segments.py. Root cause: MIOpen per-thread CUDNN-handle
            # pool race condition under concurrent conv2d threads, independent of whether
            # expandable segments are enabled. Expression covers both files.
            "(TestCuda and test_cudnn_multiple_threads_same_device)",
            # Run 28379269101 default shard 9/10, job 84077240401:
            # TestBlockStateAbsorption::test_no_triton_on_import spawns a subprocess
            # (import torch; torch.rand(2, device='cuda')) that dies with SIGABRT.
            "(TestBlockStateAbsorption and test_no_triton_on_import)",
        ],
        "cuda_expandable_segments": [
            # test_cuda_expandable_segments un-excluded from EXCLUDED_TEST_MODULES
            # (the prior "hang" was the rocprofiler shutdown bug, fixed by
            # HSA_TOOLS_DISABLE_REGISTER). Verified the faithful CI way (run_test.py
            # runs the file as a script so its __main__ sets the expandable-segments
            # allocator): only test_out_of_memory genuinely fails (test_hip_device_count
            # is already skipped above). TestCuda::test_out_of_memory asserts an OOM
            # tensor flag that is False on ROCm expandable segments. TODO: root-cause.
            # (NB: a large apparent failure cluster under raw `pytest file.py` was a
            # harness artifact — without __main__, EXPANDABLE_SEGMENTS mismatches the
            # runtime allocator — NOT real; do not re-add those.)
            "(TestCuda and test_out_of_memory)",
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
            # Run 28411211813 default shard 3/10: TestNNDeviceTypeCUDA::
            # test_module_to_empty_cuda_float32 FAILED CONSISTENTLY (regex match on the
            # expected error message fails because ROCm's Copy.cpp appends a C++
            # CapturedTraceback to the NotImplementedError string). NOTE: passes when
            # run in isolation via run_test.py, so it is order/state-dependent within
            # the full shard. Kept because it fails in the real sharded CI run.
            # TODO: find the polluting test / narrow.
            "(TestNNDeviceTypeCUDA and test_module_to_empty_cuda_float32)",
            # Run 28446166928 default shard 5/10:
            # TestConvolutionNN::test_ConvTranspose2d_output_size_downsample_upsample
            # hits the 900s pytest-timeout in F.conv2d() inside ConvTranspose2d forward.
            # CPU-path hang; reproduced consistently across all retries (~30min total).
            "(TestConvolutionNN and test_ConvTranspose2d_output_size_downsample_upsample)",
        ],
        "legacy_vmap": [
            # Run 28545046175 default shard 6/10: TestVmapAPILegacy::
            # test_fallback_masked_fill exceeds the 900s pytest-timeout on every
            # attempt (1652s total across reruns in RC0 run 28546068730 as well).
            # The vmap fallback path invokes the eager masked_fill op element-wise;
            # hangs consistently on gfx942 across both rocm7.15.0a20260629 and
            # rocm7.14.0rc0 wheels. True hang (not the rocprofiler crash pattern).
            "(TestVmapAPILegacy and test_fallback_masked_fill)",
        ],
        "optim": [
            # Run 28411211813 default shard 4/10: TestOptimRenewedCUDA::
            # test_rosenbrock_sparse_with_lrsched_False_SGD_cuda_float64 hit the 900s
            # pytest-timeout (x3 reruns) in the sparse SGD step. NOTE: passes in ~8s
            # when run in isolation via run_test.py, so it is order/state-dependent
            # within the full shard. Kept because it times out in the real sharded CI
            # run. TODO: find the polluting test / root-cause the hang.
            "(TestOptimRenewedCUDA and test_rosenbrock_sparse_with_lrsched_False_SGD_cuda_float64)",
        ],
        "ops": [
            # test_ops un-excluded from EXCLUDED_TEST_MODULES (prior crash was the
            # rocprofiler shutdown bug, fixed by HSA_TOOLS_DISABLE_REGISTER). Verified
            # faithfully via run_test.py: TestCommonCUDA::test_dtypes_sparse_sampled_addmm
            # fails CONSISTENTLY ("supported dtypes for sparse.sampled_addmm on cuda are
            # incorrect") on ROCm. TODO: root-cause the dtype-support divergence.
            "(TestCommonCUDA and test_dtypes_sparse_sampled_addmm_cuda)",
        ],
        "binary_ufuncs": [
            # Run 28379269101 default shard 6/10, job 84077240385:
            # TestBinaryUfuncs::test_laguerre_legendre_polynomial_nan_propagation
            # AssertionError: tensor(False) is not true - NaN not propagated for
            # special_laguerre_polynomial_l / special_legendre_polynomial_p on ROCm.
            "(TestBinaryUfuncs and test_laguerre_legendre_polynomial_nan_propagation)",
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
        "linalg": [
            # Run 27657923936 default shard 1/10 plus local Rock-vs-nightly probe:
            # TestLinalgCUDA::test_cholesky_solve_batched_many_batches fails in the
            # large broadcasted batched solve case (A_dims=(5,256,256), b_dims=(5,10)).
            # Nightly passes after upstream cholesky_solve batched/solver-dispatch changes.
            # test_linalg un-excluded from EXCLUDED_TEST_MODULES (the prior "0 failed
            # then SIGIOT" was the rocprofiler shutdown bug, fixed by
            # HSA_TOOLS_DISABLE_REGISTER). Verified the faithful CI way (run_test.py):
            # the only residual failures are the 4 cholesky_solve_batched_many_batches
            # dtype variants already covered by the substring skip above. (svd_lowrank
            # / test_call_count_tunableop are auto-skipped via --import-disabled-tests,
            # upstream-disabled per pytorch/pytorch#186872 — no skip needed here.)
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
            # Run 27657923936 default shard 10/10:
            # DynamicShapesCpuTests::test_tmp_not_defined_issue3_dynamic_shapes_cpu
            # still fails consistently with a small tensor mismatch
            # (3/6144 elements, max abs diff 9.1552734375e-05). Local Rock Jun12
            # source does not generate this CPU variant; nightly source generates it and passes.
            "(DynamicShapesCpuTests and test_tmp_not_defined_issue3_dynamic_shapes_cpu)",
            # Run 27473608564 default shard 1/10, job 81208818457:
            # TestGpuWrapper::test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code
            # expects at::vec:: in generated cpp wrapper; ROCm codegen omits CPU vec ISA.
            "(TestGpuWrapper and test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code)",
            # Run 28379269101 default shard 2/10, job 84077240381:
            # CudaReproTests::test_triton_interpret (TRITON_INTERPRET=1 + torch.compile)
            # spawns a child that dies with SIGABRT (subprocess.CalledProcessError).
            "(CudaReproTests and test_triton_interpret)",
            # Run 28379269101 default shard 8/10, job 84077240436:
            # TestExternKernelCaller::test_extern_kernel_benchmark_valid_timing
            # InductorError: AssertionError - incorrect result from extern-kernel choice.
            "(TestExternKernelCaller and test_extern_kernel_benchmark_valid_timing)",
        ],
        "extension_backend": [
            # Run 27361388921 default shard 4/10, job 80849478614:
            # ExtensionBackendTests::test_open_device_registration expects CPU-style
            # inductor_cpp_wrapper source but ROCm generates privateuse1 AOTI glue.
            "(ExtensionBackendTests and test_open_device_registration)",
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
        "serialization": [
            # Mirrored from pytorch_2.12.py — TestSerialization/TestOldSerialization
            # expect debug env flags set in CI; TheRock wheel CI does not enable them.
            # Run 27390088455 default shard 1/10, job 80945585178: FAILED CONSISTENTLY.
            "test_debug_set_in_ci",
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
        "matmul_cuda": [
            # Run 28446166928 default shard 6/10:
            # TestMatmulCudaCUDA::test_grouped_gemm_rocm_ck_flag_cuda asserts that the
            # Composable Kernel (CK) grouped GEMM backend is available; it is not enabled
            # in the TheRock wheel build. Genuine ROCm build-configuration gap.
            "(TestMatmulCudaCUDA and test_grouped_gemm_rocm_ck_flag_cuda)",
        ],
        "reductions": [
            # Run 28446166928 default shard 8/10:
            # ROCm's quantile kernel has a lower element limit than CUDA's reference.
            # RuntimeError: "quantile() input tensor is too large" at Sorting.cpp:289.
            "(TestReductionsCUDA and test_quantile_large_input_cuda_float32)",
            "(TestReductionsCUDA and test_quantile_large_input_cuda_float64)",
            "(TestReductionsCUDA and test_quantile_size_limit_cuda)",
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
            # Run 28528513481 distributed shard 2/3:
            # TestFileSystem::test_fsspec_without_fileno_support fails consistently.
            # Root cause TBD; confirmed genuine on rocm7.14.0rc0 debug run.
            "(TestFileSystem and test_fsspec_without_fileno_support)",
            # Child process exits with SIGABRT inside torchelastic launcher (test_run.py)
            "(ElasticLaunchTest and test_virtual_local_rank)",
            # Rock 2.13 Kineto/NCCL annotation metadata missing on recorded GPU kernels
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_True)",
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_False)",
            # NCCL symmetric memory rendezvous not supported; host communicator not found
            "test_ce_allgather",
            "test_ce_alltoall",
            # Run 27657923936 distributed shard 3/3:
            # AssertionError: Scalars are not close! fp32 numerical drift in FSDP
            # post-optimizer event. Direct Rock/nightly and local TheRock wrapper
            # runs pass at world size 8, so this appears nondeterministic.
            "(TestFullyShard1DTrainingCore and test_post_optim_event)",
        ],
    },
}
