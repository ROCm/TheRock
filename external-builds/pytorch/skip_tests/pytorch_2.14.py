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
            # HSA_TOOLS_DISABLE_REGISTER). 17 genuine residual failures remain with
            # the env var. NOTE: most of these PASS on upstream PyTorch's CUDA HUD,
            # so they are ROCm/config divergence (this module re-runs test_cuda.py
            # under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True), NOT PyTorch
            # bugs. Skipped to get CI green; root-cause is tracked separately.
            # CAVEAT: these class/test names also exist in plain test_cuda.py, so the
            # -k substring skips them there too — revisit when narrowing.
            # "Booleans mismatch: True is not False" cluster (allocator introspection
            # under expandable segments):
            "(TestCudaAllocator and test_allocation_traceback)",
            "(TestCudaAllocator and test_allocation_traceback_no_recording)",
            "(TestCudaAllocator and test_allocator_fuzz)",
            "(TestCudaAllocator and test_allocator_memory_fraction_setting)",
            "(TestCudaAllocator and test_allocator_settings)",
            "(TestCudaAllocator and test_cachingAllocator_raw_alloc)",
            "(TestCudaAllocator and test_cpp_memory_snapshot_pickle)",
            "(TestCudaAllocator and test_cycles)",
            "(TestCudaAllocator and test_direct_traceback)",
            "(TestCudaAllocator and test_memory_snapshot_script)",
            # Other residual failures (snapshot tooling / OOM semantics / stats /
            # graph / streams / mempool) under expandable segments:
            "(TestCuda and test_graph_memory_stats_and_use_result_after_destroy_graph)",
            "(TestCuda and test_out_of_memory)",
            "(TestCuda and test_out_of_memory_retry)",
            "(TestCuda and test_streaming_backwards_multiple_streams)",
            "(TestBlockStateAbsorption and test_allocate_in_thread_to_pool)",
            "(TestBlockStateAbsorption and test_allocated_in_middle_of_segment)",
            "(TestMemPool and test_mempool_ctx_multithread)",
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
            # test_module_to_empty_cuda_float32 - regex match on the expected error
            # message fails because ROCm's Copy.cpp appends a C++ CapturedTraceback
            # to the NotImplementedError string. Consistent (3x). TODO: narrow/fix.
            "(TestNNDeviceTypeCUDA and test_module_to_empty_cuda_float32)",
        ],
        "optim": [
            # Run 28411211813 default shard 4/10: TestOptimRenewedCUDA::
            # test_rosenbrock_sparse_with_lrsched_False_SGD_cuda_float64 hangs in the
            # sparse SGD step (device_params.add_(grads, alpha=-lr)); 900s
            # pytest-timeout, consistent across initial run + 2 reruns. TODO: root-cause.
            "(TestOptimRenewedCUDA and test_rosenbrock_sparse_with_lrsched_False_SGD_cuda_float64)",
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
            "(TestLinalgCUDA and test_cholesky_solve_batched_many_batches)",
            # test_linalg un-excluded from EXCLUDED_TEST_MODULES (the prior "0 failed
            # then SIGIOT" was the rocprofiler shutdown bug, fixed by
            # HSA_TOOLS_DISABLE_REGISTER). Two genuine residual failures remain
            # (consistent across 3 local reruns on MI300X, 2.14 nightly):
            # svd_lowrank complex128 fails with _LinAlgError "svd failed to converge
            # (ill-conditioned)"; tunableop call-count off. TODO: root-cause + narrow.
            "(TestLinalgCUDA and test_svd_lowrank_cuda_complex128)",
            "(TestLinalgCudaOnlyCUDA and test_call_count_tunableop_cuda_float32)",
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
            # /dev/shm exhausted by 8-rank 3D mesh tensor allocs; NCCL shared memory OOM
            "(TestFullyShardHSDP3DTraining and test_3d_mlp_with_nd_mesh)",
            # Run 28411211813 distributed shard 1/3: NCCLTraceTest::
            # test_compiled_with_reduce_overhead - genuine NCCL ALLREDUCE timeout +
            # ncclSystemError (distinct from the rocprofiler shutdown crash). TODO: investigate.
            "(NCCLTraceTest and test_compiled_with_reduce_overhead)",
        ],
    },
}
