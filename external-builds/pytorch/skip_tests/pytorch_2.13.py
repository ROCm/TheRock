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
            # Run 27662748552 default shard 6/10, job 81810307258 (expandable segments):
            # Run 27771426668 default shard 6/10, job 82172287851 (plain test_cuda.py):
            # TestCuda::test_cudnn_multiple_threads_same_device crashes consistently with
            # pure virtual method called / SIGSEGV in both test_cuda.py and
            # test_cuda_expandable_segments.py. Root cause: MIOpen per-thread CUDNN-handle
            # pool race condition under concurrent conv2d threads, independent of whether
            # expandable segments are enabled. Expression covers both files.
            "(TestCuda and test_cudnn_multiple_threads_same_device)",
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
            # Run 28126068513 default shard 6/10, job 83294069418:
            # Test passes an out-of-bounds class index to cross-entropy loss and asserts
            # a device assertion error appears in stderr. On ROCm, device assertions
            # manifest as a GPU memory access fault instead of a catchable assertion
            # message, so the expected error string is never found.
            "(TestNNDeviceTypeCUDA and test_cross_entropy_loss_2d_out_of_bounds_class_index_cuda_float16)",
            "(TestNNDeviceTypeCUDA and test_cross_entropy_loss_2d_out_of_bounds_class_index_cuda_float32)",
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
        "dynamo": [
            # Run 28109021458 job 83231166331:
            # LoggingTests::test_logs_out captures all subprocess log output and does
            # an exact string comparison against an expected log. ROCm's HSA runtime
            # emits an extra warning line into stderr:
            #   W agent.cpp:158] Attempt to enable hip visiblity for agent-2
            #   which is not visible to HSA (ROCR)
            # This warning leaks into the captured output and breaks the exact match.
            # Same agent.cpp warning family as test_hip_device_count (Bucket 3a in
            # skip_root_causes_and_fixes.md). Fix: suppress ROCm agent warnings in
            # the test subprocess env or fix the visibility mismatch causing the warning.
            "(LoggingTests and test_logs_out)",
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
            # Run 28109021458 distributed shard 2/3, job 83231166359:
            # CPFlexAttentionTest context-parallel flex attention numerical mismatch.
            # 24/1048576 elements wrong; greatest abs diff 1.60e-05 (tol 2e-06).
            # causal_mask was already a Jun1-stack monitoring candidate in FLAKY_TESTS.md
            # (failed run 26922264789, passed 26907811456); now fails again on Jun12 stack.
            # document_mask is a new consistent failure in the same run.
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",
            "(CPFlexAttentionTest and test_cp_flex_attention_document_mask)",
            # Promoted flaky entries from FLAKY_TESTS.md — failed in multiple full-suite
            # runs on the June 12 stack and promoted per the ledger's promotion policy.
            # Passed in validation run 27657923936 but that is only one passing run;
            # the ledger requires 3+ consecutive passes before removal.
            # Run 27480021236 distributed shard 1/3, job 81226065498 (+ earlier runs):
            # ComposabilityTest::test_replicate_pp_ScheduleClass3_bfloat16 hits NCCL
            # /dev/shm exhaustion under 8-rank pipeline-parallel reduce-grad.
            "(ComposabilityTest and test_replicate_pp_ScheduleClass3_bfloat16)",
            # Run 27480021236 distributed shard 1/3, job 81226065498:
            # TestStateDict::test_shared_weight fails with optimizer state_dict
            # tensor mismatch (1/3000000 elements, tiny drift).
            "(TestStateDict and test_shared_weight)",
            # Run 27480021236 distributed shard 1/3 + run 26922264789:
            # TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward fails with
            # parameter tensor mismatch after optimizer-in-backward on separate stream.
            "(TestDistBackendWithSpawn and test_ddp_apply_optim_in_backward)",
            "(TestDistBackendWithSpawn and test_ddp_apply_optim_in_backward_grad_as_bucket_view_false)",
        ],
    },
}
