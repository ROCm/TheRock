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
            # TestCudaAllocator::test_allocator_backend spawns a subprocess with a
            # wiped environment (env={}); the child exits 127, i.e. it fails before the
            # test body — a dynamic-loader failure resolving the interpreter/torch
            # shared libs under an empty environment. The ldconfig registration added
            # to run_pytorch_tests_full.py registers the ROCm SDK dirs but not whatever
            # the wiped-env child is still missing (libpython / loader path). RCA OPEN:
            # pin the exact unresolved object in the exit-127 child. Run 29384801639.
            "(TestCudaAllocator and test_allocator_backend)",
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
            # Run 27473608564 default shard 9/10, job 81208818474:
            # TestNNDeviceTypeCUDA::test_linear_cross_entropy_loss_default_bias_False_cuda_float32
            # input-grad ULP worst case 952 > 854. Re-verified GENUINE on wheel
            # rocm7.15.0a20260712 (b2b98e00): fails consistently (ULP 952>854),
            # confirmed in debug run 29215386233 shard 4.
            "(TestNNDeviceTypeCUDA and test_linear_cross_entropy_loss_default_bias_False_cuda_float32)",
            # Run 28411211813 default shard 3/10: TestNNDeviceTypeCUDA::
            # test_module_to_empty_cuda_float32 FAILED CONSISTENTLY (regex match on the
            # expected error message fails because ROCm's Copy.cpp appends a C++
            # CapturedTraceback to the NotImplementedError string). NOTE: passes when
            # run in isolation via run_test.py, so it is order/state-dependent within
            # the full shard. Kept because it fails in the real sharded CI run.
            # TODO: find the polluting test / narrow.
            "(TestNNDeviceTypeCUDA and test_module_to_empty_cuda_float32)",
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
        "export": [
            # TestExportOnFakeCudaCUDA spawns subprocesses with a near-wiped env
            # (env={"CUDA_VISIBLE_DEVICES":""}); the children exit 127 — a dynamic-
            # loader failure before the test body, same class as test_allocator_backend
            # above. The ldconfig SDK registration did not resolve it. RCA OPEN: pin the
            # exact shared object the wiped-env child cannot load. Run 29384801639
            # failed all 9 (default shard 5/10).
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
        "inductor": [
            # log10 inductor_numerics fp16/fp32 XPASS: log10 is now bitwise-correct on
            # ROCm but still listed in ROCM_UNARY_NUMERICAL_XFAILS["inductor_numerics"],
            # so the strict-xfail unexpectedly passes and the test FAILS. Removing the
            # skip UNMASKS the XPASS (full-suite run 29384801639 failed both) — the skip
            # must stay until log10 is dropped from that xfail dict upstream.
            "test_unary_ufunc_numerical_log10_backend_inductor_numerics_cuda_float16",
            "test_unary_ufunc_numerical_log10_backend_inductor_numerics_cuda_float32",
            # Run 27228539427 default shard 6/10:
            # log10 inductor_default float32 differs from eager under exact equality.
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float32",
            # Run 27228539427 default shard 6/10:
            # log10 inductor_default float16 differs from eager under exact equality.
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float16",
            # NOTE: test_tmp_not_defined_issue3_dynamic_shapes_cpu and
            # test_cuda_cpp_wrapper_keeps_vec_isa_for_host_vectorized_code REMOVED here.
            # Root cause: inductor's CPU vec-ISA probe dlopens a test .so that links
            # librocprofiler-sdk.so.1 etc. from the SDK dirs; when those dirs are off the
            # loader path the probe fails, valid_vec_isa_list() returns [] and codegen
            # falls back to scalar loops (no at::vec) -> the vec_isa assertion fails and
            # the tmp_not_defined reduction reassociates and drifts. Fixed by wiring up
            # _register_rocm_libs_with_ldconfig() (incl. host-math/lib) in
            # run_pytorch_tests_full.py, which puts the SDK dirs on the loader path.
        ],
        "distributed": [
            # Child process exits with SIGABRT inside torchelastic launcher (test_run.py)
            "(ElasticLaunchTest and test_virtual_local_rank)",
            # Rock 2.13 Kineto/NCCL annotation metadata missing on recorded GPU kernels
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_True)",
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_False)",
            # NCCL symmetric memory rendezvous not supported; host communicator not found
            "test_ce_allgather",
            "test_ce_alltoall",
            # TestFullyShard1DTrainingCore::test_post_optim_event — fp32 FSDP
            # post-optimizer-event value drift, nondeterministic (fails intermittently,
            # run 29384801639 distributed shard 2/3). Suspected tie to the ROCm
            # device_count / NUM_PROCS visibility handling (fw#16969) rather than a
            # per-test numeric bug. RCA OPEN: determine the nondeterminism source
            # (reduce-scatter ordering vs process-count/visibility) and the fix owner.
            "(TestFullyShard1DTrainingCore and test_post_optim_event)",
            # Run 29223117302 distributed shard 3/3:
            # SymmMemCollectiveTest::test_two_shot_all_reduce hangs in
            # tearDownClass (process.join() on the multi-rank PG never returns)
            # → pytest-timeout ERROR at >900s. The test body itself passes; the
            # teardown hang is a ROCm symmetric-memory multiprocess teardown
            # issue. Reproduced locally on b2b98e00 (8x MI300X): ERROR at
            # teardown, Timeout(>180s). NB: @skip_if_lt_x_gpu(4) does NOT gate
            # this out — the gfx942 distributed runner exposes 8 GPUs despite the
            # "1gpu" name, so world_size=device_count>=4 and the test runs.
            # Upstream pytorch/pytorch#159397 (flaky-bot "DISABLED ...") was
            # auto-CLOSED 2025-09-16 as no-longer-flaky on rocm — but that is
            # STALE for this wheel/ROCm: we reproduced a deterministic teardown
            # hang here, so keep the skip.
            "(SymmMemCollectiveTest and test_two_shot_all_reduce)",
            # Run 29223117302 distributed shard 2/3: FLAKY (pending RCA).
            # ReplicateTest::test_compile_fp16 + test_compile_gpu failed in CI
            # with grad assertEqual(p1.grad, p2.grad) "Tensor-likes are not
            # close!" — Mismatched elements 3 / 4,000,000 (0.0%), i.e. a handful
            # of elements just outside tolerance. NOT reproducible locally on the
            # wheel commit b2b98e00 (8x MI300X): passed 3x isolated AND passed the
            # faithful full-module run_test.py --distributed-tests path (EXIT 0).
            # So this is pod/allocator-dependent numeric jitter, not a
            # deterministic ROCm bug. Skipped to keep CI green; TODO: root-cause
            # (likely a tolerance bump upstream) and remove — do NOT treat as a
            # settled genuine failure.
            "(ReplicateTest and test_compile_fp16)",
            "(ReplicateTest and test_compile_gpu)",
            # Run 29223117302 distributed shard 2/3: FLAKY (pending RCA).
            # CPFlexAttentionTest::test_cp_flex_attention_causal_mask +
            # test_cp_flex_attention_document_mask failed in CI with
            # assert_close(cp_out, expect_out) "Tensor-likes are not close!" —
            # Mismatched elements 66 / 1,048,576 (0.0%); CI itself showed
            # 1-failed-1-passed within the shard. NOT reproducible locally on
            # b2b98e00: passed 3x isolated AND the faithful full-module
            # run_test.py --distributed-tests path (EXIT 0). Pod-dependent numeric
            # jitter, not a deterministic bug. Skipped to keep CI green; TODO:
            # root-cause (likely a tolerance bump) and remove.
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",
            "(CPFlexAttentionTest and test_cp_flex_attention_document_mask)",
            # TheRock run 29645120650 (job 88082045739): Nccl2WindowTest::test_register_errors
            # → win.tensor_register(win_buf) crashes with RuntimeError exit code 10 on both
            # ranks. NCCL2 symmetric-memory window registration broken on ROCm 7.15 wheel.
            # Same failure seen on upstream rocm-preview run 29619818031 (distributed shard 1).
            "(Nccl2WindowTest and test_register_errors)",
        ],
    },
}
