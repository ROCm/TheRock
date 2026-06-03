# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "nn": [
            # AssertionError: False is not true : Expected NaN in pdist output
            # AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
        ],
        "convolution": [
            # ROCm/MIOpen native hang in deterministic cuDNN Conv2d generated tests.
            # Covers dilation 1/2/3 across dtype variants; replaces file-level nn/test_convolution exclusion.
            "test_Conv2d_deterministic_cudnn",
        ],
        "distributions": [
            # SIGSEGV - OpenBLAS exceeds precompiled 128-thread hard limit
            # even with OPENBLAS_NUM_THREADS=64; crash in wishart.log_prob
            "test_entropy_monte_carlo",
        ],
        "dynamo": [
            # CI GPU isolation warning contaminates this exact stderr log assertion.
            "test_logs_out",
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
            # TestOpInfoPropertiesCUDA - ROCm 7.13 eager vs Triton log/log10 bitwise drift
            "test_eager_equivalence_log10_backend_inductor_default_cuda_float32",
            "test_eager_equivalence_log_backend_inductor_default_cuda_float16",
            "test_eager_equivalence_log_backend_inductor_default_cuda_float32",
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float16",
            "test_unary_ufunc_numerical_log10_backend_inductor_default_cuda_float32",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_bfloat16",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_float16",
            "test_unary_ufunc_numerical_log_backend_inductor_default_cuda_float32",
            # ExtensionBackendTests - extension_device registration/is_available handling
            "test_open_device_registration",
            # inductor/test_user_streams: stream/cudagraph structure mismatches and hangs on ROCm.
            "test_codegen_structure_parallel_matmuls",
            "test_codegen_structure_pipeline",
            "test_codegen_structure_single_stream",
            "test_explicit_current_stream_with_cudagraphs",
            "test_implicit_current_stream_with_cudagraphs",
            # inductor/test_autoheuristic: compute_cap is a string in ROCm wheel metadata.
            # pytest -k also matches the file name, so exclude neighboring tests explicitly.
            "(AutoHeuristicTest and not test_autoheuristic_a100 and not test_autoheuristic_h100 and not test_autoheuristic_pad_mm and not test_global_feedback and not test_mixed_mm_a100 and not test_pad_mm_autoheuristic_deterministic_mode)",
            # inductor/test_aot_inductor_package: AOTI C++ package tests need
            # more complete CMake/runtime library-path handling in the wheel CI lane.
            "test_compile_after_package_multi_arch",
            "test_compile_after_package_static",
            "test_compile_standalone_cos",
            "test_compile_with_exporter",
            "test_compile_with_exporter_weights",
        ],
        "fx": [
            # test_fx: backward-compatibility expectation drift.
            "test_function_back_compat",
        ],
        "schema_check": [
            # test_schema_check: multinomial bf16 schema check can hang GPU on ROCm.
            "test_schema_correctness_multinomial_cuda_bfloat16",
        ],
        "modules": [
            # TestModuleCUDA - CTCLoss cpu/gpu parity scalar mismatch
            "test_cpu_gpu_parity_nn_CTCLoss_cuda_float32",
            # TestModuleCUDA - CTCLoss forward scalar mismatch
            "test_forward_nn_CTCLoss_cuda_float32",
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
        "utils": [
            # ROCm devel/runtime-dependent UT. Skip in the PyTorch full-suite lane;
            # this is expected to run in the separate ROCm devel UT step.
            "test_load_standalone",
        ],
        "distributed": [
            # ---- ATTRIBUTION ANCHORS ----------------------------------------
            # Green stacks (Test PyTorch Wheels Full Suite, gfx94X-dcgpu,
            # 1-GPU runner). Each row is the earliest green CI run for that
            # (PyTorch source, ROCm wheel date) pair on PT 2.13; later runs
            # on the same pair exist but the earliest one is the cleanest
            # before-and-after reference.
            #
            # Apr20/PT + Apr20/ROCm  (baseline / control):
            #   torch 2.13.0a0 + rocm 7.13.0a20260420
            #   first green run 25089830237 (2026-04-29)
            #     https://github.com/ROCm/TheRock/actions/runs/25089830237
            #   last  green run 25244506667 (2026-05-02)
            #
            # Apr20/PT + May01/ROCm  (ROCm-bump attribution layer):
            #   torch 2.13.0a0 + rocm 7.13.0a20260501
            #   first green run 25884612134 (2026-05-14)
            #     https://github.com/ROCm/TheRock/actions/runs/25884612134
            #   last  green run 26317408411 (2026-05-22)
            #   Failures attributed here came from runs 25925372276 and
            #   26136844778 against the same wheels.
            #
            # May12/PT + RC2/ROCm  (PT-bump attribution layer on stable RC2):
            #   torch 2.13.0a0 + rocm 7.13.0rc2
            #   first green run 26587852775 (2026-05-28)
            #     https://github.com/ROCm/TheRock/actions/runs/26587852775
            #   last  green run 26648140224 (2026-05-29)
            #   Failures attributed here came from runs 26532128555 and
            #   26611796814 against the same wheels.
            #
            # Jun1/PT + Jun1/ROCm  (CURRENT TARGET STACK):
            #   torch 2.13.0a0 + rocm 7.14.0a20260601
            #   first green run 26767913043 (2026-06-01)
            #     https://github.com/ROCm/TheRock/actions/runs/26767913043
            #   last  green run 26894626587 (2026-06-03, validation branch)
            #   Initial failures attributed here came from runs 26794193808
            #   and 26828748341; current skip list is what kept this green.
            #
            # The remaining skips below were either reproduced as failing on
            # Jun1 (warranted), or not exercised by validation runs to date
            # (no evidence). See per-entry comments for the specific
            # validation runs.
            #
            # Validation methodology: dispatch test_pytorch_wheels_full.yml on
            # this branch with `tests_to_include=<file>` plus either
            # `debug_skips=true` (run ONLY the listed skips, to confirm they
            # still fail) or `debug_skips=false` (run the file with skips
            # active, to test downstream methods that may be blocked by a
            # SIGSEGV sibling). See run_pytorch_tests_full.py.

            # ---- WARRANTED: torch.linalg.eig requires MAGMA -----------------
            # DistMathOpsTest::test_linalg_ops fails because torch.linalg.eig
            # has no non-MAGMA backend, and this build does not link MAGMA.
            # Validation: failed in Jun1 run 26863226115 (debug_skips=true,
            # tests_to_include=distributed/tensor/test_math_ops).
            # Removal path: only when the ROCm wheel ships MAGMA or PyTorch
            # gains a non-MAGMA eig path.
            "test_linalg_ops",

            # ---- WARRANTED: bf16 norm-modules SIGSEGV in cast bucket --------
            # Both TestFullyShardMixedPrecisionCasts::test_norm_modules_bf16
            # (distributed/_composable/fsdp/test_fully_shard_mixed_precision)
            # and TestReplicateMixedPrecisionCasts::test_norm_modules_bf16
            # (distributed/_composable/test_replicate_mixed_precision) hit a
            # native Segmentation fault during the bf16 forward inside the
            # mixed-precision cast path. Same failure mode, same point in the
            # cast logic, different file.
            # First seen: Apr20/PT + May01/ROCm run 25925372276 shard 3/3,
            # job 76205215147.
            # Validation:
            #   - Jun1 run 26890203267 reproduced the FSDP SIGSEGV; the worker
            #     died and downstream methods in that class could not run.
            #   - Jun1 run 26894626587 with bf16 still skipped confirmed every
            #     other method in TestFullyShardMixedPrecisionCasts passes on
            #     Jun1 (test_norm_modules_fp16 is upstream-skipped on gfx942
            #     via @skipIfRocm; test_submodules_with_external_inputs and
            #     the other 3 methods pass).
            #   - Jun1 run 26903813112 reproduced the Replicate SIGSEGV
            #     independently (Replicate bf16 was temporarily un-skipped for
            #     that dispatch); the other 6 methods in
            #     TestReplicateMixedPrecisionCasts had passed in 26894626587.
            "(TestFullyShardMixedPrecisionCasts and test_norm_modules_bf16)",
            "(TestReplicateMixedPrecisionCasts and test_norm_modules_bf16)",

            # ---- WARRANTED: torchelastic launcher SIGABRT -------------------
            # ElasticLaunchTest::test_virtual_local_rank in
            # distributed/launcher/test_run.py launches a child via
            # torch.distributed.run; the child exits with SIGABRT (exitcode
            # -6) inside script_deviceid.py and torchelastic surfaces it as
            # ChildFailedError. Note: there is a second, different
            # ElasticLaunchTest class in distributed/launcher/api_test.py
            # without this method -- the skip targets the test_run.py one.
            # First seen: Apr20/PT + May01/ROCm run 26136844778 shard 2/3,
            # job 76873873320.
            # Validation: reproduced in Jun1 full-suite run 26907811456
            # shard 3/3 (the only failure in that run; all 25 sibling
            # ElasticLaunchTest methods passed).
            "(ElasticLaunchTest and test_virtual_local_rank)",

            # ---- WARRANTED: bf16 compile path in DDP+compiler ---------------
            # ReplicateTest::test_compile_bf16 (file
            # distributed/_composable/test_replicate_with_compiler.py) fails
            # under torch.compile + bf16 + DDP composable; attribution layer
            # was the May01/ROCm bump on top of the May01/PT source.
            # First seen: run 26296106703 distributed shard 1/3.
            # Validation: failed in Jun1 run 26858601551.
            # Note: this entry was bugged for a long time as
            # "(test_replicate_with_compiler and test_compile_bf16)" -- that
            # form matched no real pytest class (the literal was the file
            # stem, not the class). The fix to "ReplicateTest" was applied
            # alongside the cleanup amend.
            "(ReplicateTest and test_compile_bf16)",

            # ---- WARRANTED: NCCL profiler annotation kernels ----------------
            # CommTest::test_profiler_nccl_annotations_on_gpu_kernels (both
            # use_python_export={True,False}) fail because Kineto/NCCL
            # annotation metadata is missing on the recorded GPU kernels;
            # likely a regression in the profiler patch stack on the May12+
            # PyTorch sources.
            # First seen: run 26794193808 distributed shards 2/3 and 3/3.
            # Validation: both reproduced as failing in Jun1 run 26858374424
            # (single test_c10d_nccl module included; same failure mode).
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_True)",
            "(CommTest and test_profiler_nccl_annotations_on_gpu_kernels_use_python_export_False)",
        ],
    },
}
