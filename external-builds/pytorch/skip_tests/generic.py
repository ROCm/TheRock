# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    # Special notes for Windows:
    #   * Some tests hang and *must* be skipped for testing to complete.
    #     That is likely related to processes not terminating on their own:
    #     https://github.com/ROCm/TheRock/issues/999. Note that even if
    #     _test cases_ themselves terminate, the parent process still
    #     hangs though. In run_pytorch_tests.py we exit with `os.kill()` to
    #     force termination.
    #   * Linux has substantial testing on datacenter GPUs while Windows support
    #     is newer and skews towards consumer GPUs with lower specs. We disable
    #     some tests that are resource intensive or otherwise degrade CI
    #     stability. We are also conservative in disabling some tests for all
    #     pytorch versions and all GPUs. Perfect is the enemy of the good and
    #     we would rather run a subset of tests with high confidence than run
    #     all tests with low confidence.
    "windows": {
        "autograd": [
            # JIT compilation without MSVC installed then device mismatch:
            # We should fix the test to fail/skip more gracefully.
            #   Error checking compiler version for cl: [WinError 2] The system cannot find the file specified
            #   AssertionError: Object comparison failed: <torch.cuda.Stream device=cuda:0 cuda_stream=0x1fa05550b50> != <torch.cuda.Stream device=cuda:0 cuda_stream=0x0>
            "test_consumer_to_single_producer_case_2_correctness",
            # Failures on Python 3.11 (not 3.12 or 3.13) with
            #   Windows fatal exception: code 0xc0000374
            # Possibly due to use of `torch.jit.script`?
            "test_fork_join_in_middle",
            # This test JIT compiles and fails if MSVC is not installed on Windows:
            # We should fix the test to fail/skip more gracefully.
            #   subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
            "test_multi_grad_all_hooks",
            # This is/was also failing on gfx942 linux, see the per-release skip test files.
            #   AssertionError: "Simulate error" does not match "grad can be implicitly created only for scalar outputs"
            "test_reentrant_parent_error_on_cpu_cuda",
        ],
        "binary_ufuncs": [
            # Failures on Python 3.11 (not 3.12 or 3.13) with
            #   Windows fatal exception: code 0xc0000374
            # Possibly due to use of `torch.jit.script`?
            "test_div_and_floordiv_script_vs_python_cuda",
            "test_idiv_and_ifloordiv_vs_python_cuda",
        ],
        "cuda": [
            # RuntimeError: miopenStatusUnknownError
            "test_autocast_rnn",
            # On some test runners
            #   AssertionError: False is not true
            #   self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)
            "test_cublas_workspace_explicit_allocation",
            # For Python 3.11, this fails with
            #   torch.AcceleratorError: HIP error: operation not permitted when stream is capturing
            "test_cuda_graph_tensor_item_not_allowed",
            # *** Test hang (see above) ***
            "test_graph_error",
            # This test conflicts with how our test script and runners are
            # configured.
            "test_hip_device_count",
            # Multi-processing forking code that may not work on Windows?
            "test_is_pinned_no_context",
            # Bug, needs triage:
            #   AssertionError: Scalars are not equal!
            #   Expected 0 but got 2173342911312.
            "test_streams",
            # Device-side assert() does not propagate to the host on Windows ROCm:
            # the KMD has no trap handler, so the faulted queue never reports an
            # error and torch.cuda.synchronize() hangs until the CI job timeout.
            # These tests deliberately trigger a device-side assert and await it
            # with no subprocess timeout, so they hang rather than fail.
            # Re-enable once the Windows ROCm driver propagates device-side
            # faults to the runtime.
            # See https://github.com/ROCm/TheRock/issues/5565
            "test_fixed_cuda_assert_async",
            "test_index_out_of_bounds_exception_cuda",
            # Same device-side-assert-propagation issue as the two tests above:
            # spawns a subprocess that feeds invalid probabilities (negative,
            # inf, nan) to torch.multinomial, calls torch.cuda.synchronize(), and
            # asserts the device-side assert surfaces in stderr. On Windows ROCm
            # the fault never propagates, so it hangs/fails instead.
            # See https://github.com/ROCm/TheRock/issues/5565
            "test_multinomial_invalid_probs_cuda",
            # TestAmdSmiImport helpers assume Linux: they read /proc/self/maps
            # to locate libc.so and symlink it as libamd_smi.so.
            #   FileNotFoundError: [Errno 2] No such file or directory: '/proc/self/maps'
            # Added upstream in https://github.com/pytorch/pytorch/pull/197482
            # See https://github.com/ROCm/TheRock/issues/8779
            "test_amdsmi_bare_unversioned_path_uses_loader_first",
            "test_amdsmi_explicit_unversioned_path_is_redirected",
            "test_amdsmi_versioned_library_path_is_preserved",
        ],
        "nn": [
            # Hangs on some Windows ROCm runners until the job hits the 6h limit.
            # https://github.com/ROCm/TheRock/issues/5565
            "test_cross_entropy_loss_2d_out_of_bounds_class_index",
            # RuntimeError: miopenStatusUnknownError
            "test_cudnn_weight_format",
            "test_rnn_retain_variables_cuda_float16",
            "test_rnn_retain_variables_cuda_float32",
            "test_variable_sequence_cuda_float16",
            "test_variable_sequence_cuda_float32",
            # Convs are failing numerics on some test machines.
            # Possibly fixed by https://github.com/ROCm/rocm-libraries/commit/7338a0b71f43c41c3882e976ef591cb9adcd64d0
            # At least [gfx1151, torch 2.9], possibly others.
            #   Mismatched elements: 4 / 4 (100.0%)
            #   Mismatched elements: 96 / 96 (100.0%)
            # TODO: try re-enabling these tests after that commit is included in releases
            "test_Conv",  # Broad pattern - this matches 10s of test cases.
            # AssertionError: False is not true
            # self.assertTrue(torch.allclose(x.grad.cpu(), xx.grad, rtol=rtol, atol=1e-3))
            "test_warp_softmax_64bit_indexing_cuda_float16",
        ],
        "torch": [
            # Large test that isn't very CI-friendly (takes 1-180 seconds depending on runner and torch version)
            "test_conv_transposed_large_cuda",
            # *** Test hang (see above) ***
            # The callstack for this one points to _fill_mem_eff_dropout_mask, so it may be related to aotriton?
            "test_cublas_config_nondeterministic_alert_cuda",
            # Large test that isn't very CI-friendly (takes ~2 seconds, possibly hanging)
            "test_memory_format_operators_cuda",
            # Flaky tests hanging on some gfx1151 machines...
            # Maybe memory pressure? Tests use some large tensors:
            #   v = torch.FloatTensor([64000., 32., 64000.])
            # Move to gfx1151-specific skip list? Check if passing on Linux.
            # We could also skip all test_grad_*.
            "test_grad_scale_will_not_overflow_cuda",
            "test_grad_scaling_unscale_sparse_cuda_float32",
        ],
    },
}
