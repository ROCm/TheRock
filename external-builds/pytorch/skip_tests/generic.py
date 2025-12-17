skip_tests = {
    "gfx950": {
        "cuda": {
            "test_autocast_torch_bf16",
            "test_autocast_torch_fp16",
        }
    },
    "common": {
        # ----------------
        # might be failing
        # ----------------
        # "binary_ufuncs": [ "test_cuda_tensor_pow_scalar_tensor_cuda" ]
        # ----------------
        "cuda": [
            # HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES not working
            # to restrict visibility of devices
            # AssertionError: String comparison failed: '8, 1' != '8, 8'
            "test_device_count_not_cached_pre_init",
            # empty_stats() in test_cuda.py does not match stats returned
            # Returned is:
            # OrderedDict({'allocated_bytes.allocated': 0, 'allocated_bytes.current': 0, 'allocated_bytes.freed': 0,
            # 'allocated_bytes.peak': 0, 'allocation.allocated': 0, 'allocation.current': 0, 'allocation.freed': 0,
            # 'allocation.peak': 0, 'host_alloc_time.avg': 0, 'host_alloc_time.count': 0, 'host_alloc_time.max': 0,
            # 'host_alloc_time.min': 0, 'host_alloc_time.total': 0,  'host_free_time.avg': 0, 'host_free_time.count': 0,
            # 'host_free_time.max': 0, 'host_free_time.min': 0, 'host_free_time.total': 0, 'num_host_alloc': 0,
            # 'num_host_free': 0, 'reserved_bytes.allocated': 0, 'reserved_bytes.current': 0, 'reserved_bytes.freed': 0,
            # 'reserved_bytes.peak': 0, 'segment.allocated': 0, 'segment.current': 0, 'segment.freed': 0, 'segment.peak': 0})
            "test_host_memory_stats",
            # THIS IS AN OLD ERROR
            # In file included from /home/tester/.cache/torch_extensions/py312_cpu/dummy_allocator/main_hip.cpp:5:
            # /home/tester/TheRock/.venv/lib/python3.12/site-packages/torch/include/ATen/hip/Exceptions.h:4:10: fatal error: hipblas/hipblas.h: No such file or directory
            #     4 | #include <hipblas/hipblas.h>
            #     |          ^~~~~~~~~~~~~~~~~~~
            # compilation terminated.
            # NEW ERROR
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_with_allocator",
            # Change detector test (Cublaslt vs Cublas depending on gcn_arch and torch version)
            # Always skip as this test is very basic and needs manual intervention for new architectures
            # See
            #   * https://github.com/ROCm/pytorch/pull/2742
            #   * https://github.com/ROCm/pytorch/pull/2873
            "test_preferred_blas_library_settings",
            # ----------------
            # maybe failing
            # ----------------
            # "test_hip_device_count"
            # "test_nvtx"
            # ----------------
        ],
        "nn": [
            # external-builds/pytorch/pytorch/test/test_nn.py::TestNN::test_RNN_dropout_state MIOpen(HIP): Error [Compile] 'hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data())' MIOpenDropoutHIP.cpp: HIPRTC_ERROR_COMPILATION (6)
            # MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6), source file: MIOpenDropoutHIP.cpp
            # MIOpen(HIP): Warning [BuildHip] In file included from /tmp/comgr-01c423/input/MIOpenDropoutHIP.cpp:32:
            # /tmp/comgr-01c423/include/miopen_rocrand.hpp:45:10: fatal error: 'rocrand/rocrand_xorwow.h' file not found
            # 45 | #include <rocrand/rocrand_xorwow.h>
            #     |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
            # 1 error generated when compiling for gfx942.
            # MIOpen Error: /therock/src/rocm-libraries/projects/miopen/src/hipoc/hipoc_program.cpp:299: Code object build failed. Source: MIOpenDropoutHIP.cpp
            "test_RNN_dropout_state",
            # AssertionError: "Input and parameter tensors are not at the same device" does not match "Expected all tensors
            # to be on the same device, but got weight is on cpu, different from other tensors on cuda:0 (when checking
            # argument in method wrapper_CUDA__miopen_rnn)"
            "test_rnn_check_device",
        ],
        "torch": [
            # FLAKY!! AssertionError: 'tensor([2.3000+4.j, 7.0000+6.j])' != 'tensor([2.30000+4.j, 7.00000+6.j])'
            # (Note: this will also skip "test_print" in all other test modules)
            "test_print",
        ],
        "unary_ufuncs": [
            # ----------------
            # maybe failing
            # ----------------
            # this passed on gfx942
            # "test_reference_numerics_large__refs_nn_functional_mish_cuda_float16",
            # "test_reference_numerics_large_nn_functional_mish_cuda_float16",
            # ----------------
            # AttributeError: 'NoneType' object has no attribute 'dtype'
            # it is all due to the same reason "expected" being None
            # in def _test_reference_numerics(self, dtype, op, tensors, equal_nan=True):
            # actual = op(t, **torch_kwargs)
            # expected = op.ref(a, **numpy_kwargs)
            # print("torch_kwargs", torch_kwargs, "t", t, "actual", actual)
            # print("numpy_kwargs", numpy_kwargs, "a", a, "expected", expected)
            # output:
            # torch_kwargs {} t tensor([inf, inf, inf, -inf, -inf, -inf, nan, nan, nan], device='cuda:0') actual tensor([0., 0., 0., 0., 0., 0., nan, nan, nan], device='cuda:0')
            # numpy_kwargs {} a [ inf  inf  inf -inf -inf -inf  nan  nan  nan] expected None
            "test_reference_numerics_extremal__refs_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_extremal__refs_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_extremal_special_airy_ai_cuda_float32",
            "test_reference_numerics_extremal_special_airy_ai_cuda_float64",
            "test_reference_numerics_extremal_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_extremal_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_large_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_large_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_large_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_large_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_large_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_bool",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_int8",
            "test_reference_numerics_normal__refs_special_spherical_bessel_j0_cuda_uint8",
            "test_reference_numerics_normal_special_airy_ai_cuda_bool",
            "test_reference_numerics_normal_special_airy_ai_cuda_float32",
            "test_reference_numerics_normal_special_airy_ai_cuda_float64",
            "test_reference_numerics_normal_special_airy_ai_cuda_int16",
            "test_reference_numerics_normal_special_airy_ai_cuda_int32",
            "test_reference_numerics_normal_special_airy_ai_cuda_int64",
            "test_reference_numerics_normal_special_airy_ai_cuda_int8",
            "test_reference_numerics_normal_special_airy_ai_cuda_uint8",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_bool",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_int8",
            "test_reference_numerics_normal_special_spherical_bessel_j0_cuda_uint8",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_int8",
            "test_reference_numerics_small__refs_special_spherical_bessel_j0_cuda_uint8",
            "test_reference_numerics_small_special_airy_ai_cuda_float32",
            "test_reference_numerics_small_special_airy_ai_cuda_float64",
            "test_reference_numerics_small_special_airy_ai_cuda_int16",
            "test_reference_numerics_small_special_airy_ai_cuda_int32",
            "test_reference_numerics_small_special_airy_ai_cuda_int64",
            "test_reference_numerics_small_special_airy_ai_cuda_int8",
            "test_reference_numerics_small_special_airy_ai_cuda_uint8",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_float32",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_float64",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_int16",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_int32",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_int64",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_int8",
            "test_reference_numerics_small_special_spherical_bessel_j0_cuda_uint8",
        ],
    },
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
            # This test JIT compiles and fails if MSVC is not installed on Windows:
            # We should fix the test to fail/skip more gracefully.
            #   subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
            "test_multi_grad_all_hooks",
            # This is/was also failing on gfx942 linux, see the 2.9 and 2.10 skip test files.
            #   AssertionError: "Simulate error" does not match "grad can be implicitly created only for scalar outputs"
            "test_reentrant_parent_error_on_cpu_cuda",
        ],
        "cuda": [
            # RuntimeError: miopenStatusUnknownError
            "test_autocast_rnn",
            # On some test runners
            #   AssertionError: False is not true
            #   self.assertTrue(abs(check_workspace_size(a) - default_workspace_size) < 524288)
            "test_cublas_workspace_explicit_allocation",
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
        ],
        "nn": [
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
            "test_memory_format_operators_cuda"
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
