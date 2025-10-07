skip_tests = {
    "gfx950": {
        "cuda": {
            "test_preferred_blas_library_settings",
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
            # "test_graph_concurrent_replay",  # flaky on gfx942!! not anymore?
            # ----------------
            # maybe failing
            # ----------------
            # "test_hip_device_count"
            # "test_nvtx"
            #
            # gfx942
            # TestCuda under test_cuda.py, failing on gfx942 (#1143) --> not on sharkmi300x-4
            #    "test_float32_matmul_precision_get_set ",
            #
            # Explicitly deselected since givind segfault
            #    "test_unused_output_device_cuda",  # this test does not exist in nightly anymore
            #    "test_pinned_memory_empty_cache",
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
            # ----------------
            # maybe failing
            # ----------------
            # "test_index_add_correctness",
            # ----------------
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
}
