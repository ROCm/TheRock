skip_tests = [
    # Individual test function names
    "test_RNN_dropout_state",
    "test_print",
    "test_hip_device_count",
    "test_host_memory_stats",
    "test_nvtx",
    "test_mempool_with_allocator",
    "test_cuda_tensor_pow_scalar_tensor_cuda",

    # Reference numerics (known timeouts or instabilities)
    "test_reference_numerics_extremal__refs_special_spherical_bessel_j0_cuda_float32",
    "test_reference_numerics_extremal__refs_special_spherical_bessel_j0_cuda_float64",
    "test_reference_numerics_extremal_special_airy_ai_cuda_float32",
    "test_reference_numerics_extremal_special_airy_ai_cuda_float64",
    "test_reference_numerics_extremal_special_spherical_bessel_j0_cuda_float32",
    "test_reference_numerics_extremal_special_spherical_bessel_j0_cuda_float64",
    "test_reference_numerics_large__refs_nn_functional_mish_cuda_float16",
    "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_float32",
    "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_float64",
    "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int16",
    "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int32",
    "test_reference_numerics_large__refs_special_spherical_bessel_j0_cuda_int64",
    "test_reference_numerics_large_nn_functional_mish_cuda_float16",
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

    # Known failures and infrastructure issues
    "test_rnn_check_device",
    "test_device_count_not_cached_pre_init",

    # Explicitly deselected
    "test_unused_output_device_cuda",
    "test_pinned_memory_empty_cache",
]

# Detect import errors and skip full test files
try:
    import torch.testing._internal.inductor_utils  # from test_ops.py
except Exception:
    skip_tests.append("test_ops")

try:
    import torch._inductor.lowering  # from test_torchinductor.py
except Exception:
    skip_tests.append("test_torchinductor")

# Emit the -k expression
expr = "not " + " and not ".join(skip_tests)
print(expr)
