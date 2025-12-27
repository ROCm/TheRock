# PyTorch 2.8 specific skip tests
# Tests moved to generic.py have been removed to avoid duplication
skip_tests = {
    "common": {
        "cuda": [
            # This test was fixed in torch 2.9, see
            # https://github.com/ROCm/TheRock/issues/2206
            "test_hip_device_count",
        ]
    },
    "gfx950": {
        "binary_ufuncs": [
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other___rpow___cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other__refs_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_contig_vs_every_other_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig___rpow___cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig__refs_pow_cuda_complex64",
            # AssertionError: Tensor-likes are not close!
            "test_non_contig_pow_cuda_complex64",
        ]
    },
}
