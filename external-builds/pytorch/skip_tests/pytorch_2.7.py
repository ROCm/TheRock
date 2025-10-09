skip_tests = {
    "common": {
        "cuda": [
            # Explicitly deselected since givind segfault
            "test_unused_output_device_cuda",  # this test does not exist in nightly anymore
            "test_pinned_memory_empty_cache",
            "test_float32_matmul_precision_get_set",
        ]
    },
}
