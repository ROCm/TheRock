skip_tests = {
    "common": {
        "cuda": [
            # Explicitly deselected since giving segfault
            "test_unused_output_device_cuda",  # this test does not exist in nightly anymore
            "test_pinned_memory_empty_cache",
            "test_float32_matmul_precision_get_set",
            # AssertionError: Tensor-likes are not close!
            # Mismatched elements: 1 / 327680 (0.0%)
            # Greatest absolute difference: 0.03125 at index (3, 114, 184) (up to 0.01 allowed)
            # Greatest relative difference: 0.01495361328125 at index (3, 114, 184) (up to 0.01 allowed)
            "test_index_add_correctness",
            "test_graph_concurrent_replay",
        ]
    },
}
