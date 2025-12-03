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
            # passes on single run, crashes if run in a group
            "test_memory_compile_regions",
            "test_memory_plots",
            "test_memory_plots_free_segment_stack",
            #  FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmprlowr8yn.pl'
            "test_memory_snapshot",
            "test_memory_snapshot_script",
            "test_memory_snapshot_with_cpp",
            "test_mempool_ctx_multithread",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_empty_cache_inactive",
            # RuntimeError: Error building extension 'dummy_allocator_v1'
            "test_mempool_limited_memory_with_allocator",
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
