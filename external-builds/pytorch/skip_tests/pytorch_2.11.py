skip_tests = {
    "common": {
        "autograd": [
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_logging",
        ],
        "cuda": [
            # passes on single run, crashes if run in a group
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_memory_compile_regions",
            # AssertionError: False is not true
            "test_memory_plots",
            # AssertionError: Booleans mismatch: False is not True
            "test_memory_plots_free_segment_stack",
            # FileNotFoundError: [Errno 2] No such file or directory: '/github/home/.cache//flamegraph.pl'
            "test_memory_snapshot",
            # AssertionError: String comparison failed: 'test_memory_snapshot' != 'foo'
            "test_memory_snapshot_script",
            # AssertionError: False is not true
            "test_memory_snapshot_with_cpp",
            # AssertionError: Scalars are not equal!
            "test_mempool_ctx_multithread",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_mempool_empty_cache_inactive",
            # RuntimeError: Error building extension 'dummy_allocator_v1'
            "test_mempool_limited_memory_with_allocator",
            # new for pytorch 2.11
            # RuntimeError: Error building extension 'dummy_allocator_v3'
            "test_tensor_delete_after_allocator_delete",
            # RuntimeError: Error building extension 'dummy_allocator'
            "test_deleted_mempool_not_used_on_oom",
            # ModuleNotFoundError: No module named 'torchvision'
            "test_resnet",
            # RuntimeError: miopenStatusUnknownError
            "test_graph_cudnn_dropout",
        ],
        "nn": [
            # new in 2.11
            # AssertionError: Scalars are not close!
            "test_CTCLoss_cudnn_cuda",
        ],
        "torch": [
            "test_cpp_warnings_have_python_context_cuda",
            # torch._dynamo.exc.BackendCompilerFailed: backend='aot_eager' raised:
            # TypeError: 'CustomDecompTable' object is not a mapping
            "test_fx_memory_profiler_augmentation",
        ],
    },
    "gfx120X-all": {
        "autograd": [
            # AssertionError: False is not true
            "test_side_stream_backward_overlap_cuda"
        ],
        # "unary_ufuncs": [
        #     # this failed only once. maybe python version dependent? probably the run was python 3.13
        #     # AssertionError: Tensor-likes are not close!
        #     "test_batch_vs_slicing_polygamma_polygamma_n_2_cuda_float16",
        # ],
    },
    # "windows": {
    #     empty for the moment
    # },
}
