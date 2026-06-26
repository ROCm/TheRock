# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

skip_tests = {
    "common": {
        "cuda": [
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
            # ModuleNotFoundError: No module named 'torchvision'
            "test_resnet",
            # RuntimeError: miopenStatusUnknownError
            "test_graph_cudnn_dropout",
            # Fatal Python error: Segmentation fault - https://github.com/ROCm/TheRock/issues/4745
            "test_snapshot_include_traces",
        ],
        "nn": [
            # new in 2.11
            # AssertionError: Scalars are not close!
            "test_CTCLoss_cudnn_cuda",
            # AssertionError: Tensor-likes are not close! - https://github.com/ROCm/TheRock/issues/4744
            # Failed on gfx1151 and gfx942 (only with python 3.13)
            "test_Embedding_discontiguous_cuda",
        ],
        "torch": [
            "test_cpp_warnings_have_python_context_cuda",
        ],
        "distributed": [
            # Error while creating shared memory segment /dev/shm/nccl-VPyhzw (size 21823872), error: No space left on device (28)
            "test_3d_mlp_with_nd_mesh",
            # AssertionError: False is not true : cuda:0 used 2615148544.0 bytes after collective, 70% more than the status before (1495269376.0 bytes). Extra CUDA context may have been created.
            "test_extra_cuda_context",
            # Distributed failures triaged from gfx94X-dcgpu 2.11 run:
            # https://github.com/ROCm/TheRock/actions/runs/28242115579
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_device_id",
            # TestMultiProc - process join timeout (~300s)
            "test_get_pg_attr",
            # TestFullyShard1DTrainingCore - child exit code 10 (Scalars not close)
            "test_post_optim_event",
            # SymmMemEmptySetDeviceTest - child exit code 10
            # Collapsed: covers set_device_{True,False} x persistent variants
            "test_empty_strided_p2p",
            # SymmetricMemoryTest / SymmMemCollectiveTest / SymmMemPoolTest /
            # NCCLCopyEngineCollectives - Exception in worker process
            # Collapsed: covers reduce_op_{sum,avg} x symm_mem_input_{True,False}
            "test_low_contention_reduce_scatter",
            # Collapsed: covers symm_mem_input_{True,False}
            "test_low_contention_all_gather",
            "test_allow_overlapping_devices",
            "test_dispatcher_torchbind_symmetric_memory",
            "test_set_signal_pad_size_with_allocation",
            "test_large_alloc",
            "test_get_signal_pad",
            "test_subgroup",
            "test_one_shot_all_reduce",
            "test_two_shot_all_reduce",
            # Collapsed: covers test_reduce_scatter and test_reduce_scatter_corner_cases
            "test_reduce_scatter",
            "test_mempool_tensor_factory",
            "test_mempool_compute_ops",
            "test_ce_allgather",
            "test_ce_alltoall",
            # SymmMemSingleProcTest - CUDA error: out of memory
            "test_memset32",
        ],
    },
    "gfx942": {
        "cuda": [
            # new test
            # AssertionError: Scalars are not equal!
            "test_graph_capture_reclaim_shared_pool",
        ],
    },
    # "gfx120": {
    #     "unary_ufuncs": [
    #         # this failed only once. maybe python version dependent? probably the run was python 3.13
    #         # AssertionError: Tensor-likes are not close!
    #         "test_batch_vs_slicing_polygamma_polygamma_n_2_cuda_float16",
    #     ],
    # },
    # "windows": {
    #     empty for the moment
    # },
}
