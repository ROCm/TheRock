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
            # Distributed failures after un-skipping the legacy FSDP/timeout set.
            # The old FSDP/fully_shard timeouts no longer reproduce on this build;
            # the remaining failures cluster around symmetric memory / copy engine.
            # gfx94X-dcgpu distributed full suite, PyTorch 2.11 + rocm7.14:
            # https://github.com/ROCm/TheRock/actions/runs/28050755542 (users/albmalamd/unskip_distributed_for_pytorch_2.11)

            # --- Symmetric memory / NCCL copy engine (Exception in worker process) ---
            "(NCCLCopyEngineCollectives and test_ce_alltoall)",
            "(NCCLCopyEngineCollectives and test_ce_allgather)",
            "(SymmMemPoolTest and test_mempool_tensor_factory)",
            "(SymmMemPoolTest and test_mempool_compute_ops)",
            "(SymmetricMemoryTest and test_allow_overlapping_devices)",
            "(SymmetricMemoryTest and test_dispatcher_torchbind_symmetric_memory)",
            "(SymmetricMemoryTest and test_set_signal_pad_size_with_allocation)",
            "(SymmetricMemoryTest and test_get_signal_pad)",
            "(SymmetricMemoryTest and test_large_alloc)",
            "(SymmetricMemoryTest and test_subgroup)",
            # Collapsed: covers symm_mem_input_{True,False}
            "(SymmetricMemoryTest and test_low_contention_all_gather)",
            # Collapsed: covers reduce_op_{avg,sum} x symm_mem_input_{True,False}
            "(SymmetricMemoryTest and test_low_contention_reduce_scatter)",
            "(SymmMemCollectiveTest and test_one_shot_all_reduce)",
            "(SymmMemCollectiveTest and test_two_shot_all_reduce)",
            # Collapsed: covers test_reduce_scatter{,_corner_cases}
            "(SymmMemCollectiveTest and test_reduce_scatter)",

            # --- Symmetric memory P2P (child exited with error code 10) ---
            # Collapsed: covers test_empty_strided_p2p{,_persistent}_set_device_{True,False}
            "(SymmMemEmptySetDeviceTest and test_empty_strided_p2p)",

            # --- Symmetric memory (CUDA out of memory) ---
            "(SymmMemSingleProcTest and test_memset32)",

            # --- Numerical / parity (child exit 10, Tensor-likes not close) ---
            "(ReplicateTest and test_compile_gpu_ac)",
            "(ReplicateTest and test_compile_fp16)",
            "(TestZeroRedundancyOptimizerDistributed and test_ddp_zero_overlap_use_gpu_True_use_interleaved_hook_False_gradient_as_bucket_view_True_static_graph_True_shard_buckets_True)",
            "(CPFlexAttentionTest and test_cp_flex_attention_document_mask)",
            "(CPFlexAttentionTest and test_cp_flex_attention_causal_mask)",

            # --- Timeouts (pytest-timeout, multiprocessing join) ---
            "(ReplicateFullyShardInit and test_replicate_device_id)",
            "(TestMultiProc and test_get_pg_attr)",
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
