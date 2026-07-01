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
            # Distributed failures triaged from gfx94X-dcgpu 2.11 runs:
            # https://github.com/ROCm/TheRock/actions/runs/28242115579
            # https://github.com/ROCm/TheRock/actions/runs/28266889171
            # https://github.com/ROCm/TheRock/actions/runs/28447160155
            # https://github.com/ROCm/TheRock/actions/runs/29012657842 (2.11.0+rocm7.15.0a20260709)
            # ReplicateFullyShardInit - pytest-timeout (>900s)
            "test_replicate_device_id",
            # TestMultiProc - process join timeout (~300s)
            "test_get_pg_attr",
            # TestFullyShard1DTrainingCore - child exit code 10 (Scalars not close)
            "test_post_optim_event",
            # SymmMemCollectiveTest - pytest-timeout (>900s)
            "test_two_shot_all_reduce",
            # TestParityWithDDPCUDA - child exit code 10 (Scalars not close)
            "test_delayed_optim_step_offload_true_no_shard_cuda",
            "test_delayed_reduce_scatter_offload_true_no_shard_cuda",
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
