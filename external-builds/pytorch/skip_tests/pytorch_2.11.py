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
            # Dispositions/evidence: FIXES_FOR_triage_skips_2.11_0626.md
            # ReplicateFullyShardInit - pytest-timeout (>900s). Order-dependent
            # MultiProcContinuousTest hang; self-heals on rerun in a fresh
            # process. Dispatcher blocks on an untimed completion_queue.get()
            # after a worker dies (preceded by threaded-PG 'ThreadLocalWorld has
            # no attribute comms'). Proof: proofs/f1_replicate_device_id_multiproc_hang_stack.txt
            "test_replicate_device_id",
            # TestFullyShard1DTrainingCore - Scalars not close by ~1.72e-5 (allowed
            # 1e-5) at world_size=8. Benign fp reduction-order drift; NOT
            # reproducible locally (14/14 pass). Sibling class caps world_size to 2
            # citing the same drift.
            "test_post_optim_event",
            # SymmMemCollectiveTest - pytest-timeout (>900s). NOTE: the test body
            # PASSES; the hang is the *class* teardown (destroy_process_group ->
            # RCCL commReclaim/socketConnectCheck deadlock), NOT specific to
            # two-shot - test_one_shot_all_reduce alone hangs identically. Skipping
            # only two_shot is insufficient: whichever SymmMemCollectiveTest case
            # runs last will hit the same teardown deadlock. ROCm ticket:
            # proofs/ROCM_TICKET_rccl_symm_mem_teardown_deadlock.md
            "test_two_shot_all_reduce",
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
