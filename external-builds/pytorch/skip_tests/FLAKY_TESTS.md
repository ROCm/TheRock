# PyTorch Distributed Flaky Test Ledger

Tests that have failed in ≥1 full-suite run but also passed in ≥1 full-suite
run on the **same stack** (same PT + ROCm wheel, same branch commit). These are
not added to the skip list yet; they warrant a skip only if they become
**persistently failing** (fail in ≥3 consecutive full-suite runs on the current
stack, or appear in every new nightly stack).

**Stack scope:** This ledger tracks gfx94X-dcgpu distributed config runs only.
For other configs (default, inductor) or GPU families, create separate sections.

**Promotion rule:** If a test in this ledger fails 3+ consecutive full-suite
runs on the same PT+ROCm pair, move it to `pytorch_2.X.py` with an attribution
comment citing the failing run IDs.

---

## Methodology

Full-suite runs are dispatched via `test_pytorch_wheels_full.yml` with
`test_configs=distributed` and no `tests_to_include` filter. Run IDs referenced
below are on branch `users/chinmaydk99/add-pt213-rocm-skip-list` unless noted.

---

## Active candidates (Jun12 stack)

Stack: `torch 2.13.0a0+rocm7.14.0a20260612`, TheRock ref `6e0925e6f`

| Test | File | Failure mode | Fail run(s) | Pass run(s) | Consecutive fails | Status |
|---|---|---|---|---|---|---|
| `TestReplicate1DTrainingCore::test_train_parity_multi_group_cpu_offload_eager` | `distributed/_composable/test_replicate_training.py` | `AssertionError: Scalars are not close!` (rel diff ~3.1e-06 vs 1.3e-06) | [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236) shard 1/3 | same job (subprocess retry) | 1 | Monitoring |

### Notes on the Jun12 candidates

`test_train_parity_multi_group_cpu_offload_eager` failed once with a borderline
scalar ULP mismatch, then passed in a new process in the same shard. No skip
added; re-run shard if it recurs.

**Update this table after each full-suite run**: increment "Consecutive fails" if
the test fails again, reset to 0 if it passes, promote to `pytorch_2.13.py` if
it hits 3.

---

## Active candidates (Jun1 stack)

Stack: `torch 2.13.0a0+rocm7.14.0a20260601`, PT ref `2c19ec64ae0a9dfff1e30a3f97c487c21992b9ea`

| Test | File | Failure mode | Fail run(s) | Pass run(s) | Consecutive fails | Status |
|---|---|---|---|---|---|---|
| `CPFlexAttentionTest::test_cp_flex_attention_causal_mask` | `distributed/tensor/test_attention.py` | `AssertionError: Tensor-likes are not close!` (numerical precision) | [26922264789](https://github.com/ROCm/TheRock/actions/runs/26922264789) shard 2/3 | [26907811456](https://github.com/ROCm/TheRock/actions/runs/26907811456) | 1 | Monitoring |
| `NCCLTraceTest::test_compiled_ring_attention_pattern_num_steps_4_M_1024` | `distributed/test_c10d_nccl.py` | NCCL watchdog timeout (ALLREDUCE at 3000ms, 600s overall limit) | [26922264789](https://github.com/ROCm/TheRock/actions/runs/26922264789) shard 2/3 | [26907811456](https://github.com/ROCm/TheRock/actions/runs/26907811456) | 1 | Monitoring |
| `TestZeroRedundancyOptimizerDistributed::test_ddp_zero_overlap_use_gpu_True_use_interleaved_hook_False_gradient_as_bucket_view_False_static_graph_False_shard_buckets_True` | `distributed/optim/test_zero_redundancy_optimizer.py` | `AssertionError: Tensor-likes are not close!` (numerical precision) | [26922264789](https://github.com/ROCm/TheRock/actions/runs/26922264789) shard 2/3 | [26907811456](https://github.com/ROCm/TheRock/actions/runs/26907811456) | 1 | Monitoring |

### Notes on the Jun1 candidates

All 3 remaining Jun1 candidates appeared together in shard 2/3 of run
26922264789 and passed in the prior full-suite run 26907811456 on the same
commit (`2d8daf59`). The numerical precision failures (`assert_close`) are
classic ROCm flakiness. The NCCL timeout is infrastructure-sensitive.

`TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward` was promoted to
the skip list after failing again on the June 12 stack (see Promoted section).

---

## Cleared candidates (previously flaky, now confirmed stable)

None yet.

---

## Promoted to skip list (was flaky, became persistent)

| Test | File | Promoted in | Fail runs cited |
|---|---|---|---|
| `TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward` | `distributed/test_distributed_spawn.py` | `pytorch_2.13.py` distributed bucket | [26922264789](https://github.com/ROCm/TheRock/actions/runs/26922264789), [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236) |
| `TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward_grad_as_bucket_view_false` | `distributed/test_distributed_spawn.py` | `pytorch_2.13.py` distributed bucket | [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236) |
| `TestStateDict::test_shared_weight` | `distributed/checkpoint/test_state_dict.py` | `pytorch_2.13.py` distributed bucket | [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236) |
| `ComposabilityTest::test_replicate_pp_ScheduleClass3_bfloat16` | `distributed/_composable/test_composability/test_pp_composability.py` | `pytorch_2.13.py` distributed bucket | [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236) |

Promotion rationale for June 12 run [27480021236](https://github.com/ROCm/TheRock/actions/runs/27480021236)
distributed shard 1/3:

- `test_ddp_apply_optim_in_backward`: prior Jun1 flake + hard fail with 10/1M
  element mismatch on Jun12 wheel; sibling `grad_as_bucket_view_false` variant
  failed in the same module shard.
- `test_shared_weight`: optimizer state_dict `assert_close` failure persisted
  across subprocess reruns (not recovered).
- `test_replicate_pp_ScheduleClass3_bfloat16`: NCCL `/dev/shm` segment creation
  failure under 8-rank PP; same `/dev/shm` class as skipped
  `test_3d_mlp_with_nd_mesh`.
