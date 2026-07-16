## PyTorch 2.11 distributed UTs — branch-only skips (`skipping_failures_2.11_v2`)

**Source:** `git diff main -- external-builds/pytorch/skip_tests/pytorch_2.11.py`
**Platform:** gfx94X-dcgpu, PyTorch 2.11.0+rocm7.13.0a20260501
**CI runs aggregated:**
- https://github.com/ROCm/TheRock/actions/runs/25875103215
- https://github.com/ROCm/TheRock/actions/runs/25919759807
- https://github.com/ROCm/TheRock/actions/runs/26044254069
- https://github.com/ROCm/TheRock/actions/runs/26119178358
- https://github.com/ROCm/TheRock/actions/runs/26158117282

---

### Pytest 900s timeouts

- [ ] distributed/fsdp/test_fsdp_comm.py::TestExplicitUnshardCUDA::test_unshard_async_use_orig_params_True_cuda
- [ ] distributed/fsdp/test_fsdp_fine_tune.py::TestFSDPFineTuneCUDA::test_parity_with_ddp_cuda
- [ ] distributed/fsdp/test_fsdp_fine_tune.py::TestFSDPFineTuneCUDA::test_parity_with_non_frozen_fsdp_cuda
- [ ] distributed/fsdp/test_fsdp_hybrid_shard.py::TestFSDPHybridShard::test_fsdp_hybrid_shard_parity
- [ ] distributed/fsdp/test_fsdp_ignored_modules.py::TestFSDPIgnoredModules::test_ignored_modules_not_under_wrapped_root_ignore_modules_False
- [ ] distributed/fsdp/test_fsdp_misc.py::TestFSDPMiscMultiProcess::test_fsdp_device_id_use_index_False
- [ ] distributed/fsdp/test_fsdp_misc.py::TestFSDPMiscMultiProcess::test_fsdp_device_id_use_index_True
- [ ] distributed/fsdp/test_fsdp_misc.py::TestFSDPMiscMultiProcess::test_fsdp_module_no_compute_grad_use_second_layer_False_sharding_strategy0
- [ ] distributed/fsdp/test_wrap.py::TestFSDPWrap::test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_False_device_init_mode0
- [ ] distributed/fsdp/test_wrap.py::TestFSDPWrap::test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_False_device_init_mode1
- [ ] distributed/fsdp/test_wrap.py::TestFSDPWrap::test_main_wrap_api_cpu_offload0_backward_prefetch1_forward_prefetch_True_device_init_mode1
- [ ] distributed/fsdp/test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_configs0_use_orig_params_True
- [ ] distributed/fsdp/test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_configs1_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_configs1_use_orig_params_True
- [ ] distributed/fsdp/test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_cpu_offload_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_grad_acc.py::TestGradAcc::test_grad_acc_cpu_offload_use_orig_params_True
- [ ] distributed/fsdp/test_fsdp_sharded_grad_scaler.py::TestShardedGradScalerParityWithDDP::test_fsdp_ddp_parity_with_grad_scaler_offload_false_none_none_none
- [ ] distributed/fsdp/test_fsdp_sharded_grad_scaler.py::TestShardedGradScalerParityWithDDP::test_fsdp_ddp_parity_with_grad_scaler_offload_true_none_mixed_precision_none
- [ ] distributed/fsdp/test_fsdp_sharded_grad_scaler.py::TestShardedGradScalerParityWithDDP::test_fsdp_ddp_parity_with_grad_scaler_offload_true_none_none_none
- [ ] distributed/fsdp/test_fsdp_sharded_grad_scaler.py::TestShardedGradScalerParityWithDDP::test_sharded_grad_scaler_found_inf

### Process join timeout (~300s)

- [ ] distributed/fsdp/test_fsdp_comm_hooks.py::TestCommunicationHooks::test_fp16_hook_has_wrapping_True_sharding_strategy1
- [ ] distributed/tensor/test_dtensor_compile.py::TestDTensorCompileE2E::test_2d_fsdp_tp_compile_use_ca_True
- [ ] distributed/fsdp/test_fsdp_memory.py::TestFSDPMemory::test_fsdp_memory_ckpt_no_ckpt
- [ ] distributed/fsdp/test_fsdp_mixed_precision.py::TestFSDPMixedPrecisionSharded::test_mp_batchnorm_convert_sync_bn_True
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_full_optim_state_dict_keys
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_full_optim_state_dict_nested_invalid
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_optim_input_warning
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_False
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_optim_state_dict_nested_state_dict_type0_use_multiple_param_groups_False_rank0_only_False_use_diff_optim_inputs_True
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_rekey_optim_state_dict_to_ids_state_dict_type0_use_multiple_param_groups_False
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_rekey_optim_state_dict_to_ids_state_dict_type0_use_multiple_param_groups_True
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_rekey_optim_state_dict_to_names
- [ ] distributed/fsdp/test_fsdp_optim_state.py::TestFSDPOptimState::test_scatter_full_optim_state_dict_nested_halve_world_size
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_local_state_dict_cpu_offload1_fp16_False_state_dict_rank0_and_offload_False_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_mixed_precision.py::TestFSDPTrainEval::test_train_ema_eval_flow
- [ ] distributed/fsdp/test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsMultipleParamGroups::test_diff_trainability
- [ ] distributed/fsdp/test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsMultipleParamGroups::test_multiple_optimizers
- [ ] distributed/fsdp/test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsParamAccess::test_access_params_after_forward
- [ ] distributed/fsdp/test_fsdp_use_orig_params.py::TestFSDPUseOrigParamsUnshardReshard::test_summon_between_two_forwards_offload_params_False
- [ ] distributed/fsdp/test_fsdp_freezing_weights.py::TestFreezingWeights::test_freezing_weights_with_nested_trunk_True_freezing_method_FreezingMethod_GradToNone_freeze_after_wrap_fsdp_False_disable_autograd_False_forward_prefetch_True
- [ ] distributed/checkpoint/fsdp/test_fsdp_dsd.py::TestFullyShardWithDistributedStateDict::test_save_with_fsdp2_tp_and_load_with_tp
- [ ] distributed/algorithms/test_join.py::TestJoin::test_join_kwargs
- [ ] distributed/algorithms/test_join.py::TestJoin::test_multiple_joinables
- [ ] distributed/fsdp/test_fsdp_core.py::TestNoGradCUDA::test_transformer_no_grad_mixed_precision_False_cuda
- [ ] distributed/fsdp/test_fsdp_core.py::TestParityWithDDPCUDA::test_nested_always_wrap_model_offload_false_none_cuda
- [ ] distributed/fsdp/test_fsdp_core.py::TestParityWithDDPCUDA::test_nested_wrapped_model_offload_false_none_cuda
- [ ] distributed/fsdp/test_fsdp_core.py::TestParityWithDDPCUDA::test_nested_wrapped_model_offload_true_none_cuda
- [ ] distributed/fsdp/test_fsdp_core.py::TestParityWithDDPCUDA::test_transformer_offload_true_none_cuda

### Tensor parity mismatch (Tensor-likes not close)

- [ ] distributed/test_distributed_spawn.py::TestDistBackendWithSpawn::test_ddp_apply_optim_in_backward_grad_as_bucket_view_false
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_local_state_dict_cpu_offload0_fp16_False_state_dict_rank0_and_offload_False_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_sharded_state_dict_cpu_offload0_fp16_False_state_dict_rank0_and_offload_False_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_sharded_state_dict_cpu_offload0_fp16_True_state_dict_rank0_and_offload_False_use_orig_params_True
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_state_dict_cpu_offload1_fp16_True_state_dict_rank0_and_offload_False_use_orig_params_False
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_basic_save_and_load_state_dict_state_dict_type_state_dict_cpu_offload1_fp16_True_state_dict_rank0_and_offload_False_use_orig_params_True
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_state_dict_load_into_local_module_state_dict_type_sharded_state_dict_state_dict_rank0_and_offload_False_fsdp_root_False
- [ ] distributed/fsdp/test_fsdp_state_dict.py::TestFSDPStateDict::test_state_dict_load_into_local_module_state_dict_type_state_dict_state_dict_rank0_and_offload_False_fsdp_root_False

### Scalar close mismatch

- [ ] distributed/fsdp/test_fsdp_core.py::TestParityWithDDPCUDA::test_transformer_offload_true_no_shard_cuda

### Scalar equality mismatch

- [ ] distributed/test_dynamo_distributed.py::TestMultiProc::test_compiler_collectives_automatic_dynamic_tensor

### AssertionError (False is not True)

- [ ] distributed/fsdp/test_fsdp_meta.py::TestFSDPWithMetaDevice::test_nested_model_with_meta_device_reset_params_auto_wrap_True

### Subprocess killed by signal

- [ ] distributed/test_distributed_spawn.py::TestDistBackendWithSpawn::test_ddp_buffer_hook_allreduce_return_future

### RuntimeError not raised in child (exit 10)

- [ ] distributed/test_distributed_spawn.py::TestDistBackendWithSpawn::test_monitored_barrier_allreduce_hang
- [ ] distributed/test_distributed_spawn.py::TestDistBackendWithSpawn::test_monitored_barrier_allreduce_hang_wait_all_ranks

---

**Total: 62 unique tests** aggregated across the 5 CI runs above
