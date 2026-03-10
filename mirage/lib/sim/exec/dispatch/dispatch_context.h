#ifndef MIRAGE_SIM_EXEC_DISPATCH_DISPATCH_CONTEXT_H_
#define MIRAGE_SIM_EXEC_DISPATCH_DISPATCH_CONTEXT_H_

#include <cstdint>

namespace mirage::sim::exec {

struct DispatchContext {
  std::uint64_t dispatch_id = 0;
  std::uint64_t queue_id = 0;
  std::uint64_t kernel_id = 0;
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t block_x = 1;
  std::uint32_t block_y = 1;
  std::uint32_t block_z = 1;
};

enum class SyntheticKernelOpcode {
  kNop,
  kFill32,
  kVectorAddI32,
  kGfx950Program,
};

struct SyntheticDispatchArgs {
  std::uint64_t src0_va = 0;
  std::uint64_t src1_va = 0;
  std::uint64_t dst_va = 0;
  std::uint32_t element_count = 0;
  std::uint32_t immediate_u32 = 0;
  std::uint64_t code_va = 0;
  std::uint32_t code_word_count = 0;
  std::uint32_t wave_count = 1;
  // Optional exec mask buffer layout: uint64_t[wave_count].
  std::uint64_t exec_mask_va = 0;
  // SGPR buffer layout: uint32_t[wave_count][sgpr_state_count].
  std::uint64_t sgpr_state_va = 0;
  std::uint32_t sgpr_state_count = 0;
  // VGPR buffer layout:
  // uint32_t[wave_count][vgpr_state_count][WaveExecutionState::kLaneCount].
  std::uint64_t vgpr_state_va = 0;
  std::uint32_t vgpr_state_count = 0;
  std::uint64_t exec_mask = ~0ULL;
};

struct SyntheticDispatchPacket {
  DispatchContext context;
  SyntheticKernelOpcode opcode = SyntheticKernelOpcode::kNop;
  SyntheticDispatchArgs args;
};

struct CompletionRecord {
  std::uint64_t dispatch_id = 0;
  bool completed = false;
  bool success = false;
};

}  // namespace mirage::sim::exec

#endif  // MIRAGE_SIM_EXEC_DISPATCH_DISPATCH_CONTEXT_H_
