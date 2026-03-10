#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "lib/sim/isa/instruction_catalog.h"
#include "lib/sim/isa/wave_execution_state.h"
#include "lib/sim/single_gpu_simulator.h"

namespace {

template <typename T>
std::span<const std::byte> AsBytes(const std::vector<T>& values) {
  return std::as_bytes(std::span<const T>(values.data(), values.size()));
}

template <typename T>
std::span<std::byte> AsWritableBytes(std::vector<T>& values) {
  return std::as_writable_bytes(std::span<T>(values.data(), values.size()));
}

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

std::optional<std::uint32_t> FindDefaultEncodingOpcode(
    std::string_view instruction_name,
    std::string_view encoding_name) {
  using namespace mirage::sim::isa;

  const InstructionSpec* instruction = FindGfx950Instruction(instruction_name);
  if (instruction == nullptr) {
    return std::nullopt;
  }
  for (const InstructionEncodingSpec& encoding : GetEncodings(*instruction)) {
    if (encoding.encoding_name == encoding_name &&
        encoding.encoding_condition == "default") {
      return encoding.opcode;
    }
  }
  return std::nullopt;
}

constexpr std::uint32_t SetBits(std::uint32_t word,
                                std::uint32_t value,
                                std::uint32_t bit_offset,
                                std::uint32_t bit_count) {
  const std::uint32_t mask =
      (bit_count == 32) ? 0xffffffffu : ((1u << bit_count) - 1u);
  return word | ((value & mask) << bit_offset);
}

constexpr std::uint32_t MakeSopp(std::uint32_t op, std::uint32_t simm16 = 0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17f, 23, 9);
  word = SetBits(word, op, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

constexpr std::uint32_t MakeSop1(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t ssrc0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17d, 23, 9);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, op, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
  return word;
}

constexpr std::uint32_t MakeSop2(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t ssrc0,
                                 std::uint32_t ssrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x2, 30, 2);
  word = SetBits(word, op, 23, 7);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, ssrc1, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
  return word;
}

constexpr std::uint32_t MakeSopc(std::uint32_t op,
                                 std::uint32_t ssrc0,
                                 std::uint32_t ssrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17e, 23, 9);
  word = SetBits(word, op, 16, 7);
  word = SetBits(word, ssrc1, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
  return word;
}

constexpr std::uint32_t MakeVop2(std::uint32_t op,
                                 std::uint32_t vdst,
                                 std::uint32_t src0,
                                 std::uint32_t vsrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x0, 31, 1);
  word = SetBits(word, op, 25, 6);
  word = SetBits(word, vdst, 17, 8);
  word = SetBits(word, vsrc1, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

std::array<std::uint32_t, 2> MakeDs(std::uint32_t op,
                                    std::uint32_t vdst,
                                    std::uint32_t addr,
                                    std::uint32_t data0,
                                    std::uint32_t data1,
                                    std::uint32_t offset0,
                                    std::uint32_t offset1 = 0,
                                    bool gds = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(offset0 & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(offset1 & 0xffu) << 8;
  word |= static_cast<std::uint64_t>(gds ? 1u : 0u) << 16;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 17;
  word |= static_cast<std::uint64_t>(0x36u) << 26;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data0 & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(data1 & 0xffu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmem(std::uint32_t op,
                                      std::uint32_t sdata,
                                      std::uint32_t sbase_start,
                                      bool imm,
                                      std::uint32_t offset_or_soffset,
                                      bool soffset_en = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1) << 0;
  word |= static_cast<std::uint64_t>(sdata) << 6;
  word |= static_cast<std::uint64_t>(soffset_en ? 1u : 0u) << 14;
  word |= static_cast<std::uint64_t>(imm ? 1u : 0u) << 17;
  word |= static_cast<std::uint64_t>(op) << 18;
  if (imm) {
    word |= static_cast<std::uint64_t>(offset_or_soffset & 0x1fffffu) << 32;
  } else if (soffset_en) {
    word |= static_cast<std::uint64_t>(offset_or_soffset & 0x7fu) << 57;
  }
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

}  // namespace

int main() {
  using namespace mirage::sim;

  gpu::GpuProperties properties;
  properties.arch_name = "CDNA1";
  properties.gfx_target = "gfx908";
  properties.compute_units = 120;
  properties.hbm_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;

  SingleGpuSimulator simulator(properties);
  const queue::QueueId queue_id = simulator.CreateComputeQueue();
  const auto queue_state = simulator.GetQueue(queue_id);
  if (!Expect(queue_state.has_value(), "expected compute queue to exist") ||
      !Expect(queue_state->descriptor.type == queue::QueueType::kCompute,
              "expected compute queue descriptor")) {
    return 1;
  }

  const std::vector<std::int32_t> lhs{1, 2, 3, 4};
  const std::vector<std::int32_t> rhs{10, 20, 30, 40};
  const std::vector<std::int32_t> expected_sum{11, 22, 33, 44};

  const auto lhs_alloc = simulator.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                                  lhs.size() * sizeof(lhs[0]));
  const auto rhs_alloc = simulator.AllocateMemory(memory::MemoryRegionKind::kHbm,
                                                  rhs.size() * sizeof(rhs[0]));
  const auto dst_alloc =
      simulator.AllocateMemory(memory::MemoryRegionKind::kHbm,
                               expected_sum.size() * sizeof(expected_sum[0]));

  if (!Expect(lhs_alloc.mapped_va != 0, "expected lhs allocation") ||
      !Expect(rhs_alloc.mapped_va != 0, "expected rhs allocation") ||
      !Expect(dst_alloc.mapped_va != 0, "expected dst allocation") ||
      !Expect(simulator.WriteMemory(lhs_alloc.mapped_va, AsBytes(lhs)),
              "expected lhs write to succeed") ||
      !Expect(simulator.WriteMemory(rhs_alloc.mapped_va, AsBytes(rhs)),
              "expected rhs write to succeed")) {
    return 1;
  }

  exec::SyntheticDispatchPacket vector_add;
  vector_add.context.queue_id = queue_id;
  vector_add.opcode = exec::SyntheticKernelOpcode::kVectorAddI32;
  vector_add.args.src0_va = lhs_alloc.mapped_va;
  vector_add.args.src1_va = rhs_alloc.mapped_va;
  vector_add.args.dst_va = dst_alloc.mapped_va;
  vector_add.args.element_count = lhs.size();

  const exec::CompletionRecord vector_add_completion =
      simulator.Submit(queue_id, vector_add);
  if (!Expect(vector_add_completion.completed,
              "expected vector-add dispatch to complete") ||
      !Expect(vector_add_completion.success,
              "expected vector-add dispatch to succeed")) {
    return 1;
  }

  std::vector<std::int32_t> observed_sum(expected_sum.size(), 0);
  if (!Expect(simulator.ReadMemory(dst_alloc.mapped_va, AsWritableBytes(observed_sum)),
              "expected dst read to succeed") ||
      !Expect(observed_sum == expected_sum,
              "expected vector-add result to match")) {
    return 1;
  }

  const auto fill_alloc =
      simulator.AllocateMemory(memory::MemoryRegionKind::kHbm, 4 * sizeof(std::uint32_t));
  exec::SyntheticDispatchPacket fill32;
  fill32.context.queue_id = queue_id;
  fill32.opcode = exec::SyntheticKernelOpcode::kFill32;
  fill32.args.dst_va = fill_alloc.mapped_va;
  fill32.args.element_count = 4;
  fill32.args.immediate_u32 = 7;

  const exec::CompletionRecord fill_completion = simulator.Submit(queue_id, fill32);
  if (!Expect(fill_completion.completed, "expected fill dispatch to complete") ||
      !Expect(fill_completion.success, "expected fill dispatch to succeed")) {
    return 1;
  }

  std::vector<std::uint32_t> observed_fill(4, 0);
  if (!Expect(simulator.ReadMemory(fill_alloc.mapped_va, AsWritableBytes(observed_fill)),
              "expected fill read to succeed") ||
      !Expect(std::all_of(observed_fill.begin(), observed_fill.end(),
                          [](std::uint32_t value) { return value == 7; }),
              "expected fill values to match")) {
    return 1;
  }

  const auto s_mov_b32_opcode =
      FindDefaultEncodingOpcode("S_MOV_B32", "ENC_SOP1");
  const auto s_store_dword_opcode =
      FindDefaultEncodingOpcode("S_STORE_DWORD", "ENC_SMEM");
  if (!Expect(s_mov_b32_opcode.has_value(), "expected s_mov_b32 opcode lookup") ||
      !Expect(s_store_dword_opcode.has_value(),
              "expected s_store_dword opcode lookup")) {
    return 1;
  }

  const auto gfx_store_alloc =
      simulator.AllocateMemory(memory::MemoryRegionKind::kHbm, sizeof(std::uint32_t));
  const std::uint32_t output_low =
      static_cast<std::uint32_t>(gfx_store_alloc.mapped_va);
  const std::uint32_t output_high =
      static_cast<std::uint32_t>(gfx_store_alloc.mapped_va >> 32);
  const auto store_word =
      MakeSmem(*s_store_dword_opcode, 4, 0, true, 0);
  const std::vector<std::uint32_t> gfx_program_words = {
      MakeSop1(*s_mov_b32_opcode, 0, 255), output_low,
      MakeSop1(*s_mov_b32_opcode, 1, 255), output_high,
      MakeSop1(*s_mov_b32_opcode, 4, 255), 0x12345678u,
      store_word[0], store_word[1],
      MakeSopp(1),
  };
  const auto code_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      gfx_program_words.size() * sizeof(gfx_program_words[0]));
  if (!Expect(gfx_store_alloc.mapped_va != 0,
              "expected gfx store allocation") ||
      !Expect(code_alloc.mapped_va != 0, "expected code allocation") ||
      !Expect(simulator.WriteMemory(code_alloc.mapped_va, AsBytes(gfx_program_words)),
              "expected gfx program write to succeed")) {
    return 1;
  }

  exec::SyntheticDispatchPacket gfx_program_dispatch;
  gfx_program_dispatch.context.queue_id = queue_id;
  gfx_program_dispatch.opcode = exec::SyntheticKernelOpcode::kGfx950Program;
  gfx_program_dispatch.args.code_va = code_alloc.mapped_va;
  gfx_program_dispatch.args.code_word_count = gfx_program_words.size();
  gfx_program_dispatch.args.exec_mask = 0x1ULL;

  const exec::CompletionRecord gfx_program_completion =
      simulator.Submit(queue_id, gfx_program_dispatch);
  if (!Expect(gfx_program_completion.completed,
              "expected gfx950 dispatch to complete") ||
      !Expect(gfx_program_completion.success,
              "expected gfx950 dispatch to succeed")) {
    return 1;
  }

  const auto stored_value =
      simulator.ReadObject<std::uint32_t>(gfx_store_alloc.mapped_va);
  if (!Expect(stored_value.has_value(), "expected gfx950 store read") ||
      !Expect(*stored_value == 0x12345678u,
              "expected gfx950 program to write memory")) {
    return 1;
  }

  const SingleGpuSimulator::DecodeCacheStats initial_cache_stats =
      simulator.GetDecodeCacheStats();
  if (!Expect(initial_cache_stats.misses == 1,
              "expected first gfx950 dispatch to populate decode cache") ||
      !Expect(initial_cache_stats.hits == 0,
              "expected no decode-cache hits after first gfx950 dispatch") ||
      !Expect(initial_cache_stats.entries == 1,
              "expected one decode-cache entry after first gfx950 dispatch")) {
    return 1;
  }

  const exec::CompletionRecord cached_gfx_program_completion =
      simulator.Submit(queue_id, gfx_program_dispatch);
  if (!Expect(cached_gfx_program_completion.completed,
              "expected cached gfx950 dispatch to complete") ||
      !Expect(cached_gfx_program_completion.success,
              "expected cached gfx950 dispatch to succeed")) {
    return 1;
  }

  const SingleGpuSimulator::DecodeCacheStats cached_stats =
      simulator.GetDecodeCacheStats();
  if (!Expect(cached_stats.misses == 1,
              "expected cached gfx950 dispatch to avoid decode miss") ||
      !Expect(cached_stats.hits == 1,
              "expected cached gfx950 dispatch to record decode-cache hit") ||
      !Expect(cached_stats.entries == 1,
              "expected decode-cache entry count to remain stable")) {
    return 1;
  }

  const std::vector<std::uint32_t> revised_gfx_program_words = {
      MakeSop1(*s_mov_b32_opcode, 0, 255), output_low,
      MakeSop1(*s_mov_b32_opcode, 1, 255), output_high,
      MakeSop1(*s_mov_b32_opcode, 4, 255), 0xabcdef01u,
      store_word[0], store_word[1],
      MakeSopp(1),
  };
  if (!Expect(simulator.WriteMemory(code_alloc.mapped_va, AsBytes(revised_gfx_program_words)),
              "expected revised gfx program write to succeed")) {
    return 1;
  }

  const exec::CompletionRecord revised_gfx_program_completion =
      simulator.Submit(queue_id, gfx_program_dispatch);
  if (!Expect(revised_gfx_program_completion.completed,
              "expected revised gfx950 dispatch to complete") ||
      !Expect(revised_gfx_program_completion.success,
              "expected revised gfx950 dispatch to succeed")) {
    return 1;
  }

  const auto revised_stored_value =
      simulator.ReadObject<std::uint32_t>(gfx_store_alloc.mapped_va);
  const SingleGpuSimulator::DecodeCacheStats revised_stats =
      simulator.GetDecodeCacheStats();
  if (!Expect(revised_stored_value.has_value(),
              "expected revised gfx950 store read") ||
      !Expect(*revised_stored_value == 0xabcdef01u,
              "expected revised gfx950 program to update memory") ||
      !Expect(revised_stats.misses == 2,
              "expected rewritten code buffer to force decode miss") ||
      !Expect(revised_stats.hits == 1,
              "expected decode-cache hit count to remain stable after rewrite")) {
    return 1;
  }

  const auto s_add_u32_opcode =
      FindDefaultEncodingOpcode("S_ADD_U32", "ENC_SOP2");
  const auto v_add_u32_opcode =
      FindDefaultEncodingOpcode("V_ADD_U32", "ENC_VOP2");
  if (!Expect(s_add_u32_opcode.has_value(), "expected s_add_u32 opcode lookup") ||
      !Expect(v_add_u32_opcode.has_value(),
              "expected v_add_u32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> seeded_program_words = {
      MakeSop2(*s_add_u32_opcode, 2, 0, 1),
      MakeVop2(*v_add_u32_opcode, 2, 256, 1),
      MakeSopp(1),
  };
  const auto seeded_code_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      seeded_program_words.size() * sizeof(seeded_program_words[0]));
  std::vector<std::uint32_t> sgpr_state = {10u, 20u, 0u};
  const auto sgpr_state_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm, sgpr_state.size() * sizeof(sgpr_state[0]));
  std::vector<std::uint32_t> vgpr_state(
      3 * isa::WaveExecutionState::kLaneCount, 0u);
  vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 0] = 7u;
  vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 1] = 2u;
  vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 3] = 11u;
  vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 0] = 35u;
  vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 1] = 4u;
  vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 3] = 13u;
  vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 2] = 0xdeadbeefu;
  const auto vgpr_state_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm, vgpr_state.size() * sizeof(vgpr_state[0]));
  if (!Expect(seeded_code_alloc.mapped_va != 0, "expected seeded code allocation") ||
      !Expect(sgpr_state_alloc.mapped_va != 0, "expected sgpr state allocation") ||
      !Expect(vgpr_state_alloc.mapped_va != 0, "expected vgpr state allocation") ||
      !Expect(simulator.WriteMemory(seeded_code_alloc.mapped_va, AsBytes(seeded_program_words)),
              "expected seeded program write to succeed") ||
      !Expect(simulator.WriteMemory(sgpr_state_alloc.mapped_va, AsBytes(sgpr_state)),
              "expected sgpr state write to succeed") ||
      !Expect(simulator.WriteMemory(vgpr_state_alloc.mapped_va, AsBytes(vgpr_state)),
              "expected vgpr state write to succeed")) {
    return 1;
  }

  exec::SyntheticDispatchPacket seeded_dispatch;
  seeded_dispatch.context.queue_id = queue_id;
  seeded_dispatch.opcode = exec::SyntheticKernelOpcode::kGfx950Program;
  seeded_dispatch.args.code_va = seeded_code_alloc.mapped_va;
  seeded_dispatch.args.code_word_count = seeded_program_words.size();
  seeded_dispatch.args.sgpr_state_va = sgpr_state_alloc.mapped_va;
  seeded_dispatch.args.sgpr_state_count = sgpr_state.size();
  seeded_dispatch.args.vgpr_state_va = vgpr_state_alloc.mapped_va;
  seeded_dispatch.args.vgpr_state_count = 3;
  seeded_dispatch.args.exec_mask = 0b1011ULL;

  const exec::CompletionRecord seeded_dispatch_completion =
      simulator.Submit(queue_id, seeded_dispatch);
  if (!Expect(seeded_dispatch_completion.completed,
              "expected seeded gfx950 dispatch to complete") ||
      !Expect(seeded_dispatch_completion.success,
              "expected seeded gfx950 dispatch to succeed")) {
    return 1;
  }

  std::vector<std::uint32_t> observed_sgpr_state(sgpr_state.size(), 0u);
  std::vector<std::uint32_t> observed_vgpr_state(vgpr_state.size(), 0u);
  if (!Expect(simulator.ReadMemory(sgpr_state_alloc.mapped_va,
                                   AsWritableBytes(observed_sgpr_state)),
              "expected sgpr state readback") ||
      !Expect(simulator.ReadMemory(vgpr_state_alloc.mapped_va,
                                   AsWritableBytes(observed_vgpr_state)),
              "expected vgpr state readback")) {
    return 1;
  }

  if (!Expect(observed_sgpr_state[0] == 10u, "expected sgpr s0 to persist") ||
      !Expect(observed_sgpr_state[1] == 20u, "expected sgpr s1 to persist") ||
      !Expect(observed_sgpr_state[2] == 30u, "expected sgpr s2 result") ||
      !Expect(observed_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 0] == 42u,
              "expected vgpr v2 lane 0 result") ||
      !Expect(observed_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 1] == 6u,
              "expected vgpr v2 lane 1 result") ||
      !Expect(observed_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 2] ==
                  0xdeadbeefu,
              "expected inactive lane to remain untouched") ||
      !Expect(observed_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 3] == 24u,
              "expected vgpr v2 lane 3 result")) {
    return 1;
  }

  const auto ds_write_b32_opcode =
      FindDefaultEncodingOpcode("DS_WRITE_B32", "ENC_DS");
  const auto ds_add_u32_opcode =
      FindDefaultEncodingOpcode("DS_ADD_U32", "ENC_DS");
  const auto ds_read_b32_opcode =
      FindDefaultEncodingOpcode("DS_READ_B32", "ENC_DS");
  if (!Expect(ds_write_b32_opcode.has_value(),
              "expected ds_write_b32 opcode lookup") ||
      !Expect(ds_add_u32_opcode.has_value(),
              "expected ds_add_u32 opcode lookup") ||
      !Expect(ds_read_b32_opcode.has_value(),
              "expected ds_read_b32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> lds_program_words = {
      MakeDs(*ds_write_b32_opcode, 0, 0, 1, 0, 0)[0],
      MakeDs(*ds_write_b32_opcode, 0, 0, 1, 0, 0)[1],
      MakeDs(*ds_add_u32_opcode, 0, 0, 2, 0, 0)[0],
      MakeDs(*ds_add_u32_opcode, 0, 0, 2, 0, 0)[1],
      MakeDs(*ds_read_b32_opcode, 3, 0, 0, 0, 0)[0],
      MakeDs(*ds_read_b32_opcode, 3, 0, 0, 0, 0)[1],
      MakeSopp(1),
  };
  const auto lds_code_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      lds_program_words.size() * sizeof(lds_program_words[0]));
  std::vector<std::uint32_t> lds_vgpr_state(
      4 * isa::WaveExecutionState::kLaneCount, 0u);
  lds_vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 0] = 0u;
  lds_vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 1] = 4u;
  lds_vgpr_state[0 * isa::WaveExecutionState::kLaneCount + 3] = 8u;
  lds_vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 0] = 10u;
  lds_vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 1] = 20u;
  lds_vgpr_state[1 * isa::WaveExecutionState::kLaneCount + 3] = 40u;
  lds_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 0] = 1u;
  lds_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 1] = 2u;
  lds_vgpr_state[2 * isa::WaveExecutionState::kLaneCount + 3] = 4u;
  lds_vgpr_state[3 * isa::WaveExecutionState::kLaneCount + 2] = 0xdeadbeefu;
  const auto lds_vgpr_state_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm, lds_vgpr_state.size() * sizeof(lds_vgpr_state[0]));
  if (!Expect(lds_code_alloc.mapped_va != 0, "expected lds code allocation") ||
      !Expect(lds_vgpr_state_alloc.mapped_va != 0,
              "expected lds vgpr state allocation") ||
      !Expect(simulator.WriteMemory(lds_code_alloc.mapped_va, AsBytes(lds_program_words)),
              "expected lds program write to succeed") ||
      !Expect(simulator.WriteMemory(lds_vgpr_state_alloc.mapped_va, AsBytes(lds_vgpr_state)),
              "expected lds vgpr state write to succeed")) {
    return 1;
  }

  exec::SyntheticDispatchPacket lds_dispatch;
  lds_dispatch.context.queue_id = queue_id;
  lds_dispatch.opcode = exec::SyntheticKernelOpcode::kGfx950Program;
  lds_dispatch.args.code_va = lds_code_alloc.mapped_va;
  lds_dispatch.args.code_word_count = lds_program_words.size();
  lds_dispatch.args.vgpr_state_va = lds_vgpr_state_alloc.mapped_va;
  lds_dispatch.args.vgpr_state_count = 4;
  lds_dispatch.args.exec_mask = 0b1011ULL;

  const exec::CompletionRecord lds_dispatch_completion =
      simulator.Submit(queue_id, lds_dispatch);
  if (!Expect(lds_dispatch_completion.completed,
              "expected lds gfx950 dispatch to complete") ||
      !Expect(lds_dispatch_completion.success,
              "expected lds gfx950 dispatch to succeed")) {
    return 1;
  }

  std::vector<std::uint32_t> observed_lds_vgpr_state(lds_vgpr_state.size(), 0u);
  if (!Expect(simulator.ReadMemory(lds_vgpr_state_alloc.mapped_va,
                                   AsWritableBytes(observed_lds_vgpr_state)),
              "expected lds vgpr state readback")) {
    return 1;
  }

  if (!Expect(
          observed_lds_vgpr_state[3 * isa::WaveExecutionState::kLaneCount + 0] == 11u,
          "expected lds vgpr lane 0 result") ||
      !Expect(
          observed_lds_vgpr_state[3 * isa::WaveExecutionState::kLaneCount + 1] == 22u,
          "expected lds vgpr lane 1 result") ||
      !Expect(
          observed_lds_vgpr_state[3 * isa::WaveExecutionState::kLaneCount + 2] ==
              0xdeadbeefu,
          "expected inactive lds lane to remain untouched") ||
      !Expect(
          observed_lds_vgpr_state[3 * isa::WaveExecutionState::kLaneCount + 3] == 44u,
          "expected lds vgpr lane 3 result")) {
    return 1;
  }

  const auto s_cmp_eq_u32_opcode =
      FindDefaultEncodingOpcode("S_CMP_EQ_U32", "ENC_SOPC");
  const auto s_cbranch_scc1_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_SCC1", "ENC_SOPP");
  const auto s_barrier_opcode =
      FindDefaultEncodingOpcode("S_BARRIER", "ENC_SOPP");
  const auto s_endpgm_opcode =
      FindDefaultEncodingOpcode("S_ENDPGM", "ENC_SOPP");
  if (!Expect(s_cmp_eq_u32_opcode.has_value(),
              "expected s_cmp_eq_u32 opcode lookup") ||
      !Expect(s_cbranch_scc1_opcode.has_value(),
              "expected s_cbranch_scc1 opcode lookup") ||
      !Expect(s_barrier_opcode.has_value(), "expected s_barrier opcode lookup") ||
      !Expect(s_endpgm_opcode.has_value(), "expected s_endpgm opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> workgroup_program_words = {
      MakeSopc(*s_cmp_eq_u32_opcode, 0, 128),
      MakeSopp(*s_cbranch_scc1_opcode, 3),
      MakeSopp(*s_barrier_opcode),
      MakeDs(*ds_read_b32_opcode, 2, 0, 0, 0, 0)[0],
      MakeDs(*ds_read_b32_opcode, 2, 0, 0, 0, 0)[1],
      MakeSopp(*s_endpgm_opcode),
      MakeDs(*ds_write_b32_opcode, 0, 0, 1, 0, 0)[0],
      MakeDs(*ds_write_b32_opcode, 0, 0, 1, 0, 0)[1],
      MakeSopp(*s_barrier_opcode),
      MakeSopp(*s_endpgm_opcode),
  };
  const auto workgroup_code_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      workgroup_program_words.size() * sizeof(workgroup_program_words[0]));
  std::vector<std::uint64_t> workgroup_exec_masks = {0x1ULL, 0x1ULL};
  const auto workgroup_exec_mask_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      workgroup_exec_masks.size() * sizeof(workgroup_exec_masks[0]));
  std::vector<std::uint32_t> workgroup_sgpr_state = {0u, 1u};
  const auto workgroup_sgpr_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      workgroup_sgpr_state.size() * sizeof(workgroup_sgpr_state[0]));
  const std::size_t workgroup_vgpr_stride =
      3 * isa::WaveExecutionState::kLaneCount;
  std::vector<std::uint32_t> workgroup_vgpr_state(2 * workgroup_vgpr_stride, 0u);
  workgroup_vgpr_state[0 * workgroup_vgpr_stride +
                       0 * isa::WaveExecutionState::kLaneCount + 0] = 0u;
  workgroup_vgpr_state[0 * workgroup_vgpr_stride +
                       1 * isa::WaveExecutionState::kLaneCount + 0] = 99u;
  workgroup_vgpr_state[1 * workgroup_vgpr_stride +
                       0 * isa::WaveExecutionState::kLaneCount + 0] = 0u;
  workgroup_vgpr_state[1 * workgroup_vgpr_stride +
                       2 * isa::WaveExecutionState::kLaneCount + 0] =
      0xdeadbeefu;
  const auto workgroup_vgpr_alloc = simulator.AllocateMemory(
      memory::MemoryRegionKind::kHbm,
      workgroup_vgpr_state.size() * sizeof(workgroup_vgpr_state[0]));
  if (!Expect(workgroup_code_alloc.mapped_va != 0,
              "expected workgroup code allocation") ||
      !Expect(workgroup_exec_mask_alloc.mapped_va != 0,
              "expected workgroup exec mask allocation") ||
      !Expect(workgroup_sgpr_alloc.mapped_va != 0,
              "expected workgroup sgpr allocation") ||
      !Expect(workgroup_vgpr_alloc.mapped_va != 0,
              "expected workgroup vgpr allocation") ||
      !Expect(simulator.WriteMemory(workgroup_code_alloc.mapped_va,
                                    AsBytes(workgroup_program_words)),
              "expected workgroup program write to succeed") ||
      !Expect(simulator.WriteMemory(workgroup_exec_mask_alloc.mapped_va,
                                    AsBytes(workgroup_exec_masks)),
              "expected workgroup exec mask write to succeed") ||
      !Expect(simulator.WriteMemory(workgroup_sgpr_alloc.mapped_va,
                                    AsBytes(workgroup_sgpr_state)),
              "expected workgroup sgpr write to succeed") ||
      !Expect(simulator.WriteMemory(workgroup_vgpr_alloc.mapped_va,
                                    AsBytes(workgroup_vgpr_state)),
              "expected workgroup vgpr write to succeed")) {
    return 1;
  }

  exec::SyntheticDispatchPacket workgroup_dispatch;
  workgroup_dispatch.context.queue_id = queue_id;
  workgroup_dispatch.opcode = exec::SyntheticKernelOpcode::kGfx950Program;
  workgroup_dispatch.args.code_va = workgroup_code_alloc.mapped_va;
  workgroup_dispatch.args.code_word_count = workgroup_program_words.size();
  workgroup_dispatch.args.wave_count = 2;
  workgroup_dispatch.args.exec_mask_va = workgroup_exec_mask_alloc.mapped_va;
  workgroup_dispatch.args.sgpr_state_va = workgroup_sgpr_alloc.mapped_va;
  workgroup_dispatch.args.sgpr_state_count = 1;
  workgroup_dispatch.args.vgpr_state_va = workgroup_vgpr_alloc.mapped_va;
  workgroup_dispatch.args.vgpr_state_count = 3;

  const exec::CompletionRecord workgroup_dispatch_completion =
      simulator.Submit(queue_id, workgroup_dispatch);
  if (!Expect(workgroup_dispatch_completion.completed,
              "expected workgroup gfx950 dispatch to complete") ||
      !Expect(workgroup_dispatch_completion.success,
              "expected workgroup gfx950 dispatch to succeed")) {
    return 1;
  }

  std::vector<std::uint32_t> observed_workgroup_vgpr_state(
      workgroup_vgpr_state.size(), 0u);
  if (!Expect(simulator.ReadMemory(workgroup_vgpr_alloc.mapped_va,
                                   AsWritableBytes(observed_workgroup_vgpr_state)),
              "expected workgroup vgpr state readback")) {
    return 1;
  }

  if (!Expect(observed_workgroup_vgpr_state
                      [1 * workgroup_vgpr_stride +
                       2 * isa::WaveExecutionState::kLaneCount + 0] == 99u,
              "expected shared lds value to reach reader wave") ||
      !Expect(observed_workgroup_vgpr_state
                      [0 * workgroup_vgpr_stride +
                       2 * isa::WaveExecutionState::kLaneCount + 0] == 0u,
              "expected writer wave v2 to remain unchanged")) {
    return 1;
  }

  const auto final_queue_state = simulator.GetQueue(queue_id);
  if (!Expect(final_queue_state.has_value(), "expected queue to persist") ||
      !Expect(final_queue_state->doorbell.write_ptr == 8,
              "expected queue write pointer to advance") ||
      !Expect(final_queue_state->doorbell.read_ptr == 8,
              "expected queue read pointer to retire")) {
    return 1;
  }

  return 0;
}
