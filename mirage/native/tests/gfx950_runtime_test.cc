#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/decoded_instruction.h"
#include "lib/sim/isa/execution_memory.h"
#include "lib/sim/isa/gfx950_interpreter.h"
#include "lib/sim/isa/instruction_catalog.h"
#include "lib/sim/isa/wave_execution_state.h"
#include "lib/sim/single_gpu_simulator.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

template <typename T>
std::span<const std::byte> AsBytes(const std::vector<T>& values) {
  return std::as_bytes(std::span<const T>(values.data(), values.size()));
}

template <typename T>
std::span<std::byte> AsWritableBytes(std::vector<T>& values) {
  return std::as_writable_bytes(std::span<T>(values.data(), values.size()));
}

void SetLaneAddress(mirage::sim::isa::WaveExecutionState* state,
                    std::uint16_t reg,
                    std::size_t lane_index,
                    std::uint64_t address) {
  state->vgprs[reg][lane_index] = static_cast<std::uint32_t>(address);
  state->vgprs[reg + 1][lane_index] =
      static_cast<std::uint32_t>(address >> 32);
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

}  // namespace

int main(int argc, char** argv) {
  using namespace mirage::sim::isa;

  constexpr std::size_t kWarmupIterations = 200;
  constexpr std::size_t kTimedIterations = 2000;
  constexpr std::size_t kInstructionsPerDispatch = 4;
  constexpr std::uint16_t kScalarBaseReg = 0;
  constexpr std::uint16_t kLoadAddrReg = 0;
  constexpr std::uint16_t kAtomicAddrReg = 2;
  constexpr std::uint16_t kStoreAddrReg = 4;
  constexpr std::uint16_t kAtomicDataReg = 40;

  Gfx950Interpreter interpreter;
  LinearExecutionMemory memory(0x10000, 0);
  WaveExecutionState state;
  state.exec_mask = ~0ULL;
  state.sgprs[kScalarBaseReg] = 0;
  state.sgprs[kScalarBaseReg + 1] = 0;

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    const std::uint64_t load_address =
        0x1000 + static_cast<std::uint64_t>(lane_index) * 0x20u;
    const std::uint64_t store_address =
        0x4000 + static_cast<std::uint64_t>(lane_index) * 0x20u;
    const std::uint64_t atomic_address =
        0x8000 + static_cast<std::uint64_t>(lane_index) * 0x4u;

    SetLaneAddress(&state, kLoadAddrReg, lane_index, load_address);
    SetLaneAddress(&state, kAtomicAddrReg, lane_index, atomic_address);
    SetLaneAddress(&state, kStoreAddrReg, lane_index, store_address);
    state.vgprs[kAtomicDataReg][lane_index] =
        static_cast<std::uint32_t>(lane_index + 1u);

    for (std::uint32_t dword_index = 0; dword_index < 4; ++dword_index) {
      const std::uint32_t load_value =
          static_cast<std::uint32_t>(lane_index * 0x100u + dword_index);
      if (!memory.WriteU32(load_address + static_cast<std::uint64_t>(dword_index) * 4u,
                           load_value) ||
          !memory.WriteU32(store_address + static_cast<std::uint64_t>(dword_index) * 4u,
                           0u)) {
        std::cerr << "failed to seed load/store memory\n";
        return 1;
      }
    }
    if (!memory.WriteU32(atomic_address, static_cast<std::uint32_t>(lane_index))) {
      std::cerr << "failed to seed atomic memory\n";
      return 1;
    }
  }

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::FourOperand("GLOBAL_LOAD_DWORDX4",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Vgpr(kLoadAddrReg),
                                      InstructionOperand::Sgpr(kScalarBaseReg),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_ATOMIC_ADD",
                                      InstructionOperand::Vgpr(kAtomicAddrReg),
                                      InstructionOperand::Vgpr(kAtomicDataReg),
                                      InstructionOperand::Sgpr(kScalarBaseReg),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_DWORDX4",
                                      InstructionOperand::Vgpr(kStoreAddrReg),
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Sgpr(kScalarBaseReg),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::string error_message;
  std::vector<CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(program, &compiled_program, &error_message),
              error_message.c_str())) {
    return 1;
  }
  for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
    state.pc = 0;
    if (!Expect(interpreter.ExecuteProgram(compiled_program, &state, &memory,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < kTimedIterations; ++iteration) {
    state.pc = 0;
    if (!Expect(interpreter.ExecuteProgram(compiled_program, &state, &memory,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }
  }
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const double elapsed_ms =
      static_cast<double>(elapsed_ns) / 1'000'000.0;
  const double ns_per_dispatch =
      static_cast<double>(elapsed_ns) / static_cast<double>(kTimedIterations);
  const double ns_per_instruction =
      static_cast<double>(elapsed_ns) /
      static_cast<double>(kTimedIterations * kInstructionsPerDispatch);

  const std::size_t total_iterations = kWarmupIterations + kTimedIterations;
  std::uint32_t atomic_lane0 = 0;
  std::uint32_t atomic_lane63 = 0;
  std::uint32_t stored_lane0_dword3 = 0;
  if (!Expect(memory.ReadU32(0x8000, &atomic_lane0),
              "expected atomic lane 0 result") ||
      !Expect(memory.ReadU32(0x8000 + 63u * 4u, &atomic_lane63),
              "expected atomic lane 63 result") ||
      !Expect(memory.ReadU32(0x4000 + 12u, &stored_lane0_dword3),
              "expected stored lane 0 dword 3 result")) {
    return 1;
  }

  if (!Expect(atomic_lane0 == static_cast<std::uint32_t>(total_iterations),
              "unexpected lane 0 atomic accumulation") ||
      !Expect(
          atomic_lane63 ==
              static_cast<std::uint32_t>(63u + total_iterations * 64u),
          "unexpected lane 63 atomic accumulation") ||
      !Expect(stored_lane0_dword3 == 3u,
              "unexpected storeback dword value")) {
    return 1;
  }

  std::ostringstream metrics_stream;
  metrics_stream << std::fixed << std::setprecision(2)
                 << "MIRAGE_RUNTIME benchmark=gfx950_global_mix timed_iters="
                 << kTimedIterations << " elapsed_ms=" << elapsed_ms
                 << " ns_per_dispatch=" << ns_per_dispatch
                 << " ns_per_instruction=" << ns_per_instruction;
  const std::string metrics_line = metrics_stream.str();
  std::cout << metrics_line << '\n';

  const auto s_cmp_eq_u32_opcode =
      FindDefaultEncodingOpcode("S_CMP_EQ_U32", "ENC_SOPC");
  const auto s_cbranch_scc1_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_SCC1", "ENC_SOPP");
  const auto s_barrier_opcode =
      FindDefaultEncodingOpcode("S_BARRIER", "ENC_SOPP");
  const auto s_endpgm_opcode =
      FindDefaultEncodingOpcode("S_ENDPGM", "ENC_SOPP");
  const auto ds_write_b32_opcode =
      FindDefaultEncodingOpcode("DS_WRITE_B32", "ENC_DS");
  const auto ds_read_b32_opcode =
      FindDefaultEncodingOpcode("DS_READ_B32", "ENC_DS");
  if (!Expect(s_cmp_eq_u32_opcode.has_value(),
              "expected s_cmp_eq_u32 opcode lookup") ||
      !Expect(s_cbranch_scc1_opcode.has_value(),
              "expected s_cbranch_scc1 opcode lookup") ||
      !Expect(s_barrier_opcode.has_value(), "expected s_barrier opcode lookup") ||
      !Expect(s_endpgm_opcode.has_value(), "expected s_endpgm opcode lookup") ||
      !Expect(ds_write_b32_opcode.has_value(),
              "expected ds_write_b32 opcode lookup") ||
      !Expect(ds_read_b32_opcode.has_value(),
              "expected ds_read_b32 opcode lookup")) {
    return 1;
  }

  mirage::sim::gpu::GpuProperties properties;
  properties.arch_name = "CDNA4";
  properties.gfx_target = "gfx950";
  properties.compute_units = 304;
  properties.hbm_bytes = 192ULL * 1024ULL * 1024ULL;

  mirage::sim::SingleGpuSimulator simulator(properties);
  const mirage::sim::queue::QueueId queue_id = simulator.CreateComputeQueue();
  if (!Expect(queue_id != 0, "expected benchmark queue creation")) {
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
      mirage::sim::memory::MemoryRegionKind::kHbm,
      workgroup_program_words.size() * sizeof(workgroup_program_words[0]));
  std::vector<std::uint64_t> workgroup_exec_masks = {0x1ULL, 0x1ULL};
  const auto workgroup_exec_mask_alloc = simulator.AllocateMemory(
      mirage::sim::memory::MemoryRegionKind::kHbm,
      workgroup_exec_masks.size() * sizeof(workgroup_exec_masks[0]));
  std::vector<std::uint32_t> workgroup_sgpr_state = {0u, 1u};
  const auto workgroup_sgpr_alloc = simulator.AllocateMemory(
      mirage::sim::memory::MemoryRegionKind::kHbm,
      workgroup_sgpr_state.size() * sizeof(workgroup_sgpr_state[0]));
  const std::size_t workgroup_vgpr_stride =
      3 * WaveExecutionState::kLaneCount;
  std::vector<std::uint32_t> workgroup_vgpr_state(2 * workgroup_vgpr_stride, 0u);
  workgroup_vgpr_state[0 * workgroup_vgpr_stride +
                       0 * WaveExecutionState::kLaneCount + 0] = 0u;
  workgroup_vgpr_state[0 * workgroup_vgpr_stride +
                       1 * WaveExecutionState::kLaneCount + 0] = 99u;
  workgroup_vgpr_state[1 * workgroup_vgpr_stride +
                       0 * WaveExecutionState::kLaneCount + 0] = 0u;
  const auto workgroup_vgpr_alloc = simulator.AllocateMemory(
      mirage::sim::memory::MemoryRegionKind::kHbm,
      workgroup_vgpr_state.size() * sizeof(workgroup_vgpr_state[0]));
  if (!Expect(workgroup_code_alloc.mapped_va != 0,
              "expected workgroup code allocation") ||
      !Expect(workgroup_exec_mask_alloc.mapped_va != 0,
              "expected workgroup exec-mask allocation") ||
      !Expect(workgroup_sgpr_alloc.mapped_va != 0,
              "expected workgroup sgpr allocation") ||
      !Expect(workgroup_vgpr_alloc.mapped_va != 0,
              "expected workgroup vgpr allocation") ||
      !Expect(simulator.WriteMemory(workgroup_code_alloc.mapped_va,
                                    AsBytes(workgroup_program_words)),
              "expected workgroup code write") ||
      !Expect(simulator.WriteMemory(workgroup_exec_mask_alloc.mapped_va,
                                    AsBytes(workgroup_exec_masks)),
              "expected workgroup exec-mask write") ||
      !Expect(simulator.WriteMemory(workgroup_sgpr_alloc.mapped_va,
                                    AsBytes(workgroup_sgpr_state)),
              "expected workgroup sgpr write") ||
      !Expect(simulator.WriteMemory(workgroup_vgpr_alloc.mapped_va,
                                    AsBytes(workgroup_vgpr_state)),
              "expected workgroup vgpr write")) {
    return 1;
  }

  mirage::sim::exec::SyntheticDispatchPacket workgroup_dispatch;
  workgroup_dispatch.context.queue_id = queue_id;
  workgroup_dispatch.opcode =
      mirage::sim::exec::SyntheticKernelOpcode::kGfx950Program;
  workgroup_dispatch.args.code_va = workgroup_code_alloc.mapped_va;
  workgroup_dispatch.args.code_word_count = workgroup_program_words.size();
  workgroup_dispatch.args.wave_count = 2;
  workgroup_dispatch.args.exec_mask_va = workgroup_exec_mask_alloc.mapped_va;
  workgroup_dispatch.args.sgpr_state_va = workgroup_sgpr_alloc.mapped_va;
  workgroup_dispatch.args.sgpr_state_count = 1;
  workgroup_dispatch.args.vgpr_state_va = workgroup_vgpr_alloc.mapped_va;
  workgroup_dispatch.args.vgpr_state_count = 3;

  for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
    const mirage::sim::exec::CompletionRecord completion =
        simulator.Submit(queue_id, workgroup_dispatch);
    if (!Expect(completion.completed && completion.success,
                "expected workgroup warmup dispatch to succeed")) {
      return 1;
    }
  }

  const auto workgroup_start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < kTimedIterations; ++iteration) {
    const mirage::sim::exec::CompletionRecord completion =
        simulator.Submit(queue_id, workgroup_dispatch);
    if (!Expect(completion.completed && completion.success,
                "expected workgroup timed dispatch to succeed")) {
      return 1;
    }
  }
  const auto workgroup_end = std::chrono::steady_clock::now();
  const auto workgroup_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      workgroup_end - workgroup_start).count();
  const double workgroup_elapsed_ms =
      static_cast<double>(workgroup_elapsed_ns) / 1'000'000.0;
  const double workgroup_ns_per_dispatch =
      static_cast<double>(workgroup_elapsed_ns) /
      static_cast<double>(kTimedIterations);
  constexpr std::size_t kWorkgroupInstructionsPerDispatch = 8;
  const double workgroup_ns_per_instruction =
      static_cast<double>(workgroup_elapsed_ns) /
      static_cast<double>(kTimedIterations * kWorkgroupInstructionsPerDispatch);

  std::vector<std::uint32_t> observed_workgroup_vgpr_state(
      workgroup_vgpr_state.size(), 0u);
  const auto workgroup_cache_stats = simulator.GetDecodeCacheStats();
  if (!Expect(simulator.ReadMemory(workgroup_vgpr_alloc.mapped_va,
                                   AsWritableBytes(observed_workgroup_vgpr_state)),
              "expected workgroup benchmark vgpr readback") ||
      !Expect(observed_workgroup_vgpr_state
                      [1 * workgroup_vgpr_stride +
                       2 * WaveExecutionState::kLaneCount + 0] == 99u,
              "unexpected workgroup benchmark readback")) {
    return 1;
  }

  std::ostringstream workgroup_metrics_stream;
  workgroup_metrics_stream << std::fixed << std::setprecision(2)
                           << "MIRAGE_RUNTIME benchmark=gfx950_workgroup_barrier timed_iters="
                           << kTimedIterations << " elapsed_ms="
                           << workgroup_elapsed_ms
                           << " ns_per_dispatch=" << workgroup_ns_per_dispatch
                           << " ns_per_instruction="
                           << workgroup_ns_per_instruction
                           << " decode_cache_hits="
                           << workgroup_cache_stats.hits
                           << " decode_cache_misses="
                           << workgroup_cache_stats.misses;
  const std::string workgroup_metrics_line = workgroup_metrics_stream.str();
  std::cout << workgroup_metrics_line << '\n';

  std::filesystem::path metrics_path = "mirage_gfx950_runtime_metrics.txt";
  if (argc > 0 && argv != nullptr) {
    metrics_path =
        std::filesystem::path(argv[0]).parent_path() /
        "mirage_gfx950_runtime_metrics.txt";
  }
  std::ofstream metrics_file(metrics_path);
  if (metrics_file.is_open()) {
    metrics_file << metrics_line << '\n';
    metrics_file << workgroup_metrics_line << '\n';
  }

  return 0;
}
