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

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/execution_memory.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx950/interpreter.h"
#include "lib/sim/isa/instruction_catalog.h"
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

  const std::vector<DecodedInstruction> atomic_parity_program = {
      DecodedInstruction::FourOperand("GLOBAL_ATOMIC_SWAP",
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("GLOBAL_ATOMIC_CMPSWAP",
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Vgpr(6),
                                      InstructionOperand::Vgpr(8),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory atomic_parity_memory(0x2000, 0);
  WaveExecutionState atomic_parity_state;
  atomic_parity_state.exec_mask = 1u;
  atomic_parity_state.sgprs[0] = 0u;
  atomic_parity_state.sgprs[1] = 0u;
  SetLaneAddress(&atomic_parity_state, 0, 0, 0x1000u);
  SetLaneAddress(&atomic_parity_state, 2, 0, 0x0u);
  SetLaneAddress(&atomic_parity_state, 4, 0, 0x0u);
  SetLaneAddress(&atomic_parity_state, 6, 0, 0x1400u);
  SetLaneAddress(&atomic_parity_state, 8, 0, 0x0u);
  atomic_parity_state.vgprs[2][0] = 0x11112222u;
  atomic_parity_state.vgprs[8][0] = 0xbbbbbbbbu;
  atomic_parity_state.vgprs[9][0] = 0x33334444u;
  if (!Expect(atomic_parity_memory.WriteU32(0x1000u, 0xaaaa5555u),
              "expected global atomic swap seed write") ||
      !Expect(atomic_parity_memory.WriteU32(0x1400u, 0xbbbbbbbbu),
              "expected global atomic cmpswap seed write")) {
    return 1;
  }

  std::string atomic_parity_error_message;
  std::vector<CompiledInstruction> compiled_atomic_parity_program;
  std::uint32_t atomic_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(atomic_parity_program,
                                         &compiled_atomic_parity_program,
                                         &atomic_parity_error_message),
              atomic_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_atomic_parity_program,
                                         &atomic_parity_state,
                                         &atomic_parity_memory,
                                         &atomic_parity_error_message),
              atomic_parity_error_message.c_str()) ||
      !Expect(atomic_parity_state.halted,
              "expected atomic parity program to halt") ||
      !Expect(atomic_parity_state.vgprs[4][0] == 0xbbbbbbbbu,
              "expected global atomic cmpswap return value") ||
      !Expect(atomic_parity_memory.ReadU32(0x1000u, &atomic_parity_readback),
              "expected global atomic swap readback") ||
      !Expect(atomic_parity_readback == 0x11112222u,
              "expected global atomic swap memory update") ||
      !Expect(atomic_parity_memory.ReadU32(0x1400u, &atomic_parity_readback),
              "expected global atomic cmpswap readback") ||
      !Expect(atomic_parity_readback == 0x33334444u,
              "expected global atomic cmpswap memory update")) {
    return 1;
  }

  const std::vector<DecodedInstruction> atomic_x2_parity_program = {
      DecodedInstruction::FourOperand("GLOBAL_ATOMIC_ADD_X2",
                                      InstructionOperand::Vgpr(10),
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory atomic_x2_parity_memory(0x2000, 0);
  WaveExecutionState atomic_x2_parity_state;
  atomic_x2_parity_state.exec_mask = 1u;
  atomic_x2_parity_state.sgprs[0] = 0u;
  atomic_x2_parity_state.sgprs[1] = 0u;
  SetLaneAddress(&atomic_x2_parity_state, 10, 0, 0x1800u);
  atomic_x2_parity_state.vgprs[12][0] = 3u;
  atomic_x2_parity_state.vgprs[13][0] = 0u;
  if (!Expect(atomic_x2_parity_memory.WriteU32(0x1800u, 1000u),
              "expected global atomic add_x2 seed write") ||
      !Expect(atomic_x2_parity_memory.WriteU32(0x1804u, 0u),
              "expected global atomic add_x2 seed write")) {
    return 1;
  }

  std::string atomic_x2_parity_error_message;
  std::vector<CompiledInstruction> compiled_atomic_x2_parity_program;
  std::uint32_t atomic_x2_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(atomic_x2_parity_program,
                                         &compiled_atomic_x2_parity_program,
                                         &atomic_x2_parity_error_message),
              atomic_x2_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_atomic_x2_parity_program,
                                         &atomic_x2_parity_state,
                                         &atomic_x2_parity_memory,
                                         &atomic_x2_parity_error_message),
              atomic_x2_parity_error_message.c_str()) ||
      !Expect(atomic_x2_parity_state.halted,
              "expected atomic add_x2 parity program to halt") ||
      !Expect(atomic_x2_parity_state.vgprs[12][0] == 3u &&
                  atomic_x2_parity_state.vgprs[13][0] == 0u,
              "expected global atomic add_x2 data preservation") ||
      !Expect(atomic_x2_parity_memory.ReadU32(0x1800u,
                                              &atomic_x2_parity_readback),
              "expected global atomic add_x2 low readback") ||
      !Expect(atomic_x2_parity_readback == 1003u,
              "expected global atomic add_x2 low memory update") ||
      !Expect(atomic_x2_parity_memory.ReadU32(0x1804u,
                                              &atomic_x2_parity_readback),
              "expected global atomic add_x2 high readback") ||
      !Expect(atomic_x2_parity_readback == 0u,
              "expected global atomic add_x2 high memory update")) {
    return 1;
  }

  const std::vector<DecodedInstruction> flat_atomic_parity_program = {
      DecodedInstruction::ThreeOperand("FLAT_ATOMIC_ADD",
                                       InstructionOperand::Vgpr(14),
                                       InstructionOperand::Vgpr(20),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_ATOMIC_SWAP",
                                       InstructionOperand::Vgpr(16),
                                       InstructionOperand::Vgpr(21),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_ATOMIC_CMPSWAP",
                                       InstructionOperand::Vgpr(18),
                                       InstructionOperand::Vgpr(22),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_ATOMIC_ADD_X2",
                                       InstructionOperand::Vgpr(24),
                                       InstructionOperand::Vgpr(26),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory flat_atomic_parity_memory(0x2000, 0);
  WaveExecutionState flat_atomic_parity_state;
  flat_atomic_parity_state.exec_mask = 1u;
  SetLaneAddress(&flat_atomic_parity_state, 14, 0, 0x520u);
  SetLaneAddress(&flat_atomic_parity_state, 16, 0, 0x530u);
  SetLaneAddress(&flat_atomic_parity_state, 18, 0, 0x540u);
  SetLaneAddress(&flat_atomic_parity_state, 24, 0, 0x560u);
  flat_atomic_parity_state.vgprs[20][0] = 1u;
  flat_atomic_parity_state.vgprs[21][0] = 500u;
  flat_atomic_parity_state.vgprs[22][0] = 100u;
  flat_atomic_parity_state.vgprs[23][0] = 700u;
  flat_atomic_parity_state.vgprs[26][0] = 3u;
  flat_atomic_parity_state.vgprs[27][0] = 0u;
  flat_atomic_parity_state.vgprs[40][0] = 0xaaaab001u;
  flat_atomic_parity_state.vgprs[41][0] = 0xbbbbc001u;
  if (!Expect(flat_atomic_parity_memory.WriteU32(0x520u, 10u),
              "expected flat atomic add seed write") ||
      !Expect(flat_atomic_parity_memory.WriteU32(0x530u, 50u),
              "expected flat atomic swap seed write") ||
      !Expect(flat_atomic_parity_memory.WriteU32(0x540u, 100u),
              "expected flat atomic cmpswap seed write") ||
      !Expect(flat_atomic_parity_memory.WriteU32(0x560u, 1000u) &&
                  flat_atomic_parity_memory.WriteU32(0x564u, 0u),
              "expected flat atomic add_x2 seed write")) {
    return 1;
  }

  std::string flat_atomic_parity_error_message;
  std::vector<CompiledInstruction> compiled_flat_atomic_parity_program;
  if (!Expect(interpreter.CompileProgram(flat_atomic_parity_program,
                                         &compiled_flat_atomic_parity_program,
                                         &flat_atomic_parity_error_message),
              flat_atomic_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_flat_atomic_parity_program,
                                         &flat_atomic_parity_state,
                                         &flat_atomic_parity_memory,
                                         &flat_atomic_parity_error_message),
              flat_atomic_parity_error_message.c_str()) ||
      !Expect(flat_atomic_parity_state.halted,
              "expected flat atomic parity program to halt")) {
    return 1;
  }

  std::uint32_t flat_atomic_readback = 0;
  std::uint32_t flat_atomic_high_readback = 0;
  if (!Expect(flat_atomic_parity_memory.ReadU32(0x520u, &flat_atomic_readback),
              "expected flat atomic add readback") ||
      !Expect(flat_atomic_readback == 11u,
              "expected flat atomic add memory update") ||
      !Expect(flat_atomic_parity_memory.ReadU32(0x530u, &flat_atomic_readback),
              "expected flat atomic swap readback") ||
      !Expect(flat_atomic_readback == 500u,
              "expected flat atomic swap memory update") ||
      !Expect(flat_atomic_parity_memory.ReadU32(0x540u, &flat_atomic_readback),
              "expected flat atomic cmpswap readback") ||
      !Expect(flat_atomic_readback == 700u,
              "expected flat atomic cmpswap memory update") ||
      !Expect(flat_atomic_parity_memory.ReadU32(0x560u, &flat_atomic_readback),
              "expected flat atomic add_x2 low readback") ||
      !Expect(flat_atomic_readback == 1003u,
              "expected flat atomic add_x2 low memory update") ||
      !Expect(flat_atomic_parity_memory.ReadU32(0x564u,
                                               &flat_atomic_high_readback),
              "expected flat atomic add_x2 high readback") ||
      !Expect(flat_atomic_high_readback == 0u,
              "expected flat atomic add_x2 high memory update")) {
    return 1;
  }
  if (!Expect(flat_atomic_parity_state.vgprs[20][0] == 1u &&
                  flat_atomic_parity_state.vgprs[21][0] == 500u &&
                  flat_atomic_parity_state.vgprs[22][0] == 100u &&
                  flat_atomic_parity_state.vgprs[23][0] == 700u &&
                  flat_atomic_parity_state.vgprs[26][0] == 3u &&
                  flat_atomic_parity_state.vgprs[27][0] == 0u,
              "expected flat atomic parity source preservation") ||
      !Expect(flat_atomic_parity_state.vgprs[40][0] == 0xaaaab001u &&
                  flat_atomic_parity_state.vgprs[41][0] == 0xbbbbc001u,
              "expected flat atomic parity unrelated preservation")) {
    return 1;
  }

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
