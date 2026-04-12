#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
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

bool WriteU8(mirage::sim::isa::LinearExecutionMemory* memory,
             std::uint64_t address,
             std::uint8_t value) {
  std::array<std::byte, 1> bytes{std::byte{value}};
  return memory->Store(address,
                       std::span<const std::byte>(bytes.data(), bytes.size()));
}

bool WriteU16(mirage::sim::isa::LinearExecutionMemory* memory,
              std::uint64_t address,
              std::uint16_t value) {
  std::array<std::byte, 2> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(value));
  return memory->Store(address,
                       std::span<const std::byte>(bytes.data(), bytes.size()));
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

  const std::vector<DecodedInstruction> scalar_memory_parity_program = {
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX4",
                                       InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x20)),
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX8",
                                       InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x40)),
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX16",
                                       InstructionOperand::Sgpr(32),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x80)),
      DecodedInstruction::ThreeOperand("S_STORE_DWORDX2",
                                       InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(2)),
      DecodedInstruction::ThreeOperand("S_STORE_DWORDX4",
                                       InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(3)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory scalar_memory_parity_memory(0x600, 0);
  WaveExecutionState scalar_memory_parity_state{};
  scalar_memory_parity_state.exec_mask = 0b1011ULL;
  scalar_memory_parity_state.sgprs[0] = 0x100u;
  scalar_memory_parity_state.sgprs[1] = 0u;
  scalar_memory_parity_state.sgprs[2] = 0x140u;
  scalar_memory_parity_state.sgprs[3] = 0x160u;
  scalar_memory_parity_state.sgprs[96] = 0x13572468u;
  scalar_memory_parity_state.sgprs[97] = 0x24681357u;
  scalar_memory_parity_state.vgprs[18][0] = 0x10203040u;
  scalar_memory_parity_state.vgprs[18][1] = 0x50607080u;
  scalar_memory_parity_state.vgprs[18][2] = 0x90a0b0c0u;
  scalar_memory_parity_state.vgprs[18][3] = 0xd0e0f000u;
  for (std::uint32_t index = 0; index < 4u; ++index) {
    if (!Expect(scalar_memory_parity_memory.WriteU32(
                    0x120u + static_cast<std::uint64_t>(index) * 4u,
                    0x100u + index),
                "expected scalar memory x4 seed write")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 8u; ++index) {
    if (!Expect(scalar_memory_parity_memory.WriteU32(
                    0x140u + static_cast<std::uint64_t>(index) * 4u,
                    0x200u + index),
                "expected scalar memory x8 seed write")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 16u; ++index) {
    if (!Expect(scalar_memory_parity_memory.WriteU32(
                    0x180u + static_cast<std::uint64_t>(index) * 4u,
                    0x300u + index),
                "expected scalar memory x16 seed write")) {
      return 1;
    }
  }
  if (!Expect(scalar_memory_parity_memory.WriteU32(0x2f0u, 0x89abcdefu),
              "expected scalar memory unrelated seed write")) {
    return 1;
  }

  std::string scalar_memory_parity_error_message;
  std::vector<CompiledInstruction> compiled_scalar_memory_parity_program;
  if (!Expect(interpreter.CompileProgram(scalar_memory_parity_program,
                                         &compiled_scalar_memory_parity_program,
                                         &scalar_memory_parity_error_message),
              scalar_memory_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_scalar_memory_parity_program,
                                         &scalar_memory_parity_state,
                                         &scalar_memory_parity_memory,
                                         &scalar_memory_parity_error_message),
              scalar_memory_parity_error_message.c_str()) ||
      !Expect(scalar_memory_parity_state.halted,
              "expected scalar memory parity program to halt") ||
      !Expect(scalar_memory_parity_state.exec_mask == 0b1011ULL,
              "expected scalar memory parity to preserve exec") ||
      !Expect(scalar_memory_parity_state.sgprs[0] == 0x100u &&
                  scalar_memory_parity_state.sgprs[1] == 0u &&
                  scalar_memory_parity_state.sgprs[2] == 0x140u &&
                  scalar_memory_parity_state.sgprs[3] == 0x160u,
              "expected scalar memory parity to preserve controls") ||
      !Expect(scalar_memory_parity_state.sgprs[96] == 0x13572468u &&
                  scalar_memory_parity_state.sgprs[97] == 0x24681357u,
              "expected scalar memory parity to preserve unrelated sgprs") ||
      !Expect(scalar_memory_parity_state.vgprs[18][0] == 0x10203040u &&
                  scalar_memory_parity_state.vgprs[18][1] == 0x50607080u &&
                  scalar_memory_parity_state.vgprs[18][2] == 0x90a0b0c0u &&
                  scalar_memory_parity_state.vgprs[18][3] == 0xd0e0f000u,
              "expected scalar memory parity to preserve vgprs")) {
    return 1;
  }
  for (std::uint32_t index = 0; index < 4u; ++index) {
    if (!Expect(scalar_memory_parity_state.sgprs[8 + index] == 0x100u + index,
                "expected s_load_dwordx4 result")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 8u; ++index) {
    if (!Expect(scalar_memory_parity_state.sgprs[16 + index] == 0x200u + index,
                "expected s_load_dwordx8 result")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 16u; ++index) {
    if (!Expect(scalar_memory_parity_state.sgprs[32 + index] == 0x300u + index,
                "expected s_load_dwordx16 result")) {
      return 1;
    }
  }
  std::uint32_t scalar_memory_parity_readback = 0;
  for (std::uint32_t index = 0; index < 2u; ++index) {
    if (!Expect(scalar_memory_parity_memory.ReadU32(
                    0x240u + static_cast<std::uint64_t>(index) * 4u,
                    &scalar_memory_parity_readback),
                "expected scalar memory x2 store readback") ||
        !Expect(scalar_memory_parity_readback == 0x200u + index,
                "expected scalar memory x2 store result")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 4u; ++index) {
    if (!Expect(scalar_memory_parity_memory.ReadU32(
                    0x260u + static_cast<std::uint64_t>(index) * 4u,
                    &scalar_memory_parity_readback),
                "expected scalar memory x4 store readback") ||
        !Expect(scalar_memory_parity_readback == 0x100u + index,
                "expected scalar memory x4 store result")) {
      return 1;
    }
  }
  if (!Expect(scalar_memory_parity_memory.ReadU32(0x120u,
                                                  &scalar_memory_parity_readback) &&
                  scalar_memory_parity_readback == 0x100u,
              "expected scalar memory first source preserved") ||
      !Expect(scalar_memory_parity_memory.ReadU32(0x15cu,
                                                  &scalar_memory_parity_readback) &&
                  scalar_memory_parity_readback == 0x200u + 7u,
              "expected scalar memory x8 tail source preserved") ||
      !Expect(scalar_memory_parity_memory.ReadU32(0x1bcu,
                                                  &scalar_memory_parity_readback) &&
                  scalar_memory_parity_readback == 0x300u + 15u,
              "expected scalar memory x16 tail source preserved") ||
      !Expect(scalar_memory_parity_memory.ReadU32(0x2f0u,
                                                  &scalar_memory_parity_readback) &&
                  scalar_memory_parity_readback == 0x89abcdefu,
              "expected scalar memory unrelated memory preserved")) {
    return 1;
  }

  const std::vector<DecodedInstruction> scalar_buffer_parity_program = {
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORD",
                                       InstructionOperand::Sgpr(4),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x0)),
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORDX2",
                                       InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x10)),
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORDX4",
                                       InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x20)),
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORDX8",
                                       InstructionOperand::Sgpr(24),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x40)),
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORDX16",
                                       InstructionOperand::Sgpr(40),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x80)),
      DecodedInstruction::ThreeOperand("S_BUFFER_STORE_DWORD",
                                       InstructionOperand::Sgpr(4),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x200)),
      DecodedInstruction::ThreeOperand("S_BUFFER_STORE_DWORDX2",
                                       InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x210)),
      DecodedInstruction::ThreeOperand("S_BUFFER_STORE_DWORDX4",
                                       InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(72)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory scalar_buffer_parity_memory(0x700, 0);
  WaveExecutionState scalar_buffer_parity_state{};
  scalar_buffer_parity_state.exec_mask = 0b0101ULL;
  scalar_buffer_parity_state.sgprs[0] = 0x100u;
  scalar_buffer_parity_state.sgprs[1] = 0u;
  scalar_buffer_parity_state.sgprs[2] = 0x400u;
  scalar_buffer_parity_state.sgprs[3] = 0u;
  scalar_buffer_parity_state.sgprs[72] = 0x220u;
  scalar_buffer_parity_state.sgprs[96] = 0x11223344u;
  scalar_buffer_parity_state.sgprs[97] = 0x55667788u;
  scalar_buffer_parity_state.vgprs[18][0] = 0x01010101u;
  scalar_buffer_parity_state.vgprs[18][1] = 0x02020202u;
  scalar_buffer_parity_state.vgprs[18][2] = 0x03030303u;
  scalar_buffer_parity_state.vgprs[18][3] = 0x04040404u;
  if (!Expect(scalar_buffer_parity_memory.WriteU32(0x100u, 0x11110000u),
              "expected scalar buffer seed write")) {
    return 1;
  }
  for (std::uint32_t index = 0; index < 2u; ++index) {
    if (!Expect(scalar_buffer_parity_memory.WriteU32(
                    0x110u + static_cast<std::uint64_t>(index) * 4u,
                    0x22220000u + index),
                "expected scalar buffer x2 seed write")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 4u; ++index) {
    if (!Expect(scalar_buffer_parity_memory.WriteU32(
                    0x120u + static_cast<std::uint64_t>(index) * 4u,
                    0x33330000u + index),
                "expected scalar buffer x4 seed write")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 8u; ++index) {
    if (!Expect(scalar_buffer_parity_memory.WriteU32(
                    0x140u + static_cast<std::uint64_t>(index) * 4u,
                    0x44440000u + index),
                "expected scalar buffer x8 seed write")) {
      return 1;
    }
  }
  for (std::uint32_t index = 0; index < 16u; ++index) {
    if (!Expect(scalar_buffer_parity_memory.WriteU32(
                    0x180u + static_cast<std::uint64_t>(index) * 4u,
                    0x55550000u + index),
                "expected scalar buffer x16 seed write")) {
      return 1;
    }
  }
  if (!Expect(scalar_buffer_parity_memory.WriteU32(0x300u, 0u),
              "expected scalar buffer store seed write") ||
      !Expect(scalar_buffer_parity_memory.WriteU32(0x310u, 0u),
              "expected scalar buffer x2 store seed write") ||
      !Expect(scalar_buffer_parity_memory.WriteU32(0x320u, 0u),
              "expected scalar buffer x4 store seed write")) {
    return 1;
  }

  std::string scalar_buffer_parity_error_message;
  std::vector<CompiledInstruction> compiled_scalar_buffer_parity_program;
  if (!Expect(interpreter.CompileProgram(scalar_buffer_parity_program,
                                         &compiled_scalar_buffer_parity_program,
                                         &scalar_buffer_parity_error_message),
              scalar_buffer_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_scalar_buffer_parity_program,
                                         &scalar_buffer_parity_state,
                                         &scalar_buffer_parity_memory,
                                         &scalar_buffer_parity_error_message),
              scalar_buffer_parity_error_message.c_str()) ||
      !Expect(scalar_buffer_parity_state.halted,
              "expected scalar buffer parity program to halt") ||
      !Expect(scalar_buffer_parity_state.exec_mask == 0b0101ULL,
              "expected scalar buffer parity to preserve exec") ||
      !Expect(scalar_buffer_parity_state.sgprs[0] == 0x100u &&
                  scalar_buffer_parity_state.sgprs[1] == 0u &&
                  scalar_buffer_parity_state.sgprs[2] == 0x400u &&
                  scalar_buffer_parity_state.sgprs[3] == 0u &&
                  scalar_buffer_parity_state.sgprs[72] == 0x220u &&
                  scalar_buffer_parity_state.sgprs[96] == 0x11223344u &&
                  scalar_buffer_parity_state.sgprs[97] == 0x55667788u,
              "expected scalar buffer parity to preserve sgprs") ||
      !Expect(scalar_buffer_parity_state.vgprs[18][0] == 0x01010101u &&
                  scalar_buffer_parity_state.vgprs[18][1] == 0x02020202u &&
                  scalar_buffer_parity_state.vgprs[18][2] == 0x03030303u &&
                  scalar_buffer_parity_state.vgprs[18][3] == 0x04040404u,
              "expected scalar buffer parity to preserve vgprs")) {
    return 1;
  }
  if (!Expect(scalar_buffer_parity_state.sgprs[4] == 0x11110000u,
              "expected s_buffer_load_dword result") ||
      !Expect(scalar_buffer_parity_state.sgprs[8] == 0x22220000u &&
                  scalar_buffer_parity_state.sgprs[9] == 0x22220001u,
              "expected s_buffer_load_dwordx2 result") ||
      !Expect(scalar_buffer_parity_state.sgprs[16] == 0x33330000u &&
                  scalar_buffer_parity_state.sgprs[17] == 0x33330001u &&
                  scalar_buffer_parity_state.sgprs[18] == 0x33330002u &&
                  scalar_buffer_parity_state.sgprs[19] == 0x33330003u,
              "expected s_buffer_load_dwordx4 result")) {
    return 1;
  }
  if (!Expect(scalar_buffer_parity_state.sgprs[24] == 0x44440000u &&
                  scalar_buffer_parity_state.sgprs[25] == 0x44440001u &&
                  scalar_buffer_parity_state.sgprs[26] == 0x44440002u &&
                  scalar_buffer_parity_state.sgprs[27] == 0x44440003u &&
                  scalar_buffer_parity_state.sgprs[28] == 0x44440004u &&
                  scalar_buffer_parity_state.sgprs[29] == 0x44440005u &&
                  scalar_buffer_parity_state.sgprs[30] == 0x44440006u &&
                  scalar_buffer_parity_state.sgprs[31] == 0x44440007u,
              "expected s_buffer_load_dwordx8 result") ||
      !Expect(scalar_buffer_parity_state.sgprs[40] == 0x55550000u &&
                  scalar_buffer_parity_state.sgprs[41] == 0x55550001u &&
                  scalar_buffer_parity_state.sgprs[42] == 0x55550002u &&
                  scalar_buffer_parity_state.sgprs[43] == 0x55550003u &&
                  scalar_buffer_parity_state.sgprs[44] == 0x55550004u &&
                  scalar_buffer_parity_state.sgprs[45] == 0x55550005u &&
                  scalar_buffer_parity_state.sgprs[46] == 0x55550006u &&
                  scalar_buffer_parity_state.sgprs[47] == 0x55550007u &&
                  scalar_buffer_parity_state.sgprs[48] == 0x55550008u &&
                  scalar_buffer_parity_state.sgprs[49] == 0x55550009u &&
                  scalar_buffer_parity_state.sgprs[50] == 0x5555000au &&
                  scalar_buffer_parity_state.sgprs[51] == 0x5555000bu &&
                  scalar_buffer_parity_state.sgprs[52] == 0x5555000cu &&
                  scalar_buffer_parity_state.sgprs[53] == 0x5555000du &&
                  scalar_buffer_parity_state.sgprs[54] == 0x5555000eu &&
                  scalar_buffer_parity_state.sgprs[55] == 0x5555000fu,
              "expected s_buffer_load_dwordx16 result")) {
    return 1;
  }
  std::uint32_t scalar_buffer_parity_readback = 0;
  if (!Expect(scalar_buffer_parity_memory.ReadU32(0x300u,
                                                  &scalar_buffer_parity_readback),
              "expected scalar buffer dword store readback") ||
      !Expect(scalar_buffer_parity_readback == 0x11110000u,
              "expected scalar buffer dword store result") ||
      !Expect(scalar_buffer_parity_memory.ReadU32(0x310u,
                                                  &scalar_buffer_parity_readback),
              "expected scalar buffer x2 store readback") ||
      !Expect(scalar_buffer_parity_readback == 0x22220000u,
              "expected scalar buffer x2 low store result") ||
      !Expect(scalar_buffer_parity_memory.ReadU32(0x314u,
                                                  &scalar_buffer_parity_readback),
              "expected scalar buffer x2 store readback") ||
      !Expect(scalar_buffer_parity_readback == 0x22220001u,
              "expected scalar buffer x2 high store result") ||
      !Expect(scalar_buffer_parity_memory.ReadU32(0x320u,
                                                  &scalar_buffer_parity_readback),
              "expected scalar buffer x4 store readback") ||
      !Expect(scalar_buffer_parity_readback == 0x33330000u,
              "expected scalar buffer x4 store result") ||
      !Expect(scalar_buffer_parity_memory.ReadU32(0x32cu,
                                                  &scalar_buffer_parity_readback),
              "expected scalar buffer x4 store readback") ||
      !Expect(scalar_buffer_parity_readback == 0x33330003u,
              "expected scalar buffer x4 tail store result")) {
    return 1;
  }
  if (!Expect(scalar_buffer_parity_memory.ReadU32(0x100u,
                                                  &scalar_buffer_parity_readback) &&
                  scalar_buffer_parity_readback == 0x11110000u &&
                  scalar_buffer_parity_memory.ReadU32(0x114u,
                                                  &scalar_buffer_parity_readback) &&
                  scalar_buffer_parity_readback == 0x22220001u &&
                  scalar_buffer_parity_memory.ReadU32(0x12cu,
                                                  &scalar_buffer_parity_readback) &&
                  scalar_buffer_parity_readback == 0x33330003u,
              "expected scalar buffer source preservation") ||
      !Expect(scalar_buffer_parity_memory.ReadU32(0x15cu,
                                                  &scalar_buffer_parity_readback) &&
                  scalar_buffer_parity_readback == 0x44440007u &&
                  scalar_buffer_parity_memory.ReadU32(0x1bcu,
                                                  &scalar_buffer_parity_readback) &&
                  scalar_buffer_parity_readback == 0x5555000fu,
              "expected scalar buffer tail preservation")) {
    return 1;
  }

  const std::vector<DecodedInstruction> buffer_parity_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORDX4",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORDX2",
                                      InstructionOperand::Vgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x40)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORDX4",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORDX2",
                                      InstructionOperand::Vgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x60)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory buffer_parity_memory(0x2000, 0);
  WaveExecutionState buffer_parity_state{};
  buffer_parity_state.exec_mask = 1u;
  buffer_parity_state.sgprs[8] = 0x200u;
  buffer_parity_state.sgprs[9] = 0u;
  buffer_parity_state.sgprs[10] = 0x400u;
  buffer_parity_state.sgprs[11] = 0u;
  buffer_parity_state.sgprs[30] = 0x12345678u;
  buffer_parity_state.sgprs[31] = 0x9abcdef0u;
  buffer_parity_state.vgprs[40][0] = 0x11111111u;
  buffer_parity_state.vgprs[40][1] = 0x22222222u;
  buffer_parity_state.vgprs[40][2] = 0x33333333u;
  buffer_parity_state.vgprs[40][3] = 0x44444444u;
  buffer_parity_state.vgprs[41][0] = 0x55555555u;
  buffer_parity_state.vgprs[41][1] = 0x66666666u;
  buffer_parity_state.vgprs[41][2] = 0x77777777u;
  buffer_parity_state.vgprs[41][3] = 0x88888888u;
  for (std::uint32_t dword_index = 0; dword_index < 4u; ++dword_index) {
    if (!Expect(buffer_parity_memory.WriteU32(0x200u + dword_index * 4u,
                                              0x33330000u + dword_index),
                "expected buffer parity seed write") ||
        !Expect(buffer_parity_memory.WriteU32(0x220u + dword_index * 4u, 0u),
                "expected buffer parity store seed write")) {
      return 1;
    }
  }
  for (std::uint32_t dword_index = 0; dword_index < 2u; ++dword_index) {
    if (!Expect(buffer_parity_memory.WriteU32(0x240u + dword_index * 4u,
                                              0x44440000u + dword_index),
                "expected buffer parity x2 seed write") ||
        !Expect(buffer_parity_memory.WriteU32(0x260u + dword_index * 4u, 0u),
                "expected buffer parity x2 store seed write")) {
      return 1;
    }
  }
  std::string buffer_parity_error_message;
  std::vector<CompiledInstruction> compiled_buffer_parity_program;
  if (!Expect(interpreter.CompileProgram(buffer_parity_program,
                                         &compiled_buffer_parity_program,
                                         &buffer_parity_error_message),
              buffer_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_buffer_parity_program,
                                         &buffer_parity_state,
                                         &buffer_parity_memory,
                                         &buffer_parity_error_message),
              buffer_parity_error_message.c_str()) ||
      !Expect(buffer_parity_state.halted,
              "expected buffer parity program to halt") ||
      !Expect(buffer_parity_state.vgprs[20][0] == 0x33330000u &&
                  buffer_parity_state.vgprs[21][0] == 0x33330001u &&
                  buffer_parity_state.vgprs[22][0] == 0x33330002u &&
                  buffer_parity_state.vgprs[23][0] == 0x33330003u,
              "expected buffer parity load result")) {
    return 1;
  }
  if (!Expect(buffer_parity_state.vgprs[24][0] == 0x44440000u &&
                  buffer_parity_state.vgprs[25][0] == 0x44440001u,
              "expected buffer parity x2 load result") ||
      !Expect(buffer_parity_state.vgprs[28][0] == 0x0u &&
                  buffer_parity_state.vgprs[29][0] == 0x0u &&
                  buffer_parity_state.vgprs[30][0] == 0x0u,
              "expected unused buffer parity x3 registers to remain clear")) {
    return 1;
  }

  std::uint32_t buffer_parity_readback = 0;
  if (!Expect(buffer_parity_state.sgprs[8] == 0x200u &&
                  buffer_parity_state.sgprs[9] == 0u &&
                  buffer_parity_state.sgprs[10] == 0x400u &&
                  buffer_parity_state.sgprs[11] == 0u &&
                  buffer_parity_state.sgprs[30] == 0x12345678u &&
                  buffer_parity_state.sgprs[31] == 0x9abcdef0u,
              "expected buffer parity sgpr preservation") ||
      !Expect(buffer_parity_state.vgprs[40][0] == 0x11111111u &&
                  buffer_parity_state.vgprs[40][1] == 0x22222222u &&
                  buffer_parity_state.vgprs[40][2] == 0x33333333u &&
                  buffer_parity_state.vgprs[40][3] == 0x44444444u &&
                  buffer_parity_state.vgprs[41][0] == 0x55555555u &&
                  buffer_parity_state.vgprs[41][1] == 0x66666666u &&
                  buffer_parity_state.vgprs[41][2] == 0x77777777u &&
                  buffer_parity_state.vgprs[41][3] == 0x88888888u,
              "expected buffer parity vgpr preservation") ||
      !Expect(buffer_parity_memory.ReadU32(0x220u, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x33330000u &&
                  buffer_parity_memory.ReadU32(0x224u, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x33330001u &&
                  buffer_parity_memory.ReadU32(0x228u, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x33330002u &&
                  buffer_parity_memory.ReadU32(0x22cu, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x33330003u,
              "expected buffer parity store readback")) {
    return 1;
  }
  if (!Expect(buffer_parity_memory.ReadU32(0x260u, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x44440000u &&
                  buffer_parity_memory.ReadU32(0x264u, &buffer_parity_readback) &&
                  buffer_parity_readback == 0x44440001u,
              "expected buffer parity x2 store readback")) {
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

  auto make_buffer_format_descriptor_word3 = [](std::uint32_t data_format,
                                                std::uint32_t num_format) {
    return (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9) |
           (data_format << 12) | (num_format << 19);
  };

  const std::vector<DecodedInstruction> buffer_format_parity_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory buffer_format_parity_memory(0x400, 0);
  WaveExecutionState buffer_format_parity_state{};
  buffer_format_parity_state.exec_mask = 1u;
  buffer_format_parity_state.sgprs[8] = 0x100u;
  buffer_format_parity_state.sgprs[9] = 0u;
  buffer_format_parity_state.sgprs[10] = 0x80u;
  buffer_format_parity_state.sgprs[11] =
      make_buffer_format_descriptor_word3(10u, 4u);
  buffer_format_parity_state.vgprs[0][0] = 0u;
  buffer_format_parity_state.vgprs[40][0] = 0x05u;
  buffer_format_parity_state.vgprs[41][0] = 0x06u;
  buffer_format_parity_state.vgprs[42][0] = 0x07u;
  buffer_format_parity_state.vgprs[43][0] = 0x08u;
  if (!Expect(WriteU8(&buffer_format_parity_memory, 0x100u, 0x01u) &&
                  WriteU8(&buffer_format_parity_memory, 0x101u, 0x02u) &&
                  WriteU8(&buffer_format_parity_memory, 0x102u, 0x03u) &&
                  WriteU8(&buffer_format_parity_memory, 0x103u, 0x04u),
              "expected buffer format parity seed writes")) {
    return 1;
  }

  std::string buffer_format_parity_error_message;
  std::vector<CompiledInstruction> compiled_buffer_format_parity_program;
  std::uint32_t buffer_format_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(buffer_format_parity_program,
                                         &compiled_buffer_format_parity_program,
                                         &buffer_format_parity_error_message),
              buffer_format_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_buffer_format_parity_program,
                                         &buffer_format_parity_state,
                                         &buffer_format_parity_memory,
                                         &buffer_format_parity_error_message),
              buffer_format_parity_error_message.c_str()) ||
      !Expect(buffer_format_parity_state.halted,
              "expected buffer format parity program to halt") ||
      !Expect(buffer_format_parity_state.vgprs[20][0] == 0x01u &&
                  buffer_format_parity_state.vgprs[21][0] == 0x02u &&
                  buffer_format_parity_state.vgprs[22][0] == 0x03u &&
                  buffer_format_parity_state.vgprs[23][0] == 0x04u,
              "expected buffer format parity load result") ||
      !Expect(buffer_format_parity_memory.ReadU32(0x120u,
                                                  &buffer_format_parity_readback),
              "expected buffer format parity store readback") ||
      !Expect(buffer_format_parity_readback == 0x08070605u,
              "expected buffer format parity store result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> typed_buffer_parity_program = {
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_XYZW",
                                       InstructionOperand::Vgpr(20),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Sgpr(24),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(10),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_XYZW",
                                       InstructionOperand::Vgpr(40),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Sgpr(24),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(10),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  const std::uint32_t typed_buffer_dst_sel_word3 =
      (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9);
  LinearExecutionMemory typed_buffer_parity_memory(0x400, 0);
  WaveExecutionState typed_buffer_parity_state{};
  typed_buffer_parity_state.exec_mask = 1u;
  typed_buffer_parity_state.sgprs[24] = 0x200u;
  typed_buffer_parity_state.sgprs[25] = 0u;
  typed_buffer_parity_state.sgprs[26] = 0x80u;
  typed_buffer_parity_state.sgprs[27] = typed_buffer_dst_sel_word3;
  typed_buffer_parity_state.vgprs[0][0] = 0u;
  typed_buffer_parity_state.vgprs[40][0] = 0xa1u;
  typed_buffer_parity_state.vgprs[41][0] = 0xb2u;
  typed_buffer_parity_state.vgprs[42][0] = 0xc3u;
  typed_buffer_parity_state.vgprs[43][0] = 0xd4u;
  if (!Expect(WriteU8(&typed_buffer_parity_memory, 0x200u, 0x11u) &&
                  WriteU8(&typed_buffer_parity_memory, 0x201u, 0x22u) &&
                  WriteU8(&typed_buffer_parity_memory, 0x202u, 0x33u) &&
                  WriteU8(&typed_buffer_parity_memory, 0x203u, 0x44u),
              "expected typed buffer parity seed writes")) {
    return 1;
  }

  std::string typed_buffer_parity_error_message;
  std::vector<CompiledInstruction> compiled_typed_buffer_parity_program;
  std::uint32_t typed_buffer_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(typed_buffer_parity_program,
                                         &compiled_typed_buffer_parity_program,
                                         &typed_buffer_parity_error_message),
              typed_buffer_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_typed_buffer_parity_program,
                                         &typed_buffer_parity_state,
                                         &typed_buffer_parity_memory,
                                         &typed_buffer_parity_error_message),
              typed_buffer_parity_error_message.c_str()) ||
      !Expect(typed_buffer_parity_state.halted,
              "expected typed buffer parity program to halt") ||
      !Expect(typed_buffer_parity_state.vgprs[20][0] == 0x11u &&
                  typed_buffer_parity_state.vgprs[21][0] == 0x22u &&
                  typed_buffer_parity_state.vgprs[22][0] == 0x33u &&
                  typed_buffer_parity_state.vgprs[23][0] == 0x44u,
              "expected typed buffer parity load result") ||
      !Expect(typed_buffer_parity_memory.ReadU32(0x220u,
                                                 &typed_buffer_parity_readback),
              "expected typed buffer parity store readback") ||
      !Expect(typed_buffer_parity_readback == 0xd4c3b2a1u,
              "expected typed buffer parity store result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> buffer_d16_parity_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XYZW",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XYZW",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory buffer_d16_parity_memory(0x400, 0);
  WaveExecutionState buffer_d16_parity_state{};
  buffer_d16_parity_state.exec_mask = 1u;
  buffer_d16_parity_state.sgprs[8] = 0x100u;
  buffer_d16_parity_state.sgprs[9] = 0u;
  buffer_d16_parity_state.sgprs[10] = 0x80u;
  buffer_d16_parity_state.sgprs[11] = make_buffer_format_descriptor_word3(12u, 4u);
  buffer_d16_parity_state.vgprs[0][0] = 0u;
  buffer_d16_parity_state.vgprs[40][0] = 0x22221111u;
  buffer_d16_parity_state.vgprs[41][0] = 0x44443333u;
  if (!Expect(WriteU16(&buffer_d16_parity_memory, 0x100u, 0x1111u) &&
                  WriteU16(&buffer_d16_parity_memory, 0x102u, 0x2222u) &&
                  WriteU16(&buffer_d16_parity_memory, 0x104u, 0x3333u) &&
                  WriteU16(&buffer_d16_parity_memory, 0x106u, 0x4444u),
              "expected buffer d16 parity seed writes")) {
    return 1;
  }

  std::string buffer_d16_parity_error_message;
  std::vector<CompiledInstruction> compiled_buffer_d16_parity_program;
  std::uint16_t buffer_d16_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(buffer_d16_parity_program,
                                         &compiled_buffer_d16_parity_program,
                                         &buffer_d16_parity_error_message),
              buffer_d16_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_buffer_d16_parity_program,
                                         &buffer_d16_parity_state,
                                         &buffer_d16_parity_memory,
                                         &buffer_d16_parity_error_message),
              buffer_d16_parity_error_message.c_str()) ||
      !Expect(buffer_d16_parity_state.halted,
              "expected buffer d16 parity program to halt") ||
      !Expect(buffer_d16_parity_state.vgprs[20][0] == 0x22221111u &&
                  buffer_d16_parity_state.vgprs[21][0] == 0x44443333u,
              "expected buffer d16 parity load result") ||
      !Expect(buffer_d16_parity_memory.LoadU16(0x120u, &buffer_d16_parity_readback) &&
                  buffer_d16_parity_readback == 0x1111u &&
                  buffer_d16_parity_memory.LoadU16(0x122u, &buffer_d16_parity_readback) &&
                  buffer_d16_parity_readback == 0x2222u &&
                  buffer_d16_parity_memory.LoadU16(0x124u, &buffer_d16_parity_readback) &&
                  buffer_d16_parity_readback == 0x3333u &&
                  buffer_d16_parity_memory.LoadU16(0x126u, &buffer_d16_parity_readback) &&
                  buffer_d16_parity_readback == 0x4444u,
              "expected buffer d16 parity store result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> typed_buffer_d16_parity_program = {
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_D16_XYZW",
                                       InstructionOperand::Vgpr(20),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Sgpr(24),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(12),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_D16_XYZW",
                                       InstructionOperand::Vgpr(40),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Sgpr(24),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(12),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory typed_buffer_d16_parity_memory(0x400, 0);
  WaveExecutionState typed_buffer_d16_parity_state{};
  typed_buffer_d16_parity_state.exec_mask = 1u;
  typed_buffer_d16_parity_state.sgprs[24] = 0x200u;
  typed_buffer_d16_parity_state.sgprs[25] = 0u;
  typed_buffer_d16_parity_state.sgprs[26] = 0x80u;
  typed_buffer_d16_parity_state.sgprs[27] = typed_buffer_dst_sel_word3;
  typed_buffer_d16_parity_state.vgprs[0][0] = 0u;
  typed_buffer_d16_parity_state.vgprs[40][0] = 0x66665555u;
  typed_buffer_d16_parity_state.vgprs[41][0] = 0x88887777u;
  if (!Expect(WriteU16(&typed_buffer_d16_parity_memory, 0x200u, 0x5555u) &&
                  WriteU16(&typed_buffer_d16_parity_memory, 0x202u, 0x6666u) &&
                  WriteU16(&typed_buffer_d16_parity_memory, 0x204u, 0x7777u) &&
                  WriteU16(&typed_buffer_d16_parity_memory, 0x206u, 0x8888u),
              "expected typed buffer d16 parity seed writes")) {
    return 1;
  }

  std::string typed_buffer_d16_parity_error_message;
  std::vector<CompiledInstruction> compiled_typed_buffer_d16_parity_program;
  std::uint16_t typed_buffer_d16_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(typed_buffer_d16_parity_program,
                                         &compiled_typed_buffer_d16_parity_program,
                                         &typed_buffer_d16_parity_error_message),
              typed_buffer_d16_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_typed_buffer_d16_parity_program,
                                         &typed_buffer_d16_parity_state,
                                         &typed_buffer_d16_parity_memory,
                                         &typed_buffer_d16_parity_error_message),
              typed_buffer_d16_parity_error_message.c_str()) ||
      !Expect(typed_buffer_d16_parity_state.halted,
              "expected typed buffer d16 parity program to halt") ||
      !Expect(typed_buffer_d16_parity_state.vgprs[20][0] == 0x66665555u &&
                  typed_buffer_d16_parity_state.vgprs[21][0] == 0x88887777u,
              "expected typed buffer d16 parity load result") ||
      !Expect(typed_buffer_d16_parity_memory.LoadU16(0x220u,
                                                     &typed_buffer_d16_parity_readback) &&
                  typed_buffer_d16_parity_readback == 0x5555u &&
                  typed_buffer_d16_parity_memory.LoadU16(0x222u,
                                                     &typed_buffer_d16_parity_readback) &&
                  typed_buffer_d16_parity_readback == 0x6666u &&
                  typed_buffer_d16_parity_memory.LoadU16(0x224u,
                                                     &typed_buffer_d16_parity_readback) &&
                  typed_buffer_d16_parity_readback == 0x7777u &&
                  typed_buffer_d16_parity_memory.LoadU16(0x226u,
                                                     &typed_buffer_d16_parity_readback) &&
                  typed_buffer_d16_parity_readback == 0x8888u,
              "expected typed buffer d16 parity store result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> buffer_d16_hi_parity_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_HI_X",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_HI_X",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  LinearExecutionMemory buffer_d16_hi_parity_memory(0x400, 0);
  WaveExecutionState buffer_d16_hi_parity_state{};
  buffer_d16_hi_parity_state.exec_mask = 1u;
  buffer_d16_hi_parity_state.sgprs[8] = 0x100u;
  buffer_d16_hi_parity_state.sgprs[9] = 0u;
  buffer_d16_hi_parity_state.sgprs[10] = 0x80u;
  buffer_d16_hi_parity_state.sgprs[11] =
      make_buffer_format_descriptor_word3(2u, 4u);
  buffer_d16_hi_parity_state.vgprs[0][0] = 0u;
  buffer_d16_hi_parity_state.vgprs[40][0] = 0x55660000u;
  if (!Expect(WriteU16(&buffer_d16_hi_parity_memory, 0x100u, 0xabcdu),
              "expected buffer d16 hi parity seed write")) {
    return 1;
  }

  std::string buffer_d16_hi_parity_error_message;
  std::vector<CompiledInstruction> compiled_buffer_d16_hi_parity_program;
  std::uint16_t buffer_d16_hi_parity_readback = 0;
  if (!Expect(interpreter.CompileProgram(buffer_d16_hi_parity_program,
                                         &compiled_buffer_d16_hi_parity_program,
                                         &buffer_d16_hi_parity_error_message),
              buffer_d16_hi_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_buffer_d16_hi_parity_program,
                                         &buffer_d16_hi_parity_state,
                                         &buffer_d16_hi_parity_memory,
                                         &buffer_d16_hi_parity_error_message),
              buffer_d16_hi_parity_error_message.c_str()) ||
      !Expect(buffer_d16_hi_parity_state.halted,
              "expected buffer d16 hi parity program to halt") ||
      !Expect(buffer_d16_hi_parity_state.vgprs[20][0] == 0xabcd0000u,
              "expected buffer d16 hi parity load result") ||
      !Expect(buffer_d16_hi_parity_state.vgprs[40][0] == 0x55660000u,
              "expected buffer d16 hi parity source preservation") ||
      !Expect(buffer_d16_hi_parity_memory.LoadU16(0x120u,
                                                  &buffer_d16_hi_parity_readback),
              "expected buffer d16 hi parity store readback") ||
      !Expect(buffer_d16_hi_parity_readback == 0x5566u,
              "expected buffer d16 hi parity store result")) {
    return 1;
  }

  auto make_buffer_format_low_component_word3 = [](std::uint32_t data_format,
                                                   std::uint32_t num_format) {
    return (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9) |
           (data_format << 12) | (num_format << 19);
  };

  const std::vector<DecodedInstruction> buffer_low_component_parity_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_X",
                                      InstructionOperand::Vgpr(70),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_X",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XY",
                                      InstructionOperand::Vgpr(78),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XY",
                                      InstructionOperand::Vgpr(44),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x28)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_X",
                                      InstructionOperand::Vgpr(86),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_X",
                                      InstructionOperand::Vgpr(66),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x30)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  LinearExecutionMemory buffer_low_component_parity_memory(0x400, 0);
  WaveExecutionState buffer_low_component_parity_state{};
  buffer_low_component_parity_state.exec_mask = 0b1011ULL;
  buffer_low_component_parity_state.sgprs[20] = 0x100u;
  buffer_low_component_parity_state.sgprs[21] = 0u;
  buffer_low_component_parity_state.sgprs[22] = 0x80u;
  buffer_low_component_parity_state.sgprs[23] =
      make_buffer_format_low_component_word3(1u, 4u);
  buffer_low_component_parity_state.sgprs[24] = 0x140u;
  buffer_low_component_parity_state.sgprs[25] = 0u;
  buffer_low_component_parity_state.sgprs[26] = 0x80u;
  buffer_low_component_parity_state.sgprs[27] =
      make_buffer_format_low_component_word3(10u, 4u);
  buffer_low_component_parity_state.sgprs[32] = 0x1c0u;
  buffer_low_component_parity_state.sgprs[33] = 0u;
  buffer_low_component_parity_state.sgprs[34] = 0x80u;
  buffer_low_component_parity_state.sgprs[35] =
      make_buffer_format_low_component_word3(5u, 4u);
  buffer_low_component_parity_state.vgprs[0][0] = 0u;
  buffer_low_component_parity_state.vgprs[0][1] = 4u;
  buffer_low_component_parity_state.vgprs[0][2] = 0x0cu;
  buffer_low_component_parity_state.vgprs[0][3] = 8u;
  buffer_low_component_parity_state.vgprs[1][0] = 0u;
  buffer_low_component_parity_state.vgprs[1][1] = 4u;
  buffer_low_component_parity_state.vgprs[1][2] = 0x0cu;
  buffer_low_component_parity_state.vgprs[1][3] = 8u;
  buffer_low_component_parity_state.vgprs[2][0] = 0u;
  buffer_low_component_parity_state.vgprs[2][1] = 4u;
  buffer_low_component_parity_state.vgprs[2][2] = 0x0cu;
  buffer_low_component_parity_state.vgprs[2][3] = 8u;
  buffer_low_component_parity_state.vgprs[3][0] = 0u;
  buffer_low_component_parity_state.vgprs[3][1] = 4u;
  buffer_low_component_parity_state.vgprs[3][2] = 0x0cu;
  buffer_low_component_parity_state.vgprs[3][3] = 8u;
  buffer_low_component_parity_state.vgprs[30][0] = 0x55u;
  buffer_low_component_parity_state.vgprs[30][1] = 0x66u;
  buffer_low_component_parity_state.vgprs[30][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[30][3] = 0x77u;
  buffer_low_component_parity_state.vgprs[44][0] = 0x09u;
  buffer_low_component_parity_state.vgprs[44][1] = 0x19u;
  buffer_low_component_parity_state.vgprs[44][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[44][3] = 0x29u;
  buffer_low_component_parity_state.vgprs[45][0] = 0x0au;
  buffer_low_component_parity_state.vgprs[45][1] = 0x1au;
  buffer_low_component_parity_state.vgprs[45][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[45][3] = 0x2au;
  buffer_low_component_parity_state.vgprs[66][0] = 0x00001234u;
  buffer_low_component_parity_state.vgprs[66][1] = 0x00005678u;
  buffer_low_component_parity_state.vgprs[66][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[66][3] = 0x00009abcu;
  buffer_low_component_parity_state.vgprs[68][0] = 0x01020304u;
  buffer_low_component_parity_state.vgprs[68][1] = 0x11121314u;
  buffer_low_component_parity_state.vgprs[68][2] = 0x21222324u;
  buffer_low_component_parity_state.vgprs[68][3] = 0x31323334u;
  buffer_low_component_parity_state.vgprs[70][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[78][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[79][2] = 0xdeadbeefu;
  buffer_low_component_parity_state.vgprs[86][2] = 0xdeadbeefu;
  if (!Expect(WriteU8(&buffer_low_component_parity_memory, 0x100u, 0x7au),
              "expected buffer low-component x seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x104u, 0x6bu),
              "expected buffer low-component x seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x108u, 0x5cu),
              "expected buffer low-component x seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x140u, 0x01u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x141u, 0x02u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x144u, 0x11u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x145u, 0x12u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x148u, 0x21u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU8(&buffer_low_component_parity_memory, 0x149u, 0x22u),
              "expected buffer low-component xy seed write") ||
      !Expect(WriteU16(&buffer_low_component_parity_memory, 0x1c0u, 0x1234u),
              "expected buffer low-component d16 x seed write") ||
      !Expect(WriteU16(&buffer_low_component_parity_memory, 0x1c4u, 0x5678u),
              "expected buffer low-component d16 x seed write") ||
      !Expect(WriteU16(&buffer_low_component_parity_memory, 0x1c8u, 0x9abcu),
              "expected buffer low-component d16 x seed write")) {
    return 1;
  }

  std::string buffer_low_component_parity_error_message;
  std::vector<CompiledInstruction> compiled_buffer_low_component_parity_program;
  std::uint8_t buffer_low_component_byte_readback = 0;
  std::uint16_t buffer_low_component_short_readback = 0;
  if (!Expect(interpreter.CompileProgram(buffer_low_component_parity_program,
                                         &compiled_buffer_low_component_parity_program,
                                         &buffer_low_component_parity_error_message),
              buffer_low_component_parity_error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_buffer_low_component_parity_program,
                                         &buffer_low_component_parity_state,
                                         &buffer_low_component_parity_memory,
                                         &buffer_low_component_parity_error_message),
              buffer_low_component_parity_error_message.c_str()) ||
      !Expect(buffer_low_component_parity_state.halted,
              "expected buffer low-component parity program to halt") ||
      !Expect(buffer_low_component_parity_state.exec_mask == 0b1011ULL,
              "expected buffer low-component parity to preserve exec") ||
      !Expect(buffer_low_component_parity_state.vgprs[70][0] == 0x7au &&
                  buffer_low_component_parity_state.vgprs[70][1] == 0x6bu &&
                  buffer_low_component_parity_state.vgprs[70][3] == 0x5cu,
              "expected buffer low-component x load result") ||
      !Expect(buffer_low_component_parity_state.vgprs[78][0] == 0x01u &&
                  buffer_low_component_parity_state.vgprs[78][1] == 0x11u &&
                  buffer_low_component_parity_state.vgprs[78][3] == 0x21u,
              "expected buffer low-component xy low load result") ||
      !Expect(buffer_low_component_parity_state.vgprs[79][0] == 0x02u &&
                  buffer_low_component_parity_state.vgprs[79][1] == 0x12u &&
                  buffer_low_component_parity_state.vgprs[79][3] == 0x22u,
              "expected buffer low-component xy high load result") ||
      !Expect(buffer_low_component_parity_state.vgprs[86][0] == 0x00001234u &&
                  buffer_low_component_parity_state.vgprs[86][1] == 0x00005678u &&
                  buffer_low_component_parity_state.vgprs[86][3] == 0x00009abcu,
              "expected buffer low-component d16 x load result") ||
      !Expect(buffer_low_component_parity_state.vgprs[70][2] == 0xdeadbeefu,
              "expected buffer low-component x inactive lane preservation") ||
      !Expect(buffer_low_component_parity_state.vgprs[78][2] == 0xdeadbeefu,
              "expected buffer low-component xy low inactive lane preservation") ||
      !Expect(buffer_low_component_parity_state.vgprs[79][2] == 0xdeadbeefu,
              "expected buffer low-component xy high inactive lane preservation") ||
      !Expect(buffer_low_component_parity_state.vgprs[86][2] == 0xdeadbeefu,
              "expected buffer low-component d16 x inactive lane preservation") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x120u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component x store readback") ||
      !Expect(buffer_low_component_byte_readback == 0x55u,
              "expected buffer low-component x store result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x124u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component x store readback") ||
      !Expect(buffer_low_component_byte_readback == 0x66u,
              "expected buffer low-component x store result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x128u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component x store readback") ||
      !Expect(buffer_low_component_byte_readback == 0x77u,
              "expected buffer low-component x store result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x168u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 0 low read") ||
      !Expect(buffer_low_component_byte_readback == 0x09u,
              "expected buffer low-component xy store lane 0 low result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x169u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 0 high read") ||
      !Expect(buffer_low_component_byte_readback == 0x0au,
              "expected buffer low-component xy store lane 0 high result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x16cu,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 1 low read") ||
      !Expect(buffer_low_component_byte_readback == 0x19u,
              "expected buffer low-component xy store lane 1 low result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x16du,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 1 high read") ||
      !Expect(buffer_low_component_byte_readback == 0x1au,
              "expected buffer low-component xy store lane 1 high result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x170u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 3 low read") ||
      !Expect(buffer_low_component_byte_readback == 0x29u,
              "expected buffer low-component xy store lane 3 low result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x171u,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component xy store lane 3 high read") ||
      !Expect(buffer_low_component_byte_readback == 0x2au,
              "expected buffer low-component xy store lane 3 high result") ||
      !Expect(buffer_low_component_parity_memory.LoadU16(0x1f0u,
                                                         &buffer_low_component_short_readback),
              "expected buffer low-component d16 x store lane 0 read") ||
      !Expect(buffer_low_component_short_readback == 0x1234u,
              "expected buffer low-component d16 x store lane 0 result") ||
      !Expect(buffer_low_component_parity_memory.LoadU16(0x1f4u,
                                                         &buffer_low_component_short_readback),
              "expected buffer low-component d16 x store lane 1 read") ||
      !Expect(buffer_low_component_short_readback == 0x5678u,
              "expected buffer low-component d16 x store lane 1 result") ||
      !Expect(buffer_low_component_parity_memory.LoadU16(0x1f8u,
                                                         &buffer_low_component_short_readback),
              "expected buffer low-component d16 x store lane 3 read") ||
      !Expect(buffer_low_component_short_readback == 0x9abcu,
              "expected buffer low-component d16 x store lane 3 result") ||
      !Expect(buffer_low_component_parity_memory.LoadU8(0x12cu,
                                                        &buffer_low_component_byte_readback),
              "expected buffer low-component inactive x store read") ||
      !Expect(buffer_low_component_byte_readback == 0u,
              "expected buffer low-component inactive x store result") ||
      !Expect(buffer_low_component_parity_memory.LoadU16(0x174u,
                                                         &buffer_low_component_short_readback),
              "expected buffer low-component inactive xy store read") ||
      !Expect(buffer_low_component_short_readback == 0u,
              "expected buffer low-component inactive xy store result") ||
      !Expect(buffer_low_component_parity_memory.LoadU16(0x1fcu,
                                                         &buffer_low_component_short_readback),
              "expected buffer low-component inactive d16 x store read") ||
      !Expect(buffer_low_component_short_readback == 0u,
              "expected buffer low-component inactive d16 x store result")) {
    return 1;
  }

  {
  LinearExecutionMemory buffer_format_high_component_memory(0x400, 0);
  if (!Expect(WriteU8(&buffer_format_high_component_memory, 0x140u, 0x01u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x141u, 0x02u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x142u, 0x03u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x143u, 0x04u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x144u, 0x11u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x145u, 0x12u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x146u, 0x13u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x147u, 0x14u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x148u, 0x21u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x149u, 0x22u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x14au, 0x23u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_high_component_memory, 0x14bu, 0x24u),
              "expected buffer format high-component xyzw seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x180u,
                                                          0x3fc00000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x184u,
                                                          0xc0000000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x188u,
                                                          0x3e800000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x18cu,
                                                          0x40200000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x190u,
                                                          0xbf800000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x194u,
                                                          0x40400000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x1b0u,
                                                          0x40a00000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x1b4u,
                                                          0xbf000000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(buffer_format_high_component_memory.WriteU32(0x1b8u,
                                                          0x41000000u),
              "expected buffer format high-component xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1c0u, 0x0011u),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1c2u, 0x0022u),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1c4u, 0x0055u),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1c6u, 0x0066u),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1c8u, 0x00aau),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x1cau, 0x00bbu),
              "expected buffer format high-component d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x200u, 0x3c00u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x202u, 0x4000u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x204u, 0xc000u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x208u, 0x3400u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x20au, 0x3800u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x20cu, 0x3c00u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x210u, 0x4200u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x212u, 0x4400u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x214u, 0x4600u),
              "expected buffer format high-component d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x280u, 0x3800u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x282u, 0x3c00u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x284u, 0x4000u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x286u, 0xc000u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x288u, 0x3400u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x28au, 0x3800u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x28cu, 0x3c00u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x28eu, 0x4000u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x290u, 0x4200u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x292u, 0x4400u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x294u, 0x4600u),
              "expected buffer format high-component d16 xyzw seed write") ||
      !Expect(WriteU16(&buffer_format_high_component_memory, 0x296u, 0x4800u),
              "expected buffer format high-component d16 xyzw seed write")) {
    return 1;
  }

  WaveExecutionState buffer_format_high_component_state{};
  buffer_format_high_component_state.exec_mask = 0b1011ULL;
  buffer_format_high_component_state.sgprs[24] = 0x140u;
  buffer_format_high_component_state.sgprs[25] = 0u;
  buffer_format_high_component_state.sgprs[26] = 0x80u;
  buffer_format_high_component_state.sgprs[27] =
      make_buffer_format_descriptor_word3(10u, 4u);
  buffer_format_high_component_state.sgprs[28] = 0x180u;
  buffer_format_high_component_state.sgprs[29] = 0u;
  buffer_format_high_component_state.sgprs[30] = 0x80u;
  buffer_format_high_component_state.sgprs[31] =
      make_buffer_format_descriptor_word3(13u, 7u);
  buffer_format_high_component_state.sgprs[32] = 0x1c0u;
  buffer_format_high_component_state.sgprs[33] = 0u;
  buffer_format_high_component_state.sgprs[34] = 0x80u;
  buffer_format_high_component_state.sgprs[35] =
      make_buffer_format_descriptor_word3(5u, 4u);
  buffer_format_high_component_state.sgprs[36] = 0x200u;
  buffer_format_high_component_state.sgprs[37] = 0u;
  buffer_format_high_component_state.sgprs[38] = 0x80u;
  buffer_format_high_component_state.sgprs[39] =
      make_buffer_format_descriptor_word3(12u, 7u);
  buffer_format_high_component_state.sgprs[44] = 0x280u;
  buffer_format_high_component_state.sgprs[45] = 0u;
  buffer_format_high_component_state.sgprs[46] = 0x80u;
  buffer_format_high_component_state.sgprs[47] =
      make_buffer_format_descriptor_word3(12u, 7u);
  buffer_format_high_component_state.vgprs[0][0] = 0u;
  buffer_format_high_component_state.vgprs[0][1] = 4u;
  buffer_format_high_component_state.vgprs[0][2] = 0x0cu;
  buffer_format_high_component_state.vgprs[0][3] = 8u;
  buffer_format_high_component_state.vgprs[1][0] = 0u;
  buffer_format_high_component_state.vgprs[1][1] = 0x0cu;
  buffer_format_high_component_state.vgprs[1][2] = 0x24u;
  buffer_format_high_component_state.vgprs[1][3] = 0x30u;
  buffer_format_high_component_state.vgprs[2][0] = 0u;
  buffer_format_high_component_state.vgprs[2][1] = 4u;
  buffer_format_high_component_state.vgprs[2][2] = 0x0cu;
  buffer_format_high_component_state.vgprs[2][3] = 8u;
  buffer_format_high_component_state.vgprs[3][0] = 0u;
  buffer_format_high_component_state.vgprs[3][1] = 8u;
  buffer_format_high_component_state.vgprs[3][2] = 0x18u;
  buffer_format_high_component_state.vgprs[3][3] = 0x10u;
  buffer_format_high_component_state.vgprs[4][0] = 0u;
  buffer_format_high_component_state.vgprs[4][1] = 8u;
  buffer_format_high_component_state.vgprs[4][2] = 0x18u;
  buffer_format_high_component_state.vgprs[4][3] = 0x10u;
  buffer_format_high_component_state.vgprs[40][0] = 0x05u;
  buffer_format_high_component_state.vgprs[40][1] = 0x15u;
  buffer_format_high_component_state.vgprs[40][3] = 0x25u;
  buffer_format_high_component_state.vgprs[41][0] = 0x06u;
  buffer_format_high_component_state.vgprs[41][1] = 0x16u;
  buffer_format_high_component_state.vgprs[41][3] = 0x26u;
  buffer_format_high_component_state.vgprs[42][0] = 0x07u;
  buffer_format_high_component_state.vgprs[42][1] = 0x17u;
  buffer_format_high_component_state.vgprs[42][3] = 0x27u;
  buffer_format_high_component_state.vgprs[43][0] = 0x08u;
  buffer_format_high_component_state.vgprs[43][1] = 0x18u;
  buffer_format_high_component_state.vgprs[43][3] = 0x28u;
  buffer_format_high_component_state.vgprs[50][0] = 0x40200000u;
  buffer_format_high_component_state.vgprs[50][1] = 0x3f800000u;
  buffer_format_high_component_state.vgprs[50][3] = 0x40c00000u;
  buffer_format_high_component_state.vgprs[51][0] = 0xbf800000u;
  buffer_format_high_component_state.vgprs[51][1] = 0xc0000000u;
  buffer_format_high_component_state.vgprs[51][3] = 0xbf000000u;
  buffer_format_high_component_state.vgprs[52][0] = 0x40800000u;
  buffer_format_high_component_state.vgprs[52][1] = 0x40400000u;
  buffer_format_high_component_state.vgprs[52][3] = 0x41000000u;
  buffer_format_high_component_state.vgprs[60][0] = 0x00220011u;
  buffer_format_high_component_state.vgprs[60][1] = 0x00660055u;
  buffer_format_high_component_state.vgprs[60][3] = 0x00bb00aau;
  buffer_format_high_component_state.vgprs[61][0] = 0x40003c00u;
  buffer_format_high_component_state.vgprs[61][1] = 0x38003400u;
  buffer_format_high_component_state.vgprs[61][3] = 0x44004200u;
  buffer_format_high_component_state.vgprs[62][0] = 0x0000c000u;
  buffer_format_high_component_state.vgprs[62][1] = 0x00003c00u;
  buffer_format_high_component_state.vgprs[62][3] = 0x00004600u;
  buffer_format_high_component_state.vgprs[64][0] = 0x38003400u;
  buffer_format_high_component_state.vgprs[64][1] = 0x3c003800u;
  buffer_format_high_component_state.vgprs[64][3] = 0x44004200u;
  buffer_format_high_component_state.vgprs[65][0] = 0x40003c00u;
  buffer_format_high_component_state.vgprs[65][1] = 0x44004000u;
  buffer_format_high_component_state.vgprs[65][3] = 0x48004600u;
  buffer_format_high_component_state.vgprs[68][0] = 0x01020304u;
  buffer_format_high_component_state.vgprs[68][1] = 0x11121314u;
  buffer_format_high_component_state.vgprs[68][2] = 0x21222324u;
  buffer_format_high_component_state.vgprs[68][3] = 0x31323334u;
  buffer_format_high_component_state.vgprs[71][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[72][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[73][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[74][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[75][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[76][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[77][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[80][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[81][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[82][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[84][2] = 0xdeadbeefu;
  buffer_format_high_component_state.vgprs[85][2] = 0xdeadbeefu;

  const std::vector<DecodedInstruction> buffer_format_high_component_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(71),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XYZ",
                                      InstructionOperand::Vgpr(75),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Sgpr(28),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XYZ",
                                      InstructionOperand::Vgpr(50),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Sgpr(28),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XY",
                                      InstructionOperand::Vgpr(80),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XY",
                                      InstructionOperand::Vgpr(60),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XYZ",
                                      InstructionOperand::Vgpr(81),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Sgpr(36),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XYZ",
                                      InstructionOperand::Vgpr(61),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Sgpr(36),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XYZW",
                                      InstructionOperand::Vgpr(84),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Sgpr(44),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XYZW",
                                      InstructionOperand::Vgpr(64),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Sgpr(44),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<CompiledInstruction> compiled_buffer_format_high_component_program;
  std::string buffer_format_high_component_error_message;
  if (!Expect(interpreter.CompileProgram(buffer_format_high_component_program,
                                         &compiled_buffer_format_high_component_program,
                                         &buffer_format_high_component_error_message),
              buffer_format_high_component_error_message.c_str())) {
    return 1;
  }
  const LinearExecutionMemory initial_buffer_format_high_component_memory =
      buffer_format_high_component_memory;
  const WaveExecutionState initial_buffer_format_high_component_state =
      buffer_format_high_component_state;
  const auto validate_buffer_format_high_component =
      [&](const WaveExecutionState& state,
          const LinearExecutionMemory& memory,
          const char* mode) -> bool {
    static constexpr std::array<std::size_t, 3> kObservedLanes = {0u, 1u, 3u};
    const auto expect_lane_values =
        [&](std::uint16_t reg, const auto& expected, const char* label) {
          for (std::size_t index = 0; index < kObservedLanes.size(); ++index) {
            if (!Expect(state.vgprs[reg][kObservedLanes[index]] == expected[index],
                        label)) {
              return false;
            }
          }
          return true;
        };
    const auto expect_inactive_lane_preserved = [&](std::uint16_t reg,
                                                    const char* label) {
      return Expect(state.vgprs[reg][2] == 0xdeadbeefu,
                    (std::string(mode) + label).c_str());
    };
    const auto expect_sgprs_preserved =
        [&](std::uint16_t begin, std::uint16_t end, const char* label) {
          for (std::uint16_t sgpr = begin; sgpr <= end; ++sgpr) {
            if (state.sgprs[sgpr] !=
                initial_buffer_format_high_component_state.sgprs[sgpr]) {
              return Expect(false, (std::string(mode) + label).c_str());
            }
          }
          return true;
        };
    const auto expect_vgpr_range_preserved =
        [&](std::uint16_t begin, std::uint16_t end, const char* label) {
          for (std::uint16_t vgpr = begin; vgpr <= end; ++vgpr) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
              if (state.vgprs[vgpr][lane] !=
                  initial_buffer_format_high_component_state.vgprs[vgpr][lane]) {
                return Expect(false, (std::string(mode) + label).c_str());
              }
            }
          }
          return true;
        };
    const auto expect_u8_sequence =
        [&](const auto& addresses, const auto& expected, const char* label) {
          if (addresses.size() != expected.size()) {
            return Expect(false, (std::string(mode) + label).c_str());
          }
          for (std::size_t i = 0; i < addresses.size(); ++i) {
            std::uint8_t current = 0;
            if (!Expect(memory.LoadU8(addresses[i], &current),
                        (std::string(mode) + " buffer format store read").c_str()) ||
                !Expect(current == expected[i], (std::string(mode) + label).c_str())) {
              return false;
            }
          }
          return true;
        };
    const auto expect_u16_sequence =
        [&](const auto& addresses, const auto& expected, const char* label) {
          if (addresses.size() != expected.size()) {
            return Expect(false, (std::string(mode) + label).c_str());
          }
          for (std::size_t i = 0; i < addresses.size(); ++i) {
            std::uint16_t current = 0;
            if (!Expect(memory.LoadU16(addresses[i], &current),
                        (std::string(mode) + " buffer format store read").c_str()) ||
                !Expect(current == expected[i], (std::string(mode) + label).c_str())) {
              return false;
            }
          }
          return true;
        };
    const auto expect_u32_sequence =
        [&](const auto& addresses, const auto& expected, const char* label) {
          if (addresses.size() != expected.size()) {
            return Expect(false, (std::string(mode) + label).c_str());
          }
          for (std::size_t i = 0; i < addresses.size(); ++i) {
            std::uint32_t current = 0;
            if (!Expect(memory.ReadU32(addresses[i], &current),
                        (std::string(mode) + " buffer format store read").c_str()) ||
                !Expect(current == expected[i], (std::string(mode) + label).c_str())) {
              return false;
            }
          }
          return true;
        };
    static constexpr std::array<std::uint32_t, 3> kLoadXyzwX = {0x01u, 0x11u, 0x21u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzwY = {0x02u, 0x12u, 0x22u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzwZ = {0x03u, 0x13u, 0x23u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzwW = {0x04u, 0x14u, 0x24u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzX = {0x3fc00000u, 0x40200000u, 0x40a00000u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzY = {0xc0000000u, 0xbf800000u, 0xbf000000u};
    static constexpr std::array<std::uint32_t, 3> kLoadXyzZ = {0x3e800000u, 0x40400000u, 0x41000000u};
    static constexpr std::array<std::uint32_t, 3> kLoadD16Xy = {0x00220011u, 0x00660055u, 0x00bb00aau};
    static constexpr std::array<std::uint32_t, 3> kLoadD16XyzXy = {0x40003c00u, 0x38003400u, 0x44004200u};
    static constexpr std::array<std::uint32_t, 3> kLoadD16XyzZ = {0x0000c000u, 0x00003c00u, 0x00004600u};
    static constexpr std::array<std::uint32_t, 3> kLoadD16XyzwXy = {0x3c003800u, 0x38003400u, 0x44004200u};
    static constexpr std::array<std::uint32_t, 3> kLoadD16XyzwZw = {0xc0004000u, 0x40003c00u, 0x48004600u};
    static constexpr std::array<std::uint64_t, 12> kXyzwSourceAddresses = {
        0x140u, 0x141u, 0x142u, 0x143u, 0x144u, 0x145u, 0x146u, 0x147u,
        0x148u, 0x149u, 0x14au, 0x14bu};
    static constexpr std::array<std::uint8_t, 12> kXyzwSourceValues = {
        0x01u, 0x02u, 0x03u, 0x04u, 0x11u, 0x12u, 0x13u, 0x14u,
        0x21u, 0x22u, 0x23u, 0x24u};
    static constexpr std::array<std::uint64_t, 9> kXyzSourceAddresses = {
        0x180u, 0x184u, 0x188u, 0x18cu, 0x190u, 0x194u, 0x1b0u, 0x1b4u, 0x1b8u};
    static constexpr std::array<std::uint32_t, 9> kXyzSourceValues = {
        0x3fc00000u, 0xc0000000u, 0x3e800000u, 0x40200000u, 0xbf800000u,
        0x40400000u, 0x40a00000u, 0xbf000000u, 0x41000000u};
    static constexpr std::array<std::uint64_t, 6> kD16XySourceAddresses = {
        0x1c0u, 0x1c2u, 0x1c4u, 0x1c6u, 0x1c8u, 0x1cau};
    static constexpr std::array<std::uint16_t, 6> kD16XySourceValues = {
        0x0011u, 0x0022u, 0x0055u, 0x0066u, 0x00aau, 0x00bbu};
    static constexpr std::array<std::uint64_t, 9> kD16XyzSourceAddresses = {
        0x200u, 0x202u, 0x204u, 0x208u, 0x20au, 0x20cu, 0x210u, 0x212u, 0x214u};
    static constexpr std::array<std::uint16_t, 9> kD16XyzSourceValues = {
        0x3c00u, 0x4000u, 0xc000u, 0x3400u, 0x3800u, 0x3c00u, 0x4200u,
        0x4400u, 0x4600u};
    static constexpr std::array<std::uint64_t, 12> kD16XyzwSourceAddresses = {
        0x280u, 0x282u, 0x284u, 0x286u, 0x288u, 0x28au, 0x28cu, 0x28eu,
        0x290u, 0x292u, 0x294u, 0x296u};
    static constexpr std::array<std::uint16_t, 12> kD16XyzwSourceValues = {
        0x3800u, 0x3c00u, 0x4000u, 0xc000u, 0x3400u, 0x3800u, 0x3c00u,
        0x4000u, 0x4200u, 0x4400u, 0x4600u, 0x4800u};
    static constexpr std::array<std::uint64_t, 12> kXyzwStoreAddresses = {
        0x160u, 0x161u, 0x162u, 0x163u, 0x164u, 0x165u, 0x166u, 0x167u,
        0x168u, 0x169u, 0x16au, 0x16bu};
    static constexpr std::array<std::uint8_t, 12> kXyzwStoreValues = {
        0x05u, 0x06u, 0x07u, 0x08u, 0x15u, 0x16u, 0x17u, 0x18u,
        0x25u, 0x26u, 0x27u, 0x28u};
    static constexpr std::array<std::uint64_t, 9> kXyzStoreAddresses = {
        0x1a0u, 0x1a4u, 0x1a8u, 0x1acu, 0x1b0u, 0x1b4u, 0x1d0u, 0x1d4u, 0x1d8u};
    static constexpr std::array<std::uint32_t, 9> kXyzStoreValues = {
        0x40200000u, 0xbf800000u, 0x40800000u, 0x3f800000u, 0xc0000000u,
        0x40400000u, 0x40c00000u, 0xbf000000u, 0x41000000u};
    static constexpr std::array<std::uint64_t, 6> kD16XyStoreAddresses = {
        0x1e0u, 0x1e2u, 0x1e4u, 0x1e6u, 0x1e8u, 0x1eau};
    static constexpr std::array<std::uint16_t, 6> kD16XyStoreValues = {
        0x0011u, 0x0022u, 0x0055u, 0x0066u, 0x00aau, 0x00bbu};
    static constexpr std::array<std::uint64_t, 9> kD16XyzStoreAddresses = {
        0x220u, 0x222u, 0x224u, 0x228u, 0x22au, 0x22cu, 0x230u, 0x232u, 0x234u};
    static constexpr std::array<std::uint16_t, 9> kD16XyzStoreValues = {
        0x3c00u, 0x4000u, 0xc000u, 0x3400u, 0x3800u, 0x3c00u, 0x4200u,
        0x4400u, 0x4600u};
    static constexpr std::array<std::uint64_t, 12> kD16XyzwStoreAddresses = {
        0x2a0u, 0x2a2u, 0x2a4u, 0x2a6u, 0x2a8u, 0x2aau, 0x2acu, 0x2aeu,
        0x2b0u, 0x2b2u, 0x2b4u, 0x2b6u};
    static constexpr std::array<std::uint16_t, 12> kD16XyzwStoreValues = {
        0x3400u, 0x3800u, 0x3c00u, 0x4000u, 0x3800u, 0x3c00u, 0x4000u,
        0x4400u, 0x4200u, 0x4400u, 0x4600u, 0x4800u};
    return Expect(state.halted,
                  (std::string(mode) + " buffer format high-component program to halt").c_str()) &&
           Expect(state.exec_mask == 0b1011ULL,
                  (std::string(mode) + " buffer format high-component preserves exec").c_str()) &&
           expect_sgprs_preserved(24u, 47u,
                                  " buffer format high-component preserves descriptors") &&
           expect_vgpr_range_preserved(0u, 4u,
                                       " buffer format high-component preserves addresses") &&
           expect_vgpr_range_preserved(40u, 43u,
                                       " buffer format high-component preserves xyzw store sources") &&
           expect_vgpr_range_preserved(50u, 52u,
                                       " buffer format high-component preserves xyz store sources") &&
           expect_vgpr_range_preserved(60u, 65u,
                                       " buffer format high-component preserves d16 store sources") &&
           expect_vgpr_range_preserved(68u, 68u,
                                       " buffer format high-component preserves unrelated vgprs") &&
           expect_lane_values(71, kLoadXyzwX,
                              (std::string(mode) + " buffer format xyzw load x result").c_str()) &&
           expect_lane_values(72, kLoadXyzwY,
                              (std::string(mode) + " buffer format xyzw load y result").c_str()) &&
           expect_lane_values(73, kLoadXyzwZ,
                              (std::string(mode) + " buffer format xyzw load z result").c_str()) &&
           expect_lane_values(74, kLoadXyzwW,
                              (std::string(mode) + " buffer format xyzw load w result").c_str()) &&
           expect_lane_values(75, kLoadXyzX,
                              (std::string(mode) + " buffer format xyz load x result").c_str()) &&
           expect_lane_values(76, kLoadXyzY,
                              (std::string(mode) + " buffer format xyz load y result").c_str()) &&
           expect_lane_values(77, kLoadXyzZ,
                              (std::string(mode) + " buffer format xyz load z result").c_str()) &&
           expect_lane_values(80, kLoadD16Xy,
                              (std::string(mode) + " buffer format d16 xy load result").c_str()) &&
           expect_lane_values(81, kLoadD16XyzXy,
                              (std::string(mode) + " buffer format d16 xyz packed xy result").c_str()) &&
           expect_lane_values(82, kLoadD16XyzZ,
                              (std::string(mode) + " buffer format d16 xyz z result").c_str()) &&
           expect_lane_values(84, kLoadD16XyzwXy,
                              (std::string(mode) + " buffer format d16 xyzw packed xy result").c_str()) &&
           expect_lane_values(85, kLoadD16XyzwZw,
                              (std::string(mode) + " buffer format d16 xyzw packed zw result").c_str()) &&
           expect_inactive_lane_preserved(71, " inactive buffer format xyzw lane remains untouched") &&
           expect_inactive_lane_preserved(75, " inactive buffer format xyz lane remains untouched") &&
           expect_inactive_lane_preserved(80, " inactive buffer format d16 xy lane remains untouched") &&
           expect_inactive_lane_preserved(81, " inactive buffer format d16 xyz packed xy lane remains untouched") &&
           expect_inactive_lane_preserved(82, " inactive buffer format d16 xyz z lane remains untouched") &&
           expect_inactive_lane_preserved(84, " inactive buffer format d16 xyzw packed xy lane remains untouched") &&
           expect_inactive_lane_preserved(85, " inactive buffer format d16 xyzw packed zw lane remains untouched") &&
           expect_u8_sequence(kXyzwStoreAddresses,
                              kXyzwStoreValues,
                              " buffer format high-component xyzw store result") &&
           expect_u32_sequence(kXyzStoreAddresses,
                               kXyzStoreValues,
                               " buffer format high-component xyz store result") &&
           expect_u16_sequence(kD16XyStoreAddresses,
                               kD16XyStoreValues,
                               " buffer format high-component d16 xy store result") &&
           expect_u16_sequence(kD16XyzStoreAddresses,
                               kD16XyzStoreValues,
                               " buffer format high-component d16 xyz store result") &&
           expect_u16_sequence(kD16XyzwStoreAddresses,
                               kD16XyzwStoreValues,
                               " buffer format high-component d16 xyzw store result");
  };
  if (!Expect(interpreter.ExecuteProgram(buffer_format_high_component_program,
                                         &buffer_format_high_component_state,
                                         &buffer_format_high_component_memory,
                                         &buffer_format_high_component_error_message),
              buffer_format_high_component_error_message.c_str()) ||
      !validate_buffer_format_high_component(buffer_format_high_component_state,
                                             buffer_format_high_component_memory,
                                             "decoded")) {
    return 1;
  }

  LinearExecutionMemory compiled_buffer_format_high_component_memory =
      initial_buffer_format_high_component_memory;
  WaveExecutionState compiled_buffer_format_high_component_state =
      initial_buffer_format_high_component_state;
  if (!Expect(interpreter.ExecuteProgram(
                  compiled_buffer_format_high_component_program,
                  &compiled_buffer_format_high_component_state,
                  &compiled_buffer_format_high_component_memory,
                  &buffer_format_high_component_error_message),
              buffer_format_high_component_error_message.c_str()) ||
      !validate_buffer_format_high_component(
          compiled_buffer_format_high_component_state,
          compiled_buffer_format_high_component_memory, "compiled")) {
    return 1;
  }
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
