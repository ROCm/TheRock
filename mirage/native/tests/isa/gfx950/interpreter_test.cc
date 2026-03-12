#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/execution_memory.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx950/interpreter.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

void SplitU64(std::uint64_t value,
              std::uint32_t* low,
              std::uint32_t* high) {
  if (low != nullptr) {
    *low = static_cast<std::uint32_t>(value);
  }
  if (high != nullptr) {
    *high = static_cast<std::uint32_t>(value >> 32);
  }
}

std::array<std::uint32_t, 4> OneDword(std::uint32_t value) {
  return {value, 0u, 0u, 0u};
}

std::array<std::uint32_t, 4> TwoDwords(std::uint64_t value) {
  std::array<std::uint32_t, 4> dwords{};
  SplitU64(value, &dwords[0], &dwords[1]);
  return dwords;
}

std::array<std::uint32_t, 4> FourDwords(std::uint64_t low,
                                        std::uint64_t high) {
  std::array<std::uint32_t, 4> dwords{};
  SplitU64(low, &dwords[0], &dwords[1]);
  SplitU64(high, &dwords[2], &dwords[3]);
  return dwords;
}

std::uint64_t ComposeU64(std::uint32_t low, std::uint32_t high) {
  return static_cast<std::uint64_t>(low) |
         (static_cast<std::uint64_t>(high) << 32);
}

bool WriteU64(mirage::sim::isa::LinearExecutionMemory* memory,
              std::uint64_t address,
              std::uint64_t value) {
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  SplitU64(value, &low, &high);
  return memory->WriteU32(address, low) && memory->WriteU32(address + 4, high);
}

bool ReadU64(const mirage::sim::isa::LinearExecutionMemory& memory,
             std::uint64_t address,
             std::uint64_t* value) {
  if (value == nullptr) {
    return false;
  }
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  return memory.ReadU32(address, &low) && memory.ReadU32(address + 4, &high) &&
         ((*value = ComposeU64(low, high)), true);
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

bool ReadU8(const mirage::sim::isa::LinearExecutionMemory& memory,
            std::uint64_t address,
            std::uint8_t* value) {
  if (value == nullptr) {
    return false;
  }
  std::array<std::byte, 1> bytes{};
  if (!memory.Load(address, std::span<std::byte>(bytes.data(), bytes.size()))) {
    return false;
  }
  *value = static_cast<std::uint8_t>(bytes[0]);
  return true;
}

bool ReadU16(const mirage::sim::isa::LinearExecutionMemory& memory,
             std::uint64_t address,
             std::uint16_t* value) {
  if (value == nullptr) {
    return false;
  }
  std::array<std::byte, 2> bytes{};
  if (!memory.Load(address, std::span<std::byte>(bytes.data(), bytes.size()))) {
    return false;
  }
  std::memcpy(value, bytes.data(), sizeof(*value));
  return true;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t DoubleBits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void SetLane0VgprU64(mirage::sim::isa::WaveExecutionState* state,
                     std::uint16_t reg,
                     std::uint64_t value) {
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  SplitU64(value, &low, &high);
  state->vgprs[reg][0] = low;
  state->vgprs[reg + 1][0] = high;
}

std::uint64_t ReadLane0VgprU64(const mirage::sim::isa::WaveExecutionState& state,
                               std::uint16_t reg) {
  return ComposeU64(state.vgprs[reg][0], state.vgprs[reg + 1][0]);
}

struct AtomicSemanticCase {
  std::string_view opcode;
  std::uint8_t memory_dword_count = 0;
  std::uint8_t data_dword_count = 0;
  std::array<std::uint32_t, 4> initial_memory{};
  std::array<std::uint32_t, 4> data{};
  std::array<std::uint32_t, 4> expected_memory{};
  std::array<std::uint32_t, 4> expected_return{};
};

struct SaveexecSemanticCase {
  std::string_view opcode;
  std::uint64_t initial_exec = 0;
  std::uint64_t source = 0;
  std::uint64_t expected_exec = 0;
};

struct ScalarUnaryCase {
  std::string_view opcode;
  std::uint32_t source = 0;
  std::uint32_t initial_dest = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarBinaryCase {
  std::string_view opcode;
  std::uint32_t lhs = 0;
  std::uint32_t rhs = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairUnaryCase {
  std::string_view opcode;
  std::uint64_t source = 0;
  std::uint64_t initial_dest = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairFromScalarUnaryCase {
  std::string_view opcode;
  std::uint32_t source = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarFromPairUnaryCase {
  std::string_view opcode;
  std::uint64_t source = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairBinaryCase {
  std::string_view opcode;
  std::uint64_t lhs = 0;
  std::uint64_t rhs = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairCompareCase {
  std::string_view opcode;
  std::uint64_t lhs = 0;
  std::uint64_t rhs = 0;
  bool expected_scc = false;
};

bool RunAtomicSemanticCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const AtomicSemanticCase& test_case) {
  using namespace mirage::sim::isa;

  constexpr std::uint64_t kAtomicAddress = 0x100;
  constexpr std::uint16_t kAddressReg = 0;
  constexpr std::uint16_t kDataReg = 20;
  constexpr std::uint16_t kReturnReg = 40;
  constexpr std::uint16_t kScalarBaseReg = 2;

  LinearExecutionMemory memory(0x400, 0);
  for (std::uint8_t dword_index = 0; dword_index < test_case.memory_dword_count;
       ++dword_index) {
    if (!memory.WriteU32(kAtomicAddress + static_cast<std::uint64_t>(dword_index) * 4u,
                         test_case.initial_memory[dword_index])) {
      std::cerr << test_case.opcode << ": failed to seed memory\n";
      return false;
    }
  }

  static thread_local WaveExecutionState state;
  state = {};
  state.exec_mask = 0x1ULL;
  SetLane0VgprU64(&state, kAddressReg, kAtomicAddress);
  state.sgprs[kScalarBaseReg] = 0;
  state.sgprs[kScalarBaseReg + 1] = 0;
  for (std::uint8_t dword_index = 0; dword_index < test_case.data_dword_count;
       ++dword_index) {
    state.vgprs[kDataReg + dword_index][0] = test_case.data[dword_index];
  }
  for (std::uint8_t dword_index = 0; dword_index < test_case.memory_dword_count;
       ++dword_index) {
    state.vgprs[kReturnReg + dword_index][0] = 0xdeadbeefu;
  }

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::FiveOperand(test_case.opcode,
                                      InstructionOperand::Vgpr(kReturnReg),
                                      InstructionOperand::Vgpr(kAddressReg),
                                      InstructionOperand::Vgpr(kDataReg),
                                      InstructionOperand::Sgpr(kScalarBaseReg),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::string error_message;
  if (!interpreter.ExecuteProgram(program, &state, &memory, &error_message)) {
    std::cerr << test_case.opcode << ": " << error_message << '\n';
    return false;
  }

  for (std::uint8_t dword_index = 0; dword_index < test_case.memory_dword_count;
       ++dword_index) {
    std::uint32_t actual_value = 0;
    if (!memory.ReadU32(kAtomicAddress + static_cast<std::uint64_t>(dword_index) * 4u,
                        &actual_value)) {
      std::cerr << test_case.opcode << ": failed to read result memory\n";
      return false;
    }
    if (actual_value != test_case.expected_memory[dword_index]) {
      std::cerr << test_case.opcode << ": memory dword " << +dword_index
                << " mismatch\n";
      return false;
    }
    if (state.vgprs[kReturnReg + dword_index][0] !=
        test_case.expected_return[dword_index]) {
      std::cerr << test_case.opcode << ": return dword " << +dword_index
                << " mismatch\n";
      return false;
    }
  }
  return true;
}

bool RunBufferAtomicSemanticCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const AtomicSemanticCase& test_case,
    bool return_prior_value) {
  using namespace mirage::sim::isa;

  constexpr std::uint64_t kAtomicAddress = 0x100;
  constexpr std::uint16_t kDataReg = 20;
  constexpr std::uint16_t kResourceReg = 2;

  LinearExecutionMemory memory(0x400, 0);
  for (std::uint8_t dword_index = 0; dword_index < test_case.memory_dword_count;
       ++dword_index) {
    if (!memory.WriteU32(kAtomicAddress + static_cast<std::uint64_t>(dword_index) * 4u,
                         test_case.initial_memory[dword_index])) {
      std::cerr << test_case.opcode << ": failed to seed buffer atomic memory\n";
      return false;
    }
  }

  static thread_local WaveExecutionState state;
  state = {};
  state.exec_mask = 0x1ULL;
  state.sgprs[kResourceReg] = static_cast<std::uint32_t>(kAtomicAddress);
  state.sgprs[kResourceReg + 1] = 0u;
  state.sgprs[kResourceReg + 2] = 0x100u;
  state.sgprs[kResourceReg + 3] = 0u;
  for (std::uint8_t dword_index = 0; dword_index < test_case.data_dword_count;
       ++dword_index) {
    state.vgprs[kDataReg + dword_index][0] = test_case.data[dword_index];
  }

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::SixOperand(test_case.opcode,
                                     InstructionOperand::Vgpr(kDataReg),
                                     InstructionOperand::Imm32(0),
                                     InstructionOperand::Sgpr(kResourceReg),
                                     InstructionOperand::Imm32(0),
                                     InstructionOperand::Imm32(0),
                                     InstructionOperand::Imm32(return_prior_value
                                                                   ? 1u
                                                                   : 0u)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::string error_message;
  if (!interpreter.ExecuteProgram(program, &state, &memory, &error_message)) {
    std::cerr << test_case.opcode << ": " << error_message << '\n';
    return false;
  }

  for (std::uint8_t dword_index = 0; dword_index < test_case.memory_dword_count;
       ++dword_index) {
    std::uint32_t actual_value = 0;
    if (!memory.ReadU32(kAtomicAddress + static_cast<std::uint64_t>(dword_index) * 4u,
                        &actual_value)) {
      std::cerr << test_case.opcode
                << ": failed to read buffer atomic result memory\n";
      return false;
    }
    if (actual_value != test_case.expected_memory[dword_index]) {
      std::cerr << test_case.opcode << ": buffer atomic memory dword "
                << +dword_index << " mismatch\n";
      return false;
    }
  }

  for (std::uint8_t dword_index = 0; dword_index < test_case.data_dword_count;
       ++dword_index) {
    const std::uint32_t expected_value =
        (return_prior_value && dword_index < test_case.memory_dword_count)
            ? test_case.expected_return[dword_index]
            : test_case.data[dword_index];
    if (state.vgprs[kDataReg + dword_index][0] != expected_value) {
      std::cerr << test_case.opcode << ": buffer atomic data dword "
                << +dword_index << " mismatch\n";
      return false;
    }
  }
  return true;
}

bool RunSaveexecSemanticCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const SaveexecSemanticCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary(test_case.opcode, InstructionOperand::Sgpr(40),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  auto state = std::make_unique<WaveExecutionState>();
  state->exec_mask = test_case.initial_exec;
  SplitU64(test_case.source, &state->sgprs[20], &state->sgprs[21]);

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, state.get(),
                                    &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, state.get(),
                                         &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const std::uint64_t saved_exec =
      ComposeU64(state->sgprs[40], state->sgprs[41]);
  const char* mode = use_compiled_program ? "compiled" : "decoded";
  if (!Expect(state->halted, "expected saveexec test program to halt")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(saved_exec == test_case.initial_exec,
              "expected saveexec destination to capture previous exec")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(state->exec_mask == test_case.expected_exec,
              "expected saveexec result exec mask")) {
    std::cerr << test_case.opcode << ' ' << mode << " actual=0x" << std::hex
              << state->exec_mask << " expected=0x" << test_case.expected_exec
              << std::dec << '\n';
    return false;
  }
  if (!Expect(state->scc == (test_case.expected_exec != 0),
              "expected saveexec SCC to reflect result nonzero")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  return true;
}

bool RunScalarUnaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarUnaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary(test_case.opcode, InstructionOperand::Sgpr(40),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.source;
  state.sgprs[40] = test_case.initial_dest;

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const char* mode = use_compiled_program ? "compiled" : "decoded";
  if (!Expect(state.halted, "expected scalar unary test to halt")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(state.sgprs[40] == test_case.expected,
              "expected scalar unary result")) {
    std::cerr << test_case.opcode << ' ' << mode << " actual=0x" << std::hex
              << state.sgprs[40] << " expected=0x" << test_case.expected
              << std::dec << '\n';
    return false;
  }
  if (!Expect(state.scc == test_case.expected_scc,
              "expected scalar unary SCC")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  return true;
}

bool RunScalarBinaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarBinaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary(test_case.opcode, InstructionOperand::Sgpr(40),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Sgpr(24)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.lhs;
  state.sgprs[24] = test_case.rhs;

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const char* mode = use_compiled_program ? "compiled" : "decoded";
  if (!Expect(state.halted, "expected scalar binary test to halt")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(state.sgprs[40] == test_case.expected,
              "expected scalar binary result")) {
    std::cerr << test_case.opcode << ' ' << mode << " actual=0x" << std::hex
              << state.sgprs[40] << " expected=0x" << test_case.expected
              << std::dec << '\n';
    return false;
  }
  if (!Expect(state.scc == test_case.expected_scc,
              "expected scalar binary SCC")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairUnaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairUnaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary(test_case.opcode, InstructionOperand::Sgpr(40),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  SplitU64(test_case.initial_dest, &state.sgprs[40], &state.sgprs[41]);
  SplitU64(test_case.source, &state.sgprs[20], &state.sgprs[21]);

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected scalar pair unary test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair unary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair unary SCC")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairCompareCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairCompareCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::TwoOperand(test_case.opcode, InstructionOperand::Sgpr(20),
                                     InstructionOperand::Sgpr(24)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  SplitU64(test_case.lhs, &state.sgprs[20], &state.sgprs[21]);
  SplitU64(test_case.rhs, &state.sgprs[24], &state.sgprs[25]);

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const char* mode = use_compiled_program ? "compiled" : "decoded";
  if (!Expect(state.halted, "expected scalar pair compare test to halt")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(state.scc == test_case.expected_scc,
              "expected scalar pair compare SCC")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairFromScalarUnaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairFromScalarUnaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary(test_case.opcode, InstructionOperand::Sgpr(40),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.source;

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected scalar pair-from-scalar unary test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair-from-scalar unary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair-from-scalar unary SCC")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }
  return true;
}

bool RunScalarFromPairUnaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarFromPairUnaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary(test_case.opcode, InstructionOperand::Sgpr(40),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  SplitU64(test_case.source, &state.sgprs[20], &state.sgprs[21]);

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const char* mode = use_compiled_program ? "compiled" : "decoded";
  if (!Expect(state.halted, "expected scalar-from-pair unary test to halt")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  if (!Expect(state.sgprs[40] == test_case.expected,
              "expected scalar-from-pair unary result")) {
    std::cerr << test_case.opcode << ' ' << mode << " actual=0x" << std::hex
              << state.sgprs[40] << " expected=0x" << test_case.expected
              << std::dec << '\n';
    return false;
  }
  if (!Expect(state.scc == test_case.expected_scc,
              "expected scalar-from-pair unary SCC")) {
    std::cerr << test_case.opcode << ' ' << mode << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairBinaryCase(
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairBinaryCase& test_case,
    bool use_compiled_program) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Binary(test_case.opcode, InstructionOperand::Sgpr(40),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Sgpr(24)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  static thread_local WaveExecutionState state;
  state = {};
  state.scc = test_case.initial_scc;
  SplitU64(test_case.lhs, &state.sgprs[20], &state.sgprs[21]);
  SplitU64(test_case.rhs, &state.sgprs[24], &state.sgprs[25]);

  std::string error_message;
  if (use_compiled_program) {
    std::vector<CompiledInstruction> compiled_program;
    if (!interpreter.CompileProgram(program, &compiled_program, &error_message)) {
      std::cerr << test_case.opcode << " compile: " << error_message << '\n';
      return false;
    }
    if (!interpreter.ExecuteProgram(compiled_program, &state, &error_message)) {
      std::cerr << test_case.opcode << " compiled execute: " << error_message
                << '\n';
      return false;
    }
  } else if (!interpreter.ExecuteProgram(program, &state, &error_message)) {
    std::cerr << test_case.opcode << " decoded execute: " << error_message
              << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected scalar pair binary test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair binary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair binary SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << result
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunDsPairReturnProgramTest(
    const mirage::sim::isa::Gfx950Interpreter& interpreter) {
  using namespace mirage::sim::isa;

  std::string error_message;
  const std::vector<DecodedInstruction> ds_pair_return_program = {
      DecodedInstruction::SixOperand(
          "DS_WRXCHG2_RTN_B32", InstructionOperand::Vgpr(40),
          InstructionOperand::Vgpr(0), InstructionOperand::Vgpr(2),
          InstructionOperand::Vgpr(3), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(3)),
      DecodedInstruction::SixOperand(
          "DS_WRXCHG2ST64_RTN_B32", InstructionOperand::Vgpr(42),
          InstructionOperand::Vgpr(4), InstructionOperand::Vgpr(6),
          InstructionOperand::Vgpr(7), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(2)),
      DecodedInstruction::SixOperand(
          "DS_WRXCHG2_RTN_B64", InstructionOperand::Vgpr(50),
          InstructionOperand::Vgpr(8), InstructionOperand::Vgpr(12),
          InstructionOperand::Vgpr(14), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(2)),
      DecodedInstruction::SixOperand(
          "DS_WRXCHG2ST64_RTN_B64", InstructionOperand::Vgpr(54),
          InstructionOperand::Vgpr(16), InstructionOperand::Vgpr(18),
          InstructionOperand::Vgpr(20), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(2)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_pair_return_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[0][0] = 64u;
    state.vgprs[0][1] = 96u;
    state.vgprs[0][3] = 128u;
    state.vgprs[2][0] = 1001u;
    state.vgprs[2][1] = 2001u;
    state.vgprs[2][3] = 3001u;
    state.vgprs[3][0] = 1002u;
    state.vgprs[3][1] = 2002u;
    state.vgprs[3][3] = 3002u;

    state.vgprs[4][0] = 1536u;
    state.vgprs[4][1] = 2560u;
    state.vgprs[4][3] = 3584u;
    state.vgprs[6][0] = 4101u;
    state.vgprs[6][1] = 4201u;
    state.vgprs[6][3] = 4301u;
    state.vgprs[7][0] = 4102u;
    state.vgprs[7][1] = 4202u;
    state.vgprs[7][3] = 4302u;

    state.vgprs[8][0] = 4608u;
    state.vgprs[8][1] = 4640u;
    state.vgprs[8][3] = 4672u;
    state.vgprs[16][0] = 5120u;
    state.vgprs[16][1] = 6656u;
    state.vgprs[16][3] = 8192u;

    auto set_lane_u64 = [&](std::uint16_t reg, std::size_t lane,
                            std::uint64_t value) {
      std::uint32_t low = 0;
      std::uint32_t high = 0;
      SplitU64(value, &low, &high);
      state.vgprs[reg][lane] = low;
      state.vgprs[reg + 1][lane] = high;
    };
    set_lane_u64(12, 0u, 0x1111222233334444ULL);
    set_lane_u64(12, 1u, 0x9999aaaabbbbccccULL);
    set_lane_u64(12, 3u, 0x0123456789abcdefULL);
    set_lane_u64(14, 0u, 0x5555666677778888ULL);
    set_lane_u64(14, 1u, 0xddddeeeeffff0001ULL);
    set_lane_u64(14, 3u, 0xfedcba9876543210ULL);
    set_lane_u64(18, 0u, 0xaaaaaaaa00000001ULL);
    set_lane_u64(18, 1u, 0xbbbbbbbb00000002ULL);
    set_lane_u64(18, 3u, 0xcccccccc00000003ULL);
    set_lane_u64(20, 0u, 0xdddddddd00000004ULL);
    set_lane_u64(20, 1u, 0xeeeeeeee00000005ULL);
    set_lane_u64(20, 3u, 0xffffffff00000006ULL);

    auto write_lds_u32 = [&](std::uint64_t address, std::uint32_t value) {
      std::memcpy(state.lds_bytes.data() + address, &value, sizeof(value));
    };
    auto write_lds_u64 = [&](std::uint64_t address, std::uint64_t value) {
      std::memcpy(state.lds_bytes.data() + address, &value, sizeof(value));
    };
    write_lds_u32(68u, 101u);
    write_lds_u32(76u, 102u);
    write_lds_u32(100u, 201u);
    write_lds_u32(108u, 202u);
    write_lds_u32(132u, 301u);
    write_lds_u32(140u, 302u);
    write_lds_u32(1792u, 1101u);
    write_lds_u32(2048u, 1102u);
    write_lds_u32(2816u, 1201u);
    write_lds_u32(3072u, 1202u);
    write_lds_u32(3840u, 1301u);
    write_lds_u32(4096u, 1302u);
    write_lds_u64(4616u, 0x1111111122222222ULL);
    write_lds_u64(4624u, 0x3333333344444444ULL);
    write_lds_u64(4648u, 0x5555555566666666ULL);
    write_lds_u64(4656u, 0x7777777788888888ULL);
    write_lds_u64(4680u, 0x99999999aaaabbbbULL);
    write_lds_u64(4688u, 0xccccddddeeeeffffULL);
    write_lds_u64(5632u, 0x1010101020202020ULL);
    write_lds_u64(6144u, 0x3030303040404040ULL);
    write_lds_u64(7168u, 0x5050505060606060ULL);
    write_lds_u64(7680u, 0x7070707080808080ULL);
    write_lds_u64(8704u, 0x90909090a0a0a0a0ULL);
    write_lds_u64(9216u, 0xb0b0b0b0c0c0c0c0ULL);

    for (std::uint16_t vgpr = 40; vgpr <= 57; ++vgpr) {
      state.vgprs[vgpr][2] = 0x80000000u + vgpr;
    }
    return state;
  };
  auto validate_ds_pair_return_state = [&](const WaveExecutionState& state,
                                           const char* mode) {
    if (!Expect(state.halted, "expected ds pair-return program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }

    static constexpr std::array<std::size_t, 3> lanes = {0u, 1u, 3u};
    static constexpr std::array<std::array<std::uint32_t, 2>, 3> b32_returns = {{
        {{101u, 102u}},
        {{201u, 202u}},
        {{301u, 302u}},
    }};
    static constexpr std::array<std::array<std::uint32_t, 2>, 3> b32_updates = {{
        {{1001u, 1002u}},
        {{2001u, 2002u}},
        {{3001u, 3002u}},
    }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3> b32_addresses = {{
        {{68u, 76u}},
        {{100u, 108u}},
        {{132u, 140u}},
    }};
    static constexpr std::array<std::array<std::uint32_t, 2>, 3>
        b32_st64_returns = {{
            {{1101u, 1102u}},
            {{1201u, 1202u}},
            {{1301u, 1302u}},
        }};
    static constexpr std::array<std::array<std::uint32_t, 2>, 3>
        b32_st64_updates = {{
            {{4101u, 4102u}},
            {{4201u, 4202u}},
            {{4301u, 4302u}},
        }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3>
        b32_st64_addresses = {{
            {{1792u, 2048u}},
            {{2816u, 3072u}},
            {{3840u, 4096u}},
        }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3> b64_returns = {{
        {{0x1111111122222222ULL, 0x3333333344444444ULL}},
        {{0x5555555566666666ULL, 0x7777777788888888ULL}},
        {{0x99999999aaaabbbbULL, 0xccccddddeeeeffffULL}},
    }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3> b64_updates = {{
        {{0x1111222233334444ULL, 0x5555666677778888ULL}},
        {{0x9999aaaabbbbccccULL, 0xddddeeeeffff0001ULL}},
        {{0x0123456789abcdefULL, 0xfedcba9876543210ULL}},
    }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3>
        b64_addresses = {{
            {{4616u, 4624u}},
            {{4648u, 4656u}},
            {{4680u, 4688u}},
        }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3>
        b64_st64_returns = {{
            {{0x1010101020202020ULL, 0x3030303040404040ULL}},
            {{0x5050505060606060ULL, 0x7070707080808080ULL}},
            {{0x90909090a0a0a0a0ULL, 0xb0b0b0b0c0c0c0c0ULL}},
        }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3>
        b64_st64_updates = {{
            {{0xaaaaaaaa00000001ULL, 0xdddddddd00000004ULL}},
            {{0xbbbbbbbb00000002ULL, 0xeeeeeeee00000005ULL}},
            {{0xcccccccc00000003ULL, 0xffffffff00000006ULL}},
        }};
    static constexpr std::array<std::array<std::uint64_t, 2>, 3>
        b64_st64_addresses = {{
            {{5632u, 6144u}},
            {{7168u, 7680u}},
            {{8704u, 9216u}},
        }};

    for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
      const std::size_t lane = lanes[lane_index];
      if (!Expect(state.vgprs[40][lane] == b32_returns[lane_index][0],
                  "expected ds wrxchg2 rtn b32 low return") ||
          !Expect(state.vgprs[41][lane] == b32_returns[lane_index][1],
                  "expected ds wrxchg2 rtn b32 high return") ||
          !Expect(state.vgprs[42][lane] == b32_st64_returns[lane_index][0],
                  "expected ds wrxchg2st64 rtn b32 low return") ||
          !Expect(state.vgprs[43][lane] == b32_st64_returns[lane_index][1],
                  "expected ds wrxchg2st64 rtn b32 high return") ||
          !Expect(ComposeU64(state.vgprs[50][lane], state.vgprs[51][lane]) ==
                      b64_returns[lane_index][0],
                  "expected ds wrxchg2 rtn b64 low return") ||
          !Expect(ComposeU64(state.vgprs[52][lane], state.vgprs[53][lane]) ==
                      b64_returns[lane_index][1],
                  "expected ds wrxchg2 rtn b64 high return") ||
          !Expect(ComposeU64(state.vgprs[54][lane], state.vgprs[55][lane]) ==
                      b64_st64_returns[lane_index][0],
                  "expected ds wrxchg2st64 rtn b64 low return") ||
          !Expect(ComposeU64(state.vgprs[56][lane], state.vgprs[57][lane]) ==
                      b64_st64_returns[lane_index][1],
                  "expected ds wrxchg2st64 rtn b64 high return")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }

      std::uint32_t lds_u32 = 0;
      std::memcpy(&lds_u32, state.lds_bytes.data() + b32_addresses[lane_index][0],
                  sizeof(lds_u32));
      if (!Expect(lds_u32 == b32_updates[lane_index][0],
                  "expected ds wrxchg2 rtn b32 low lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u32, state.lds_bytes.data() + b32_addresses[lane_index][1],
                  sizeof(lds_u32));
      if (!Expect(lds_u32 == b32_updates[lane_index][1],
                  "expected ds wrxchg2 rtn b32 high lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u32,
                  state.lds_bytes.data() + b32_st64_addresses[lane_index][0],
                  sizeof(lds_u32));
      if (!Expect(lds_u32 == b32_st64_updates[lane_index][0],
                  "expected ds wrxchg2st64 rtn b32 low lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u32,
                  state.lds_bytes.data() + b32_st64_addresses[lane_index][1],
                  sizeof(lds_u32));
      if (!Expect(lds_u32 == b32_st64_updates[lane_index][1],
                  "expected ds wrxchg2st64 rtn b32 high lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }

      std::uint64_t lds_u64 = 0;
      std::memcpy(&lds_u64, state.lds_bytes.data() + b64_addresses[lane_index][0],
                  sizeof(lds_u64));
      if (!Expect(lds_u64 == b64_updates[lane_index][0],
                  "expected ds wrxchg2 rtn b64 low lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u64, state.lds_bytes.data() + b64_addresses[lane_index][1],
                  sizeof(lds_u64));
      if (!Expect(lds_u64 == b64_updates[lane_index][1],
                  "expected ds wrxchg2 rtn b64 high lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u64,
                  state.lds_bytes.data() + b64_st64_addresses[lane_index][0],
                  sizeof(lds_u64));
      if (!Expect(lds_u64 == b64_st64_updates[lane_index][0],
                  "expected ds wrxchg2st64 rtn b64 low lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
      std::memcpy(&lds_u64,
                  state.lds_bytes.data() + b64_st64_addresses[lane_index][1],
                  sizeof(lds_u64));
      if (!Expect(lds_u64 == b64_st64_updates[lane_index][1],
                  "expected ds wrxchg2st64 rtn b64 high lds")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
    }

    for (std::uint16_t vgpr = 40; vgpr <= 57; ++vgpr) {
      const std::uint32_t expected = 0x80000000u + vgpr;
      if (!Expect(state.vgprs[vgpr][2] == expected,
                  "expected inactive ds pair-return destination preservation")) {
        std::cerr << mode << " vgpr=" << vgpr << '\n';
        return false;
      }
    }
    return true;
  };

  WaveExecutionState decoded_ds_pair_return_state = make_ds_pair_return_state();
  if (!Expect(interpreter.ExecuteProgram(ds_pair_return_program,
                                         &decoded_ds_pair_return_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_pair_return_state(decoded_ds_pair_return_state, "decoded")) {
    return false;
  }

  std::vector<CompiledInstruction> compiled_ds_pair_return_program;
  if (!Expect(interpreter.CompileProgram(ds_pair_return_program,
                                         &compiled_ds_pair_return_program,
                                         &error_message),
              error_message.c_str())) {
    return false;
  }
  WaveExecutionState compiled_ds_pair_return_state = make_ds_pair_return_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_pair_return_program,
                                         &compiled_ds_pair_return_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_pair_return_state(compiled_ds_pair_return_state,
                                     "compiled")) {
    return false;
  }

  return true;
}

bool RunDsReturnTests(const mirage::sim::isa::Gfx950Interpreter& interpreter) {
  using namespace mirage::sim::isa;

  struct DsReturnCase {
    std::string_view opcode;
    std::array<std::uint32_t, 3> initial_values;
    std::array<std::uint32_t, 3> data_values;
    std::array<std::uint32_t, 3> expected_final_values;
  };
  const auto run_ds_return_case =
      [&](const DsReturnCase& test_case, bool use_compiled_program) {
        const std::vector<DecodedInstruction> program = {
            DecodedInstruction::ThreeOperand(
                "DS_WRITE_B32", InstructionOperand::Vgpr(0),
                InstructionOperand::Vgpr(1), InstructionOperand::Imm32(0)),
            DecodedInstruction::FourOperand(
                test_case.opcode, InstructionOperand::Vgpr(2),
                InstructionOperand::Vgpr(0), InstructionOperand::Vgpr(3),
                InstructionOperand::Imm32(0)),
            DecodedInstruction::ThreeOperand(
                "DS_READ_B32", InstructionOperand::Vgpr(4),
                InstructionOperand::Vgpr(0), InstructionOperand::Imm32(0)),
            DecodedInstruction::Nullary("S_ENDPGM"),
        };

        WaveExecutionState state;
        state.exec_mask = 0b1011ULL;
        state.vgprs[0][0] = 0u;
        state.vgprs[0][1] = 4u;
        state.vgprs[0][3] = 8u;
        state.vgprs[1][0] = test_case.initial_values[0];
        state.vgprs[1][1] = test_case.initial_values[1];
        state.vgprs[1][3] = test_case.initial_values[2];
        state.vgprs[3][0] = test_case.data_values[0];
        state.vgprs[3][1] = test_case.data_values[1];
        state.vgprs[3][3] = test_case.data_values[2];
        state.vgprs[2][2] = 0xdeadbeefu;
        state.vgprs[4][2] = 0xcafebabeu;

        std::string case_error;
        if (use_compiled_program) {
          std::vector<CompiledInstruction> compiled_program;
          if (!interpreter.CompileProgram(program, &compiled_program, &case_error)) {
            std::cerr << test_case.opcode << " compile: " << case_error << '\n';
            return false;
          }
          if (!interpreter.ExecuteProgram(compiled_program, &state, &case_error)) {
            std::cerr << test_case.opcode << " compiled execute: " << case_error
                      << '\n';
            return false;
          }
        } else if (!interpreter.ExecuteProgram(program, &state, &case_error)) {
          std::cerr << test_case.opcode << " decoded execute: " << case_error
                    << '\n';
          return false;
        }

        const char* mode = use_compiled_program ? "compiled" : "decoded";
        if (!Expect(state.halted, "expected ds return test to halt")) {
          std::cerr << test_case.opcode << ' ' << mode << '\n';
          return false;
        }

        static constexpr std::array<std::size_t, 3> kObservedLanes = {0u, 1u, 3u};
        static constexpr std::array<std::size_t, 3> kObservedAddresses = {0u, 4u,
                                                                           8u};
        for (std::size_t index = 0; index < kObservedLanes.size(); ++index) {
          const std::size_t lane = kObservedLanes[index];
          const std::uint32_t expected_old = test_case.initial_values[index];
          const std::uint32_t expected_final =
              test_case.expected_final_values[index];
          if (!Expect(state.vgprs[2][lane] == expected_old,
                      "expected ds return old value") ||
              !Expect(state.vgprs[4][lane] == expected_final,
                      "expected ds return final value")) {
            std::cerr << test_case.opcode << ' ' << mode << " lane=" << lane
                      << '\n';
            return false;
          }

          std::uint32_t lds_value = 0;
          std::memcpy(&lds_value, state.lds_bytes.data() + kObservedAddresses[index],
                      sizeof(lds_value));
          if (!Expect(lds_value == expected_final,
                      "expected ds return lds final value")) {
            std::cerr << test_case.opcode << ' ' << mode << " lane=" << lane
                      << '\n';
            return false;
          }
        }

        if (!Expect(state.vgprs[2][2] == 0xdeadbeefu,
                    "expected inactive ds return destination preservation") ||
            !Expect(state.vgprs[4][2] == 0xcafebabeu,
                    "expected inactive ds return read preservation")) {
          std::cerr << test_case.opcode << ' ' << mode << '\n';
          return false;
        }
        return true;
      };
  const std::array<DsReturnCase, 18> kDsReturnCases = {{
      {"DS_ADD_RTN_U32", {10u, 20u, 40u}, {1u, 2u, 4u}, {11u, 22u, 44u}},
      {"DS_SUB_RTN_U32", {10u, 20u, 40u}, {1u, 2u, 4u}, {9u, 18u, 36u}},
      {"DS_RSUB_RTN_U32", {10u, 20u, 40u}, {15u, 25u, 45u}, {5u, 5u, 5u}},
      {"DS_INC_RTN_U32", {5u, 9u, 0u}, {7u, 9u, 4u}, {6u, 0u, 1u}},
      {"DS_DEC_RTN_U32", {5u, 0u, 3u}, {7u, 9u, 3u}, {4u, 9u, 2u}},
      {"DS_MIN_RTN_I32",
       {0xfffffff0u, 5u, 0xffffff00u},
       {0xfffffff8u, 0xffffffffu, 0xffffff80u},
       {0xfffffff0u, 0xffffffffu, 0xffffff00u}},
      {"DS_MAX_RTN_I32",
       {0xfffffff0u, 5u, 0xffffff00u},
       {0xfffffff8u, 0xffffffffu, 0xffffff80u},
       {0xfffffff8u, 5u, 0xffffff80u}},
      {"DS_MIN_RTN_U32", {3u, 7u, 1u}, {5u, 9u, 2u}, {3u, 7u, 1u}},
      {"DS_MAX_RTN_U32", {3u, 7u, 1u}, {5u, 9u, 2u}, {5u, 9u, 2u}},
      {"DS_AND_RTN_B32",
       {0x0f0f0f0fu, 0xff00ff00u, 0xaaaaaaaau},
       {0xf0f0ffffu, 0x00ff00ffu, 0x0f0f0f0fu},
       {0x00000f0fu, 0x00000000u, 0x0a0a0a0au}},
      {"DS_OR_RTN_B32",
       {0x11003300u, 0x0000ff00u, 0xaaaa0000u},
       {0x0000cc11u, 0x12340000u, 0x00005555u},
       {0x1100ff11u, 0x1234ff00u, 0xaaaa5555u}},
      {"DS_XOR_RTN_B32",
       {0x12345678u, 0xffffffffu, 0x0f0f0f0fu},
       {0x00ff00ffu, 0x0f0f0f0fu, 0xffffffffu},
       {0x12cb5687u, 0xf0f0f0f0u, 0xf0f0f0f0u}},
      {"DS_WRXCHG_RTN_B32",
       {0x12345678u, 0xffffffffu, 0x0f0f0f0fu},
       {0x00ff00ffu, 0x0f0f0f0fu, 0xffffffffu},
       {0x00ff00ffu, 0x0f0f0f0fu, 0xffffffffu}},
      {"DS_ADD_RTN_F32",
       {FloatBits(1.5f), FloatBits(-2.0f), FloatBits(10.0f)},
       {FloatBits(2.25f), FloatBits(0.5f), FloatBits(-5.0f)},
       {FloatBits(3.75f), FloatBits(-1.5f), FloatBits(5.0f)}},
      {"DS_MIN_RTN_F32",
       {FloatBits(4.0f), FloatBits(-2.0f), FloatBits(1.5f)},
       {FloatBits(3.0f), FloatBits(-3.5f), FloatBits(2.0f)},
       {FloatBits(3.0f), FloatBits(-3.5f), FloatBits(1.5f)}},
      {"DS_MAX_RTN_F32",
       {FloatBits(4.0f), FloatBits(-2.0f), FloatBits(1.5f)},
       {FloatBits(3.0f), FloatBits(-3.5f), FloatBits(2.0f)},
       {FloatBits(4.0f), FloatBits(-2.0f), FloatBits(2.0f)}},
      {"DS_PK_ADD_RTN_F16",
       {0x40003c00u, 0x0000bc00u, 0x4400c000u},
       {0x3c004000u, 0x3c003800u, 0x40003c00u},
       {0x42004200u, 0x3c00b800u, 0x4600bc00u}},
      {"DS_PK_ADD_RTN_BF16",
       {0x40003f80u, 0x0000bf80u, 0x4080c000u},
       {0x3f804000u, 0x3f803f00u, 0x40003f80u},
       {0x40404040u, 0x3f80bf00u, 0x40c0bf80u}},
  }};
  for (const DsReturnCase& test_case : kDsReturnCases) {
    if (!run_ds_return_case(test_case, false) ||
        !run_ds_return_case(test_case, true)) {
      return false;
    }
  }
  return true;
}

bool RunDsDualDataTests(
    const mirage::sim::isa::Gfx950Interpreter& interpreter) {
  using namespace mirage::sim::isa;

  struct DsDualDataCase {
    std::string_view opcode;
    bool has_return;
    std::array<std::uint32_t, 3> initial_values;
    std::array<std::uint32_t, 3> data0_values;
    std::array<std::uint32_t, 3> data1_values;
    std::array<std::uint32_t, 3> expected_final_values;
  };
  const auto run_ds_dual_data_case =
      [&](const DsDualDataCase& test_case, bool use_compiled_program) {
        std::vector<DecodedInstruction> program = {
            DecodedInstruction::ThreeOperand(
                "DS_WRITE_B32", InstructionOperand::Vgpr(0),
                InstructionOperand::Vgpr(1), InstructionOperand::Imm32(0)),
        };
        if (test_case.has_return) {
          program.push_back(DecodedInstruction::FiveOperand(
              test_case.opcode, InstructionOperand::Vgpr(2),
              InstructionOperand::Vgpr(0), InstructionOperand::Vgpr(3),
              InstructionOperand::Vgpr(4), InstructionOperand::Imm32(0)));
        } else {
          program.push_back(DecodedInstruction::FourOperand(
              test_case.opcode, InstructionOperand::Vgpr(0),
              InstructionOperand::Vgpr(3), InstructionOperand::Vgpr(4),
              InstructionOperand::Imm32(0)));
        }
        program.push_back(DecodedInstruction::ThreeOperand(
            "DS_READ_B32", InstructionOperand::Vgpr(5), InstructionOperand::Vgpr(0),
            InstructionOperand::Imm32(0)));
        program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

        WaveExecutionState state;
        state.exec_mask = 0b1011ULL;
        state.vgprs[0][0] = 0u;
        state.vgprs[0][1] = 4u;
        state.vgprs[0][3] = 8u;
        state.vgprs[1][0] = test_case.initial_values[0];
        state.vgprs[1][1] = test_case.initial_values[1];
        state.vgprs[1][3] = test_case.initial_values[2];
        state.vgprs[3][0] = test_case.data0_values[0];
        state.vgprs[3][1] = test_case.data0_values[1];
        state.vgprs[3][3] = test_case.data0_values[2];
        state.vgprs[4][0] = test_case.data1_values[0];
        state.vgprs[4][1] = test_case.data1_values[1];
        state.vgprs[4][3] = test_case.data1_values[2];
        state.vgprs[2][2] = 0xdeadbeefu;
        state.vgprs[5][2] = 0xcafebabeu;

        std::string case_error;
        if (use_compiled_program) {
          std::vector<CompiledInstruction> compiled_program;
          if (!interpreter.CompileProgram(program, &compiled_program, &case_error)) {
            std::cerr << test_case.opcode << " compile: " << case_error << '\n';
            return false;
          }
          if (!interpreter.ExecuteProgram(compiled_program, &state, &case_error)) {
            std::cerr << test_case.opcode << " compiled execute: " << case_error
                      << '\n';
            return false;
          }
        } else if (!interpreter.ExecuteProgram(program, &state, &case_error)) {
          std::cerr << test_case.opcode << " decoded execute: " << case_error
                    << '\n';
          return false;
        }

        const char* mode = use_compiled_program ? "compiled" : "decoded";
        if (!Expect(state.halted, "expected ds dual-data test to halt")) {
          std::cerr << test_case.opcode << ' ' << mode << '\n';
          return false;
        }

        static constexpr std::array<std::size_t, 3> kObservedLanes = {0u, 1u, 3u};
        static constexpr std::array<std::size_t, 3> kObservedAddresses = {0u, 4u,
                                                                           8u};
        for (std::size_t index = 0; index < kObservedLanes.size(); ++index) {
          const std::size_t lane = kObservedLanes[index];
          const std::uint32_t expected_final =
              test_case.expected_final_values[index];
          if (test_case.has_return &&
              !Expect(state.vgprs[2][lane] == test_case.initial_values[index],
                      "expected ds dual-data old value")) {
            std::cerr << test_case.opcode << ' ' << mode << " lane=" << lane
                      << '\n';
            return false;
          }
          if (!Expect(state.vgprs[5][lane] == expected_final,
                      "expected ds dual-data final value")) {
            std::cerr << test_case.opcode << ' ' << mode << " lane=" << lane
                      << '\n';
            return false;
          }

          std::uint32_t lds_value = 0;
          std::memcpy(&lds_value, state.lds_bytes.data() + kObservedAddresses[index],
                      sizeof(lds_value));
          if (!Expect(lds_value == expected_final,
                      "expected ds dual-data lds final value")) {
            std::cerr << test_case.opcode << ' ' << mode << " lane=" << lane
                      << '\n';
            return false;
          }
        }

        if (test_case.has_return &&
            !Expect(state.vgprs[2][2] == 0xdeadbeefu,
                    "expected inactive ds dual-data return preservation")) {
          std::cerr << test_case.opcode << ' ' << mode << '\n';
          return false;
        }
        if (!Expect(state.vgprs[5][2] == 0xcafebabeu,
                    "expected inactive ds dual-data read preservation")) {
          std::cerr << test_case.opcode << ' ' << mode << '\n';
          return false;
        }
        return true;
      };
  const std::array<DsDualDataCase, 7> kDsDualDataCases = {{
      {"DS_MSKOR_B32",
       false,
       {0xaaaa5555u, 0xf0f0f0f0u, 0x12345678u},
       {0x00ff00ffu, 0x0f0f0000u, 0xffffffffu},
       {0x11002200u, 0x00001234u, 0x00000009u},
       {0xbb007700u, 0xf0f0f2f4u, 0x00000009u}},
      {"DS_CMPST_B32",
       false,
       {10u, 20u, 30u},
       {10u, 5u, 30u},
       {111u, 222u, 333u},
       {111u, 20u, 333u}},
      {"DS_CMPST_F32",
       false,
       {FloatBits(1.0f), FloatBits(2.5f), FloatBits(-0.0f)},
       {FloatBits(1.0f), FloatBits(3.0f), FloatBits(0.0f)},
       {FloatBits(4.0f), FloatBits(5.0f), FloatBits(6.0f)},
       {FloatBits(4.0f), FloatBits(2.5f), FloatBits(6.0f)}},
      {"DS_MSKOR_RTN_B32",
       true,
       {0xaaaa5555u, 0xf0f0f0f0u, 0x12345678u},
       {0x00ff00ffu, 0x0f0f0000u, 0xffffffffu},
       {0x11002200u, 0x00001234u, 0x00000009u},
       {0xbb007700u, 0xf0f0f2f4u, 0x00000009u}},
      {"DS_CMPST_RTN_B32",
       true,
       {10u, 20u, 30u},
       {10u, 5u, 30u},
       {111u, 222u, 333u},
       {111u, 20u, 333u}},
      {"DS_CMPST_RTN_F32",
       true,
       {FloatBits(1.0f), FloatBits(2.5f), FloatBits(-0.0f)},
       {FloatBits(1.0f), FloatBits(3.0f), FloatBits(0.0f)},
       {FloatBits(4.0f), FloatBits(5.0f), FloatBits(6.0f)},
       {FloatBits(4.0f), FloatBits(2.5f), FloatBits(6.0f)}},
      {"DS_WRAP_RTN_B32",
       true,
       {0u, 5u, 12u},
       {111u, 222u, 333u},
       {7u, 8u, 10u},
       {111u, 4u, 333u}},
  }};
  for (const DsDualDataCase& test_case : kDsDualDataCases) {
    if (!run_ds_dual_data_case(test_case, false) ||
        !run_ds_dual_data_case(test_case, true)) {
      return false;
    }
  }
  return true;
}

bool RunDsWaveCounterTests(
    const mirage::sim::isa::Gfx950Interpreter& interpreter) {
  using namespace mirage::sim::isa;

  const std::vector<DecodedInstruction> ds_wave_counter_program = {
      DecodedInstruction::TwoOperand("DS_CONSUME", InstructionOperand::Vgpr(10),
                                     InstructionOperand::Imm32(0x120)),
      DecodedInstruction::TwoOperand("DS_APPEND", InstructionOperand::Vgpr(11),
                                     InstructionOperand::Imm32(0x450)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_wave_counter_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[10][2] = 0xdeadbeefu;
    state.vgprs[11][2] = 0xcafebabeu;
    const std::uint32_t consume_initial = 20u;
    const std::uint32_t append_initial = 50u;
    std::memcpy(state.lds_bytes.data() + 0x120u, &consume_initial,
                sizeof(consume_initial));
    std::memcpy(state.lds_bytes.data() + 0x450u, &append_initial,
                sizeof(append_initial));
    return state;
  };
  auto validate_ds_wave_counter_state = [&](const WaveExecutionState& state,
                                            const char* mode) {
    if (!Expect(state.halted, "expected ds wave-counter program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }

    static constexpr std::array<std::size_t, 3> kObservedLanes = {0u, 1u, 3u};
    for (std::size_t lane : kObservedLanes) {
      if (!Expect(state.vgprs[10][lane] == 20u,
                  "expected ds consume return value") ||
          !Expect(state.vgprs[11][lane] == 50u,
                  "expected ds append return value")) {
        std::cerr << mode << " lane=" << lane << '\n';
        return false;
      }
    }

    if (!Expect(state.vgprs[10][2] == 0xdeadbeefu,
                "expected inactive ds consume destination preservation") ||
        !Expect(state.vgprs[11][2] == 0xcafebabeu,
                "expected inactive ds append destination preservation")) {
      std::cerr << mode << '\n';
      return false;
    }

    std::uint32_t consume_value = 0;
    std::uint32_t append_value = 0;
    std::memcpy(&consume_value, state.lds_bytes.data() + 0x120u,
                sizeof(consume_value));
    std::memcpy(&append_value, state.lds_bytes.data() + 0x450u,
                sizeof(append_value));
    if (!Expect(consume_value == 17u, "expected ds consume final lds value") ||
        !Expect(append_value == 53u, "expected ds append final lds value")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  std::string error_message;
  WaveExecutionState decoded_ds_wave_counter_state =
      make_ds_wave_counter_state();
  if (!Expect(interpreter.ExecuteProgram(ds_wave_counter_program,
                                         &decoded_ds_wave_counter_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_wave_counter_state(decoded_ds_wave_counter_state,
                                      "decoded")) {
    return false;
  }

  std::vector<CompiledInstruction> compiled_ds_wave_counter_program;
  if (!Expect(interpreter.CompileProgram(ds_wave_counter_program,
                                         &compiled_ds_wave_counter_program,
                                         &error_message),
              error_message.c_str())) {
    return false;
  }
  WaveExecutionState compiled_ds_wave_counter_state =
      make_ds_wave_counter_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_wave_counter_program,
                                         &compiled_ds_wave_counter_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_wave_counter_state(compiled_ds_wave_counter_state,
                                      "compiled")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  Gfx950Interpreter interpreter;
  if (!Expect(interpreter.Supports("S_MOV_B32"), "expected S_MOV_B32 support") ||
      !Expect(interpreter.Supports("S_MOV_B64"),
              "expected S_MOV_B64 support") ||
      !Expect(interpreter.Supports("S_CMOV_B32"),
              "expected S_CMOV_B32 support") ||
      !Expect(interpreter.Supports("S_CMOV_B64"),
              "expected S_CMOV_B64 support") ||
      !Expect(interpreter.Supports("S_CMOVK_I32"),
              "expected S_CMOVK_I32 support") ||
      !Expect(interpreter.Supports("S_MOVK_I32"),
              "expected S_MOVK_I32 support") ||
      !Expect(interpreter.Supports("S_NOT_B32"),
              "expected S_NOT_B32 support") ||
      !Expect(interpreter.Supports("S_NOT_B64"),
              "expected S_NOT_B64 support") ||
      !Expect(interpreter.Supports("S_ABS_I32"),
              "expected S_ABS_I32 support") ||
      !Expect(interpreter.Supports("S_BREV_B32"),
              "expected S_BREV_B32 support") ||
      !Expect(interpreter.Supports("S_BREV_B64"),
              "expected S_BREV_B64 support") ||
      !Expect(interpreter.Supports("S_BCNT0_I32_B32"),
              "expected S_BCNT0_I32_B32 support") ||
      !Expect(interpreter.Supports("S_BCNT0_I32_B64"),
              "expected S_BCNT0_I32_B64 support") ||
      !Expect(interpreter.Supports("S_BCNT1_I32_B32"),
              "expected S_BCNT1_I32_B32 support") ||
      !Expect(interpreter.Supports("S_BCNT1_I32_B64"),
              "expected S_BCNT1_I32_B64 support") ||
      !Expect(interpreter.Supports("S_FF0_I32_B32"),
              "expected S_FF0_I32_B32 support") ||
      !Expect(interpreter.Supports("S_FF0_I32_B64"),
              "expected S_FF0_I32_B64 support") ||
      !Expect(interpreter.Supports("S_FF1_I32_B32"),
              "expected S_FF1_I32_B32 support") ||
      !Expect(interpreter.Supports("S_FF1_I32_B64"),
              "expected S_FF1_I32_B64 support") ||
      !Expect(interpreter.Supports("S_FLBIT_I32_B32"),
              "expected S_FLBIT_I32_B32 support") ||
      !Expect(interpreter.Supports("S_FLBIT_I32_B64"),
              "expected S_FLBIT_I32_B64 support") ||
      !Expect(interpreter.Supports("S_FLBIT_I32"),
              "expected S_FLBIT_I32 support") ||
      !Expect(interpreter.Supports("S_FLBIT_I32_I64"),
              "expected S_FLBIT_I32_I64 support") ||
      !Expect(interpreter.Supports("S_BITREPLICATE_B64_B32"),
              "expected S_BITREPLICATE_B64_B32 support") ||
      !Expect(interpreter.Supports("S_QUADMASK_B32"),
              "expected S_QUADMASK_B32 support") ||
      !Expect(interpreter.Supports("S_QUADMASK_B64"),
              "expected S_QUADMASK_B64 support") ||
      !Expect(interpreter.Supports("S_SEXT_I32_I8"),
              "expected S_SEXT_I32_I8 support") ||
      !Expect(interpreter.Supports("S_SEXT_I32_I16"),
              "expected S_SEXT_I32_I16 support") ||
      !Expect(interpreter.Supports("S_BITSET0_B32"),
              "expected S_BITSET0_B32 support") ||
      !Expect(interpreter.Supports("S_BITSET0_B64"),
              "expected S_BITSET0_B64 support") ||
      !Expect(interpreter.Supports("S_BITSET1_B32"),
              "expected S_BITSET1_B32 support") ||
      !Expect(interpreter.Supports("S_BITSET1_B64"),
              "expected S_BITSET1_B64 support") ||
      !Expect(interpreter.Supports("S_ADDK_I32"),
              "expected S_ADDK_I32 support") ||
      !Expect(interpreter.Supports("S_ADD_I32"),
              "expected S_ADD_I32 support") ||
      !Expect(interpreter.Supports("S_ADDC_U32"),
              "expected S_ADDC_U32 support") ||
      !Expect(interpreter.Supports("S_SUB_I32"),
              "expected S_SUB_I32 support") ||
      !Expect(interpreter.Supports("S_MULK_I32"),
              "expected S_MULK_I32 support") ||
      !Expect(interpreter.Supports("S_MUL_I32"),
              "expected S_MUL_I32 support") ||
      !Expect(interpreter.Supports("S_MUL_HI_U32"),
              "expected S_MUL_HI_U32 support") ||
      !Expect(interpreter.Supports("S_MUL_HI_I32"),
              "expected S_MUL_HI_I32 support") ||
      !Expect(interpreter.Supports("S_SUBB_U32"),
              "expected S_SUBB_U32 support") ||
      !Expect(interpreter.Supports("S_MIN_I32"),
              "expected S_MIN_I32 support") ||
      !Expect(interpreter.Supports("S_MIN_U32"),
              "expected S_MIN_U32 support") ||
      !Expect(interpreter.Supports("S_MAX_I32"),
              "expected S_MAX_I32 support") ||
      !Expect(interpreter.Supports("S_MAX_U32"),
              "expected S_MAX_U32 support") ||
      !Expect(interpreter.Supports("S_LSHL1_ADD_U32"),
              "expected S_LSHL1_ADD_U32 support") ||
      !Expect(interpreter.Supports("S_LSHL2_ADD_U32"),
              "expected S_LSHL2_ADD_U32 support") ||
      !Expect(interpreter.Supports("S_LSHL3_ADD_U32"),
              "expected S_LSHL3_ADD_U32 support") ||
      !Expect(interpreter.Supports("S_LSHL4_ADD_U32"),
              "expected S_LSHL4_ADD_U32 support") ||
      !Expect(interpreter.Supports("S_LSHL_B32"),
              "expected S_LSHL_B32 support") ||
      !Expect(interpreter.Supports("S_LSHL_B64"),
              "expected S_LSHL_B64 support") ||
      !Expect(interpreter.Supports("S_LSHR_B32"),
              "expected S_LSHR_B32 support") ||
      !Expect(interpreter.Supports("S_LSHR_B64"),
              "expected S_LSHR_B64 support") ||
      !Expect(interpreter.Supports("S_ASHR_I32"),
              "expected S_ASHR_I32 support") ||
      !Expect(interpreter.Supports("S_ASHR_I64"),
              "expected S_ASHR_I64 support") ||
      !Expect(interpreter.Supports("S_BFM_B32"),
              "expected S_BFM_B32 support") ||
      !Expect(interpreter.Supports("S_BFM_B64"),
              "expected S_BFM_B64 support") ||
      !Expect(interpreter.Supports("S_PACK_LL_B32_B16"),
              "expected S_PACK_LL_B32_B16 support") ||
      !Expect(interpreter.Supports("S_PACK_LH_B32_B16"),
              "expected S_PACK_LH_B32_B16 support") ||
      !Expect(interpreter.Supports("S_PACK_HH_B32_B16"),
              "expected S_PACK_HH_B32_B16 support") ||
      !Expect(interpreter.Supports("S_CSELECT_B32"),
              "expected S_CSELECT_B32 support") ||
      !Expect(interpreter.Supports("S_CSELECT_B64"),
              "expected S_CSELECT_B64 support") ||
      !Expect(interpreter.Supports("S_ABSDIFF_I32"),
              "expected S_ABSDIFF_I32 support") ||
      !Expect(interpreter.Supports("S_BFE_U32"),
              "expected S_BFE_U32 support") ||
      !Expect(interpreter.Supports("S_BFE_I32"),
              "expected S_BFE_I32 support") ||
      !Expect(interpreter.Supports("S_BFE_U64"),
              "expected S_BFE_U64 support") ||
      !Expect(interpreter.Supports("S_BFE_I64"),
              "expected S_BFE_I64 support") ||
      !Expect(interpreter.Supports("S_ANDN2_B32"),
              "expected S_ANDN2_B32 support") ||
      !Expect(interpreter.Supports("S_NAND_B32"),
              "expected S_NAND_B32 support") ||
      !Expect(interpreter.Supports("S_ORN2_B32"),
              "expected S_ORN2_B32 support") ||
      !Expect(interpreter.Supports("S_NOR_B32"),
              "expected S_NOR_B32 support") ||
      !Expect(interpreter.Supports("S_XNOR_B32"),
              "expected S_XNOR_B32 support") ||
      !Expect(interpreter.Supports("S_AND_B64"),
              "expected S_AND_B64 support") ||
      !Expect(interpreter.Supports("S_ANDN2_B64"),
              "expected S_ANDN2_B64 support") ||
      !Expect(interpreter.Supports("S_NAND_B64"),
              "expected S_NAND_B64 support") ||
      !Expect(interpreter.Supports("S_OR_B64"),
              "expected S_OR_B64 support") ||
      !Expect(interpreter.Supports("S_ORN2_B64"),
              "expected S_ORN2_B64 support") ||
      !Expect(interpreter.Supports("S_NOR_B64"),
              "expected S_NOR_B64 support") ||
      !Expect(interpreter.Supports("S_XOR_B64"),
              "expected S_XOR_B64 support") ||
      !Expect(interpreter.Supports("S_XNOR_B64"),
              "expected S_XNOR_B64 support") ||
      !Expect(interpreter.Supports("V_ADD_U32"), "expected V_ADD_U32 support") ||
      !Expect(interpreter.Supports("V_ADD_F32"), "expected V_ADD_F32 support") ||
      !Expect(interpreter.Supports("V_SUB_F32"), "expected V_SUB_F32 support") ||
      !Expect(interpreter.Supports("V_MUL_F32"), "expected V_MUL_F32 support") ||
      !Expect(interpreter.Supports("V_MIN_F32"), "expected V_MIN_F32 support") ||
      !Expect(interpreter.Supports("V_MAX_F32"), "expected V_MAX_F32 support") ||
      !Expect(interpreter.Supports("V_ADD_F64"), "expected V_ADD_F64 support") ||
      !Expect(interpreter.Supports("V_MUL_F64"), "expected V_MUL_F64 support") ||
      !Expect(interpreter.Supports("V_FMA_F32"), "expected V_FMA_F32 support") ||
      !Expect(interpreter.Supports("V_FMA_F64"), "expected V_FMA_F64 support") ||
      !Expect(interpreter.Supports("V_ADD_CO_U32"),
              "expected V_ADD_CO_U32 support") ||
      !Expect(interpreter.Supports("V_ADDC_CO_U32"),
              "expected V_ADDC_CO_U32 support") ||
      !Expect(interpreter.Supports("V_MOV_B64"),
              "expected V_MOV_B64 support") ||
      !Expect(interpreter.Supports("V_MUL_LO_U32"),
              "expected V_MUL_LO_U32 support") ||
      !Expect(interpreter.Supports("V_MUL_HI_U32"),
              "expected V_MUL_HI_U32 support") ||
      !Expect(interpreter.Supports("V_MUL_HI_I32"),
              "expected V_MUL_HI_I32 support") ||
      !Expect(interpreter.Supports("V_BCNT_U32_B32"),
              "expected V_BCNT_U32_B32 support") ||
      !Expect(interpreter.Supports("V_BFM_B32"),
              "expected V_BFM_B32 support") ||
      !Expect(interpreter.Supports("V_MBCNT_LO_U32_B32"),
              "expected V_MBCNT_LO_U32_B32 support") ||
      !Expect(interpreter.Supports("V_MBCNT_HI_U32_B32"),
              "expected V_MBCNT_HI_U32_B32 support") ||
      !Expect(interpreter.Supports("V_LSHLREV_B64"),
              "expected V_LSHLREV_B64 support") ||
      !Expect(interpreter.Supports("V_LSHRREV_B64"),
              "expected V_LSHRREV_B64 support") ||
      !Expect(interpreter.Supports("V_ASHRREV_I64"),
              "expected V_ASHRREV_I64 support") ||
      !Expect(interpreter.Supports("V_SUBREV_U32"),
              "expected V_SUBREV_U32 support") ||
      !Expect(interpreter.Supports("V_SUB_CO_U32"),
              "expected V_SUB_CO_U32 support") ||
      !Expect(interpreter.Supports("V_SUBREV_CO_U32"),
              "expected V_SUBREV_CO_U32 support") ||
      !Expect(interpreter.Supports("V_SUBB_CO_U32"),
              "expected V_SUBB_CO_U32 support") ||
      !Expect(interpreter.Supports("V_SUBBREV_CO_U32"),
              "expected V_SUBBREV_CO_U32 support") ||
      !Expect(interpreter.Supports("V_ADD3_U32"),
              "expected V_ADD3_U32 support") ||
      !Expect(interpreter.Supports("V_LSHL_ADD_U32"),
              "expected V_LSHL_ADD_U32 support") ||
      !Expect(interpreter.Supports("V_LSHL_ADD_U64"),
              "expected V_LSHL_ADD_U64 support") ||
      !Expect(interpreter.Supports("V_ADD_LSHL_U32"),
              "expected V_ADD_LSHL_U32 support") ||
      !Expect(interpreter.Supports("V_LSHL_OR_B32"),
              "expected V_LSHL_OR_B32 support") ||
      !Expect(interpreter.Supports("V_AND_OR_B32"),
              "expected V_AND_OR_B32 support") ||
      !Expect(interpreter.Supports("V_OR3_B32"),
              "expected V_OR3_B32 support") ||
      !Expect(interpreter.Supports("V_XAD_U32"),
              "expected V_XAD_U32 support") ||
      !Expect(interpreter.Supports("V_LERP_U8"),
              "expected V_LERP_U8 support") ||
      !Expect(interpreter.Supports("V_PERM_B32"),
              "expected V_PERM_B32 support") ||
      !Expect(interpreter.Supports("V_BFE_U32"),
              "expected V_BFE_U32 support") ||
      !Expect(interpreter.Supports("V_BFE_I32"),
              "expected V_BFE_I32 support") ||
      !Expect(interpreter.Supports("V_BFI_B32"),
              "expected V_BFI_B32 support") ||
      !Expect(interpreter.Supports("V_ALIGNBIT_B32"),
              "expected V_ALIGNBIT_B32 support") ||
      !Expect(interpreter.Supports("V_ALIGNBYTE_B32"),
              "expected V_ALIGNBYTE_B32 support") ||
      !Expect(interpreter.Supports("V_MIN3_I32"),
              "expected V_MIN3_I32 support") ||
      !Expect(interpreter.Supports("V_MIN3_U32"),
              "expected V_MIN3_U32 support") ||
      !Expect(interpreter.Supports("V_MAX3_I32"),
              "expected V_MAX3_I32 support") ||
      !Expect(interpreter.Supports("V_MAX3_U32"),
              "expected V_MAX3_U32 support") ||
      !Expect(interpreter.Supports("V_MED3_I32"),
              "expected V_MED3_I32 support") ||
      !Expect(interpreter.Supports("V_MED3_U32"),
              "expected V_MED3_U32 support") ||
      !Expect(interpreter.Supports("V_SAD_U8"),
              "expected V_SAD_U8 support") ||
      !Expect(interpreter.Supports("V_SAD_HI_U8"),
              "expected V_SAD_HI_U8 support") ||
      !Expect(interpreter.Supports("V_SAD_U16"),
              "expected V_SAD_U16 support") ||
      !Expect(interpreter.Supports("V_SAD_U32"),
              "expected V_SAD_U32 support") ||
      !Expect(interpreter.Supports("V_MAD_I32_I24"),
              "expected V_MAD_I32_I24 support") ||
      !Expect(interpreter.Supports("V_MAD_U32_U24"),
              "expected V_MAD_U32_U24 support") ||
      !Expect(interpreter.Supports("V_MAD_U64_U32"),
              "expected V_MAD_U64_U32 support") ||
      !Expect(interpreter.Supports("V_MAD_I64_I32"),
              "expected V_MAD_I64_I32 support") ||
      !Expect(interpreter.Supports("V_CMP_EQ_U32"),
              "expected V_CMP_EQ_U32 support") ||
      !Expect(interpreter.Supports("V_CMP_LT_I32"),
              "expected V_CMP_LT_I32 support") ||
      !Expect(interpreter.Supports("V_CMP_GE_U32"),
              "expected V_CMP_GE_U32 support") ||
      !Expect(interpreter.Supports("V_CNDMASK_B32"),
              "expected V_CNDMASK_B32 support") ||
      !Expect(interpreter.Supports("V_MIN_I32"),
              "expected V_MIN_I32 support") ||
      !Expect(interpreter.Supports("V_MAX_I32"),
              "expected V_MAX_I32 support") ||
      !Expect(interpreter.Supports("V_MIN_U32"),
              "expected V_MIN_U32 support") ||
      !Expect(interpreter.Supports("V_MAX_U32"),
              "expected V_MAX_U32 support") ||
      !Expect(interpreter.Supports("V_LSHRREV_B32"),
              "expected V_LSHRREV_B32 support") ||
      !Expect(interpreter.Supports("V_ASHRREV_I32"),
              "expected V_ASHRREV_I32 support") ||
      !Expect(interpreter.Supports("V_LSHLREV_B32"),
              "expected V_LSHLREV_B32 support") ||
      !Expect(interpreter.Supports("V_AND_B32"),
              "expected V_AND_B32 support") ||
      !Expect(interpreter.Supports("V_OR_B32"),
              "expected V_OR_B32 support") ||
      !Expect(interpreter.Supports("V_XOR_B32"),
              "expected V_XOR_B32 support") ||
      !Expect(interpreter.Supports("V_NOT_B32"),
              "expected V_NOT_B32 support") ||
      !Expect(interpreter.Supports("V_BFREV_B32"),
              "expected V_BFREV_B32 support") ||
      !Expect(interpreter.Supports("V_FFBH_U32"),
              "expected V_FFBH_U32 support") ||
      !Expect(interpreter.Supports("V_FFBL_B32"),
              "expected V_FFBL_B32 support") ||
      !Expect(interpreter.Supports("V_FFBH_I32"),
              "expected V_FFBH_I32 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_I32"),
              "expected V_CVT_F32_I32 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_U32"),
              "expected V_CVT_F32_U32 support") ||
      !Expect(interpreter.Supports("V_CVT_U32_F32"),
              "expected V_CVT_U32_F32 support") ||
      !Expect(interpreter.Supports("V_CVT_I32_F32"),
              "expected V_CVT_I32_F32 support") ||
      !Expect(interpreter.Supports("V_CVT_I32_F64"),
              "expected V_CVT_I32_F64 support") ||
      !Expect(interpreter.Supports("V_CVT_U32_F64"),
              "expected V_CVT_U32_F64 support") ||
      !Expect(interpreter.Supports("V_CVT_F16_F32"),
              "expected V_CVT_F16_F32 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_F16"),
              "expected V_CVT_F32_F16 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_F64"),
              "expected V_CVT_F32_F64 support") ||
      !Expect(interpreter.Supports("V_CVT_F64_F32"),
              "expected V_CVT_F64_F32 support") ||
      !Expect(interpreter.Supports("V_CVT_F64_I32"),
              "expected V_CVT_F64_I32 support") ||
      !Expect(interpreter.Supports("V_CVT_F64_U32"),
              "expected V_CVT_F64_U32 support") ||
      !Expect(interpreter.Supports("FLAT_LOAD_DWORD"),
              "expected FLAT_LOAD_DWORD support") ||
      !Expect(interpreter.Supports("FLAT_LOAD_SBYTE_D16_HI"),
              "expected FLAT_LOAD_SBYTE_D16_HI support") ||
      !Expect(interpreter.Supports("FLAT_STORE_SHORT_D16_HI"),
              "expected FLAT_STORE_SHORT_D16_HI support") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_DWORD"),
              "expected GLOBAL_STORE_DWORD support") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_SHORT_D16_HI"),
              "expected GLOBAL_LOAD_SHORT_D16_HI support") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_BYTE_D16_HI"),
              "expected GLOBAL_STORE_BYTE_D16_HI support") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_DWORDX2"),
              "expected GLOBAL_LOAD_DWORDX2 support") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_DWORDX2"),
              "expected GLOBAL_STORE_DWORDX2 support") ||
      !Expect(interpreter.Supports("GLOBAL_LOAD_DWORDX4"),
              "expected GLOBAL_LOAD_DWORDX4 support") ||
      !Expect(interpreter.Supports("GLOBAL_STORE_DWORDX4"),
              "expected GLOBAL_STORE_DWORDX4 support") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_ADD"),
              "expected GLOBAL_ATOMIC_ADD support") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_SWAP"),
              "expected GLOBAL_ATOMIC_SWAP support") ||
      !Expect(interpreter.Supports("GLOBAL_ATOMIC_CMPSWAP"),
              "expected GLOBAL_ATOMIC_CMPSWAP support") ||
      !Expect(interpreter.Supports("S_CMP_EQ_U32"),
              "expected S_CMP_EQ_U32 support") ||
      !Expect(interpreter.Supports("S_CMP_EQ_U64"),
              "expected S_CMP_EQ_U64 support") ||
      !Expect(interpreter.Supports("S_CMP_LG_U64"),
              "expected S_CMP_LG_U64 support") ||
      !Expect(interpreter.Supports("S_CMP_GT_I32"),
              "expected S_CMP_GT_I32 support") ||
      !Expect(interpreter.Supports("S_CMP_GT_U32"),
              "expected S_CMP_GT_U32 support") ||
      !Expect(interpreter.Supports("S_BITCMP0_B32"),
              "expected S_BITCMP0_B32 support") ||
      !Expect(interpreter.Supports("S_BITCMP1_B32"),
              "expected S_BITCMP1_B32 support") ||
      !Expect(interpreter.Supports("S_BITCMP0_B64"),
              "expected S_BITCMP0_B64 support") ||
      !Expect(interpreter.Supports("S_BITCMP1_B64"),
              "expected S_BITCMP1_B64 support") ||
      !Expect(interpreter.Supports("S_CMPK_LT_I32"),
              "expected S_CMPK_LT_I32 support") ||
      !Expect(interpreter.Supports("S_CBRANCH_SCC1"),
              "expected S_CBRANCH_SCC1 support") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCZ"),
              "expected S_CBRANCH_VCCZ support") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCNZ"),
              "expected S_CBRANCH_VCCNZ support") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECZ"),
              "expected S_CBRANCH_EXECZ support") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECNZ"),
              "expected S_CBRANCH_EXECNZ support") ||
      !Expect(interpreter.Supports("S_AND_SAVEEXEC_B64"),
              "expected S_AND_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ANDN1_SAVEEXEC_B64"),
              "expected S_ANDN1_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ANDN2_SAVEEXEC_B64"),
              "expected S_ANDN2_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_NAND_SAVEEXEC_B64"),
              "expected S_NAND_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_OR_SAVEEXEC_B64"),
              "expected S_OR_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ORN1_SAVEEXEC_B64"),
              "expected S_ORN1_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ORN2_SAVEEXEC_B64"),
              "expected S_ORN2_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_NOR_SAVEEXEC_B64"),
              "expected S_NOR_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_XOR_SAVEEXEC_B64"),
              "expected S_XOR_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_XNOR_SAVEEXEC_B64"),
              "expected S_XNOR_SAVEEXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ANDN1_WREXEC_B64"),
              "expected S_ANDN1_WREXEC_B64 support") ||
      !Expect(interpreter.Supports("S_ANDN2_WREXEC_B64"),
              "expected S_ANDN2_WREXEC_B64 support") ||
      !Expect(interpreter.Supports("S_BARRIER"),
              "expected S_BARRIER support") ||
      !Expect(interpreter.Supports("V_NOP"),
              "expected V_NOP support") ||
      !Expect(interpreter.Supports("V_READFIRSTLANE_B32"),
              "expected V_READFIRSTLANE_B32 support") ||
      !Expect(interpreter.Supports("V_READLANE_B32"),
              "expected V_READLANE_B32 support") ||
      !Expect(interpreter.Supports("V_WRITELANE_B32"),
              "expected V_WRITELANE_B32 support") ||
      !Expect(interpreter.Supports("V_CVT_F16_U16"),
              "expected V_CVT_F16_U16 support") ||
      !Expect(interpreter.Supports("V_CVT_F16_I16"),
              "expected V_CVT_F16_I16 support") ||
      !Expect(interpreter.Supports("V_CVT_U16_F16"),
              "expected V_CVT_U16_F16 support") ||
      !Expect(interpreter.Supports("V_CVT_I16_F16"),
              "expected V_CVT_I16_F16 support") ||
      !Expect(interpreter.Supports("V_SAT_PK_U8_I16"),
              "expected V_SAT_PK_U8_I16 support") ||
      !Expect(interpreter.Supports("V_EXP_LEGACY_F32"),
              "expected V_EXP_LEGACY_F32 support") ||
      !Expect(interpreter.Supports("V_LOG_LEGACY_F32"),
              "expected V_LOG_LEGACY_F32 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE0"),
              "expected V_CVT_F32_UBYTE0 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE1"),
              "expected V_CVT_F32_UBYTE1 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE2"),
              "expected V_CVT_F32_UBYTE2 support") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE3"),
              "expected V_CVT_F32_UBYTE3 support") ||
      !Expect(interpreter.Supports("DS_NOP"),
              "expected DS_NOP support") ||
      !Expect(interpreter.Supports("DS_ADD_U32"),
              "expected DS_ADD_U32 support") ||
      !Expect(interpreter.Supports("DS_READ_B32"),
              "expected DS_READ_B32 support") ||
      !Expect(interpreter.Supports("DS_WRITE_B32"),
              "expected DS_WRITE_B32 support") ||
      !Expect(interpreter.Supports("S_ICACHE_INV"),
              "expected S_ICACHE_INV support") ||
      !Expect(interpreter.Supports("BUFFER_WBL2"),
              "expected BUFFER_WBL2 support") ||
      !Expect(interpreter.Supports("BUFFER_INV"),
              "expected BUFFER_INV support") ||
      !Expect(interpreter.Supports("BUFFER_LOAD_DWORD"),
              "expected BUFFER_LOAD_DWORD support")) {
    return 1;
  }

  const std::array<std::string_view, 18> kExtendedDsOpcodes = {
      "DS_SUB_U32", "DS_RSUB_U32", "DS_INC_U32", "DS_DEC_U32",
      "DS_MIN_I32", "DS_MAX_I32",  "DS_MIN_U32", "DS_MAX_U32",
      "DS_AND_B32", "DS_OR_B32",   "DS_XOR_B32", "DS_ADD_F32",
      "DS_MIN_F32", "DS_MAX_F32",  "DS_WRITE_B8", "DS_WRITE_B16",
      "DS_PK_ADD_F16", "DS_PK_ADD_BF16",
  };
  for (std::string_view opcode : kExtendedDsOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected extended DS opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kDsLaneRoutingOpcodes = {
      "DS_SWIZZLE_B32",
      "DS_PERMUTE_B32",
      "DS_BPERMUTE_B32",
  };
  for (std::string_view opcode : kDsLaneRoutingOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds lane-routing opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kDsDualDataOpcodes = {
      "DS_MSKOR_B32",
      "DS_CMPST_B32",
      "DS_CMPST_F32",
  };
  for (std::string_view opcode : kDsDualDataOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds dual-data opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsWideWriteOpcodes = {
      "DS_WRITE_B64",
  };
  for (std::string_view opcode : kDsWideWriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide-write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsMultiDwordWriteOpcodes = {
      "DS_WRITE_B96",
      "DS_WRITE_B128",
  };
  for (std::string_view opcode : kDsMultiDwordWriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds multi-dword write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairWriteOpcodes = {
      "DS_WRITE2_B32",
      "DS_WRITE2ST64_B32",
  };
  for (std::string_view opcode : kDsPairWriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds pair-write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairWriteOpcodes = {
      "DS_WRITE2_B64",
      "DS_WRITE2ST64_B64",
  };
  for (std::string_view opcode : kDsWidePairWriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide pair-write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsWideReadOpcodes = {
      "DS_READ_B64",
  };
  for (std::string_view opcode : kDsWideReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide-read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsMultiDwordReadOpcodes = {
      "DS_READ_B96",
      "DS_READ_B128",
  };
  for (std::string_view opcode : kDsMultiDwordReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds multi-dword read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairReadOpcodes = {
      "DS_READ2_B32",
      "DS_READ2ST64_B32",
  };
  for (std::string_view opcode : kDsPairReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds pair-read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairReturnOpcodes = {
      "DS_WRXCHG2_RTN_B32",
      "DS_WRXCHG2ST64_RTN_B32",
  };
  for (std::string_view opcode : kDsPairReturnOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds pair-return opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairReadOpcodes = {
      "DS_READ2_B64",
      "DS_READ2ST64_B64",
  };
  for (std::string_view opcode : kDsWidePairReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide pair-read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairReturnOpcodes = {
      "DS_WRXCHG2_RTN_B64",
      "DS_WRXCHG2ST64_RTN_B64",
  };
  for (std::string_view opcode : kDsWidePairReturnOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide pair-return opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 15> kDsWideUpdateOpcodes = {
      "DS_ADD_U64", "DS_SUB_U64", "DS_RSUB_U64", "DS_INC_U64", "DS_DEC_U64",
      "DS_MIN_I64", "DS_MAX_I64", "DS_MIN_U64", "DS_MAX_U64", "DS_AND_B64",
      "DS_OR_B64",  "DS_XOR_B64", "DS_ADD_F64",  "DS_MIN_F64", "DS_MAX_F64",
  };
  for (std::string_view opcode : kDsWideUpdateOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide update opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 4> kDsNarrowReadOpcodes = {
      "DS_READ_I8",
      "DS_READ_U8",
      "DS_READ_I16",
      "DS_READ_U16",
  };
  for (std::string_view opcode : kDsNarrowReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds narrow-read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsD16WriteOpcodes = {
      "DS_WRITE_B8_D16_HI",
      "DS_WRITE_B16_D16_HI",
  };
  for (std::string_view opcode : kDsD16WriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds d16 write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 6> kDsD16ReadOpcodes = {
      "DS_READ_U8_D16",
      "DS_READ_U8_D16_HI",
      "DS_READ_I8_D16",
      "DS_READ_I8_D16_HI",
      "DS_READ_U16_D16",
      "DS_READ_U16_D16_HI",
  };
  for (std::string_view opcode : kDsD16ReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds d16 read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsAddTidWriteOpcodes = {
      "DS_WRITE_ADDTID_B32",
  };
  for (std::string_view opcode : kDsAddTidWriteOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds addtid write opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsAddTidReadOpcodes = {
      "DS_READ_ADDTID_B32",
  };
  for (std::string_view opcode : kDsAddTidReadOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds addtid read opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWaveCounterOpcodes = {
      "DS_CONSUME",
      "DS_APPEND",
  };
  for (std::string_view opcode : kDsWaveCounterOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wave-counter opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 4> kDsDualDataReturnOpcodes = {
      "DS_MSKOR_RTN_B32",
      "DS_CMPST_RTN_B32",
      "DS_CMPST_RTN_F32",
      "DS_WRAP_RTN_B32",
  };
  for (std::string_view opcode : kDsDualDataReturnOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds dual-data return opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kDsWideDualDataOpcodes = {
      "DS_MSKOR_B64",
      "DS_CMPST_B64",
      "DS_CMPST_F64",
  };
  for (std::string_view opcode : kDsWideDualDataOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected ds wide dual-data opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 18> kReturningDsOpcodes = {
      "DS_ADD_RTN_U32", "DS_SUB_RTN_U32", "DS_RSUB_RTN_U32",
      "DS_INC_RTN_U32", "DS_DEC_RTN_U32", "DS_MIN_RTN_I32",
      "DS_MAX_RTN_I32", "DS_MIN_RTN_U32", "DS_MAX_RTN_U32",
      "DS_AND_RTN_B32", "DS_OR_RTN_B32",  "DS_XOR_RTN_B32",
      "DS_WRXCHG_RTN_B32", "DS_ADD_RTN_F32", "DS_MIN_RTN_F32",
      "DS_MAX_RTN_F32", "DS_PK_ADD_RTN_F16", "DS_PK_ADD_RTN_BF16",
  };
  for (std::string_view opcode : kReturningDsOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected returning DS opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kWideReturningDsDualDataOpcodes = {
      "DS_MSKOR_RTN_B64",
      "DS_CMPST_RTN_B64",
      "DS_CMPST_RTN_F64",
  };
  for (std::string_view opcode : kWideReturningDsDualDataOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected wide returning ds dual-data opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 17> kWideReturningDsOpcodes = {
      "DS_ADD_RTN_U64", "DS_SUB_RTN_U64", "DS_RSUB_RTN_U64",
      "DS_INC_RTN_U64", "DS_DEC_RTN_U64", "DS_MIN_RTN_I64",
      "DS_MAX_RTN_I64", "DS_MIN_RTN_U64", "DS_MAX_RTN_U64",
      "DS_AND_RTN_B64", "DS_OR_RTN_B64",  "DS_XOR_RTN_B64",
      "DS_WRXCHG_RTN_B64", "DS_ADD_RTN_F64", "DS_MIN_RTN_F64",
      "DS_CONDXCHG32_RTN_B64", "DS_MAX_RTN_F64",
  };
  for (std::string_view opcode : kWideReturningDsOpcodes) {
    if (!Expect(interpreter.Supports(opcode),
                "expected wide returning ds opcode support")) {
      std::cerr << opcode << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 16> kVectorCompare64Opcodes = {
      "V_CMP_F_I64",  "V_CMP_LT_I64", "V_CMP_EQ_I64", "V_CMP_LE_I64",
      "V_CMP_GT_I64", "V_CMP_NE_I64", "V_CMP_GE_I64", "V_CMP_T_I64",
      "V_CMP_F_U64",  "V_CMP_LT_U64", "V_CMP_EQ_U64", "V_CMP_LE_U64",
      "V_CMP_GT_U64", "V_CMP_NE_U64", "V_CMP_GE_U64", "V_CMP_T_U64",
  };
  const std::array<std::string_view, 16> kVectorCompareF32Opcodes = {
      "V_CMP_F_F32",   "V_CMP_LT_F32",  "V_CMP_EQ_F32",  "V_CMP_LE_F32",
      "V_CMP_GT_F32",  "V_CMP_LG_F32",  "V_CMP_GE_F32",  "V_CMP_O_F32",
      "V_CMP_U_F32",   "V_CMP_NGE_F32", "V_CMP_NLG_F32", "V_CMP_NGT_F32",
      "V_CMP_NLE_F32", "V_CMP_NEQ_F32", "V_CMP_NLT_F32", "V_CMP_TRU_F32",
  };
  const std::array<std::string_view, 16> kVectorCompareF64Opcodes = {
      "V_CMP_F_F64",   "V_CMP_LT_F64",  "V_CMP_EQ_F64",  "V_CMP_LE_F64",
      "V_CMP_GT_F64",  "V_CMP_LG_F64",  "V_CMP_GE_F64",  "V_CMP_O_F64",
      "V_CMP_U_F64",   "V_CMP_NGE_F64", "V_CMP_NLG_F64", "V_CMP_NGT_F64",
      "V_CMP_NLE_F64", "V_CMP_NEQ_F64", "V_CMP_NLT_F64", "V_CMP_TRU_F64",
  };
  for (std::string_view opcode : kVectorCompareF32Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }
  for (std::string_view opcode : kVectorCompareF64Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }
  for (std::string_view opcode : kVectorCompare64Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 16> kVectorCmpxF32Opcodes = {
      "V_CMPX_F_F32",   "V_CMPX_LT_F32",  "V_CMPX_EQ_F32",  "V_CMPX_LE_F32",
      "V_CMPX_GT_F32",  "V_CMPX_LG_F32",  "V_CMPX_GE_F32",  "V_CMPX_O_F32",
      "V_CMPX_U_F32",   "V_CMPX_NGE_F32", "V_CMPX_NLG_F32", "V_CMPX_NGT_F32",
      "V_CMPX_NLE_F32", "V_CMPX_NEQ_F32", "V_CMPX_NLT_F32", "V_CMPX_TRU_F32",
  };
  const std::array<std::string_view, 16> kVectorCmpxF64Opcodes = {
      "V_CMPX_F_F64",   "V_CMPX_LT_F64",  "V_CMPX_EQ_F64",  "V_CMPX_LE_F64",
      "V_CMPX_GT_F64",  "V_CMPX_LG_F64",  "V_CMPX_GE_F64",  "V_CMPX_O_F64",
      "V_CMPX_U_F64",   "V_CMPX_NGE_F64", "V_CMPX_NLG_F64", "V_CMPX_NGT_F64",
      "V_CMPX_NLE_F64", "V_CMPX_NEQ_F64", "V_CMPX_NLT_F64", "V_CMPX_TRU_F64",
  };
  const std::array<std::string_view, 4> kVectorClassOpcodes = {
      "V_CMP_CLASS_F32", "V_CMPX_CLASS_F32",
      "V_CMP_CLASS_F64", "V_CMPX_CLASS_F64",
  };
  for (std::string_view opcode : kVectorCmpxF32Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }
  for (std::string_view opcode : kVectorCmpxF64Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }
  for (std::string_view opcode : kVectorClassOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 32> kVectorCmpxOpcodes = {
      "V_CMPX_F_I32",  "V_CMPX_LT_I32", "V_CMPX_EQ_I32", "V_CMPX_LE_I32",
      "V_CMPX_GT_I32", "V_CMPX_NE_I32", "V_CMPX_GE_I32", "V_CMPX_T_I32",
      "V_CMPX_F_U32",  "V_CMPX_LT_U32", "V_CMPX_EQ_U32", "V_CMPX_LE_U32",
      "V_CMPX_GT_U32", "V_CMPX_NE_U32", "V_CMPX_GE_U32", "V_CMPX_T_U32",
      "V_CMPX_F_I64",  "V_CMPX_LT_I64", "V_CMPX_EQ_I64", "V_CMPX_LE_I64",
      "V_CMPX_GT_I64", "V_CMPX_NE_I64", "V_CMPX_GE_I64", "V_CMPX_T_I64",
      "V_CMPX_F_U64",  "V_CMPX_LT_U64", "V_CMPX_EQ_U64", "V_CMPX_LE_U64",
      "V_CMPX_GT_U64", "V_CMPX_NE_U64", "V_CMPX_GE_U64", "V_CMPX_T_U64",
  };
  for (std::string_view opcode : kVectorCmpxOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<SaveexecSemanticCase, 10> kSaveexecCases = {{
      {"S_AND_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0x00000000000000c0ULL},
      {"S_ANDN1_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0x0000000000000030ULL},
      {"S_ANDN2_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0x0000000000000c03ULL},
      {"S_NAND_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0xffffffffffffff3fULL},
      {"S_OR_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0x0000000000000cf3ULL},
      {"S_ORN1_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0xfffffffffffff3fcULL},
      {"S_ORN2_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0xffffffffffffffcfULL},
      {"S_NOR_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0xfffffffffffff30cULL},
      {"S_XOR_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0x0000000000000c33ULL},
      {"S_XNOR_SAVEEXEC_B64", 0x00000000000000f0ULL, 0x0000000000000cc3ULL,
       0xfffffffffffff3ccULL},
  }};
  for (const SaveexecSemanticCase& test_case : kSaveexecCases) {
    if (!RunSaveexecSemanticCase(interpreter, test_case, false) ||
        !RunSaveexecSemanticCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarUnaryCase, 19> kScalarUnaryCases = {{
      {"S_CMOV_B32", 0x11223344U, 0x55667788U, 0x55667788U, false, false},
      {"S_CMOV_B32", 0x11223344U, 0x55667788U, 0x11223344U, true, true},
      {"S_NOT_B32", 0x00ff00aaU, 0U, 0xff00ff55U, false, true},
      {"S_NOT_B32", 0xffffffffU, 0U, 0x00000000U, true, false},
      {"S_ABS_I32", 0xffffff80U, 0U, 0x00000080U, false, true},
      {"S_BREV_B32", 0x0000000dU, 0U, 0xb0000000U, true, true},
      {"S_BCNT0_I32_B32", 0xffffffffU, 0U, 0U, true, false},
      {"S_BCNT1_I32_B32", 0xf0f0f0f0U, 0U, 16U, false, true},
      {"S_FF0_I32_B32", 0xfffffff7U, 0U, 3U, true, true},
      {"S_FF1_I32_B32", 0x00000010U, 0U, 4U, false, false},
      {"S_FLBIT_I32_B32", 0x00000010U, 0U, 27U, true, true},
      {"S_FLBIT_I32", 0xfffffff0U, 0U, 28U, false, false},
      {"S_QUADMASK_B32", 0x0000f00fU, 0U, 0x00000009U, false, true},
      {"S_SEXT_I32_I8", 0x00000080U, 0U, 0xffffff80U, true, true},
      {"S_SEXT_I32_I16", 0x00008001U, 0U, 0xffff8001U, false, false},
      {"S_BITSET0_B32", 5U, 0xffffffffU, 0xffffffdfU, true, true},
      {"S_BITSET0_B32", 3U, 0x00000000U, 0x00000000U, false, false},
      {"S_BITSET1_B32", 5U, 0x00000000U, 0x00000020U, true, true},
      {"S_BITSET1_B32", 31U, 0x7fffffffU, 0xffffffffU, false, false},
  }};
  for (const ScalarUnaryCase& test_case : kScalarUnaryCases) {
    if (!RunScalarUnaryCase(interpreter, test_case, false) ||
        !RunScalarUnaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarBinaryCase, 30> kScalarBinaryCases = {{
      {"S_CSELECT_B32", 0x11111111U, 0x22222222U, 0x11111111U, true, true},
      {"S_CSELECT_B32", 0x11111111U, 0x22222222U, 0x22222222U, false, false},
      {"S_ABSDIFF_I32", 5U, 17U, 12U, false, true},
      {"S_ADD_I32", 0xffffffffU, 1U, 0x00000000U, false, true},
      {"S_SUB_I32", 0U, 1U, 0xffffffffU, false, false},
      {"S_MIN_I32", 0xfffffffbu, 3U, 0xfffffffbu, false, true},
      {"S_MIN_U32", 5U, 3U, 3U, true, false},
      {"S_MAX_I32", 0xfffffffbu, 3U, 3U, true, false},
      {"S_MAX_U32", 5U, 3U, 5U, false, true},
      {"S_BFE_U32", 0x12345678U, 0x00080008U, 0x00000056U, false, true},
      {"S_BFE_I32", 0x0000f000U, 0x0004000cU, 0xffffffffU, false, true},
      {"S_ANDN2_B32", 0x55ff0f0fU, 0x3300aa55U, 0x44ff050aU, false, true},
      {"S_NAND_B32", 0x55ff0f0fU, 0x3300aa55U, 0xeefff5faU, false, true},
      {"S_ORN2_B32", 0x55ff0f0fU, 0x3300aa55U, 0xddff5fafU, false, true},
      {"S_NOR_B32", 0x55ff0f0fU, 0x3300aa55U, 0x880050a0U, false, true},
      {"S_XNOR_B32", 0x55ff0f0fU, 0x3300aa55U, 0x99005aa5U, false, true},
      {"S_ADDC_U32", 0xffffffffU, 0U, 0x00000000U, true, true},
      {"S_SUBB_U32", 0U, 0U, 0xffffffffU, true, false},
      {"S_LSHL1_ADD_U32", 0x40000000U, 0x80000000U, 0x00000000U, false, true},
      {"S_LSHL2_ADD_U32", 0x10000000U, 0xc0000000U, 0x00000000U, false, true},
      {"S_LSHL3_ADD_U32", 0x02000000U, 0xf0000000U, 0x00000000U, false, true},
      {"S_LSHL4_ADD_U32", 0x01000000U, 0xf0000000U, 0x00000000U, false, true},
      {"S_LSHL_B32", 0x00000011U, 4U, 0x00000110U, false, true},
      {"S_LSHR_B32", 0x80000000U, 4U, 0x08000000U, false, true},
      {"S_ASHR_I32", 0x80000000U, 4U, 0xf8000000U, false, true},
      {"S_BFM_B32", 5U, 3U, 0x000000f8U, true, true},
      {"S_BFM_B32", 0U, 9U, 0x00000000U, false, false},
      {"S_PACK_LL_B32_B16", 0x12345678U, 0xabcdef90U, 0xef905678U, true, true},
      {"S_PACK_LH_B32_B16", 0x12345678U, 0xabcdef90U, 0xabcd5678U, false, false},
      {"S_PACK_HH_B32_B16", 0x12345678U, 0xabcdef90U, 0xabcd1234U, true, true},
  }};
  for (const ScalarBinaryCase& test_case : kScalarBinaryCases) {
    if (!RunScalarBinaryCase(interpreter, test_case, false) ||
        !RunScalarBinaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarPairUnaryCase, 9> kScalarPairUnaryCases = {{
      {"S_CMOV_B64", 0x0123456789abcdefULL, 0xfedcba9876543210ULL,
       0xfedcba9876543210ULL, false, false},
      {"S_CMOV_B64", 0x0123456789abcdefULL, 0xfedcba9876543210ULL,
       0x0123456789abcdefULL, true, true},
      {"S_MOV_B64", 0x123456789abcdef0ULL, 0ULL, 0x123456789abcdef0ULL, false,
       true},
      {"S_NOT_B64", 0x123456789abcdef0ULL, 0ULL, 0xedcba9876543210fULL, false,
       true},
      {"S_NOT_B64", 0xffffffffffffffffULL, 0ULL, 0x0000000000000000ULL, true,
       false},
      {"S_BREV_B64", 0x000000000000000dULL, 0ULL, 0xb000000000000000ULL, true,
       true},
      {"S_QUADMASK_B64", 0xf00000000000000fULL, 0ULL, 0x0000000000008001ULL,
       false, true},
      {"S_BITSET0_B64", 5ULL, 0xffffffffffffffffULL, 0xffffffffffffffdfULL, true,
       true},
      {"S_BITSET1_B64", 63ULL, 0x7fffffffffffffffULL, 0xffffffffffffffffULL, false,
       false},
  }};
  for (const ScalarPairUnaryCase& test_case : kScalarPairUnaryCases) {
    if (!RunScalarPairUnaryCase(interpreter, test_case, false) ||
        !RunScalarPairUnaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarPairFromScalarUnaryCase, 1>
      kScalarPairFromScalarUnaryCases = {{
          {"S_BITREPLICATE_B64_B32", 0x00000005U, 0x0000000000000033ULL, true,
           true},
      }};
  for (const ScalarPairFromScalarUnaryCase& test_case :
       kScalarPairFromScalarUnaryCases) {
    if (!RunScalarPairFromScalarUnaryCase(interpreter, test_case, false) ||
        !RunScalarPairFromScalarUnaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarFromPairUnaryCase, 6> kScalarFromPairUnaryCases = {{
      {"S_BCNT0_I32_B64", 0xffffffffffffffffULL, 0U, true, false},
      {"S_BCNT1_I32_B64", 0xf0f0f0f00f0f0f0fULL, 32U, false, true},
      {"S_FF0_I32_B64", 0xfffffffffffffff7ULL, 3U, true, true},
      {"S_FF1_I32_B64", 0x0000000000000010ULL, 4U, false, false},
      {"S_FLBIT_I32_B64", 0x0000001000000000ULL, 27U, true, true},
      {"S_FLBIT_I32_I64", 0xfffffffffffffff0ULL, 60U, false, false},
  }};
  for (const ScalarFromPairUnaryCase& test_case : kScalarFromPairUnaryCases) {
    if (!RunScalarFromPairUnaryCase(interpreter, test_case, false) ||
        !RunScalarFromPairUnaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarPairBinaryCase, 17> kScalarPairBinaryCases = {{
      {"S_CSELECT_B64", 0x1111111122222222ULL, 0x3333333344444444ULL,
       0x1111111122222222ULL, true, true},
      {"S_CSELECT_B64", 0x1111111122222222ULL, 0x3333333344444444ULL,
       0x3333333344444444ULL, false, false},
      {"S_BFE_U64", 0x123456789abcdef0ULL, 0x0000000000100008ULL,
       0x000000000000bcdeULL, false, true},
      {"S_BFE_I64", 0x000000000000f000ULL, 0x000000000004000cULL,
       0xffffffffffffffffULL, false, true},
      {"S_AND_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0x000000aa11000a05ULL, false, true},
      {"S_AND_B64", 0x00000000ffffffffULL, 0xffffffff00000000ULL,
       0x0000000000000000ULL, true, false},
      {"S_ANDN2_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0x00f0000044ff050aULL, false, true},
      {"S_NAND_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0xffffff55eefff5faULL, false, true},
      {"S_OR_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0x0fff00ff77ffaf5fULL, false, true},
      {"S_ORN2_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0xf0f0ffaaddff5fafULL, false, true},
      {"S_NOR_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0xf000ff00880050a0ULL, false, true},
      {"S_XOR_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0x0fff005566ffa55aULL, false, true},
      {"S_XNOR_B64", 0x00f000aa55ff0f0fULL, 0x0f0f00ff3300aa55ULL,
       0xf000ffaa99005aa5ULL, false, true},
      {"S_LSHL_B64", 0x0000000100000001ULL, 4ULL, 0x0000001000000010ULL,
       false, true},
      {"S_LSHR_B64", 0x8000000000000000ULL, 4ULL, 0x0800000000000000ULL,
       false, true},
      {"S_ASHR_I64", 0x8000000000000000ULL, 4ULL, 0xf800000000000000ULL,
       false, true},
      {"S_BFM_B64", 9ULL, 12ULL, 0x00000000001ff000ULL, true, true},
  }};
  for (const ScalarPairBinaryCase& test_case : kScalarPairBinaryCases) {
    if (!RunScalarPairBinaryCase(interpreter, test_case, false) ||
        !RunScalarPairBinaryCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<ScalarPairCompareCase, 4> kScalarPairCompareCases = {{
      {"S_CMP_EQ_U64", 0x123456789abcdef0ULL, 0x123456789abcdef0ULL, true},
      {"S_CMP_EQ_U64", 0x123456789abcdef0ULL, 0x123456789abcdef1ULL, false},
      {"S_CMP_LG_U64", 0x0000000000000001ULL, 0x0000000000000002ULL, true},
      {"S_CMP_LG_U64", 0xabcdef0123456789ULL, 0xabcdef0123456789ULL, false},
  }};
  for (const ScalarPairCompareCase& test_case : kScalarPairCompareCases) {
    if (!RunScalarPairCompareCase(interpreter, test_case, false) ||
        !RunScalarPairCompareCase(interpreter, test_case, true)) {
      return 1;
    }
  }

  const std::array<std::string_view, 7> kGlobalLoadLdsOpcodes = {
      "GLOBAL_LOAD_LDS_UBYTE",  "GLOBAL_LOAD_LDS_SBYTE",
      "GLOBAL_LOAD_LDS_USHORT", "GLOBAL_LOAD_LDS_SSHORT",
      "GLOBAL_LOAD_LDS_DWORD",  "GLOBAL_LOAD_LDS_DWORDX3",
      "GLOBAL_LOAD_LDS_DWORDX4",
  };
  for (std::string_view opcode : kGlobalLoadLdsOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 5> kWideScalarMemoryOpcodes = {
      "S_LOAD_DWORDX4", "S_LOAD_DWORDX8", "S_LOAD_DWORDX16",
      "S_STORE_DWORDX2", "S_STORE_DWORDX4",
  };
  for (std::string_view opcode : kWideScalarMemoryOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 8> kScalarBufferMemoryOpcodes = {
      "S_BUFFER_LOAD_DWORD",   "S_BUFFER_LOAD_DWORDX2",
      "S_BUFFER_LOAD_DWORDX4", "S_BUFFER_LOAD_DWORDX8",
      "S_BUFFER_LOAD_DWORDX16", "S_BUFFER_STORE_DWORD",
      "S_BUFFER_STORE_DWORDX2", "S_BUFFER_STORE_DWORDX4",
  };
  for (std::string_view opcode : kScalarBufferMemoryOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 12> kScalarBufferAtomic32Opcodes = {
      "S_BUFFER_ATOMIC_SWAP", "S_BUFFER_ATOMIC_ADD",
      "S_BUFFER_ATOMIC_SUB",  "S_BUFFER_ATOMIC_SMIN",
      "S_BUFFER_ATOMIC_UMIN", "S_BUFFER_ATOMIC_SMAX",
      "S_BUFFER_ATOMIC_UMAX", "S_BUFFER_ATOMIC_AND",
      "S_BUFFER_ATOMIC_OR",   "S_BUFFER_ATOMIC_XOR",
      "S_BUFFER_ATOMIC_INC",  "S_BUFFER_ATOMIC_DEC",
  };
  for (std::string_view opcode : kScalarBufferAtomic32Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 12> kScalarBufferAtomic64Opcodes = {
      "S_BUFFER_ATOMIC_SWAP_X2", "S_BUFFER_ATOMIC_ADD_X2",
      "S_BUFFER_ATOMIC_SUB_X2",  "S_BUFFER_ATOMIC_SMIN_X2",
      "S_BUFFER_ATOMIC_UMIN_X2", "S_BUFFER_ATOMIC_SMAX_X2",
      "S_BUFFER_ATOMIC_UMAX_X2", "S_BUFFER_ATOMIC_AND_X2",
      "S_BUFFER_ATOMIC_OR_X2",   "S_BUFFER_ATOMIC_XOR_X2",
      "S_BUFFER_ATOMIC_INC_X2",  "S_BUFFER_ATOMIC_DEC_X2",
  };
  for (std::string_view opcode : kScalarBufferAtomic64Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 12> kScalarAtomic32Opcodes = {
      "S_ATOMIC_SWAP", "S_ATOMIC_ADD",  "S_ATOMIC_SUB", "S_ATOMIC_SMIN",
      "S_ATOMIC_UMIN", "S_ATOMIC_SMAX", "S_ATOMIC_UMAX", "S_ATOMIC_AND",
      "S_ATOMIC_OR",   "S_ATOMIC_XOR",  "S_ATOMIC_INC",  "S_ATOMIC_DEC",
  };
  for (std::string_view opcode : kScalarAtomic32Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 12> kScalarAtomic64Opcodes = {
      "S_ATOMIC_SWAP_X2", "S_ATOMIC_ADD_X2",  "S_ATOMIC_SUB_X2",
      "S_ATOMIC_SMIN_X2", "S_ATOMIC_UMIN_X2", "S_ATOMIC_SMAX_X2",
      "S_ATOMIC_UMAX_X2", "S_ATOMIC_AND_X2",  "S_ATOMIC_OR_X2",
      "S_ATOMIC_XOR_X2",  "S_ATOMIC_INC_X2",  "S_ATOMIC_DEC_X2",
  };
  for (std::string_view opcode : kScalarAtomic64Opcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 4> kBlockedScalarAtomicOpcodes = {
      "S_BUFFER_ATOMIC_CMPSWAP", "S_BUFFER_ATOMIC_CMPSWAP_X2",
      "S_ATOMIC_CMPSWAP",        "S_ATOMIC_CMPSWAP_X2",
  };
  for (std::string_view opcode : kBlockedScalarAtomicOpcodes) {
    const std::string message = "expected " + std::string(opcode) +
                                " to remain unsupported";
    if (!Expect(!interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 10> kScalarMaintenanceOpcodes = {
      "S_DCACHE_INV",      "S_DCACHE_WB",      "S_DCACHE_INV_VOL",
      "S_DCACHE_WB_VOL",   "S_DCACHE_DISCARD", "S_DCACHE_DISCARD_X2",
      "S_MEMTIME",         "S_MEMREALTIME",    "S_ATC_PROBE",
      "S_ATC_PROBE_BUFFER",
  };
  for (std::string_view opcode : kScalarMaintenanceOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 56> kBufferMemoryOpcodes = {
      "BUFFER_LOAD_FORMAT_X",     "BUFFER_LOAD_FORMAT_XY",
      "BUFFER_LOAD_FORMAT_XYZ",   "BUFFER_LOAD_FORMAT_XYZW",
      "BUFFER_STORE_FORMAT_X",    "BUFFER_STORE_FORMAT_XY",
      "BUFFER_STORE_FORMAT_XYZ",  "BUFFER_STORE_FORMAT_XYZW",
      "BUFFER_LOAD_FORMAT_D16_X", "BUFFER_LOAD_FORMAT_D16_XY",
      "BUFFER_LOAD_FORMAT_D16_XYZ", "BUFFER_LOAD_FORMAT_D16_XYZW",
      "BUFFER_STORE_FORMAT_D16_X",  "BUFFER_STORE_FORMAT_D16_XY",
      "BUFFER_STORE_FORMAT_D16_XYZ", "BUFFER_STORE_FORMAT_D16_XYZW",
      "BUFFER_LOAD_FORMAT_D16_HI_X", "BUFFER_STORE_FORMAT_D16_HI_X",
      "TBUFFER_LOAD_FORMAT_X",      "TBUFFER_LOAD_FORMAT_XY",
      "TBUFFER_LOAD_FORMAT_XYZ",    "TBUFFER_LOAD_FORMAT_XYZW",
      "TBUFFER_STORE_FORMAT_X",     "TBUFFER_STORE_FORMAT_XY",
      "TBUFFER_STORE_FORMAT_XYZ",   "TBUFFER_STORE_FORMAT_XYZW",
      "TBUFFER_LOAD_FORMAT_D16_X",  "TBUFFER_LOAD_FORMAT_D16_XY",
      "TBUFFER_LOAD_FORMAT_D16_XYZ","TBUFFER_LOAD_FORMAT_D16_XYZW",
      "TBUFFER_STORE_FORMAT_D16_X", "TBUFFER_STORE_FORMAT_D16_XY",
      "TBUFFER_STORE_FORMAT_D16_XYZ","TBUFFER_STORE_FORMAT_D16_XYZW",
      "BUFFER_LOAD_UBYTE",        "BUFFER_LOAD_SBYTE",
      "BUFFER_LOAD_USHORT",       "BUFFER_LOAD_SSHORT",
      "BUFFER_LOAD_DWORD",        "BUFFER_LOAD_DWORDX2",
      "BUFFER_LOAD_DWORDX3",      "BUFFER_LOAD_DWORDX4",
      "BUFFER_STORE_BYTE",        "BUFFER_STORE_BYTE_D16_HI",
      "BUFFER_STORE_SHORT",       "BUFFER_STORE_SHORT_D16_HI",
      "BUFFER_STORE_DWORD",       "BUFFER_STORE_DWORDX2",
      "BUFFER_STORE_DWORDX3",     "BUFFER_STORE_DWORDX4",
      "BUFFER_LOAD_UBYTE_D16",    "BUFFER_LOAD_UBYTE_D16_HI",
      "BUFFER_LOAD_SBYTE_D16",    "BUFFER_LOAD_SBYTE_D16_HI",
      "BUFFER_LOAD_SHORT_D16",    "BUFFER_LOAD_SHORT_D16_HI",
  };
  for (std::string_view opcode : kBufferMemoryOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 32> kBufferAtomicOpcodes = {
      "BUFFER_ATOMIC_SWAP",      "BUFFER_ATOMIC_CMPSWAP",
      "BUFFER_ATOMIC_ADD",       "BUFFER_ATOMIC_SUB",
      "BUFFER_ATOMIC_SMIN",      "BUFFER_ATOMIC_UMIN",
      "BUFFER_ATOMIC_SMAX",      "BUFFER_ATOMIC_UMAX",
      "BUFFER_ATOMIC_AND",       "BUFFER_ATOMIC_OR",
      "BUFFER_ATOMIC_XOR",       "BUFFER_ATOMIC_INC",
      "BUFFER_ATOMIC_DEC",       "BUFFER_ATOMIC_ADD_F32",
      "BUFFER_ATOMIC_PK_ADD_F16","BUFFER_ATOMIC_ADD_F64",
      "BUFFER_ATOMIC_MIN_F64",   "BUFFER_ATOMIC_MAX_F64",
      "BUFFER_ATOMIC_PK_ADD_BF16","BUFFER_ATOMIC_SWAP_X2",
      "BUFFER_ATOMIC_CMPSWAP_X2","BUFFER_ATOMIC_ADD_X2",
      "BUFFER_ATOMIC_SUB_X2",    "BUFFER_ATOMIC_SMIN_X2",
      "BUFFER_ATOMIC_UMIN_X2",   "BUFFER_ATOMIC_SMAX_X2",
      "BUFFER_ATOMIC_UMAX_X2",   "BUFFER_ATOMIC_AND_X2",
      "BUFFER_ATOMIC_OR_X2",     "BUFFER_ATOMIC_XOR_X2",
      "BUFFER_ATOMIC_INC_X2",    "BUFFER_ATOMIC_DEC_X2",
  };
  for (std::string_view opcode : kBufferAtomicOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 32> kGlobalAtomicOpcodes = {
      "GLOBAL_ATOMIC_SWAP",
      "GLOBAL_ATOMIC_CMPSWAP",
      "GLOBAL_ATOMIC_ADD",
      "GLOBAL_ATOMIC_SUB",
      "GLOBAL_ATOMIC_SMIN",
      "GLOBAL_ATOMIC_UMIN",
      "GLOBAL_ATOMIC_SMAX",
      "GLOBAL_ATOMIC_UMAX",
      "GLOBAL_ATOMIC_AND",
      "GLOBAL_ATOMIC_OR",
      "GLOBAL_ATOMIC_XOR",
      "GLOBAL_ATOMIC_INC",
      "GLOBAL_ATOMIC_DEC",
      "GLOBAL_ATOMIC_ADD_F32",
      "GLOBAL_ATOMIC_PK_ADD_F16",
      "GLOBAL_ATOMIC_ADD_F64",
      "GLOBAL_ATOMIC_MIN_F64",
      "GLOBAL_ATOMIC_MAX_F64",
      "GLOBAL_ATOMIC_PK_ADD_BF16",
      "GLOBAL_ATOMIC_SWAP_X2",
      "GLOBAL_ATOMIC_CMPSWAP_X2",
      "GLOBAL_ATOMIC_ADD_X2",
      "GLOBAL_ATOMIC_SUB_X2",
      "GLOBAL_ATOMIC_SMIN_X2",
      "GLOBAL_ATOMIC_UMIN_X2",
      "GLOBAL_ATOMIC_SMAX_X2",
      "GLOBAL_ATOMIC_UMAX_X2",
      "GLOBAL_ATOMIC_AND_X2",
      "GLOBAL_ATOMIC_OR_X2",
      "GLOBAL_ATOMIC_XOR_X2",
      "GLOBAL_ATOMIC_INC_X2",
      "GLOBAL_ATOMIC_DEC_X2",
  };
  for (std::string_view opcode : kGlobalAtomicOpcodes) {
    const std::string message = "expected " + std::string(opcode) + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }
  for (std::string_view global_opcode : kGlobalAtomicOpcodes) {
    const std::string opcode =
        "FLAT_" + std::string(global_opcode.substr(7));
    const std::string message = "expected " + opcode + " support";
    if (!Expect(interpreter.Supports(opcode), message.c_str())) {
      return 1;
    }
  }

  WaveExecutionState state;
  state.exec_mask = 0b1011ULL;

  const std::vector<DecodedInstruction> program = {
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(7)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(5)),
      DecodedInstruction::Binary("S_ADD_U32", InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(1)),
      DecodedInstruction::Binary("S_XOR_B32", InstructionOperand::Sgpr(3),
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Imm32(0x0F)),
      DecodedInstruction::Unary("V_MOV_B32", InstructionOperand::Vgpr(0),
                                InstructionOperand::Imm32(3)),
      DecodedInstruction::Unary("V_MOV_B32", InstructionOperand::Vgpr(1),
                                InstructionOperand::Sgpr(2)),
      DecodedInstruction::Binary("V_ADD_U32", InstructionOperand::Vgpr(2),
                                 InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Binary("V_SUB_U32", InstructionOperand::Vgpr(3),
                                 InstructionOperand::Vgpr(2),
                                 InstructionOperand::Imm32(1)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  std::string error_message;
  if (!Expect(interpreter.ExecuteProgram(program, &state, &error_message),
              error_message.c_str())) {
    return 1;
  }

  if (!Expect(state.halted, "expected program to halt") ||
      !Expect(state.pc == 8, "expected PC to stop on S_ENDPGM") ||
      !Expect(state.sgprs[2] == 12, "expected S_ADD_U32 result") ||
      !Expect(state.sgprs[3] == (12U ^ 0x0FU), "expected S_XOR_B32 result") ||
      !Expect(state.scc, "expected non-zero SCC after XOR")) {
    return 1;
  }

  if (!Expect(state.vgprs[2][0] == 15, "expected lane 0 V_ADD_U32 result") ||
      !Expect(state.vgprs[2][1] == 15, "expected lane 1 V_ADD_U32 result") ||
      !Expect(state.vgprs[2][2] == 0, "expected inactive lane to remain untouched") ||
      !Expect(state.vgprs[2][3] == 15, "expected lane 3 V_ADD_U32 result") ||
      !Expect(state.vgprs[3][0] == 14, "expected lane 0 V_SUB_U32 result") ||
      !Expect(state.vgprs[3][2] == 0, "expected inactive lane to remain untouched")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(program, &compiled_program, &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_state;
  compiled_state.exec_mask = 0b1011ULL;
  if (!Expect(interpreter.ExecuteProgram(compiled_program, &compiled_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_state.halted, "expected compiled program to halt") ||
      !Expect(compiled_state.sgprs[2] == 12,
              "expected compiled S_ADD_U32 result") ||
      !Expect(compiled_state.vgprs[2][0] == 15,
              "expected compiled lane 0 V_ADD_U32 result") ||
      !Expect(compiled_state.vgprs[3][3] == 14,
              "expected compiled lane 3 V_SUB_U32 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_extended_program = {
      DecodedInstruction::Binary("V_MIN_I32", InstructionOperand::Vgpr(10),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Vgpr(0)),
      DecodedInstruction::Binary("V_MAX_I32", InstructionOperand::Vgpr(11),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Vgpr(0)),
      DecodedInstruction::Binary("V_MIN_U32", InstructionOperand::Vgpr(12),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Binary("V_MAX_U32", InstructionOperand::Vgpr(13),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(1)),
      DecodedInstruction::Binary("V_LSHLREV_B32", InstructionOperand::Vgpr(14),
                                 InstructionOperand::Sgpr(2),
                                 InstructionOperand::Vgpr(2)),
      DecodedInstruction::Binary("V_LSHRREV_B32", InstructionOperand::Vgpr(15),
                                 InstructionOperand::Sgpr(3),
                                 InstructionOperand::Vgpr(3)),
      DecodedInstruction::Binary("V_ASHRREV_I32", InstructionOperand::Vgpr(16),
                                 InstructionOperand::Sgpr(3),
                                 InstructionOperand::Vgpr(3)),
      DecodedInstruction::Binary("V_AND_B32", InstructionOperand::Vgpr(17),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(13)),
      DecodedInstruction::Binary("V_OR_B32", InstructionOperand::Vgpr(18),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(12)),
      DecodedInstruction::Binary("V_XOR_B32", InstructionOperand::Vgpr(19),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Vgpr(13)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_extended_state;
  vector_extended_state.exec_mask = 0b1011ULL;
  vector_extended_state.sgprs[0] = 0xfffffff0u;
  vector_extended_state.sgprs[1] = 15u;
  vector_extended_state.sgprs[2] = 4u;
  vector_extended_state.sgprs[3] = 2u;
  vector_extended_state.vgprs[0][0] = 3u;
  vector_extended_state.vgprs[0][1] = 0xfffffff6u;
  vector_extended_state.vgprs[0][3] = 7u;
  vector_extended_state.vgprs[1][0] = 2u;
  vector_extended_state.vgprs[1][1] = 20u;
  vector_extended_state.vgprs[1][3] = 8u;
  vector_extended_state.vgprs[2][0] = 1u;
  vector_extended_state.vgprs[2][1] = 2u;
  vector_extended_state.vgprs[2][3] = 4u;
  vector_extended_state.vgprs[3][0] = 0xfffffff8u;
  vector_extended_state.vgprs[3][1] = 0xfffffffcu;
  vector_extended_state.vgprs[3][3] = 16u;
  if (!Expect(interpreter.ExecuteProgram(vector_extended_program,
                                         &vector_extended_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_extended_state.halted,
              "expected vector extended program to halt") ||
      !Expect(vector_extended_state.vgprs[10][0] == 0xfffffff0u,
              "expected v_min_i32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[10][1] == 0xfffffff0u,
              "expected v_min_i32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[10][2] == 0u,
              "expected inactive lane v_min_i32 result") ||
      !Expect(vector_extended_state.vgprs[11][0] == 3u,
              "expected v_max_i32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[11][1] == 0xfffffff6u,
              "expected v_max_i32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[12][0] == 2u,
              "expected v_min_u32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[12][1] == 15u,
              "expected v_min_u32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[13][1] == 20u,
              "expected v_max_u32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[14][0] == 16u,
              "expected v_lshlrev_b32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[14][1] == 32u,
              "expected v_lshlrev_b32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[14][3] == 64u,
              "expected v_lshlrev_b32 lane 3 result") ||
      !Expect(vector_extended_state.vgprs[15][0] == 0x3ffffffeu,
              "expected v_lshrrev_b32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[15][1] == 0x3fffffffu,
              "expected v_lshrrev_b32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[16][0] == 0xfffffffeu,
              "expected v_ashrrev_i32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[16][1] == 0xffffffffu,
              "expected v_ashrrev_i32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[17][1] == 4u,
              "expected v_and_b32 lane 1 result") ||
      !Expect(vector_extended_state.vgprs[18][3] == 15u,
              "expected v_or_b32 lane 3 result") ||
      !Expect(vector_extended_state.vgprs[19][0] == 0u,
              "expected v_xor_b32 lane 0 result") ||
      !Expect(vector_extended_state.vgprs[19][1] == 27u,
              "expected v_xor_b32 lane 1 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_extended_program;
  if (!Expect(interpreter.CompileProgram(vector_extended_program,
                                         &compiled_vector_extended_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_extended_state;
  compiled_vector_extended_state.exec_mask = 0b1011ULL;
  compiled_vector_extended_state.sgprs[0] = 0xfffffff0u;
  compiled_vector_extended_state.sgprs[1] = 15u;
  compiled_vector_extended_state.sgprs[2] = 4u;
  compiled_vector_extended_state.sgprs[3] = 2u;
  compiled_vector_extended_state.vgprs[0][0] = 3u;
  compiled_vector_extended_state.vgprs[0][1] = 0xfffffff6u;
  compiled_vector_extended_state.vgprs[0][3] = 7u;
  compiled_vector_extended_state.vgprs[1][0] = 2u;
  compiled_vector_extended_state.vgprs[1][1] = 20u;
  compiled_vector_extended_state.vgprs[1][3] = 8u;
  compiled_vector_extended_state.vgprs[2][0] = 1u;
  compiled_vector_extended_state.vgprs[2][1] = 2u;
  compiled_vector_extended_state.vgprs[2][3] = 4u;
  compiled_vector_extended_state.vgprs[3][0] = 0xfffffff8u;
  compiled_vector_extended_state.vgprs[3][1] = 0xfffffffcu;
  compiled_vector_extended_state.vgprs[3][3] = 16u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_extended_program,
                                         &compiled_vector_extended_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_extended_state.halted,
              "expected compiled vector extended program to halt") ||
      !Expect(compiled_vector_extended_state.vgprs[10][0] == 0xfffffff0u,
              "expected compiled v_min_i32 lane 0 result") ||
      !Expect(compiled_vector_extended_state.vgprs[11][1] == 0xfffffff6u,
              "expected compiled v_max_i32 lane 1 result") ||
      !Expect(compiled_vector_extended_state.vgprs[12][1] == 15u,
              "expected compiled v_min_u32 lane 1 result") ||
      !Expect(compiled_vector_extended_state.vgprs[13][1] == 20u,
              "expected compiled v_max_u32 lane 1 result") ||
      !Expect(compiled_vector_extended_state.vgprs[14][3] == 64u,
              "expected compiled v_lshlrev_b32 lane 3 result") ||
      !Expect(compiled_vector_extended_state.vgprs[15][0] == 0x3ffffffeu,
              "expected compiled v_lshrrev_b32 lane 0 result") ||
      !Expect(compiled_vector_extended_state.vgprs[16][1] == 0xffffffffu,
              "expected compiled v_ashrrev_i32 lane 1 result") ||
      !Expect(compiled_vector_extended_state.vgprs[17][1] == 4u,
              "expected compiled v_and_b32 lane 1 result") ||
      !Expect(compiled_vector_extended_state.vgprs[18][3] == 15u,
              "expected compiled v_or_b32 lane 3 result") ||
      !Expect(compiled_vector_extended_state.vgprs[19][1] == 27u,
              "expected compiled v_xor_b32 lane 1 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_unary_program = {
      DecodedInstruction::Unary("V_NOT_B32", InstructionOperand::Vgpr(25),
                                InstructionOperand::Sgpr(4)),
      DecodedInstruction::Unary("V_BFREV_B32", InstructionOperand::Vgpr(26),
                                InstructionOperand::Sgpr(5)),
      DecodedInstruction::Unary("V_FFBH_U32", InstructionOperand::Vgpr(27),
                                InstructionOperand::Sgpr(6)),
      DecodedInstruction::Unary("V_FFBL_B32", InstructionOperand::Vgpr(28),
                                InstructionOperand::Sgpr(6)),
      DecodedInstruction::Unary("V_FFBH_I32", InstructionOperand::Vgpr(29),
                                InstructionOperand::Sgpr(7)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_unary_state;
  vector_unary_state.exec_mask = 0b1011ULL;
  vector_unary_state.sgprs[4] = 0x0f0f0000u;
  vector_unary_state.sgprs[5] = 0x0000000fu;
  vector_unary_state.sgprs[6] = 0x00f00000u;
  vector_unary_state.sgprs[7] = 0xffff0000u;
  vector_unary_state.vgprs[25][2] = 0xdeadbeefu;
  vector_unary_state.vgprs[26][2] = 0xdeadbeefu;
  vector_unary_state.vgprs[27][2] = 0xdeadbeefu;
  vector_unary_state.vgprs[28][2] = 0xdeadbeefu;
  vector_unary_state.vgprs[29][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_unary_program,
                                         &vector_unary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_unary_state.halted,
              "expected vector unary program to halt") ||
      !Expect(vector_unary_state.vgprs[25][0] == 0xf0f0ffffu,
              "expected v_not_b32 lane 0 result") ||
      !Expect(vector_unary_state.vgprs[25][2] == 0xdeadbeefu,
              "expected inactive lane v_not_b32 result") ||
      !Expect(vector_unary_state.vgprs[26][1] == 0xf0000000u,
              "expected v_bfrev_b32 lane 1 result") ||
      !Expect(vector_unary_state.vgprs[27][3] == 8u,
              "expected v_ffbh_u32 lane 3 result") ||
      !Expect(vector_unary_state.vgprs[28][0] == 20u,
              "expected v_ffbl_b32 lane 0 result") ||
      !Expect(vector_unary_state.vgprs[29][1] == 16u,
              "expected v_ffbh_i32 lane 1 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_unary_program;
  if (!Expect(interpreter.CompileProgram(vector_unary_program,
                                         &compiled_vector_unary_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_unary_state;
  compiled_vector_unary_state.exec_mask = 0b1011ULL;
  compiled_vector_unary_state.sgprs[4] = 0x0f0f0000u;
  compiled_vector_unary_state.sgprs[5] = 0x0000000fu;
  compiled_vector_unary_state.sgprs[6] = 0x00f00000u;
  compiled_vector_unary_state.sgprs[7] = 0xffff0000u;
  compiled_vector_unary_state.vgprs[25][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_unary_program,
                                         &compiled_vector_unary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_unary_state.halted,
              "expected compiled vector unary program to halt") ||
      !Expect(compiled_vector_unary_state.vgprs[25][0] == 0xf0f0ffffu,
              "expected compiled v_not_b32 lane 0 result") ||
      !Expect(compiled_vector_unary_state.vgprs[26][1] == 0xf0000000u,
              "expected compiled v_bfrev_b32 lane 1 result") ||
      !Expect(compiled_vector_unary_state.vgprs[27][3] == 8u,
              "expected compiled v_ffbh_u32 lane 3 result") ||
      !Expect(compiled_vector_unary_state.vgprs[28][0] == 20u,
              "expected compiled v_ffbl_b32 lane 0 result") ||
      !Expect(compiled_vector_unary_state.vgprs[29][1] == 16u,
              "expected compiled v_ffbh_i32 lane 1 result") ||
      !Expect(compiled_vector_unary_state.vgprs[25][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_not_b32 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_conversion_program = {
      DecodedInstruction::Unary("V_CVT_F32_I32", InstructionOperand::Vgpr(60),
                                InstructionOperand::Sgpr(90)),
      DecodedInstruction::Unary("V_CVT_F32_U32", InstructionOperand::Vgpr(61),
                                InstructionOperand::Sgpr(91)),
      DecodedInstruction::Unary("V_CVT_I32_F32", InstructionOperand::Vgpr(62),
                                InstructionOperand::Vgpr(70)),
      DecodedInstruction::Unary("V_CVT_U32_F32", InstructionOperand::Vgpr(63),
                                InstructionOperand::Vgpr(71)),
      DecodedInstruction::Unary("V_CVT_F16_F32", InstructionOperand::Vgpr(64),
                                InstructionOperand::Vgpr(72)),
      DecodedInstruction::Unary("V_CVT_F32_F16", InstructionOperand::Vgpr(65),
                                InstructionOperand::Vgpr(73)),
      DecodedInstruction::Unary("V_CVT_F64_F32", InstructionOperand::Vgpr(66),
                                InstructionOperand::Sgpr(92)),
      DecodedInstruction::Unary("V_CVT_F32_F64", InstructionOperand::Vgpr(68),
                                InstructionOperand::Vgpr(74)),
      DecodedInstruction::Unary("V_CVT_I32_F64", InstructionOperand::Vgpr(84),
                                InstructionOperand::Vgpr(74)),
      DecodedInstruction::Unary("V_CVT_U32_F64", InstructionOperand::Vgpr(85),
                                InstructionOperand::Vgpr(82)),
      DecodedInstruction::Unary("V_CVT_F64_I32", InstructionOperand::Vgpr(86),
                                InstructionOperand::Sgpr(93)),
      DecodedInstruction::Unary("V_CVT_F64_U32", InstructionOperand::Vgpr(88),
                                InstructionOperand::Sgpr(94)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_conversion_state;
  vector_conversion_state.exec_mask = 0b1011ULL;
  vector_conversion_state.sgprs[90] = static_cast<std::uint32_t>(-7);
  vector_conversion_state.sgprs[91] = 9u;
  vector_conversion_state.sgprs[92] = FloatBits(-1.25f);
  vector_conversion_state.sgprs[93] = static_cast<std::uint32_t>(-11);
  vector_conversion_state.sgprs[94] = 12u;
  vector_conversion_state.vgprs[70][0] = FloatBits(3.75f);
  vector_conversion_state.vgprs[70][1] = FloatBits(-2.75f);
  vector_conversion_state.vgprs[70][3] = FloatBits(-0.5f);
  vector_conversion_state.vgprs[71][0] = FloatBits(7.75f);
  vector_conversion_state.vgprs[71][1] = FloatBits(1.0f);
  vector_conversion_state.vgprs[71][3] = FloatBits(0.5f);
  vector_conversion_state.vgprs[72][0] = FloatBits(1.5f);
  vector_conversion_state.vgprs[72][1] = FloatBits(-2.0f);
  vector_conversion_state.vgprs[72][3] = FloatBits(0.5f);
  vector_conversion_state.vgprs[73][0] = 0x00003e00u;
  vector_conversion_state.vgprs[73][1] = 0x0000c000u;
  vector_conversion_state.vgprs[73][3] = 0x00003800u;
  SplitU64(DoubleBits(2.5), &vector_conversion_state.vgprs[74][0],
           &vector_conversion_state.vgprs[75][0]);
  SplitU64(DoubleBits(-0.25), &vector_conversion_state.vgprs[74][1],
           &vector_conversion_state.vgprs[75][1]);
  SplitU64(DoubleBits(8.0), &vector_conversion_state.vgprs[74][3],
           &vector_conversion_state.vgprs[75][3]);
  SplitU64(DoubleBits(9.5), &vector_conversion_state.vgprs[82][0],
           &vector_conversion_state.vgprs[83][0]);
  SplitU64(DoubleBits(1.0), &vector_conversion_state.vgprs[82][1],
           &vector_conversion_state.vgprs[83][1]);
  SplitU64(DoubleBits(0.5), &vector_conversion_state.vgprs[82][3],
           &vector_conversion_state.vgprs[83][3]);
  vector_conversion_state.vgprs[60][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[61][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[62][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[63][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[64][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[65][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[66][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[67][2] = 0xcafebabeu;
  vector_conversion_state.vgprs[68][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[84][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[85][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[86][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[87][2] = 0xcafebabeu;
  vector_conversion_state.vgprs[88][2] = 0xdeadbeefu;
  vector_conversion_state.vgprs[89][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(vector_conversion_program,
                                         &vector_conversion_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_conversion_state.halted,
              "expected vector conversion program to halt") ||
      !Expect(vector_conversion_state.vgprs[60][0] == FloatBits(-7.0f),
              "expected decoded v_cvt_f32_i32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[60][1] == FloatBits(-7.0f),
              "expected decoded v_cvt_f32_i32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[60][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_i32 result") ||
      !Expect(vector_conversion_state.vgprs[60][3] == FloatBits(-7.0f),
              "expected decoded v_cvt_f32_i32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[61][0] == FloatBits(9.0f),
              "expected decoded v_cvt_f32_u32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[61][1] == FloatBits(9.0f),
              "expected decoded v_cvt_f32_u32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[61][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_u32 result") ||
      !Expect(vector_conversion_state.vgprs[61][3] == FloatBits(9.0f),
              "expected decoded v_cvt_f32_u32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[62][0] == 3u,
              "expected decoded v_cvt_i32_f32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[62][1] == static_cast<std::uint32_t>(-2),
              "expected decoded v_cvt_i32_f32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[62][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_i32_f32 result") ||
      !Expect(vector_conversion_state.vgprs[62][3] == 0u,
              "expected decoded v_cvt_i32_f32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[63][0] == 7u,
              "expected decoded v_cvt_u32_f32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[63][1] == 1u,
              "expected decoded v_cvt_u32_f32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[63][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_u32_f32 result") ||
      !Expect(vector_conversion_state.vgprs[63][3] == 0u,
              "expected decoded v_cvt_u32_f32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[64][0] == 0x00003e00u,
              "expected decoded v_cvt_f16_f32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[64][1] == 0x0000c000u,
              "expected decoded v_cvt_f16_f32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[64][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f16_f32 result") ||
      !Expect(vector_conversion_state.vgprs[64][3] == 0x00003800u,
              "expected decoded v_cvt_f16_f32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[65][0] == FloatBits(1.5f),
              "expected decoded v_cvt_f32_f16 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[65][1] == FloatBits(-2.0f),
              "expected decoded v_cvt_f32_f16 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[65][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_f16 result") ||
      !Expect(vector_conversion_state.vgprs[65][3] == FloatBits(0.5f),
              "expected decoded v_cvt_f32_f16 lane 3 result") ||
      !Expect(ReadLane0VgprU64(vector_conversion_state, 66) == DoubleBits(-1.25),
              "expected decoded v_cvt_f64_f32 lane 0 result") ||
      !Expect(ComposeU64(vector_conversion_state.vgprs[66][1],
                         vector_conversion_state.vgprs[67][1]) ==
                  DoubleBits(-1.25),
              "expected decoded v_cvt_f64_f32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[66][2] == 0xdeadbeefu &&
                  vector_conversion_state.vgprs[67][2] == 0xcafebabeu,
              "expected inactive decoded v_cvt_f64_f32 result") ||
      !Expect(ComposeU64(vector_conversion_state.vgprs[66][3],
                         vector_conversion_state.vgprs[67][3]) ==
                  DoubleBits(-1.25),
              "expected decoded v_cvt_f64_f32 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[68][0] == FloatBits(2.5f),
              "expected decoded v_cvt_f32_f64 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[68][1] == FloatBits(-0.25f),
              "expected decoded v_cvt_f32_f64 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[68][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_f64 result") ||
      !Expect(vector_conversion_state.vgprs[68][3] == FloatBits(8.0f),
              "expected decoded v_cvt_f32_f64 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[84][0] == 2u,
              "expected decoded v_cvt_i32_f64 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[84][1] == 0u,
              "expected decoded v_cvt_i32_f64 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[84][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_i32_f64 result") ||
      !Expect(vector_conversion_state.vgprs[84][3] == 8u,
              "expected decoded v_cvt_i32_f64 lane 3 result") ||
      !Expect(vector_conversion_state.vgprs[85][0] == 9u,
              "expected decoded v_cvt_u32_f64 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[85][1] == 1u,
              "expected decoded v_cvt_u32_f64 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[85][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_u32_f64 result") ||
      !Expect(vector_conversion_state.vgprs[85][3] == 0u,
              "expected decoded v_cvt_u32_f64 lane 3 result") ||
      !Expect(ComposeU64(vector_conversion_state.vgprs[86][0],
                         vector_conversion_state.vgprs[87][0]) ==
                  DoubleBits(-11.0),
              "expected decoded v_cvt_f64_i32 lane 0 result") ||
      !Expect(vector_conversion_state.vgprs[86][2] == 0xdeadbeefu &&
                  vector_conversion_state.vgprs[87][2] == 0xcafebabeu,
              "expected inactive decoded v_cvt_f64_i32 result") ||
      !Expect(ComposeU64(vector_conversion_state.vgprs[88][1],
                         vector_conversion_state.vgprs[89][1]) ==
                  DoubleBits(12.0),
              "expected decoded v_cvt_f64_u32 lane 1 result") ||
      !Expect(vector_conversion_state.vgprs[88][2] == 0xdeadbeefu &&
                  vector_conversion_state.vgprs[89][2] == 0xcafebabeu,
              "expected inactive decoded v_cvt_f64_u32 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_conversion_program;
  if (!Expect(interpreter.CompileProgram(vector_conversion_program,
                                         &compiled_vector_conversion_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_conversion_state;
  compiled_vector_conversion_state.exec_mask = 0b1011ULL;
  compiled_vector_conversion_state.sgprs[90] = static_cast<std::uint32_t>(-7);
  compiled_vector_conversion_state.sgprs[91] = 9u;
  compiled_vector_conversion_state.sgprs[92] = FloatBits(-1.25f);
  compiled_vector_conversion_state.sgprs[93] = static_cast<std::uint32_t>(-11);
  compiled_vector_conversion_state.sgprs[94] = 12u;
  compiled_vector_conversion_state.vgprs[70][0] = FloatBits(3.75f);
  compiled_vector_conversion_state.vgprs[70][1] = FloatBits(-2.75f);
  compiled_vector_conversion_state.vgprs[70][3] = FloatBits(-0.5f);
  compiled_vector_conversion_state.vgprs[71][0] = FloatBits(7.75f);
  compiled_vector_conversion_state.vgprs[71][1] = FloatBits(1.0f);
  compiled_vector_conversion_state.vgprs[71][3] = FloatBits(0.5f);
  compiled_vector_conversion_state.vgprs[72][0] = FloatBits(1.5f);
  compiled_vector_conversion_state.vgprs[72][1] = FloatBits(-2.0f);
  compiled_vector_conversion_state.vgprs[72][3] = FloatBits(0.5f);
  compiled_vector_conversion_state.vgprs[73][0] = 0x00003e00u;
  compiled_vector_conversion_state.vgprs[73][1] = 0x0000c000u;
  compiled_vector_conversion_state.vgprs[73][3] = 0x00003800u;
  SplitU64(DoubleBits(2.5), &compiled_vector_conversion_state.vgprs[74][0],
           &compiled_vector_conversion_state.vgprs[75][0]);
  SplitU64(DoubleBits(-0.25), &compiled_vector_conversion_state.vgprs[74][1],
           &compiled_vector_conversion_state.vgprs[75][1]);
  SplitU64(DoubleBits(8.0), &compiled_vector_conversion_state.vgprs[74][3],
           &compiled_vector_conversion_state.vgprs[75][3]);
  SplitU64(DoubleBits(9.5), &compiled_vector_conversion_state.vgprs[82][0],
           &compiled_vector_conversion_state.vgprs[83][0]);
  SplitU64(DoubleBits(1.0), &compiled_vector_conversion_state.vgprs[82][1],
           &compiled_vector_conversion_state.vgprs[83][1]);
  SplitU64(DoubleBits(0.5), &compiled_vector_conversion_state.vgprs[82][3],
           &compiled_vector_conversion_state.vgprs[83][3]);
  compiled_vector_conversion_state.vgprs[60][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[61][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[62][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[63][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[64][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[65][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[66][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[67][2] = 0xcafebabeu;
  compiled_vector_conversion_state.vgprs[68][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[84][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[85][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[86][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[87][2] = 0xcafebabeu;
  compiled_vector_conversion_state.vgprs[88][2] = 0xdeadbeefu;
  compiled_vector_conversion_state.vgprs[89][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_conversion_program,
                                         &compiled_vector_conversion_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_conversion_state.halted,
              "expected compiled vector conversion program to halt") ||
      !Expect(compiled_vector_conversion_state.vgprs[60][0] == FloatBits(-7.0f),
              "expected compiled v_cvt_f32_i32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[60][1] == FloatBits(-7.0f),
              "expected compiled v_cvt_f32_i32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[60][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_i32 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[60][3] == FloatBits(-7.0f),
              "expected compiled v_cvt_f32_i32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[61][0] == FloatBits(9.0f),
              "expected compiled v_cvt_f32_u32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[61][1] == FloatBits(9.0f),
              "expected compiled v_cvt_f32_u32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[61][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_u32 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[61][3] == FloatBits(9.0f),
              "expected compiled v_cvt_f32_u32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[62][0] == 3u,
              "expected compiled v_cvt_i32_f32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[62][1] ==
                  static_cast<std::uint32_t>(-2),
              "expected compiled v_cvt_i32_f32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[62][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_i32_f32 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[62][3] == 0u,
              "expected compiled v_cvt_i32_f32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[63][0] == 7u,
              "expected compiled v_cvt_u32_f32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[63][1] == 1u,
              "expected compiled v_cvt_u32_f32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[63][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_u32_f32 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[63][3] == 0u,
              "expected compiled v_cvt_u32_f32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[64][0] == 0x00003e00u,
              "expected compiled v_cvt_f16_f32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[64][1] == 0x0000c000u,
              "expected compiled v_cvt_f16_f32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[64][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_f16_f32 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[64][3] == 0x00003800u,
              "expected compiled v_cvt_f16_f32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[65][0] == FloatBits(1.5f),
              "expected compiled v_cvt_f32_f16 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[65][1] == FloatBits(-2.0f),
              "expected compiled v_cvt_f32_f16 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[65][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_f16 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[65][3] == FloatBits(0.5f),
              "expected compiled v_cvt_f32_f16 lane 3 result") ||
      !Expect(ReadLane0VgprU64(compiled_vector_conversion_state, 66) ==
                  DoubleBits(-1.25),
              "expected compiled v_cvt_f64_f32 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_conversion_state.vgprs[66][1],
                         compiled_vector_conversion_state.vgprs[67][1]) ==
                  DoubleBits(-1.25),
              "expected compiled v_cvt_f64_f32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[66][2] == 0xdeadbeefu &&
                  compiled_vector_conversion_state.vgprs[67][2] == 0xcafebabeu,
              "expected inactive compiled v_cvt_f64_f32 result") ||
      !Expect(ComposeU64(compiled_vector_conversion_state.vgprs[66][3],
                         compiled_vector_conversion_state.vgprs[67][3]) ==
                  DoubleBits(-1.25),
              "expected compiled v_cvt_f64_f32 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[68][0] == FloatBits(2.5f),
              "expected compiled v_cvt_f32_f64 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[68][1] ==
                  FloatBits(-0.25f),
              "expected compiled v_cvt_f32_f64 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[68][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_f64 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[68][3] == FloatBits(8.0f),
              "expected compiled v_cvt_f32_f64 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[84][0] == 2u,
              "expected compiled v_cvt_i32_f64 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[84][1] == 0u,
              "expected compiled v_cvt_i32_f64 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[84][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_i32_f64 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[84][3] == 8u,
              "expected compiled v_cvt_i32_f64 lane 3 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[85][0] == 9u,
              "expected compiled v_cvt_u32_f64 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[85][1] == 1u,
              "expected compiled v_cvt_u32_f64 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[85][2] == 0xdeadbeefu,
              "expected inactive compiled v_cvt_u32_f64 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[85][3] == 0u,
              "expected compiled v_cvt_u32_f64 lane 3 result") ||
      !Expect(ComposeU64(compiled_vector_conversion_state.vgprs[86][0],
                         compiled_vector_conversion_state.vgprs[87][0]) ==
                  DoubleBits(-11.0),
              "expected compiled v_cvt_f64_i32 lane 0 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[86][2] == 0xdeadbeefu &&
                  compiled_vector_conversion_state.vgprs[87][2] == 0xcafebabeu,
              "expected inactive compiled v_cvt_f64_i32 result") ||
      !Expect(ComposeU64(compiled_vector_conversion_state.vgprs[88][1],
                         compiled_vector_conversion_state.vgprs[89][1]) ==
                  DoubleBits(12.0),
              "expected compiled v_cvt_f64_u32 lane 1 result") ||
      !Expect(compiled_vector_conversion_state.vgprs[88][2] == 0xdeadbeefu &&
                  compiled_vector_conversion_state.vgprs[89][2] == 0xcafebabeu,
              "expected inactive compiled v_cvt_f64_u32 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_byte_conversion_program = {
      DecodedInstruction::Nullary("V_NOP"),
      DecodedInstruction::Unary("V_CVT_F16_U16", InstructionOperand::Vgpr(96),
                                InstructionOperand::Vgpr(90)),
      DecodedInstruction::Unary("V_CVT_F32_UBYTE0",
                                InstructionOperand::Vgpr(97),
                                InstructionOperand::Vgpr(91)),
      DecodedInstruction::Unary("V_CVT_F32_UBYTE1",
                                InstructionOperand::Vgpr(98),
                                InstructionOperand::Vgpr(91)),
      DecodedInstruction::Unary("V_CVT_F32_UBYTE2",
                                InstructionOperand::Vgpr(99),
                                InstructionOperand::Vgpr(91)),
      DecodedInstruction::Unary("V_CVT_F32_UBYTE3",
                                InstructionOperand::Vgpr(100),
                                InstructionOperand::Vgpr(91)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_byte_conversion_state;
  vector_byte_conversion_state.exec_mask = 0b1011ULL;
  vector_byte_conversion_state.vgprs[90][0] = 1u;
  vector_byte_conversion_state.vgprs[90][1] = 2u;
  vector_byte_conversion_state.vgprs[90][3] = 3u;
  vector_byte_conversion_state.vgprs[91][0] = 0x44332211u;
  vector_byte_conversion_state.vgprs[91][1] = 0xaabbccddu;
  vector_byte_conversion_state.vgprs[91][3] = 0x01020304u;
  vector_byte_conversion_state.vgprs[96][2] = 0xdeadbeefu;
  vector_byte_conversion_state.vgprs[97][2] = 0xdeadbeefu;
  vector_byte_conversion_state.vgprs[98][2] = 0xdeadbeefu;
  vector_byte_conversion_state.vgprs[99][2] = 0xdeadbeefu;
  vector_byte_conversion_state.vgprs[100][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_byte_conversion_program,
                                         &vector_byte_conversion_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_byte_conversion_state.halted,
              "expected decoded vector byte conversion program to halt") ||
      !Expect(vector_byte_conversion_state.vgprs[96][0] == 0x00003c00u,
              "expected decoded v_cvt_f16_u16 lane 0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[96][1] == 0x00004000u,
              "expected decoded v_cvt_f16_u16 lane 1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[96][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f16_u16 result") ||
      !Expect(vector_byte_conversion_state.vgprs[96][3] == 0x00004200u,
              "expected decoded v_cvt_f16_u16 lane 3 result") ||
      !Expect(vector_byte_conversion_state.vgprs[97][0] == FloatBits(17.0f),
              "expected decoded v_cvt_f32_ubyte0 lane 0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[97][1] == FloatBits(221.0f),
              "expected decoded v_cvt_f32_ubyte0 lane 1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[97][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_ubyte0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[97][3] == FloatBits(4.0f),
              "expected decoded v_cvt_f32_ubyte0 lane 3 result") ||
      !Expect(vector_byte_conversion_state.vgprs[98][0] == FloatBits(34.0f),
              "expected decoded v_cvt_f32_ubyte1 lane 0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[98][1] == FloatBits(204.0f),
              "expected decoded v_cvt_f32_ubyte1 lane 1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[98][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_ubyte1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[98][3] == FloatBits(3.0f),
              "expected decoded v_cvt_f32_ubyte1 lane 3 result") ||
      !Expect(vector_byte_conversion_state.vgprs[99][0] == FloatBits(51.0f),
              "expected decoded v_cvt_f32_ubyte2 lane 0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[99][1] == FloatBits(187.0f),
              "expected decoded v_cvt_f32_ubyte2 lane 1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[99][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_ubyte2 result") ||
      !Expect(vector_byte_conversion_state.vgprs[99][3] == FloatBits(2.0f),
              "expected decoded v_cvt_f32_ubyte2 lane 3 result") ||
      !Expect(vector_byte_conversion_state.vgprs[100][0] == FloatBits(68.0f),
              "expected decoded v_cvt_f32_ubyte3 lane 0 result") ||
      !Expect(vector_byte_conversion_state.vgprs[100][1] == FloatBits(170.0f),
              "expected decoded v_cvt_f32_ubyte3 lane 1 result") ||
      !Expect(vector_byte_conversion_state.vgprs[100][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f32_ubyte3 result") ||
      !Expect(vector_byte_conversion_state.vgprs[100][3] == FloatBits(1.0f),
              "expected decoded v_cvt_f32_ubyte3 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_byte_conversion_program;
  if (!Expect(interpreter.CompileProgram(vector_byte_conversion_program,
                                         &compiled_vector_byte_conversion_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_byte_conversion_state;
  compiled_vector_byte_conversion_state.exec_mask = 0b1011ULL;
  compiled_vector_byte_conversion_state.vgprs[90][0] = 1u;
  compiled_vector_byte_conversion_state.vgprs[90][1] = 2u;
  compiled_vector_byte_conversion_state.vgprs[90][3] = 3u;
  compiled_vector_byte_conversion_state.vgprs[91][0] = 0x44332211u;
  compiled_vector_byte_conversion_state.vgprs[91][1] = 0xaabbccddu;
  compiled_vector_byte_conversion_state.vgprs[91][3] = 0x01020304u;
  compiled_vector_byte_conversion_state.vgprs[96][2] = 0xdeadbeefu;
  compiled_vector_byte_conversion_state.vgprs[97][2] = 0xdeadbeefu;
  compiled_vector_byte_conversion_state.vgprs[98][2] = 0xdeadbeefu;
  compiled_vector_byte_conversion_state.vgprs[99][2] = 0xdeadbeefu;
  compiled_vector_byte_conversion_state.vgprs[100][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(
                  compiled_vector_byte_conversion_program,
                  &compiled_vector_byte_conversion_state, &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_byte_conversion_state.halted,
              "expected compiled vector byte conversion program to halt") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[96][0] == 0x00003c00u,
              "expected compiled v_cvt_f16_u16 lane 0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[96][1] == 0x00004000u,
              "expected compiled v_cvt_f16_u16 lane 1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[96][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f16_u16 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[96][3] == 0x00004200u,
              "expected compiled v_cvt_f16_u16 lane 3 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[97][0] ==
                  FloatBits(17.0f),
              "expected compiled v_cvt_f32_ubyte0 lane 0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[97][1] ==
                  FloatBits(221.0f),
              "expected compiled v_cvt_f32_ubyte0 lane 1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[97][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_ubyte0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[97][3] ==
                  FloatBits(4.0f),
              "expected compiled v_cvt_f32_ubyte0 lane 3 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[98][0] ==
                  FloatBits(34.0f),
              "expected compiled v_cvt_f32_ubyte1 lane 0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[98][1] ==
                  FloatBits(204.0f),
              "expected compiled v_cvt_f32_ubyte1 lane 1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[98][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_ubyte1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[98][3] ==
                  FloatBits(3.0f),
              "expected compiled v_cvt_f32_ubyte1 lane 3 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[99][0] ==
                  FloatBits(51.0f),
              "expected compiled v_cvt_f32_ubyte2 lane 0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[99][1] ==
                  FloatBits(187.0f),
              "expected compiled v_cvt_f32_ubyte2 lane 1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[99][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_ubyte2 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[99][3] ==
                  FloatBits(2.0f),
              "expected compiled v_cvt_f32_ubyte2 lane 3 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[100][0] ==
                  FloatBits(68.0f),
              "expected compiled v_cvt_f32_ubyte3 lane 0 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[100][1] ==
                  FloatBits(170.0f),
              "expected compiled v_cvt_f32_ubyte3 lane 1 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[100][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f32_ubyte3 result") ||
      !Expect(compiled_vector_byte_conversion_state.vgprs[100][3] ==
                  FloatBits(1.0f),
              "expected compiled v_cvt_f32_ubyte3 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_half_int_conversion_program = {
      DecodedInstruction::Unary("V_CVT_F16_I16", InstructionOperand::Vgpr(101),
                                InstructionOperand::Vgpr(92)),
      DecodedInstruction::Unary("V_CVT_U16_F16", InstructionOperand::Vgpr(102),
                                InstructionOperand::Vgpr(93)),
      DecodedInstruction::Unary("V_CVT_I16_F16", InstructionOperand::Vgpr(103),
                                InstructionOperand::Vgpr(94)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  auto vector_half_int_conversion_state = std::make_unique<WaveExecutionState>();
  vector_half_int_conversion_state->exec_mask = 0b1011ULL;
  vector_half_int_conversion_state->vgprs[92][0] = 0xffffu;
  vector_half_int_conversion_state->vgprs[92][1] = 0x0002u;
  vector_half_int_conversion_state->vgprs[92][3] = 0xfffdu;
  vector_half_int_conversion_state->vgprs[93][0] = 0x00003e00u;
  vector_half_int_conversion_state->vgprs[93][1] = 0x00004000u;
  vector_half_int_conversion_state->vgprs[93][3] = 0x00004300u;
  vector_half_int_conversion_state->vgprs[94][0] = 0x0000be00u;
  vector_half_int_conversion_state->vgprs[94][1] = 0x00004000u;
  vector_half_int_conversion_state->vgprs[94][3] = 0x0000c200u;
  vector_half_int_conversion_state->vgprs[101][2] = 0xdeadbeefu;
  vector_half_int_conversion_state->vgprs[102][2] = 0xdeadbeefu;
  vector_half_int_conversion_state->vgprs[103][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_half_int_conversion_program,
                                         vector_half_int_conversion_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_half_int_conversion_state->halted,
              "expected decoded vector half/int conversion program to halt") ||
      !Expect(vector_half_int_conversion_state->vgprs[101][0] == 0x0000bc00u,
              "expected decoded v_cvt_f16_i16 lane 0 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[101][1] == 0x00004000u,
              "expected decoded v_cvt_f16_i16 lane 1 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[101][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_f16_i16 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[101][3] == 0x0000c200u,
              "expected decoded v_cvt_f16_i16 lane 3 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[102][0] == 1u,
              "expected decoded v_cvt_u16_f16 lane 0 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[102][1] == 2u,
              "expected decoded v_cvt_u16_f16 lane 1 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[102][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_u16_f16 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[102][3] == 3u,
              "expected decoded v_cvt_u16_f16 lane 3 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[103][0] == 0x0000ffffu,
              "expected decoded v_cvt_i16_f16 lane 0 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[103][1] == 0x00000002u,
              "expected decoded v_cvt_i16_f16 lane 1 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[103][2] == 0xdeadbeefu,
              "expected inactive decoded v_cvt_i16_f16 result") ||
      !Expect(vector_half_int_conversion_state->vgprs[103][3] == 0x0000fffdu,
              "expected decoded v_cvt_i16_f16 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_half_int_conversion_program;
  if (!Expect(interpreter.CompileProgram(vector_half_int_conversion_program,
                                         &compiled_vector_half_int_conversion_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  auto compiled_vector_half_int_conversion_state =
      std::make_unique<WaveExecutionState>();
  compiled_vector_half_int_conversion_state->exec_mask = 0b1011ULL;
  compiled_vector_half_int_conversion_state->vgprs[92][0] = 0xffffu;
  compiled_vector_half_int_conversion_state->vgprs[92][1] = 0x0002u;
  compiled_vector_half_int_conversion_state->vgprs[92][3] = 0xfffdu;
  compiled_vector_half_int_conversion_state->vgprs[93][0] = 0x00003e00u;
  compiled_vector_half_int_conversion_state->vgprs[93][1] = 0x00004000u;
  compiled_vector_half_int_conversion_state->vgprs[93][3] = 0x00004300u;
  compiled_vector_half_int_conversion_state->vgprs[94][0] = 0x0000be00u;
  compiled_vector_half_int_conversion_state->vgprs[94][1] = 0x00004000u;
  compiled_vector_half_int_conversion_state->vgprs[94][3] = 0x0000c200u;
  compiled_vector_half_int_conversion_state->vgprs[101][2] = 0xdeadbeefu;
  compiled_vector_half_int_conversion_state->vgprs[102][2] = 0xdeadbeefu;
  compiled_vector_half_int_conversion_state->vgprs[103][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(
                  compiled_vector_half_int_conversion_program,
                  compiled_vector_half_int_conversion_state.get(),
                  &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_half_int_conversion_state->halted,
              "expected compiled vector half/int conversion program to halt") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[101][0] ==
                  0x0000bc00u,
              "expected compiled v_cvt_f16_i16 lane 0 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[101][1] ==
                  0x00004000u,
              "expected compiled v_cvt_f16_i16 lane 1 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[101][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_f16_i16 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[101][3] ==
                  0x0000c200u,
              "expected compiled v_cvt_f16_i16 lane 3 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[102][0] == 1u,
              "expected compiled v_cvt_u16_f16 lane 0 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[102][1] == 2u,
              "expected compiled v_cvt_u16_f16 lane 1 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[102][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_u16_f16 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[102][3] == 3u,
              "expected compiled v_cvt_u16_f16 lane 3 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[103][0] ==
                  0x0000ffffu,
              "expected compiled v_cvt_i16_f16 lane 0 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[103][1] ==
                  0x00000002u,
              "expected compiled v_cvt_i16_f16 lane 1 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[103][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_cvt_i16_f16 result") ||
      !Expect(compiled_vector_half_int_conversion_state->vgprs[103][3] ==
                  0x0000fffdu,
              "expected compiled v_cvt_i16_f16 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_sat_pk_program = {
      DecodedInstruction::Unary("V_SAT_PK_U8_I16", InstructionOperand::Vgpr(106),
                                InstructionOperand::Vgpr(97)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  auto vector_sat_pk_state = std::make_unique<WaveExecutionState>();
  vector_sat_pk_state->exec_mask = 0b1011ULL;
  vector_sat_pk_state->vgprs[97][0] = 0x007f0100u;
  vector_sat_pk_state->vgprs[97][1] = 0xffff0001u;
  vector_sat_pk_state->vgprs[97][3] = 0x12340080u;
  vector_sat_pk_state->vgprs[106][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_sat_pk_program,
                                         vector_sat_pk_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_sat_pk_state->halted,
              "expected decoded v_sat_pk_u8_i16 program to halt") ||
      !Expect(vector_sat_pk_state->vgprs[106][0] == 0x00007fffu,
              "expected decoded v_sat_pk_u8_i16 lane 0 result") ||
      !Expect(vector_sat_pk_state->vgprs[106][1] == 0x00000001u,
              "expected decoded v_sat_pk_u8_i16 lane 1 result") ||
      !Expect(vector_sat_pk_state->vgprs[106][2] == 0xdeadbeefu,
              "expected inactive decoded v_sat_pk_u8_i16 result") ||
      !Expect(vector_sat_pk_state->vgprs[106][3] == 0x0000ff80u,
              "expected decoded v_sat_pk_u8_i16 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_sat_pk_program;
  if (!Expect(interpreter.CompileProgram(vector_sat_pk_program,
                                         &compiled_vector_sat_pk_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  auto compiled_vector_sat_pk_state = std::make_unique<WaveExecutionState>();
  compiled_vector_sat_pk_state->exec_mask = 0b1011ULL;
  compiled_vector_sat_pk_state->vgprs[97][0] = 0x007f0100u;
  compiled_vector_sat_pk_state->vgprs[97][1] = 0xffff0001u;
  compiled_vector_sat_pk_state->vgprs[97][3] = 0x12340080u;
  compiled_vector_sat_pk_state->vgprs[106][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_sat_pk_program,
                                         compiled_vector_sat_pk_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_sat_pk_state->halted,
              "expected compiled v_sat_pk_u8_i16 program to halt") ||
      !Expect(compiled_vector_sat_pk_state->vgprs[106][0] == 0x00007fffu,
              "expected compiled v_sat_pk_u8_i16 lane 0 result") ||
      !Expect(compiled_vector_sat_pk_state->vgprs[106][1] == 0x00000001u,
              "expected compiled v_sat_pk_u8_i16 lane 1 result") ||
      !Expect(compiled_vector_sat_pk_state->vgprs[106][2] == 0xdeadbeefu,
              "expected inactive compiled v_sat_pk_u8_i16 result") ||
      !Expect(compiled_vector_sat_pk_state->vgprs[106][3] == 0x0000ff80u,
              "expected compiled v_sat_pk_u8_i16 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_legacy_float_math_program = {
      DecodedInstruction::Unary("V_EXP_LEGACY_F32", InstructionOperand::Vgpr(104),
                                InstructionOperand::Vgpr(95)),
      DecodedInstruction::Unary("V_LOG_LEGACY_F32", InstructionOperand::Vgpr(105),
                                InstructionOperand::Vgpr(96)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  auto vector_legacy_float_math_state = std::make_unique<WaveExecutionState>();
  vector_legacy_float_math_state->exec_mask = 0b1011ULL;
  vector_legacy_float_math_state->vgprs[95][0] = FloatBits(1.0f);
  vector_legacy_float_math_state->vgprs[95][1] = FloatBits(2.0f);
  vector_legacy_float_math_state->vgprs[95][3] = FloatBits(-1.0f);
  vector_legacy_float_math_state->vgprs[96][0] = FloatBits(1.0f);
  vector_legacy_float_math_state->vgprs[96][1] = FloatBits(8.0f);
  vector_legacy_float_math_state->vgprs[96][3] = FloatBits(0.5f);
  vector_legacy_float_math_state->vgprs[104][2] = 0xdeadbeefu;
  vector_legacy_float_math_state->vgprs[105][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_legacy_float_math_program,
                                         vector_legacy_float_math_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_legacy_float_math_state->halted,
              "expected decoded legacy float math program to halt") ||
      !Expect(vector_legacy_float_math_state->vgprs[104][0] ==
                  FloatBits(2.0f),
              "expected decoded v_exp_legacy_f32 lane 0 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[104][1] ==
                  FloatBits(4.0f),
              "expected decoded v_exp_legacy_f32 lane 1 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[104][2] == 0xdeadbeefu,
              "expected inactive decoded v_exp_legacy_f32 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[104][3] ==
                  FloatBits(0.5f),
              "expected decoded v_exp_legacy_f32 lane 3 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[105][0] ==
                  FloatBits(0.0f),
              "expected decoded v_log_legacy_f32 lane 0 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[105][1] ==
                  FloatBits(3.0f),
              "expected decoded v_log_legacy_f32 lane 1 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[105][2] == 0xdeadbeefu,
              "expected inactive decoded v_log_legacy_f32 result") ||
      !Expect(vector_legacy_float_math_state->vgprs[105][3] ==
                  FloatBits(-1.0f),
              "expected decoded v_log_legacy_f32 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_legacy_float_math_program;
  if (!Expect(interpreter.CompileProgram(vector_legacy_float_math_program,
                                         &compiled_vector_legacy_float_math_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  auto compiled_vector_legacy_float_math_state =
      std::make_unique<WaveExecutionState>();
  compiled_vector_legacy_float_math_state->exec_mask = 0b1011ULL;
  compiled_vector_legacy_float_math_state->vgprs[95][0] = FloatBits(1.0f);
  compiled_vector_legacy_float_math_state->vgprs[95][1] = FloatBits(2.0f);
  compiled_vector_legacy_float_math_state->vgprs[95][3] = FloatBits(-1.0f);
  compiled_vector_legacy_float_math_state->vgprs[96][0] = FloatBits(1.0f);
  compiled_vector_legacy_float_math_state->vgprs[96][1] = FloatBits(8.0f);
  compiled_vector_legacy_float_math_state->vgprs[96][3] = FloatBits(0.5f);
  compiled_vector_legacy_float_math_state->vgprs[104][2] = 0xdeadbeefu;
  compiled_vector_legacy_float_math_state->vgprs[105][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(
                  compiled_vector_legacy_float_math_program,
                  compiled_vector_legacy_float_math_state.get(),
                  &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_legacy_float_math_state->halted,
              "expected compiled legacy float math program to halt") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[104][0] ==
                  FloatBits(2.0f),
              "expected compiled v_exp_legacy_f32 lane 0 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[104][1] ==
                  FloatBits(4.0f),
              "expected compiled v_exp_legacy_f32 lane 1 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[104][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_exp_legacy_f32 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[104][3] ==
                  FloatBits(0.5f),
              "expected compiled v_exp_legacy_f32 lane 3 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[105][0] ==
                  FloatBits(0.0f),
              "expected compiled v_log_legacy_f32 lane 0 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[105][1] ==
                  FloatBits(3.0f),
              "expected compiled v_log_legacy_f32 lane 1 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[105][2] ==
                  0xdeadbeefu,
              "expected inactive compiled v_log_legacy_f32 result") ||
      !Expect(compiled_vector_legacy_float_math_state->vgprs[105][3] ==
                  FloatBits(-1.0f),
              "expected compiled v_log_legacy_f32 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_vop3_program = {
      DecodedInstruction::Binary("V_MUL_LO_U32", InstructionOperand::Vgpr(30),
                                 InstructionOperand::Sgpr(8),
                                 InstructionOperand::Vgpr(8)),
      DecodedInstruction::Binary("V_MUL_HI_U32", InstructionOperand::Vgpr(31),
                                 InstructionOperand::Sgpr(9),
                                 InstructionOperand::Vgpr(9)),
      DecodedInstruction::Binary("V_MUL_HI_I32", InstructionOperand::Vgpr(32),
                                 InstructionOperand::Sgpr(10),
                                 InstructionOperand::Vgpr(10)),
      DecodedInstruction::Ternary("V_ADD3_U32", InstructionOperand::Vgpr(33),
                                  InstructionOperand::Sgpr(11),
                                  InstructionOperand::Vgpr(11),
                                  InstructionOperand::Imm32(5)),
      DecodedInstruction::Ternary("V_LERP_U8", InstructionOperand::Vgpr(49),
                                  InstructionOperand::Vgpr(26),
                                  InstructionOperand::Sgpr(28),
                                  InstructionOperand::Sgpr(27)),
      DecodedInstruction::Ternary("V_PERM_B32", InstructionOperand::Vgpr(50),
                                  InstructionOperand::Sgpr(29),
                                  InstructionOperand::Vgpr(27),
                                  InstructionOperand::Vgpr(28)),
      DecodedInstruction::Ternary("V_BFE_U32", InstructionOperand::Vgpr(34),
                                  InstructionOperand::Vgpr(12),
                                  InstructionOperand::Vgpr(16),
                                  InstructionOperand::Sgpr(20)),
      DecodedInstruction::Ternary("V_BFE_I32", InstructionOperand::Vgpr(35),
                                  InstructionOperand::Vgpr(13),
                                  InstructionOperand::Vgpr(17),
                                  InstructionOperand::Sgpr(21)),
      DecodedInstruction::Ternary("V_BFI_B32", InstructionOperand::Vgpr(36),
                                  InstructionOperand::Sgpr(22),
                                  InstructionOperand::Vgpr(14),
                                  InstructionOperand::Vgpr(15)),
      DecodedInstruction::Ternary("V_ALIGNBIT_B32", InstructionOperand::Vgpr(37),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Imm32(4)),
      DecodedInstruction::Ternary("V_ALIGNBYTE_B32", InstructionOperand::Vgpr(38),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Imm32(1)),
      DecodedInstruction::Ternary("V_MIN3_I32", InstructionOperand::Vgpr(39),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_MAX3_I32", InstructionOperand::Vgpr(40),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_MED3_I32", InstructionOperand::Vgpr(41),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_MIN3_U32", InstructionOperand::Vgpr(42),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_MAX3_U32", InstructionOperand::Vgpr(43),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_MED3_U32", InstructionOperand::Vgpr(44),
                                  InstructionOperand::Vgpr(18),
                                  InstructionOperand::Sgpr(23),
                                  InstructionOperand::Vgpr(19)),
      DecodedInstruction::Ternary("V_SAD_U8", InstructionOperand::Vgpr(45),
                                  InstructionOperand::Vgpr(20),
                                  InstructionOperand::Sgpr(24),
                                  InstructionOperand::Vgpr(21)),
      DecodedInstruction::Ternary("V_SAD_HI_U8", InstructionOperand::Vgpr(46),
                                  InstructionOperand::Vgpr(20),
                                  InstructionOperand::Sgpr(24),
                                  InstructionOperand::Vgpr(21)),
      DecodedInstruction::Ternary("V_SAD_U16", InstructionOperand::Vgpr(47),
                                  InstructionOperand::Vgpr(22),
                                  InstructionOperand::Sgpr(25),
                                  InstructionOperand::Vgpr(23)),
      DecodedInstruction::Ternary("V_SAD_U32", InstructionOperand::Vgpr(48),
                                  InstructionOperand::Vgpr(24),
                                  InstructionOperand::Sgpr(26),
                                  InstructionOperand::Vgpr(25)),
      DecodedInstruction::Ternary("V_MAD_I32_I24",
                                  InstructionOperand::Vgpr(51),
                                  InstructionOperand::Vgpr(53),
                                  InstructionOperand::Sgpr(30),
                                  InstructionOperand::Vgpr(54)),
      DecodedInstruction::Ternary("V_MAD_U32_U24",
                                  InstructionOperand::Vgpr(52),
                                  InstructionOperand::Vgpr(55),
                                  InstructionOperand::Sgpr(31),
                                  InstructionOperand::Vgpr(56)),
      DecodedInstruction::Ternary("V_LSHL_ADD_U32",
                                  InstructionOperand::Vgpr(57),
                                  InstructionOperand::Vgpr(61),
                                  InstructionOperand::Sgpr(32),
                                  InstructionOperand::Vgpr(62)),
      DecodedInstruction::Binary("V_BCNT_U32_B32", InstructionOperand::Vgpr(58),
                                 InstructionOperand::Vgpr(63),
                                 InstructionOperand::Sgpr(33)),
      DecodedInstruction::Binary("V_BFM_B32", InstructionOperand::Vgpr(59),
                                 InstructionOperand::Vgpr(64),
                                 InstructionOperand::Sgpr(34)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_vop3_state;
  vector_vop3_state.exec_mask = 0b1011ULL;
  vector_vop3_state.sgprs[8] = 7u;
  vector_vop3_state.sgprs[9] = 0x80000000u;
  vector_vop3_state.sgprs[10] = 0x80000000u;
  vector_vop3_state.sgprs[11] = 1u;
  vector_vop3_state.sgprs[20] = 8u;
  vector_vop3_state.sgprs[21] = 4u;
  vector_vop3_state.sgprs[22] = 0x00ff00ffu;
  vector_vop3_state.sgprs[23] = 0x55667788u;
  vector_vop3_state.sgprs[24] = 0x01020304u;
  vector_vop3_state.sgprs[25] = 0x00010002u;
  vector_vop3_state.sgprs[26] = 75u;
  vector_vop3_state.sgprs[27] = 0x01000101u;
  vector_vop3_state.sgprs[28] = 0x05060708u;
  vector_vop3_state.sgprs[29] = 0x80017fffu;
  vector_vop3_state.sgprs[30] = 0x00fffffeu;
  vector_vop3_state.sgprs[31] = 0x0000ffffu;
  vector_vop3_state.sgprs[32] = 4u;
  vector_vop3_state.sgprs[33] = 100u;
  vector_vop3_state.sgprs[34] = 3u;
  vector_vop3_state.vgprs[8][0] = 9u;
  vector_vop3_state.vgprs[8][1] = 0xffffffffu;
  vector_vop3_state.vgprs[8][3] = 0x12345678u;
  vector_vop3_state.vgprs[9][0] = 4u;
  vector_vop3_state.vgprs[9][1] = 3u;
  vector_vop3_state.vgprs[9][3] = 1u;
  vector_vop3_state.vgprs[10][0] = 2u;
  vector_vop3_state.vgprs[10][1] = 0xffffffffu;
  vector_vop3_state.vgprs[10][3] = 3u;
  vector_vop3_state.vgprs[11][0] = 2u;
  vector_vop3_state.vgprs[11][1] = 0xffffffffu;
  vector_vop3_state.vgprs[11][3] = 7u;
  vector_vop3_state.vgprs[12][0] = 0x12345678u;
  vector_vop3_state.vgprs[12][1] = 0x87654321u;
  vector_vop3_state.vgprs[12][3] = 0xfedcba98u;
  vector_vop3_state.vgprs[13][0] = 0x0000f000u;
  vector_vop3_state.vgprs[13][1] = 0x00f00000u;
  vector_vop3_state.vgprs[13][3] = 0x12345678u;
  vector_vop3_state.vgprs[14][0] = 0x11111111u;
  vector_vop3_state.vgprs[14][1] = 0xaaaaaaaau;
  vector_vop3_state.vgprs[14][3] = 0xffffffffu;
  vector_vop3_state.vgprs[15][0] = 0x22222222u;
  vector_vop3_state.vgprs[15][1] = 0x55555555u;
  vector_vop3_state.vgprs[15][3] = 0x00000000u;
  vector_vop3_state.vgprs[16][0] = 8u;
  vector_vop3_state.vgprs[16][1] = 4u;
  vector_vop3_state.vgprs[16][3] = 28u;
  vector_vop3_state.vgprs[17][0] = 12u;
  vector_vop3_state.vgprs[17][1] = 20u;
  vector_vop3_state.vgprs[17][3] = 4u;
  vector_vop3_state.vgprs[18][0] = 0x11223344u;
  vector_vop3_state.vgprs[18][1] = 0x89abcdefu;
  vector_vop3_state.vgprs[18][3] = 0xf0f0f0f0u;
  vector_vop3_state.vgprs[19][0] = 0x80000000u;
  vector_vop3_state.vgprs[19][1] = 0xffffffffu;
  vector_vop3_state.vgprs[19][3] = 0x00000010u;
  vector_vop3_state.vgprs[20][0] = 0x11121314u;
  vector_vop3_state.vgprs[20][1] = 0x02040608u;
  vector_vop3_state.vgprs[20][3] = 0x00010203u;
  vector_vop3_state.vgprs[21][0] = 5u;
  vector_vop3_state.vgprs[21][1] = 7u;
  vector_vop3_state.vgprs[21][3] = 9u;
  vector_vop3_state.vgprs[22][0] = 0x00040008u;
  vector_vop3_state.vgprs[22][1] = 0x00020001u;
  vector_vop3_state.vgprs[22][3] = 0x00100020u;
  vector_vop3_state.vgprs[23][0] = 10u;
  vector_vop3_state.vgprs[23][1] = 12u;
  vector_vop3_state.vgprs[23][3] = 14u;
  vector_vop3_state.vgprs[24][0] = 100u;
  vector_vop3_state.vgprs[24][1] = 50u;
  vector_vop3_state.vgprs[24][3] = 400u;
  vector_vop3_state.vgprs[25][0] = 3u;
  vector_vop3_state.vgprs[25][1] = 5u;
  vector_vop3_state.vgprs[25][3] = 7u;
  vector_vop3_state.vgprs[26][0] = 0x01020304u;
  vector_vop3_state.vgprs[26][1] = 0x10111213u;
  vector_vop3_state.vgprs[26][3] = 0xffffffffu;
  vector_vop3_state.vgprs[27][0] = 0x11223344u;
  vector_vop3_state.vgprs[27][1] = 0x7fff8000u;
  vector_vop3_state.vgprs[27][3] = 0xa1b2c3d4u;
  vector_vop3_state.vgprs[28][0] = 0x0d0c0500u;
  vector_vop3_state.vgprs[28][1] = 0x0b0a0908u;
  vector_vop3_state.vgprs[28][3] = 0x07060403u;
  vector_vop3_state.vgprs[53][0] = 3u;
  vector_vop3_state.vgprs[53][1] = 0x00fffffeu;
  vector_vop3_state.vgprs[53][3] = 0x00800001u;
  vector_vop3_state.vgprs[54][0] = 5u;
  vector_vop3_state.vgprs[54][1] = 7u;
  vector_vop3_state.vgprs[54][3] = 9u;
  vector_vop3_state.vgprs[55][0] = 2u;
  vector_vop3_state.vgprs[55][1] = 0x00010000u;
  vector_vop3_state.vgprs[55][3] = 10u;
  vector_vop3_state.vgprs[56][0] = 11u;
  vector_vop3_state.vgprs[56][1] = 13u;
  vector_vop3_state.vgprs[56][3] = 17u;
  vector_vop3_state.vgprs[61][0] = 3u;
  vector_vop3_state.vgprs[61][1] = 0xffffffffu;
  vector_vop3_state.vgprs[61][3] = 0x12345678u;
  vector_vop3_state.vgprs[62][0] = 5u;
  vector_vop3_state.vgprs[62][1] = 7u;
  vector_vop3_state.vgprs[62][3] = 9u;
  vector_vop3_state.vgprs[63][0] = 0xf0f0f0f0u;
  vector_vop3_state.vgprs[63][1] = 0x00000003u;
  vector_vop3_state.vgprs[63][3] = 0xffffffffu;
  vector_vop3_state.vgprs[64][0] = 5u;
  vector_vop3_state.vgprs[64][1] = 7u;
  vector_vop3_state.vgprs[64][3] = 1u;
  vector_vop3_state.vgprs[30][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[31][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[32][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[33][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[34][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[35][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[36][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[37][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[38][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[39][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[40][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[41][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[42][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[43][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[44][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[45][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[46][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[47][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[48][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[49][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[50][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[51][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[52][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[57][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[58][2] = 0xdeadbeefu;
  vector_vop3_state.vgprs[59][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_vop3_program, &vector_vop3_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_vop3_state.halted, "expected vector VOP3 program to halt") ||
      !Expect(vector_vop3_state.vgprs[30][0] == 63u,
              "expected v_mul_lo_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[30][1] == 0xfffffff9u,
              "expected v_mul_lo_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[30][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_lo_u32 result") ||
      !Expect(vector_vop3_state.vgprs[30][3] == 0x7f6e5d48u,
              "expected v_mul_lo_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[31][0] == 2u,
              "expected v_mul_hi_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[31][1] == 1u,
              "expected v_mul_hi_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_hi_u32 result") ||
      !Expect(vector_vop3_state.vgprs[31][3] == 0u,
              "expected v_mul_hi_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[32][0] == 0xffffffffu,
              "expected v_mul_hi_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[32][1] == 0u,
              "expected v_mul_hi_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[32][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_hi_i32 result") ||
      !Expect(vector_vop3_state.vgprs[32][3] == 0xfffffffeu,
              "expected v_mul_hi_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[33][0] == 8u,
              "expected v_add3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[33][1] == 5u,
              "expected v_add3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[33][2] == 0xdeadbeefu,
              "expected inactive lane v_add3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[33][3] == 13u,
              "expected v_add3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[49][0] == 0x03040506u,
              "expected v_lerp_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[49][1] == 0x0b0b0d0eu,
              "expected v_lerp_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[49][2] == 0xdeadbeefu,
              "expected inactive lane v_lerp_u8 result") ||
      !Expect(vector_vop3_state.vgprs[49][3] == 0x82828384u,
              "expected v_lerp_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[50][0] == 0xff007f44u,
              "expected v_perm_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[50][1] == 0xff0000ffu,
              "expected v_perm_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[50][2] == 0xdeadbeefu,
              "expected inactive lane v_perm_b32 result") ||
      !Expect(vector_vop3_state.vgprs[50][3] == 0x8001ffa1u,
              "expected v_perm_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[34][0] == 0x56u,
              "expected v_bfe_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[34][1] == 0x32u,
              "expected v_bfe_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[34][2] == 0xdeadbeefu,
              "expected inactive lane v_bfe_u32 result") ||
      !Expect(vector_vop3_state.vgprs[34][3] == 0x0fu,
              "expected v_bfe_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[35][0] == 0xffffffffu,
              "expected v_bfe_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[35][1] == 0xffffffffu,
              "expected v_bfe_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[35][2] == 0xdeadbeefu,
              "expected inactive lane v_bfe_i32 result") ||
      !Expect(vector_vop3_state.vgprs[35][3] == 0x00000007u,
              "expected v_bfe_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[36][0] == 0x22112211u,
              "expected v_bfi_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[36][1] == 0x55aa55aau,
              "expected v_bfi_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[36][2] == 0xdeadbeefu,
              "expected inactive lane v_bfi_b32 result") ||
      !Expect(vector_vop3_state.vgprs[36][3] == 0x00ff00ffu,
              "expected v_bfi_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[37][0] == 0x45566778u,
              "expected v_alignbit_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[37][1] == 0xf5566778u,
              "expected v_alignbit_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[37][2] == 0xdeadbeefu,
              "expected inactive lane v_alignbit_b32 result") ||
      !Expect(vector_vop3_state.vgprs[37][3] == 0x05566778u,
              "expected v_alignbit_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[38][0] == 0x44556677u,
              "expected v_alignbyte_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[38][1] == 0xef556677u,
              "expected v_alignbyte_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[38][2] == 0xdeadbeefu,
              "expected inactive lane v_alignbyte_b32 result") ||
      !Expect(vector_vop3_state.vgprs[38][3] == 0xf0556677u,
              "expected v_alignbyte_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[39][0] == 0x80000000u,
              "expected v_min3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[39][1] == 0x89abcdefu,
              "expected v_min3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[39][2] == 0xdeadbeefu,
              "expected inactive lane v_min3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[39][3] == 0xf0f0f0f0u,
              "expected v_min3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[40][0] == 0x55667788u,
              "expected v_max3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[40][1] == 0x55667788u,
              "expected v_max3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[40][2] == 0xdeadbeefu,
              "expected inactive lane v_max3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[40][3] == 0x55667788u,
              "expected v_max3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[41][0] == 0x11223344u,
              "expected v_med3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[41][1] == 0xffffffffu,
              "expected v_med3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[41][2] == 0xdeadbeefu,
              "expected inactive lane v_med3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[41][3] == 0x00000010u,
              "expected v_med3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[42][0] == 0x11223344u,
              "expected v_min3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[42][1] == 0x55667788u,
              "expected v_min3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[42][2] == 0xdeadbeefu,
              "expected inactive lane v_min3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[42][3] == 0x00000010u,
              "expected v_min3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[43][0] == 0x80000000u,
              "expected v_max3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[43][1] == 0xffffffffu,
              "expected v_max3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[43][2] == 0xdeadbeefu,
              "expected inactive lane v_max3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[43][3] == 0xf0f0f0f0u,
              "expected v_max3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[44][0] == 0x55667788u,
              "expected v_med3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[44][1] == 0x89abcdefu,
              "expected v_med3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[44][2] == 0xdeadbeefu,
              "expected inactive lane v_med3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[44][3] == 0x55667788u,
              "expected v_med3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[45][0] == 69u,
              "expected v_sad_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[45][1] == 17u,
              "expected v_sad_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[45][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u8 result") ||
      !Expect(vector_vop3_state.vgprs[45][3] == 13u,
              "expected v_sad_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[46][0] == 0x00400005u,
              "expected v_sad_hi_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[46][1] == 0x000a0007u,
              "expected v_sad_hi_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[46][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_hi_u8 result") ||
      !Expect(vector_vop3_state.vgprs[46][3] == 0x00040009u,
              "expected v_sad_hi_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[47][0] == 19u,
              "expected v_sad_u16 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[47][1] == 14u,
              "expected v_sad_u16 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[47][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u16 result") ||
      !Expect(vector_vop3_state.vgprs[47][3] == 59u,
              "expected v_sad_u16 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[48][0] == 28u,
              "expected v_sad_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[48][1] == 30u,
              "expected v_sad_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[48][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u32 result") ||
      !Expect(vector_vop3_state.vgprs[48][3] == 332u,
              "expected v_sad_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[51][0] == 0xffffffffu,
              "expected v_mad_i32_i24 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[51][1] == 11u,
              "expected v_mad_i32_i24 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[51][2] == 0xdeadbeefu,
              "expected inactive lane v_mad_i32_i24 result") ||
      !Expect(vector_vop3_state.vgprs[51][3] == 0x01000007u,
              "expected v_mad_i32_i24 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[52][0] == 0x00020009u,
              "expected v_mad_u32_u24 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[52][1] == 0xffff000du,
              "expected v_mad_u32_u24 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[52][2] == 0xdeadbeefu,
              "expected inactive lane v_mad_u32_u24 result") ||
      !Expect(vector_vop3_state.vgprs[52][3] == 0x000a0007u,
              "expected v_mad_u32_u24 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[57][0] == 53u,
              "expected v_lshl_add_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[57][1] == 0xfffffff7u,
              "expected v_lshl_add_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[57][2] == 0xdeadbeefu,
              "expected inactive lane v_lshl_add_u32 result") ||
      !Expect(vector_vop3_state.vgprs[57][3] == 0x23456789u,
              "expected v_lshl_add_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[58][0] == 116u,
              "expected v_bcnt_u32_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[58][1] == 102u,
              "expected v_bcnt_u32_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[58][2] == 0xdeadbeefu,
              "expected inactive lane v_bcnt_u32_b32 result") ||
      !Expect(vector_vop3_state.vgprs[58][3] == 132u,
              "expected v_bcnt_u32_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[59][0] == 0x000000f8u,
              "expected v_bfm_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[59][1] == 0x000003f8u,
              "expected v_bfm_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[59][2] == 0xdeadbeefu,
              "expected inactive lane v_bfm_b32 result") ||
      !Expect(vector_vop3_state.vgprs[59][3] == 0x00000008u,
              "expected v_bfm_b32 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_vop3_program;
  if (!Expect(interpreter.CompileProgram(vector_vop3_program,
                                         &compiled_vector_vop3_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_vop3_state;
  compiled_vector_vop3_state.exec_mask = 0b1011ULL;
  compiled_vector_vop3_state.sgprs[8] = 7u;
  compiled_vector_vop3_state.sgprs[9] = 0x80000000u;
  compiled_vector_vop3_state.sgprs[10] = 0x80000000u;
  compiled_vector_vop3_state.sgprs[11] = 1u;
  compiled_vector_vop3_state.sgprs[20] = 8u;
  compiled_vector_vop3_state.sgprs[21] = 4u;
  compiled_vector_vop3_state.sgprs[22] = 0x00ff00ffu;
  compiled_vector_vop3_state.sgprs[23] = 0x55667788u;
  compiled_vector_vop3_state.sgprs[24] = 0x01020304u;
  compiled_vector_vop3_state.sgprs[25] = 0x00010002u;
  compiled_vector_vop3_state.sgprs[26] = 75u;
  compiled_vector_vop3_state.sgprs[27] = 0x01000101u;
  compiled_vector_vop3_state.sgprs[28] = 0x05060708u;
  compiled_vector_vop3_state.sgprs[29] = 0x80017fffu;
  compiled_vector_vop3_state.sgprs[30] = 0x00fffffeu;
  compiled_vector_vop3_state.sgprs[31] = 0x0000ffffu;
  compiled_vector_vop3_state.sgprs[32] = 4u;
  compiled_vector_vop3_state.sgprs[33] = 100u;
  compiled_vector_vop3_state.sgprs[34] = 3u;
  compiled_vector_vop3_state.vgprs[8][0] = 9u;
  compiled_vector_vop3_state.vgprs[8][1] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[8][3] = 0x12345678u;
  compiled_vector_vop3_state.vgprs[9][0] = 4u;
  compiled_vector_vop3_state.vgprs[9][1] = 3u;
  compiled_vector_vop3_state.vgprs[9][3] = 1u;
  compiled_vector_vop3_state.vgprs[10][0] = 2u;
  compiled_vector_vop3_state.vgprs[10][1] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[10][3] = 3u;
  compiled_vector_vop3_state.vgprs[11][0] = 2u;
  compiled_vector_vop3_state.vgprs[11][1] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[11][3] = 7u;
  compiled_vector_vop3_state.vgprs[12][0] = 0x12345678u;
  compiled_vector_vop3_state.vgprs[12][1] = 0x87654321u;
  compiled_vector_vop3_state.vgprs[12][3] = 0xfedcba98u;
  compiled_vector_vop3_state.vgprs[13][0] = 0x0000f000u;
  compiled_vector_vop3_state.vgprs[13][1] = 0x00f00000u;
  compiled_vector_vop3_state.vgprs[13][3] = 0x12345678u;
  compiled_vector_vop3_state.vgprs[14][0] = 0x11111111u;
  compiled_vector_vop3_state.vgprs[14][1] = 0xaaaaaaaau;
  compiled_vector_vop3_state.vgprs[14][3] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[15][0] = 0x22222222u;
  compiled_vector_vop3_state.vgprs[15][1] = 0x55555555u;
  compiled_vector_vop3_state.vgprs[15][3] = 0x00000000u;
  compiled_vector_vop3_state.vgprs[16][0] = 8u;
  compiled_vector_vop3_state.vgprs[16][1] = 4u;
  compiled_vector_vop3_state.vgprs[16][3] = 28u;
  compiled_vector_vop3_state.vgprs[17][0] = 12u;
  compiled_vector_vop3_state.vgprs[17][1] = 20u;
  compiled_vector_vop3_state.vgprs[17][3] = 4u;
  compiled_vector_vop3_state.vgprs[18][0] = 0x11223344u;
  compiled_vector_vop3_state.vgprs[18][1] = 0x89abcdefu;
  compiled_vector_vop3_state.vgprs[18][3] = 0xf0f0f0f0u;
  compiled_vector_vop3_state.vgprs[19][0] = 0x80000000u;
  compiled_vector_vop3_state.vgprs[19][1] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[19][3] = 0x00000010u;
  compiled_vector_vop3_state.vgprs[20][0] = 0x11121314u;
  compiled_vector_vop3_state.vgprs[20][1] = 0x02040608u;
  compiled_vector_vop3_state.vgprs[20][3] = 0x00010203u;
  compiled_vector_vop3_state.vgprs[21][0] = 5u;
  compiled_vector_vop3_state.vgprs[21][1] = 7u;
  compiled_vector_vop3_state.vgprs[21][3] = 9u;
  compiled_vector_vop3_state.vgprs[22][0] = 0x00040008u;
  compiled_vector_vop3_state.vgprs[22][1] = 0x00020001u;
  compiled_vector_vop3_state.vgprs[22][3] = 0x00100020u;
  compiled_vector_vop3_state.vgprs[23][0] = 10u;
  compiled_vector_vop3_state.vgprs[23][1] = 12u;
  compiled_vector_vop3_state.vgprs[23][3] = 14u;
  compiled_vector_vop3_state.vgprs[24][0] = 100u;
  compiled_vector_vop3_state.vgprs[24][1] = 50u;
  compiled_vector_vop3_state.vgprs[24][3] = 400u;
  compiled_vector_vop3_state.vgprs[25][0] = 3u;
  compiled_vector_vop3_state.vgprs[25][1] = 5u;
  compiled_vector_vop3_state.vgprs[25][3] = 7u;
  compiled_vector_vop3_state.vgprs[26][0] = 0x01020304u;
  compiled_vector_vop3_state.vgprs[26][1] = 0x10111213u;
  compiled_vector_vop3_state.vgprs[26][3] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[27][0] = 0x11223344u;
  compiled_vector_vop3_state.vgprs[27][1] = 0x7fff8000u;
  compiled_vector_vop3_state.vgprs[27][3] = 0xa1b2c3d4u;
  compiled_vector_vop3_state.vgprs[28][0] = 0x0d0c0500u;
  compiled_vector_vop3_state.vgprs[28][1] = 0x0b0a0908u;
  compiled_vector_vop3_state.vgprs[28][3] = 0x07060403u;
  compiled_vector_vop3_state.vgprs[53][0] = 3u;
  compiled_vector_vop3_state.vgprs[53][1] = 0x00fffffeu;
  compiled_vector_vop3_state.vgprs[53][3] = 0x00800001u;
  compiled_vector_vop3_state.vgprs[54][0] = 5u;
  compiled_vector_vop3_state.vgprs[54][1] = 7u;
  compiled_vector_vop3_state.vgprs[54][3] = 9u;
  compiled_vector_vop3_state.vgprs[55][0] = 2u;
  compiled_vector_vop3_state.vgprs[55][1] = 0x00010000u;
  compiled_vector_vop3_state.vgprs[55][3] = 10u;
  compiled_vector_vop3_state.vgprs[56][0] = 11u;
  compiled_vector_vop3_state.vgprs[56][1] = 13u;
  compiled_vector_vop3_state.vgprs[56][3] = 17u;
  compiled_vector_vop3_state.vgprs[61][0] = 3u;
  compiled_vector_vop3_state.vgprs[61][1] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[61][3] = 0x12345678u;
  compiled_vector_vop3_state.vgprs[62][0] = 5u;
  compiled_vector_vop3_state.vgprs[62][1] = 7u;
  compiled_vector_vop3_state.vgprs[62][3] = 9u;
  compiled_vector_vop3_state.vgprs[63][0] = 0xf0f0f0f0u;
  compiled_vector_vop3_state.vgprs[63][1] = 0x00000003u;
  compiled_vector_vop3_state.vgprs[63][3] = 0xffffffffu;
  compiled_vector_vop3_state.vgprs[64][0] = 5u;
  compiled_vector_vop3_state.vgprs[64][1] = 7u;
  compiled_vector_vop3_state.vgprs[64][3] = 1u;
  compiled_vector_vop3_state.vgprs[30][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[31][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[32][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[33][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[34][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[35][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[36][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[37][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[38][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[39][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[40][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[41][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[42][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[43][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[44][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[45][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[46][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[47][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[48][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[49][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[50][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[51][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[52][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[57][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[58][2] = 0xdeadbeefu;
  compiled_vector_vop3_state.vgprs[59][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_vop3_program,
                                         &compiled_vector_vop3_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_vop3_state.halted,
              "expected compiled vector VOP3 program to halt") ||
      !Expect(compiled_vector_vop3_state.vgprs[30][0] == 63u,
              "expected compiled v_mul_lo_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[30][1] == 0xfffffff9u,
              "expected compiled v_mul_lo_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[30][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_mul_lo_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[30][3] == 0x7f6e5d48u,
              "expected compiled v_mul_lo_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[31][0] == 2u,
              "expected compiled v_mul_hi_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[31][1] == 1u,
              "expected compiled v_mul_hi_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[31][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_mul_hi_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[31][3] == 0u,
              "expected compiled v_mul_hi_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[32][0] == 0xffffffffu,
              "expected compiled v_mul_hi_i32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[32][1] == 0u,
              "expected compiled v_mul_hi_i32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[32][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_mul_hi_i32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[32][3] == 0xfffffffeu,
              "expected compiled v_mul_hi_i32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[33][0] == 8u,
              "expected compiled v_add3_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[33][1] == 5u,
              "expected compiled v_add3_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[33][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_add3_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[33][3] == 13u,
              "expected compiled v_add3_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[49][0] == 0x03040506u,
              "expected compiled v_lerp_u8 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[49][1] == 0x0b0b0d0eu,
              "expected compiled v_lerp_u8 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[49][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_lerp_u8 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[49][3] == 0x82828384u,
              "expected compiled v_lerp_u8 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[50][0] == 0xff007f44u,
              "expected compiled v_perm_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[50][1] == 0xff0000ffu,
              "expected compiled v_perm_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[50][2] == 0xdeadbeefu,
              "expected compiled inactive v_perm_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[50][3] == 0x8001ffa1u,
              "expected compiled v_perm_b32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[34][0] == 0x56u,
              "expected compiled v_bfe_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[34][1] == 0x32u,
              "expected compiled v_bfe_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[34][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_bfe_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[34][3] == 0x0fu,
              "expected compiled v_bfe_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[35][0] == 0xffffffffu,
              "expected compiled v_bfe_i32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[35][1] == 0xffffffffu,
              "expected compiled v_bfe_i32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[35][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_bfe_i32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[35][3] == 0x00000007u,
              "expected compiled v_bfe_i32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[36][0] == 0x22112211u,
              "expected compiled v_bfi_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[36][1] == 0x55aa55aau,
              "expected compiled v_bfi_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[36][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_bfi_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[36][3] == 0x00ff00ffu,
              "expected compiled v_bfi_b32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[37][0] == 0x45566778u,
              "expected compiled v_alignbit_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[37][1] == 0xf5566778u,
              "expected compiled v_alignbit_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[37][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_alignbit_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[37][3] == 0x05566778u,
              "expected compiled v_alignbit_b32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[38][0] == 0x44556677u,
              "expected compiled v_alignbyte_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[38][1] == 0xef556677u,
              "expected compiled v_alignbyte_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[38][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_alignbyte_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[38][3] == 0xf0556677u,
              "expected compiled v_alignbyte_b32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[39][0] == 0x80000000u,
              "expected compiled v_min3_i32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[39][1] == 0x89abcdefu,
              "expected compiled v_min3_i32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[39][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_min3_i32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[39][3] == 0xf0f0f0f0u,
              "expected compiled v_min3_i32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[40][0] == 0x55667788u,
              "expected compiled v_max3_i32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[40][1] == 0x55667788u,
              "expected compiled v_max3_i32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[40][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_max3_i32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[40][3] == 0x55667788u,
              "expected compiled v_max3_i32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[41][0] == 0x11223344u,
              "expected compiled v_med3_i32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[41][1] == 0xffffffffu,
              "expected compiled v_med3_i32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[41][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_med3_i32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[41][3] == 0x00000010u,
              "expected compiled v_med3_i32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[42][0] == 0x11223344u,
              "expected compiled v_min3_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[42][1] == 0x55667788u,
              "expected compiled v_min3_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[42][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_min3_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[42][3] == 0x00000010u,
              "expected compiled v_min3_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[43][0] == 0x80000000u,
              "expected compiled v_max3_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[43][1] == 0xffffffffu,
              "expected compiled v_max3_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[43][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_max3_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[43][3] == 0xf0f0f0f0u,
              "expected compiled v_max3_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[44][0] == 0x55667788u,
              "expected compiled v_med3_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[44][1] == 0x89abcdefu,
              "expected compiled v_med3_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[44][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_med3_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[44][3] == 0x55667788u,
              "expected compiled v_med3_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[45][0] == 69u,
              "expected compiled v_sad_u8 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[45][1] == 17u,
              "expected compiled v_sad_u8 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[45][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_sad_u8 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[45][3] == 13u,
              "expected compiled v_sad_u8 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[46][0] == 0x00400005u,
              "expected compiled v_sad_hi_u8 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[46][1] == 0x000a0007u,
              "expected compiled v_sad_hi_u8 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[46][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_sad_hi_u8 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[46][3] == 0x00040009u,
              "expected compiled v_sad_hi_u8 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[47][0] == 19u,
              "expected compiled v_sad_u16 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[47][1] == 14u,
              "expected compiled v_sad_u16 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[47][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_sad_u16 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[47][3] == 59u,
              "expected compiled v_sad_u16 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[48][0] == 28u,
              "expected compiled v_sad_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[48][1] == 30u,
              "expected compiled v_sad_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[48][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_sad_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[48][3] == 332u,
              "expected compiled v_sad_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[51][0] == 0xffffffffu,
              "expected compiled v_mad_i32_i24 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[51][1] == 11u,
              "expected compiled v_mad_i32_i24 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[51][2] == 0xdeadbeefu,
              "expected compiled inactive v_mad_i32_i24 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[51][3] == 0x01000007u,
              "expected compiled v_mad_i32_i24 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[52][0] == 0x00020009u,
              "expected compiled v_mad_u32_u24 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[52][1] == 0xffff000du,
              "expected compiled v_mad_u32_u24 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[52][2] == 0xdeadbeefu,
              "expected compiled inactive v_mad_u32_u24 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[52][3] == 0x000a0007u,
              "expected compiled v_mad_u32_u24 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[57][0] == 53u,
              "expected compiled v_lshl_add_u32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[57][1] == 0xfffffff7u,
              "expected compiled v_lshl_add_u32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[57][2] == 0xdeadbeefu,
              "expected compiled inactive v_lshl_add_u32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[57][3] == 0x23456789u,
              "expected compiled v_lshl_add_u32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[58][0] == 116u,
              "expected compiled v_bcnt_u32_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[58][1] == 102u,
              "expected compiled v_bcnt_u32_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[58][2] == 0xdeadbeefu,
              "expected compiled inactive v_bcnt_u32_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[58][3] == 132u,
              "expected compiled v_bcnt_u32_b32 lane 3 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[59][0] == 0x000000f8u,
              "expected compiled v_bfm_b32 lane 0 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[59][1] == 0x000003f8u,
              "expected compiled v_bfm_b32 lane 1 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[59][2] == 0xdeadbeefu,
              "expected compiled inactive v_bfm_b32 result") ||
      !Expect(compiled_vector_vop3_state.vgprs[59][3] == 0x00000008u,
              "expected compiled v_bfm_b32 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_mbcnt_program = {
      DecodedInstruction::Binary("V_MBCNT_LO_U32_B32",
                                 InstructionOperand::Vgpr(60),
                                 InstructionOperand::Vgpr(70),
                                 InstructionOperand::Sgpr(35)),
      DecodedInstruction::Binary("V_MBCNT_HI_U32_B32",
                                 InstructionOperand::Vgpr(61),
                                 InstructionOperand::Vgpr(70),
                                 InstructionOperand::Sgpr(36)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_mbcnt_state;
  vector_mbcnt_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 32) | (1ULL << 35);
  vector_mbcnt_state.sgprs[35] = 10u;
  vector_mbcnt_state.sgprs[36] = 40u;
  vector_mbcnt_state.vgprs[70][0] = 0x000000b5u;
  vector_mbcnt_state.vgprs[70][1] = 0x000000b5u;
  vector_mbcnt_state.vgprs[70][32] = 0x000000b5u;
  vector_mbcnt_state.vgprs[70][35] = 0x000000b5u;
  vector_mbcnt_state.vgprs[60][2] = 0xdeadbeefu;
  vector_mbcnt_state.vgprs[61][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_mbcnt_program, &vector_mbcnt_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_mbcnt_state.halted, "expected vector MBCNT program to halt") ||
      !Expect(vector_mbcnt_state.vgprs[60][0] == 10u,
              "expected v_mbcnt_lo_u32_b32 lane 0 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][1] == 11u,
              "expected v_mbcnt_lo_u32_b32 lane 1 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][2] == 0xdeadbeefu,
              "expected inactive v_mbcnt_lo_u32_b32 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][32] == 15u,
              "expected v_mbcnt_lo_u32_b32 lane 32 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][35] == 15u,
              "expected v_mbcnt_lo_u32_b32 lane 35 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][0] == 40u,
              "expected v_mbcnt_hi_u32_b32 lane 0 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][1] == 40u,
              "expected v_mbcnt_hi_u32_b32 lane 1 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][2] == 0xdeadbeefu,
              "expected inactive v_mbcnt_hi_u32_b32 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][32] == 40u,
              "expected v_mbcnt_hi_u32_b32 lane 32 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][35] == 42u,
              "expected v_mbcnt_hi_u32_b32 lane 35 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_mbcnt_program;
  if (!Expect(interpreter.CompileProgram(vector_mbcnt_program,
                                         &compiled_vector_mbcnt_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_mbcnt_state;
  compiled_vector_mbcnt_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 32) | (1ULL << 35);
  compiled_vector_mbcnt_state.sgprs[35] = 10u;
  compiled_vector_mbcnt_state.sgprs[36] = 40u;
  compiled_vector_mbcnt_state.vgprs[70][0] = 0x000000b5u;
  compiled_vector_mbcnt_state.vgprs[70][1] = 0x000000b5u;
  compiled_vector_mbcnt_state.vgprs[70][32] = 0x000000b5u;
  compiled_vector_mbcnt_state.vgprs[70][35] = 0x000000b5u;
  compiled_vector_mbcnt_state.vgprs[60][2] = 0xdeadbeefu;
  compiled_vector_mbcnt_state.vgprs[61][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_mbcnt_program,
                                         &compiled_vector_mbcnt_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_mbcnt_state.halted,
              "expected compiled vector MBCNT program to halt") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[60][0] == 10u,
              "expected compiled v_mbcnt_lo_u32_b32 lane 0 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[60][1] == 11u,
              "expected compiled v_mbcnt_lo_u32_b32 lane 1 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[60][2] == 0xdeadbeefu,
              "expected compiled inactive v_mbcnt_lo_u32_b32 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[60][32] == 15u,
              "expected compiled v_mbcnt_lo_u32_b32 lane 32 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[60][35] == 15u,
              "expected compiled v_mbcnt_lo_u32_b32 lane 35 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[61][0] == 40u,
              "expected compiled v_mbcnt_hi_u32_b32 lane 0 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[61][1] == 40u,
              "expected compiled v_mbcnt_hi_u32_b32 lane 1 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[61][2] == 0xdeadbeefu,
              "expected compiled inactive v_mbcnt_hi_u32_b32 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[61][32] == 40u,
              "expected compiled v_mbcnt_hi_u32_b32 lane 32 result") ||
      !Expect(compiled_vector_mbcnt_state.vgprs[61][35] == 42u,
              "expected compiled v_mbcnt_hi_u32_b32 lane 35 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_move64_program = {
      DecodedInstruction::Unary("V_MOV_B64", InstructionOperand::Vgpr(68),
                                InstructionOperand::Vgpr(80)),
      DecodedInstruction::Unary("V_MOV_B64", InstructionOperand::Vgpr(70),
                                InstructionOperand::Sgpr(44)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_move64_state;
  vector_move64_state.exec_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  vector_move64_state.sgprs[44] = 0x89abcdefu;
  vector_move64_state.sgprs[45] = 0x01234567u;
  SplitU64(0xaaaaaaaa55555555ULL, &vector_move64_state.vgprs[80][0],
           &vector_move64_state.vgprs[81][0]);
  SplitU64(0x0123456789abcdefULL, &vector_move64_state.vgprs[80][1],
           &vector_move64_state.vgprs[81][1]);
  SplitU64(0xfedcba9876543210ULL, &vector_move64_state.vgprs[80][35],
           &vector_move64_state.vgprs[81][35]);
  vector_move64_state.vgprs[68][2] = 0xdeadbeefu;
  vector_move64_state.vgprs[69][2] = 0xcafebabeu;
  vector_move64_state.vgprs[70][2] = 0xdeadbeefu;
  vector_move64_state.vgprs[71][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(vector_move64_program,
                                         &vector_move64_state, &error_message),
              error_message.c_str()) ||
      !Expect(vector_move64_state.halted,
              "expected vector move64 program to halt") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][0],
                         vector_move64_state.vgprs[69][0]) ==
                  0xaaaaaaaa55555555ULL,
              "expected v_mov_b64 from vgpr lane 0 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][1],
                         vector_move64_state.vgprs[69][1]) ==
                  0x0123456789abcdefULL,
              "expected v_mov_b64 from vgpr lane 1 result") ||
      !Expect(vector_move64_state.vgprs[68][2] == 0xdeadbeefu &&
                  vector_move64_state.vgprs[69][2] == 0xcafebabeu,
              "expected inactive v_mov_b64 from vgpr result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][35],
                         vector_move64_state.vgprs[69][35]) ==
                  0xfedcba9876543210ULL,
              "expected v_mov_b64 from vgpr lane 35 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][0],
                         vector_move64_state.vgprs[71][0]) ==
                  0x0123456789abcdefULL,
              "expected v_mov_b64 from sgpr lane 0 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][1],
                         vector_move64_state.vgprs[71][1]) ==
                  0x0123456789abcdefULL,
              "expected v_mov_b64 from sgpr lane 1 result") ||
      !Expect(vector_move64_state.vgprs[70][2] == 0xdeadbeefu &&
                  vector_move64_state.vgprs[71][2] == 0xcafebabeu,
              "expected inactive v_mov_b64 from sgpr result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][35],
                         vector_move64_state.vgprs[71][35]) ==
                  0x0123456789abcdefULL,
              "expected v_mov_b64 from sgpr lane 35 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_move64_program;
  if (!Expect(interpreter.CompileProgram(vector_move64_program,
                                         &compiled_vector_move64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_move64_state;
  compiled_vector_move64_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  compiled_vector_move64_state.sgprs[44] = 0x89abcdefu;
  compiled_vector_move64_state.sgprs[45] = 0x01234567u;
  SplitU64(0xaaaaaaaa55555555ULL, &compiled_vector_move64_state.vgprs[80][0],
           &compiled_vector_move64_state.vgprs[81][0]);
  SplitU64(0x0123456789abcdefULL, &compiled_vector_move64_state.vgprs[80][1],
           &compiled_vector_move64_state.vgprs[81][1]);
  SplitU64(0xfedcba9876543210ULL, &compiled_vector_move64_state.vgprs[80][35],
           &compiled_vector_move64_state.vgprs[81][35]);
  compiled_vector_move64_state.vgprs[68][2] = 0xdeadbeefu;
  compiled_vector_move64_state.vgprs[69][2] = 0xcafebabeu;
  compiled_vector_move64_state.vgprs[70][2] = 0xdeadbeefu;
  compiled_vector_move64_state.vgprs[71][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_move64_program,
                                         &compiled_vector_move64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_move64_state.halted,
              "expected compiled vector move64 program to halt") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[68][0],
                         compiled_vector_move64_state.vgprs[69][0]) ==
                  0xaaaaaaaa55555555ULL,
              "expected compiled v_mov_b64 from vgpr lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[68][1],
                         compiled_vector_move64_state.vgprs[69][1]) ==
                  0x0123456789abcdefULL,
              "expected compiled v_mov_b64 from vgpr lane 1 result") ||
      !Expect(compiled_vector_move64_state.vgprs[68][2] == 0xdeadbeefu &&
                  compiled_vector_move64_state.vgprs[69][2] == 0xcafebabeu,
              "expected compiled inactive v_mov_b64 from vgpr result") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[68][35],
                         compiled_vector_move64_state.vgprs[69][35]) ==
                  0xfedcba9876543210ULL,
              "expected compiled v_mov_b64 from vgpr lane 35 result") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[70][0],
                         compiled_vector_move64_state.vgprs[71][0]) ==
                  0x0123456789abcdefULL,
              "expected compiled v_mov_b64 from sgpr lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[70][1],
                         compiled_vector_move64_state.vgprs[71][1]) ==
                  0x0123456789abcdefULL,
              "expected compiled v_mov_b64 from sgpr lane 1 result") ||
      !Expect(compiled_vector_move64_state.vgprs[70][2] == 0xdeadbeefu &&
                  compiled_vector_move64_state.vgprs[71][2] == 0xcafebabeu,
              "expected compiled inactive v_mov_b64 from sgpr result") ||
      !Expect(ComposeU64(compiled_vector_move64_state.vgprs[70][35],
                         compiled_vector_move64_state.vgprs[71][35]) ==
                  0x0123456789abcdefULL,
              "expected compiled v_mov_b64 from sgpr lane 35 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_shift64_program = {
      DecodedInstruction::Binary("V_LSHLREV_B64", InstructionOperand::Vgpr(72),
                                 InstructionOperand::Sgpr(40),
                                 InstructionOperand::Vgpr(80)),
      DecodedInstruction::Binary("V_LSHRREV_B64", InstructionOperand::Vgpr(74),
                                 InstructionOperand::Sgpr(41),
                                 InstructionOperand::Vgpr(80)),
      DecodedInstruction::Binary("V_ASHRREV_I64", InstructionOperand::Vgpr(76),
                                 InstructionOperand::Sgpr(42),
                                 InstructionOperand::Vgpr(80)),
      DecodedInstruction::Ternary("V_LSHL_ADD_U64",
                                  InstructionOperand::Vgpr(78),
                                  InstructionOperand::Vgpr(80),
                                  InstructionOperand::Vgpr(84),
                                  InstructionOperand::Vgpr(82)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_shift64_state;
  vector_shift64_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  vector_shift64_state.sgprs[40] = 4u;
  vector_shift64_state.sgprs[41] = 4u;
  vector_shift64_state.sgprs[42] = 4u;
  SplitU64(0x0123456789abcdefULL, &vector_shift64_state.vgprs[80][0],
           &vector_shift64_state.vgprs[81][0]);
  SplitU64(0x8000000000000000ULL, &vector_shift64_state.vgprs[80][1],
           &vector_shift64_state.vgprs[81][1]);
  SplitU64(0xfffffffffffffff0ULL, &vector_shift64_state.vgprs[80][35],
           &vector_shift64_state.vgprs[81][35]);
  SplitU64(0x1111111111111111ULL, &vector_shift64_state.vgprs[82][0],
           &vector_shift64_state.vgprs[83][0]);
  SplitU64(0x0000000000000003ULL, &vector_shift64_state.vgprs[82][1],
           &vector_shift64_state.vgprs[83][1]);
  SplitU64(0x0000000000000010ULL, &vector_shift64_state.vgprs[82][35],
           &vector_shift64_state.vgprs[83][35]);
  vector_shift64_state.vgprs[84][0] = 4u;
  vector_shift64_state.vgprs[84][1] = 1u;
  vector_shift64_state.vgprs[84][35] = 4u;
  vector_shift64_state.vgprs[72][2] = 0xdeadbeefu;
  vector_shift64_state.vgprs[73][2] = 0xcafebabeu;
  vector_shift64_state.vgprs[74][2] = 0xdeadbeefu;
  vector_shift64_state.vgprs[75][2] = 0xcafebabeu;
  vector_shift64_state.vgprs[76][2] = 0xdeadbeefu;
  vector_shift64_state.vgprs[77][2] = 0xcafebabeu;
  vector_shift64_state.vgprs[78][2] = 0xdeadbeefu;
  vector_shift64_state.vgprs[79][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(vector_shift64_program,
                                         &vector_shift64_state, &error_message),
              error_message.c_str()) ||
      !Expect(vector_shift64_state.halted,
              "expected vector shift64 program to halt") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][0],
                         vector_shift64_state.vgprs[73][0]) ==
                  0x123456789abcdef0ULL,
              "expected v_lshlrev_b64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][1],
                         vector_shift64_state.vgprs[73][1]) == 0ULL,
              "expected v_lshlrev_b64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[72][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[73][2] == 0xcafebabeu,
              "expected inactive v_lshlrev_b64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][35],
                         vector_shift64_state.vgprs[73][35]) ==
                  0xffffffffffffff00ULL,
              "expected v_lshlrev_b64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][0],
                         vector_shift64_state.vgprs[75][0]) ==
                  0x00123456789abcdeULL,
              "expected v_lshrrev_b64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][1],
                         vector_shift64_state.vgprs[75][1]) ==
                  0x0800000000000000ULL,
              "expected v_lshrrev_b64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[74][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[75][2] == 0xcafebabeu,
              "expected inactive v_lshrrev_b64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][35],
                         vector_shift64_state.vgprs[75][35]) ==
                  0x0fffffffffffffffULL,
              "expected v_lshrrev_b64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][0],
                         vector_shift64_state.vgprs[77][0]) ==
                  0x00123456789abcdeULL,
              "expected v_ashrrev_i64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][1],
                         vector_shift64_state.vgprs[77][1]) ==
                  0xf800000000000000ULL,
              "expected v_ashrrev_i64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[76][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[77][2] == 0xcafebabeu,
              "expected inactive v_ashrrev_i64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][35],
                         vector_shift64_state.vgprs[77][35]) ==
                  0xffffffffffffffffULL,
              "expected v_ashrrev_i64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][0],
                         vector_shift64_state.vgprs[79][0]) ==
                  0x23456789abcdf001ULL,
              "expected v_lshl_add_u64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][1],
                         vector_shift64_state.vgprs[79][1]) == 3ULL,
              "expected v_lshl_add_u64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[78][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[79][2] == 0xcafebabeu,
              "expected inactive v_lshl_add_u64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][35],
                         vector_shift64_state.vgprs[79][35]) ==
                  0xffffffffffffff10ULL,
              "expected v_lshl_add_u64 lane 35 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_shift64_program;
  if (!Expect(interpreter.CompileProgram(vector_shift64_program,
                                         &compiled_vector_shift64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_shift64_state;
  compiled_vector_shift64_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  compiled_vector_shift64_state.sgprs[40] = 4u;
  compiled_vector_shift64_state.sgprs[41] = 4u;
  compiled_vector_shift64_state.sgprs[42] = 4u;
  SplitU64(0x0123456789abcdefULL, &compiled_vector_shift64_state.vgprs[80][0],
           &compiled_vector_shift64_state.vgprs[81][0]);
  SplitU64(0x8000000000000000ULL, &compiled_vector_shift64_state.vgprs[80][1],
           &compiled_vector_shift64_state.vgprs[81][1]);
  SplitU64(0xfffffffffffffff0ULL, &compiled_vector_shift64_state.vgprs[80][35],
           &compiled_vector_shift64_state.vgprs[81][35]);
  SplitU64(0x1111111111111111ULL, &compiled_vector_shift64_state.vgprs[82][0],
           &compiled_vector_shift64_state.vgprs[83][0]);
  SplitU64(0x0000000000000003ULL, &compiled_vector_shift64_state.vgprs[82][1],
           &compiled_vector_shift64_state.vgprs[83][1]);
  SplitU64(0x0000000000000010ULL, &compiled_vector_shift64_state.vgprs[82][35],
           &compiled_vector_shift64_state.vgprs[83][35]);
  compiled_vector_shift64_state.vgprs[84][0] = 4u;
  compiled_vector_shift64_state.vgprs[84][1] = 1u;
  compiled_vector_shift64_state.vgprs[84][35] = 4u;
  compiled_vector_shift64_state.vgprs[72][2] = 0xdeadbeefu;
  compiled_vector_shift64_state.vgprs[73][2] = 0xcafebabeu;
  compiled_vector_shift64_state.vgprs[74][2] = 0xdeadbeefu;
  compiled_vector_shift64_state.vgprs[75][2] = 0xcafebabeu;
  compiled_vector_shift64_state.vgprs[76][2] = 0xdeadbeefu;
  compiled_vector_shift64_state.vgprs[77][2] = 0xcafebabeu;
  compiled_vector_shift64_state.vgprs[78][2] = 0xdeadbeefu;
  compiled_vector_shift64_state.vgprs[79][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_shift64_program,
                                         &compiled_vector_shift64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_shift64_state.halted,
              "expected compiled vector shift64 program to halt") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[72][0],
                         compiled_vector_shift64_state.vgprs[73][0]) ==
                  0x123456789abcdef0ULL,
              "expected compiled v_lshlrev_b64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[72][1],
                         compiled_vector_shift64_state.vgprs[73][1]) == 0ULL,
              "expected compiled v_lshlrev_b64 lane 1 result") ||
      !Expect(compiled_vector_shift64_state.vgprs[72][2] == 0xdeadbeefu &&
                  compiled_vector_shift64_state.vgprs[73][2] == 0xcafebabeu,
              "expected compiled inactive v_lshlrev_b64 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[72][35],
                         compiled_vector_shift64_state.vgprs[73][35]) ==
                  0xffffffffffffff00ULL,
              "expected compiled v_lshlrev_b64 lane 35 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[74][0],
                         compiled_vector_shift64_state.vgprs[75][0]) ==
                  0x00123456789abcdeULL,
              "expected compiled v_lshrrev_b64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[74][1],
                         compiled_vector_shift64_state.vgprs[75][1]) ==
                  0x0800000000000000ULL,
              "expected compiled v_lshrrev_b64 lane 1 result") ||
      !Expect(compiled_vector_shift64_state.vgprs[74][2] == 0xdeadbeefu &&
                  compiled_vector_shift64_state.vgprs[75][2] == 0xcafebabeu,
              "expected compiled inactive v_lshrrev_b64 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[74][35],
                         compiled_vector_shift64_state.vgprs[75][35]) ==
                  0x0fffffffffffffffULL,
              "expected compiled v_lshrrev_b64 lane 35 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[76][0],
                         compiled_vector_shift64_state.vgprs[77][0]) ==
                  0x00123456789abcdeULL,
              "expected compiled v_ashrrev_i64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[76][1],
                         compiled_vector_shift64_state.vgprs[77][1]) ==
                  0xf800000000000000ULL,
              "expected compiled v_ashrrev_i64 lane 1 result") ||
      !Expect(compiled_vector_shift64_state.vgprs[76][2] == 0xdeadbeefu &&
                  compiled_vector_shift64_state.vgprs[77][2] == 0xcafebabeu,
              "expected compiled inactive v_ashrrev_i64 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[76][35],
                         compiled_vector_shift64_state.vgprs[77][35]) ==
                  0xffffffffffffffffULL,
              "expected compiled v_ashrrev_i64 lane 35 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[78][0],
                         compiled_vector_shift64_state.vgprs[79][0]) ==
                  0x23456789abcdf001ULL,
              "expected compiled v_lshl_add_u64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[78][1],
                         compiled_vector_shift64_state.vgprs[79][1]) == 3ULL,
              "expected compiled v_lshl_add_u64 lane 1 result") ||
      !Expect(compiled_vector_shift64_state.vgprs[78][2] == 0xdeadbeefu &&
                  compiled_vector_shift64_state.vgprs[79][2] == 0xcafebabeu,
              "expected compiled inactive v_lshl_add_u64 result") ||
      !Expect(ComposeU64(compiled_vector_shift64_state.vgprs[78][35],
                         compiled_vector_shift64_state.vgprs[79][35]) ==
                  0xffffffffffffff10ULL,
              "expected compiled v_lshl_add_u64 lane 35 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_mad64_program = {
      DecodedInstruction::FiveOperand("V_MAD_U64_U32",
                                      InstructionOperand::Vgpr(86),
                                      InstructionOperand::Sgpr(118),
                                      InstructionOperand::Sgpr(46),
                                      InstructionOperand::Vgpr(88),
                                      InstructionOperand::Vgpr(82)),
      DecodedInstruction::FiveOperand("V_MAD_I64_I32",
                                      InstructionOperand::Vgpr(90),
                                      InstructionOperand::Sgpr(106),
                                      InstructionOperand::Sgpr(47),
                                      InstructionOperand::Vgpr(89),
                                      InstructionOperand::Vgpr(84)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_mad64_state;
  vector_mad64_state.exec_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  vector_mad64_state.vcc_mask = 0x4ULL;
  vector_mad64_state.sgprs[46] = 0xffffffffu;
  vector_mad64_state.sgprs[47] = 0x7fffffffu;
  vector_mad64_state.sgprs[106] = 0x00000004u;
  vector_mad64_state.sgprs[107] = 0u;
  vector_mad64_state.sgprs[118] = 0x00000008u;
  vector_mad64_state.sgprs[119] = 0u;
  vector_mad64_state.vgprs[88][0] = 2u;
  vector_mad64_state.vgprs[88][1] = 1u;
  vector_mad64_state.vgprs[88][35] = 0u;
  vector_mad64_state.vgprs[89][0] = 2u;
  vector_mad64_state.vgprs[89][1] = 0u;
  vector_mad64_state.vgprs[89][35] = 1u;
  SplitU64(0x0000000000000001ULL, &vector_mad64_state.vgprs[82][0],
           &vector_mad64_state.vgprs[83][0]);
  SplitU64(0xffffffffffffffffULL, &vector_mad64_state.vgprs[82][1],
           &vector_mad64_state.vgprs[83][1]);
  SplitU64(0x0000000000000005ULL, &vector_mad64_state.vgprs[82][35],
           &vector_mad64_state.vgprs[83][35]);
  SplitU64(0x7fffffff00000002ULL, &vector_mad64_state.vgprs[84][0],
           &vector_mad64_state.vgprs[85][0]);
  SplitU64(0x0000000000000005ULL, &vector_mad64_state.vgprs[84][1],
           &vector_mad64_state.vgprs[85][1]);
  SplitU64(0x000000000000000aULL, &vector_mad64_state.vgprs[84][35],
           &vector_mad64_state.vgprs[85][35]);
  vector_mad64_state.vgprs[86][2] = 0xdeadbeefu;
  vector_mad64_state.vgprs[87][2] = 0xcafebabeu;
  vector_mad64_state.vgprs[90][2] = 0xdeadbeefu;
  vector_mad64_state.vgprs[91][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(vector_mad64_program, &vector_mad64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_mad64_state.halted,
              "expected vector mad64 program to halt") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][0],
                         vector_mad64_state.vgprs[87][0]) ==
                  0x00000001ffffffffULL,
              "expected v_mad_u64_u32 lane 0 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][1],
                         vector_mad64_state.vgprs[87][1]) ==
                  0x00000000fffffffeULL,
              "expected v_mad_u64_u32 lane 1 result") ||
      !Expect(vector_mad64_state.vgprs[86][2] == 0xdeadbeefu &&
                  vector_mad64_state.vgprs[87][2] == 0xcafebabeu,
              "expected inactive v_mad_u64_u32 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][35],
                         vector_mad64_state.vgprs[87][35]) ==
                  0x0000000000000005ULL,
              "expected v_mad_u64_u32 lane 35 result") ||
      !Expect(vector_mad64_state.sgprs[118] == 0x0000000au &&
                  vector_mad64_state.sgprs[119] == 0u,
              "expected v_mad_u64_u32 sdst mask") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][0],
                         vector_mad64_state.vgprs[91][0]) ==
                  0x8000000000000000ULL,
              "expected v_mad_i64_i32 lane 0 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][1],
                         vector_mad64_state.vgprs[91][1]) ==
                  0x0000000000000005ULL,
              "expected v_mad_i64_i32 lane 1 result") ||
      !Expect(vector_mad64_state.vgprs[90][2] == 0xdeadbeefu &&
                  vector_mad64_state.vgprs[91][2] == 0xcafebabeu,
              "expected inactive v_mad_i64_i32 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][35],
                         vector_mad64_state.vgprs[91][35]) ==
                  0x0000000080000009ULL,
              "expected v_mad_i64_i32 lane 35 result") ||
      !Expect(vector_mad64_state.sgprs[106] == 0x00000005u &&
                  vector_mad64_state.sgprs[107] == 0u,
              "expected v_mad_i64_i32 sdst mask") ||
      !Expect(vector_mad64_state.vcc_mask == 0x0000000000000005ULL,
              "expected final vcc mask after v_mad_i64_i32")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_mad64_program;
  if (!Expect(interpreter.CompileProgram(vector_mad64_program,
                                         &compiled_vector_mad64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_mad64_state;
  compiled_vector_mad64_state.exec_mask =
      (1ULL << 0) | (1ULL << 1) | (1ULL << 35);
  compiled_vector_mad64_state.vcc_mask = 0x4ULL;
  compiled_vector_mad64_state.sgprs[46] = 0xffffffffu;
  compiled_vector_mad64_state.sgprs[47] = 0x7fffffffu;
  compiled_vector_mad64_state.sgprs[106] = 0x00000004u;
  compiled_vector_mad64_state.sgprs[107] = 0u;
  compiled_vector_mad64_state.sgprs[118] = 0x00000008u;
  compiled_vector_mad64_state.sgprs[119] = 0u;
  compiled_vector_mad64_state.vgprs[88][0] = 2u;
  compiled_vector_mad64_state.vgprs[88][1] = 1u;
  compiled_vector_mad64_state.vgprs[88][35] = 0u;
  compiled_vector_mad64_state.vgprs[89][0] = 2u;
  compiled_vector_mad64_state.vgprs[89][1] = 0u;
  compiled_vector_mad64_state.vgprs[89][35] = 1u;
  SplitU64(0x0000000000000001ULL,
           &compiled_vector_mad64_state.vgprs[82][0],
           &compiled_vector_mad64_state.vgprs[83][0]);
  SplitU64(0xffffffffffffffffULL,
           &compiled_vector_mad64_state.vgprs[82][1],
           &compiled_vector_mad64_state.vgprs[83][1]);
  SplitU64(0x0000000000000005ULL,
           &compiled_vector_mad64_state.vgprs[82][35],
           &compiled_vector_mad64_state.vgprs[83][35]);
  SplitU64(0x7fffffff00000002ULL,
           &compiled_vector_mad64_state.vgprs[84][0],
           &compiled_vector_mad64_state.vgprs[85][0]);
  SplitU64(0x0000000000000005ULL,
           &compiled_vector_mad64_state.vgprs[84][1],
           &compiled_vector_mad64_state.vgprs[85][1]);
  SplitU64(0x000000000000000aULL,
           &compiled_vector_mad64_state.vgprs[84][35],
           &compiled_vector_mad64_state.vgprs[85][35]);
  compiled_vector_mad64_state.vgprs[86][2] = 0xdeadbeefu;
  compiled_vector_mad64_state.vgprs[87][2] = 0xcafebabeu;
  compiled_vector_mad64_state.vgprs[90][2] = 0xdeadbeefu;
  compiled_vector_mad64_state.vgprs[91][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_mad64_program,
                                         &compiled_vector_mad64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_mad64_state.halted,
              "expected compiled vector mad64 program to halt") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[86][0],
                         compiled_vector_mad64_state.vgprs[87][0]) ==
                  0x00000001ffffffffULL,
              "expected compiled v_mad_u64_u32 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[86][1],
                         compiled_vector_mad64_state.vgprs[87][1]) ==
                  0x00000000fffffffeULL,
              "expected compiled v_mad_u64_u32 lane 1 result") ||
      !Expect(compiled_vector_mad64_state.vgprs[86][2] == 0xdeadbeefu &&
                  compiled_vector_mad64_state.vgprs[87][2] == 0xcafebabeu,
              "expected compiled inactive v_mad_u64_u32 result") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[86][35],
                         compiled_vector_mad64_state.vgprs[87][35]) ==
                  0x0000000000000005ULL,
              "expected compiled v_mad_u64_u32 lane 35 result") ||
      !Expect(compiled_vector_mad64_state.sgprs[118] == 0x0000000au &&
                  compiled_vector_mad64_state.sgprs[119] == 0u,
              "expected compiled v_mad_u64_u32 sdst mask") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[90][0],
                         compiled_vector_mad64_state.vgprs[91][0]) ==
                  0x8000000000000000ULL,
              "expected compiled v_mad_i64_i32 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[90][1],
                         compiled_vector_mad64_state.vgprs[91][1]) ==
                  0x0000000000000005ULL,
              "expected compiled v_mad_i64_i32 lane 1 result") ||
      !Expect(compiled_vector_mad64_state.vgprs[90][2] == 0xdeadbeefu &&
                  compiled_vector_mad64_state.vgprs[91][2] == 0xcafebabeu,
              "expected compiled inactive v_mad_i64_i32 result") ||
      !Expect(ComposeU64(compiled_vector_mad64_state.vgprs[90][35],
                         compiled_vector_mad64_state.vgprs[91][35]) ==
                  0x0000000080000009ULL,
              "expected compiled v_mad_i64_i32 lane 35 result") ||
      !Expect(compiled_vector_mad64_state.sgprs[106] == 0x00000005u &&
                  compiled_vector_mad64_state.sgprs[107] == 0u,
              "expected compiled v_mad_i64_i32 sdst mask") ||
      !Expect(compiled_vector_mad64_state.vcc_mask == 0x0000000000000005ULL,
              "expected compiled final vcc mask after v_mad_i64_i32")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_ternary_logic_program = {
      DecodedInstruction::Ternary("V_ADD_LSHL_U32",
                                  InstructionOperand::Vgpr(82),
                                  InstructionOperand::Vgpr(83),
                                  InstructionOperand::Sgpr(43),
                                  InstructionOperand::Vgpr(84)),
      DecodedInstruction::Ternary("V_LSHL_OR_B32", InstructionOperand::Vgpr(85),
                                  InstructionOperand::Vgpr(86),
                                  InstructionOperand::Sgpr(44),
                                  InstructionOperand::Vgpr(87)),
      DecodedInstruction::Ternary("V_AND_OR_B32", InstructionOperand::Vgpr(88),
                                  InstructionOperand::Vgpr(89),
                                  InstructionOperand::Sgpr(45),
                                  InstructionOperand::Vgpr(90)),
      DecodedInstruction::Ternary("V_OR3_B32", InstructionOperand::Vgpr(91),
                                  InstructionOperand::Vgpr(92),
                                  InstructionOperand::Sgpr(46),
                                  InstructionOperand::Vgpr(93)),
      DecodedInstruction::Ternary("V_XAD_U32", InstructionOperand::Vgpr(94),
                                  InstructionOperand::Vgpr(95),
                                  InstructionOperand::Sgpr(47),
                                  InstructionOperand::Vgpr(96)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_ternary_logic_state;
  vector_ternary_logic_state.exec_mask = 0b1011ULL;
  vector_ternary_logic_state.sgprs[43] = 3u;
  vector_ternary_logic_state.sgprs[44] = 4u;
  vector_ternary_logic_state.sgprs[45] = 0x0ff00ff0u;
  vector_ternary_logic_state.sgprs[46] = 0x00ff0000u;
  vector_ternary_logic_state.sgprs[47] = 0x0f0f0f0fu;
  vector_ternary_logic_state.vgprs[83][0] = 5u;
  vector_ternary_logic_state.vgprs[83][1] = 0xfffffffeu;
  vector_ternary_logic_state.vgprs[83][3] = 10u;
  vector_ternary_logic_state.vgprs[84][0] = 1u;
  vector_ternary_logic_state.vgprs[84][1] = 2u;
  vector_ternary_logic_state.vgprs[84][3] = 4u;
  vector_ternary_logic_state.vgprs[86][0] = 0x12u;
  vector_ternary_logic_state.vgprs[86][1] = 1u;
  vector_ternary_logic_state.vgprs[86][3] = 0x80000000u;
  vector_ternary_logic_state.vgprs[87][0] = 0x00000003u;
  vector_ternary_logic_state.vgprs[87][1] = 0xffff0000u;
  vector_ternary_logic_state.vgprs[87][3] = 0x0000000fu;
  vector_ternary_logic_state.vgprs[89][0] = 0xf0f0f0f0u;
  vector_ternary_logic_state.vgprs[89][1] = 0xaaaaaaaau;
  vector_ternary_logic_state.vgprs[89][3] = 0x0f0f0f0fu;
  vector_ternary_logic_state.vgprs[90][0] = 0x0000000fu;
  vector_ternary_logic_state.vgprs[90][1] = 0x000000f0u;
  vector_ternary_logic_state.vgprs[90][3] = 0xf0000000u;
  vector_ternary_logic_state.vgprs[92][0] = 0x0000f000u;
  vector_ternary_logic_state.vgprs[92][1] = 0xf0000000u;
  vector_ternary_logic_state.vgprs[92][3] = 0x00000000u;
  vector_ternary_logic_state.vgprs[93][0] = 0x0000000fu;
  vector_ternary_logic_state.vgprs[93][1] = 0x0000f000u;
  vector_ternary_logic_state.vgprs[93][3] = 0x0f0f0f0fu;
  vector_ternary_logic_state.vgprs[95][0] = 0xffffffffu;
  vector_ternary_logic_state.vgprs[95][1] = 0x12345678u;
  vector_ternary_logic_state.vgprs[95][3] = 0x00000000u;
  vector_ternary_logic_state.vgprs[96][0] = 1u;
  vector_ternary_logic_state.vgprs[96][1] = 2u;
  vector_ternary_logic_state.vgprs[96][3] = 0xfffffff0u;
  vector_ternary_logic_state.vgprs[82][2] = 0xdeadbeefu;
  vector_ternary_logic_state.vgprs[85][2] = 0xdeadbeefu;
  vector_ternary_logic_state.vgprs[88][2] = 0xdeadbeefu;
  vector_ternary_logic_state.vgprs[91][2] = 0xdeadbeefu;
  vector_ternary_logic_state.vgprs[94][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_ternary_logic_program,
                                         &vector_ternary_logic_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_ternary_logic_state.halted,
              "expected vector ternary logic program to halt") ||
      !Expect(vector_ternary_logic_state.vgprs[82][0] == 16u,
              "expected v_add_lshl_u32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][1] == 4u,
              "expected v_add_lshl_u32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][2] == 0xdeadbeefu,
              "expected inactive v_add_lshl_u32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][3] == 208u,
              "expected v_add_lshl_u32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][0] == 0x00000123u,
              "expected v_lshl_or_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][1] == 0xffff0010u,
              "expected v_lshl_or_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][2] == 0xdeadbeefu,
              "expected inactive v_lshl_or_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][3] == 0x0000000fu,
              "expected v_lshl_or_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][0] == 0x00f000ffu,
              "expected v_and_or_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][1] == 0x0aa00af0u,
              "expected v_and_or_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][2] == 0xdeadbeefu,
              "expected inactive v_and_or_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][3] == 0xff000f00u,
              "expected v_and_or_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][0] == 0x00fff00fu,
              "expected v_or3_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][1] == 0xf0fff000u,
              "expected v_or3_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][2] == 0xdeadbeefu,
              "expected inactive v_or3_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][3] == 0x0fff0f0fu,
              "expected v_or3_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][0] == 0xf0f0f0f1u,
              "expected v_xad_u32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][1] == 0x1d3b5979u,
              "expected v_xad_u32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][2] == 0xdeadbeefu,
              "expected inactive v_xad_u32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][3] == 0x0f0f0effu,
              "expected v_xad_u32 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_ternary_logic_program;
  if (!Expect(interpreter.CompileProgram(vector_ternary_logic_program,
                                         &compiled_vector_ternary_logic_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_ternary_logic_state;
  compiled_vector_ternary_logic_state.exec_mask = 0b1011ULL;
  compiled_vector_ternary_logic_state.sgprs[43] = 3u;
  compiled_vector_ternary_logic_state.sgprs[44] = 4u;
  compiled_vector_ternary_logic_state.sgprs[45] = 0x0ff00ff0u;
  compiled_vector_ternary_logic_state.sgprs[46] = 0x00ff0000u;
  compiled_vector_ternary_logic_state.sgprs[47] = 0x0f0f0f0fu;
  compiled_vector_ternary_logic_state.vgprs[83][0] = 5u;
  compiled_vector_ternary_logic_state.vgprs[83][1] = 0xfffffffeu;
  compiled_vector_ternary_logic_state.vgprs[83][3] = 10u;
  compiled_vector_ternary_logic_state.vgprs[84][0] = 1u;
  compiled_vector_ternary_logic_state.vgprs[84][1] = 2u;
  compiled_vector_ternary_logic_state.vgprs[84][3] = 4u;
  compiled_vector_ternary_logic_state.vgprs[86][0] = 0x12u;
  compiled_vector_ternary_logic_state.vgprs[86][1] = 1u;
  compiled_vector_ternary_logic_state.vgprs[86][3] = 0x80000000u;
  compiled_vector_ternary_logic_state.vgprs[87][0] = 0x00000003u;
  compiled_vector_ternary_logic_state.vgprs[87][1] = 0xffff0000u;
  compiled_vector_ternary_logic_state.vgprs[87][3] = 0x0000000fu;
  compiled_vector_ternary_logic_state.vgprs[89][0] = 0xf0f0f0f0u;
  compiled_vector_ternary_logic_state.vgprs[89][1] = 0xaaaaaaaau;
  compiled_vector_ternary_logic_state.vgprs[89][3] = 0x0f0f0f0fu;
  compiled_vector_ternary_logic_state.vgprs[90][0] = 0x0000000fu;
  compiled_vector_ternary_logic_state.vgprs[90][1] = 0x000000f0u;
  compiled_vector_ternary_logic_state.vgprs[90][3] = 0xf0000000u;
  compiled_vector_ternary_logic_state.vgprs[92][0] = 0x0000f000u;
  compiled_vector_ternary_logic_state.vgprs[92][1] = 0xf0000000u;
  compiled_vector_ternary_logic_state.vgprs[92][3] = 0x00000000u;
  compiled_vector_ternary_logic_state.vgprs[93][0] = 0x0000000fu;
  compiled_vector_ternary_logic_state.vgprs[93][1] = 0x0000f000u;
  compiled_vector_ternary_logic_state.vgprs[93][3] = 0x0f0f0f0fu;
  compiled_vector_ternary_logic_state.vgprs[95][0] = 0xffffffffu;
  compiled_vector_ternary_logic_state.vgprs[95][1] = 0x12345678u;
  compiled_vector_ternary_logic_state.vgprs[95][3] = 0x00000000u;
  compiled_vector_ternary_logic_state.vgprs[96][0] = 1u;
  compiled_vector_ternary_logic_state.vgprs[96][1] = 2u;
  compiled_vector_ternary_logic_state.vgprs[96][3] = 0xfffffff0u;
  compiled_vector_ternary_logic_state.vgprs[82][2] = 0xdeadbeefu;
  compiled_vector_ternary_logic_state.vgprs[85][2] = 0xdeadbeefu;
  compiled_vector_ternary_logic_state.vgprs[88][2] = 0xdeadbeefu;
  compiled_vector_ternary_logic_state.vgprs[91][2] = 0xdeadbeefu;
  compiled_vector_ternary_logic_state.vgprs[94][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_ternary_logic_program,
                                         &compiled_vector_ternary_logic_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_ternary_logic_state.halted,
              "expected compiled vector ternary logic program to halt") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[82][0] == 16u,
              "expected compiled v_add_lshl_u32 lane 0 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[82][1] == 4u,
              "expected compiled v_add_lshl_u32 lane 1 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[82][2] == 0xdeadbeefu,
              "expected compiled inactive v_add_lshl_u32 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[82][3] == 208u,
              "expected compiled v_add_lshl_u32 lane 3 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[85][0] == 0x00000123u,
              "expected compiled v_lshl_or_b32 lane 0 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[85][1] == 0xffff0010u,
              "expected compiled v_lshl_or_b32 lane 1 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[85][2] == 0xdeadbeefu,
              "expected compiled inactive v_lshl_or_b32 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[85][3] == 0x0000000fu,
              "expected compiled v_lshl_or_b32 lane 3 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[88][0] == 0x00f000ffu,
              "expected compiled v_and_or_b32 lane 0 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[88][1] == 0x0aa00af0u,
              "expected compiled v_and_or_b32 lane 1 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[88][2] == 0xdeadbeefu,
              "expected compiled inactive v_and_or_b32 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[88][3] == 0xff000f00u,
              "expected compiled v_and_or_b32 lane 3 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[91][0] == 0x00fff00fu,
              "expected compiled v_or3_b32 lane 0 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[91][1] == 0xf0fff000u,
              "expected compiled v_or3_b32 lane 1 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[91][2] == 0xdeadbeefu,
              "expected compiled inactive v_or3_b32 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[91][3] == 0x0fff0f0fu,
              "expected compiled v_or3_b32 lane 3 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[94][0] == 0xf0f0f0f1u,
              "expected compiled v_xad_u32 lane 0 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[94][1] == 0x1d3b5979u,
              "expected compiled v_xad_u32 lane 1 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[94][2] == 0xdeadbeefu,
              "expected compiled inactive v_xad_u32 result") ||
      !Expect(compiled_vector_ternary_logic_state.vgprs[94][3] == 0x0f0f0effu,
              "expected compiled v_xad_u32 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_binary_extra_program = {
      DecodedInstruction::Binary("V_SUBREV_U32", InstructionOperand::Vgpr(98),
                                 InstructionOperand::Sgpr(48),
                                 InstructionOperand::Vgpr(97)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_binary_extra_state;
  vector_binary_extra_state.exec_mask = 0b1011ULL;
  vector_binary_extra_state.sgprs[48] = 10u;
  vector_binary_extra_state.vgprs[97][0] = 20u;
  vector_binary_extra_state.vgprs[97][1] = 5u;
  vector_binary_extra_state.vgprs[97][3] = 0u;
  vector_binary_extra_state.vgprs[98][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_binary_extra_program,
                                         &vector_binary_extra_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_binary_extra_state.halted,
              "expected vector binary extra program to halt") ||
      !Expect(vector_binary_extra_state.vgprs[98][0] == 10u,
              "expected v_subrev_u32 lane 0 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][1] == 0xfffffffbu,
              "expected v_subrev_u32 lane 1 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][2] == 0xdeadbeefu,
              "expected inactive v_subrev_u32 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][3] == 0xfffffff6u,
              "expected v_subrev_u32 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_binary_extra_program;
  if (!Expect(interpreter.CompileProgram(vector_binary_extra_program,
                                         &compiled_vector_binary_extra_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_binary_extra_state;
  compiled_vector_binary_extra_state.exec_mask = 0b1011ULL;
  compiled_vector_binary_extra_state.sgprs[48] = 10u;
  compiled_vector_binary_extra_state.vgprs[97][0] = 20u;
  compiled_vector_binary_extra_state.vgprs[97][1] = 5u;
  compiled_vector_binary_extra_state.vgprs[97][3] = 0u;
  compiled_vector_binary_extra_state.vgprs[98][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_binary_extra_program,
                                         &compiled_vector_binary_extra_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_binary_extra_state.halted,
              "expected compiled vector binary extra program to halt") ||
      !Expect(compiled_vector_binary_extra_state.vgprs[98][0] == 10u,
              "expected compiled v_subrev_u32 lane 0 result") ||
      !Expect(compiled_vector_binary_extra_state.vgprs[98][1] == 0xfffffffbu,
              "expected compiled v_subrev_u32 lane 1 result") ||
      !Expect(compiled_vector_binary_extra_state.vgprs[98][2] == 0xdeadbeefu,
              "expected compiled inactive v_subrev_u32 result") ||
      !Expect(compiled_vector_binary_extra_state.vgprs[98][3] == 0xfffffff6u,
              "expected compiled v_subrev_u32 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_float_f32_program = {
      DecodedInstruction::Binary("V_ADD_F32", InstructionOperand::Vgpr(30),
                                 InstructionOperand::Sgpr(60),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Binary("V_SUB_F32", InstructionOperand::Vgpr(31),
                                 InstructionOperand::Sgpr(61),
                                 InstructionOperand::Vgpr(21)),
      DecodedInstruction::Binary("V_MUL_F32", InstructionOperand::Vgpr(32),
                                 InstructionOperand::Sgpr(62),
                                 InstructionOperand::Vgpr(22)),
      DecodedInstruction::Binary("V_MIN_F32", InstructionOperand::Vgpr(33),
                                 InstructionOperand::Sgpr(63),
                                 InstructionOperand::Vgpr(23)),
      DecodedInstruction::Binary("V_MAX_F32", InstructionOperand::Vgpr(34),
                                 InstructionOperand::Sgpr(64),
                                 InstructionOperand::Vgpr(24)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_float_f32_state;
  vector_float_f32_state.exec_mask = 0b1011ULL;
  vector_float_f32_state.sgprs[60] = FloatBits(1.5f);
  vector_float_f32_state.sgprs[61] = FloatBits(5.0f);
  vector_float_f32_state.sgprs[62] = FloatBits(-2.0f);
  vector_float_f32_state.sgprs[63] = FloatBits(1.0f);
  vector_float_f32_state.sgprs[64] = FloatBits(1.5f);
  vector_float_f32_state.vgprs[20][0] = FloatBits(2.0f);
  vector_float_f32_state.vgprs[20][1] = FloatBits(-2.25f);
  vector_float_f32_state.vgprs[20][3] = FloatBits(0.5f);
  vector_float_f32_state.vgprs[21][0] = FloatBits(1.25f);
  vector_float_f32_state.vgprs[21][1] = FloatBits(8.0f);
  vector_float_f32_state.vgprs[21][3] = FloatBits(-0.5f);
  vector_float_f32_state.vgprs[22][0] = FloatBits(1.5f);
  vector_float_f32_state.vgprs[22][1] = FloatBits(-0.5f);
  vector_float_f32_state.vgprs[22][3] = FloatBits(4.0f);
  vector_float_f32_state.vgprs[23][0] = FloatBits(0.0f);
  vector_float_f32_state.vgprs[23][1] = FloatBits(2.0f);
  vector_float_f32_state.vgprs[23][3] = FloatBits(-5.0f);
  vector_float_f32_state.vgprs[24][0] = FloatBits(2.0f);
  vector_float_f32_state.vgprs[24][1] = FloatBits(-3.0f);
  vector_float_f32_state.vgprs[24][3] = FloatBits(1.5f);
  vector_float_f32_state.vgprs[30][2] = 0xdeadbeefu;
  vector_float_f32_state.vgprs[31][2] = 0xdeadbeefu;
  vector_float_f32_state.vgprs[32][2] = 0xdeadbeefu;
  vector_float_f32_state.vgprs[33][2] = 0xdeadbeefu;
  vector_float_f32_state.vgprs[34][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_float_f32_program,
                                         &vector_float_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_float_f32_state.halted,
              "expected vector float f32 program to halt") ||
      !Expect(vector_float_f32_state.vgprs[30][0] == FloatBits(3.5f),
              "expected decoded v_add_f32 lane 0 result") ||
      !Expect(vector_float_f32_state.vgprs[30][1] == FloatBits(-0.75f),
              "expected decoded v_add_f32 lane 1 result") ||
      !Expect(vector_float_f32_state.vgprs[30][2] == 0xdeadbeefu,
              "expected inactive decoded v_add_f32 result") ||
      !Expect(vector_float_f32_state.vgprs[30][3] == FloatBits(2.0f),
              "expected decoded v_add_f32 lane 3 result") ||
      !Expect(vector_float_f32_state.vgprs[31][0] == FloatBits(3.75f),
              "expected decoded v_sub_f32 lane 0 result") ||
      !Expect(vector_float_f32_state.vgprs[31][1] == FloatBits(-3.0f),
              "expected decoded v_sub_f32 lane 1 result") ||
      !Expect(vector_float_f32_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive decoded v_sub_f32 result") ||
      !Expect(vector_float_f32_state.vgprs[31][3] == FloatBits(5.5f),
              "expected decoded v_sub_f32 lane 3 result") ||
      !Expect(vector_float_f32_state.vgprs[32][0] == FloatBits(-3.0f),
              "expected decoded v_mul_f32 lane 0 result") ||
      !Expect(vector_float_f32_state.vgprs[32][1] == FloatBits(1.0f),
              "expected decoded v_mul_f32 lane 1 result") ||
      !Expect(vector_float_f32_state.vgprs[32][2] == 0xdeadbeefu,
              "expected inactive decoded v_mul_f32 result") ||
      !Expect(vector_float_f32_state.vgprs[32][3] == FloatBits(-8.0f),
              "expected decoded v_mul_f32 lane 3 result") ||
      !Expect(vector_float_f32_state.vgprs[33][0] == FloatBits(0.0f),
              "expected decoded v_min_f32 lane 0 result") ||
      !Expect(vector_float_f32_state.vgprs[33][1] == FloatBits(1.0f),
              "expected decoded v_min_f32 lane 1 result") ||
      !Expect(vector_float_f32_state.vgprs[33][2] == 0xdeadbeefu,
              "expected inactive decoded v_min_f32 result") ||
      !Expect(vector_float_f32_state.vgprs[33][3] == FloatBits(-5.0f),
              "expected decoded v_min_f32 lane 3 result") ||
      !Expect(vector_float_f32_state.vgprs[34][0] == FloatBits(2.0f),
              "expected decoded v_max_f32 lane 0 result") ||
      !Expect(vector_float_f32_state.vgprs[34][1] == FloatBits(1.5f),
              "expected decoded v_max_f32 lane 1 result") ||
      !Expect(vector_float_f32_state.vgprs[34][2] == 0xdeadbeefu,
              "expected inactive decoded v_max_f32 result") ||
      !Expect(vector_float_f32_state.vgprs[34][3] == FloatBits(1.5f),
              "expected decoded v_max_f32 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_float_f32_program;
  if (!Expect(interpreter.CompileProgram(vector_float_f32_program,
                                         &compiled_vector_float_f32_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_float_f32_state;
  compiled_vector_float_f32_state.exec_mask = 0b1011ULL;
  compiled_vector_float_f32_state.sgprs[60] = FloatBits(1.5f);
  compiled_vector_float_f32_state.sgprs[61] = FloatBits(5.0f);
  compiled_vector_float_f32_state.sgprs[62] = FloatBits(-2.0f);
  compiled_vector_float_f32_state.sgprs[63] = FloatBits(1.0f);
  compiled_vector_float_f32_state.sgprs[64] = FloatBits(1.5f);
  compiled_vector_float_f32_state.vgprs[20][0] = FloatBits(2.0f);
  compiled_vector_float_f32_state.vgprs[20][1] = FloatBits(-2.25f);
  compiled_vector_float_f32_state.vgprs[20][3] = FloatBits(0.5f);
  compiled_vector_float_f32_state.vgprs[21][0] = FloatBits(1.25f);
  compiled_vector_float_f32_state.vgprs[21][1] = FloatBits(8.0f);
  compiled_vector_float_f32_state.vgprs[21][3] = FloatBits(-0.5f);
  compiled_vector_float_f32_state.vgprs[22][0] = FloatBits(1.5f);
  compiled_vector_float_f32_state.vgprs[22][1] = FloatBits(-0.5f);
  compiled_vector_float_f32_state.vgprs[22][3] = FloatBits(4.0f);
  compiled_vector_float_f32_state.vgprs[23][0] = FloatBits(0.0f);
  compiled_vector_float_f32_state.vgprs[23][1] = FloatBits(2.0f);
  compiled_vector_float_f32_state.vgprs[23][3] = FloatBits(-5.0f);
  compiled_vector_float_f32_state.vgprs[24][0] = FloatBits(2.0f);
  compiled_vector_float_f32_state.vgprs[24][1] = FloatBits(-3.0f);
  compiled_vector_float_f32_state.vgprs[24][3] = FloatBits(1.5f);
  compiled_vector_float_f32_state.vgprs[30][2] = 0xdeadbeefu;
  compiled_vector_float_f32_state.vgprs[31][2] = 0xdeadbeefu;
  compiled_vector_float_f32_state.vgprs[32][2] = 0xdeadbeefu;
  compiled_vector_float_f32_state.vgprs[33][2] = 0xdeadbeefu;
  compiled_vector_float_f32_state.vgprs[34][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_float_f32_program,
                                         &compiled_vector_float_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_float_f32_state.halted,
              "expected compiled vector float f32 program to halt") ||
      !Expect(compiled_vector_float_f32_state.vgprs[30][0] == FloatBits(3.5f),
              "expected compiled v_add_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[30][1] ==
                  FloatBits(-0.75f),
              "expected compiled v_add_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[30][2] == 0xdeadbeefu,
              "expected inactive compiled v_add_f32 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[30][3] == FloatBits(2.0f),
              "expected compiled v_add_f32 lane 3 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[31][0] ==
                  FloatBits(3.75f),
              "expected compiled v_sub_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[31][1] == FloatBits(-3.0f),
              "expected compiled v_sub_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive compiled v_sub_f32 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[31][3] == FloatBits(5.5f),
              "expected compiled v_sub_f32 lane 3 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[32][0] == FloatBits(-3.0f),
              "expected compiled v_mul_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[32][1] == FloatBits(1.0f),
              "expected compiled v_mul_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[32][2] == 0xdeadbeefu,
              "expected inactive compiled v_mul_f32 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[32][3] == FloatBits(-8.0f),
              "expected compiled v_mul_f32 lane 3 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[33][0] == FloatBits(0.0f),
              "expected compiled v_min_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[33][1] == FloatBits(1.0f),
              "expected compiled v_min_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[33][2] == 0xdeadbeefu,
              "expected inactive compiled v_min_f32 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[33][3] == FloatBits(-5.0f),
              "expected compiled v_min_f32 lane 3 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[34][0] == FloatBits(2.0f),
              "expected compiled v_max_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[34][1] == FloatBits(1.5f),
              "expected compiled v_max_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[34][2] == 0xdeadbeefu,
              "expected inactive compiled v_max_f32 result") ||
      !Expect(compiled_vector_float_f32_state.vgprs[34][3] == FloatBits(1.5f),
              "expected compiled v_max_f32 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_float_f64_program = {
      DecodedInstruction::Binary("V_ADD_F64", InstructionOperand::Vgpr(50),
                                 InstructionOperand::Sgpr(80),
                                 InstructionOperand::Vgpr(40)),
      DecodedInstruction::Binary("V_MUL_F64", InstructionOperand::Vgpr(52),
                                 InstructionOperand::Sgpr(82),
                                 InstructionOperand::Vgpr(42)),
      DecodedInstruction::Ternary("V_FMA_F32", InstructionOperand::Vgpr(54),
                                  InstructionOperand::Sgpr(65),
                                  InstructionOperand::Vgpr(44),
                                  InstructionOperand::Vgpr(45)),
      DecodedInstruction::Ternary("V_FMA_F64", InstructionOperand::Vgpr(56),
                                  InstructionOperand::Sgpr(84),
                                  InstructionOperand::Vgpr(46),
                                  InstructionOperand::Vgpr(48)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_float_f64_state;
  vector_float_f64_state.exec_mask = 0b1011ULL;
  vector_float_f64_state.sgprs[65] = FloatBits(1.5f);
  SplitU64(DoubleBits(1.25), &vector_float_f64_state.sgprs[80],
           &vector_float_f64_state.sgprs[81]);
  SplitU64(DoubleBits(-2.0), &vector_float_f64_state.sgprs[82],
           &vector_float_f64_state.sgprs[83]);
  SplitU64(DoubleBits(1.5), &vector_float_f64_state.sgprs[84],
           &vector_float_f64_state.sgprs[85]);
  SplitU64(DoubleBits(2.5), &vector_float_f64_state.vgprs[40][0],
           &vector_float_f64_state.vgprs[41][0]);
  SplitU64(DoubleBits(-0.25), &vector_float_f64_state.vgprs[40][1],
           &vector_float_f64_state.vgprs[41][1]);
  SplitU64(DoubleBits(0.75), &vector_float_f64_state.vgprs[40][3],
           &vector_float_f64_state.vgprs[41][3]);
  SplitU64(DoubleBits(1.5), &vector_float_f64_state.vgprs[42][0],
           &vector_float_f64_state.vgprs[43][0]);
  SplitU64(DoubleBits(-0.5), &vector_float_f64_state.vgprs[42][1],
           &vector_float_f64_state.vgprs[43][1]);
  SplitU64(DoubleBits(4.0), &vector_float_f64_state.vgprs[42][3],
           &vector_float_f64_state.vgprs[43][3]);
  vector_float_f64_state.vgprs[44][0] = FloatBits(2.0f);
  vector_float_f64_state.vgprs[44][1] = FloatBits(-2.0f);
  vector_float_f64_state.vgprs[44][3] = FloatBits(4.0f);
  vector_float_f64_state.vgprs[45][0] = FloatBits(0.5f);
  vector_float_f64_state.vgprs[45][1] = FloatBits(1.0f);
  vector_float_f64_state.vgprs[45][3] = FloatBits(-1.0f);
  SplitU64(DoubleBits(2.0), &vector_float_f64_state.vgprs[46][0],
           &vector_float_f64_state.vgprs[47][0]);
  SplitU64(DoubleBits(-2.0), &vector_float_f64_state.vgprs[46][1],
           &vector_float_f64_state.vgprs[47][1]);
  SplitU64(DoubleBits(4.0), &vector_float_f64_state.vgprs[46][3],
           &vector_float_f64_state.vgprs[47][3]);
  SplitU64(DoubleBits(0.5), &vector_float_f64_state.vgprs[48][0],
           &vector_float_f64_state.vgprs[49][0]);
  SplitU64(DoubleBits(1.0), &vector_float_f64_state.vgprs[48][1],
           &vector_float_f64_state.vgprs[49][1]);
  SplitU64(DoubleBits(-1.0), &vector_float_f64_state.vgprs[48][3],
           &vector_float_f64_state.vgprs[49][3]);
  vector_float_f64_state.vgprs[50][2] = 0xdeadbeefu;
  vector_float_f64_state.vgprs[51][2] = 0xcafebabeu;
  vector_float_f64_state.vgprs[52][2] = 0xdeadbeefu;
  vector_float_f64_state.vgprs[53][2] = 0xcafebabeu;
  vector_float_f64_state.vgprs[54][2] = 0xdeadbeefu;
  vector_float_f64_state.vgprs[56][2] = 0xdeadbeefu;
  vector_float_f64_state.vgprs[57][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(vector_float_f64_program,
                                         &vector_float_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_float_f64_state.halted,
              "expected vector float f64 program to halt") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[50][0],
                         vector_float_f64_state.vgprs[51][0]) ==
                  DoubleBits(3.75),
              "expected decoded v_add_f64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[50][1],
                         vector_float_f64_state.vgprs[51][1]) ==
                  DoubleBits(1.0),
              "expected decoded v_add_f64 lane 1 result") ||
      !Expect(vector_float_f64_state.vgprs[50][2] == 0xdeadbeefu &&
                  vector_float_f64_state.vgprs[51][2] == 0xcafebabeu,
              "expected inactive decoded v_add_f64 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[50][3],
                         vector_float_f64_state.vgprs[51][3]) ==
                  DoubleBits(2.0),
              "expected decoded v_add_f64 lane 3 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[52][0],
                         vector_float_f64_state.vgprs[53][0]) ==
                  DoubleBits(-3.0),
              "expected decoded v_mul_f64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[52][1],
                         vector_float_f64_state.vgprs[53][1]) ==
                  DoubleBits(1.0),
              "expected decoded v_mul_f64 lane 1 result") ||
      !Expect(vector_float_f64_state.vgprs[52][2] == 0xdeadbeefu &&
                  vector_float_f64_state.vgprs[53][2] == 0xcafebabeu,
              "expected inactive decoded v_mul_f64 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[52][3],
                         vector_float_f64_state.vgprs[53][3]) ==
                  DoubleBits(-8.0),
              "expected decoded v_mul_f64 lane 3 result") ||
      !Expect(vector_float_f64_state.vgprs[54][0] == FloatBits(3.5f),
              "expected decoded v_fma_f32 lane 0 result") ||
      !Expect(vector_float_f64_state.vgprs[54][1] == FloatBits(-2.0f),
              "expected decoded v_fma_f32 lane 1 result") ||
      !Expect(vector_float_f64_state.vgprs[54][2] == 0xdeadbeefu,
              "expected inactive decoded v_fma_f32 result") ||
      !Expect(vector_float_f64_state.vgprs[54][3] == FloatBits(5.0f),
              "expected decoded v_fma_f32 lane 3 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[56][0],
                         vector_float_f64_state.vgprs[57][0]) ==
                  DoubleBits(3.5),
              "expected decoded v_fma_f64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[56][1],
                         vector_float_f64_state.vgprs[57][1]) ==
                  DoubleBits(-2.0),
              "expected decoded v_fma_f64 lane 1 result") ||
      !Expect(vector_float_f64_state.vgprs[56][2] == 0xdeadbeefu &&
                  vector_float_f64_state.vgprs[57][2] == 0xcafebabeu,
              "expected inactive decoded v_fma_f64 result") ||
      !Expect(ComposeU64(vector_float_f64_state.vgprs[56][3],
                         vector_float_f64_state.vgprs[57][3]) ==
                  DoubleBits(5.0),
              "expected decoded v_fma_f64 lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_float_f64_program;
  if (!Expect(interpreter.CompileProgram(vector_float_f64_program,
                                         &compiled_vector_float_f64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_float_f64_state;
  compiled_vector_float_f64_state.exec_mask = 0b1011ULL;
  compiled_vector_float_f64_state.sgprs[65] = FloatBits(1.5f);
  SplitU64(DoubleBits(1.25), &compiled_vector_float_f64_state.sgprs[80],
           &compiled_vector_float_f64_state.sgprs[81]);
  SplitU64(DoubleBits(-2.0), &compiled_vector_float_f64_state.sgprs[82],
           &compiled_vector_float_f64_state.sgprs[83]);
  SplitU64(DoubleBits(1.5), &compiled_vector_float_f64_state.sgprs[84],
           &compiled_vector_float_f64_state.sgprs[85]);
  SplitU64(DoubleBits(2.5), &compiled_vector_float_f64_state.vgprs[40][0],
           &compiled_vector_float_f64_state.vgprs[41][0]);
  SplitU64(DoubleBits(-0.25), &compiled_vector_float_f64_state.vgprs[40][1],
           &compiled_vector_float_f64_state.vgprs[41][1]);
  SplitU64(DoubleBits(0.75), &compiled_vector_float_f64_state.vgprs[40][3],
           &compiled_vector_float_f64_state.vgprs[41][3]);
  SplitU64(DoubleBits(1.5), &compiled_vector_float_f64_state.vgprs[42][0],
           &compiled_vector_float_f64_state.vgprs[43][0]);
  SplitU64(DoubleBits(-0.5), &compiled_vector_float_f64_state.vgprs[42][1],
           &compiled_vector_float_f64_state.vgprs[43][1]);
  SplitU64(DoubleBits(4.0), &compiled_vector_float_f64_state.vgprs[42][3],
           &compiled_vector_float_f64_state.vgprs[43][3]);
  compiled_vector_float_f64_state.vgprs[44][0] = FloatBits(2.0f);
  compiled_vector_float_f64_state.vgprs[44][1] = FloatBits(-2.0f);
  compiled_vector_float_f64_state.vgprs[44][3] = FloatBits(4.0f);
  compiled_vector_float_f64_state.vgprs[45][0] = FloatBits(0.5f);
  compiled_vector_float_f64_state.vgprs[45][1] = FloatBits(1.0f);
  compiled_vector_float_f64_state.vgprs[45][3] = FloatBits(-1.0f);
  SplitU64(DoubleBits(2.0), &compiled_vector_float_f64_state.vgprs[46][0],
           &compiled_vector_float_f64_state.vgprs[47][0]);
  SplitU64(DoubleBits(-2.0), &compiled_vector_float_f64_state.vgprs[46][1],
           &compiled_vector_float_f64_state.vgprs[47][1]);
  SplitU64(DoubleBits(4.0), &compiled_vector_float_f64_state.vgprs[46][3],
           &compiled_vector_float_f64_state.vgprs[47][3]);
  SplitU64(DoubleBits(0.5), &compiled_vector_float_f64_state.vgprs[48][0],
           &compiled_vector_float_f64_state.vgprs[49][0]);
  SplitU64(DoubleBits(1.0), &compiled_vector_float_f64_state.vgprs[48][1],
           &compiled_vector_float_f64_state.vgprs[49][1]);
  SplitU64(DoubleBits(-1.0), &compiled_vector_float_f64_state.vgprs[48][3],
           &compiled_vector_float_f64_state.vgprs[49][3]);
  compiled_vector_float_f64_state.vgprs[50][2] = 0xdeadbeefu;
  compiled_vector_float_f64_state.vgprs[51][2] = 0xcafebabeu;
  compiled_vector_float_f64_state.vgprs[52][2] = 0xdeadbeefu;
  compiled_vector_float_f64_state.vgprs[53][2] = 0xcafebabeu;
  compiled_vector_float_f64_state.vgprs[54][2] = 0xdeadbeefu;
  compiled_vector_float_f64_state.vgprs[56][2] = 0xdeadbeefu;
  compiled_vector_float_f64_state.vgprs[57][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_float_f64_program,
                                         &compiled_vector_float_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_float_f64_state.halted,
              "expected compiled vector float f64 program to halt") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[50][0],
                         compiled_vector_float_f64_state.vgprs[51][0]) ==
                  DoubleBits(3.75),
              "expected compiled v_add_f64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[50][1],
                         compiled_vector_float_f64_state.vgprs[51][1]) ==
                  DoubleBits(1.0),
              "expected compiled v_add_f64 lane 1 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[50][2] == 0xdeadbeefu &&
                  compiled_vector_float_f64_state.vgprs[51][2] == 0xcafebabeu,
              "expected inactive compiled v_add_f64 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[50][3],
                         compiled_vector_float_f64_state.vgprs[51][3]) ==
                  DoubleBits(2.0),
              "expected compiled v_add_f64 lane 3 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[52][0],
                         compiled_vector_float_f64_state.vgprs[53][0]) ==
                  DoubleBits(-3.0),
              "expected compiled v_mul_f64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[52][1],
                         compiled_vector_float_f64_state.vgprs[53][1]) ==
                  DoubleBits(1.0),
              "expected compiled v_mul_f64 lane 1 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[52][2] == 0xdeadbeefu &&
                  compiled_vector_float_f64_state.vgprs[53][2] == 0xcafebabeu,
              "expected inactive compiled v_mul_f64 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[52][3],
                         compiled_vector_float_f64_state.vgprs[53][3]) ==
                  DoubleBits(-8.0),
              "expected compiled v_mul_f64 lane 3 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[54][0] == FloatBits(3.5f),
              "expected compiled v_fma_f32 lane 0 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[54][1] == FloatBits(-2.0f),
              "expected compiled v_fma_f32 lane 1 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[54][2] == 0xdeadbeefu,
              "expected inactive compiled v_fma_f32 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[54][3] == FloatBits(5.0f),
              "expected compiled v_fma_f32 lane 3 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[56][0],
                         compiled_vector_float_f64_state.vgprs[57][0]) ==
                  DoubleBits(3.5),
              "expected compiled v_fma_f64 lane 0 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[56][1],
                         compiled_vector_float_f64_state.vgprs[57][1]) ==
                  DoubleBits(-2.0),
              "expected compiled v_fma_f64 lane 1 result") ||
      !Expect(compiled_vector_float_f64_state.vgprs[56][2] == 0xdeadbeefu &&
                  compiled_vector_float_f64_state.vgprs[57][2] == 0xcafebabeu,
              "expected inactive compiled v_fma_f64 result") ||
      !Expect(ComposeU64(compiled_vector_float_f64_state.vgprs[56][3],
                         compiled_vector_float_f64_state.vgprs[57][3]) ==
                  DoubleBits(5.0),
              "expected compiled v_fma_f64 lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_carry_binary_program = {
      DecodedInstruction::FourOperand("V_ADD_CO_U32", InstructionOperand::Vgpr(103),
                                      InstructionOperand::Sgpr(106),
                                      InstructionOperand::Sgpr(51),
                                      InstructionOperand::Vgpr(104)),
      DecodedInstruction::FourOperand("V_SUB_CO_U32", InstructionOperand::Vgpr(105),
                                      InstructionOperand::Sgpr(108),
                                      InstructionOperand::Sgpr(52),
                                      InstructionOperand::Vgpr(106)),
      DecodedInstruction::FourOperand("V_SUBREV_CO_U32",
                                      InstructionOperand::Vgpr(107),
                                      InstructionOperand::Sgpr(110),
                                      InstructionOperand::Sgpr(53),
                                      InstructionOperand::Vgpr(108)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_carry_binary_state;
  vector_carry_binary_state.exec_mask = 0b1011ULL;
  vector_carry_binary_state.vcc_mask = 0b0100ULL;
  vector_carry_binary_state.sgprs[51] = 0xffffffffu;
  vector_carry_binary_state.sgprs[52] = 0u;
  vector_carry_binary_state.sgprs[53] = 1u;
  vector_carry_binary_state.vgprs[104][0] = 1u;
  vector_carry_binary_state.vgprs[104][1] = 5u;
  vector_carry_binary_state.vgprs[104][3] = 0x7fffffffu;
  vector_carry_binary_state.vgprs[106][0] = 1u;
  vector_carry_binary_state.vgprs[106][1] = 0u;
  vector_carry_binary_state.vgprs[106][3] = 0xffffffffu;
  vector_carry_binary_state.vgprs[108][0] = 0u;
  vector_carry_binary_state.vgprs[108][1] = 1u;
  vector_carry_binary_state.vgprs[108][3] = 5u;
  vector_carry_binary_state.vgprs[103][2] = 0xdeadbeefu;
  vector_carry_binary_state.vgprs[105][2] = 0xdeadbeefu;
  vector_carry_binary_state.vgprs[107][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_carry_binary_program,
                                         &vector_carry_binary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_carry_binary_state.halted,
              "expected vector carry binary program to halt") ||
      !Expect(vector_carry_binary_state.vgprs[103][0] == 0u,
              "expected v_add_co_u32 lane 0 result") ||
      !Expect(vector_carry_binary_state.vgprs[103][1] == 4u,
              "expected v_add_co_u32 lane 1 result") ||
      !Expect(vector_carry_binary_state.vgprs[103][2] == 0xdeadbeefu,
              "expected inactive v_add_co_u32 result") ||
      !Expect(vector_carry_binary_state.vgprs[103][3] == 0x7ffffffeu,
              "expected v_add_co_u32 lane 3 result") ||
      !Expect(vector_carry_binary_state.sgprs[106] == 0x0000000fu &&
                  vector_carry_binary_state.sgprs[107] == 0u,
              "expected v_add_co_u32 carry mask") ||
      !Expect(vector_carry_binary_state.vgprs[105][0] == 0xffffffffu,
              "expected v_sub_co_u32 lane 0 result") ||
      !Expect(vector_carry_binary_state.vgprs[105][1] == 0u,
              "expected v_sub_co_u32 lane 1 result") ||
      !Expect(vector_carry_binary_state.vgprs[105][2] == 0xdeadbeefu,
              "expected inactive v_sub_co_u32 result") ||
      !Expect(vector_carry_binary_state.vgprs[105][3] == 1u,
              "expected v_sub_co_u32 lane 3 result") ||
      !Expect(vector_carry_binary_state.sgprs[108] == 0x0000000du &&
                  vector_carry_binary_state.sgprs[109] == 0u,
              "expected v_sub_co_u32 carry mask") ||
      !Expect(vector_carry_binary_state.vgprs[107][0] == 0xffffffffu,
              "expected v_subrev_co_u32 lane 0 result") ||
      !Expect(vector_carry_binary_state.vgprs[107][1] == 0u,
              "expected v_subrev_co_u32 lane 1 result") ||
      !Expect(vector_carry_binary_state.vgprs[107][2] == 0xdeadbeefu,
              "expected inactive v_subrev_co_u32 result") ||
      !Expect(vector_carry_binary_state.vgprs[107][3] == 4u,
              "expected v_subrev_co_u32 lane 3 result") ||
      !Expect(vector_carry_binary_state.sgprs[110] == 0x00000005u &&
                  vector_carry_binary_state.sgprs[111] == 0u,
              "expected v_subrev_co_u32 carry mask") ||
      !Expect(vector_carry_binary_state.vcc_mask == 0x0000000000000005ULL,
              "expected final vcc mask after carry ops")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_carry_binary_program;
  if (!Expect(interpreter.CompileProgram(vector_carry_binary_program,
                                         &compiled_vector_carry_binary_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_carry_binary_state;
  compiled_vector_carry_binary_state.exec_mask = 0b1011ULL;
  compiled_vector_carry_binary_state.vcc_mask = 0b0100ULL;
  compiled_vector_carry_binary_state.sgprs[51] = 0xffffffffu;
  compiled_vector_carry_binary_state.sgprs[52] = 0u;
  compiled_vector_carry_binary_state.sgprs[53] = 1u;
  compiled_vector_carry_binary_state.vgprs[104][0] = 1u;
  compiled_vector_carry_binary_state.vgprs[104][1] = 5u;
  compiled_vector_carry_binary_state.vgprs[104][3] = 0x7fffffffu;
  compiled_vector_carry_binary_state.vgprs[106][0] = 1u;
  compiled_vector_carry_binary_state.vgprs[106][1] = 0u;
  compiled_vector_carry_binary_state.vgprs[106][3] = 0xffffffffu;
  compiled_vector_carry_binary_state.vgprs[108][0] = 0u;
  compiled_vector_carry_binary_state.vgprs[108][1] = 1u;
  compiled_vector_carry_binary_state.vgprs[108][3] = 5u;
  compiled_vector_carry_binary_state.vgprs[103][2] = 0xdeadbeefu;
  compiled_vector_carry_binary_state.vgprs[105][2] = 0xdeadbeefu;
  compiled_vector_carry_binary_state.vgprs[107][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_carry_binary_program,
                                         &compiled_vector_carry_binary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_carry_binary_state.halted,
              "expected compiled vector carry binary program to halt") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[103][0] == 0u,
              "expected compiled v_add_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[103][1] == 4u,
              "expected compiled v_add_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[103][2] == 0xdeadbeefu,
              "expected compiled inactive v_add_co_u32 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[103][3] == 0x7ffffffeu,
              "expected compiled v_add_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_binary_state.sgprs[106] == 0x0000000fu &&
                  compiled_vector_carry_binary_state.sgprs[107] == 0u,
              "expected compiled v_add_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[105][0] == 0xffffffffu,
              "expected compiled v_sub_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[105][1] == 0u,
              "expected compiled v_sub_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[105][2] == 0xdeadbeefu,
              "expected compiled inactive v_sub_co_u32 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[105][3] == 1u,
              "expected compiled v_sub_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_binary_state.sgprs[108] == 0x0000000du &&
                  compiled_vector_carry_binary_state.sgprs[109] == 0u,
              "expected compiled v_sub_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[107][0] == 0xffffffffu,
              "expected compiled v_subrev_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[107][1] == 0u,
              "expected compiled v_subrev_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[107][2] == 0xdeadbeefu,
              "expected compiled inactive v_subrev_co_u32 result") ||
      !Expect(compiled_vector_carry_binary_state.vgprs[107][3] == 4u,
              "expected compiled v_subrev_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_binary_state.sgprs[110] == 0x00000005u &&
                  compiled_vector_carry_binary_state.sgprs[111] == 0u,
              "expected compiled v_subrev_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_binary_state.vcc_mask == 0x0000000000000005ULL,
              "expected compiled final vcc mask after carry ops")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_carry_in_binary_program = {
      DecodedInstruction::FiveOperand(
          "V_ADDC_CO_U32", InstructionOperand::Vgpr(112),
          InstructionOperand::Sgpr(118), InstructionOperand::Sgpr(54),
          InstructionOperand::Vgpr(109), InstructionOperand::Sgpr(120)),
      DecodedInstruction::FiveOperand(
          "V_SUBB_CO_U32", InstructionOperand::Vgpr(113),
          InstructionOperand::Sgpr(122), InstructionOperand::Sgpr(55),
          InstructionOperand::Vgpr(110), InstructionOperand::Sgpr(124)),
      DecodedInstruction::FiveOperand(
          "V_SUBBREV_CO_U32", InstructionOperand::Vgpr(114),
          InstructionOperand::Sgpr(126), InstructionOperand::Sgpr(56),
          InstructionOperand::Vgpr(111), InstructionOperand::Sgpr(116)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_carry_in_binary_state;
  vector_carry_in_binary_state.exec_mask = 0b1011ULL;
  vector_carry_in_binary_state.vcc_mask = 0b0101ULL;
  vector_carry_in_binary_state.sgprs[54] = 0xfffffffeu;
  vector_carry_in_binary_state.sgprs[55] = 1u;
  vector_carry_in_binary_state.sgprs[56] = 1u;
  vector_carry_in_binary_state.sgprs[120] = 0x00000005u;
  vector_carry_in_binary_state.sgprs[121] = 0u;
  vector_carry_in_binary_state.sgprs[124] = 0x0000000eu;
  vector_carry_in_binary_state.sgprs[125] = 0u;
  vector_carry_in_binary_state.sgprs[116] = 0x00000006u;
  vector_carry_in_binary_state.sgprs[117] = 0u;
  vector_carry_in_binary_state.vgprs[109][0] = 1u;
  vector_carry_in_binary_state.vgprs[109][1] = 1u;
  vector_carry_in_binary_state.vgprs[109][3] = 2u;
  vector_carry_in_binary_state.vgprs[110][0] = 1u;
  vector_carry_in_binary_state.vgprs[110][1] = 0u;
  vector_carry_in_binary_state.vgprs[110][3] = 1u;
  vector_carry_in_binary_state.vgprs[111][0] = 1u;
  vector_carry_in_binary_state.vgprs[111][1] = 2u;
  vector_carry_in_binary_state.vgprs[111][3] = 0u;
  vector_carry_in_binary_state.vgprs[112][2] = 0xdeadbeefu;
  vector_carry_in_binary_state.vgprs[113][2] = 0xdeadbeefu;
  vector_carry_in_binary_state.vgprs[114][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_carry_in_binary_program,
                                         &vector_carry_in_binary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_carry_in_binary_state.halted,
              "expected vector carry-in binary program to halt") ||
      !Expect(vector_carry_in_binary_state.vgprs[112][0] == 0u,
              "expected v_addc_co_u32 lane 0 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[112][1] == 0xffffffffu,
              "expected v_addc_co_u32 lane 1 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[112][2] == 0xdeadbeefu,
              "expected inactive v_addc_co_u32 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[112][3] == 0u,
              "expected v_addc_co_u32 lane 3 result") ||
      !Expect(vector_carry_in_binary_state.sgprs[118] == 0x0000000du &&
                  vector_carry_in_binary_state.sgprs[119] == 0u,
              "expected v_addc_co_u32 carry mask") ||
      !Expect(vector_carry_in_binary_state.vgprs[113][0] == 0u,
              "expected v_subb_co_u32 lane 0 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[113][1] == 0u,
              "expected v_subb_co_u32 lane 1 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[113][2] == 0xdeadbeefu,
              "expected inactive v_subb_co_u32 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[113][3] == 0xffffffffu,
              "expected v_subb_co_u32 lane 3 result") ||
      !Expect(vector_carry_in_binary_state.sgprs[122] == 0x0000000cu &&
                  vector_carry_in_binary_state.sgprs[123] == 0u,
              "expected v_subb_co_u32 carry mask") ||
      !Expect(vector_carry_in_binary_state.vgprs[114][0] == 0u,
              "expected v_subbrev_co_u32 lane 0 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[114][1] == 0u,
              "expected v_subbrev_co_u32 lane 1 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[114][2] == 0xdeadbeefu,
              "expected inactive v_subbrev_co_u32 result") ||
      !Expect(vector_carry_in_binary_state.vgprs[114][3] == 0xffffffffu,
              "expected v_subbrev_co_u32 lane 3 result") ||
      !Expect(vector_carry_in_binary_state.sgprs[126] == 0x0000000cu &&
                  vector_carry_in_binary_state.sgprs[127] == 0u,
              "expected v_subbrev_co_u32 carry mask") ||
      !Expect(vector_carry_in_binary_state.vcc_mask == 0x000000000000000cULL,
              "expected final vcc mask after carry-in ops")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_carry_in_binary_program;
  if (!Expect(interpreter.CompileProgram(vector_carry_in_binary_program,
                                         &compiled_vector_carry_in_binary_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_carry_in_binary_state;
  compiled_vector_carry_in_binary_state.exec_mask = 0b1011ULL;
  compiled_vector_carry_in_binary_state.vcc_mask = 0b0101ULL;
  compiled_vector_carry_in_binary_state.sgprs[54] = 0xfffffffeu;
  compiled_vector_carry_in_binary_state.sgprs[55] = 1u;
  compiled_vector_carry_in_binary_state.sgprs[56] = 1u;
  compiled_vector_carry_in_binary_state.sgprs[120] = 0x00000005u;
  compiled_vector_carry_in_binary_state.sgprs[121] = 0u;
  compiled_vector_carry_in_binary_state.sgprs[124] = 0x0000000eu;
  compiled_vector_carry_in_binary_state.sgprs[125] = 0u;
  compiled_vector_carry_in_binary_state.sgprs[116] = 0x00000006u;
  compiled_vector_carry_in_binary_state.sgprs[117] = 0u;
  compiled_vector_carry_in_binary_state.vgprs[109][0] = 1u;
  compiled_vector_carry_in_binary_state.vgprs[109][1] = 1u;
  compiled_vector_carry_in_binary_state.vgprs[109][3] = 2u;
  compiled_vector_carry_in_binary_state.vgprs[110][0] = 1u;
  compiled_vector_carry_in_binary_state.vgprs[110][1] = 0u;
  compiled_vector_carry_in_binary_state.vgprs[110][3] = 1u;
  compiled_vector_carry_in_binary_state.vgprs[111][0] = 1u;
  compiled_vector_carry_in_binary_state.vgprs[111][1] = 2u;
  compiled_vector_carry_in_binary_state.vgprs[111][3] = 0u;
  compiled_vector_carry_in_binary_state.vgprs[112][2] = 0xdeadbeefu;
  compiled_vector_carry_in_binary_state.vgprs[113][2] = 0xdeadbeefu;
  compiled_vector_carry_in_binary_state.vgprs[114][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_carry_in_binary_program,
                                         &compiled_vector_carry_in_binary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_carry_in_binary_state.halted,
              "expected compiled vector carry-in binary program to halt") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[112][0] == 0u,
              "expected compiled v_addc_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[112][1] ==
                  0xffffffffu,
              "expected compiled v_addc_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[112][2] ==
                  0xdeadbeefu,
              "expected compiled inactive v_addc_co_u32 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[112][3] == 0u,
              "expected compiled v_addc_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_in_binary_state.sgprs[118] == 0x0000000du &&
                  compiled_vector_carry_in_binary_state.sgprs[119] == 0u,
              "expected compiled v_addc_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[113][0] == 0u,
              "expected compiled v_subb_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[113][1] == 0u,
              "expected compiled v_subb_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[113][2] ==
                  0xdeadbeefu,
              "expected compiled inactive v_subb_co_u32 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[113][3] ==
                  0xffffffffu,
              "expected compiled v_subb_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_in_binary_state.sgprs[122] == 0x0000000cu &&
                  compiled_vector_carry_in_binary_state.sgprs[123] == 0u,
              "expected compiled v_subb_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[114][0] == 0u,
              "expected compiled v_subbrev_co_u32 lane 0 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[114][1] == 0u,
              "expected compiled v_subbrev_co_u32 lane 1 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[114][2] ==
                  0xdeadbeefu,
              "expected compiled inactive v_subbrev_co_u32 result") ||
      !Expect(compiled_vector_carry_in_binary_state.vgprs[114][3] ==
                  0xffffffffu,
              "expected compiled v_subbrev_co_u32 lane 3 result") ||
      !Expect(compiled_vector_carry_in_binary_state.sgprs[126] == 0x0000000cu &&
                  compiled_vector_carry_in_binary_state.sgprs[127] == 0u,
              "expected compiled v_subbrev_co_u32 carry mask") ||
      !Expect(compiled_vector_carry_in_binary_state.vcc_mask ==
                  0x000000000000000cULL,
              "expected compiled final vcc mask after carry-in ops")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_compare_program = {
      DecodedInstruction::Binary("V_CMP_EQ_U32", InstructionOperand::Sgpr(106),
                                 InstructionOperand::Sgpr(12),
                                 InstructionOperand::Vgpr(12)),
      DecodedInstruction::Binary("V_CMP_LT_I32", InstructionOperand::Sgpr(108),
                                 InstructionOperand::Sgpr(13),
                                 InstructionOperand::Vgpr(13)),
      DecodedInstruction::Binary("V_CMP_GE_U32", InstructionOperand::Sgpr(110),
                                 InstructionOperand::Sgpr(14),
                                 InstructionOperand::Vgpr(14)),
      DecodedInstruction::Binary("V_CNDMASK_B32", InstructionOperand::Vgpr(35),
                                 InstructionOperand::Sgpr(15),
                                 InstructionOperand::Vgpr(15)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_compare_state;
  vector_compare_state.exec_mask = 0b1011ULL;
  vector_compare_state.vcc_mask = 0b0100ULL;
  vector_compare_state.sgprs[12] = 7u;
  vector_compare_state.sgprs[13] = static_cast<std::uint32_t>(-2);
  vector_compare_state.sgprs[14] = 10u;
  vector_compare_state.sgprs[15] = 99u;
  vector_compare_state.vgprs[12][0] = 7u;
  vector_compare_state.vgprs[12][1] = 5u;
  vector_compare_state.vgprs[12][3] = 7u;
  vector_compare_state.vgprs[13][0] = static_cast<std::uint32_t>(-1);
  vector_compare_state.vgprs[13][1] = static_cast<std::uint32_t>(-3);
  vector_compare_state.vgprs[13][3] = static_cast<std::uint32_t>(-2);
  vector_compare_state.vgprs[14][0] = 10u;
  vector_compare_state.vgprs[14][1] = 11u;
  vector_compare_state.vgprs[14][3] = 2u;
  vector_compare_state.vgprs[15][0] = 100u;
  vector_compare_state.vgprs[15][1] = 200u;
  vector_compare_state.vgprs[15][3] = 300u;
  vector_compare_state.vgprs[35][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_compare_program,
                                         &vector_compare_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_state.halted,
              "expected vector compare program to halt") ||
      !Expect(vector_compare_state.sgprs[106] == 13u,
              "expected v_cmp_eq_u32 low mask result") ||
      !Expect(vector_compare_state.sgprs[107] == 0u,
              "expected v_cmp_eq_u32 high mask result") ||
      !Expect(vector_compare_state.sgprs[108] == 5u,
              "expected v_cmp_lt_i32 low mask result") ||
      !Expect(vector_compare_state.sgprs[109] == 0u,
              "expected v_cmp_lt_i32 high mask result") ||
      !Expect(vector_compare_state.sgprs[110] == 13u,
              "expected v_cmp_ge_u32 low mask result") ||
      !Expect(vector_compare_state.sgprs[111] == 0u,
              "expected v_cmp_ge_u32 high mask result") ||
      !Expect(vector_compare_state.vgprs[35][0] == 100u,
              "expected v_cndmask_b32 lane 0 result") ||
      !Expect(vector_compare_state.vgprs[35][1] == 99u,
              "expected v_cndmask_b32 lane 1 result") ||
      !Expect(vector_compare_state.vgprs[35][2] == 0xdeadbeefu,
              "expected inactive lane v_cndmask_b32 result") ||
      !Expect(vector_compare_state.vgprs[35][3] == 300u,
              "expected v_cndmask_b32 lane 3 result") ||
      !Expect(vector_compare_state.vcc_mask == 13u,
              "expected final VCC mask result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_compare_program;
  if (!Expect(interpreter.CompileProgram(vector_compare_program,
                                         &compiled_vector_compare_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_compare_state;
  compiled_vector_compare_state.exec_mask = 0b1011ULL;
  compiled_vector_compare_state.vcc_mask = 0b0100ULL;
  compiled_vector_compare_state.sgprs[12] = 7u;
  compiled_vector_compare_state.sgprs[13] = static_cast<std::uint32_t>(-2);
  compiled_vector_compare_state.sgprs[14] = 10u;
  compiled_vector_compare_state.sgprs[15] = 99u;
  compiled_vector_compare_state.vgprs[12][0] = 7u;
  compiled_vector_compare_state.vgprs[12][1] = 5u;
  compiled_vector_compare_state.vgprs[12][3] = 7u;
  compiled_vector_compare_state.vgprs[13][0] = static_cast<std::uint32_t>(-1);
  compiled_vector_compare_state.vgprs[13][1] = static_cast<std::uint32_t>(-3);
  compiled_vector_compare_state.vgprs[13][3] = static_cast<std::uint32_t>(-2);
  compiled_vector_compare_state.vgprs[14][0] = 10u;
  compiled_vector_compare_state.vgprs[14][1] = 11u;
  compiled_vector_compare_state.vgprs[14][3] = 2u;
  compiled_vector_compare_state.vgprs[15][0] = 100u;
  compiled_vector_compare_state.vgprs[15][1] = 200u;
  compiled_vector_compare_state.vgprs[15][3] = 300u;
  compiled_vector_compare_state.vgprs[35][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_compare_program,
                                         &compiled_vector_compare_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_compare_state.halted,
              "expected compiled vector compare program to halt") ||
      !Expect(compiled_vector_compare_state.sgprs[106] == 13u,
              "expected compiled v_cmp_eq_u32 low mask result") ||
      !Expect(compiled_vector_compare_state.sgprs[107] == 0u,
              "expected compiled v_cmp_eq_u32 high mask result") ||
      !Expect(compiled_vector_compare_state.sgprs[108] == 5u,
              "expected compiled v_cmp_lt_i32 low mask result") ||
      !Expect(compiled_vector_compare_state.sgprs[109] == 0u,
              "expected compiled v_cmp_lt_i32 high mask result") ||
      !Expect(compiled_vector_compare_state.sgprs[110] == 13u,
              "expected compiled v_cmp_ge_u32 low mask result") ||
      !Expect(compiled_vector_compare_state.sgprs[111] == 0u,
              "expected compiled v_cmp_ge_u32 high mask result") ||
      !Expect(compiled_vector_compare_state.vgprs[35][0] == 100u,
              "expected compiled v_cndmask_b32 lane 0 result") ||
      !Expect(compiled_vector_compare_state.vgprs[35][1] == 99u,
              "expected compiled v_cndmask_b32 lane 1 result") ||
      !Expect(compiled_vector_compare_state.vgprs[35][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_cndmask_b32 result") ||
      !Expect(compiled_vector_compare_state.vgprs[35][3] == 300u,
              "expected compiled v_cndmask_b32 lane 3 result") ||
      !Expect(compiled_vector_compare_state.vcc_mask == 13u,
              "expected compiled final VCC mask result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_compare_f32_program = {
      DecodedInstruction::Binary("V_CMP_LT_F32", InstructionOperand::Sgpr(112),
                                 InstructionOperand::Sgpr(16),
                                 InstructionOperand::Vgpr(16)),
      DecodedInstruction::Binary("V_CMP_U_F32", InstructionOperand::Sgpr(114),
                                 InstructionOperand::Sgpr(17),
                                 InstructionOperand::Vgpr(17)),
      DecodedInstruction::Binary("V_CMP_NGT_F32", InstructionOperand::Sgpr(116),
                                 InstructionOperand::Sgpr(18),
                                 InstructionOperand::Vgpr(18)),
      DecodedInstruction::Binary("V_CNDMASK_B32", InstructionOperand::Vgpr(37),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_compare_f32_state;
  vector_compare_f32_state.exec_mask = 0b1011ULL;
  vector_compare_f32_state.sgprs[16] = FloatBits(1.5f);
  vector_compare_f32_state.sgprs[17] = 0x7fc00000u;
  vector_compare_f32_state.sgprs[18] = FloatBits(2.0f);
  vector_compare_f32_state.sgprs[20] = 55u;
  vector_compare_f32_state.vgprs[16][0] = FloatBits(2.0f);
  vector_compare_f32_state.vgprs[16][1] = FloatBits(1.0f);
  vector_compare_f32_state.vgprs[16][3] = FloatBits(1.5f);
  vector_compare_f32_state.vgprs[17][0] = FloatBits(0.0f);
  vector_compare_f32_state.vgprs[17][1] = FloatBits(4.0f);
  vector_compare_f32_state.vgprs[17][3] = FloatBits(-1.0f);
  vector_compare_f32_state.vgprs[18][0] = FloatBits(1.0f);
  vector_compare_f32_state.vgprs[18][1] = FloatBits(2.0f);
  vector_compare_f32_state.vgprs[18][3] = 0x7fc00000u;
  vector_compare_f32_state.vgprs[20][0] = 100u;
  vector_compare_f32_state.vgprs[20][1] = 200u;
  vector_compare_f32_state.vgprs[20][3] = 300u;
  vector_compare_f32_state.vgprs[37][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_compare_f32_program,
                                         &vector_compare_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_f32_state.halted,
              "expected vector compare f32 program to halt") ||
      !Expect(vector_compare_f32_state.sgprs[112] == 1u,
              "expected v_cmp_lt_f32 low mask result") ||
      !Expect(vector_compare_f32_state.sgprs[114] == 11u,
              "expected v_cmp_u_f32 low mask result") ||
      !Expect(vector_compare_f32_state.sgprs[116] == 10u,
              "expected v_cmp_ngt_f32 low mask result") ||
      !Expect(vector_compare_f32_state.vgprs[37][0] == 55u,
              "expected v_cndmask_b32 lane 0 f32 result") ||
      !Expect(vector_compare_f32_state.vgprs[37][1] == 200u,
              "expected v_cndmask_b32 lane 1 f32 result") ||
      !Expect(vector_compare_f32_state.vgprs[37][2] == 0xdeadbeefu,
              "expected inactive lane v_cndmask_b32 f32 result") ||
      !Expect(vector_compare_f32_state.vgprs[37][3] == 300u,
              "expected v_cndmask_b32 lane 3 f32 result") ||
      !Expect(vector_compare_f32_state.vcc_mask == 10u,
              "expected final F32 VCC mask result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_compare_f32_program;
  if (!Expect(interpreter.CompileProgram(vector_compare_f32_program,
                                         &compiled_vector_compare_f32_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_compare_f32_state;
  compiled_vector_compare_f32_state.exec_mask = 0b1011ULL;
  compiled_vector_compare_f32_state.sgprs[16] = FloatBits(1.5f);
  compiled_vector_compare_f32_state.sgprs[17] = 0x7fc00000u;
  compiled_vector_compare_f32_state.sgprs[18] = FloatBits(2.0f);
  compiled_vector_compare_f32_state.sgprs[20] = 55u;
  compiled_vector_compare_f32_state.vgprs[16][0] = FloatBits(2.0f);
  compiled_vector_compare_f32_state.vgprs[16][1] = FloatBits(1.0f);
  compiled_vector_compare_f32_state.vgprs[16][3] = FloatBits(1.5f);
  compiled_vector_compare_f32_state.vgprs[17][0] = FloatBits(0.0f);
  compiled_vector_compare_f32_state.vgprs[17][1] = FloatBits(4.0f);
  compiled_vector_compare_f32_state.vgprs[17][3] = FloatBits(-1.0f);
  compiled_vector_compare_f32_state.vgprs[18][0] = FloatBits(1.0f);
  compiled_vector_compare_f32_state.vgprs[18][1] = FloatBits(2.0f);
  compiled_vector_compare_f32_state.vgprs[18][3] = 0x7fc00000u;
  compiled_vector_compare_f32_state.vgprs[20][0] = 100u;
  compiled_vector_compare_f32_state.vgprs[20][1] = 200u;
  compiled_vector_compare_f32_state.vgprs[20][3] = 300u;
  compiled_vector_compare_f32_state.vgprs[37][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_compare_f32_program,
                                         &compiled_vector_compare_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_compare_f32_state.halted,
              "expected compiled vector compare f32 program to halt") ||
      !Expect(compiled_vector_compare_f32_state.sgprs[112] == 1u,
              "expected compiled v_cmp_lt_f32 low mask result") ||
      !Expect(compiled_vector_compare_f32_state.sgprs[114] == 11u,
              "expected compiled v_cmp_u_f32 low mask result") ||
      !Expect(compiled_vector_compare_f32_state.sgprs[116] == 10u,
              "expected compiled v_cmp_ngt_f32 low mask result") ||
      !Expect(compiled_vector_compare_f32_state.vgprs[37][0] == 55u,
              "expected compiled v_cndmask_b32 lane 0 f32 result") ||
      !Expect(compiled_vector_compare_f32_state.vgprs[37][1] == 200u,
              "expected compiled v_cndmask_b32 lane 1 f32 result") ||
      !Expect(compiled_vector_compare_f32_state.vgprs[37][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_cndmask_b32 f32 result") ||
      !Expect(compiled_vector_compare_f32_state.vgprs[37][3] == 300u,
              "expected compiled v_cndmask_b32 lane 3 f32 result") ||
      !Expect(compiled_vector_compare_f32_state.vcc_mask == 10u,
              "expected compiled final F32 VCC mask result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_compare_f64_program = {
      DecodedInstruction::Binary("V_CMP_LT_F64", InstructionOperand::Sgpr(120),
                                 InstructionOperand::Sgpr(70),
                                 InstructionOperand::Vgpr(30)),
      DecodedInstruction::Binary("V_CMP_U_F64", InstructionOperand::Sgpr(122),
                                 InstructionOperand::Sgpr(72),
                                 InstructionOperand::Vgpr(32)),
      DecodedInstruction::Binary("V_CMP_NGT_F64", InstructionOperand::Sgpr(124),
                                 InstructionOperand::Sgpr(74),
                                 InstructionOperand::Vgpr(34)),
      DecodedInstruction::Binary("V_CNDMASK_B32", InstructionOperand::Vgpr(38),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  constexpr std::uint64_t kQuietNan64 = 0x7ff8000000000000ULL;
  WaveExecutionState vector_compare_f64_state;
  vector_compare_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &vector_compare_f64_state.sgprs[70],
           &vector_compare_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_compare_f64_state.sgprs[72],
           &vector_compare_f64_state.sgprs[73]);
  SplitU64(DoubleBits(2.0), &vector_compare_f64_state.sgprs[74],
           &vector_compare_f64_state.sgprs[75]);
  vector_compare_f64_state.sgprs[20] = 55u;
  SplitU64(DoubleBits(2.0), &vector_compare_f64_state.vgprs[30][0],
           &vector_compare_f64_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &vector_compare_f64_state.vgprs[30][1],
           &vector_compare_f64_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &vector_compare_f64_state.vgprs[30][3],
           &vector_compare_f64_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &vector_compare_f64_state.vgprs[32][0],
           &vector_compare_f64_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &vector_compare_f64_state.vgprs[32][1],
           &vector_compare_f64_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &vector_compare_f64_state.vgprs[32][3],
           &vector_compare_f64_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &vector_compare_f64_state.vgprs[34][0],
           &vector_compare_f64_state.vgprs[35][0]);
  SplitU64(DoubleBits(2.0), &vector_compare_f64_state.vgprs[34][1],
           &vector_compare_f64_state.vgprs[35][1]);
  SplitU64(kQuietNan64, &vector_compare_f64_state.vgprs[34][3],
           &vector_compare_f64_state.vgprs[35][3]);
  vector_compare_f64_state.vgprs[20][0] = 100u;
  vector_compare_f64_state.vgprs[20][1] = 200u;
  vector_compare_f64_state.vgprs[20][3] = 300u;
  vector_compare_f64_state.vgprs[38][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_compare_f64_program,
                                         &vector_compare_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_f64_state.halted,
              "expected vector compare f64 program to halt") ||
      !Expect(vector_compare_f64_state.sgprs[120] == 1u &&
                  vector_compare_f64_state.sgprs[121] == 0u,
              "expected v_cmp_lt_f64 mask result") ||
      !Expect(vector_compare_f64_state.sgprs[122] == 11u &&
                  vector_compare_f64_state.sgprs[123] == 0u,
              "expected v_cmp_u_f64 mask result") ||
      !Expect(vector_compare_f64_state.sgprs[124] == 10u &&
                  vector_compare_f64_state.sgprs[125] == 0u,
              "expected v_cmp_ngt_f64 mask result") ||
      !Expect(vector_compare_f64_state.vgprs[38][0] == 55u,
              "expected v_cndmask_b32 lane 0 f64 result") ||
      !Expect(vector_compare_f64_state.vgprs[38][1] == 200u,
              "expected v_cndmask_b32 lane 1 f64 result") ||
      !Expect(vector_compare_f64_state.vgprs[38][2] == 0xdeadbeefu,
              "expected inactive lane v_cndmask_b32 f64 result") ||
      !Expect(vector_compare_f64_state.vgprs[38][3] == 300u,
              "expected v_cndmask_b32 lane 3 f64 result") ||
      !Expect(vector_compare_f64_state.vcc_mask == 10u,
              "expected final F64 VCC mask result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_compare_f64_program;
  if (!Expect(interpreter.CompileProgram(vector_compare_f64_program,
                                         &compiled_vector_compare_f64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_compare_f64_state;
  compiled_vector_compare_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &compiled_vector_compare_f64_state.sgprs[70],
           &compiled_vector_compare_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &compiled_vector_compare_f64_state.sgprs[72],
           &compiled_vector_compare_f64_state.sgprs[73]);
  SplitU64(DoubleBits(2.0), &compiled_vector_compare_f64_state.sgprs[74],
           &compiled_vector_compare_f64_state.sgprs[75]);
  compiled_vector_compare_f64_state.sgprs[20] = 55u;
  SplitU64(DoubleBits(2.0), &compiled_vector_compare_f64_state.vgprs[30][0],
           &compiled_vector_compare_f64_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &compiled_vector_compare_f64_state.vgprs[30][1],
           &compiled_vector_compare_f64_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &compiled_vector_compare_f64_state.vgprs[30][3],
           &compiled_vector_compare_f64_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &compiled_vector_compare_f64_state.vgprs[32][0],
           &compiled_vector_compare_f64_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &compiled_vector_compare_f64_state.vgprs[32][1],
           &compiled_vector_compare_f64_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &compiled_vector_compare_f64_state.vgprs[32][3],
           &compiled_vector_compare_f64_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &compiled_vector_compare_f64_state.vgprs[34][0],
           &compiled_vector_compare_f64_state.vgprs[35][0]);
  SplitU64(DoubleBits(2.0), &compiled_vector_compare_f64_state.vgprs[34][1],
           &compiled_vector_compare_f64_state.vgprs[35][1]);
  SplitU64(kQuietNan64, &compiled_vector_compare_f64_state.vgprs[34][3],
           &compiled_vector_compare_f64_state.vgprs[35][3]);
  compiled_vector_compare_f64_state.vgprs[20][0] = 100u;
  compiled_vector_compare_f64_state.vgprs[20][1] = 200u;
  compiled_vector_compare_f64_state.vgprs[20][3] = 300u;
  compiled_vector_compare_f64_state.vgprs[38][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_compare_f64_program,
                                         &compiled_vector_compare_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_compare_f64_state.halted,
              "expected compiled vector compare f64 program to halt") ||
      !Expect(compiled_vector_compare_f64_state.sgprs[120] == 1u &&
                  compiled_vector_compare_f64_state.sgprs[121] == 0u,
              "expected compiled v_cmp_lt_f64 mask result") ||
      !Expect(compiled_vector_compare_f64_state.sgprs[122] == 11u &&
                  compiled_vector_compare_f64_state.sgprs[123] == 0u,
              "expected compiled v_cmp_u_f64 mask result") ||
      !Expect(compiled_vector_compare_f64_state.sgprs[124] == 10u &&
                  compiled_vector_compare_f64_state.sgprs[125] == 0u,
              "expected compiled v_cmp_ngt_f64 mask result") ||
      !Expect(compiled_vector_compare_f64_state.vgprs[38][0] == 55u,
              "expected compiled v_cndmask_b32 lane 0 f64 result") ||
      !Expect(compiled_vector_compare_f64_state.vgprs[38][1] == 200u,
              "expected compiled v_cndmask_b32 lane 1 f64 result") ||
      !Expect(compiled_vector_compare_f64_state.vgprs[38][2] == 0xdeadbeefu,
              "expected compiled inactive lane v_cndmask_b32 f64 result") ||
      !Expect(compiled_vector_compare_f64_state.vgprs[38][3] == 300u,
              "expected compiled v_cndmask_b32 lane 3 f64 result") ||
      !Expect(compiled_vector_compare_f64_state.vcc_mask == 10u,
              "expected compiled final F64 VCC mask result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_class_f32_program = {
      DecodedInstruction::Binary("V_CMP_CLASS_F32", InstructionOperand::Sgpr(118),
                                 InstructionOperand::Sgpr(16),
                                 InstructionOperand::Vgpr(40)),
      DecodedInstruction::Binary("V_CMP_CLASS_F32", InstructionOperand::Sgpr(120),
                                 InstructionOperand::Sgpr(17),
                                 InstructionOperand::Vgpr(41)),
      DecodedInstruction::Binary("V_CNDMASK_B32", InstructionOperand::Vgpr(42),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_class_f32_state;
  vector_class_f32_state.exec_mask = 0b1011ULL;
  vector_class_f32_state.sgprs[16] = FloatBits(-0.0f);
  vector_class_f32_state.sgprs[17] = 0x7fc00000u;
  vector_class_f32_state.sgprs[20] = 55u;
  vector_class_f32_state.vgprs[40][0] = 0x20u;
  vector_class_f32_state.vgprs[40][1] = 0x40u;
  vector_class_f32_state.vgprs[40][3] = 0x60u;
  vector_class_f32_state.vgprs[41][0] = 0x1u;
  vector_class_f32_state.vgprs[41][1] = 0x2u;
  vector_class_f32_state.vgprs[41][3] = 0x3u;
  vector_class_f32_state.vgprs[20][0] = 100u;
  vector_class_f32_state.vgprs[20][1] = 200u;
  vector_class_f32_state.vgprs[20][3] = 300u;
  vector_class_f32_state.vgprs[42][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_class_f32_program,
                                         &vector_class_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_class_f32_state.halted,
              "expected vector class f32 program to halt") ||
      !Expect(vector_class_f32_state.sgprs[118] == 9u &&
                  vector_class_f32_state.sgprs[119] == 0u,
              "expected v_cmp_class_f32 first mask result") ||
      !Expect(vector_class_f32_state.sgprs[120] == 10u &&
                  vector_class_f32_state.sgprs[121] == 0u,
              "expected v_cmp_class_f32 second mask result") ||
      !Expect(vector_class_f32_state.vgprs[42][0] == 55u,
              "expected v_cmp_class_f32 lane 0 cndmask result") ||
      !Expect(vector_class_f32_state.vgprs[42][1] == 200u,
              "expected v_cmp_class_f32 lane 1 cndmask result") ||
      !Expect(vector_class_f32_state.vgprs[42][2] == 0xdeadbeefu,
              "expected v_cmp_class_f32 inactive lane result") ||
      !Expect(vector_class_f32_state.vgprs[42][3] == 300u,
              "expected v_cmp_class_f32 lane 3 cndmask result") ||
      !Expect(vector_class_f32_state.vcc_mask == 10u,
              "expected final VCC mask after vector class f32 ops")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_class_f32_program;
  if (!Expect(interpreter.CompileProgram(vector_class_f32_program,
                                         &compiled_vector_class_f32_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_class_f32_state;
  compiled_vector_class_f32_state.exec_mask = 0b1011ULL;
  compiled_vector_class_f32_state.sgprs[16] = FloatBits(-0.0f);
  compiled_vector_class_f32_state.sgprs[17] = 0x7fc00000u;
  compiled_vector_class_f32_state.sgprs[20] = 55u;
  compiled_vector_class_f32_state.vgprs[40][0] = 0x20u;
  compiled_vector_class_f32_state.vgprs[40][1] = 0x40u;
  compiled_vector_class_f32_state.vgprs[40][3] = 0x60u;
  compiled_vector_class_f32_state.vgprs[41][0] = 0x1u;
  compiled_vector_class_f32_state.vgprs[41][1] = 0x2u;
  compiled_vector_class_f32_state.vgprs[41][3] = 0x3u;
  compiled_vector_class_f32_state.vgprs[20][0] = 100u;
  compiled_vector_class_f32_state.vgprs[20][1] = 200u;
  compiled_vector_class_f32_state.vgprs[20][3] = 300u;
  compiled_vector_class_f32_state.vgprs[42][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_class_f32_program,
                                         &compiled_vector_class_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_class_f32_state.halted,
              "expected compiled vector class f32 program to halt") ||
      !Expect(compiled_vector_class_f32_state.sgprs[118] == 9u &&
                  compiled_vector_class_f32_state.sgprs[119] == 0u,
              "expected compiled v_cmp_class_f32 first mask result") ||
      !Expect(compiled_vector_class_f32_state.sgprs[120] == 10u &&
                  compiled_vector_class_f32_state.sgprs[121] == 0u,
              "expected compiled v_cmp_class_f32 second mask result") ||
      !Expect(compiled_vector_class_f32_state.vgprs[42][0] == 55u,
              "expected compiled v_cmp_class_f32 lane 0 cndmask result") ||
      !Expect(compiled_vector_class_f32_state.vgprs[42][1] == 200u,
              "expected compiled v_cmp_class_f32 lane 1 cndmask result") ||
      !Expect(compiled_vector_class_f32_state.vgprs[42][2] == 0xdeadbeefu,
              "expected compiled v_cmp_class_f32 inactive lane result") ||
      !Expect(compiled_vector_class_f32_state.vgprs[42][3] == 300u,
              "expected compiled v_cmp_class_f32 lane 3 cndmask result") ||
      !Expect(compiled_vector_class_f32_state.vcc_mask == 10u,
              "expected compiled final VCC mask after vector class f32 ops")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_class_f64_program = {
      DecodedInstruction::Binary("V_CMP_CLASS_F64", InstructionOperand::Sgpr(122),
                                 InstructionOperand::Sgpr(70),
                                 InstructionOperand::Vgpr(40)),
      DecodedInstruction::Binary("V_CMP_CLASS_F64", InstructionOperand::Sgpr(124),
                                 InstructionOperand::Sgpr(72),
                                 InstructionOperand::Vgpr(41)),
      DecodedInstruction::Binary("V_CNDMASK_B32", InstructionOperand::Vgpr(43),
                                 InstructionOperand::Sgpr(20),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_class_f64_state;
  vector_class_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(-0.0), &vector_class_f64_state.sgprs[70],
           &vector_class_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_class_f64_state.sgprs[72],
           &vector_class_f64_state.sgprs[73]);
  vector_class_f64_state.sgprs[20] = 55u;
  vector_class_f64_state.vgprs[40][0] = 0x20u;
  vector_class_f64_state.vgprs[40][1] = 0x40u;
  vector_class_f64_state.vgprs[40][3] = 0x60u;
  vector_class_f64_state.vgprs[41][0] = 0x1u;
  vector_class_f64_state.vgprs[41][1] = 0x2u;
  vector_class_f64_state.vgprs[41][3] = 0x3u;
  vector_class_f64_state.vgprs[20][0] = 100u;
  vector_class_f64_state.vgprs[20][1] = 200u;
  vector_class_f64_state.vgprs[20][3] = 300u;
  vector_class_f64_state.vgprs[43][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(vector_class_f64_program,
                                         &vector_class_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_class_f64_state.halted,
              "expected vector class f64 program to halt") ||
      !Expect(vector_class_f64_state.sgprs[122] == 9u &&
                  vector_class_f64_state.sgprs[123] == 0u,
              "expected v_cmp_class_f64 first mask result") ||
      !Expect(vector_class_f64_state.sgprs[124] == 10u &&
                  vector_class_f64_state.sgprs[125] == 0u,
              "expected v_cmp_class_f64 second mask result") ||
      !Expect(vector_class_f64_state.vgprs[43][0] == 55u,
              "expected v_cmp_class_f64 lane 0 cndmask result") ||
      !Expect(vector_class_f64_state.vgprs[43][1] == 200u,
              "expected v_cmp_class_f64 lane 1 cndmask result") ||
      !Expect(vector_class_f64_state.vgprs[43][2] == 0xdeadbeefu,
              "expected v_cmp_class_f64 inactive lane result") ||
      !Expect(vector_class_f64_state.vgprs[43][3] == 300u,
              "expected v_cmp_class_f64 lane 3 cndmask result") ||
      !Expect(vector_class_f64_state.vcc_mask == 10u,
              "expected final VCC mask after vector class f64 ops")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_class_f64_program;
  if (!Expect(interpreter.CompileProgram(vector_class_f64_program,
                                         &compiled_vector_class_f64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_class_f64_state;
  compiled_vector_class_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(-0.0), &compiled_vector_class_f64_state.sgprs[70],
           &compiled_vector_class_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &compiled_vector_class_f64_state.sgprs[72],
           &compiled_vector_class_f64_state.sgprs[73]);
  compiled_vector_class_f64_state.sgprs[20] = 55u;
  compiled_vector_class_f64_state.vgprs[40][0] = 0x20u;
  compiled_vector_class_f64_state.vgprs[40][1] = 0x40u;
  compiled_vector_class_f64_state.vgprs[40][3] = 0x60u;
  compiled_vector_class_f64_state.vgprs[41][0] = 0x1u;
  compiled_vector_class_f64_state.vgprs[41][1] = 0x2u;
  compiled_vector_class_f64_state.vgprs[41][3] = 0x3u;
  compiled_vector_class_f64_state.vgprs[20][0] = 100u;
  compiled_vector_class_f64_state.vgprs[20][1] = 200u;
  compiled_vector_class_f64_state.vgprs[20][3] = 300u;
  compiled_vector_class_f64_state.vgprs[43][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_class_f64_program,
                                         &compiled_vector_class_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_class_f64_state.halted,
              "expected compiled vector class f64 program to halt") ||
      !Expect(compiled_vector_class_f64_state.sgprs[122] == 9u &&
                  compiled_vector_class_f64_state.sgprs[123] == 0u,
              "expected compiled v_cmp_class_f64 first mask result") ||
      !Expect(compiled_vector_class_f64_state.sgprs[124] == 10u &&
                  compiled_vector_class_f64_state.sgprs[125] == 0u,
              "expected compiled v_cmp_class_f64 second mask result") ||
      !Expect(compiled_vector_class_f64_state.vgprs[43][0] == 55u,
              "expected compiled v_cmp_class_f64 lane 0 cndmask result") ||
      !Expect(compiled_vector_class_f64_state.vgprs[43][1] == 200u,
              "expected compiled v_cmp_class_f64 lane 1 cndmask result") ||
      !Expect(compiled_vector_class_f64_state.vgprs[43][2] == 0xdeadbeefu,
              "expected compiled v_cmp_class_f64 inactive lane result") ||
      !Expect(compiled_vector_class_f64_state.vgprs[43][3] == 300u,
              "expected compiled v_cmp_class_f64 lane 3 cndmask result") ||
      !Expect(compiled_vector_class_f64_state.vcc_mask == 10u,
              "expected compiled final VCC mask after vector class f64 ops")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_compare64_program = {
      DecodedInstruction::Binary("V_CMP_LT_I64", InstructionOperand::Sgpr(112),
                                 InstructionOperand::Sgpr(60),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::Binary("V_CMP_GE_U64", InstructionOperand::Sgpr(106),
                                 InstructionOperand::Sgpr(62),
                                 InstructionOperand::Vgpr(22)),
      DecodedInstruction::Binary("V_CMP_F_I64", InstructionOperand::Sgpr(114),
                                 InstructionOperand::Sgpr(64),
                                 InstructionOperand::Vgpr(24)),
      DecodedInstruction::Binary("V_CMP_T_U64", InstructionOperand::Sgpr(116),
                                 InstructionOperand::Sgpr(66),
                                 InstructionOperand::Vgpr(26)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_compare64_state;
  vector_compare64_state.exec_mask = 0b1011ULL;
  vector_compare64_state.vcc_mask = 0b0100ULL;
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &vector_compare64_state.sgprs[60], &vector_compare64_state.sgprs[61]);
  SplitU64(10ULL, &vector_compare64_state.sgprs[62],
           &vector_compare64_state.sgprs[63]);
  SplitU64(0x12345678ULL, &vector_compare64_state.sgprs[64],
           &vector_compare64_state.sgprs[65]);
  SplitU64(0xabcdefULL, &vector_compare64_state.sgprs[66],
           &vector_compare64_state.sgprs[67]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-4)),
           &vector_compare64_state.vgprs[20][0],
           &vector_compare64_state.vgprs[21][0]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-6)),
           &vector_compare64_state.vgprs[20][1],
           &vector_compare64_state.vgprs[21][1]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &vector_compare64_state.vgprs[20][3],
           &vector_compare64_state.vgprs[21][3]);
  SplitU64(20ULL, &vector_compare64_state.vgprs[22][0],
           &vector_compare64_state.vgprs[23][0]);
  SplitU64(10ULL, &vector_compare64_state.vgprs[22][1],
           &vector_compare64_state.vgprs[23][1]);
  SplitU64(1ULL, &vector_compare64_state.vgprs[22][3],
           &vector_compare64_state.vgprs[23][3]);
  SplitU64(0ULL, &vector_compare64_state.vgprs[24][0],
           &vector_compare64_state.vgprs[25][0]);
  SplitU64(1ULL, &vector_compare64_state.vgprs[24][1],
           &vector_compare64_state.vgprs[25][1]);
  SplitU64(2ULL, &vector_compare64_state.vgprs[24][3],
           &vector_compare64_state.vgprs[25][3]);
  SplitU64(3ULL, &vector_compare64_state.vgprs[26][0],
           &vector_compare64_state.vgprs[27][0]);
  SplitU64(4ULL, &vector_compare64_state.vgprs[26][1],
           &vector_compare64_state.vgprs[27][1]);
  SplitU64(5ULL, &vector_compare64_state.vgprs[26][3],
           &vector_compare64_state.vgprs[27][3]);
  if (!Expect(interpreter.ExecuteProgram(vector_compare64_program,
                                         &vector_compare64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare64_state.halted,
              "expected vector compare64 program to halt") ||
      !Expect(vector_compare64_state.sgprs[112] == 5u &&
                  vector_compare64_state.sgprs[113] == 0u,
              "expected v_cmp_lt_i64 mask result") ||
      !Expect(vector_compare64_state.sgprs[106] == 14u &&
                  vector_compare64_state.sgprs[107] == 0u,
              "expected v_cmp_ge_u64 mask result") ||
      !Expect(vector_compare64_state.sgprs[114] == 4u &&
                  vector_compare64_state.sgprs[115] == 0u,
              "expected v_cmp_f_i64 mask result") ||
      !Expect(vector_compare64_state.sgprs[116] == 15u &&
                  vector_compare64_state.sgprs[117] == 0u,
              "expected v_cmp_t_u64 mask result") ||
      !Expect(vector_compare64_state.vcc_mask == 15u,
              "expected final VCC mask after vector compare64 ops")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_compare64_program;
  if (!Expect(interpreter.CompileProgram(vector_compare64_program,
                                         &compiled_vector_compare64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_compare64_state;
  compiled_vector_compare64_state.exec_mask = 0b1011ULL;
  compiled_vector_compare64_state.vcc_mask = 0b0100ULL;
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &compiled_vector_compare64_state.sgprs[60],
           &compiled_vector_compare64_state.sgprs[61]);
  SplitU64(10ULL, &compiled_vector_compare64_state.sgprs[62],
           &compiled_vector_compare64_state.sgprs[63]);
  SplitU64(0x12345678ULL, &compiled_vector_compare64_state.sgprs[64],
           &compiled_vector_compare64_state.sgprs[65]);
  SplitU64(0xabcdefULL, &compiled_vector_compare64_state.sgprs[66],
           &compiled_vector_compare64_state.sgprs[67]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-4)),
           &compiled_vector_compare64_state.vgprs[20][0],
           &compiled_vector_compare64_state.vgprs[21][0]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-6)),
           &compiled_vector_compare64_state.vgprs[20][1],
           &compiled_vector_compare64_state.vgprs[21][1]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &compiled_vector_compare64_state.vgprs[20][3],
           &compiled_vector_compare64_state.vgprs[21][3]);
  SplitU64(20ULL, &compiled_vector_compare64_state.vgprs[22][0],
           &compiled_vector_compare64_state.vgprs[23][0]);
  SplitU64(10ULL, &compiled_vector_compare64_state.vgprs[22][1],
           &compiled_vector_compare64_state.vgprs[23][1]);
  SplitU64(1ULL, &compiled_vector_compare64_state.vgprs[22][3],
           &compiled_vector_compare64_state.vgprs[23][3]);
  SplitU64(0ULL, &compiled_vector_compare64_state.vgprs[24][0],
           &compiled_vector_compare64_state.vgprs[25][0]);
  SplitU64(1ULL, &compiled_vector_compare64_state.vgprs[24][1],
           &compiled_vector_compare64_state.vgprs[25][1]);
  SplitU64(2ULL, &compiled_vector_compare64_state.vgprs[24][3],
           &compiled_vector_compare64_state.vgprs[25][3]);
  SplitU64(3ULL, &compiled_vector_compare64_state.vgprs[26][0],
           &compiled_vector_compare64_state.vgprs[27][0]);
  SplitU64(4ULL, &compiled_vector_compare64_state.vgprs[26][1],
           &compiled_vector_compare64_state.vgprs[27][1]);
  SplitU64(5ULL, &compiled_vector_compare64_state.vgprs[26][3],
           &compiled_vector_compare64_state.vgprs[27][3]);
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_compare64_program,
                                         &compiled_vector_compare64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_compare64_state.halted,
              "expected compiled vector compare64 program to halt") ||
      !Expect(compiled_vector_compare64_state.sgprs[112] == 5u &&
                  compiled_vector_compare64_state.sgprs[113] == 0u,
              "expected compiled v_cmp_lt_i64 mask result") ||
      !Expect(compiled_vector_compare64_state.sgprs[106] == 14u &&
                  compiled_vector_compare64_state.sgprs[107] == 0u,
              "expected compiled v_cmp_ge_u64 mask result") ||
      !Expect(compiled_vector_compare64_state.sgprs[114] == 4u &&
                  compiled_vector_compare64_state.sgprs[115] == 0u,
              "expected compiled v_cmp_f_i64 mask result") ||
      !Expect(compiled_vector_compare64_state.sgprs[116] == 15u &&
                  compiled_vector_compare64_state.sgprs[117] == 0u,
              "expected compiled v_cmp_t_u64 mask result") ||
      !Expect(compiled_vector_compare64_state.vcc_mask == 15u,
              "expected compiled final VCC mask after vector compare64 ops")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_cmpx_program = {
      DecodedInstruction::Binary("V_CMPX_EQ_U32", InstructionOperand::Sgpr(106),
                                 InstructionOperand::Sgpr(12),
                                 InstructionOperand::Vgpr(12)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Binary("V_CMPX_LT_I64", InstructionOperand::Sgpr(108),
                                 InstructionOperand::Sgpr(60),
                                 InstructionOperand::Vgpr(20)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Binary("V_CMPX_GT_U64", InstructionOperand::Sgpr(110),
                                 InstructionOperand::Sgpr(62),
                                 InstructionOperand::Vgpr(22)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(2),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_cmpx_state;
  vector_cmpx_state.exec_mask = 0b1011ULL;
  vector_cmpx_state.vcc_mask = 0b0100ULL;
  vector_cmpx_state.sgprs[12] = 7u;
  vector_cmpx_state.vgprs[12][0] = 7u;
  vector_cmpx_state.vgprs[12][1] = 5u;
  vector_cmpx_state.vgprs[12][3] = 7u;
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &vector_cmpx_state.sgprs[60], &vector_cmpx_state.sgprs[61]);
  SplitU64(10ULL, &vector_cmpx_state.sgprs[62], &vector_cmpx_state.sgprs[63]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-4)),
           &vector_cmpx_state.vgprs[20][0], &vector_cmpx_state.vgprs[21][0]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-6)),
           &vector_cmpx_state.vgprs[20][1], &vector_cmpx_state.vgprs[21][1]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &vector_cmpx_state.vgprs[20][3], &vector_cmpx_state.vgprs[21][3]);
  SplitU64(20ULL, &vector_cmpx_state.vgprs[22][0], &vector_cmpx_state.vgprs[23][0]);
  SplitU64(10ULL, &vector_cmpx_state.vgprs[22][1], &vector_cmpx_state.vgprs[23][1]);
  SplitU64(1ULL, &vector_cmpx_state.vgprs[22][3], &vector_cmpx_state.vgprs[23][3]);
  if (!Expect(interpreter.ExecuteProgram(vector_cmpx_program, &vector_cmpx_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_state.halted, "expected vector cmpx program to halt") ||
      !Expect(vector_cmpx_state.sgprs[106] == 9u &&
                  vector_cmpx_state.sgprs[107] == 0u,
              "expected v_cmpx_eq_u32 mask result") ||
      !Expect(vector_cmpx_state.sgprs[108] == 1u &&
                  vector_cmpx_state.sgprs[109] == 0u,
              "expected v_cmpx_lt_i64 mask result") ||
      !Expect(vector_cmpx_state.sgprs[110] == 0u &&
                  vector_cmpx_state.sgprs[111] == 0u,
              "expected v_cmpx_gt_u64 mask result") ||
      !Expect(vector_cmpx_state.sgprs[0] == 111u,
              "expected EXECZ fallthrough after first cmpx") ||
      !Expect(vector_cmpx_state.sgprs[1] == 0u,
              "expected EXECNZ branch to skip second move") ||
      !Expect(vector_cmpx_state.sgprs[2] == 0u,
              "expected EXECZ branch to skip third move") ||
      !Expect(vector_cmpx_state.exec_mask == 0u,
              "expected final exec mask after cmpx chain") ||
      !Expect(vector_cmpx_state.vcc_mask == 0u,
              "expected final vcc mask after cmpx chain")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_cmpx_program;
  if (!Expect(interpreter.CompileProgram(vector_cmpx_program,
                                         &compiled_vector_cmpx_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_cmpx_state;
  compiled_vector_cmpx_state.exec_mask = 0b1011ULL;
  compiled_vector_cmpx_state.vcc_mask = 0b0100ULL;
  compiled_vector_cmpx_state.sgprs[12] = 7u;
  compiled_vector_cmpx_state.vgprs[12][0] = 7u;
  compiled_vector_cmpx_state.vgprs[12][1] = 5u;
  compiled_vector_cmpx_state.vgprs[12][3] = 7u;
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &compiled_vector_cmpx_state.sgprs[60],
           &compiled_vector_cmpx_state.sgprs[61]);
  SplitU64(10ULL, &compiled_vector_cmpx_state.sgprs[62],
           &compiled_vector_cmpx_state.sgprs[63]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-4)),
           &compiled_vector_cmpx_state.vgprs[20][0],
           &compiled_vector_cmpx_state.vgprs[21][0]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-6)),
           &compiled_vector_cmpx_state.vgprs[20][1],
           &compiled_vector_cmpx_state.vgprs[21][1]);
  SplitU64(static_cast<std::uint64_t>(static_cast<std::int64_t>(-5)),
           &compiled_vector_cmpx_state.vgprs[20][3],
           &compiled_vector_cmpx_state.vgprs[21][3]);
  SplitU64(20ULL, &compiled_vector_cmpx_state.vgprs[22][0],
           &compiled_vector_cmpx_state.vgprs[23][0]);
  SplitU64(10ULL, &compiled_vector_cmpx_state.vgprs[22][1],
           &compiled_vector_cmpx_state.vgprs[23][1]);
  SplitU64(1ULL, &compiled_vector_cmpx_state.vgprs[22][3],
           &compiled_vector_cmpx_state.vgprs[23][3]);
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_cmpx_program,
                                         &compiled_vector_cmpx_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_cmpx_state.halted,
              "expected compiled vector cmpx program to halt") ||
      !Expect(compiled_vector_cmpx_state.sgprs[106] == 9u &&
                  compiled_vector_cmpx_state.sgprs[107] == 0u,
              "expected compiled v_cmpx_eq_u32 mask result") ||
      !Expect(compiled_vector_cmpx_state.sgprs[108] == 1u &&
                  compiled_vector_cmpx_state.sgprs[109] == 0u,
              "expected compiled v_cmpx_lt_i64 mask result") ||
      !Expect(compiled_vector_cmpx_state.sgprs[110] == 0u &&
                  compiled_vector_cmpx_state.sgprs[111] == 0u,
              "expected compiled v_cmpx_gt_u64 mask result") ||
      !Expect(compiled_vector_cmpx_state.sgprs[0] == 111u,
              "expected compiled EXECZ fallthrough after first cmpx") ||
      !Expect(compiled_vector_cmpx_state.sgprs[1] == 0u,
              "expected compiled EXECNZ branch to skip second move") ||
      !Expect(compiled_vector_cmpx_state.sgprs[2] == 0u,
              "expected compiled EXECZ branch to skip third move") ||
      !Expect(compiled_vector_cmpx_state.exec_mask == 0u,
              "expected compiled final exec mask after cmpx chain") ||
      !Expect(compiled_vector_cmpx_state.vcc_mask == 0u,
              "expected compiled final vcc mask after cmpx chain")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_cmpx_f32_program = {
      DecodedInstruction::Binary("V_CMPX_LT_F32", InstructionOperand::Sgpr(112),
                                 InstructionOperand::Sgpr(16),
                                 InstructionOperand::Vgpr(16)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(5),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Binary("V_CMPX_U_F32", InstructionOperand::Sgpr(114),
                                 InstructionOperand::Sgpr(17),
                                 InstructionOperand::Vgpr(17)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(6),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Binary("V_CMPX_O_F32", InstructionOperand::Sgpr(116),
                                 InstructionOperand::Sgpr(19),
                                 InstructionOperand::Vgpr(19)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(7),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_cmpx_f32_state;
  vector_cmpx_f32_state.exec_mask = 0b1011ULL;
  vector_cmpx_f32_state.sgprs[16] = FloatBits(1.5f);
  vector_cmpx_f32_state.sgprs[17] = 0x7fc00000u;
  vector_cmpx_f32_state.sgprs[19] = 0x7fc00000u;
  vector_cmpx_f32_state.vgprs[16][0] = FloatBits(2.0f);
  vector_cmpx_f32_state.vgprs[16][1] = FloatBits(1.0f);
  vector_cmpx_f32_state.vgprs[16][3] = FloatBits(1.5f);
  vector_cmpx_f32_state.vgprs[17][0] = FloatBits(0.0f);
  vector_cmpx_f32_state.vgprs[17][1] = FloatBits(4.0f);
  vector_cmpx_f32_state.vgprs[17][3] = FloatBits(-1.0f);
  vector_cmpx_f32_state.vgprs[19][0] = FloatBits(1.0f);
  vector_cmpx_f32_state.vgprs[19][1] = FloatBits(0.0f);
  vector_cmpx_f32_state.vgprs[19][3] = FloatBits(2.0f);
  if (!Expect(interpreter.ExecuteProgram(vector_cmpx_f32_program,
                                         &vector_cmpx_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_f32_state.halted,
              "expected vector cmpx f32 program to halt") ||
      !Expect(vector_cmpx_f32_state.sgprs[112] == 1u &&
                  vector_cmpx_f32_state.sgprs[113] == 0u,
              "expected v_cmpx_lt_f32 mask result") ||
      !Expect(vector_cmpx_f32_state.sgprs[114] == 1u &&
                  vector_cmpx_f32_state.sgprs[115] == 0u,
              "expected v_cmpx_u_f32 mask result") ||
      !Expect(vector_cmpx_f32_state.sgprs[116] == 0u &&
                  vector_cmpx_f32_state.sgprs[117] == 0u,
              "expected v_cmpx_o_f32 mask result") ||
      !Expect(vector_cmpx_f32_state.sgprs[5] == 111u,
              "expected EXECZ fallthrough after first f32 cmpx") ||
      !Expect(vector_cmpx_f32_state.sgprs[6] == 0u,
              "expected EXECNZ branch to skip second move for f32 cmpx") ||
      !Expect(vector_cmpx_f32_state.sgprs[7] == 0u,
              "expected EXECZ branch to skip third move for f32 cmpx") ||
      !Expect(vector_cmpx_f32_state.exec_mask == 0u,
              "expected final exec mask after f32 cmpx chain") ||
      !Expect(vector_cmpx_f32_state.vcc_mask == 0u,
              "expected final vcc mask after f32 cmpx chain")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_cmpx_f32_program;
  if (!Expect(interpreter.CompileProgram(vector_cmpx_f32_program,
                                         &compiled_vector_cmpx_f32_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_cmpx_f32_state;
  compiled_vector_cmpx_f32_state.exec_mask = 0b1011ULL;
  compiled_vector_cmpx_f32_state.sgprs[16] = FloatBits(1.5f);
  compiled_vector_cmpx_f32_state.sgprs[17] = 0x7fc00000u;
  compiled_vector_cmpx_f32_state.sgprs[19] = 0x7fc00000u;
  compiled_vector_cmpx_f32_state.vgprs[16][0] = FloatBits(2.0f);
  compiled_vector_cmpx_f32_state.vgprs[16][1] = FloatBits(1.0f);
  compiled_vector_cmpx_f32_state.vgprs[16][3] = FloatBits(1.5f);
  compiled_vector_cmpx_f32_state.vgprs[17][0] = FloatBits(0.0f);
  compiled_vector_cmpx_f32_state.vgprs[17][1] = FloatBits(4.0f);
  compiled_vector_cmpx_f32_state.vgprs[17][3] = FloatBits(-1.0f);
  compiled_vector_cmpx_f32_state.vgprs[19][0] = FloatBits(1.0f);
  compiled_vector_cmpx_f32_state.vgprs[19][1] = FloatBits(0.0f);
  compiled_vector_cmpx_f32_state.vgprs[19][3] = FloatBits(2.0f);
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_cmpx_f32_program,
                                         &compiled_vector_cmpx_f32_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_cmpx_f32_state.halted,
              "expected compiled vector cmpx f32 program to halt") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[112] == 1u &&
                  compiled_vector_cmpx_f32_state.sgprs[113] == 0u,
              "expected compiled v_cmpx_lt_f32 mask result") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[114] == 1u &&
                  compiled_vector_cmpx_f32_state.sgprs[115] == 0u,
              "expected compiled v_cmpx_u_f32 mask result") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[116] == 0u &&
                  compiled_vector_cmpx_f32_state.sgprs[117] == 0u,
              "expected compiled v_cmpx_o_f32 mask result") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[5] == 111u,
              "expected compiled EXECZ fallthrough after first f32 cmpx") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[6] == 0u,
              "expected compiled EXECNZ branch to skip second move for f32 cmpx") ||
      !Expect(compiled_vector_cmpx_f32_state.sgprs[7] == 0u,
              "expected compiled EXECZ branch to skip third move for f32 cmpx") ||
      !Expect(compiled_vector_cmpx_f32_state.exec_mask == 0u,
              "expected compiled final exec mask after f32 cmpx chain") ||
      !Expect(compiled_vector_cmpx_f32_state.vcc_mask == 0u,
              "expected compiled final vcc mask after f32 cmpx chain")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_cmpx_f64_program = {
      DecodedInstruction::Binary("V_CMPX_LT_F64", InstructionOperand::Sgpr(112),
                                 InstructionOperand::Sgpr(70),
                                 InstructionOperand::Vgpr(30)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(8),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Binary("V_CMPX_U_F64", InstructionOperand::Sgpr(114),
                                 InstructionOperand::Sgpr(72),
                                 InstructionOperand::Vgpr(32)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(9),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Binary("V_CMPX_O_F64", InstructionOperand::Sgpr(116),
                                 InstructionOperand::Sgpr(74),
                                 InstructionOperand::Vgpr(34)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(10),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_cmpx_f64_state;
  vector_cmpx_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &vector_cmpx_f64_state.sgprs[70],
           &vector_cmpx_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_cmpx_f64_state.sgprs[72],
           &vector_cmpx_f64_state.sgprs[73]);
  SplitU64(kQuietNan64, &vector_cmpx_f64_state.sgprs[74],
           &vector_cmpx_f64_state.sgprs[75]);
  SplitU64(DoubleBits(2.0), &vector_cmpx_f64_state.vgprs[30][0],
           &vector_cmpx_f64_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_f64_state.vgprs[30][1],
           &vector_cmpx_f64_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &vector_cmpx_f64_state.vgprs[30][3],
           &vector_cmpx_f64_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &vector_cmpx_f64_state.vgprs[32][0],
           &vector_cmpx_f64_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &vector_cmpx_f64_state.vgprs[32][1],
           &vector_cmpx_f64_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &vector_cmpx_f64_state.vgprs[32][3],
           &vector_cmpx_f64_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_f64_state.vgprs[34][0],
           &vector_cmpx_f64_state.vgprs[35][0]);
  SplitU64(DoubleBits(0.0), &vector_cmpx_f64_state.vgprs[34][1],
           &vector_cmpx_f64_state.vgprs[35][1]);
  SplitU64(DoubleBits(2.0), &vector_cmpx_f64_state.vgprs[34][3],
           &vector_cmpx_f64_state.vgprs[35][3]);
  if (!Expect(interpreter.ExecuteProgram(vector_cmpx_f64_program,
                                         &vector_cmpx_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_f64_state.halted,
              "expected vector cmpx f64 program to halt") ||
      !Expect(vector_cmpx_f64_state.sgprs[112] == 1u &&
                  vector_cmpx_f64_state.sgprs[113] == 0u,
              "expected v_cmpx_lt_f64 mask result") ||
      !Expect(vector_cmpx_f64_state.sgprs[114] == 1u &&
                  vector_cmpx_f64_state.sgprs[115] == 0u,
              "expected v_cmpx_u_f64 mask result") ||
      !Expect(vector_cmpx_f64_state.sgprs[116] == 0u &&
                  vector_cmpx_f64_state.sgprs[117] == 0u,
              "expected v_cmpx_o_f64 mask result") ||
      !Expect(vector_cmpx_f64_state.sgprs[8] == 111u,
              "expected EXECZ fallthrough after first f64 cmpx") ||
      !Expect(vector_cmpx_f64_state.sgprs[9] == 0u,
              "expected EXECNZ branch to skip second move for f64 cmpx") ||
      !Expect(vector_cmpx_f64_state.sgprs[10] == 0u,
              "expected EXECZ branch to skip third move for f64 cmpx") ||
      !Expect(vector_cmpx_f64_state.exec_mask == 0u,
              "expected final exec mask after f64 cmpx chain") ||
      !Expect(vector_cmpx_f64_state.vcc_mask == 0u,
              "expected final vcc mask after f64 cmpx chain")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_cmpx_f64_program;
  if (!Expect(interpreter.CompileProgram(vector_cmpx_f64_program,
                                         &compiled_vector_cmpx_f64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_cmpx_f64_state;
  compiled_vector_cmpx_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &compiled_vector_cmpx_f64_state.sgprs[70],
           &compiled_vector_cmpx_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &compiled_vector_cmpx_f64_state.sgprs[72],
           &compiled_vector_cmpx_f64_state.sgprs[73]);
  SplitU64(kQuietNan64, &compiled_vector_cmpx_f64_state.sgprs[74],
           &compiled_vector_cmpx_f64_state.sgprs[75]);
  SplitU64(DoubleBits(2.0), &compiled_vector_cmpx_f64_state.vgprs[30][0],
           &compiled_vector_cmpx_f64_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &compiled_vector_cmpx_f64_state.vgprs[30][1],
           &compiled_vector_cmpx_f64_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &compiled_vector_cmpx_f64_state.vgprs[30][3],
           &compiled_vector_cmpx_f64_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &compiled_vector_cmpx_f64_state.vgprs[32][0],
           &compiled_vector_cmpx_f64_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &compiled_vector_cmpx_f64_state.vgprs[32][1],
           &compiled_vector_cmpx_f64_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &compiled_vector_cmpx_f64_state.vgprs[32][3],
           &compiled_vector_cmpx_f64_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &compiled_vector_cmpx_f64_state.vgprs[34][0],
           &compiled_vector_cmpx_f64_state.vgprs[35][0]);
  SplitU64(DoubleBits(0.0), &compiled_vector_cmpx_f64_state.vgprs[34][1],
           &compiled_vector_cmpx_f64_state.vgprs[35][1]);
  SplitU64(DoubleBits(2.0), &compiled_vector_cmpx_f64_state.vgprs[34][3],
           &compiled_vector_cmpx_f64_state.vgprs[35][3]);
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_cmpx_f64_program,
                                         &compiled_vector_cmpx_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_cmpx_f64_state.halted,
              "expected compiled vector cmpx f64 program to halt") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[112] == 1u &&
                  compiled_vector_cmpx_f64_state.sgprs[113] == 0u,
              "expected compiled v_cmpx_lt_f64 mask result") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[114] == 1u &&
                  compiled_vector_cmpx_f64_state.sgprs[115] == 0u,
              "expected compiled v_cmpx_u_f64 mask result") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[116] == 0u &&
                  compiled_vector_cmpx_f64_state.sgprs[117] == 0u,
              "expected compiled v_cmpx_o_f64 mask result") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[8] == 111u,
              "expected compiled EXECZ fallthrough after first f64 cmpx") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[9] == 0u,
              "expected compiled EXECNZ branch to skip second move for f64 cmpx") ||
      !Expect(compiled_vector_cmpx_f64_state.sgprs[10] == 0u,
              "expected compiled EXECZ branch to skip third move for f64 cmpx") ||
      !Expect(compiled_vector_cmpx_f64_state.exec_mask == 0u,
              "expected compiled final exec mask after f64 cmpx chain") ||
      !Expect(compiled_vector_cmpx_f64_state.vcc_mask == 0u,
              "expected compiled final vcc mask after f64 cmpx chain")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_cmpx_class_f64_program = {
      DecodedInstruction::Binary("V_CMPX_CLASS_F64", InstructionOperand::Sgpr(112),
                                 InstructionOperand::Sgpr(70),
                                 InstructionOperand::Vgpr(40)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(11),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Binary("V_CMPX_CLASS_F64", InstructionOperand::Sgpr(114),
                                 InstructionOperand::Sgpr(72),
                                 InstructionOperand::Vgpr(41)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(12),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Binary("V_CMPX_CLASS_F64", InstructionOperand::Sgpr(116),
                                 InstructionOperand::Sgpr(74),
                                 InstructionOperand::Vgpr(42)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(13),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState vector_cmpx_class_f64_state;
  vector_cmpx_class_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(-0.0), &vector_cmpx_class_f64_state.sgprs[70],
           &vector_cmpx_class_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_cmpx_class_f64_state.sgprs[72],
           &vector_cmpx_class_f64_state.sgprs[73]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_class_f64_state.sgprs[74],
           &vector_cmpx_class_f64_state.sgprs[75]);
  vector_cmpx_class_f64_state.vgprs[40][0] = 0x20u;
  vector_cmpx_class_f64_state.vgprs[40][1] = 0x40u;
  vector_cmpx_class_f64_state.vgprs[40][3] = 0x60u;
  vector_cmpx_class_f64_state.vgprs[41][0] = 0x2u;
  vector_cmpx_class_f64_state.vgprs[41][3] = 0x1u;
  vector_cmpx_class_f64_state.vgprs[42][0] = 0x20u;
  if (!Expect(interpreter.ExecuteProgram(vector_cmpx_class_f64_program,
                                         &vector_cmpx_class_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_class_f64_state.halted,
              "expected vector cmpx class f64 program to halt") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[112] == 9u &&
                  vector_cmpx_class_f64_state.sgprs[113] == 0u,
              "expected v_cmpx_class_f64 first mask result") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[114] == 1u &&
                  vector_cmpx_class_f64_state.sgprs[115] == 0u,
              "expected v_cmpx_class_f64 second mask result") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[116] == 0u &&
                  vector_cmpx_class_f64_state.sgprs[117] == 0u,
              "expected v_cmpx_class_f64 third mask result") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[11] == 111u,
              "expected EXECZ fallthrough after first class cmpx") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[12] == 0u,
              "expected EXECNZ branch to skip second move after class cmpx") ||
      !Expect(vector_cmpx_class_f64_state.sgprs[13] == 0u,
              "expected EXECZ branch to skip third move after class cmpx") ||
      !Expect(vector_cmpx_class_f64_state.exec_mask == 0u,
              "expected final exec mask after class cmpx chain") ||
      !Expect(vector_cmpx_class_f64_state.vcc_mask == 0u,
              "expected final vcc mask after class cmpx chain")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_cmpx_class_f64_program;
  if (!Expect(interpreter.CompileProgram(vector_cmpx_class_f64_program,
                                         &compiled_vector_cmpx_class_f64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vector_cmpx_class_f64_state;
  compiled_vector_cmpx_class_f64_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(-0.0), &compiled_vector_cmpx_class_f64_state.sgprs[70],
           &compiled_vector_cmpx_class_f64_state.sgprs[71]);
  SplitU64(kQuietNan64, &compiled_vector_cmpx_class_f64_state.sgprs[72],
           &compiled_vector_cmpx_class_f64_state.sgprs[73]);
  SplitU64(DoubleBits(1.0), &compiled_vector_cmpx_class_f64_state.sgprs[74],
           &compiled_vector_cmpx_class_f64_state.sgprs[75]);
  compiled_vector_cmpx_class_f64_state.vgprs[40][0] = 0x20u;
  compiled_vector_cmpx_class_f64_state.vgprs[40][1] = 0x40u;
  compiled_vector_cmpx_class_f64_state.vgprs[40][3] = 0x60u;
  compiled_vector_cmpx_class_f64_state.vgprs[41][0] = 0x2u;
  compiled_vector_cmpx_class_f64_state.vgprs[41][3] = 0x1u;
  compiled_vector_cmpx_class_f64_state.vgprs[42][0] = 0x20u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_cmpx_class_f64_program,
                                         &compiled_vector_cmpx_class_f64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vector_cmpx_class_f64_state.halted,
              "expected compiled vector cmpx class f64 program to halt") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[112] == 9u &&
                  compiled_vector_cmpx_class_f64_state.sgprs[113] == 0u,
              "expected compiled v_cmpx_class_f64 first mask result") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[114] == 1u &&
                  compiled_vector_cmpx_class_f64_state.sgprs[115] == 0u,
              "expected compiled v_cmpx_class_f64 second mask result") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[116] == 0u &&
                  compiled_vector_cmpx_class_f64_state.sgprs[117] == 0u,
              "expected compiled v_cmpx_class_f64 third mask result") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[11] == 111u,
              "expected compiled EXECZ fallthrough after first class cmpx") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[12] == 0u,
              "expected compiled EXECNZ branch to skip second move after class cmpx") ||
      !Expect(compiled_vector_cmpx_class_f64_state.sgprs[13] == 0u,
              "expected compiled EXECZ branch to skip third move after class cmpx") ||
      !Expect(compiled_vector_cmpx_class_f64_state.exec_mask == 0u,
              "expected compiled final exec mask after class cmpx chain") ||
      !Expect(compiled_vector_cmpx_class_f64_state.vcc_mask == 0u,
              "expected compiled final vcc mask after class cmpx chain")) {
    return 1;
  }

  const std::vector<DecodedInstruction> scalar_extended_program = {
      DecodedInstruction::Unary("S_MOVK_I32", InstructionOperand::Sgpr(5),
                                InstructionOperand::Imm32(
                                    static_cast<std::uint32_t>(-3))),
      DecodedInstruction::Binary("S_ADDK_I32", InstructionOperand::Sgpr(5),
                                 InstructionOperand::Sgpr(5),
                                 InstructionOperand::Imm32(5)),
      DecodedInstruction::Binary("S_MULK_I32", InstructionOperand::Sgpr(5),
                                 InstructionOperand::Sgpr(5),
                                 InstructionOperand::Imm32(
                                     static_cast<std::uint32_t>(-4))),
      DecodedInstruction::Binary("S_MUL_HI_U32", InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Imm32(2)),
      DecodedInstruction::Binary("S_MUL_HI_I32", InstructionOperand::Sgpr(3),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(2)),
      DecodedInstruction::Binary("S_MUL_I32", InstructionOperand::Sgpr(4),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(2)),
      DecodedInstruction::TwoOperand("S_CMP_GT_U32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_CMOVK_I32", InstructionOperand::Sgpr(6),
                                InstructionOperand::Imm32(42)),
      DecodedInstruction::TwoOperand("S_CMP_LT_I32", InstructionOperand::Sgpr(5),
                                     InstructionOperand::Imm32(
                                         static_cast<std::uint32_t>(-1))),
      DecodedInstruction::Unary("S_CMOVK_I32", InstructionOperand::Sgpr(7),
                                InstructionOperand::Imm32(
                                    static_cast<std::uint32_t>(-7))),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState scalar_extended_state;
  scalar_extended_state.sgprs[0] = 0xffffffffu;
  scalar_extended_state.sgprs[1] = 0x80000000u;
  if (!Expect(interpreter.ExecuteProgram(scalar_extended_program,
                                         &scalar_extended_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(scalar_extended_state.halted,
              "expected scalar extended program to halt") ||
      !Expect(scalar_extended_state.sgprs[2] == 1u,
              "expected s_mul_hi_u32 result") ||
      !Expect(scalar_extended_state.sgprs[3] == 0xffffffffu,
              "expected s_mul_hi_i32 result") ||
      !Expect(scalar_extended_state.sgprs[4] == 0u,
              "expected s_mul_i32 result") ||
      !Expect(scalar_extended_state.sgprs[5] == 0xfffffff8u,
              "expected k-form scalar result") ||
      !Expect(scalar_extended_state.sgprs[6] == 42u,
              "expected s_cmovk_i32 true-path result") ||
      !Expect(scalar_extended_state.sgprs[7] == 0xfffffff9u,
              "expected s_cmovk_i32 post-compare result") ||
      !Expect(scalar_extended_state.scc,
              "expected final signed compare to set SCC")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_scalar_extended_program;
  if (!Expect(interpreter.CompileProgram(scalar_extended_program,
                                         &compiled_scalar_extended_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_scalar_extended_state;
  compiled_scalar_extended_state.sgprs[0] = 0xffffffffu;
  compiled_scalar_extended_state.sgprs[1] = 0x80000000u;
  if (!Expect(interpreter.ExecuteProgram(compiled_scalar_extended_program,
                                         &compiled_scalar_extended_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_scalar_extended_state.halted,
              "expected compiled scalar extended program to halt") ||
      !Expect(compiled_scalar_extended_state.sgprs[2] == 1u,
              "expected compiled s_mul_hi_u32 result") ||
      !Expect(compiled_scalar_extended_state.sgprs[3] == 0xffffffffu,
              "expected compiled s_mul_hi_i32 result") ||
      !Expect(compiled_scalar_extended_state.sgprs[4] == 0u,
              "expected compiled s_mul_i32 result") ||
      !Expect(compiled_scalar_extended_state.sgprs[5] == 0xfffffff8u,
              "expected compiled k-form scalar result") ||
      !Expect(compiled_scalar_extended_state.sgprs[6] == 42u,
              "expected compiled s_cmovk_i32 true-path result") ||
      !Expect(compiled_scalar_extended_state.sgprs[7] == 0xfffffff9u,
              "expected compiled s_cmovk_i32 post-compare result") ||
      !Expect(compiled_scalar_extended_state.scc,
              "expected compiled final signed compare to set SCC")) {
    return 1;
  }

  WaveExecutionState unsupported_state;
  const std::vector<DecodedInstruction> unsupported_program = {
      DecodedInstruction::Nullary("BUFFER_LOAD_DWORD"),
  };
  if (!Expect(!interpreter.ExecuteProgram(unsupported_program, &unsupported_state,
                                          &error_message),
              "expected malformed buffer opcode to fail") ||
      !Expect(!error_message.empty(), "expected malformed buffer opcode error")) {
    return 1;
  }

  WaveExecutionState ds_state;
  ds_state.exec_mask = 0b1011ULL;
  ds_state.vgprs[0][0] = 0u;
  ds_state.vgprs[0][1] = 4u;
  ds_state.vgprs[0][3] = 8u;
  ds_state.vgprs[1][0] = 10u;
  ds_state.vgprs[1][1] = 20u;
  ds_state.vgprs[1][3] = 40u;
  ds_state.vgprs[2][0] = 1u;
  ds_state.vgprs[2][1] = 2u;
  ds_state.vgprs[2][3] = 4u;
  ds_state.vgprs[3][2] = 0xdeadbeefu;
  const std::vector<DecodedInstruction> ds_program = {
      DecodedInstruction::Nullary("DS_NOP"),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_ADD_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(3),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(ds_program, &ds_state, &error_message),
              error_message.c_str()) ||
      !Expect(ds_state.vgprs[3][0] == 11u, "expected ds lane 0 read result") ||
      !Expect(ds_state.vgprs[3][1] == 22u, "expected ds lane 1 read result") ||
      !Expect(ds_state.vgprs[3][2] == 0xdeadbeefu,
              "expected inactive lane ds destination to remain untouched") ||
      !Expect(ds_state.vgprs[3][3] == 44u, "expected ds lane 3 read result")) {
    return 1;
  }

  std::uint32_t lds_lane0 = 0;
  std::uint32_t lds_lane1 = 0;
  std::uint32_t lds_lane3 = 0;
  std::memcpy(&lds_lane0, ds_state.lds_bytes.data() + 0, sizeof(lds_lane0));
  std::memcpy(&lds_lane1, ds_state.lds_bytes.data() + 4, sizeof(lds_lane1));
  std::memcpy(&lds_lane3, ds_state.lds_bytes.data() + 8, sizeof(lds_lane3));
  if (!Expect(lds_lane0 == 11u, "expected lds lane 0 result") ||
      !Expect(lds_lane1 == 22u, "expected lds lane 1 result") ||
      !Expect(lds_lane3 == 44u, "expected lds lane 3 result")) {
    return 1;
  }

  WaveExecutionState ds_integer_state;
  ds_integer_state.exec_mask = 0b1011ULL;
  ds_integer_state.vgprs[0][0] = 0u;
  ds_integer_state.vgprs[0][1] = 4u;
  ds_integer_state.vgprs[0][3] = 8u;
  ds_integer_state.vgprs[1][0] = 10u;
  ds_integer_state.vgprs[1][1] = 20u;
  ds_integer_state.vgprs[1][3] = 40u;
  ds_integer_state.vgprs[2][0] = 1u;
  ds_integer_state.vgprs[2][1] = 2u;
  ds_integer_state.vgprs[2][3] = 4u;
  ds_integer_state.vgprs[3][0] = 100u;
  ds_integer_state.vgprs[3][1] = 100u;
  ds_integer_state.vgprs[3][3] = 100u;
  ds_integer_state.vgprs[4][0] = 100u;
  ds_integer_state.vgprs[4][1] = 82u;
  ds_integer_state.vgprs[4][3] = 63u;
  ds_integer_state.vgprs[5][0] = 10u;
  ds_integer_state.vgprs[5][1] = 0u;
  ds_integer_state.vgprs[5][3] = 5u;
  ds_integer_state.vgprs[6][0] = 0xfffffff0u;
  ds_integer_state.vgprs[6][1] = 5u;
  ds_integer_state.vgprs[6][3] = 0xffffff00u;
  ds_integer_state.vgprs[7][0] = 0xfffffff8u;
  ds_integer_state.vgprs[7][1] = 0xffffffffu;
  ds_integer_state.vgprs[7][3] = 0xffffff80u;
  ds_integer_state.vgprs[8][0] = 3u;
  ds_integer_state.vgprs[8][1] = 7u;
  ds_integer_state.vgprs[8][3] = 1u;
  ds_integer_state.vgprs[9][0] = 5u;
  ds_integer_state.vgprs[9][1] = 9u;
  ds_integer_state.vgprs[9][3] = 2u;
  ds_integer_state.vgprs[10][0] = 7u;
  ds_integer_state.vgprs[10][1] = 6u;
  ds_integer_state.vgprs[10][3] = 3u;
  ds_integer_state.vgprs[11][0] = 8u;
  ds_integer_state.vgprs[11][1] = 1u;
  ds_integer_state.vgprs[11][3] = 4u;
  ds_integer_state.vgprs[12][0] = 2u;
  ds_integer_state.vgprs[12][1] = 3u;
  ds_integer_state.vgprs[12][3] = 5u;
  ds_integer_state.vgprs[13][2] = 0xdeadbeefu;
  const std::vector<DecodedInstruction> ds_integer_program = {
      DecodedInstruction::Nullary("DS_NOP"),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_SUB_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_RSUB_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(3),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_INC_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_DEC_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(5),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_I32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_I32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(7),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(8),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_U32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(9),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_AND_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(10),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_OR_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(11),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_XOR_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(12),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(13),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<CompiledInstruction> compiled_ds_integer_program;
  if (!Expect(interpreter.CompileProgram(ds_integer_program,
                                         &compiled_ds_integer_program,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_ds_integer_program,
                                         &ds_integer_state, &error_message),
              error_message.c_str()) ||
      !Expect(ds_integer_state.vgprs[13][0] == 15u,
              "expected compiled ds integer lane 0 result") ||
      !Expect(ds_integer_state.vgprs[13][1] == 2u,
              "expected compiled ds integer lane 1 result") ||
      !Expect(ds_integer_state.vgprs[13][2] == 0xdeadbeefu,
              "expected compiled ds integer inactive lane result") ||
      !Expect(ds_integer_state.vgprs[13][3] == 3u,
              "expected compiled ds integer lane 3 result")) {
    return 1;
  }

  lds_lane0 = 0;
  lds_lane1 = 0;
  lds_lane3 = 0;
  std::memcpy(&lds_lane0, ds_integer_state.lds_bytes.data() + 0,
              sizeof(lds_lane0));
  std::memcpy(&lds_lane1, ds_integer_state.lds_bytes.data() + 4,
              sizeof(lds_lane1));
  std::memcpy(&lds_lane3, ds_integer_state.lds_bytes.data() + 8,
              sizeof(lds_lane3));
  if (!Expect(lds_lane0 == 15u, "expected compiled ds integer lds lane 0") ||
      !Expect(lds_lane1 == 2u, "expected compiled ds integer lds lane 1") ||
      !Expect(lds_lane3 == 3u, "expected compiled ds integer lds lane 3")) {
    return 1;
  }

  WaveExecutionState ds_float_state;
  ds_float_state.exec_mask = 0b1011ULL;
  ds_float_state.vgprs[0][0] = 0u;
  ds_float_state.vgprs[0][1] = 4u;
  ds_float_state.vgprs[0][3] = 8u;
  ds_float_state.vgprs[1][0] = 0x3fc00000u;
  ds_float_state.vgprs[1][1] = 0xc0000000u;
  ds_float_state.vgprs[1][3] = 0x41200000u;
  ds_float_state.vgprs[2][0] = 0x40100000u;
  ds_float_state.vgprs[2][1] = 0x3f800000u;
  ds_float_state.vgprs[2][3] = 0xc0a00000u;
  ds_float_state.vgprs[3][0] = 0x40800000u;
  ds_float_state.vgprs[3][1] = 0xc0400000u;
  ds_float_state.vgprs[3][3] = 0x40c00000u;
  ds_float_state.vgprs[4][0] = 0x40600000u;
  ds_float_state.vgprs[4][1] = 0xc0200000u;
  ds_float_state.vgprs[4][3] = 0x40e00000u;
  ds_float_state.vgprs[5][2] = 0xdeadbeefu;
  ds_float_state.vgprs[6][0] = 16u;
  ds_float_state.vgprs[6][1] = 20u;
  ds_float_state.vgprs[6][3] = 24u;
  ds_float_state.vgprs[7][0] = 0x11223344u;
  ds_float_state.vgprs[7][1] = 0xaabbccddu;
  ds_float_state.vgprs[7][3] = 0x01020304u;
  ds_float_state.vgprs[8][0] = 0x77u;
  ds_float_state.vgprs[8][1] = 0x66u;
  ds_float_state.vgprs[8][3] = 0xcdu;
  ds_float_state.vgprs[10][2] = 0xdeadbeefu;
  ds_float_state.vgprs[11][0] = 32u;
  ds_float_state.vgprs[11][1] = 36u;
  ds_float_state.vgprs[11][3] = 40u;
  ds_float_state.vgprs[12][0] = 0x11223344u;
  ds_float_state.vgprs[12][1] = 0xaabbccddu;
  ds_float_state.vgprs[12][3] = 0x01020304u;
  ds_float_state.vgprs[14][0] = 0x5566u;
  ds_float_state.vgprs[14][1] = 0x1234u;
  ds_float_state.vgprs[14][3] = 0xabcdu;
  ds_float_state.vgprs[15][2] = 0xdeadbeefu;
  const std::vector<DecodedInstruction> ds_float_program = {
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_ADD_F32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_F32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(3),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_F32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(5),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(6),
                                       InstructionOperand::Vgpr(7),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B8", InstructionOperand::Vgpr(6),
                                       InstructionOperand::Vgpr(8),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(10),
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(11),
                                       InstructionOperand::Vgpr(12),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B16", InstructionOperand::Vgpr(11),
                                       InstructionOperand::Vgpr(14),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(15),
                                       InstructionOperand::Vgpr(11),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<CompiledInstruction> compiled_ds_float_program;
  if (!Expect(interpreter.CompileProgram(ds_float_program,
                                         &compiled_ds_float_program,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(interpreter.ExecuteProgram(compiled_ds_float_program,
                                         &ds_float_state, &error_message),
              error_message.c_str()) ||
      !Expect(ds_float_state.vgprs[5][0] == 0x40700000u,
              "expected compiled ds float lane 0 result") ||
      !Expect(ds_float_state.vgprs[5][1] == 0xc0200000u,
              "expected compiled ds float lane 1 result") ||
      !Expect(ds_float_state.vgprs[5][2] == 0xdeadbeefu,
              "expected compiled ds float inactive lane result") ||
      !Expect(ds_float_state.vgprs[5][3] == 0x40e00000u,
              "expected compiled ds float lane 3 result") ||
      !Expect(ds_float_state.vgprs[10][0] == 0x11223377u,
              "expected compiled ds byte-write lane 0 result") ||
      !Expect(ds_float_state.vgprs[10][1] == 0xaabbcc66u,
              "expected compiled ds byte-write lane 1 result") ||
      !Expect(ds_float_state.vgprs[10][2] == 0xdeadbeefu,
              "expected compiled ds byte-write inactive lane result") ||
      !Expect(ds_float_state.vgprs[10][3] == 0x010203cdu,
              "expected compiled ds byte-write lane 3 result") ||
      !Expect(ds_float_state.vgprs[15][0] == 0x11225566u,
              "expected compiled ds half-write lane 0 result") ||
      !Expect(ds_float_state.vgprs[15][1] == 0xaabb1234u,
              "expected compiled ds half-write lane 1 result") ||
      !Expect(ds_float_state.vgprs[15][2] == 0xdeadbeefu,
              "expected compiled ds half-write inactive lane result") ||
      !Expect(ds_float_state.vgprs[15][3] == 0x0102abcdu,
              "expected compiled ds half-write lane 3 result")) {
    return 1;
  }

  {
  const std::vector<DecodedInstruction> ds_packed_add_program = {
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_PK_ADD_F16", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(3),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(4),
                                       InstructionOperand::Vgpr(5),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_PK_ADD_BF16",
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(7),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_packed_add_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[0][0] = 0u;
    state.vgprs[0][1] = 4u;
    state.vgprs[0][3] = 8u;
    state.vgprs[1][0] = 0x40003c00u;
    state.vgprs[1][1] = 0x0000bc00u;
    state.vgprs[1][3] = 0x4400c000u;
    state.vgprs[2][0] = 0x3c004000u;
    state.vgprs[2][1] = 0x3c003800u;
    state.vgprs[2][3] = 0x40003c00u;
    state.vgprs[4][0] = 16u;
    state.vgprs[4][1] = 20u;
    state.vgprs[4][3] = 24u;
    state.vgprs[5][0] = 0x40003f80u;
    state.vgprs[5][1] = 0x0000bf80u;
    state.vgprs[5][3] = 0x4080c000u;
    state.vgprs[6][0] = 0x3f804000u;
    state.vgprs[6][1] = 0x3f803f00u;
    state.vgprs[6][3] = 0x40003f80u;
    state.vgprs[3][2] = 0xdeadbeefu;
    state.vgprs[7][2] = 0xcafebabeu;
    return state;
  };
  auto validate_ds_packed_add_state = [&](const WaveExecutionState& state,
                                          const char* mode) {
    if (!Expect(state.halted, "expected ds packed-add program to halt") ||
        !Expect(state.vgprs[3][0] == 0x42004200u,
                "expected ds_pk_add_f16 lane 0 result") ||
        !Expect(state.vgprs[3][1] == 0x3c00b800u,
                "expected ds_pk_add_f16 lane 1 result") ||
        !Expect(state.vgprs[3][2] == 0xdeadbeefu,
                "expected ds_pk_add_f16 inactive lane result") ||
        !Expect(state.vgprs[3][3] == 0x4600bc00u,
                "expected ds_pk_add_f16 lane 3 result") ||
        !Expect(state.vgprs[7][0] == 0x40404040u,
                "expected ds_pk_add_bf16 lane 0 result") ||
        !Expect(state.vgprs[7][1] == 0x3f80bf00u,
                "expected ds_pk_add_bf16 lane 1 result") ||
        !Expect(state.vgprs[7][2] == 0xcafebabeu,
                "expected ds_pk_add_bf16 inactive lane result") ||
        !Expect(state.vgprs[7][3] == 0x40c0bf80u,
                "expected ds_pk_add_bf16 lane 3 result")) {
      std::cerr << mode << '\n';
      return false;
    }

    const auto expect_lds_value = [&](std::size_t address,
                                      std::uint32_t expected,
                                      const char* label) {
      std::uint32_t value = 0;
      std::memcpy(&value, state.lds_bytes.data() + address, sizeof(value));
      return Expect(value == expected, label);
    };
    if (!expect_lds_value(0u, 0x42004200u, "expected ds_pk_add_f16 lane 0 lds") ||
        !expect_lds_value(4u, 0x3c00b800u, "expected ds_pk_add_f16 lane 1 lds") ||
        !expect_lds_value(8u, 0x4600bc00u, "expected ds_pk_add_f16 lane 3 lds") ||
        !expect_lds_value(16u, 0x40404040u, "expected ds_pk_add_bf16 lane 0 lds") ||
        !expect_lds_value(20u, 0x3f80bf00u, "expected ds_pk_add_bf16 lane 1 lds") ||
        !expect_lds_value(24u, 0x40c0bf80u, "expected ds_pk_add_bf16 lane 3 lds")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_packed_add_state = make_ds_packed_add_state();
  if (!Expect(interpreter.ExecuteProgram(ds_packed_add_program,
                                         &decoded_ds_packed_add_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_packed_add_state(decoded_ds_packed_add_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_packed_add_program;
  if (!Expect(interpreter.CompileProgram(ds_packed_add_program,
                                         &compiled_ds_packed_add_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_packed_add_state = make_ds_packed_add_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_packed_add_program,
                                         &compiled_ds_packed_add_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_packed_add_state(compiled_ds_packed_add_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_swizzle_program = {
      DecodedInstruction::ThreeOperand("DS_SWIZZLE_B32",
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0x041fu)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_swizzle_state = []() {
    WaveExecutionState state;
    state.exec_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 3) |
                      (1ULL << 32) | (1ULL << 33);
    state.vgprs[0][0] = 11u;
    state.vgprs[0][1] = 22u;
    state.vgprs[0][2] = 33u;
    state.vgprs[0][3] = 44u;
    state.vgprs[0][32] = 320u;
    state.vgprs[0][33] = 330u;
    state.vgprs[1][2] = 0xdeadbeefu;
    return state;
  };
  auto validate_ds_swizzle_state = [&](const WaveExecutionState& state,
                                       const char* mode) {
    if (!Expect(state.halted, "expected ds swizzle program to halt") ||
        !Expect(state.vgprs[1][0] == 22u,
                "expected ds swizzle lane 0 result") ||
        !Expect(state.vgprs[1][1] == 11u,
                "expected ds swizzle lane 1 result") ||
        !Expect(state.vgprs[1][2] == 0xdeadbeefu,
                "expected ds swizzle inactive lane preservation") ||
        !Expect(state.vgprs[1][3] == 0u,
                "expected ds swizzle inactive-source zero result") ||
        !Expect(state.vgprs[1][32] == 330u,
                "expected ds swizzle lane 32 result") ||
        !Expect(state.vgprs[1][33] == 320u,
                "expected ds swizzle lane 33 result")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_swizzle_state = make_ds_swizzle_state();
  if (!Expect(interpreter.ExecuteProgram(ds_swizzle_program,
                                         &decoded_ds_swizzle_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_swizzle_state(decoded_ds_swizzle_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_swizzle_program;
  if (!Expect(interpreter.CompileProgram(ds_swizzle_program,
                                         &compiled_ds_swizzle_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_swizzle_state = make_ds_swizzle_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_swizzle_program,
                                         &compiled_ds_swizzle_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_swizzle_state(compiled_ds_swizzle_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_permute_program = {
      DecodedInstruction::FourOperand("DS_BPERMUTE_B32",
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_PERMUTE_B32",
                                      InstructionOperand::Vgpr(6),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Vgpr(5),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_permute_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0x0fULL;
    state.vgprs[0][0] = 8u;
    state.vgprs[0][1] = 0u;
    state.vgprs[0][2] = 12u;
    state.vgprs[0][3] = 4u;
    state.vgprs[1][0] = 101u;
    state.vgprs[1][1] = 202u;
    state.vgprs[1][2] = 303u;
    state.vgprs[1][3] = 404u;
    state.vgprs[4][0] = 4u;
    state.vgprs[4][1] = 0u;
    state.vgprs[4][2] = 12u;
    state.vgprs[4][3] = 0u;
    state.vgprs[5][0] = 1001u;
    state.vgprs[5][1] = 1002u;
    state.vgprs[5][2] = 1003u;
    state.vgprs[5][3] = 1004u;
    state.vgprs[2][4] = 0xdeadbeefu;
    state.vgprs[6][4] = 0xcafebabeu;
    return state;
  };
  auto validate_ds_permute_state = [&](const WaveExecutionState& state,
                                       const char* mode) {
    if (!Expect(state.halted, "expected ds permute program to halt") ||
        !Expect(state.vgprs[2][0] == 303u,
                "expected ds_bpermute lane 0 result") ||
        !Expect(state.vgprs[2][1] == 101u,
                "expected ds_bpermute lane 1 result") ||
        !Expect(state.vgprs[2][2] == 404u,
                "expected ds_bpermute lane 2 result") ||
        !Expect(state.vgprs[2][3] == 202u,
                "expected ds_bpermute lane 3 result") ||
        !Expect(state.vgprs[2][4] == 0xdeadbeefu,
                "expected ds_bpermute inactive lane preservation") ||
        !Expect(state.vgprs[6][0] == 1004u,
                "expected ds_permute lane 0 result") ||
        !Expect(state.vgprs[6][1] == 1001u,
                "expected ds_permute lane 1 result") ||
        !Expect(state.vgprs[6][2] == 0u,
                "expected ds_permute lane 2 zero result") ||
        !Expect(state.vgprs[6][3] == 1003u,
                "expected ds_permute lane 3 result") ||
        !Expect(state.vgprs[6][4] == 0xcafebabeu,
                "expected ds_permute inactive lane preservation")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_permute_state = make_ds_permute_state();
  if (!Expect(interpreter.ExecuteProgram(ds_permute_program,
                                         &decoded_ds_permute_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_permute_state(decoded_ds_permute_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_permute_program;
  if (!Expect(interpreter.CompileProgram(ds_permute_program,
                                         &compiled_ds_permute_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_permute_state = make_ds_permute_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_permute_program,
                                         &compiled_ds_permute_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_permute_state(compiled_ds_permute_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_addtid_program = {
      DecodedInstruction::TwoOperand("DS_WRITE_ADDTID_B32",
                                     InstructionOperand::Vgpr(1),
                                     InstructionOperand::Imm32(16)),
      DecodedInstruction::TwoOperand("DS_READ_ADDTID_B32",
                                     InstructionOperand::Vgpr(2),
                                     InstructionOperand::Imm32(16)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_addtid_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.sgprs[124] = 0x00010020u;
    state.vgprs[1][0] = 111u;
    state.vgprs[1][1] = 222u;
    state.vgprs[1][3] = 444u;
    state.vgprs[2][2] = 0xdeadbeefu;
    return state;
  };
  auto validate_ds_addtid_state = [&](const WaveExecutionState& state,
                                      const char* mode) {
    if (!Expect(state.halted, "expected ds addtid program to halt") ||
        !Expect(state.vgprs[2][0] == 111u,
                "expected ds read_addtid lane 0 result") ||
        !Expect(state.vgprs[2][1] == 222u,
                "expected ds read_addtid lane 1 result") ||
        !Expect(state.vgprs[2][2] == 0xdeadbeefu,
                "expected ds read_addtid inactive lane result") ||
        !Expect(state.vgprs[2][3] == 444u,
                "expected ds read_addtid lane 3 result")) {
      std::cerr << mode << '\n';
      return false;
    }

    const auto expect_lds_value = [&](std::size_t address,
                                      std::uint32_t expected,
                                      const char* label) {
      std::uint32_t value = 0;
      std::memcpy(&value, state.lds_bytes.data() + address, sizeof(value));
      return Expect(value == expected, label);
    };
    if (!expect_lds_value(48u, 111u, "expected ds write_addtid lane 0 lds") ||
        !expect_lds_value(52u, 222u, "expected ds write_addtid lane 1 lds") ||
        !expect_lds_value(60u, 444u, "expected ds write_addtid lane 3 lds")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_addtid_state = make_ds_addtid_state();
  if (!Expect(interpreter.ExecuteProgram(ds_addtid_program,
                                         &decoded_ds_addtid_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_addtid_state(decoded_ds_addtid_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_addtid_program;
  if (!Expect(interpreter.CompileProgram(ds_addtid_program,
                                         &compiled_ds_addtid_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_addtid_state = make_ds_addtid_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_addtid_program,
                                         &compiled_ds_addtid_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_addtid_state(compiled_ds_addtid_state, "compiled")) {
    return 1;
  }
  }

  if (!RunDsWaveCounterTests(interpreter)) {
    return 1;
  }

  if (!RunDsReturnTests(interpreter)) {
    return 1;
  }

  if (!RunDsDualDataTests(interpreter)) {
    return 1;
  }

  {
  const std::vector<DecodedInstruction> ds_access_program = {
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_I8", InstructionOperand::Vgpr(10),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(3)),
      DecodedInstruction::ThreeOperand("DS_READ_U8", InstructionOperand::Vgpr(11),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(2)),
      DecodedInstruction::ThreeOperand("DS_READ_I16", InstructionOperand::Vgpr(12),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(2)),
      DecodedInstruction::ThreeOperand("DS_READ_U16", InstructionOperand::Vgpr(13),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("DS_WRITE2_B32", InstructionOperand::Vgpr(2),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Imm32(1),
                                      InstructionOperand::Imm32(3)),
      DecodedInstruction::FourOperand("DS_READ2_B32", InstructionOperand::Vgpr(14),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Imm32(1),
                                      InstructionOperand::Imm32(3)),
      DecodedInstruction::FiveOperand(
          "DS_WRITE2ST64_B32", InstructionOperand::Vgpr(5),
          InstructionOperand::Vgpr(6), InstructionOperand::Vgpr(7),
          InstructionOperand::Imm32(1), InstructionOperand::Imm32(2)),
      DecodedInstruction::FourOperand(
          "DS_READ2ST64_B32", InstructionOperand::Vgpr(16),
          InstructionOperand::Vgpr(5), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(2)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B16", InstructionOperand::Vgpr(18),
                                       InstructionOperand::Vgpr(19),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand(
          "DS_WRITE_B8_D16_HI", InstructionOperand::Vgpr(18),
          InstructionOperand::Vgpr(20), InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_U8_D16", InstructionOperand::Vgpr(21),
                                       InstructionOperand::Vgpr(18),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand(
          "DS_READ_U8_D16_HI", InstructionOperand::Vgpr(22),
          InstructionOperand::Vgpr(18), InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_I8_D16", InstructionOperand::Vgpr(23),
                                       InstructionOperand::Vgpr(18),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand(
          "DS_READ_I8_D16_HI", InstructionOperand::Vgpr(24),
          InstructionOperand::Vgpr(18), InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(25),
                                       InstructionOperand::Vgpr(26),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand(
          "DS_WRITE_B16_D16_HI", InstructionOperand::Vgpr(25),
          InstructionOperand::Vgpr(27), InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_U16_D16", InstructionOperand::Vgpr(28),
                                       InstructionOperand::Vgpr(25),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand(
          "DS_READ_U16_D16_HI", InstructionOperand::Vgpr(29),
          InstructionOperand::Vgpr(25), InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(30),
                                       InstructionOperand::Vgpr(25),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_access_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[0][0] = 0u;
    state.vgprs[0][1] = 4u;
    state.vgprs[0][3] = 8u;
    state.vgprs[1][0] = 0x80ff1122u;
    state.vgprs[1][1] = 0x7f008877u;
    state.vgprs[1][3] = 0x1234fedcu;
    state.vgprs[2][0] = 32u;
    state.vgprs[2][1] = 64u;
    state.vgprs[2][3] = 96u;
    state.vgprs[3][0] = 0xaaaaaaaau;
    state.vgprs[3][1] = 0x11111111u;
    state.vgprs[3][3] = 0xfeedfaceu;
    state.vgprs[4][0] = 0xbbbbbbbbu;
    state.vgprs[4][1] = 0x22222222u;
    state.vgprs[4][3] = 0xcafebeefu;
    state.vgprs[5][0] = 128u;
    state.vgprs[5][1] = 160u;
    state.vgprs[5][3] = 192u;
    state.vgprs[6][0] = 0x01020304u;
    state.vgprs[6][1] = 0x05060708u;
    state.vgprs[6][3] = 0x090a0b0cu;
    state.vgprs[7][0] = 0xa0b0c0d0u;
    state.vgprs[7][1] = 0x0badf00du;
    state.vgprs[7][3] = 0x13579bdfu;
    state.vgprs[18][0] = 224u;
    state.vgprs[18][1] = 228u;
    state.vgprs[18][3] = 232u;
    state.vgprs[19][0] = 0x1234u;
    state.vgprs[19][1] = 0x8856u;
    state.vgprs[19][3] = 0x00cdu;
    state.vgprs[20][0] = 0x00000180u;
    state.vgprs[20][1] = 0x1234007fu;
    state.vgprs[20][3] = 0xffff00ffu;
    state.vgprs[25][0] = 256u;
    state.vgprs[25][1] = 260u;
    state.vgprs[25][3] = 264u;
    state.vgprs[26][0] = 0x11223344u;
    state.vgprs[26][1] = 0xaabbccddu;
    state.vgprs[26][3] = 0x01020304u;
    state.vgprs[27][0] = 0xaaaa5566u;
    state.vgprs[27][1] = 0xbbbb1234u;
    state.vgprs[27][3] = 0xccccabcdu;
    for (std::uint16_t vgpr = 10; vgpr <= 30; ++vgpr) {
      state.vgprs[vgpr][2] = (vgpr & 1u) == 0u ? 0xdeadbeefu : 0xcafebabeu;
    }
    return state;
  };
  auto validate_ds_access_state = [&](const WaveExecutionState& state,
                                      const char* mode) {
    if (!Expect(state.halted, "expected ds access program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }

    if (!Expect(state.vgprs[10][0] == 0xffffff80u,
                "expected ds_read_i8 lane 0 result") ||
        !Expect(state.vgprs[10][1] == 0x7fu,
                "expected ds_read_i8 lane 1 result") ||
        !Expect(state.vgprs[10][3] == 0x12u,
                "expected ds_read_i8 lane 3 result") ||
        !Expect(state.vgprs[11][0] == 0xffu,
                "expected ds_read_u8 lane 0 result") ||
        !Expect(state.vgprs[11][1] == 0x0u,
                "expected ds_read_u8 lane 1 result") ||
        !Expect(state.vgprs[11][3] == 0x34u,
                "expected ds_read_u8 lane 3 result") ||
        !Expect(state.vgprs[12][0] == 0xffff80ffu,
                "expected ds_read_i16 lane 0 result") ||
        !Expect(state.vgprs[12][1] == 0x7f00u,
                "expected ds_read_i16 lane 1 result") ||
        !Expect(state.vgprs[12][3] == 0x1234u,
                "expected ds_read_i16 lane 3 result") ||
        !Expect(state.vgprs[13][0] == 0x1122u,
                "expected ds_read_u16 lane 0 result") ||
        !Expect(state.vgprs[13][1] == 0x8877u,
                "expected ds_read_u16 lane 1 result") ||
        !Expect(state.vgprs[13][3] == 0xfedcu,
                "expected ds_read_u16 lane 3 result") ||
        !Expect(state.vgprs[14][0] == 0xaaaaaaaau,
                "expected ds_read2 low lane 0 result") ||
        !Expect(state.vgprs[14][1] == 0x11111111u,
                "expected ds_read2 low lane 1 result") ||
        !Expect(state.vgprs[14][3] == 0xfeedfaceu,
                "expected ds_read2 low lane 3 result") ||
        !Expect(state.vgprs[15][0] == 0xbbbbbbbbu,
                "expected ds_read2 high lane 0 result") ||
        !Expect(state.vgprs[15][1] == 0x22222222u,
                "expected ds_read2 high lane 1 result") ||
        !Expect(state.vgprs[15][3] == 0xcafebeefu,
                "expected ds_read2 high lane 3 result") ||
        !Expect(state.vgprs[16][0] == 0x01020304u,
                "expected ds_read2st64 low lane 0 result") ||
        !Expect(state.vgprs[16][1] == 0x05060708u,
                "expected ds_read2st64 low lane 1 result") ||
        !Expect(state.vgprs[16][3] == 0x090a0b0cu,
                "expected ds_read2st64 low lane 3 result") ||
        !Expect(state.vgprs[17][0] == 0xa0b0c0d0u,
                "expected ds_read2st64 high lane 0 result") ||
        !Expect(state.vgprs[17][1] == 0x0badf00du,
                "expected ds_read2st64 high lane 1 result") ||
        !Expect(state.vgprs[17][3] == 0x13579bdfu,
                "expected ds_read2st64 high lane 3 result") ||
        !Expect(state.vgprs[21][0] == 0x34u,
                "expected ds_read_u8_d16 lane 0 result") ||
        !Expect(state.vgprs[21][1] == 0x56u,
                "expected ds_read_u8_d16 lane 1 result") ||
        !Expect(state.vgprs[21][3] == 0xcdu,
                "expected ds_read_u8_d16 lane 3 result") ||
        !Expect(state.vgprs[22][0] == 0x80u,
                "expected ds_read_u8_d16_hi lane 0 result") ||
        !Expect(state.vgprs[22][1] == 0x7fu,
                "expected ds_read_u8_d16_hi lane 1 result") ||
        !Expect(state.vgprs[22][3] == 0xffu,
                "expected ds_read_u8_d16_hi lane 3 result") ||
        !Expect(state.vgprs[23][0] == 0x34u,
                "expected ds_read_i8_d16 lane 0 result") ||
        !Expect(state.vgprs[23][1] == 0x56u,
                "expected ds_read_i8_d16 lane 1 result") ||
        !Expect(state.vgprs[23][3] == 0xffffffcdu,
                "expected ds_read_i8_d16 lane 3 result") ||
        !Expect(state.vgprs[24][0] == 0xffffff80u,
                "expected ds_read_i8_d16_hi lane 0 result") ||
        !Expect(state.vgprs[24][1] == 0x7fu,
                "expected ds_read_i8_d16_hi lane 1 result") ||
        !Expect(state.vgprs[24][3] == 0xffffffffu,
                "expected ds_read_i8_d16_hi lane 3 result") ||
        !Expect(state.vgprs[28][0] == 0x3344u,
                "expected ds_read_u16_d16 lane 0 result") ||
        !Expect(state.vgprs[28][1] == 0xccddu,
                "expected ds_read_u16_d16 lane 1 result") ||
        !Expect(state.vgprs[28][3] == 0x0304u,
                "expected ds_read_u16_d16 lane 3 result") ||
        !Expect(state.vgprs[29][0] == 0x5566u,
                "expected ds_read_u16_d16_hi lane 0 result") ||
        !Expect(state.vgprs[29][1] == 0x1234u,
                "expected ds_read_u16_d16_hi lane 1 result") ||
        !Expect(state.vgprs[29][3] == 0xabcdu,
                "expected ds_read_u16_d16_hi lane 3 result") ||
        !Expect(state.vgprs[30][0] == 0x55663344u,
                "expected ds d16 full lane 0 result") ||
        !Expect(state.vgprs[30][1] == 0x1234ccddu,
                "expected ds d16 full lane 1 result") ||
        !Expect(state.vgprs[30][3] == 0xabcd0304u,
                "expected ds d16 full lane 3 result")) {
      std::cerr << mode << '\n';
      return false;
    }

    if (!Expect(state.vgprs[10][2] == 0xdeadbeefu,
                "expected inactive ds_read_i8 preservation") ||
        !Expect(state.vgprs[11][2] == 0xcafebabeu,
                "expected inactive ds_read_u8 preservation") ||
        !Expect(state.vgprs[12][2] == 0xdeadbeefu,
                "expected inactive ds_read_i16 preservation") ||
        !Expect(state.vgprs[13][2] == 0xcafebabeu,
                "expected inactive ds_read_u16 preservation") ||
        !Expect(state.vgprs[14][2] == 0xdeadbeefu,
                "expected inactive ds_read2 low preservation") ||
        !Expect(state.vgprs[15][2] == 0xcafebabeu,
                "expected inactive ds_read2 high preservation") ||
        !Expect(state.vgprs[16][2] == 0xdeadbeefu,
                "expected inactive ds_read2st64 low preservation") ||
        !Expect(state.vgprs[17][2] == 0xcafebabeu,
                "expected inactive ds_read2st64 high preservation") ||
        !Expect(state.vgprs[21][2] == 0xcafebabeu,
                "expected inactive ds_read_u8_d16 preservation") ||
        !Expect(state.vgprs[22][2] == 0xdeadbeefu,
                "expected inactive ds_read_u8_d16_hi preservation") ||
        !Expect(state.vgprs[23][2] == 0xcafebabeu,
                "expected inactive ds_read_i8_d16 preservation") ||
        !Expect(state.vgprs[24][2] == 0xdeadbeefu,
                "expected inactive ds_read_i8_d16_hi preservation") ||
        !Expect(state.vgprs[28][2] == 0xdeadbeefu,
                "expected inactive ds_read_u16_d16 preservation") ||
        !Expect(state.vgprs[29][2] == 0xcafebabeu,
                "expected inactive ds_read_u16_d16_hi preservation") ||
        !Expect(state.vgprs[30][2] == 0xdeadbeefu,
                "expected inactive ds d16 full preservation")) {
      std::cerr << mode << '\n';
      return false;
    }

    std::uint32_t value = 0;
    if (!Expect((std::memcpy(&value, state.lds_bytes.data() + 36, sizeof(value)), value) ==
                    0xaaaaaaaau,
                "expected ds_write2 lane 0 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 44, sizeof(value)), value) ==
                    0xbbbbbbbbu,
                "expected ds_write2 lane 0 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 68, sizeof(value)), value) ==
                    0x11111111u,
                "expected ds_write2 lane 1 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 76, sizeof(value)), value) ==
                    0x22222222u,
                "expected ds_write2 lane 1 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 100, sizeof(value)), value) ==
                    0xfeedfaceu,
                "expected ds_write2 lane 3 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 108, sizeof(value)), value) ==
                    0xcafebeefu,
                "expected ds_write2 lane 3 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 384, sizeof(value)), value) ==
                    0x01020304u,
                "expected ds_write2st64 lane 0 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 640, sizeof(value)), value) ==
                    0xa0b0c0d0u,
                "expected ds_write2st64 lane 0 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 416, sizeof(value)), value) ==
                    0x05060708u,
                "expected ds_write2st64 lane 1 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 672, sizeof(value)), value) ==
                    0x0badf00du,
                "expected ds_write2st64 lane 1 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 448, sizeof(value)), value) ==
                    0x090a0b0cu,
                "expected ds_write2st64 lane 3 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 704, sizeof(value)), value) ==
                    0x13579bdfu,
                "expected ds_write2st64 lane 3 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 224, sizeof(std::uint16_t)), value & 0xffffu) ==
                    0x8034u,
                "expected ds_write_b8_d16_hi lane 0 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 228, sizeof(std::uint16_t)), value & 0xffffu) ==
                    0x7f56u,
                "expected ds_write_b8_d16_hi lane 1 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 232, sizeof(std::uint16_t)), value & 0xffffu) ==
                    0xffcdu,
                "expected ds_write_b8_d16_hi lane 3 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 256, sizeof(value)), value) ==
                    0x55663344u,
                "expected ds_write_b16_d16_hi lane 0 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 260, sizeof(value)), value) ==
                    0x1234ccddu,
                "expected ds_write_b16_d16_hi lane 1 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 264, sizeof(value)), value) ==
                    0xabcd0304u,
                "expected ds_write_b16_d16_hi lane 3 store")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_access_state = make_ds_access_state();
  if (!Expect(interpreter.ExecuteProgram(ds_access_program, &decoded_ds_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_access_state(decoded_ds_access_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_access_program;
  if (!Expect(interpreter.CompileProgram(ds_access_program,
                                         &compiled_ds_access_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_access_state = make_ds_access_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_access_program,
                                         &compiled_ds_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_access_state(compiled_ds_access_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_b64_access_program = {
      DecodedInstruction::ThreeOperand("DS_WRITE_B64", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_READ_B64", InstructionOperand::Vgpr(20),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("DS_WRITE2_B64", InstructionOperand::Vgpr(3),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Vgpr(6),
                                      InstructionOperand::Imm32(1),
                                      InstructionOperand::Imm32(3)),
      DecodedInstruction::FourOperand("DS_READ2_B64", InstructionOperand::Vgpr(22),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Imm32(1),
                                      InstructionOperand::Imm32(3)),
      DecodedInstruction::FiveOperand(
          "DS_WRITE2ST64_B64", InstructionOperand::Vgpr(8),
          InstructionOperand::Vgpr(9), InstructionOperand::Vgpr(11),
          InstructionOperand::Imm32(1), InstructionOperand::Imm32(2)),
      DecodedInstruction::FourOperand(
          "DS_READ2ST64_B64", InstructionOperand::Vgpr(26),
          InstructionOperand::Vgpr(8), InstructionOperand::Imm32(1),
          InstructionOperand::Imm32(2)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_b64_access_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[0][0] = 0u;
    state.vgprs[0][1] = 8u;
    state.vgprs[0][3] = 16u;
    state.vgprs[1][0] = 0x11111111u;
    state.vgprs[2][0] = 0xaaaaaaaau;
    state.vgprs[1][1] = 0x22222222u;
    state.vgprs[2][1] = 0xbbbbbbbbu;
    state.vgprs[1][3] = 0x33333333u;
    state.vgprs[2][3] = 0xccccccccu;

    state.vgprs[3][0] = 32u;
    state.vgprs[3][1] = 64u;
    state.vgprs[3][3] = 96u;
    state.vgprs[4][0] = 0x44444444u;
    state.vgprs[5][0] = 0xddddddddu;
    state.vgprs[6][0] = 0x55555555u;
    state.vgprs[7][0] = 0xeeeeeeeeu;
    state.vgprs[4][1] = 0x66666666u;
    state.vgprs[5][1] = 0xf0f0f0f0u;
    state.vgprs[6][1] = 0x77777777u;
    state.vgprs[7][1] = 0x12345678u;
    state.vgprs[4][3] = 0x88888888u;
    state.vgprs[5][3] = 0x9abcdef0u;
    state.vgprs[6][3] = 0x99999999u;
    state.vgprs[7][3] = 0x0fedcba9u;

    state.vgprs[8][0] = 128u;
    state.vgprs[8][1] = 160u;
    state.vgprs[8][3] = 192u;
    state.vgprs[9][0] = 0x01010101u;
    state.vgprs[10][0] = 0x11111111u;
    state.vgprs[11][0] = 0x02020202u;
    state.vgprs[12][0] = 0x22222222u;
    state.vgprs[9][1] = 0x03030303u;
    state.vgprs[10][1] = 0x33333333u;
    state.vgprs[11][1] = 0x04040404u;
    state.vgprs[12][1] = 0x44444444u;
    state.vgprs[9][3] = 0x05050505u;
    state.vgprs[10][3] = 0x55555555u;
    state.vgprs[11][3] = 0x06060606u;
    state.vgprs[12][3] = 0x66666666u;

    for (std::uint16_t vgpr = 20; vgpr <= 29; ++vgpr) {
      state.vgprs[vgpr][2] = 0x90000000u + vgpr;
    }
    return state;
  };
  auto validate_ds_b64_access_state = [&](const WaveExecutionState& state,
                                          const char* mode) {
    if (!Expect(state.halted, "expected ds b64 access program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }

    if (!Expect(ComposeU64(state.vgprs[20][0], state.vgprs[21][0]) ==
                    0xaaaaaaaa11111111ULL,
                "expected ds_read_b64 lane 0 result") ||
        !Expect(ComposeU64(state.vgprs[20][1], state.vgprs[21][1]) ==
                    0xbbbbbbbb22222222ULL,
                "expected ds_read_b64 lane 1 result") ||
        !Expect(ComposeU64(state.vgprs[20][3], state.vgprs[21][3]) ==
                    0xcccccccc33333333ULL,
                "expected ds_read_b64 lane 3 result") ||
        !Expect(ComposeU64(state.vgprs[22][0], state.vgprs[23][0]) ==
                    0xdddddddd44444444ULL,
                "expected ds_read2_b64 low lane 0 result") ||
        !Expect(ComposeU64(state.vgprs[24][0], state.vgprs[25][0]) ==
                    0xeeeeeeee55555555ULL,
                "expected ds_read2_b64 high lane 0 result") ||
        !Expect(ComposeU64(state.vgprs[22][1], state.vgprs[23][1]) ==
                    0xf0f0f0f066666666ULL,
                "expected ds_read2_b64 low lane 1 result") ||
        !Expect(ComposeU64(state.vgprs[24][1], state.vgprs[25][1]) ==
                    0x1234567877777777ULL,
                "expected ds_read2_b64 high lane 1 result") ||
        !Expect(ComposeU64(state.vgprs[22][3], state.vgprs[23][3]) ==
                    0x9abcdef088888888ULL,
                "expected ds_read2_b64 low lane 3 result") ||
        !Expect(ComposeU64(state.vgprs[24][3], state.vgprs[25][3]) ==
                    0x0fedcba999999999ULL,
                "expected ds_read2_b64 high lane 3 result") ||
        !Expect(ComposeU64(state.vgprs[26][0], state.vgprs[27][0]) ==
                    0x1111111101010101ULL,
                "expected ds_read2st64_b64 low lane 0 result") ||
        !Expect(ComposeU64(state.vgprs[28][0], state.vgprs[29][0]) ==
                    0x2222222202020202ULL,
                "expected ds_read2st64_b64 high lane 0 result") ||
        !Expect(ComposeU64(state.vgprs[26][1], state.vgprs[27][1]) ==
                    0x3333333303030303ULL,
                "expected ds_read2st64_b64 low lane 1 result") ||
        !Expect(ComposeU64(state.vgprs[28][1], state.vgprs[29][1]) ==
                    0x4444444404040404ULL,
                "expected ds_read2st64_b64 high lane 1 result") ||
        !Expect(ComposeU64(state.vgprs[26][3], state.vgprs[27][3]) ==
                    0x5555555505050505ULL,
                "expected ds_read2st64_b64 low lane 3 result") ||
        !Expect(ComposeU64(state.vgprs[28][3], state.vgprs[29][3]) ==
                    0x6666666606060606ULL,
                "expected ds_read2st64_b64 high lane 3 result")) {
      std::cerr << mode << '\n';
      return false;
    }

    for (std::uint16_t vgpr = 20; vgpr <= 29; ++vgpr) {
      const std::uint32_t expected = 0x90000000u + vgpr;
      if (!Expect(state.vgprs[vgpr][2] == expected,
                  "expected inactive ds b64 destination preservation")) {
        std::cerr << mode << " vgpr=" << vgpr << '\n';
        return false;
      }
    }

    std::uint64_t value = 0;
    if (!Expect((std::memcpy(&value, state.lds_bytes.data() + 0, sizeof(value)), value) ==
                    0xaaaaaaaa11111111ULL,
                "expected ds_write_b64 lane 0 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 8, sizeof(value)), value) ==
                    0xbbbbbbbb22222222ULL,
                "expected ds_write_b64 lane 1 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 16, sizeof(value)), value) ==
                    0xcccccccc33333333ULL,
                "expected ds_write_b64 lane 3 store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 40, sizeof(value)), value) ==
                    0xdddddddd44444444ULL,
                "expected ds_write2_b64 lane 0 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 56, sizeof(value)), value) ==
                    0xeeeeeeee55555555ULL,
                "expected ds_write2_b64 lane 0 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 72, sizeof(value)), value) ==
                    0xf0f0f0f066666666ULL,
                "expected ds_write2_b64 lane 1 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 88, sizeof(value)), value) ==
                    0x1234567877777777ULL,
                "expected ds_write2_b64 lane 1 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 104, sizeof(value)), value) ==
                    0x9abcdef088888888ULL,
                "expected ds_write2_b64 lane 3 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 120, sizeof(value)), value) ==
                    0x0fedcba999999999ULL,
                "expected ds_write2_b64 lane 3 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 640, sizeof(value)), value) ==
                    0x1111111101010101ULL,
                "expected ds_write2st64_b64 lane 0 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 1152, sizeof(value)), value) ==
                    0x2222222202020202ULL,
                "expected ds_write2st64_b64 lane 0 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 672, sizeof(value)), value) ==
                    0x3333333303030303ULL,
                "expected ds_write2st64_b64 lane 1 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 1184, sizeof(value)), value) ==
                    0x4444444404040404ULL,
                "expected ds_write2st64_b64 lane 1 high store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 704, sizeof(value)), value) ==
                    0x5555555505050505ULL,
                "expected ds_write2st64_b64 lane 3 low store") ||
        !Expect((std::memcpy(&value, state.lds_bytes.data() + 1216, sizeof(value)), value) ==
                    0x6666666606060606ULL,
                "expected ds_write2st64_b64 lane 3 high store")) {
      std::cerr << mode << '\n';
      return false;
    }
    return true;
  };

  WaveExecutionState decoded_ds_b64_access_state = make_ds_b64_access_state();
  if (!Expect(interpreter.ExecuteProgram(ds_b64_access_program,
                                         &decoded_ds_b64_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_access_state(decoded_ds_b64_access_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_b64_access_program;
  if (!Expect(interpreter.CompileProgram(ds_b64_access_program,
                                         &compiled_ds_b64_access_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_b64_access_state = make_ds_b64_access_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_b64_access_program,
                                         &compiled_ds_b64_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_access_state(compiled_ds_b64_access_state, "compiled")) {
    return 1;
  }
  }

  if (!RunDsPairReturnProgramTest(interpreter)) {
    return 1;
  }

  {
  const std::vector<DecodedInstruction> ds_multi_dword_access_program = {
      DecodedInstruction::ThreeOperand("DS_WRITE_B96", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::ThreeOperand("DS_READ_B96", InstructionOperand::Vgpr(10),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::ThreeOperand("DS_WRITE_B128", InstructionOperand::Vgpr(4),
                                       InstructionOperand::Vgpr(5),
                                       InstructionOperand::Imm32(8)),
      DecodedInstruction::ThreeOperand("DS_READ_B128", InstructionOperand::Vgpr(20),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(8)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_multi_dword_access_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    state.vgprs[0][0] = 0u;
    state.vgprs[0][1] = 16u;
    state.vgprs[0][3] = 32u;
    state.vgprs[1][0] = 0x11111111u;
    state.vgprs[1][1] = 0x44444444u;
    state.vgprs[1][3] = 0x77777777u;
    state.vgprs[2][0] = 0x22222222u;
    state.vgprs[2][1] = 0x55555555u;
    state.vgprs[2][3] = 0x88888888u;
    state.vgprs[3][0] = 0x33333333u;
    state.vgprs[3][1] = 0x66666666u;
    state.vgprs[3][3] = 0x99999999u;
    state.vgprs[4][0] = 64u;
    state.vgprs[4][1] = 96u;
    state.vgprs[4][3] = 128u;
    state.vgprs[5][0] = 0x0a0b0c0du;
    state.vgprs[5][1] = 0x10203040u;
    state.vgprs[5][3] = 0x55667788u;
    state.vgprs[6][0] = 0x1a1b1c1du;
    state.vgprs[6][1] = 0x20304050u;
    state.vgprs[6][3] = 0x66778899u;
    state.vgprs[7][0] = 0x2a2b2c2du;
    state.vgprs[7][1] = 0x30405060u;
    state.vgprs[7][3] = 0x778899aau;
    state.vgprs[8][0] = 0x3a3b3c3du;
    state.vgprs[8][1] = 0x40506070u;
    state.vgprs[8][3] = 0x8899aabbu;
    for (std::uint16_t vgpr = 10; vgpr <= 23; ++vgpr) {
      state.vgprs[vgpr][2] = (vgpr & 1u) == 0u ? 0xdeadbeefu : 0xcafebabeu;
    }
    return state;
  };
  auto validate_ds_multi_dword_access_state =
      [&](const WaveExecutionState& state, const char* mode) {
        if (!Expect(state.halted,
                    "expected ds multi-dword access program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }
        const std::array<std::size_t, 3> lanes = {0u, 1u, 3u};
        const std::array<std::array<std::uint32_t, 3>, 3> b96_expected = {{
            {{0x11111111u, 0x22222222u, 0x33333333u}},
            {{0x44444444u, 0x55555555u, 0x66666666u}},
            {{0x77777777u, 0x88888888u, 0x99999999u}},
        }};
        const std::array<std::array<std::uint32_t, 4>, 3> b128_expected = {{
            {{0x0a0b0c0du, 0x1a1b1c1du, 0x2a2b2c2du, 0x3a3b3c3du}},
            {{0x10203040u, 0x20304050u, 0x30405060u, 0x40506070u}},
            {{0x55667788u, 0x66778899u, 0x778899aau, 0x8899aabbu}},
        }};
        for (std::size_t lane_index = 0; lane_index < lanes.size();
             ++lane_index) {
          const std::size_t lane = lanes[lane_index];
          for (std::size_t dword_index = 0; dword_index < 3; ++dword_index) {
            if (!Expect(state.vgprs[static_cast<std::uint16_t>(10 + dword_index)]
                                       [lane] ==
                           b96_expected[lane_index][dword_index],
                       "expected ds_read_b96 result")) {
              std::cerr << mode << " lane=" << lane
                        << " dword=" << dword_index << '\n';
              return false;
            }
          }
          for (std::size_t dword_index = 0; dword_index < 4; ++dword_index) {
            if (!Expect(state.vgprs[static_cast<std::uint16_t>(20 + dword_index)]
                                       [lane] ==
                           b128_expected[lane_index][dword_index],
                       "expected ds_read_b128 result")) {
              std::cerr << mode << " lane=" << lane
                        << " dword=" << dword_index << '\n';
              return false;
            }
          }
        }
        for (std::uint16_t vgpr = 10; vgpr <= 23; ++vgpr) {
          const std::uint32_t expected = (vgpr & 1u) == 0u ? 0xdeadbeefu
                                                           : 0xcafebabeu;
          if (!Expect(state.vgprs[vgpr][2] == expected,
                      "expected inactive ds multi-dword destination preservation")) {
            std::cerr << mode << " vgpr=" << vgpr << '\n';
            return false;
          }
        }
        const auto expect_lds_value = [&](std::uint64_t address,
                                          std::uint32_t expected,
                                          const char* label) {
          std::uint32_t value = 0;
          std::memcpy(&value, state.lds_bytes.data() + address, sizeof(value));
          return Expect(value == expected, label);
        };
        if (!expect_lds_value(4u, 0x11111111u, "expected ds_write_b96 lane 0 dword 0") ||
            !expect_lds_value(8u, 0x22222222u, "expected ds_write_b96 lane 0 dword 1") ||
            !expect_lds_value(12u, 0x33333333u, "expected ds_write_b96 lane 0 dword 2") ||
            !expect_lds_value(20u, 0x44444444u, "expected ds_write_b96 lane 1 dword 0") ||
            !expect_lds_value(24u, 0x55555555u, "expected ds_write_b96 lane 1 dword 1") ||
            !expect_lds_value(28u, 0x66666666u, "expected ds_write_b96 lane 1 dword 2") ||
            !expect_lds_value(36u, 0x77777777u, "expected ds_write_b96 lane 3 dword 0") ||
            !expect_lds_value(40u, 0x88888888u, "expected ds_write_b96 lane 3 dword 1") ||
            !expect_lds_value(44u, 0x99999999u, "expected ds_write_b96 lane 3 dword 2") ||
            !expect_lds_value(72u, 0x0a0b0c0du, "expected ds_write_b128 lane 0 dword 0") ||
            !expect_lds_value(76u, 0x1a1b1c1du, "expected ds_write_b128 lane 0 dword 1") ||
            !expect_lds_value(80u, 0x2a2b2c2du, "expected ds_write_b128 lane 0 dword 2") ||
            !expect_lds_value(84u, 0x3a3b3c3du, "expected ds_write_b128 lane 0 dword 3") ||
            !expect_lds_value(104u, 0x10203040u, "expected ds_write_b128 lane 1 dword 0") ||
            !expect_lds_value(108u, 0x20304050u, "expected ds_write_b128 lane 1 dword 1") ||
            !expect_lds_value(112u, 0x30405060u, "expected ds_write_b128 lane 1 dword 2") ||
            !expect_lds_value(116u, 0x40506070u, "expected ds_write_b128 lane 1 dword 3") ||
            !expect_lds_value(136u, 0x55667788u, "expected ds_write_b128 lane 3 dword 0") ||
            !expect_lds_value(140u, 0x66778899u, "expected ds_write_b128 lane 3 dword 1") ||
            !expect_lds_value(144u, 0x778899aau, "expected ds_write_b128 lane 3 dword 2") ||
            !expect_lds_value(148u, 0x8899aabbu, "expected ds_write_b128 lane 3 dword 3")) {
          std::cerr << mode << '\n';
          return false;
        }
        return true;
      };

  WaveExecutionState decoded_ds_multi_dword_access_state =
      make_ds_multi_dword_access_state();
  if (!Expect(interpreter.ExecuteProgram(ds_multi_dword_access_program,
                                         &decoded_ds_multi_dword_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_multi_dword_access_state(decoded_ds_multi_dword_access_state,
                                            "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_multi_dword_access_program;
  if (!Expect(interpreter.CompileProgram(ds_multi_dword_access_program,
                                         &compiled_ds_multi_dword_access_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_multi_dword_access_state =
      make_ds_multi_dword_access_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_multi_dword_access_program,
                                         &compiled_ds_multi_dword_access_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_multi_dword_access_state(compiled_ds_multi_dword_access_state,
                                            "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_b64_update_program = {
      DecodedInstruction::ThreeOperand("DS_ADD_U64", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_SUB_U64", InstructionOperand::Vgpr(3),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_RSUB_U64", InstructionOperand::Vgpr(6),
                                       InstructionOperand::Vgpr(7),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_INC_U64", InstructionOperand::Vgpr(9),
                                       InstructionOperand::Vgpr(10),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_DEC_U64", InstructionOperand::Vgpr(12),
                                       InstructionOperand::Vgpr(13),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_I64", InstructionOperand::Vgpr(15),
                                       InstructionOperand::Vgpr(16),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_I64", InstructionOperand::Vgpr(18),
                                       InstructionOperand::Vgpr(19),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_U64", InstructionOperand::Vgpr(21),
                                       InstructionOperand::Vgpr(22),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_U64", InstructionOperand::Vgpr(24),
                                       InstructionOperand::Vgpr(25),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_AND_B64", InstructionOperand::Vgpr(27),
                                       InstructionOperand::Vgpr(28),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_OR_B64", InstructionOperand::Vgpr(30),
                                       InstructionOperand::Vgpr(31),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_XOR_B64", InstructionOperand::Vgpr(33),
                                       InstructionOperand::Vgpr(34),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MSKOR_B64", InstructionOperand::Vgpr(36),
                                      InstructionOperand::Vgpr(37),
                                      InstructionOperand::Vgpr(39),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_CMPST_B64", InstructionOperand::Vgpr(42),
                                      InstructionOperand::Vgpr(43),
                                      InstructionOperand::Vgpr(45),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_CMPST_F64", InstructionOperand::Vgpr(48),
                                      InstructionOperand::Vgpr(49),
                                      InstructionOperand::Vgpr(51),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_ADD_F64", InstructionOperand::Vgpr(54),
                                       InstructionOperand::Vgpr(55),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MIN_F64", InstructionOperand::Vgpr(57),
                                       InstructionOperand::Vgpr(58),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("DS_MAX_F64", InstructionOperand::Vgpr(60),
                                       InstructionOperand::Vgpr(61),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  struct DsLds64Expectation {
    std::uint64_t address;
    std::uint64_t expected;
    const char* label;
  };
  auto make_ds_b64_update_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0x1ULL;
    auto write_lds_u64 = [](WaveExecutionState* wave,
                            std::uint64_t address,
                            std::uint64_t value) {
      std::memcpy(wave->lds_bytes.data() + address, &value, sizeof(value));
    };

    state.vgprs[0][0] = 0u;
    SetLane0VgprU64(&state, 1, 5u);
    state.vgprs[3][0] = 16u;
    SetLane0VgprU64(&state, 4, 6u);
    state.vgprs[6][0] = 32u;
    SetLane0VgprU64(&state, 7, 10u);
    state.vgprs[9][0] = 48u;
    SetLane0VgprU64(&state, 10, 7u);
    state.vgprs[12][0] = 64u;
    SetLane0VgprU64(&state, 13, 11u);
    state.vgprs[15][0] = 80u;
    SetLane0VgprU64(&state, 16, 2u);
    state.vgprs[18][0] = 96u;
    SetLane0VgprU64(&state, 19, 2u);
    state.vgprs[21][0] = 112u;
    SetLane0VgprU64(&state, 22, 4u);
    state.vgprs[24][0] = 128u;
    SetLane0VgprU64(&state, 25, 14u);
    state.vgprs[27][0] = 144u;
    SetLane0VgprU64(&state, 28, 0x0f0f0f0f0f0f0f0fULL);
    state.vgprs[30][0] = 160u;
    SetLane0VgprU64(&state, 31, 0x1100110011001100ULL);
    state.vgprs[33][0] = 176u;
    SetLane0VgprU64(&state, 34, 0x00ff00ff00ff00ffULL);
    state.vgprs[36][0] = 192u;
    SetLane0VgprU64(&state, 37, 0xff00ff0000ff00ffULL);
    SetLane0VgprU64(&state, 39, 0x005500aa550000aaULL);
    state.vgprs[42][0] = 208u;
    SetLane0VgprU64(&state, 43, 0x1122334455667788ULL);
    SetLane0VgprU64(&state, 45, 0x8877665544332211ULL);
    state.vgprs[48][0] = 224u;
    SetLane0VgprU64(&state, 49, DoubleBits(3.5));
    SetLane0VgprU64(&state, 51, DoubleBits(9.0));
    state.vgprs[54][0] = 240u;
    SetLane0VgprU64(&state, 55, DoubleBits(2.25));
    state.vgprs[57][0] = 256u;
    SetLane0VgprU64(&state, 58, DoubleBits(-1.0));
    state.vgprs[60][0] = 272u;
    SetLane0VgprU64(&state, 61, DoubleBits(8.0));

    write_lds_u64(&state, 0u, 10u);
    write_lds_u64(&state, 16u, 20u);
    write_lds_u64(&state, 32u, 4u);
    write_lds_u64(&state, 48u, 7u);
    write_lds_u64(&state, 64u, 0u);
    write_lds_u64(&state, 80u, 0xfffffffffffffffcULL);
    write_lds_u64(&state, 96u, 0xfffffffffffffffdULL);
    write_lds_u64(&state, 112u, 9u);
    write_lds_u64(&state, 128u, 9u);
    write_lds_u64(&state, 144u, 0xff00ff00ff00ff00ULL);
    write_lds_u64(&state, 160u, 0x0011001100110011ULL);
    write_lds_u64(&state, 176u, 0xffff0000ffff0000ULL);
    write_lds_u64(&state, 192u, 0xffff0000aaaa5555ULL);
    write_lds_u64(&state, 208u, 0x1122334455667788ULL);
    write_lds_u64(&state, 224u, DoubleBits(2.5));
    write_lds_u64(&state, 240u, DoubleBits(1.5));
    write_lds_u64(&state, 256u, DoubleBits(4.0));
    write_lds_u64(&state, 272u, DoubleBits(4.0));
    return state;
  };
  auto validate_ds_b64_update_state = [&](const WaveExecutionState& state,
                                          const char* mode) {
    if (!Expect(state.halted, "expected ds b64 update program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }
    const auto read_lds_u64 = [&](std::uint64_t address) {
      std::uint64_t value = 0;
      std::memcpy(&value, state.lds_bytes.data() + address, sizeof(value));
      return value;
    };
    const std::array<DsLds64Expectation, 18> expectations = {{
        {0u, 15u, "expected ds_add_u64 result"},
        {16u, 14u, "expected ds_sub_u64 result"},
        {32u, 6u, "expected ds_rsub_u64 result"},
        {48u, 0u, "expected ds_inc_u64 result"},
        {64u, 11u, "expected ds_dec_u64 result"},
        {80u, 0xfffffffffffffffcULL, "expected ds_min_i64 result"},
        {96u, 2u, "expected ds_max_i64 result"},
        {112u, 4u, "expected ds_min_u64 result"},
        {128u, 14u, "expected ds_max_u64 result"},
        {144u, 0x0f000f000f000f00ULL, "expected ds_and_b64 result"},
        {160u, 0x1111111111111111ULL, "expected ds_or_b64 result"},
        {176u, 0xff0000ffff0000ffULL, "expected ds_xor_b64 result"},
        {192u,
         (0xffff0000aaaa5555ULL & ~0xff00ff0000ff00ffULL) |
             0x005500aa550000aaULL,
         "expected ds_mskor_b64 result"},
        {208u, 0x8877665544332211ULL, "expected ds_cmpst_b64 result"},
        {224u, DoubleBits(2.5), "expected ds_cmpst_f64 result"},
        {240u, DoubleBits(3.75), "expected ds_add_f64 result"},
        {256u, DoubleBits(-1.0), "expected ds_min_f64 result"},
        {272u, DoubleBits(8.0), "expected ds_max_f64 result"},
    }};
    for (const DsLds64Expectation& expectation : expectations) {
      if (!Expect(read_lds_u64(expectation.address) == expectation.expected,
                  expectation.label)) {
        std::cerr << mode << " address=" << expectation.address << '\n';
        return false;
      }
    }
    return true;
  };

  WaveExecutionState decoded_ds_b64_update_state = make_ds_b64_update_state();
  if (!Expect(interpreter.ExecuteProgram(ds_b64_update_program,
                                         &decoded_ds_b64_update_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_update_state(decoded_ds_b64_update_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_b64_update_program;
  if (!Expect(interpreter.CompileProgram(ds_b64_update_program,
                                         &compiled_ds_b64_update_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_b64_update_state = make_ds_b64_update_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_b64_update_program,
                                         &compiled_ds_b64_update_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_update_state(compiled_ds_b64_update_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_b64_return_program = {
      DecodedInstruction::FourOperand("DS_ADD_RTN_U64", InstructionOperand::Vgpr(80),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_SUB_RTN_U64", InstructionOperand::Vgpr(83),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_RSUB_RTN_U64", InstructionOperand::Vgpr(86),
                                      InstructionOperand::Vgpr(6),
                                      InstructionOperand::Vgpr(7),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_INC_RTN_U64", InstructionOperand::Vgpr(89),
                                      InstructionOperand::Vgpr(9),
                                      InstructionOperand::Vgpr(10),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_DEC_RTN_U64", InstructionOperand::Vgpr(92),
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Vgpr(13),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MIN_RTN_I64", InstructionOperand::Vgpr(95),
                                      InstructionOperand::Vgpr(15),
                                      InstructionOperand::Vgpr(16),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MAX_RTN_I64", InstructionOperand::Vgpr(98),
                                      InstructionOperand::Vgpr(18),
                                      InstructionOperand::Vgpr(19),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MIN_RTN_U64", InstructionOperand::Vgpr(101),
                                      InstructionOperand::Vgpr(21),
                                      InstructionOperand::Vgpr(22),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MAX_RTN_U64", InstructionOperand::Vgpr(104),
                                      InstructionOperand::Vgpr(24),
                                      InstructionOperand::Vgpr(25),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_AND_RTN_B64", InstructionOperand::Vgpr(107),
                                      InstructionOperand::Vgpr(27),
                                      InstructionOperand::Vgpr(28),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_OR_RTN_B64", InstructionOperand::Vgpr(110),
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(31),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_XOR_RTN_B64", InstructionOperand::Vgpr(113),
                                      InstructionOperand::Vgpr(33),
                                      InstructionOperand::Vgpr(34),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand(
          "DS_WRXCHG_RTN_B64", InstructionOperand::Vgpr(116),
          InstructionOperand::Vgpr(36), InstructionOperand::Vgpr(37),
          InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_ADD_RTN_F64", InstructionOperand::Vgpr(119),
                                      InstructionOperand::Vgpr(42),
                                      InstructionOperand::Vgpr(43),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MIN_RTN_F64", InstructionOperand::Vgpr(122),
                                      InstructionOperand::Vgpr(45),
                                      InstructionOperand::Vgpr(46),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("DS_MAX_RTN_F64", InstructionOperand::Vgpr(125),
                                      InstructionOperand::Vgpr(48),
                                      InstructionOperand::Vgpr(49),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand(
          "DS_MSKOR_RTN_B64", InstructionOperand::Vgpr(68),
          InstructionOperand::Vgpr(51), InstructionOperand::Vgpr(52),
          InstructionOperand::Vgpr(54), InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand(
          "DS_CMPST_RTN_B64", InstructionOperand::Vgpr(71),
          InstructionOperand::Vgpr(57), InstructionOperand::Vgpr(58),
          InstructionOperand::Vgpr(60), InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand(
          "DS_CMPST_RTN_F64", InstructionOperand::Vgpr(74),
          InstructionOperand::Vgpr(63), InstructionOperand::Vgpr(64),
          InstructionOperand::Vgpr(66), InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  struct DsReturn64Expectation {
    std::uint16_t destination_reg;
    std::uint64_t expected_return;
    std::uint64_t address;
    std::uint64_t expected_memory;
    const char* return_label;
    const char* memory_label;
  };
  auto make_ds_b64_return_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0x1ULL;
    auto write_lds_u64 = [](WaveExecutionState* wave,
                            std::uint64_t address,
                            std::uint64_t value) {
      std::memcpy(wave->lds_bytes.data() + address, &value, sizeof(value));
    };

    state.vgprs[0][0] = 0u;
    SetLane0VgprU64(&state, 1, 5u);
    state.vgprs[3][0] = 16u;
    SetLane0VgprU64(&state, 4, 6u);
    state.vgprs[6][0] = 32u;
    SetLane0VgprU64(&state, 7, 10u);
    state.vgprs[9][0] = 48u;
    SetLane0VgprU64(&state, 10, 7u);
    state.vgprs[12][0] = 64u;
    SetLane0VgprU64(&state, 13, 11u);
    state.vgprs[15][0] = 80u;
    SetLane0VgprU64(&state, 16, 2u);
    state.vgprs[18][0] = 96u;
    SetLane0VgprU64(&state, 19, 2u);
    state.vgprs[21][0] = 112u;
    SetLane0VgprU64(&state, 22, 4u);
    state.vgprs[24][0] = 128u;
    SetLane0VgprU64(&state, 25, 14u);
    state.vgprs[27][0] = 144u;
    SetLane0VgprU64(&state, 28, 0x0f0f0f0f0f0f0f0fULL);
    state.vgprs[30][0] = 160u;
    SetLane0VgprU64(&state, 31, 0x1100110011001100ULL);
    state.vgprs[33][0] = 176u;
    SetLane0VgprU64(&state, 34, 0x00ff00ff00ff00ffULL);
    state.vgprs[36][0] = 192u;
    SetLane0VgprU64(&state, 37, 0xfedcba9876543210ULL);
    state.vgprs[42][0] = 208u;
    SetLane0VgprU64(&state, 43, DoubleBits(2.25));
    state.vgprs[45][0] = 224u;
    SetLane0VgprU64(&state, 46, DoubleBits(-1.0));
    state.vgprs[48][0] = 240u;
    SetLane0VgprU64(&state, 49, DoubleBits(8.0));
    state.vgprs[51][0] = 256u;
    SetLane0VgprU64(&state, 52, 0xff00ff0000ff00ffULL);
    SetLane0VgprU64(&state, 54, 0x005500aa550000aaULL);
    state.vgprs[57][0] = 272u;
    SetLane0VgprU64(&state, 58, 0x1122334455667788ULL);
    SetLane0VgprU64(&state, 60, 0x8877665544332211ULL);
    state.vgprs[63][0] = 288u;
    SetLane0VgprU64(&state, 64, DoubleBits(3.5));
    SetLane0VgprU64(&state, 66, DoubleBits(9.0));

    for (std::uint16_t reg = 68; reg <= 125; ++reg) {
      state.vgprs[reg][0] = 0xdead0000u + reg;
    }

    write_lds_u64(&state, 0u, 10u);
    write_lds_u64(&state, 16u, 20u);
    write_lds_u64(&state, 32u, 4u);
    write_lds_u64(&state, 48u, 7u);
    write_lds_u64(&state, 64u, 0u);
    write_lds_u64(&state, 80u, 0xfffffffffffffffcULL);
    write_lds_u64(&state, 96u, 0xfffffffffffffffdULL);
    write_lds_u64(&state, 112u, 9u);
    write_lds_u64(&state, 128u, 9u);
    write_lds_u64(&state, 144u, 0xff00ff00ff00ff00ULL);
    write_lds_u64(&state, 160u, 0x0011001100110011ULL);
    write_lds_u64(&state, 176u, 0xffff0000ffff0000ULL);
    write_lds_u64(&state, 192u, 0x0123456789abcdefULL);
    write_lds_u64(&state, 208u, DoubleBits(1.5));
    write_lds_u64(&state, 224u, DoubleBits(4.0));
    write_lds_u64(&state, 240u, DoubleBits(4.0));
    write_lds_u64(&state, 256u, 0xffff0000aaaa5555ULL);
    write_lds_u64(&state, 272u, 0x1122334455667788ULL);
    write_lds_u64(&state, 288u, DoubleBits(2.5));
    return state;
  };
  auto validate_ds_b64_return_state = [&](const WaveExecutionState& state,
                                          const char* mode) {
    if (!Expect(state.halted, "expected ds b64 return program to halt")) {
      std::cerr << mode << '\n';
      return false;
    }
    const auto read_lds_u64 = [&](std::uint64_t address) {
      std::uint64_t value = 0;
      std::memcpy(&value, state.lds_bytes.data() + address, sizeof(value));
      return value;
    };
    const std::array<DsReturn64Expectation, 19> expectations = {{
        {80u, 10u, 0u, 15u, "expected ds_add_rtn_u64 return",
         "expected ds_add_rtn_u64 memory"},
        {83u, 20u, 16u, 14u, "expected ds_sub_rtn_u64 return",
         "expected ds_sub_rtn_u64 memory"},
        {86u, 4u, 32u, 6u, "expected ds_rsub_rtn_u64 return",
         "expected ds_rsub_rtn_u64 memory"},
        {89u, 7u, 48u, 0u, "expected ds_inc_rtn_u64 return",
         "expected ds_inc_rtn_u64 memory"},
        {92u, 0u, 64u, 11u, "expected ds_dec_rtn_u64 return",
         "expected ds_dec_rtn_u64 memory"},
        {95u, 0xfffffffffffffffcULL, 80u, 0xfffffffffffffffcULL,
         "expected ds_min_rtn_i64 return", "expected ds_min_rtn_i64 memory"},
        {98u, 0xfffffffffffffffdULL, 96u, 2u,
         "expected ds_max_rtn_i64 return", "expected ds_max_rtn_i64 memory"},
        {101u, 9u, 112u, 4u, "expected ds_min_rtn_u64 return",
         "expected ds_min_rtn_u64 memory"},
        {104u, 9u, 128u, 14u, "expected ds_max_rtn_u64 return",
         "expected ds_max_rtn_u64 memory"},
        {107u, 0xff00ff00ff00ff00ULL, 144u, 0x0f000f000f000f00ULL,
         "expected ds_and_rtn_b64 return", "expected ds_and_rtn_b64 memory"},
        {110u, 0x0011001100110011ULL, 160u, 0x1111111111111111ULL,
         "expected ds_or_rtn_b64 return", "expected ds_or_rtn_b64 memory"},
        {113u, 0xffff0000ffff0000ULL, 176u, 0xff0000ffff0000ffULL,
         "expected ds_xor_rtn_b64 return", "expected ds_xor_rtn_b64 memory"},
        {116u, 0x0123456789abcdefULL, 192u, 0xfedcba9876543210ULL,
         "expected ds_wrxchg_rtn_b64 return",
         "expected ds_wrxchg_rtn_b64 memory"},
        {119u, DoubleBits(1.5), 208u, DoubleBits(3.75),
         "expected ds_add_rtn_f64 return", "expected ds_add_rtn_f64 memory"},
        {122u, DoubleBits(4.0), 224u, DoubleBits(-1.0),
         "expected ds_min_rtn_f64 return", "expected ds_min_rtn_f64 memory"},
        {125u, DoubleBits(4.0), 240u, DoubleBits(8.0),
         "expected ds_max_rtn_f64 return", "expected ds_max_rtn_f64 memory"},
        {68u, 0xffff0000aaaa5555ULL, 256u,
         (0xffff0000aaaa5555ULL & ~0xff00ff0000ff00ffULL) |
             0x005500aa550000aaULL,
         "expected ds_mskor_rtn_b64 return", "expected ds_mskor_rtn_b64 memory"},
        {71u, 0x1122334455667788ULL, 272u, 0x8877665544332211ULL,
         "expected ds_cmpst_rtn_b64 return", "expected ds_cmpst_rtn_b64 memory"},
        {74u, DoubleBits(2.5), 288u, DoubleBits(2.5),
         "expected ds_cmpst_rtn_f64 return", "expected ds_cmpst_rtn_f64 memory"},
    }};
    for (const DsReturn64Expectation& expectation : expectations) {
      if (!Expect(ReadLane0VgprU64(state, expectation.destination_reg) ==
                      expectation.expected_return,
                  expectation.return_label) ||
          !Expect(read_lds_u64(expectation.address) == expectation.expected_memory,
                  expectation.memory_label)) {
        std::cerr << mode << " dest=" << expectation.destination_reg << '\n';
        return false;
      }
    }
    return true;
  };

  WaveExecutionState decoded_ds_b64_return_state = make_ds_b64_return_state();
  if (!Expect(interpreter.ExecuteProgram(ds_b64_return_program,
                                         &decoded_ds_b64_return_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_return_state(decoded_ds_b64_return_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_b64_return_program;
  if (!Expect(interpreter.CompileProgram(ds_b64_return_program,
                                         &compiled_ds_b64_return_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_b64_return_state = make_ds_b64_return_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_b64_return_program,
                                         &compiled_ds_b64_return_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_b64_return_state(compiled_ds_b64_return_state, "compiled")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> ds_condxchg32_rtn_b64_program = {
      DecodedInstruction::FourOperand("DS_CONDXCHG32_RTN_B64",
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Vgpr(0),
                                      InstructionOperand::Vgpr(1),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto make_ds_condxchg32_rtn_b64_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0x1ULL;
    state.vgprs[0][0] = 64u;
    SetLane0VgprU64(&state, 1, ComposeU64(0x80000033u, 0x00000044u));
    state.vgprs[4][0] = 0xdeadbeefu;
    state.vgprs[5][0] = 0xcafebabeu;
    const std::uint64_t initial_value = ComposeU64(0x11111111u, 0x22222222u);
    std::memcpy(state.lds_bytes.data() + 64u, &initial_value, sizeof(initial_value));
    return state;
  };
  auto validate_ds_condxchg32_rtn_b64_state =
      [&](const WaveExecutionState& state, const char* mode) {
        if (!Expect(state.halted,
                    "expected ds condxchg32 rtn b64 program to halt") ||
            !Expect(ReadLane0VgprU64(state, 4) ==
                        ComposeU64(0x11111111u, 0x22222222u),
                    "expected ds condxchg32 rtn b64 return value")) {
          std::cerr << mode << '\n';
          return false;
        }
        std::uint64_t lds_value = 0;
        std::memcpy(&lds_value, state.lds_bytes.data() + 64u, sizeof(lds_value));
        if (!Expect(lds_value == ComposeU64(0x80000033u, 0x22222222u),
                    "expected ds condxchg32 rtn b64 final value")) {
          std::cerr << mode << '\n';
          return false;
        }
        return true;
      };

  WaveExecutionState decoded_ds_condxchg32_rtn_b64_state =
      make_ds_condxchg32_rtn_b64_state();
  if (!Expect(interpreter.ExecuteProgram(ds_condxchg32_rtn_b64_program,
                                         &decoded_ds_condxchg32_rtn_b64_state,
                                         &error_message),
              error_message.c_str()) ||
      !validate_ds_condxchg32_rtn_b64_state(
          decoded_ds_condxchg32_rtn_b64_state, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_ds_condxchg32_rtn_b64_program;
  if (!Expect(interpreter.CompileProgram(ds_condxchg32_rtn_b64_program,
                                         &compiled_ds_condxchg32_rtn_b64_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  WaveExecutionState compiled_ds_condxchg32_rtn_b64_state =
      make_ds_condxchg32_rtn_b64_state();
  if (!Expect(interpreter.ExecuteProgram(
                  compiled_ds_condxchg32_rtn_b64_program,
                  &compiled_ds_condxchg32_rtn_b64_state, &error_message),
              error_message.c_str()) ||
      !validate_ds_condxchg32_rtn_b64_state(
          compiled_ds_condxchg32_rtn_b64_state, "compiled")) {
    return 1;
  }
  }

  WaveExecutionState writer_wave;
  writer_wave.exec_mask = 0x1ULL;
  writer_wave.workgroup_wave_count = 2;
  writer_wave.sgprs[0] = 0u;
  writer_wave.vgprs[0][0] = 0u;
  writer_wave.vgprs[1][0] = 99u;
  WaveExecutionState reader_wave;
  reader_wave.exec_mask = 0x1ULL;
  reader_wave.workgroup_wave_count = 2;
  reader_wave.sgprs[0] = 1u;
  reader_wave.vgprs[0][0] = 0u;
  reader_wave.vgprs[2][0] = 0xdeadbeefu;
  std::vector<std::byte> shared_lds(WaveExecutionState::kLdsSizeBytes);
  WorkgroupExecutionContext workgroup;
  workgroup.shared_lds = std::span<std::byte>(shared_lds.data(), shared_lds.size());
  workgroup.wave_count = 2;
  const std::vector<DecodedInstruction> barrier_program = {
      DecodedInstruction::TwoOperand("S_CMP_EQ_U32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(0)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC1",
                                     InstructionOperand::Imm32(3)),
      DecodedInstruction::Nullary("S_BARRIER"),
      DecodedInstruction::ThreeOperand("DS_READ_B32", InstructionOperand::Vgpr(2),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
      DecodedInstruction::ThreeOperand("DS_WRITE_B32", InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(1),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_BARRIER"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::array<WaveExecutionState*, 2> waves = {&writer_wave, &reader_wave};
  bool barrier_failed = false;
  for (;;) {
    bool all_done = true;
    std::size_t blocked_waves = 0;
    for (WaveExecutionState* wave : waves) {
      if (wave->halted || wave->pc >= barrier_program.size()) {
        continue;
      }
      all_done = false;
      if (wave->waiting_on_barrier) {
        ++blocked_waves;
        continue;
      }

      ProgramRunState run_state = ProgramRunState::kCompleted;
      if (!Expect(interpreter.ExecuteProgramUntilYield(
                      barrier_program, wave, nullptr, &workgroup, &run_state,
                      &error_message),
                  error_message.c_str())) {
        return 1;
      }
      if (run_state == ProgramRunState::kBlockedOnBarrier) {
        ++blocked_waves;
      }
    }

    if (all_done) {
      break;
    }
    if (blocked_waves != 0) {
      if (!Expect(blocked_waves == waves.size(),
                  "expected all workgroup waves to rendezvous at barrier")) {
        barrier_failed = true;
        break;
      }
      for (WaveExecutionState* wave : waves) {
        wave->waiting_on_barrier = false;
      }
    }
  }
  if (barrier_failed) {
    return 1;
  }

  std::uint32_t shared_lds_value = 0;
  std::memcpy(&shared_lds_value, shared_lds.data(), sizeof(shared_lds_value));
  if (!Expect(writer_wave.halted, "expected writer wave to halt") ||
      !Expect(reader_wave.halted, "expected reader wave to halt") ||
      !Expect(reader_wave.vgprs[2][0] == 99u,
              "expected reader wave to observe shared lds write") ||
      !Expect(writer_wave.vgprs[2][0] == 0u,
              "expected writer wave destination register to remain untouched") ||
      !Expect(shared_lds_value == 99u, "expected shared lds value")) {
    return 1;
  }

  WaveExecutionState branch_state;
  const std::vector<DecodedInstruction> branch_program = {
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(3)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(0)),
      DecodedInstruction::Binary("S_ADD_U32", InstructionOperand::Sgpr(1),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(1)),
      DecodedInstruction::Binary("S_SUB_U32", InstructionOperand::Sgpr(0),
                                 InstructionOperand::Sgpr(0),
                                 InstructionOperand::Imm32(1)),
      DecodedInstruction::TwoOperand("S_CMP_LG_U32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(0)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC1",
                                     InstructionOperand::Imm32(
                                         static_cast<std::uint32_t>(-4))),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(branch_program, &branch_state, &error_message),
              error_message.c_str()) ||
      !Expect(branch_state.halted, "expected branch program to halt") ||
      !Expect(branch_state.sgprs[0] == 0, "expected loop counter to reach zero") ||
      !Expect(branch_state.sgprs[1] == 3, "expected loop body to run three times") ||
      !Expect(!branch_state.scc, "expected final SCC to be zero")) {
    return 1;
  }

  WaveExecutionState compare_branch_state;
  const std::vector<DecodedInstruction> compare_branch_program = {
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(5)),
      DecodedInstruction::TwoOperand("S_CMP_EQ_U32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(4)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC0",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(2),
                                InstructionOperand::Imm32(99)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(compare_branch_program,
                                         &compare_branch_state, &error_message),
              error_message.c_str()) ||
      !Expect(compare_branch_state.halted, "expected compare/branch program to halt") ||
      !Expect(compare_branch_state.sgprs[2] == 0,
              "expected SCC0 branch to skip the wrong-path move") ||
      !Expect(!compare_branch_state.scc, "expected compare-equal to leave SCC false")) {
    return 1;
  }

  const std::vector<DecodedInstruction> bitcmp_program = {
      DecodedInstruction::TwoOperand("S_BITCMP0_B32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(10),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::TwoOperand("S_BITCMP1_B32", InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC0",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(11),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::TwoOperand("S_BITCMP1_B64", InstructionOperand::Sgpr(2),
                                     InstructionOperand::Imm32(63)),
      DecodedInstruction::OneOperand("S_CBRANCH_SCC0",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(12),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState bitcmp_state;
  bitcmp_state.sgprs[0] = 0x8u;
  bitcmp_state.sgprs[2] = 0u;
  bitcmp_state.sgprs[3] = 0x80000000u;
  if (!Expect(interpreter.ExecuteProgram(bitcmp_program, &bitcmp_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(bitcmp_state.halted, "expected bitcmp program to halt") ||
      !Expect(bitcmp_state.sgprs[10] == 111u,
              "expected bitcmp0 true-path move result") ||
      !Expect(bitcmp_state.sgprs[11] == 0u,
              "expected bitcmp1 false-path branch to skip move") ||
      !Expect(bitcmp_state.sgprs[12] == 333u,
              "expected bitcmp1_b64 true-path move result") ||
      !Expect(bitcmp_state.scc, "expected final bitcmp SCC to be true")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_bitcmp_program;
  if (!Expect(interpreter.CompileProgram(bitcmp_program,
                                         &compiled_bitcmp_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_bitcmp_state;
  compiled_bitcmp_state.sgprs[0] = 0x8u;
  compiled_bitcmp_state.sgprs[2] = 0u;
  compiled_bitcmp_state.sgprs[3] = 0x80000000u;
  if (!Expect(interpreter.ExecuteProgram(compiled_bitcmp_program,
                                         &compiled_bitcmp_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_bitcmp_state.halted,
              "expected compiled bitcmp program to halt") ||
      !Expect(compiled_bitcmp_state.sgprs[10] == 111u,
              "expected compiled bitcmp0 true-path move result") ||
      !Expect(compiled_bitcmp_state.sgprs[11] == 0u,
              "expected compiled bitcmp1 false-path branch to skip move") ||
      !Expect(compiled_bitcmp_state.sgprs[12] == 333u,
              "expected compiled bitcmp1_b64 true-path move result") ||
      !Expect(compiled_bitcmp_state.scc,
              "expected compiled final bitcmp SCC to be true")) {
    return 1;
  }

  WaveExecutionState vcc_branch_state;
  const std::vector<DecodedInstruction> vcc_branch_program = {
      DecodedInstruction::Binary("V_CMP_EQ_U32", InstructionOperand::Sgpr(106),
                                 InstructionOperand::Sgpr(12),
                                 InstructionOperand::Vgpr(12)),
      DecodedInstruction::OneOperand("S_CBRANCH_VCCNZ",
                                     InstructionOperand::Imm32(2)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(2),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::OneOperand("S_CBRANCH_VCCZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(3),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  vcc_branch_state.exec_mask = 0b1011ULL;
  vcc_branch_state.sgprs[12] = 7u;
  vcc_branch_state.vgprs[12][0] = 7u;
  vcc_branch_state.vgprs[12][1] = 5u;
  vcc_branch_state.vgprs[12][3] = 7u;
  if (!Expect(interpreter.ExecuteProgram(vcc_branch_program, &vcc_branch_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vcc_branch_state.halted, "expected VCC branch program to halt") ||
      !Expect(vcc_branch_state.sgprs[2] == 0u,
              "expected VCCNZ branch to skip first move") ||
      !Expect(vcc_branch_state.sgprs[3] == 222u,
              "expected VCCZ branch to fall through when VCC is nonzero") ||
      !Expect(vcc_branch_state.vcc_mask == 9u,
              "expected VCC branch program to preserve compare mask")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vcc_branch_program;
  if (!Expect(interpreter.CompileProgram(vcc_branch_program,
                                         &compiled_vcc_branch_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_vcc_branch_state;
  compiled_vcc_branch_state.exec_mask = 0b1011ULL;
  compiled_vcc_branch_state.sgprs[12] = 7u;
  compiled_vcc_branch_state.vgprs[12][0] = 7u;
  compiled_vcc_branch_state.vgprs[12][1] = 5u;
  compiled_vcc_branch_state.vgprs[12][3] = 7u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vcc_branch_program,
                                         &compiled_vcc_branch_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_vcc_branch_state.halted,
              "expected compiled VCC branch program to halt") ||
      !Expect(compiled_vcc_branch_state.sgprs[2] == 0u,
              "expected compiled VCCNZ branch to skip first move") ||
      !Expect(compiled_vcc_branch_state.sgprs[3] == 222u,
              "expected compiled VCCZ branch to fall through when VCC is nonzero") ||
      !Expect(compiled_vcc_branch_state.vcc_mask == 9u,
              "expected compiled VCC branch program to preserve compare mask")) {
    return 1;
  }

  const std::vector<DecodedInstruction> exec_mask_program = {
      DecodedInstruction::Unary("S_AND_SAVEEXEC_B64", InstructionOperand::Sgpr(30),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::Unary("V_MOV_B32", InstructionOperand::Vgpr(20),
                                InstructionOperand::Sgpr(26)),
      DecodedInstruction::Unary("S_AND_SAVEEXEC_B64", InstructionOperand::Sgpr(32),
                                InstructionOperand::Sgpr(22)),
      DecodedInstruction::Unary("V_MOV_B32", InstructionOperand::Vgpr(21),
                                InstructionOperand::Sgpr(27)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(0),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Unary("S_OR_SAVEEXEC_B64", InstructionOperand::Sgpr(34),
                                InstructionOperand::Sgpr(24)),
      DecodedInstruction::Unary("V_MOV_B32", InstructionOperand::Vgpr(22),
                                InstructionOperand::Sgpr(28)),
      DecodedInstruction::Unary("V_READFIRSTLANE_B32", InstructionOperand::Sgpr(2),
                                InstructionOperand::Vgpr(22)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState exec_mask_state;
  exec_mask_state.exec_mask = 0b1011ULL;
  exec_mask_state.sgprs[20] = 0b1001u;
  exec_mask_state.sgprs[21] = 0u;
  exec_mask_state.sgprs[22] = 0u;
  exec_mask_state.sgprs[23] = 0u;
  exec_mask_state.sgprs[24] = 0b0010u;
  exec_mask_state.sgprs[25] = 0u;
  exec_mask_state.sgprs[26] = 55u;
  exec_mask_state.sgprs[27] = 66u;
  exec_mask_state.sgprs[28] = 77u;
  exec_mask_state.vgprs[21][2] = 0xdeadbeefu;
  exec_mask_state.vgprs[22][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(exec_mask_program, &exec_mask_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(exec_mask_state.halted, "expected exec-mask program to halt") ||
      !Expect(exec_mask_state.sgprs[30] == 0b1011u,
              "expected first saveexec low result") ||
      !Expect(exec_mask_state.sgprs[31] == 0u,
              "expected first saveexec high result") ||
      !Expect(exec_mask_state.sgprs[32] == 0b1001u,
              "expected second saveexec low result") ||
      !Expect(exec_mask_state.sgprs[33] == 0u,
              "expected second saveexec high result") ||
      !Expect(exec_mask_state.sgprs[34] == 0u,
              "expected third saveexec low result") ||
      !Expect(exec_mask_state.sgprs[35] == 0u,
              "expected third saveexec high result") ||
      !Expect(exec_mask_state.sgprs[0] == 111u,
              "expected EXECNZ fallthrough move result") ||
      !Expect(exec_mask_state.sgprs[1] == 0u,
              "expected EXECZ branch to skip move") ||
      !Expect(exec_mask_state.sgprs[2] == 77u,
              "expected v_readfirstlane_b32 result") ||
      !Expect(exec_mask_state.exec_mask == 0b0010u,
              "expected final exec mask") ||
      !Expect(exec_mask_state.scc,
              "expected final saveexec/or to leave SCC true") ||
      !Expect(exec_mask_state.vgprs[20][0] == 55u,
              "expected exec-masked vector move lane 0 result") ||
      !Expect(exec_mask_state.vgprs[20][1] == 0u,
              "expected exec-masked vector move lane 1 result") ||
      !Expect(exec_mask_state.vgprs[20][3] == 55u,
              "expected exec-masked vector move lane 3 result") ||
      !Expect(exec_mask_state.vgprs[21][0] == 0u,
              "expected zero-exec vector move lane 0 result") ||
      !Expect(exec_mask_state.vgprs[21][2] == 0xdeadbeefu,
              "expected zero-exec inactive lane preservation") ||
      !Expect(exec_mask_state.vgprs[22][0] == 0u,
              "expected restored-exec vector move lane 0 result") ||
      !Expect(exec_mask_state.vgprs[22][1] == 77u,
              "expected restored-exec vector move lane 1 result") ||
      !Expect(exec_mask_state.vgprs[22][2] == 0xdeadbeefu,
              "expected restored-exec inactive lane preservation") ||
      !Expect(exec_mask_state.vgprs[22][3] == 0u,
              "expected restored-exec vector move lane 3 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_exec_mask_program;
  if (!Expect(interpreter.CompileProgram(exec_mask_program,
                                         &compiled_exec_mask_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_exec_mask_state;
  compiled_exec_mask_state.exec_mask = 0b1011ULL;
  compiled_exec_mask_state.sgprs[20] = 0b1001u;
  compiled_exec_mask_state.sgprs[21] = 0u;
  compiled_exec_mask_state.sgprs[22] = 0u;
  compiled_exec_mask_state.sgprs[23] = 0u;
  compiled_exec_mask_state.sgprs[24] = 0b0010u;
  compiled_exec_mask_state.sgprs[25] = 0u;
  compiled_exec_mask_state.sgprs[26] = 55u;
  compiled_exec_mask_state.sgprs[27] = 66u;
  compiled_exec_mask_state.sgprs[28] = 77u;
  compiled_exec_mask_state.vgprs[21][2] = 0xdeadbeefu;
  compiled_exec_mask_state.vgprs[22][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(compiled_exec_mask_program,
                                         &compiled_exec_mask_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_exec_mask_state.halted,
              "expected compiled exec-mask program to halt") ||
      !Expect(compiled_exec_mask_state.sgprs[30] == 0b1011u,
              "expected compiled first saveexec low result") ||
      !Expect(compiled_exec_mask_state.sgprs[31] == 0u,
              "expected compiled first saveexec high result") ||
      !Expect(compiled_exec_mask_state.sgprs[32] == 0b1001u,
              "expected compiled second saveexec low result") ||
      !Expect(compiled_exec_mask_state.sgprs[33] == 0u,
              "expected compiled second saveexec high result") ||
      !Expect(compiled_exec_mask_state.sgprs[34] == 0u,
              "expected compiled third saveexec low result") ||
      !Expect(compiled_exec_mask_state.sgprs[35] == 0u,
              "expected compiled third saveexec high result") ||
      !Expect(compiled_exec_mask_state.sgprs[0] == 111u,
              "expected compiled EXECNZ fallthrough move result") ||
      !Expect(compiled_exec_mask_state.sgprs[1] == 0u,
              "expected compiled EXECZ branch to skip move") ||
      !Expect(compiled_exec_mask_state.sgprs[2] == 77u,
              "expected compiled v_readfirstlane_b32 result") ||
      !Expect(compiled_exec_mask_state.exec_mask == 0b0010u,
              "expected compiled final exec mask") ||
      !Expect(compiled_exec_mask_state.scc,
              "expected compiled final saveexec/or to leave SCC true") ||
      !Expect(compiled_exec_mask_state.vgprs[20][0] == 55u,
              "expected compiled exec-masked vector move lane 0 result") ||
      !Expect(compiled_exec_mask_state.vgprs[20][1] == 0u,
              "expected compiled exec-masked vector move lane 1 result") ||
      !Expect(compiled_exec_mask_state.vgprs[20][3] == 55u,
              "expected compiled exec-masked vector move lane 3 result") ||
      !Expect(compiled_exec_mask_state.vgprs[21][0] == 0u,
              "expected compiled zero-exec vector move lane 0 result") ||
      !Expect(compiled_exec_mask_state.vgprs[21][2] == 0xdeadbeefu,
              "expected compiled zero-exec inactive lane preservation") ||
      !Expect(compiled_exec_mask_state.vgprs[22][0] == 0u,
              "expected compiled restored-exec vector move lane 0 result") ||
      !Expect(compiled_exec_mask_state.vgprs[22][1] == 77u,
              "expected compiled restored-exec vector move lane 1 result") ||
      !Expect(compiled_exec_mask_state.vgprs[22][2] == 0xdeadbeefu,
              "expected compiled restored-exec inactive lane preservation") ||
      !Expect(compiled_exec_mask_state.vgprs[22][3] == 0u,
              "expected compiled restored-exec vector move lane 3 result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> exec_pair_program = {
      DecodedInstruction::Unary("S_MOV_B64", InstructionOperand::Sgpr(126),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(10),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Binary("S_AND_B64", InstructionOperand::Sgpr(126),
                                 InstructionOperand::Sgpr(126),
                                 InstructionOperand::Sgpr(22)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(11),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Binary("S_OR_B64", InstructionOperand::Sgpr(126),
                                 InstructionOperand::Sgpr(126),
                                 InstructionOperand::Sgpr(24)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(12),
                                InstructionOperand::Imm32(333)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState exec_pair_state;
  exec_pair_state.exec_mask = 0b1011ULL;
  exec_pair_state.sgprs[20] = 0b1001u;
  exec_pair_state.sgprs[21] = 0u;
  exec_pair_state.sgprs[22] = 0b0010u;
  exec_pair_state.sgprs[23] = 0u;
  exec_pair_state.sgprs[24] = 0b0100u;
  exec_pair_state.sgprs[25] = 0u;
  if (!Expect(interpreter.ExecuteProgram(exec_pair_program, &exec_pair_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(exec_pair_state.halted, "expected exec-pair program to halt") ||
      !Expect(exec_pair_state.sgprs[10] == 111u,
              "expected first exec-pair fallthrough move") ||
      !Expect(exec_pair_state.sgprs[11] == 0u,
              "expected execz branch to skip second move") ||
      !Expect(exec_pair_state.sgprs[12] == 0u,
              "expected execnz branch to skip third move") ||
      !Expect(exec_pair_state.exec_mask == 0b0100u,
              "expected final exec mask from scalar pair writes")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_exec_pair_program;
  if (!Expect(interpreter.CompileProgram(exec_pair_program,
                                         &compiled_exec_pair_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_exec_pair_state;
  compiled_exec_pair_state.exec_mask = 0b1011ULL;
  compiled_exec_pair_state.sgprs[20] = 0b1001u;
  compiled_exec_pair_state.sgprs[21] = 0u;
  compiled_exec_pair_state.sgprs[22] = 0b0010u;
  compiled_exec_pair_state.sgprs[23] = 0u;
  compiled_exec_pair_state.sgprs[24] = 0b0100u;
  compiled_exec_pair_state.sgprs[25] = 0u;
  if (!Expect(interpreter.ExecuteProgram(compiled_exec_pair_program,
                                         &compiled_exec_pair_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_exec_pair_state.halted,
              "expected compiled exec-pair program to halt") ||
      !Expect(compiled_exec_pair_state.sgprs[10] == 111u,
              "expected compiled first exec-pair fallthrough move") ||
      !Expect(compiled_exec_pair_state.sgprs[11] == 0u,
              "expected compiled execz branch to skip second move") ||
      !Expect(compiled_exec_pair_state.sgprs[12] == 0u,
              "expected compiled execnz branch to skip third move") ||
      !Expect(compiled_exec_pair_state.exec_mask == 0b0100u,
              "expected compiled final exec mask from scalar pair writes")) {
    return 1;
  }

  const std::vector<DecodedInstruction> wrexec_program = {
      DecodedInstruction::Unary("S_ANDN1_WREXEC_B64",
                                InstructionOperand::Sgpr(30),
                                InstructionOperand::Sgpr(20)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECNZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(16),
                                InstructionOperand::Imm32(111)),
      DecodedInstruction::Unary("S_ANDN2_WREXEC_B64",
                                InstructionOperand::Sgpr(32),
                                InstructionOperand::Sgpr(22)),
      DecodedInstruction::OneOperand("S_CBRANCH_EXECZ",
                                     InstructionOperand::Imm32(1)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(17),
                                InstructionOperand::Imm32(222)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState wrexec_state;
  wrexec_state.exec_mask = 0b1011ULL;
  wrexec_state.sgprs[20] = 0b1001u;
  wrexec_state.sgprs[21] = 0u;
  wrexec_state.sgprs[22] = 0b0010u;
  wrexec_state.sgprs[23] = 0u;
  if (!Expect(interpreter.ExecuteProgram(wrexec_program, &wrexec_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(wrexec_state.halted, "expected wrexec program to halt") ||
      !Expect(wrexec_state.sgprs[30] == 0b0010u &&
                  wrexec_state.sgprs[31] == 0u,
              "expected andn1_wrexec result in destination") ||
      !Expect(wrexec_state.sgprs[32] == 0u &&
                  wrexec_state.sgprs[33] == 0u,
              "expected andn2_wrexec result in destination") ||
      !Expect(wrexec_state.sgprs[16] == 0u,
              "expected execnz branch to skip first move") ||
      !Expect(wrexec_state.sgprs[17] == 0u,
              "expected execz branch to skip second move") ||
      !Expect(wrexec_state.exec_mask == 0u,
              "expected final exec mask after wrexec ops") ||
      !Expect(!wrexec_state.scc,
              "expected final scc after zero wrexec result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_wrexec_program;
  if (!Expect(interpreter.CompileProgram(wrexec_program,
                                         &compiled_wrexec_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_wrexec_state;
  compiled_wrexec_state.exec_mask = 0b1011ULL;
  compiled_wrexec_state.sgprs[20] = 0b1001u;
  compiled_wrexec_state.sgprs[21] = 0u;
  compiled_wrexec_state.sgprs[22] = 0b0010u;
  compiled_wrexec_state.sgprs[23] = 0u;
  if (!Expect(interpreter.ExecuteProgram(compiled_wrexec_program,
                                         &compiled_wrexec_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_wrexec_state.halted,
              "expected compiled wrexec program to halt") ||
      !Expect(compiled_wrexec_state.sgprs[30] == 0b0010u &&
                  compiled_wrexec_state.sgprs[31] == 0u,
              "expected compiled andn1_wrexec result in destination") ||
      !Expect(compiled_wrexec_state.sgprs[32] == 0u &&
                  compiled_wrexec_state.sgprs[33] == 0u,
              "expected compiled andn2_wrexec result in destination") ||
      !Expect(compiled_wrexec_state.sgprs[16] == 0u,
              "expected compiled execnz branch to skip first move") ||
      !Expect(compiled_wrexec_state.sgprs[17] == 0u,
              "expected compiled execz branch to skip second move") ||
      !Expect(compiled_wrexec_state.exec_mask == 0u,
              "expected compiled final exec mask after wrexec ops") ||
      !Expect(!compiled_wrexec_state.scc,
              "expected compiled final scc after zero wrexec result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> special_source_program = {
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(13),
                                InstructionOperand::Sgpr(251)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(14),
                                InstructionOperand::Sgpr(252)),
      DecodedInstruction::Unary("S_MOV_B32", InstructionOperand::Sgpr(15),
                                InstructionOperand::Sgpr(253)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState special_source_state;
  special_source_state.exec_mask = 0u;
  special_source_state.vcc_mask = 0u;
  special_source_state.scc = true;
  if (!Expect(interpreter.ExecuteProgram(special_source_program,
                                         &special_source_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(special_source_state.sgprs[13] == 1u,
              "expected src_vccz decoded result") ||
      !Expect(special_source_state.sgprs[14] == 1u,
              "expected src_execz decoded result") ||
      !Expect(special_source_state.sgprs[15] == 1u,
              "expected src_scc decoded result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_special_source_program;
  if (!Expect(interpreter.CompileProgram(special_source_program,
                                         &compiled_special_source_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_special_source_state;
  compiled_special_source_state.exec_mask = 0u;
  compiled_special_source_state.vcc_mask = 0u;
  compiled_special_source_state.scc = true;
  if (!Expect(interpreter.ExecuteProgram(compiled_special_source_program,
                                         &compiled_special_source_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_special_source_state.sgprs[13] == 1u,
              "expected compiled src_vccz result") ||
      !Expect(compiled_special_source_state.sgprs[14] == 1u,
              "expected compiled src_execz result") ||
      !Expect(compiled_special_source_state.sgprs[15] == 1u,
              "expected compiled src_scc result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> lane_ops_program = {
      DecodedInstruction::Binary("V_READLANE_B32", InstructionOperand::Sgpr(4),
                                 InstructionOperand::Vgpr(10),
                                 InstructionOperand::Sgpr(6)),
      DecodedInstruction::Binary("V_WRITELANE_B32", InstructionOperand::Vgpr(12),
                                 InstructionOperand::Sgpr(8),
                                 InstructionOperand::Sgpr(9)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  WaveExecutionState lane_ops_state;
  lane_ops_state.exec_mask = 0x1ULL;
  lane_ops_state.sgprs[6] = 5u;
  lane_ops_state.sgprs[8] = 0xfeedfaceu;
  lane_ops_state.sgprs[9] = 7u;
  lane_ops_state.vgprs[10][5] = 0x12345678u;
  lane_ops_state.vgprs[12][6] = 0xaaaaaaaau;
  lane_ops_state.vgprs[12][7] = 0xbbbbbbbbu;
  if (!Expect(interpreter.ExecuteProgram(lane_ops_program, &lane_ops_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(lane_ops_state.halted, "expected decoded lane-ops program to halt") ||
      !Expect(lane_ops_state.sgprs[4] == 0x12345678u,
              "expected decoded v_readlane_b32 result") ||
      !Expect(lane_ops_state.vgprs[12][6] == 0xaaaaaaaau,
              "expected decoded v_writelane_b32 to preserve neighboring lane") ||
      !Expect(lane_ops_state.vgprs[12][7] == 0xfeedfaceu,
              "expected decoded v_writelane_b32 result")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_lane_ops_program;
  if (!Expect(interpreter.CompileProgram(lane_ops_program,
                                         &compiled_lane_ops_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }

  WaveExecutionState compiled_lane_ops_state;
  compiled_lane_ops_state.exec_mask = 0x1ULL;
  compiled_lane_ops_state.sgprs[6] = 5u;
  compiled_lane_ops_state.sgprs[8] = 0xfeedfaceu;
  compiled_lane_ops_state.sgprs[9] = 7u;
  compiled_lane_ops_state.vgprs[10][5] = 0x12345678u;
  compiled_lane_ops_state.vgprs[12][6] = 0xaaaaaaaau;
  compiled_lane_ops_state.vgprs[12][7] = 0xbbbbbbbbu;
  if (!Expect(interpreter.ExecuteProgram(compiled_lane_ops_program,
                                         &compiled_lane_ops_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(compiled_lane_ops_state.halted,
              "expected compiled lane-ops program to halt") ||
      !Expect(compiled_lane_ops_state.sgprs[4] == 0x12345678u,
              "expected compiled v_readlane_b32 result") ||
      !Expect(compiled_lane_ops_state.vgprs[12][6] == 0xaaaaaaaau,
              "expected compiled v_writelane_b32 to preserve neighboring lane") ||
      !Expect(compiled_lane_ops_state.vgprs[12][7] == 0xfeedfaceu,
              "expected compiled v_writelane_b32 result")) {
    return 1;
  }

  LinearExecutionMemory memory(0x400, 0);
  if (!Expect(memory.WriteU32(0x100, 0x11223344u), "expected memory seed write") ||
      !Expect(memory.WriteU32(0x104, 0x55667788u), "expected memory seed write") ||
      !Expect(memory.WriteU32(0x108, 0x99aabbccu), "expected memory seed write")) {
    return 1;
  }

  WaveExecutionState memory_state;
  memory_state.sgprs[0] = 0x100;
  memory_state.sgprs[1] = 0x0;
  memory_state.sgprs[2] = 0x10;
  const std::vector<DecodedInstruction> memory_program = {
      DecodedInstruction::ThreeOperand("S_LOAD_DWORD", InstructionOperand::Sgpr(4),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX2", InstructionOperand::Sgpr(6),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::ThreeOperand("S_STORE_DWORD", InstructionOperand::Sgpr(4),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(2)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(memory_program, &memory_state, &memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(memory_state.halted, "expected memory program to halt") ||
      !Expect(memory_state.sgprs[4] == 0x11223344u, "expected s_load_dword result") ||
      !Expect(memory_state.sgprs[6] == 0x55667788u,
              "expected s_load_dwordx2 low result") ||
      !Expect(memory_state.sgprs[7] == 0x99aabbccu,
              "expected s_load_dwordx2 high result")) {
    return 1;
  }

  std::uint32_t stored_value = 0;
  if (!Expect(memory.ReadU32(0x110, &stored_value), "expected stored value read") ||
      !Expect(stored_value == 0x11223344u, "expected s_store_dword result")) {
    return 1;
  }

  {
  const auto seed_wide_scalar_memory = [](LinearExecutionMemory* memory) {
    if (memory == nullptr) {
      return false;
    }
    for (std::uint32_t index = 0; index < 4; ++index) {
      if (!memory->WriteU32(0x120u + index * 4u, 0x100u + index)) {
        return false;
      }
    }
    for (std::uint32_t index = 0; index < 8; ++index) {
      if (!memory->WriteU32(0x140u + index * 4u, 0x200u + index)) {
        return false;
      }
    }
    for (std::uint32_t index = 0; index < 16; ++index) {
      if (!memory->WriteU32(0x180u + index * 4u, 0x300u + index)) {
        return false;
      }
    }
    return true;
  };
  const auto make_wide_scalar_memory_state = []() {
    WaveExecutionState state{};
    state.sgprs[0] = 0x100u;
    state.sgprs[1] = 0u;
    state.sgprs[2] = 0x140u;
    state.sgprs[3] = 0x160u;
    return state;
  };
  const auto validate_wide_scalar_memory_state =
      [](const WaveExecutionState& state,
         const LinearExecutionMemory& memory,
         const char* mode) {
        if (!Expect(state.halted, "expected wide scalar memory program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }
        for (std::uint32_t index = 0; index < 4; ++index) {
          if (!Expect(state.sgprs[8 + index] == 0x100u + index,
                      "expected s_load_dwordx4 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 8; ++index) {
          if (!Expect(state.sgprs[16 + index] == 0x200u + index,
                      "expected s_load_dwordx8 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 16; ++index) {
          if (!Expect(state.sgprs[32 + index] == 0x300u + index,
                      "expected s_load_dwordx16 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 2; ++index) {
          std::uint32_t value = 0;
          if (!Expect(memory.ReadU32(0x240u + index * 4u, &value),
                      "expected s_store_dwordx2 read") ||
              !Expect(value == 0x200u + index,
                      "expected s_store_dwordx2 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 4; ++index) {
          std::uint32_t value = 0;
          if (!Expect(memory.ReadU32(0x260u + index * 4u, &value),
                      "expected s_store_dwordx4 read") ||
              !Expect(value == 0x100u + index,
                      "expected s_store_dwordx4 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        return true;
      };

  const std::vector<DecodedInstruction> wide_scalar_memory_program = {
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX4", InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x20)),
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX8", InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x40)),
      DecodedInstruction::ThreeOperand("S_LOAD_DWORDX16", InstructionOperand::Sgpr(32),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x80)),
      DecodedInstruction::ThreeOperand("S_STORE_DWORDX2", InstructionOperand::Sgpr(16),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(2)),
      DecodedInstruction::ThreeOperand("S_STORE_DWORDX4", InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Sgpr(3)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };

  LinearExecutionMemory decoded_wide_scalar_memory(0x600, 0);
  if (!Expect(seed_wide_scalar_memory(&decoded_wide_scalar_memory),
              "expected decoded wide scalar memory seed writes")) {
    return 1;
  }
  WaveExecutionState decoded_wide_scalar_memory_state =
      make_wide_scalar_memory_state();
  if (!Expect(interpreter.ExecuteProgram(wide_scalar_memory_program,
                                         &decoded_wide_scalar_memory_state,
                                         &decoded_wide_scalar_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_wide_scalar_memory_state(decoded_wide_scalar_memory_state,
                                         decoded_wide_scalar_memory,
                                         "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_wide_scalar_memory_program;
  if (!Expect(interpreter.CompileProgram(wide_scalar_memory_program,
                                         &compiled_wide_scalar_memory_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  LinearExecutionMemory compiled_wide_scalar_memory(0x600, 0);
  if (!Expect(seed_wide_scalar_memory(&compiled_wide_scalar_memory),
              "expected compiled wide scalar memory seed writes")) {
    return 1;
  }
  WaveExecutionState compiled_wide_scalar_memory_state =
      make_wide_scalar_memory_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_wide_scalar_memory_program,
                                         &compiled_wide_scalar_memory_state,
                                         &compiled_wide_scalar_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_wide_scalar_memory_state(compiled_wide_scalar_memory_state,
                                         compiled_wide_scalar_memory,
                                         "compiled")) {
    return 1;
  }
  }

  {
  const auto seed_scalar_buffer_memory = [](LinearExecutionMemory* memory) {
    if (memory == nullptr) {
      return false;
    }
    if (!memory->WriteU32(0x100u, 0x11110000u)) {
      return false;
    }
    for (std::uint32_t index = 0; index < 2; ++index) {
      if (!memory->WriteU32(0x110u + index * 4u, 0x22220000u + index)) {
        return false;
      }
    }
    for (std::uint32_t index = 0; index < 4; ++index) {
      if (!memory->WriteU32(0x120u + index * 4u, 0x33330000u + index)) {
        return false;
      }
    }
    for (std::uint32_t index = 0; index < 8; ++index) {
      if (!memory->WriteU32(0x140u + index * 4u, 0x44440000u + index)) {
        return false;
      }
    }
    for (std::uint32_t index = 0; index < 16; ++index) {
      if (!memory->WriteU32(0x180u + index * 4u, 0x55550000u + index)) {
        return false;
      }
    }
    return true;
  };
  const auto make_scalar_buffer_state = []() {
    WaveExecutionState state{};
    state.sgprs[0] = 0x100u;
    state.sgprs[1] = 0u;
    state.sgprs[2] = 0x400u;
    state.sgprs[3] = 0u;
    state.sgprs[72] = 0x220u;
    return state;
  };
  const auto validate_scalar_buffer_state =
      [](const WaveExecutionState& state,
         const LinearExecutionMemory& memory,
         const char* mode) {
        if (!Expect(state.halted,
                    "expected scalar buffer memory program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }
        if (!Expect(state.sgprs[4] == 0x11110000u,
                    "expected s_buffer_load_dword result")) {
          std::cerr << mode << '\n';
          return false;
        }
        for (std::uint32_t index = 0; index < 2; ++index) {
          if (!Expect(state.sgprs[8 + index] == 0x22220000u + index,
                      "expected s_buffer_load_dwordx2 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 4; ++index) {
          if (!Expect(state.sgprs[16 + index] == 0x33330000u + index,
                      "expected s_buffer_load_dwordx4 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 8; ++index) {
          if (!Expect(state.sgprs[24 + index] == 0x44440000u + index,
                      "expected s_buffer_load_dwordx8 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 16; ++index) {
          if (!Expect(state.sgprs[40 + index] == 0x55550000u + index,
                      "expected s_buffer_load_dwordx16 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        std::uint32_t value = 0;
        if (!Expect(memory.ReadU32(0x300u, &value),
                    "expected s_buffer_store_dword read") ||
            !Expect(value == 0x11110000u,
                    "expected s_buffer_store_dword result")) {
          std::cerr << mode << '\n';
          return false;
        }
        for (std::uint32_t index = 0; index < 2; ++index) {
          if (!Expect(memory.ReadU32(0x310u + index * 4u, &value),
                      "expected s_buffer_store_dwordx2 read") ||
              !Expect(value == 0x22220000u + index,
                      "expected s_buffer_store_dwordx2 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        for (std::uint32_t index = 0; index < 4; ++index) {
          if (!Expect(memory.ReadU32(0x320u + index * 4u, &value),
                      "expected s_buffer_store_dwordx4 read") ||
              !Expect(value == 0x33330000u + index,
                      "expected s_buffer_store_dwordx4 result")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
        }
        return true;
      };

  const std::vector<DecodedInstruction> scalar_buffer_memory_program = {
      DecodedInstruction::ThreeOperand("S_BUFFER_LOAD_DWORD",
                                       InstructionOperand::Sgpr(4),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0)),
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

  LinearExecutionMemory decoded_scalar_buffer_memory(0x700, 0);
  if (!Expect(seed_scalar_buffer_memory(&decoded_scalar_buffer_memory),
              "expected decoded scalar buffer seed writes")) {
    return 1;
  }
  WaveExecutionState decoded_scalar_buffer_memory_state =
      make_scalar_buffer_state();
  if (!Expect(interpreter.ExecuteProgram(scalar_buffer_memory_program,
                                         &decoded_scalar_buffer_memory_state,
                                         &decoded_scalar_buffer_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_scalar_buffer_state(decoded_scalar_buffer_memory_state,
                                    decoded_scalar_buffer_memory, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_scalar_buffer_memory_program;
  if (!Expect(interpreter.CompileProgram(scalar_buffer_memory_program,
                                         &compiled_scalar_buffer_memory_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  LinearExecutionMemory compiled_scalar_buffer_memory(0x700, 0);
  if (!Expect(seed_scalar_buffer_memory(&compiled_scalar_buffer_memory),
              "expected compiled scalar buffer seed writes")) {
    return 1;
  }
  WaveExecutionState compiled_scalar_buffer_memory_state =
      make_scalar_buffer_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_scalar_buffer_memory_program,
                                         &compiled_scalar_buffer_memory_state,
                                         &compiled_scalar_buffer_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_scalar_buffer_state(compiled_scalar_buffer_memory_state,
                                    compiled_scalar_buffer_memory,
                                    "compiled")) {
    return 1;
  }
  }

  {
  struct ScalarAtomicCase32 {
    std::string_view opcode;
    std::uint32_t offset;
    std::uint32_t data_value;
    std::uint32_t old_value;
    std::uint32_t new_value;
  };
  const std::array<ScalarAtomicCase32, 12> kScalarAtomicCases32 = {{
      {"S_ATOMIC_SWAP", 0x00u, 0xaaaa5555u, 0x11110000u, 0xaaaa5555u},
      {"S_ATOMIC_ADD", 0x04u, 4u, 9u, 13u},
      {"S_ATOMIC_SUB", 0x08u, 4u, 9u, 5u},
      {"S_ATOMIC_SMIN", 0x0cu, static_cast<std::uint32_t>(-3), 5u,
       static_cast<std::uint32_t>(-3)},
      {"S_ATOMIC_UMIN", 0x10u, 4u, 9u, 4u},
      {"S_ATOMIC_SMAX", 0x14u, 3u, static_cast<std::uint32_t>(-10), 3u},
      {"S_ATOMIC_UMAX", 0x18u, 9u, 2u, 9u},
      {"S_ATOMIC_AND", 0x1cu, 0x00000ff0u, 0x0000f0f0u, 0x000000f0u},
      {"S_ATOMIC_OR", 0x20u, 0x000000f1u, 0x0000f000u, 0x0000f0f1u},
      {"S_ATOMIC_XOR", 0x24u, 0x0f0f0f0fu, 0xf0f0f0f0u, 0xffffffffu},
      {"S_ATOMIC_INC", 0x28u, 5u, 3u, 4u},
      {"S_ATOMIC_DEC", 0x2cu, 9u, 0u, 9u},
  }};
  const std::array<std::string_view, 12> kScalarBufferAtomicCase32Opcodes = {
      "S_BUFFER_ATOMIC_SWAP", "S_BUFFER_ATOMIC_ADD",
      "S_BUFFER_ATOMIC_SUB",  "S_BUFFER_ATOMIC_SMIN",
      "S_BUFFER_ATOMIC_UMIN", "S_BUFFER_ATOMIC_SMAX",
      "S_BUFFER_ATOMIC_UMAX", "S_BUFFER_ATOMIC_AND",
      "S_BUFFER_ATOMIC_OR",   "S_BUFFER_ATOMIC_XOR",
      "S_BUFFER_ATOMIC_INC",  "S_BUFFER_ATOMIC_DEC",
  };

  struct ScalarAtomicCase64 {
    std::string_view opcode;
    std::uint32_t offset;
    std::uint64_t data_value;
    std::uint64_t old_value;
    std::uint64_t new_value;
  };
  const std::array<ScalarAtomicCase64, 12> kScalarAtomicCases64 = {{
      {"S_ATOMIC_SWAP_X2", 0x80u, 0xaaaabbbbccccddddULL,
       0x1111222233334444ULL, 0xaaaabbbbccccddddULL},
      {"S_ATOMIC_ADD_X2", 0x88u, 4u, 9u, 13u},
      {"S_ATOMIC_SUB_X2", 0x90u, 4u, 9u, 5u},
      {"S_ATOMIC_SMIN_X2", 0x98u, static_cast<std::uint64_t>(-3LL), 5u,
       static_cast<std::uint64_t>(-3LL)},
      {"S_ATOMIC_UMIN_X2", 0xa0u, 4u, 9u, 4u},
      {"S_ATOMIC_SMAX_X2", 0xa8u, 3u, static_cast<std::uint64_t>(-10LL), 3u},
      {"S_ATOMIC_UMAX_X2", 0xb0u, 9u, 2u, 9u},
      {"S_ATOMIC_AND_X2", 0xb8u, 0x0f0f0f0f0f0f0f0fULL,
       0xff00ff00ff00ff00ULL, 0x0f000f000f000f00ULL},
      {"S_ATOMIC_OR_X2", 0xc0u, 0x00000000000000f1ULL,
       0xf000000000000000ULL, 0xf0000000000000f1ULL},
      {"S_ATOMIC_XOR_X2", 0xc8u, 0x0f0f0f0f0f0f0f0fULL,
       0xf0f0f0f0f0f0f0f0ULL, 0xffffffffffffffffULL},
      {"S_ATOMIC_INC_X2", 0xd0u, 5u, 3u, 4u},
      {"S_ATOMIC_DEC_X2", 0xd8u, 9u, 0u, 9u},
  }};
  const std::array<std::string_view, 12> kScalarBufferAtomicCase64Opcodes = {
      "S_BUFFER_ATOMIC_SWAP_X2", "S_BUFFER_ATOMIC_ADD_X2",
      "S_BUFFER_ATOMIC_SUB_X2",  "S_BUFFER_ATOMIC_SMIN_X2",
      "S_BUFFER_ATOMIC_UMIN_X2", "S_BUFFER_ATOMIC_SMAX_X2",
      "S_BUFFER_ATOMIC_UMAX_X2", "S_BUFFER_ATOMIC_AND_X2",
      "S_BUFFER_ATOMIC_OR_X2",   "S_BUFFER_ATOMIC_XOR_X2",
      "S_BUFFER_ATOMIC_INC_X2",  "S_BUFFER_ATOMIC_DEC_X2",
  };

  const auto run_scalar_atomic_cases32 =
      [&](bool uses_buffer_descriptor,
          const auto& cases,
          const auto& opcodes,
          const char* mode) {
        WaveExecutionState state{};
        LinearExecutionMemory memory(0x400, 0);
        state.sgprs[0] = 0x100u;
        state.sgprs[1] = 0u;
        if (uses_buffer_descriptor) {
          state.sgprs[2] = 0x200u;
          state.sgprs[3] = 0u;
        }

        std::vector<DecodedInstruction> program;
        program.reserve(cases.size() + 1u);
        for (std::size_t index = 0; index < cases.size(); ++index) {
          const auto& atomic_case = cases[index];
          const std::uint16_t data_sgpr =
              static_cast<std::uint16_t>(4u + index);
          state.sgprs[data_sgpr] = atomic_case.data_value;
          if (!Expect(memory.WriteU32(0x100u + atomic_case.offset,
                                      atomic_case.old_value),
                      "expected scalar atomic seed write")) {
            std::cerr << mode << " " << atomic_case.opcode << '\n';
            return false;
          }
          program.push_back(DecodedInstruction::ThreeOperand(
              opcodes[index],
              InstructionOperand::Sgpr(data_sgpr), InstructionOperand::Sgpr(0),
              InstructionOperand::Imm32(atomic_case.offset)));
        }
        program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

        if (!Expect(interpreter.ExecuteProgram(program, &state, &memory,
                                               &error_message),
                    error_message.c_str()) ||
            !Expect(state.halted, "expected scalar atomic program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }

        for (std::size_t index = 0; index < cases.size(); ++index) {
          const auto& atomic_case = cases[index];
          const std::uint16_t data_sgpr =
              static_cast<std::uint16_t>(4u + index);
          std::uint32_t memory_value = 0;
          if (!Expect(state.sgprs[data_sgpr] == atomic_case.old_value,
                      "expected scalar atomic old value return") ||
              !Expect(memory.ReadU32(0x100u + atomic_case.offset, &memory_value),
                      "expected scalar atomic memory read") ||
              !Expect(memory_value == atomic_case.new_value,
                      "expected scalar atomic memory update")) {
            std::cerr << mode << " " << atomic_case.opcode << '\n';
            return false;
          }
        }
        return true;
      };

  const auto run_scalar_atomic_cases64 =
      [&](bool uses_buffer_descriptor,
          const auto& cases,
          const auto& opcodes,
          const char* mode) {
        WaveExecutionState state{};
        LinearExecutionMemory memory(0x400, 0);
        state.sgprs[0] = 0x100u;
        state.sgprs[1] = 0u;
        if (uses_buffer_descriptor) {
          state.sgprs[2] = 0x200u;
          state.sgprs[3] = 0u;
        }

        std::vector<DecodedInstruction> program;
        program.reserve(cases.size() + 1u);
        for (std::size_t index = 0; index < cases.size(); ++index) {
          const auto& atomic_case = cases[index];
          const std::uint16_t data_sgpr =
              static_cast<std::uint16_t>(32u + index * 2u);
          SplitU64(atomic_case.data_value, &state.sgprs[data_sgpr],
                   &state.sgprs[data_sgpr + 1u]);
          if (!Expect(WriteU64(&memory, 0x100u + atomic_case.offset,
                               atomic_case.old_value),
                      "expected scalar atomic x2 seed write")) {
            std::cerr << mode << " " << atomic_case.opcode << '\n';
            return false;
          }
          program.push_back(DecodedInstruction::ThreeOperand(
              opcodes[index],
              InstructionOperand::Sgpr(data_sgpr), InstructionOperand::Sgpr(0),
              InstructionOperand::Imm32(atomic_case.offset)));
        }
        program.push_back(DecodedInstruction::Nullary("S_ENDPGM"));

        if (!Expect(interpreter.ExecuteProgram(program, &state, &memory,
                                               &error_message),
                    error_message.c_str()) ||
            !Expect(state.halted, "expected scalar atomic x2 program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }

        for (std::size_t index = 0; index < cases.size(); ++index) {
          const auto& atomic_case = cases[index];
          const std::uint16_t data_sgpr =
              static_cast<std::uint16_t>(32u + index * 2u);
          std::uint64_t memory_value = 0;
          if (!Expect(ComposeU64(state.sgprs[data_sgpr],
                                 state.sgprs[data_sgpr + 1u]) ==
                          atomic_case.old_value,
                      "expected scalar atomic x2 old value return") ||
              !Expect(ReadU64(memory, 0x100u + atomic_case.offset, &memory_value),
                      "expected scalar atomic x2 memory read") ||
              !Expect(memory_value == atomic_case.new_value,
                      "expected scalar atomic x2 memory update")) {
            std::cerr << mode << " " << atomic_case.opcode << '\n';
            return false;
          }
        }
        return true;
      };

  if (!run_scalar_atomic_cases32(false, kScalarAtomicCases32,
                                 kScalarAtomic32Opcodes,
                                 "decoded scalar atomic") ||
      !run_scalar_atomic_cases32(true, kScalarAtomicCases32,
                                 kScalarBufferAtomicCase32Opcodes,
                                 "decoded scalar buffer atomic") ||
      !run_scalar_atomic_cases64(false, kScalarAtomicCases64,
                                 kScalarAtomic64Opcodes,
                                 "decoded scalar atomic x2") ||
      !run_scalar_atomic_cases64(true, kScalarAtomicCases64,
                                 kScalarBufferAtomicCase64Opcodes,
                                 "decoded scalar buffer atomic x2")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> scalar_maintenance_program = {
      DecodedInstruction::Nullary("S_DCACHE_INV"),
      DecodedInstruction::Nullary("S_DCACHE_WB"),
      DecodedInstruction::Nullary("S_DCACHE_INV_VOL"),
      DecodedInstruction::Nullary("S_DCACHE_WB_VOL"),
      DecodedInstruction::TwoOperand("S_DCACHE_DISCARD",
                                     InstructionOperand::Sgpr(0),
                                     InstructionOperand::Imm32(0x40)),
      DecodedInstruction::TwoOperand("S_DCACHE_DISCARD_X2",
                                     InstructionOperand::Sgpr(0),
                                     InstructionOperand::Sgpr(2)),
      DecodedInstruction::OneOperand("S_MEMTIME", InstructionOperand::Sgpr(4)),
      DecodedInstruction::OneOperand("S_MEMREALTIME", InstructionOperand::Sgpr(6)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  WaveExecutionState scalar_maintenance_state{};
  scalar_maintenance_state.sgprs[0] = 0x200u;
  scalar_maintenance_state.sgprs[1] = 0u;
  scalar_maintenance_state.sgprs[2] = 0x80u;
  if (!Expect(interpreter.ExecuteProgram(scalar_maintenance_program,
                                         &scalar_maintenance_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(scalar_maintenance_state.halted,
              "expected scalar maintenance program to halt") ||
      !Expect(scalar_maintenance_state.sgprs[0] == 0x200u,
              "expected scalar maintenance to preserve base") ||
      !Expect(scalar_maintenance_state.sgprs[2] == 0x80u,
              "expected scalar maintenance to preserve offset") ||
      !Expect(scalar_maintenance_state.sgprs[4] != 0u ||
                  scalar_maintenance_state.sgprs[5] != 0u,
              "expected s_memtime to write a timestamp") ||
      !Expect(scalar_maintenance_state.sgprs[6] != 0u ||
                  scalar_maintenance_state.sgprs[7] != 0u,
              "expected s_memrealtime to write a timestamp")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> scalar_probe_program = {
      DecodedInstruction::ThreeOperand("S_ATC_PROBE",
                                       InstructionOperand::Imm32(0x2a),
                                       InstructionOperand::Sgpr(0),
                                       InstructionOperand::Imm32(0x44)),
      DecodedInstruction::ThreeOperand("S_ATC_PROBE_BUFFER",
                                       InstructionOperand::Imm32(0x55),
                                       InstructionOperand::Sgpr(8),
                                       InstructionOperand::Sgpr(12)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  WaveExecutionState scalar_probe_state{};
  scalar_probe_state.sgprs[0] = 0x180u;
  scalar_probe_state.sgprs[1] = 0u;
  scalar_probe_state.sgprs[8] = 0x240u;
  scalar_probe_state.sgprs[9] = 0u;
  scalar_probe_state.sgprs[10] = 0x100u;
  scalar_probe_state.sgprs[11] = 0u;
  scalar_probe_state.sgprs[12] = 0x20u;
  if (!Expect(interpreter.ExecuteProgram(scalar_probe_program,
                                         &scalar_probe_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(scalar_probe_state.halted,
              "expected scalar probe program to halt") ||
      !Expect(scalar_probe_state.sgprs[0] == 0x180u,
              "expected s_atc_probe to preserve base") ||
      !Expect(scalar_probe_state.sgprs[8] == 0x240u,
              "expected s_atc_probe_buffer to preserve descriptor") ||
      !Expect(scalar_probe_state.sgprs[12] == 0x20u,
              "expected s_atc_probe_buffer to preserve soffset")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> buffer_maintenance_program = {
      DecodedInstruction::Nullary("BUFFER_WBL2"),
      DecodedInstruction::Nullary("BUFFER_INV"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  WaveExecutionState buffer_maintenance_state{};
  if (!Expect(interpreter.ExecuteProgram(buffer_maintenance_program,
                                         &buffer_maintenance_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(buffer_maintenance_state.halted,
              "expected buffer maintenance program to halt")) {
    return 1;
  }
  }

  {
  LinearExecutionMemory buffer_memory(0x500, 0);
  if (!Expect(buffer_memory.WriteU32(0x100u, 0x11111111u),
              "expected buffer load seed write") ||
      !Expect(buffer_memory.WriteU32(0x114u, 0x22220001u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x118u, 0x22220002u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x124u, 0x22220011u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x128u, 0x22220012u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x134u, 0x22220031u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x138u, 0x22220032u),
              "expected buffer loadx2 seed write") ||
      !Expect(buffer_memory.WriteU32(0x150u, 0x33330001u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x154u, 0x33330002u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x158u, 0x33330003u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x160u, 0x33330011u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x164u, 0x33330012u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x168u, 0x33330013u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x170u, 0x33330031u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x174u, 0x33330032u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x178u, 0x33330033u),
              "expected buffer loadx3 seed write") ||
      !Expect(buffer_memory.WriteU32(0x190u, 0x44440001u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x194u, 0x44440002u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x198u, 0x44440003u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x19cu, 0x44440004u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1a0u, 0x44440011u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1a4u, 0x44440012u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1a8u, 0x44440013u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1acu, 0x44440014u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1b0u, 0x44440031u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1b4u, 0x44440032u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1b8u, 0x44440033u),
              "expected buffer loadx4 seed write") ||
      !Expect(buffer_memory.WriteU32(0x1bcu, 0x44440034u),
              "expected buffer loadx4 seed write")) {
    return 1;
  }

  WaveExecutionState buffer_memory_state{};
  buffer_memory_state.exec_mask = 0b1011ULL;
  buffer_memory_state.sgprs[8] = 0x100u;
  buffer_memory_state.sgprs[9] = 0u;
  buffer_memory_state.sgprs[10] = 0x300u;
  buffer_memory_state.sgprs[11] = 0u;
  buffer_memory_state.sgprs[70] = 0x80u;
  buffer_memory_state.sgprs[71] = 0xf0u;
  buffer_memory_state.vgprs[2][0] = 0x10u;
  buffer_memory_state.vgprs[2][1] = 0x20u;
  buffer_memory_state.vgprs[2][3] = 0x30u;
  buffer_memory_state.vgprs[50][0] = 0x50000001u;
  buffer_memory_state.vgprs[50][1] = 0x50000002u;
  buffer_memory_state.vgprs[50][3] = 0x50000004u;
  buffer_memory_state.vgprs[52][0] = 0x52000001u;
  buffer_memory_state.vgprs[53][0] = 0x53000001u;
  buffer_memory_state.vgprs[52][1] = 0x52000002u;
  buffer_memory_state.vgprs[53][1] = 0x53000002u;
  buffer_memory_state.vgprs[52][3] = 0x52000004u;
  buffer_memory_state.vgprs[53][3] = 0x53000004u;
  buffer_memory_state.vgprs[56][0] = 0x56000001u;
  buffer_memory_state.vgprs[57][0] = 0x57000001u;
  buffer_memory_state.vgprs[58][0] = 0x58000001u;
  buffer_memory_state.vgprs[56][1] = 0x56000002u;
  buffer_memory_state.vgprs[57][1] = 0x57000002u;
  buffer_memory_state.vgprs[58][1] = 0x58000002u;
  buffer_memory_state.vgprs[56][3] = 0x56000004u;
  buffer_memory_state.vgprs[57][3] = 0x57000004u;
  buffer_memory_state.vgprs[58][3] = 0x58000004u;
  buffer_memory_state.vgprs[60][0] = 0x60000001u;
  buffer_memory_state.vgprs[61][0] = 0x61000001u;
  buffer_memory_state.vgprs[62][0] = 0x62000001u;
  buffer_memory_state.vgprs[63][0] = 0x63000001u;
  buffer_memory_state.vgprs[60][1] = 0x60000002u;
  buffer_memory_state.vgprs[61][1] = 0x61000002u;
  buffer_memory_state.vgprs[62][1] = 0x62000002u;
  buffer_memory_state.vgprs[63][1] = 0x63000002u;
  buffer_memory_state.vgprs[60][3] = 0x60000004u;
  buffer_memory_state.vgprs[61][3] = 0x61000004u;
  buffer_memory_state.vgprs[62][3] = 0x62000004u;
  buffer_memory_state.vgprs[63][3] = 0x63000004u;

  const std::vector<DecodedInstruction> buffer_memory_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORD",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORDX2",
                                      InstructionOperand::Vgpr(24),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(4)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORDX3",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x40)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_DWORDX4",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Sgpr(70),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORD",
                                      InstructionOperand::Vgpr(50),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0xc0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORDX2",
                                      InstructionOperand::Vgpr(52),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0xc0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORDX3",
                                      InstructionOperand::Vgpr(56),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0xe0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_DWORDX4",
                                      InstructionOperand::Vgpr(60),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Sgpr(71),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(buffer_memory_program,
                                         &buffer_memory_state, &buffer_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(buffer_memory_state.halted,
              "expected buffer memory program to halt") ||
      !Expect(buffer_memory_state.vgprs[20][0] == 0x11111111u,
              "expected buffer load lane 0 result") ||
      !Expect(buffer_memory_state.vgprs[20][1] == 0x11111111u,
              "expected buffer load lane 1 result") ||
      !Expect(buffer_memory_state.vgprs[20][2] == 0u,
              "expected inactive buffer load lane to remain untouched") ||
      !Expect(buffer_memory_state.vgprs[20][3] == 0x11111111u,
              "expected buffer load lane 3 result") ||
      !Expect(buffer_memory_state.vgprs[24][0] == 0x22220001u &&
                  buffer_memory_state.vgprs[25][0] == 0x22220002u,
              "expected buffer loadx2 lane 0 result") ||
      !Expect(buffer_memory_state.vgprs[24][1] == 0x22220011u &&
                  buffer_memory_state.vgprs[25][1] == 0x22220012u,
              "expected buffer loadx2 lane 1 result") ||
      !Expect(buffer_memory_state.vgprs[24][3] == 0x22220031u &&
                  buffer_memory_state.vgprs[25][3] == 0x22220032u,
              "expected buffer loadx2 lane 3 result") ||
      !Expect(buffer_memory_state.vgprs[30][1] == 0x33330011u &&
                  buffer_memory_state.vgprs[31][1] == 0x33330012u &&
                  buffer_memory_state.vgprs[32][1] == 0x33330013u,
              "expected buffer loadx3 lane 1 result") ||
      !Expect(buffer_memory_state.vgprs[40][3] == 0x44440031u &&
                  buffer_memory_state.vgprs[41][3] == 0x44440032u &&
                  buffer_memory_state.vgprs[42][3] == 0x44440033u &&
                  buffer_memory_state.vgprs[43][3] == 0x44440034u,
              "expected buffer loadx4 lane 3 result")) {
    return 1;
  }

  std::uint32_t value = 0;
  if (!Expect(buffer_memory.ReadU32(0x1c0u, &value),
              "expected buffer store read") ||
      !Expect(value == 0x50000004u,
              "expected scalar-address buffer store last-lane result") ||
      !Expect(buffer_memory.ReadU32(0x1d0u, &value),
              "expected buffer storex2 read") ||
      !Expect(value == 0x52000001u, "expected buffer storex2 lane 0 result") ||
      !Expect(buffer_memory.ReadU32(0x1d4u, &value),
              "expected buffer storex2 read") ||
      !Expect(value == 0x53000001u, "expected buffer storex2 lane 0 result") ||
      !Expect(buffer_memory.ReadU32(0x1e0u, &value),
              "expected buffer storex2 read") ||
      !Expect(value == 0x52000002u, "expected buffer storex2 lane 1 result") ||
      !Expect(buffer_memory.ReadU32(0x1f0u, &value),
              "expected buffer storex3 read") ||
      !Expect(value == 0x56000001u, "expected buffer storex3 lane 0 result") ||
      !Expect(buffer_memory.ReadU32(0x1f8u, &value),
              "expected buffer storex3 read") ||
      !Expect(value == 0x58000001u, "expected buffer storex3 lane 0 result") ||
      !Expect(buffer_memory.ReadU32(0x200u, &value),
              "expected buffer storex4 read") ||
      !Expect(value == 0x60000001u, "expected buffer storex4 lane 0 result") ||
      !Expect(buffer_memory.ReadU32(0x22cu, &value),
              "expected buffer storex4 read") ||
      !Expect(value == 0x63000004u, "expected buffer storex4 lane 3 result")) {
    return 1;
  }
  }

  {
  LinearExecutionMemory buffer_subword_memory(0x500, 0);
  if (!Expect(WriteU8(&buffer_subword_memory, 0x100u, 0x7au),
              "expected buffer ubyte seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x110u, 0x80u),
              "expected buffer sbyte seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x120u, 0x7fu),
              "expected buffer sbyte seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x130u, 0xfeu),
              "expected buffer sbyte seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x150u, 0x1234u),
              "expected buffer ushort seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x160u, 0x5678u),
              "expected buffer ushort seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x170u, 0xabcdu),
              "expected buffer ushort seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x190u, 0x8001u),
              "expected buffer sshort seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x1a0u, 0x7fffu),
              "expected buffer sshort seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x1b0u, 0xff00u),
              "expected buffer sshort seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x1d0u, 0x34u),
              "expected buffer ubyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x1e0u, 0x56u),
              "expected buffer ubyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x1f0u, 0x78u),
              "expected buffer ubyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x240u, 0x9au),
              "expected buffer ubyte d16 hi seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x250u, 0xbcu),
              "expected buffer ubyte d16 hi seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x260u, 0xdeu),
              "expected buffer ubyte d16 hi seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x270u, 0x81u),
              "expected buffer sbyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x280u, 0x82u),
              "expected buffer sbyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x290u, 0x7fu),
              "expected buffer sbyte d16 seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x2b0u, 0xfeu),
              "expected buffer sbyte d16 hi seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x2c0u, 0x80u),
              "expected buffer sbyte d16 hi seed write") ||
      !Expect(WriteU8(&buffer_subword_memory, 0x2d0u, 0x01u),
              "expected buffer sbyte d16 hi seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x2f0u, 0x1122u),
              "expected buffer short d16 seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x300u, 0x3344u),
              "expected buffer short d16 seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x310u, 0x5566u),
              "expected buffer short d16 seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x340u, 0x89abu),
              "expected buffer short d16 hi seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x350u, 0xcdefu),
              "expected buffer short d16 hi seed write") ||
      !Expect(WriteU16(&buffer_subword_memory, 0x360u, 0x1357u),
              "expected buffer short d16 hi seed write")) {
    return 1;
  }

  WaveExecutionState buffer_subword_state{};
  buffer_subword_state.exec_mask = 0b1011ULL;
  buffer_subword_state.sgprs[8] = 0x100u;
  buffer_subword_state.sgprs[9] = 0u;
  buffer_subword_state.sgprs[10] = 0x400u;
  buffer_subword_state.sgprs[11] = 0u;
  buffer_subword_state.vgprs[2][0] = 0x10u;
  buffer_subword_state.vgprs[2][1] = 0x20u;
  buffer_subword_state.vgprs[2][3] = 0x30u;
  buffer_subword_state.vgprs[40][0] = 0x111111abu;
  buffer_subword_state.vgprs[40][1] = 0x222222bcu;
  buffer_subword_state.vgprs[40][3] = 0x333333deu;
  buffer_subword_state.vgprs[41][0] = 0x00340000u;
  buffer_subword_state.vgprs[41][1] = 0x00ab0000u;
  buffer_subword_state.vgprs[41][3] = 0x00fe0000u;
  buffer_subword_state.vgprs[42][0] = 0x000089abu;
  buffer_subword_state.vgprs[42][1] = 0x0000cdefu;
  buffer_subword_state.vgprs[42][3] = 0x00001357u;
  buffer_subword_state.vgprs[43][0] = 0x24680000u;
  buffer_subword_state.vgprs[43][1] = 0x13570000u;
  buffer_subword_state.vgprs[43][3] = 0xabcd0000u;

  const std::vector<DecodedInstruction> buffer_subword_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_UBYTE",
                                      InstructionOperand::Vgpr(60),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SBYTE",
                                      InstructionOperand::Vgpr(61),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_USHORT",
                                      InstructionOperand::Vgpr(62),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x40)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SSHORT",
                                      InstructionOperand::Vgpr(63),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x80)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_UBYTE_D16",
                                      InstructionOperand::Vgpr(64),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0xc0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_UBYTE_D16_HI",
                                      InstructionOperand::Vgpr(65),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x130)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SBYTE_D16",
                                      InstructionOperand::Vgpr(66),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x160)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SBYTE_D16_HI",
                                      InstructionOperand::Vgpr(67),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x1a0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SHORT_D16",
                                      InstructionOperand::Vgpr(68),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x1e0)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_SHORT_D16_HI",
                                      InstructionOperand::Vgpr(69),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x230)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_BYTE",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x270)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_BYTE_D16_HI",
                                      InstructionOperand::Vgpr(41),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x2a0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_SHORT",
                                      InstructionOperand::Vgpr(42),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x2e0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_SHORT_D16_HI",
                                      InstructionOperand::Vgpr(43),
                                      InstructionOperand::Vgpr(2),
                                      InstructionOperand::Sgpr(8),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x320)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(buffer_subword_program,
                                         &buffer_subword_state,
                                         &buffer_subword_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(buffer_subword_state.halted,
              "expected buffer subword program to halt") ||
      !Expect(buffer_subword_state.vgprs[60][0] == 0x7au &&
                  buffer_subword_state.vgprs[60][1] == 0x7au &&
                  buffer_subword_state.vgprs[60][3] == 0x7au,
              "expected buffer ubyte load result") ||
      !Expect(buffer_subword_state.vgprs[61][0] == 0xffffff80u &&
                  buffer_subword_state.vgprs[61][1] == 0x0000007fu &&
                  buffer_subword_state.vgprs[61][3] == 0xfffffffeu,
              "expected buffer sbyte load result") ||
      !Expect(buffer_subword_state.vgprs[62][1] == 0x5678u &&
                  buffer_subword_state.vgprs[62][3] == 0xabcdu,
              "expected buffer ushort load result") ||
      !Expect(buffer_subword_state.vgprs[63][0] == 0xffff8001u &&
                  buffer_subword_state.vgprs[63][1] == 0x00007fffu &&
                  buffer_subword_state.vgprs[63][3] == 0xffffff00u,
              "expected buffer sshort load result") ||
      !Expect(buffer_subword_state.vgprs[64][0] == 0x00000034u &&
                  buffer_subword_state.vgprs[64][1] == 0x00000056u &&
                  buffer_subword_state.vgprs[64][3] == 0x00000078u,
              "expected buffer ubyte d16 load result") ||
      !Expect(buffer_subword_state.vgprs[65][0] == 0x009a0000u &&
                  buffer_subword_state.vgprs[65][1] == 0x00bc0000u &&
                  buffer_subword_state.vgprs[65][3] == 0x00de0000u,
              "expected buffer ubyte d16 hi load result") ||
      !Expect(buffer_subword_state.vgprs[66][0] == 0x0000ff81u &&
                  buffer_subword_state.vgprs[66][1] == 0x0000ff82u &&
                  buffer_subword_state.vgprs[66][3] == 0x0000007fu,
              "expected buffer sbyte d16 load result") ||
      !Expect(buffer_subword_state.vgprs[67][0] == 0xfffe0000u &&
                  buffer_subword_state.vgprs[67][1] == 0xff800000u &&
                  buffer_subword_state.vgprs[67][3] == 0x00010000u,
              "expected buffer sbyte d16 hi load result") ||
      !Expect(buffer_subword_state.vgprs[68][1] == 0x00003344u &&
                  buffer_subword_state.vgprs[68][3] == 0x00005566u,
              "expected buffer short d16 load result") ||
      !Expect(buffer_subword_state.vgprs[69][0] == 0x89ab0000u &&
                  buffer_subword_state.vgprs[69][1] == 0xcdef0000u &&
                  buffer_subword_state.vgprs[69][3] == 0x13570000u,
              "expected buffer short d16 hi load result")) {
    return 1;
  }

  std::uint8_t stored_byte = 0;
  std::uint16_t stored_short = 0;
  if (!Expect(ReadU8(buffer_subword_memory, 0x380u, &stored_byte),
              "expected buffer byte store read") ||
      !Expect(stored_byte == 0xabu, "expected buffer byte store lane 0 result") ||
      !Expect(ReadU8(buffer_subword_memory, 0x390u, &stored_byte),
              "expected buffer byte store read") ||
      !Expect(stored_byte == 0xbcu, "expected buffer byte store lane 1 result") ||
      !Expect(ReadU8(buffer_subword_memory, 0x3a0u, &stored_byte),
              "expected buffer byte store read") ||
      !Expect(stored_byte == 0xdeu, "expected buffer byte store lane 3 result") ||
      !Expect(ReadU8(buffer_subword_memory, 0x3b0u, &stored_byte),
              "expected buffer byte d16 hi store read") ||
      !Expect(stored_byte == 0x34u,
              "expected buffer byte d16 hi store lane 0 result") ||
      !Expect(ReadU8(buffer_subword_memory, 0x3c0u, &stored_byte),
              "expected buffer byte d16 hi store read") ||
      !Expect(stored_byte == 0xabu,
              "expected buffer byte d16 hi store lane 1 result") ||
      !Expect(ReadU8(buffer_subword_memory, 0x3d0u, &stored_byte),
              "expected buffer byte d16 hi store read") ||
      !Expect(stored_byte == 0xfeu,
              "expected buffer byte d16 hi store lane 3 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x3f0u, &stored_short),
              "expected buffer short store read") ||
      !Expect(stored_short == 0x89abu,
              "expected buffer short store lane 0 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x400u, &stored_short),
              "expected buffer short store read") ||
      !Expect(stored_short == 0xcdefu,
              "expected buffer short store lane 1 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x410u, &stored_short),
              "expected buffer short store read") ||
      !Expect(stored_short == 0x1357u,
              "expected buffer short store lane 3 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x430u, &stored_short),
              "expected buffer short d16 hi store read") ||
      !Expect(stored_short == 0x2468u,
              "expected buffer short d16 hi store lane 0 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x440u, &stored_short),
              "expected buffer short d16 hi store read") ||
      !Expect(stored_short == 0x1357u,
              "expected buffer short d16 hi store lane 1 result") ||
      !Expect(ReadU16(buffer_subword_memory, 0x450u, &stored_short),
              "expected buffer short d16 hi store read") ||
      !Expect(stored_short == 0xabcdu,
              "expected buffer short d16 hi store lane 3 result")) {
    return 1;
  }
  }

  {
  auto make_buffer_format_descriptor_word3 = [](std::uint32_t data_format,
                                                std::uint32_t num_format) {
    return (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9) |
           (data_format << 12) | (num_format << 19);
  };

  LinearExecutionMemory buffer_format_memory(0x400, 0);
  if (!Expect(WriteU8(&buffer_format_memory, 0x100u, 0x7au),
              "expected buffer format x seed write") ||
      !Expect(WriteU8(&buffer_format_memory, 0x140u, 0x01u),
              "expected buffer format xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_memory, 0x141u, 0x02u),
              "expected buffer format xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_memory, 0x142u, 0x03u),
              "expected buffer format xyzw seed write") ||
      !Expect(WriteU8(&buffer_format_memory, 0x143u, 0x04u),
              "expected buffer format xyzw seed write") ||
      !Expect(buffer_format_memory.WriteU32(0x180u, 0x3fc00000u),
              "expected buffer format xyz seed write") ||
      !Expect(buffer_format_memory.WriteU32(0x184u, 0xc0000000u),
              "expected buffer format xyz seed write") ||
      !Expect(buffer_format_memory.WriteU32(0x188u, 0x3e800000u),
              "expected buffer format xyz seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x1c0u, 0x0011u),
              "expected buffer format d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x1c2u, 0x0022u),
              "expected buffer format d16 xy seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x200u, 0x3c00u),
              "expected buffer format d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x202u, 0x4000u),
              "expected buffer format d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x204u, 0xc000u),
              "expected buffer format d16 xyz seed write") ||
      !Expect(WriteU16(&buffer_format_memory, 0x240u, 0x3344u),
              "expected buffer format d16 hi seed write")) {
    return 1;
  }

  WaveExecutionState buffer_format_state{};
  buffer_format_state.exec_mask = 0x1ULL;
  buffer_format_state.sgprs[20] = 0x100u;
  buffer_format_state.sgprs[21] = 0u;
  buffer_format_state.sgprs[22] = 0x80u;
  buffer_format_state.sgprs[23] =
      make_buffer_format_descriptor_word3(1u, 4u);
  buffer_format_state.sgprs[24] = 0x140u;
  buffer_format_state.sgprs[25] = 0u;
  buffer_format_state.sgprs[26] = 0x80u;
  buffer_format_state.sgprs[27] =
      make_buffer_format_descriptor_word3(10u, 4u);
  buffer_format_state.sgprs[28] = 0x180u;
  buffer_format_state.sgprs[29] = 0u;
  buffer_format_state.sgprs[30] = 0x80u;
  buffer_format_state.sgprs[31] =
      make_buffer_format_descriptor_word3(13u, 7u);
  buffer_format_state.sgprs[32] = 0x1c0u;
  buffer_format_state.sgprs[33] = 0u;
  buffer_format_state.sgprs[34] = 0x80u;
  buffer_format_state.sgprs[35] =
      make_buffer_format_descriptor_word3(5u, 4u);
  buffer_format_state.sgprs[36] = 0x200u;
  buffer_format_state.sgprs[37] = 0u;
  buffer_format_state.sgprs[38] = 0x80u;
  buffer_format_state.sgprs[39] =
      make_buffer_format_descriptor_word3(12u, 7u);
  buffer_format_state.sgprs[40] = 0x240u;
  buffer_format_state.sgprs[41] = 0u;
  buffer_format_state.sgprs[42] = 0x80u;
  buffer_format_state.sgprs[43] =
      make_buffer_format_descriptor_word3(2u, 4u);
  buffer_format_state.vgprs[30][0] = 0x55u;
  buffer_format_state.vgprs[40][0] = 0x05u;
  buffer_format_state.vgprs[41][0] = 0x06u;
  buffer_format_state.vgprs[42][0] = 0x07u;
  buffer_format_state.vgprs[43][0] = 0x08u;
  buffer_format_state.vgprs[50][0] = 0x40200000u;
  buffer_format_state.vgprs[51][0] = 0xbf800000u;
  buffer_format_state.vgprs[52][0] = 0x40800000u;
  buffer_format_state.vgprs[60][0] = 0x00220011u;
  buffer_format_state.vgprs[61][0] = 0x40003c00u;
  buffer_format_state.vgprs[62][0] = 0x0000c000u;
  buffer_format_state.vgprs[63][0] = 0x33440000u;

  const std::vector<DecodedInstruction> buffer_format_program = {
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_X",
                                      InstructionOperand::Vgpr(70),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_X",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(20),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(71),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XYZW",
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(24),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_XYZ",
                                      InstructionOperand::Vgpr(75),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(28),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_XYZ",
                                      InstructionOperand::Vgpr(50),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(28),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XY",
                                      InstructionOperand::Vgpr(80),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XY",
                                      InstructionOperand::Vgpr(60),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(32),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_XYZ",
                                      InstructionOperand::Vgpr(81),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(36),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_XYZ",
                                      InstructionOperand::Vgpr(61),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(36),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::FiveOperand("BUFFER_LOAD_FORMAT_D16_HI_X",
                                      InstructionOperand::Vgpr(83),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(40),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("BUFFER_STORE_FORMAT_D16_HI_X",
                                      InstructionOperand::Vgpr(63),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Sgpr(40),
                                      InstructionOperand::Imm32(0),
                                      InstructionOperand::Imm32(0x20)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(buffer_format_program,
                                         &buffer_format_state,
                                         &buffer_format_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(buffer_format_state.halted,
              "expected buffer format program to halt") ||
      !Expect(buffer_format_state.vgprs[70][0] == 0x7au,
              "expected buffer format x load result") ||
      !Expect(buffer_format_state.vgprs[71][0] == 0x01u &&
                  buffer_format_state.vgprs[72][0] == 0x02u &&
                  buffer_format_state.vgprs[73][0] == 0x03u &&
                  buffer_format_state.vgprs[74][0] == 0x04u,
              "expected buffer format xyzw load result") ||
      !Expect(buffer_format_state.vgprs[75][0] == 0x3fc00000u &&
                  buffer_format_state.vgprs[76][0] == 0xc0000000u &&
                  buffer_format_state.vgprs[77][0] == 0x3e800000u,
              "expected buffer format xyz load result") ||
      !Expect(buffer_format_state.vgprs[80][0] == 0x00220011u,
              "expected buffer format d16 xy load result") ||
      !Expect(buffer_format_state.vgprs[81][0] == 0x40003c00u &&
                  buffer_format_state.vgprs[82][0] == 0x0000c000u,
              "expected buffer format d16 xyz load result") ||
      !Expect(buffer_format_state.vgprs[83][0] == 0x33440000u,
              "expected buffer format d16 hi load result")) {
    return 1;
  }

  std::uint8_t buffer_format_byte = 0;
  std::uint16_t buffer_format_short = 0;
  std::uint32_t buffer_format_dword = 0;
  if (!Expect(ReadU8(buffer_format_memory, 0x120u, &buffer_format_byte),
              "expected buffer format x store read") ||
      !Expect(buffer_format_byte == 0x55u,
              "expected buffer format x store result") ||
      !Expect(ReadU8(buffer_format_memory, 0x160u, &buffer_format_byte),
              "expected buffer format xyzw store read") ||
      !Expect(buffer_format_byte == 0x05u,
              "expected buffer format xyzw store result") ||
      !Expect(ReadU8(buffer_format_memory, 0x161u, &buffer_format_byte),
              "expected buffer format xyzw store read") ||
      !Expect(buffer_format_byte == 0x06u,
              "expected buffer format xyzw store result") ||
      !Expect(ReadU8(buffer_format_memory, 0x162u, &buffer_format_byte),
              "expected buffer format xyzw store read") ||
      !Expect(buffer_format_byte == 0x07u,
              "expected buffer format xyzw store result") ||
      !Expect(ReadU8(buffer_format_memory, 0x163u, &buffer_format_byte),
              "expected buffer format xyzw store read") ||
      !Expect(buffer_format_byte == 0x08u,
              "expected buffer format xyzw store result") ||
      !Expect(buffer_format_memory.ReadU32(0x1a0u, &buffer_format_dword),
              "expected buffer format xyz store read") ||
      !Expect(buffer_format_dword == 0x40200000u,
              "expected buffer format xyz store result") ||
      !Expect(buffer_format_memory.ReadU32(0x1a4u, &buffer_format_dword),
              "expected buffer format xyz store read") ||
      !Expect(buffer_format_dword == 0xbf800000u,
              "expected buffer format xyz store result") ||
      !Expect(buffer_format_memory.ReadU32(0x1a8u, &buffer_format_dword),
              "expected buffer format xyz store read") ||
      !Expect(buffer_format_dword == 0x40800000u,
              "expected buffer format xyz store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x1e0u, &buffer_format_short),
              "expected buffer format d16 xy store read") ||
      !Expect(buffer_format_short == 0x0011u,
              "expected buffer format d16 xy store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x1e2u, &buffer_format_short),
              "expected buffer format d16 xy store read") ||
      !Expect(buffer_format_short == 0x0022u,
              "expected buffer format d16 xy store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x220u, &buffer_format_short),
              "expected buffer format d16 xyz store read") ||
      !Expect(buffer_format_short == 0x3c00u,
              "expected buffer format d16 xyz store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x222u, &buffer_format_short),
              "expected buffer format d16 xyz store read") ||
      !Expect(buffer_format_short == 0x4000u,
              "expected buffer format d16 xyz store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x224u, &buffer_format_short),
              "expected buffer format d16 xyz store read") ||
      !Expect(buffer_format_short == 0xc000u,
              "expected buffer format d16 xyz store result") ||
      !Expect(ReadU16(buffer_format_memory, 0x260u, &buffer_format_short),
              "expected buffer format d16 hi store read") ||
      !Expect(buffer_format_short == 0x3344u,
              "expected buffer format d16 hi store result")) {
    return 1;
  }
  }

  {
  const std::uint32_t typed_buffer_dst_sel_word3 =
      (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9);

  LinearExecutionMemory typed_buffer_memory(0x500, 0);
  if (!Expect(WriteU8(&typed_buffer_memory, 0x280u, 0x11u),
              "expected typed buffer xyzw seed write") ||
      !Expect(WriteU8(&typed_buffer_memory, 0x281u, 0x22u),
              "expected typed buffer xyzw seed write") ||
      !Expect(WriteU8(&typed_buffer_memory, 0x282u, 0x33u),
              "expected typed buffer xyzw seed write") ||
      !Expect(WriteU8(&typed_buffer_memory, 0x283u, 0x44u),
              "expected typed buffer xyzw seed write") ||
      !Expect(typed_buffer_memory.WriteU32(0x2c0u, 0x3f800000u),
              "expected typed buffer xyz seed write") ||
      !Expect(typed_buffer_memory.WriteU32(0x2c4u, 0xbf400000u),
              "expected typed buffer xyz seed write") ||
      !Expect(typed_buffer_memory.WriteU32(0x2c8u, 0x40400000u),
              "expected typed buffer xyz seed write") ||
      !Expect(WriteU16(&typed_buffer_memory, 0x300u, 0x0102u),
              "expected typed buffer d16 xy seed write") ||
      !Expect(WriteU16(&typed_buffer_memory, 0x302u, 0x0304u),
              "expected typed buffer d16 xy seed write") ||
      !Expect(WriteU16(&typed_buffer_memory, 0x340u, 0x3800u),
              "expected typed buffer d16 xyz seed write") ||
      !Expect(WriteU16(&typed_buffer_memory, 0x342u, 0x3c00u),
              "expected typed buffer d16 xyz seed write") ||
      !Expect(WriteU16(&typed_buffer_memory, 0x344u, 0x4000u),
              "expected typed buffer d16 xyz seed write")) {
    return 1;
  }

  WaveExecutionState typed_buffer_state{};
  typed_buffer_state.exec_mask = 0x1ULL;
  typed_buffer_state.sgprs[44] = 0x280u;
  typed_buffer_state.sgprs[45] = 0u;
  typed_buffer_state.sgprs[46] = 0x80u;
  typed_buffer_state.sgprs[47] = typed_buffer_dst_sel_word3;
  typed_buffer_state.sgprs[48] = 0x2c0u;
  typed_buffer_state.sgprs[49] = 0u;
  typed_buffer_state.sgprs[50] = 0x80u;
  typed_buffer_state.sgprs[51] = typed_buffer_dst_sel_word3;
  typed_buffer_state.sgprs[52] = 0x300u;
  typed_buffer_state.sgprs[53] = 0u;
  typed_buffer_state.sgprs[54] = 0x80u;
  typed_buffer_state.sgprs[55] = typed_buffer_dst_sel_word3;
  typed_buffer_state.sgprs[56] = 0x340u;
  typed_buffer_state.sgprs[57] = 0u;
  typed_buffer_state.sgprs[58] = 0x80u;
  typed_buffer_state.sgprs[59] = typed_buffer_dst_sel_word3;
  typed_buffer_state.vgprs[100][0] = 0xa1u;
  typed_buffer_state.vgprs[101][0] = 0xb2u;
  typed_buffer_state.vgprs[102][0] = 0xc3u;
  typed_buffer_state.vgprs[103][0] = 0xd4u;
  typed_buffer_state.vgprs[110][0] = 0x3f000000u;
  typed_buffer_state.vgprs[111][0] = 0xbf800000u;
  typed_buffer_state.vgprs[112][0] = 0x40200000u;
  typed_buffer_state.vgprs[120][0] = 0x03040102u;
  typed_buffer_state.vgprs[121][0] = 0x3c003800u;
  typed_buffer_state.vgprs[122][0] = 0x00004000u;

  const std::vector<DecodedInstruction> typed_buffer_program = {
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_XYZW",
                                       InstructionOperand::Vgpr(90),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(44),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(10),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_XYZW",
                                       InstructionOperand::Vgpr(100),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(44),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(10),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_XYZ",
                                       InstructionOperand::Vgpr(94),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(48),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(13),
                                       InstructionOperand::Imm32(7)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_XYZ",
                                       InstructionOperand::Vgpr(110),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(48),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(13),
                                       InstructionOperand::Imm32(7)),
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_D16_XY",
                                       InstructionOperand::Vgpr(97),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(52),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(5),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_D16_XY",
                                       InstructionOperand::Vgpr(120),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(52),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(5),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::SevenOperand("TBUFFER_LOAD_FORMAT_D16_XYZ",
                                       InstructionOperand::Vgpr(98),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(56),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(12),
                                       InstructionOperand::Imm32(7)),
      DecodedInstruction::SevenOperand("TBUFFER_STORE_FORMAT_D16_XYZ",
                                       InstructionOperand::Vgpr(121),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Sgpr(56),
                                       InstructionOperand::Imm32(0),
                                       InstructionOperand::Imm32(0x20),
                                       InstructionOperand::Imm32(12),
                                       InstructionOperand::Imm32(7)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(typed_buffer_program,
                                         &typed_buffer_state,
                                         &typed_buffer_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(typed_buffer_state.halted,
              "expected typed buffer program to halt") ||
      !Expect(typed_buffer_state.vgprs[90][0] == 0x11u &&
                  typed_buffer_state.vgprs[91][0] == 0x22u &&
                  typed_buffer_state.vgprs[92][0] == 0x33u &&
                  typed_buffer_state.vgprs[93][0] == 0x44u,
              "expected typed buffer xyzw load result") ||
      !Expect(typed_buffer_state.vgprs[94][0] == 0x3f800000u &&
                  typed_buffer_state.vgprs[95][0] == 0xbf400000u &&
                  typed_buffer_state.vgprs[96][0] == 0x40400000u,
              "expected typed buffer xyz load result") ||
      !Expect(typed_buffer_state.vgprs[97][0] == 0x03040102u,
              "expected typed buffer d16 xy load result") ||
      !Expect(typed_buffer_state.vgprs[98][0] == 0x3c003800u &&
                  typed_buffer_state.vgprs[99][0] == 0x00004000u,
              "expected typed buffer d16 xyz load result")) {
    return 1;
  }

  std::uint8_t typed_buffer_byte = 0;
  std::uint16_t typed_buffer_short = 0;
  std::uint32_t typed_buffer_dword = 0;
  if (!Expect(ReadU8(typed_buffer_memory, 0x2a0u, &typed_buffer_byte),
              "expected typed buffer xyzw store read") ||
      !Expect(typed_buffer_byte == 0xa1u,
              "expected typed buffer xyzw store result") ||
      !Expect(ReadU8(typed_buffer_memory, 0x2a1u, &typed_buffer_byte),
              "expected typed buffer xyzw store read") ||
      !Expect(typed_buffer_byte == 0xb2u,
              "expected typed buffer xyzw store result") ||
      !Expect(ReadU8(typed_buffer_memory, 0x2a2u, &typed_buffer_byte),
              "expected typed buffer xyzw store read") ||
      !Expect(typed_buffer_byte == 0xc3u,
              "expected typed buffer xyzw store result") ||
      !Expect(ReadU8(typed_buffer_memory, 0x2a3u, &typed_buffer_byte),
              "expected typed buffer xyzw store read") ||
      !Expect(typed_buffer_byte == 0xd4u,
              "expected typed buffer xyzw store result") ||
      !Expect(typed_buffer_memory.ReadU32(0x2e0u, &typed_buffer_dword),
              "expected typed buffer xyz store read") ||
      !Expect(typed_buffer_dword == 0x3f000000u,
              "expected typed buffer xyz store result") ||
      !Expect(typed_buffer_memory.ReadU32(0x2e4u, &typed_buffer_dword),
              "expected typed buffer xyz store read") ||
      !Expect(typed_buffer_dword == 0xbf800000u,
              "expected typed buffer xyz store result") ||
      !Expect(typed_buffer_memory.ReadU32(0x2e8u, &typed_buffer_dword),
              "expected typed buffer xyz store read") ||
      !Expect(typed_buffer_dword == 0x40200000u,
              "expected typed buffer xyz store result") ||
      !Expect(ReadU16(typed_buffer_memory, 0x320u, &typed_buffer_short),
              "expected typed buffer d16 xy store read") ||
      !Expect(typed_buffer_short == 0x0102u,
              "expected typed buffer d16 xy store result") ||
      !Expect(ReadU16(typed_buffer_memory, 0x322u, &typed_buffer_short),
              "expected typed buffer d16 xy store read") ||
      !Expect(typed_buffer_short == 0x0304u,
              "expected typed buffer d16 xy store result") ||
      !Expect(ReadU16(typed_buffer_memory, 0x360u, &typed_buffer_short),
              "expected typed buffer d16 xyz store read") ||
      !Expect(typed_buffer_short == 0x3800u,
              "expected typed buffer d16 xyz store result") ||
      !Expect(ReadU16(typed_buffer_memory, 0x362u, &typed_buffer_short),
              "expected typed buffer d16 xyz store read") ||
      !Expect(typed_buffer_short == 0x3c00u,
              "expected typed buffer d16 xyz store result") ||
      !Expect(ReadU16(typed_buffer_memory, 0x364u, &typed_buffer_short),
              "expected typed buffer d16 xyz store read") ||
      !Expect(typed_buffer_short == 0x4000u,
              "expected typed buffer d16 xyz store result")) {
    return 1;
  }
  }

  {
  const std::vector<DecodedInstruction> icache_program = {
      DecodedInstruction::Nullary("S_ICACHE_INV"),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  WaveExecutionState icache_state{};
  if (!Expect(interpreter.ExecuteProgram(icache_program, &icache_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(icache_state.halted,
              "expected s_icache_inv program to halt")) {
    return 1;
  }
  }

  LinearExecutionMemory vector_memory(0x800, 0);
  if (!Expect(vector_memory.WriteU32(0x184, 0x11111111u),
              "expected flat load seed write") ||
      !Expect(vector_memory.WriteU32(0x188, 0x22222222u),
              "expected flat load seed write") ||
      !Expect(vector_memory.WriteU32(0x190, 0x33333333u),
              "expected flat load seed write") ||
      !Expect(vector_memory.WriteU32(0x220, 0xaaaabbbbu),
              "expected global load seed write") ||
      !Expect(vector_memory.WriteU32(0x224, 0xccccddddu),
              "expected global load seed write") ||
      !Expect(vector_memory.WriteU32(0x22c, 0x12345678u),
              "expected global load seed write")) {
    return 1;
  }

  WaveExecutionState vector_memory_state;
  vector_memory_state.exec_mask = 0b1011ULL;
  vector_memory_state.sgprs[0] = 0x200;
  vector_memory_state.sgprs[1] = 0x0;
  vector_memory_state.vgprs[0][0] = 0x180;
  vector_memory_state.vgprs[0][1] = 0x184;
  vector_memory_state.vgprs[0][3] = 0x18c;
  vector_memory_state.vgprs[1][0] = 0x0;
  vector_memory_state.vgprs[1][1] = 0x0;
  vector_memory_state.vgprs[1][3] = 0x0;
  vector_memory_state.vgprs[2][0] = 0xdead0001u;
  vector_memory_state.vgprs[2][1] = 0xdead0002u;
  vector_memory_state.vgprs[2][3] = 0xdead0004u;
  vector_memory_state.vgprs[3][0] = 0xbeef0011u;
  vector_memory_state.vgprs[3][1] = 0xbeef0022u;
  vector_memory_state.vgprs[3][3] = 0xbeef0044u;
  vector_memory_state.vgprs[4][0] = 0x24;
  vector_memory_state.vgprs[4][1] = 0x28;
  vector_memory_state.vgprs[4][3] = 0x30;
  vector_memory_state.vgprs[5][0] = 0x0;
  vector_memory_state.vgprs[5][1] = 0x0;
  vector_memory_state.vgprs[5][3] = 0x0;
  const std::vector<DecodedInstruction> vector_memory_program = {
      DecodedInstruction::ThreeOperand("FLAT_LOAD_DWORD",
                                       InstructionOperand::Vgpr(10),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(4)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_DWORD",
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand(
          "GLOBAL_LOAD_DWORD", InstructionOperand::Vgpr(11),
          InstructionOperand::Vgpr(4), InstructionOperand::Sgpr(0),
          InstructionOperand::Imm32(static_cast<std::uint32_t>(-4))),
      DecodedInstruction::FourOperand("GLOBAL_STORE_DWORD",
                                      InstructionOperand::Vgpr(4),
                                      InstructionOperand::Vgpr(3),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(4)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(vector_memory_program,
                                         &vector_memory_state, &vector_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_memory_state.vgprs[10][0] == 0x11111111u,
              "expected lane 0 flat load result") ||
      !Expect(vector_memory_state.vgprs[10][1] == 0x22222222u,
              "expected lane 1 flat load result") ||
      !Expect(vector_memory_state.vgprs[10][2] == 0x0u,
              "expected inactive flat load lane to remain untouched") ||
      !Expect(vector_memory_state.vgprs[10][3] == 0x33333333u,
              "expected lane 3 flat load result") ||
      !Expect(vector_memory_state.vgprs[11][0] == 0xaaaabbbbu,
              "expected lane 0 global load result") ||
      !Expect(vector_memory_state.vgprs[11][1] == 0xccccddddu,
              "expected lane 1 global load result") ||
      !Expect(vector_memory_state.vgprs[11][2] == 0x0u,
              "expected inactive global load lane to remain untouched") ||
      !Expect(vector_memory_state.vgprs[11][3] == 0x12345678u,
              "expected lane 3 global load result")) {
    return 1;
  }

  std::uint32_t flat_store_lane0 = 0;
  std::uint32_t flat_store_lane1 = 0;
  std::uint32_t flat_store_lane3 = 0;
  std::uint32_t global_store_lane0 = 0;
  std::uint32_t global_store_lane1 = 0;
  std::uint32_t global_store_lane3 = 0;
  if (!Expect(vector_memory.ReadU32(0x180, &flat_store_lane0),
              "expected flat store lane 0 read") ||
      !Expect(vector_memory.ReadU32(0x184, &flat_store_lane1),
              "expected flat store lane 1 read") ||
      !Expect(vector_memory.ReadU32(0x18c, &flat_store_lane3),
              "expected flat store lane 3 read") ||
      !Expect(flat_store_lane0 == 0xdead0001u,
              "expected lane 0 flat store result") ||
      !Expect(flat_store_lane1 == 0xdead0002u,
              "expected lane 1 flat store result") ||
      !Expect(flat_store_lane3 == 0xdead0004u,
              "expected lane 3 flat store result") ||
      !Expect(vector_memory.ReadU32(0x228, &global_store_lane0),
              "expected global store lane 0 read") ||
      !Expect(vector_memory.ReadU32(0x22c, &global_store_lane1),
              "expected global store lane 1 read") ||
      !Expect(vector_memory.ReadU32(0x234, &global_store_lane3),
              "expected global store lane 3 read") ||
      !Expect(global_store_lane0 == 0xbeef0011u,
              "expected lane 0 global store result") ||
      !Expect(global_store_lane1 == 0xbeef0022u,
              "expected lane 1 global store result") ||
      !Expect(global_store_lane3 == 0xbeef0044u,
              "expected lane 3 global store result")) {
    return 1;
  }

  LinearExecutionMemory subword_memory(0x2000, 0);
  if (!Expect(WriteU8(&subword_memory, 0x600, 0x7au),
              "expected flat ubyte seed write") ||
      !Expect(WriteU8(&subword_memory, 0x610, 0x80u),
              "expected flat sbyte seed write") ||
      !Expect(WriteU16(&subword_memory, 0x620, 0x1234u),
              "expected flat ushort seed write") ||
      !Expect(WriteU16(&subword_memory, 0x630, 0x8001u),
              "expected flat sshort seed write") ||
      !Expect(WriteU8(&subword_memory, 0xa20, 0xa5u),
              "expected global ubyte seed write") ||
      !Expect(WriteU8(&subword_memory, 0xa30, 0xf0u),
              "expected global sbyte seed write") ||
      !Expect(WriteU16(&subword_memory, 0xa40, 0x5678u),
              "expected global ushort seed write") ||
      !Expect(WriteU16(&subword_memory, 0xa50, 0x8002u),
              "expected global sshort seed write")) {
    return 1;
  }

  WaveExecutionState subword_state;
  subword_state.exec_mask = 0x1ULL;
  subword_state.sgprs[0] = 0xa00;
  subword_state.sgprs[1] = 0x0;
  subword_state.vgprs[0][0] = 0x600;
  subword_state.vgprs[2][0] = 0x610;
  subword_state.vgprs[4][0] = 0x620;
  subword_state.vgprs[6][0] = 0x630;
  subword_state.vgprs[8][0] = 0x20;
  subword_state.vgprs[10][0] = 0x30;
  subword_state.vgprs[12][0] = 0x40;
  subword_state.vgprs[14][0] = 0x50;
  subword_state.vgprs[16][0] = 0x640;
  subword_state.vgprs[18][0] = 0x650;
  subword_state.vgprs[20][0] = 0x60;
  subword_state.vgprs[22][0] = 0x70;
  subword_state.vgprs[40][0] = 0x123456abu;
  subword_state.vgprs[41][0] = 0x89abcdefu;
  subword_state.vgprs[42][0] = 0x55667788u;
  subword_state.vgprs[43][0] = 0xa1b2c3d4u;
  const std::vector<DecodedInstruction> subword_program = {
      DecodedInstruction::ThreeOperand("FLAT_LOAD_UBYTE",
                                       InstructionOperand::Vgpr(30),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SBYTE",
                                       InstructionOperand::Vgpr(31),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_USHORT",
                                       InstructionOperand::Vgpr(32),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SSHORT",
                                       InstructionOperand::Vgpr(33),
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_UBYTE",
                                      InstructionOperand::Vgpr(34),
                                      InstructionOperand::Vgpr(8),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SBYTE",
                                      InstructionOperand::Vgpr(35),
                                      InstructionOperand::Vgpr(10),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_USHORT",
                                      InstructionOperand::Vgpr(36),
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SSHORT",
                                      InstructionOperand::Vgpr(37),
                                      InstructionOperand::Vgpr(14),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_BYTE",
                                       InstructionOperand::Vgpr(16),
                                       InstructionOperand::Vgpr(40),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_SHORT",
                                       InstructionOperand::Vgpr(18),
                                       InstructionOperand::Vgpr(41),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_BYTE",
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Vgpr(42),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_SHORT",
                                      InstructionOperand::Vgpr(22),
                                      InstructionOperand::Vgpr(43),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(subword_program, &subword_state,
                                         &subword_memory, &error_message),
              error_message.c_str()) ||
      !Expect(subword_state.vgprs[30][0] == 0x7au,
              "expected flat ubyte load result") ||
      !Expect(subword_state.vgprs[31][0] == 0xffffff80u,
              "expected flat sbyte load result") ||
      !Expect(subword_state.vgprs[32][0] == 0x1234u,
              "expected flat ushort load result") ||
      !Expect(subword_state.vgprs[33][0] == 0xffff8001u,
              "expected flat sshort load result") ||
      !Expect(subword_state.vgprs[34][0] == 0xa5u,
              "expected global ubyte load result") ||
      !Expect(subword_state.vgprs[35][0] == 0xfffffff0u,
              "expected global sbyte load result") ||
      !Expect(subword_state.vgprs[36][0] == 0x5678u,
              "expected global ushort load result") ||
      !Expect(subword_state.vgprs[37][0] == 0xffff8002u,
              "expected global sshort load result")) {
    return 1;
  }

  std::uint8_t stored_byte = 0;
  std::uint16_t stored_short = 0;
  if (!Expect(ReadU8(subword_memory, 0x640, &stored_byte),
              "expected flat byte store read") ||
      !Expect(stored_byte == 0xabu, "expected flat byte store result") ||
      !Expect(ReadU16(subword_memory, 0x650, &stored_short),
              "expected flat short store read") ||
      !Expect(stored_short == 0xcdefu, "expected flat short store result") ||
      !Expect(ReadU8(subword_memory, 0xa60, &stored_byte),
              "expected global byte store read") ||
      !Expect(stored_byte == 0x88u, "expected global byte store result") ||
      !Expect(ReadU16(subword_memory, 0xa70, &stored_short),
              "expected global short store read") ||
      !Expect(stored_short == 0xc3d4u, "expected global short store result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> vector_memory_d16_program = {
      DecodedInstruction::ThreeOperand("FLAT_LOAD_UBYTE_D16",
                                       InstructionOperand::Vgpr(60),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_UBYTE_D16_HI",
                                       InstructionOperand::Vgpr(61),
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SBYTE_D16",
                                       InstructionOperand::Vgpr(62),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SBYTE_D16_HI",
                                       InstructionOperand::Vgpr(63),
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SHORT_D16",
                                       InstructionOperand::Vgpr(64),
                                       InstructionOperand::Vgpr(8),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_SHORT_D16_HI",
                                       InstructionOperand::Vgpr(65),
                                       InstructionOperand::Vgpr(10),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_UBYTE_D16",
                                      InstructionOperand::Vgpr(66),
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_UBYTE_D16_HI",
                                      InstructionOperand::Vgpr(67),
                                      InstructionOperand::Vgpr(14),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SBYTE_D16",
                                      InstructionOperand::Vgpr(68),
                                      InstructionOperand::Vgpr(16),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SBYTE_D16_HI",
                                      InstructionOperand::Vgpr(69),
                                      InstructionOperand::Vgpr(18),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SHORT_D16",
                                      InstructionOperand::Vgpr(70),
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_SHORT_D16_HI",
                                      InstructionOperand::Vgpr(71),
                                      InstructionOperand::Vgpr(22),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_BYTE_D16_HI",
                                       InstructionOperand::Vgpr(24),
                                       InstructionOperand::Vgpr(40),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_SHORT_D16_HI",
                                       InstructionOperand::Vgpr(26),
                                       InstructionOperand::Vgpr(41),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_BYTE_D16_HI",
                                      InstructionOperand::Vgpr(28),
                                      InstructionOperand::Vgpr(42),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_SHORT_D16_HI",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(43),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  const auto seed_vector_memory_d16 =
      [](LinearExecutionMemory* memory) -> bool {
    return memory != nullptr &&
           WriteU8(memory, 0x700, 0x34u) &&
           WriteU8(memory, 0x710, 0x56u) &&
           WriteU8(memory, 0x720, 0x80u) &&
           WriteU8(memory, 0x730, 0xf0u) &&
           WriteU16(memory, 0x740, 0x1234u) &&
           WriteU16(memory, 0x750, 0xabcdu) &&
           WriteU8(memory, 0xc20, 0x78u) &&
           WriteU8(memory, 0xc30, 0x9au) &&
           WriteU8(memory, 0xc40, 0x81u) &&
           WriteU8(memory, 0xc50, 0x82u) &&
           WriteU16(memory, 0xc60, 0x4567u) &&
           WriteU16(memory, 0xc70, 0x89abu);
  };
  const auto make_vector_memory_d16_state = []() {
    WaveExecutionState state{};
    state.exec_mask = 0x1ULL;
    state.sgprs[0] = 0xc00;
    state.sgprs[1] = 0x0;
    state.vgprs[0][0] = 0x700;
    state.vgprs[2][0] = 0x710;
    state.vgprs[4][0] = 0x720;
    state.vgprs[6][0] = 0x730;
    state.vgprs[8][0] = 0x740;
    state.vgprs[10][0] = 0x750;
    state.vgprs[12][0] = 0x20;
    state.vgprs[14][0] = 0x30;
    state.vgprs[16][0] = 0x40;
    state.vgprs[18][0] = 0x50;
    state.vgprs[20][0] = 0x60;
    state.vgprs[22][0] = 0x70;
    state.vgprs[24][0] = 0x760;
    state.vgprs[26][0] = 0x770;
    state.vgprs[28][0] = 0x80;
    state.vgprs[30][0] = 0x90;
    state.vgprs[40][0] = 0x1234abcdu;
    state.vgprs[41][0] = 0x89abcdefu;
    state.vgprs[42][0] = 0x13572468u;
    state.vgprs[43][0] = 0x2468ace0u;
    return state;
  };
  const auto validate_vector_memory_d16 =
      [](const WaveExecutionState& state,
         const LinearExecutionMemory& memory,
         const char* mode) -> bool {
    std::uint8_t byte_value = 0;
    std::uint16_t short_value = 0;
    return Expect(state.vgprs[60][0] == 0x00000034u,
                  (std::string(mode) + " flat load ubyte d16").c_str()) &&
           Expect(state.vgprs[61][0] == 0x00560000u,
                  (std::string(mode) + " flat load ubyte d16 hi").c_str()) &&
           Expect(state.vgprs[62][0] == 0x0000ff80u,
                  (std::string(mode) + " flat load sbyte d16").c_str()) &&
           Expect(state.vgprs[63][0] == 0xfff00000u,
                  (std::string(mode) + " flat load sbyte d16 hi").c_str()) &&
           Expect(state.vgprs[64][0] == 0x00001234u,
                  (std::string(mode) + " flat load short d16").c_str()) &&
           Expect(state.vgprs[65][0] == 0xabcd0000u,
                  (std::string(mode) + " flat load short d16 hi").c_str()) &&
           Expect(state.vgprs[66][0] == 0x00000078u,
                  (std::string(mode) + " global load ubyte d16").c_str()) &&
           Expect(state.vgprs[67][0] == 0x009a0000u,
                  (std::string(mode) + " global load ubyte d16 hi").c_str()) &&
           Expect(state.vgprs[68][0] == 0x0000ff81u,
                  (std::string(mode) + " global load sbyte d16").c_str()) &&
           Expect(state.vgprs[69][0] == 0xff820000u,
                  (std::string(mode) + " global load sbyte d16 hi").c_str()) &&
           Expect(state.vgprs[70][0] == 0x00004567u,
                  (std::string(mode) + " global load short d16").c_str()) &&
           Expect(state.vgprs[71][0] == 0x89ab0000u,
                  (std::string(mode) + " global load short d16 hi").c_str()) &&
           Expect(ReadU8(memory, 0x760, &byte_value),
                  (std::string(mode) + " flat store byte d16 hi read").c_str()) &&
           Expect(byte_value == 0x34u,
                  (std::string(mode) + " flat store byte d16 hi").c_str()) &&
           Expect(ReadU16(memory, 0x770, &short_value),
                  (std::string(mode) + " flat store short d16 hi read").c_str()) &&
           Expect(short_value == 0x89abu,
                  (std::string(mode) + " flat store short d16 hi").c_str()) &&
           Expect(ReadU8(memory, 0xc80, &byte_value),
                  (std::string(mode) + " global store byte d16 hi read").c_str()) &&
           Expect(byte_value == 0x57u,
                  (std::string(mode) + " global store byte d16 hi").c_str()) &&
           Expect(ReadU16(memory, 0xc90, &short_value),
                  (std::string(mode) + " global store short d16 hi read").c_str()) &&
           Expect(short_value == 0x2468u,
                  (std::string(mode) + " global store short d16 hi").c_str());
  };

  LinearExecutionMemory vector_memory_d16_memory(0x2000, 0);
  if (!Expect(seed_vector_memory_d16(&vector_memory_d16_memory),
              "expected vector memory d16 seed writes")) {
    return 1;
  }
  WaveExecutionState vector_memory_d16_state = make_vector_memory_d16_state();
  if (!Expect(interpreter.ExecuteProgram(vector_memory_d16_program,
                                         &vector_memory_d16_state,
                                         &vector_memory_d16_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_vector_memory_d16(vector_memory_d16_state,
                                  vector_memory_d16_memory, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_vector_memory_d16_program;
  if (!Expect(interpreter.CompileProgram(vector_memory_d16_program,
                                         &compiled_vector_memory_d16_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  LinearExecutionMemory compiled_vector_memory_d16_memory(0x2000, 0);
  if (!Expect(seed_vector_memory_d16(&compiled_vector_memory_d16_memory),
              "expected compiled vector memory d16 seed writes")) {
    return 1;
  }
  WaveExecutionState compiled_vector_memory_d16_state =
      make_vector_memory_d16_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_memory_d16_program,
                                         &compiled_vector_memory_d16_state,
                                         &compiled_vector_memory_d16_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_vector_memory_d16(compiled_vector_memory_d16_state,
                                  compiled_vector_memory_d16_memory,
                                  "compiled")) {
    return 1;
  }

  LinearExecutionMemory global_x2_memory(0x1000, 0);
  if (!Expect(global_x2_memory.WriteU32(0x340, 0x01020304u),
              "expected global x2 load seed write") ||
      !Expect(global_x2_memory.WriteU32(0x344, 0x05060708u),
              "expected global x2 load seed write") ||
      !Expect(global_x2_memory.WriteU32(0x348, 0x11121314u),
              "expected global x2 load seed write") ||
      !Expect(global_x2_memory.WriteU32(0x34c, 0x15161718u),
              "expected global x2 load seed write") ||
      !Expect(global_x2_memory.WriteU32(0x358, 0x21222324u),
              "expected global x2 load seed write") ||
      !Expect(global_x2_memory.WriteU32(0x35c, 0x25262728u),
              "expected global x2 load seed write")) {
    return 1;
  }

  WaveExecutionState global_x2_state;
  global_x2_state.exec_mask = 0b1011ULL;
  global_x2_state.sgprs[0] = 0x300;
  global_x2_state.sgprs[1] = 0x0;
  global_x2_state.vgprs[6][0] = 0x40;
  global_x2_state.vgprs[6][1] = 0x48;
  global_x2_state.vgprs[6][3] = 0x58;
  global_x2_state.vgprs[7][0] = 0x0;
  global_x2_state.vgprs[7][1] = 0x0;
  global_x2_state.vgprs[7][3] = 0x0;
  global_x2_state.vgprs[8][0] = 0x80;
  global_x2_state.vgprs[8][1] = 0x88;
  global_x2_state.vgprs[8][3] = 0x98;
  global_x2_state.vgprs[9][0] = 0x0;
  global_x2_state.vgprs[9][1] = 0x0;
  global_x2_state.vgprs[9][3] = 0x0;
  global_x2_state.vgprs[20][0] = 0xaaaabbbb;
  global_x2_state.vgprs[20][1] = 0xccccdddd;
  global_x2_state.vgprs[20][3] = 0xeeeeffff;
  global_x2_state.vgprs[21][0] = 0x11112222;
  global_x2_state.vgprs[21][1] = 0x33334444;
  global_x2_state.vgprs[21][3] = 0x55556666;
  const std::vector<DecodedInstruction> global_x2_program = {
      DecodedInstruction::FourOperand("GLOBAL_LOAD_DWORDX2",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(6),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_DWORDX2",
                                      InstructionOperand::Vgpr(8),
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(global_x2_program, &global_x2_state,
                                         &global_x2_memory, &error_message),
              error_message.c_str()) ||
      !Expect(global_x2_state.vgprs[30][0] == 0x01020304u,
              "expected lane 0 global x2 low load result") ||
      !Expect(global_x2_state.vgprs[31][0] == 0x05060708u,
              "expected lane 0 global x2 high load result") ||
      !Expect(global_x2_state.vgprs[30][1] == 0x11121314u,
              "expected lane 1 global x2 low load result") ||
      !Expect(global_x2_state.vgprs[31][1] == 0x15161718u,
              "expected lane 1 global x2 high load result") ||
      !Expect(global_x2_state.vgprs[30][2] == 0x0u,
              "expected inactive lane global x2 low load to remain untouched") ||
      !Expect(global_x2_state.vgprs[31][2] == 0x0u,
              "expected inactive lane global x2 high load to remain untouched") ||
      !Expect(global_x2_state.vgprs[30][3] == 0x21222324u,
              "expected lane 3 global x2 low load result") ||
      !Expect(global_x2_state.vgprs[31][3] == 0x25262728u,
              "expected lane 3 global x2 high load result")) {
    return 1;
  }

  std::uint32_t global_x2_store_low = 0;
  std::uint32_t global_x2_store_high = 0;
  if (!Expect(global_x2_memory.ReadU32(0x380, &global_x2_store_low),
              "expected global x2 store lane 0 low read") ||
      !Expect(global_x2_memory.ReadU32(0x384, &global_x2_store_high),
              "expected global x2 store lane 0 high read") ||
      !Expect(global_x2_store_low == 0xaaaabbbbu,
              "expected lane 0 global x2 low store result") ||
      !Expect(global_x2_store_high == 0x11112222u,
              "expected lane 0 global x2 high store result") ||
      !Expect(global_x2_memory.ReadU32(0x388, &global_x2_store_low),
              "expected global x2 store lane 1 low read") ||
      !Expect(global_x2_memory.ReadU32(0x38c, &global_x2_store_high),
              "expected global x2 store lane 1 high read") ||
      !Expect(global_x2_store_low == 0xccccddddu,
              "expected lane 1 global x2 low store result") ||
      !Expect(global_x2_store_high == 0x33334444u,
              "expected lane 1 global x2 high store result") ||
      !Expect(global_x2_memory.ReadU32(0x398, &global_x2_store_low),
              "expected global x2 store lane 3 low read") ||
      !Expect(global_x2_memory.ReadU32(0x39c, &global_x2_store_high),
              "expected global x2 store lane 3 high read") ||
      !Expect(global_x2_store_low == 0xeeeeffffu,
              "expected lane 3 global x2 low store result") ||
      !Expect(global_x2_store_high == 0x55556666u,
              "expected lane 3 global x2 high store result")) {
    return 1;
  }

  LinearExecutionMemory global_x4_memory(0x2000, 0);
  if (!Expect(global_x4_memory.WriteU32(0x440, 0x10111213u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x444, 0x14151617u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x448, 0x18191a1bu),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x44c, 0x1c1d1e1fu),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x450, 0x20212223u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x454, 0x24252627u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x458, 0x28292a2bu),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x45c, 0x2c2d2e2fu),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x470, 0x30313233u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x474, 0x34353637u),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x478, 0x38393a3bu),
              "expected global x4 load seed write") ||
      !Expect(global_x4_memory.WriteU32(0x47c, 0x3c3d3e3fu),
              "expected global x4 load seed write")) {
    return 1;
  }

  WaveExecutionState global_x4_state;
  global_x4_state.exec_mask = 0b1011ULL;
  global_x4_state.sgprs[0] = 0x400;
  global_x4_state.sgprs[1] = 0x0;
  global_x4_state.vgprs[10][0] = 0x40;
  global_x4_state.vgprs[10][1] = 0x50;
  global_x4_state.vgprs[10][3] = 0x70;
  global_x4_state.vgprs[11][0] = 0x0;
  global_x4_state.vgprs[11][1] = 0x0;
  global_x4_state.vgprs[11][3] = 0x0;
  global_x4_state.vgprs[12][0] = 0x80;
  global_x4_state.vgprs[12][1] = 0x90;
  global_x4_state.vgprs[12][3] = 0xb0;
  global_x4_state.vgprs[13][0] = 0x0;
  global_x4_state.vgprs[13][1] = 0x0;
  global_x4_state.vgprs[13][3] = 0x0;
  global_x4_state.vgprs[40][0] = 0xa0a1a2a3u;
  global_x4_state.vgprs[40][1] = 0xb0b1b2b3u;
  global_x4_state.vgprs[40][3] = 0xc0c1c2c3u;
  global_x4_state.vgprs[41][0] = 0xa4a5a6a7u;
  global_x4_state.vgprs[41][1] = 0xb4b5b6b7u;
  global_x4_state.vgprs[41][3] = 0xc4c5c6c7u;
  global_x4_state.vgprs[42][0] = 0xa8a9aaabu;
  global_x4_state.vgprs[42][1] = 0xb8b9babbu;
  global_x4_state.vgprs[42][3] = 0xc8c9cacbu;
  global_x4_state.vgprs[43][0] = 0xacadaeafu;
  global_x4_state.vgprs[43][1] = 0xbcbdbebfu;
  global_x4_state.vgprs[43][3] = 0xcccdcecfu;
  const std::vector<DecodedInstruction> global_x4_program = {
      DecodedInstruction::FourOperand("GLOBAL_LOAD_DWORDX4",
                                      InstructionOperand::Vgpr(50),
                                      InstructionOperand::Vgpr(10),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_DWORDX4",
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Vgpr(40),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(global_x4_program, &global_x4_state,
                                         &global_x4_memory, &error_message),
              error_message.c_str()) ||
      !Expect(global_x4_state.vgprs[50][0] == 0x10111213u,
              "expected lane 0 global x4 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][0] == 0x14151617u,
              "expected lane 0 global x4 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][0] == 0x18191a1bu,
              "expected lane 0 global x4 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][0] == 0x1c1d1e1fu,
              "expected lane 0 global x4 dword 3 load result") ||
      !Expect(global_x4_state.vgprs[50][1] == 0x20212223u,
              "expected lane 1 global x4 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][1] == 0x24252627u,
              "expected lane 1 global x4 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][1] == 0x28292a2bu,
              "expected lane 1 global x4 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][1] == 0x2c2d2e2fu,
              "expected lane 1 global x4 dword 3 load result") ||
      !Expect(global_x4_state.vgprs[50][2] == 0x0u,
              "expected inactive lane global x4 dword 0 load to remain untouched") ||
      !Expect(global_x4_state.vgprs[53][2] == 0x0u,
              "expected inactive lane global x4 dword 3 load to remain untouched") ||
      !Expect(global_x4_state.vgprs[50][3] == 0x30313233u,
              "expected lane 3 global x4 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][3] == 0x34353637u,
              "expected lane 3 global x4 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][3] == 0x38393a3bu,
              "expected lane 3 global x4 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][3] == 0x3c3d3e3fu,
              "expected lane 3 global x4 dword 3 load result")) {
    return 1;
  }

  std::uint32_t global_x4_store_value = 0;
  if (!Expect(global_x4_memory.ReadU32(0x480, &global_x4_store_value),
              "expected global x4 store lane 0 dword 0 read") ||
      !Expect(global_x4_store_value == 0xa0a1a2a3u,
              "expected lane 0 global x4 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x484, &global_x4_store_value),
              "expected global x4 store lane 0 dword 1 read") ||
      !Expect(global_x4_store_value == 0xa4a5a6a7u,
              "expected lane 0 global x4 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x488, &global_x4_store_value),
              "expected global x4 store lane 0 dword 2 read") ||
      !Expect(global_x4_store_value == 0xa8a9aaabu,
              "expected lane 0 global x4 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x48c, &global_x4_store_value),
              "expected global x4 store lane 0 dword 3 read") ||
      !Expect(global_x4_store_value == 0xacadaeafu,
              "expected lane 0 global x4 dword 3 store result") ||
      !Expect(global_x4_memory.ReadU32(0x490, &global_x4_store_value),
              "expected global x4 store lane 1 dword 0 read") ||
      !Expect(global_x4_store_value == 0xb0b1b2b3u,
              "expected lane 1 global x4 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x494, &global_x4_store_value),
              "expected global x4 store lane 1 dword 1 read") ||
      !Expect(global_x4_store_value == 0xb4b5b6b7u,
              "expected lane 1 global x4 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x498, &global_x4_store_value),
              "expected global x4 store lane 1 dword 2 read") ||
      !Expect(global_x4_store_value == 0xb8b9babbu,
              "expected lane 1 global x4 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x49c, &global_x4_store_value),
              "expected global x4 store lane 1 dword 3 read") ||
      !Expect(global_x4_store_value == 0xbcbdbebfu,
              "expected lane 1 global x4 dword 3 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b0, &global_x4_store_value),
              "expected global x4 store lane 3 dword 0 read") ||
      !Expect(global_x4_store_value == 0xc0c1c2c3u,
              "expected lane 3 global x4 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b4, &global_x4_store_value),
              "expected global x4 store lane 3 dword 1 read") ||
      !Expect(global_x4_store_value == 0xc4c5c6c7u,
              "expected lane 3 global x4 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b8, &global_x4_store_value),
              "expected global x4 store lane 3 dword 2 read") ||
      !Expect(global_x4_store_value == 0xc8c9cacbu,
              "expected lane 3 global x4 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4bc, &global_x4_store_value),
              "expected global x4 store lane 3 dword 3 read") ||
      !Expect(global_x4_store_value == 0xcccdcecfu,
              "expected lane 3 global x4 dword 3 store result")) {
    return 1;
  }

  LinearExecutionMemory mixed_width_memory(0x3000, 0);
  if (!Expect(mixed_width_memory.WriteU32(0x900, 0x10101010u),
              "expected flat x2 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x904, 0x20202020u),
              "expected flat x2 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x920, 0x30303030u),
              "expected flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x924, 0x40404040u),
              "expected flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x928, 0x50505050u),
              "expected flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x940, 0x60606060u),
              "expected flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x944, 0x70707070u),
              "expected flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x948, 0x80808080u),
              "expected flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x94c, 0x90909090u),
              "expected flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc20, 0xa0a0a0a0u),
              "expected global x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc24, 0xb0b0b0b0u),
              "expected global x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc28, 0xc0c0c0c0u),
              "expected global x3 load seed write")) {
    return 1;
  }

  WaveExecutionState mixed_width_state;
  mixed_width_state.exec_mask = 0x1ULL;
  mixed_width_state.sgprs[0] = 0xc00;
  mixed_width_state.sgprs[1] = 0x0;
  mixed_width_state.vgprs[0][0] = 0x900;
  mixed_width_state.vgprs[2][0] = 0x980;
  mixed_width_state.vgprs[4][0] = 0x920;
  mixed_width_state.vgprs[6][0] = 0x9a0;
  mixed_width_state.vgprs[8][0] = 0x940;
  mixed_width_state.vgprs[10][0] = 0x9c0;
  mixed_width_state.vgprs[12][0] = 0x20;
  mixed_width_state.vgprs[14][0] = 0x40;
  mixed_width_state.vgprs[70][0] = 0xd1d2d3d4u;
  mixed_width_state.vgprs[71][0] = 0xe1e2e3e4u;
  mixed_width_state.vgprs[72][0] = 0xf1f2f3f4u;
  mixed_width_state.vgprs[73][0] = 0x11121314u;
  mixed_width_state.vgprs[74][0] = 0x21222324u;
  mixed_width_state.vgprs[75][0] = 0x31323334u;
  mixed_width_state.vgprs[76][0] = 0x41424344u;
  mixed_width_state.vgprs[79][0] = 0x51525354u;
  mixed_width_state.vgprs[80][0] = 0x61626364u;
  mixed_width_state.vgprs[81][0] = 0x71727374u;
  const std::vector<DecodedInstruction> mixed_width_program = {
      DecodedInstruction::ThreeOperand("FLAT_LOAD_DWORDX2",
                                       InstructionOperand::Vgpr(50),
                                       InstructionOperand::Vgpr(0),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_DWORDX2",
                                       InstructionOperand::Vgpr(2),
                                       InstructionOperand::Vgpr(70),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_DWORDX3",
                                       InstructionOperand::Vgpr(52),
                                       InstructionOperand::Vgpr(4),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_DWORDX3",
                                       InstructionOperand::Vgpr(6),
                                       InstructionOperand::Vgpr(72),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_LOAD_DWORDX4",
                                       InstructionOperand::Vgpr(55),
                                       InstructionOperand::Vgpr(8),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_STORE_DWORDX4",
                                       InstructionOperand::Vgpr(10),
                                       InstructionOperand::Vgpr(73),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_LOAD_DWORDX3",
                                      InstructionOperand::Vgpr(59),
                                      InstructionOperand::Vgpr(12),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_STORE_DWORDX3",
                                      InstructionOperand::Vgpr(14),
                                      InstructionOperand::Vgpr(79),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(mixed_width_program, &mixed_width_state,
                                         &mixed_width_memory, &error_message),
              error_message.c_str()) ||
      !Expect(mixed_width_state.vgprs[50][0] == 0x10101010u,
              "expected flat x2 low load result") ||
      !Expect(mixed_width_state.vgprs[51][0] == 0x20202020u,
              "expected flat x2 high load result") ||
      !Expect(mixed_width_state.vgprs[52][0] == 0x30303030u,
              "expected flat x3 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[53][0] == 0x40404040u,
              "expected flat x3 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[54][0] == 0x50505050u,
              "expected flat x3 dword 2 load result") ||
      !Expect(mixed_width_state.vgprs[55][0] == 0x60606060u,
              "expected flat x4 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[56][0] == 0x70707070u,
              "expected flat x4 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[57][0] == 0x80808080u,
              "expected flat x4 dword 2 load result") ||
      !Expect(mixed_width_state.vgprs[58][0] == 0x90909090u,
              "expected flat x4 dword 3 load result") ||
      !Expect(mixed_width_state.vgprs[59][0] == 0xa0a0a0a0u,
              "expected global x3 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[60][0] == 0xb0b0b0b0u,
              "expected global x3 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[61][0] == 0xc0c0c0c0u,
              "expected global x3 dword 2 load result")) {
    return 1;
  }

  std::uint32_t mixed_width_value = 0;
  if (!Expect(mixed_width_memory.ReadU32(0x980, &mixed_width_value),
              "expected flat x2 store dword 0 read") ||
      !Expect(mixed_width_value == 0xd1d2d3d4u,
              "expected flat x2 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x984, &mixed_width_value),
              "expected flat x2 store dword 1 read") ||
      !Expect(mixed_width_value == 0xe1e2e3e4u,
              "expected flat x2 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a0, &mixed_width_value),
              "expected flat x3 store dword 0 read") ||
      !Expect(mixed_width_value == 0xf1f2f3f4u,
              "expected flat x3 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a4, &mixed_width_value),
              "expected flat x3 store dword 1 read") ||
      !Expect(mixed_width_value == 0x11121314u,
              "expected flat x3 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a8, &mixed_width_value),
              "expected flat x3 store dword 2 read") ||
      !Expect(mixed_width_value == 0x21222324u,
              "expected flat x3 store dword 2 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c0, &mixed_width_value),
              "expected flat x4 store dword 0 read") ||
      !Expect(mixed_width_value == 0x11121314u,
              "expected flat x4 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c4, &mixed_width_value),
              "expected flat x4 store dword 1 read") ||
      !Expect(mixed_width_value == 0x21222324u,
              "expected flat x4 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c8, &mixed_width_value),
              "expected flat x4 store dword 2 read") ||
      !Expect(mixed_width_value == 0x31323334u,
              "expected flat x4 store dword 2 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9cc, &mixed_width_value),
              "expected flat x4 store dword 3 read") ||
      !Expect(mixed_width_value == 0x41424344u,
              "expected flat x4 store dword 3 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc40, &mixed_width_value),
              "expected global x3 store dword 0 read") ||
      !Expect(mixed_width_value == 0x51525354u,
              "expected global x3 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc44, &mixed_width_value),
              "expected global x3 store dword 1 read") ||
      !Expect(mixed_width_value == 0x61626364u,
              "expected global x3 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc48, &mixed_width_value),
              "expected global x3 store dword 2 read") ||
      !Expect(mixed_width_value == 0x71727374u,
              "expected global x3 store dword 2 result")) {
    return 1;
  }

  {
  auto set_lane_u64 = [](WaveExecutionState* state,
                         std::uint16_t reg,
                         std::size_t lane,
                         std::uint64_t value) {
    if (state == nullptr) {
      return;
    }
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    SplitU64(value, &low, &high);
    state->vgprs[reg][lane] = low;
    state->vgprs[static_cast<std::uint16_t>(reg + 1u)][lane] = high;
  };
  auto make_global_load_lds_state =
      [&](std::uint32_t m0_base,
          std::array<std::uint64_t, 3> lane_addresses) {
        WaveExecutionState state{};
        state.exec_mask = 0b1011ULL;
        state.sgprs[0] = 0u;
        state.sgprs[1] = 0u;
        state.sgprs[124] = m0_base;
        set_lane_u64(&state, 0u, 0u, lane_addresses[0]);
        set_lane_u64(&state, 0u, 1u, lane_addresses[1]);
        set_lane_u64(&state, 0u, 3u, lane_addresses[2]);
        return state;
      };
  auto expect_lds_dwords =
      [&](const WaveExecutionState& state,
          std::uint32_t base_address,
          std::initializer_list<std::uint32_t> expected_values,
          const char* mode) {
        std::size_t index = 0;
        for (std::uint32_t expected_value : expected_values) {
          std::uint32_t observed_value = 0;
          std::memcpy(&observed_value,
                      state.lds_bytes.data() + base_address +
                          index * sizeof(std::uint32_t),
                      sizeof(observed_value));
          if (!Expect(observed_value == expected_value,
                      "expected global_load_lds LDS value")) {
            std::cerr << mode << " index=" << index << '\n';
            return false;
          }
          ++index;
        }
        return true;
      };
  auto run_global_load_lds_case =
      [&](std::string_view opcode,
          std::int32_t offset,
          std::uint32_t m0_base,
          std::array<std::uint64_t, 3> lane_addresses,
          auto seed_memory,
          std::initializer_list<std::uint32_t> expected_values) {
        const std::vector<DecodedInstruction> program = {
            DecodedInstruction::ThreeOperand(
                opcode, InstructionOperand::Vgpr(0),
                InstructionOperand::Sgpr(0), InstructionOperand::Imm32(offset)),
            DecodedInstruction::Nullary("S_ENDPGM"),
        };

        LinearExecutionMemory decoded_memory(0x4000, 0);
        if (!Expect(seed_memory(&decoded_memory),
                    "expected decoded global_load_lds seed writes")) {
          return false;
        }
        WaveExecutionState decoded_state =
            make_global_load_lds_state(m0_base, lane_addresses);
        if (!Expect(interpreter.ExecuteProgram(program, &decoded_state,
                                               &decoded_memory, &error_message),
                    error_message.c_str()) ||
            !Expect(decoded_state.halted,
                    "expected decoded global_load_lds program to halt") ||
            !expect_lds_dwords(decoded_state,
                               static_cast<std::uint32_t>(m0_base + offset),
                               expected_values, "decoded")) {
          std::cerr << opcode << '\n';
          return false;
        }

        std::vector<CompiledInstruction> compiled_program;
        if (!Expect(interpreter.CompileProgram(program, &compiled_program,
                                               &error_message),
                    error_message.c_str())) {
          std::cerr << opcode << '\n';
          return false;
        }
        LinearExecutionMemory compiled_memory(0x4000, 0);
        if (!Expect(seed_memory(&compiled_memory),
                    "expected compiled global_load_lds seed writes")) {
          return false;
        }
        WaveExecutionState compiled_state =
            make_global_load_lds_state(m0_base, lane_addresses);
        if (!Expect(interpreter.ExecuteProgram(compiled_program, &compiled_state,
                                               &compiled_memory, &error_message),
                    error_message.c_str()) ||
            !Expect(compiled_state.halted,
                    "expected compiled global_load_lds program to halt") ||
            !expect_lds_dwords(compiled_state,
                               static_cast<std::uint32_t>(m0_base + offset),
                               expected_values, "compiled")) {
          std::cerr << opcode << '\n';
          return false;
        }
        return true;
      };

  if (!run_global_load_lds_case(
          "GLOBAL_LOAD_LDS_UBYTE", 0, 0x40u, {0x200u, 0x210u, 0x230u},
          [](LinearExecutionMemory* memory) {
            return memory != nullptr && WriteU8(memory, 0x200u, 0x7fu) &&
                   WriteU8(memory, 0x210u, 0x80u) &&
                   WriteU8(memory, 0x230u, 0xfeu);
          },
          {0x0000007fu, 0x00000080u, 0x000000feu}) ||
      !run_global_load_lds_case(
          "GLOBAL_LOAD_LDS_SSHORT", 0, 0x80u, {0x300u, 0x320u, 0x360u},
          [](LinearExecutionMemory* memory) {
            return memory != nullptr && WriteU16(memory, 0x300u, 0x0001u) &&
                   WriteU16(memory, 0x320u, 0xff80u) &&
                   WriteU16(memory, 0x360u, 0x7f01u);
          },
          {0x00000001u, 0xffffff80u, 0x00007f01u}) ||
      !run_global_load_lds_case(
          "GLOBAL_LOAD_LDS_DWORD", 0x10, 0x120u,
          {0x3f0u, 0x410u, 0x450u},
          [](LinearExecutionMemory* memory) {
            return memory != nullptr && memory->WriteU32(0x400u, 0x11223344u) &&
                   memory->WriteU32(0x420u, 0xaabbccddu) &&
                   memory->WriteU32(0x460u, 0x01020304u);
          },
          {0x11223344u, 0xaabbccddu, 0x01020304u}) ||
      !run_global_load_lds_case(
          "GLOBAL_LOAD_LDS_DWORDX3", 0, 0x180u,
          {0x500u, 0x520u, 0x560u},
          [](LinearExecutionMemory* memory) {
            return memory != nullptr && memory->WriteU32(0x500u, 11u) &&
                   memory->WriteU32(0x504u, 12u) &&
                   memory->WriteU32(0x508u, 13u) &&
                   memory->WriteU32(0x520u, 21u) &&
                   memory->WriteU32(0x524u, 22u) &&
                   memory->WriteU32(0x528u, 23u) &&
                   memory->WriteU32(0x560u, 31u) &&
                   memory->WriteU32(0x564u, 32u) &&
                   memory->WriteU32(0x568u, 33u);
          },
          {11u, 12u, 13u, 0u, 21u, 22u, 23u, 0u, 31u, 32u, 33u, 0u}) ||
      !run_global_load_lds_case(
          "GLOBAL_LOAD_LDS_DWORDX4", 0, 0x1c0u,
          {0x600u, 0x620u, 0x660u},
          [](LinearExecutionMemory* memory) {
            return memory != nullptr && memory->WriteU32(0x600u, 101u) &&
                   memory->WriteU32(0x604u, 102u) &&
                   memory->WriteU32(0x608u, 103u) &&
                   memory->WriteU32(0x60cu, 104u) &&
                   memory->WriteU32(0x620u, 201u) &&
                   memory->WriteU32(0x624u, 202u) &&
                   memory->WriteU32(0x628u, 203u) &&
                   memory->WriteU32(0x62cu, 204u) &&
                   memory->WriteU32(0x660u, 301u) &&
                   memory->WriteU32(0x664u, 302u) &&
                   memory->WriteU32(0x668u, 303u) &&
                   memory->WriteU32(0x66cu, 304u);
          },
          {101u, 102u, 103u, 104u, 201u, 202u, 203u, 204u,
           301u, 302u, 303u, 304u})) {
    return 1;
  }
  }

  LinearExecutionMemory atomic_memory(0x1000, 0);
  if (!Expect(atomic_memory.WriteU32(0x520, 10u),
              "expected atomic add seed write") ||
      !Expect(atomic_memory.WriteU32(0x524, 20u),
              "expected atomic add seed write") ||
      !Expect(atomic_memory.WriteU32(0x52c, 40u),
              "expected atomic add seed write") ||
      !Expect(atomic_memory.WriteU32(0x530, 50u),
              "expected atomic swap seed write") ||
      !Expect(atomic_memory.WriteU32(0x534, 60u),
              "expected atomic swap seed write") ||
      !Expect(atomic_memory.WriteU32(0x53c, 80u),
              "expected atomic swap seed write") ||
      !Expect(atomic_memory.WriteU32(0x540, 100u),
              "expected atomic cmpswap seed write") ||
      !Expect(atomic_memory.WriteU32(0x544, 110u),
              "expected atomic cmpswap seed write") ||
      !Expect(atomic_memory.WriteU32(0x54c, 130u),
              "expected atomic cmpswap seed write")) {
    return 1;
  }

  WaveExecutionState atomic_state;
  atomic_state.exec_mask = 0b1011ULL;
  atomic_state.sgprs[0] = 0x500;
  atomic_state.sgprs[1] = 0x0;
  atomic_state.vgprs[14][0] = 0x20;
  atomic_state.vgprs[14][1] = 0x24;
  atomic_state.vgprs[14][3] = 0x2c;
  atomic_state.vgprs[15][0] = 0x0;
  atomic_state.vgprs[15][1] = 0x0;
  atomic_state.vgprs[15][3] = 0x0;
  atomic_state.vgprs[16][0] = 0x30;
  atomic_state.vgprs[16][1] = 0x34;
  atomic_state.vgprs[16][3] = 0x3c;
  atomic_state.vgprs[17][0] = 0x0;
  atomic_state.vgprs[17][1] = 0x0;
  atomic_state.vgprs[17][3] = 0x0;
  atomic_state.vgprs[18][0] = 0x40;
  atomic_state.vgprs[18][1] = 0x44;
  atomic_state.vgprs[18][3] = 0x4c;
  atomic_state.vgprs[19][0] = 0x0;
  atomic_state.vgprs[19][1] = 0x0;
  atomic_state.vgprs[19][3] = 0x0;
  atomic_state.vgprs[20][0] = 1u;
  atomic_state.vgprs[20][1] = 2u;
  atomic_state.vgprs[20][3] = 4u;
  atomic_state.vgprs[21][0] = 500u;
  atomic_state.vgprs[21][1] = 600u;
  atomic_state.vgprs[21][3] = 800u;
  atomic_state.vgprs[22][0] = 100u;
  atomic_state.vgprs[22][1] = 999u;
  atomic_state.vgprs[22][3] = 130u;
  atomic_state.vgprs[23][0] = 700u;
  atomic_state.vgprs[23][1] = 777u;
  atomic_state.vgprs[23][3] = 900u;
  atomic_state.vgprs[31][0] = 0xdeadbeefu;
  atomic_state.vgprs[31][2] = 0xdeadbeefu;
  const std::vector<DecodedInstruction> atomic_program = {
      DecodedInstruction::FiveOperand("GLOBAL_ATOMIC_ADD",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(14),
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("GLOBAL_ATOMIC_SWAP",
                                      InstructionOperand::Vgpr(16),
                                      InstructionOperand::Vgpr(21),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FiveOperand("GLOBAL_ATOMIC_CMPSWAP",
                                      InstructionOperand::Vgpr(31),
                                      InstructionOperand::Vgpr(18),
                                      InstructionOperand::Vgpr(22),
                                      InstructionOperand::Sgpr(0),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  if (!Expect(interpreter.ExecuteProgram(atomic_program, &atomic_state,
                                         &atomic_memory, &error_message),
              error_message.c_str()) ||
      !Expect(atomic_state.vgprs[30][0] == 10u,
              "expected lane 0 atomic add return value") ||
      !Expect(atomic_state.vgprs[30][1] == 20u,
              "expected lane 1 atomic add return value") ||
      !Expect(atomic_state.vgprs[30][2] == 0u,
              "expected inactive lane atomic add return to remain untouched") ||
      !Expect(atomic_state.vgprs[30][3] == 40u,
              "expected lane 3 atomic add return value") ||
      !Expect(atomic_state.vgprs[31][0] == 100u,
              "expected lane 0 atomic cmpswap return value") ||
      !Expect(atomic_state.vgprs[31][1] == 110u,
              "expected lane 1 atomic cmpswap return value") ||
      !Expect(atomic_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive lane atomic cmpswap destination to remain untouched") ||
      !Expect(atomic_state.vgprs[31][3] == 130u,
              "expected lane 3 atomic cmpswap return value")) {
    return 1;
  }

  std::uint32_t atomic_value = 0;
  if (!Expect(atomic_memory.ReadU32(0x520, &atomic_value),
              "expected lane 0 atomic add read") ||
      !Expect(atomic_value == 11u, "expected lane 0 atomic add result") ||
      !Expect(atomic_memory.ReadU32(0x524, &atomic_value),
              "expected lane 1 atomic add read") ||
      !Expect(atomic_value == 22u, "expected lane 1 atomic add result") ||
      !Expect(atomic_memory.ReadU32(0x52c, &atomic_value),
              "expected lane 3 atomic add read") ||
      !Expect(atomic_value == 44u, "expected lane 3 atomic add result") ||
      !Expect(atomic_memory.ReadU32(0x530, &atomic_value),
              "expected lane 0 atomic swap read") ||
      !Expect(atomic_value == 500u, "expected lane 0 atomic swap result") ||
      !Expect(atomic_memory.ReadU32(0x534, &atomic_value),
              "expected lane 1 atomic swap read") ||
      !Expect(atomic_value == 600u, "expected lane 1 atomic swap result") ||
      !Expect(atomic_memory.ReadU32(0x53c, &atomic_value),
              "expected lane 3 atomic swap read") ||
      !Expect(atomic_value == 800u, "expected lane 3 atomic swap result") ||
      !Expect(atomic_memory.ReadU32(0x540, &atomic_value),
              "expected lane 0 atomic cmpswap read") ||
      !Expect(atomic_value == 700u, "expected lane 0 atomic cmpswap result") ||
      !Expect(atomic_memory.ReadU32(0x544, &atomic_value),
              "expected lane 1 atomic cmpswap read") ||
      !Expect(atomic_value == 110u, "expected lane 1 atomic cmpswap mismatch result") ||
      !Expect(atomic_memory.ReadU32(0x54c, &atomic_value),
              "expected lane 3 atomic cmpswap read") ||
      !Expect(atomic_value == 900u, "expected lane 3 atomic cmpswap result")) {
    return 1;
  }

  const std::vector<DecodedInstruction> flat_atomic_program = {
      DecodedInstruction::FourOperand("FLAT_ATOMIC_ADD",
                                      InstructionOperand::Vgpr(30),
                                      InstructionOperand::Vgpr(14),
                                      InstructionOperand::Vgpr(20),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::ThreeOperand("FLAT_ATOMIC_SWAP",
                                       InstructionOperand::Vgpr(16),
                                       InstructionOperand::Vgpr(21),
                                       InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("FLAT_ATOMIC_CMPSWAP",
                                      InstructionOperand::Vgpr(31),
                                      InstructionOperand::Vgpr(18),
                                      InstructionOperand::Vgpr(22),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::FourOperand("FLAT_ATOMIC_ADD_X2",
                                      InstructionOperand::Vgpr(32),
                                      InstructionOperand::Vgpr(24),
                                      InstructionOperand::Vgpr(26),
                                      InstructionOperand::Imm32(0)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  auto seed_flat_atomic_memory = [](LinearExecutionMemory* memory) {
    if (memory == nullptr) {
      return false;
    }
    auto write_u64 = [&](std::uint64_t address, std::uint64_t value) {
      std::uint32_t low = 0;
      std::uint32_t high = 0;
      SplitU64(value, &low, &high);
      return memory->WriteU32(address, low) &&
             memory->WriteU32(address + 4u, high);
    };
    return memory->WriteU32(0x520, 10u) && memory->WriteU32(0x524, 20u) &&
           memory->WriteU32(0x52c, 40u) && memory->WriteU32(0x530, 50u) &&
           memory->WriteU32(0x534, 60u) && memory->WriteU32(0x53c, 80u) &&
           memory->WriteU32(0x540, 100u) && memory->WriteU32(0x544, 110u) &&
           memory->WriteU32(0x54c, 130u) && write_u64(0x560, 1000ULL) &&
           write_u64(0x568, 2000ULL) && write_u64(0x578, 3000ULL);
  };
  auto make_flat_atomic_state = []() {
    WaveExecutionState state;
    state.exec_mask = 0b1011ULL;
    auto set_lane_u64 = [&](std::uint16_t reg, std::size_t lane,
                            std::uint64_t value) {
      std::uint32_t low = 0;
      std::uint32_t high = 0;
      SplitU64(value, &low, &high);
      state.vgprs[reg][lane] = low;
      state.vgprs[reg + 1][lane] = high;
    };
    set_lane_u64(14, 0u, 0x520ULL);
    set_lane_u64(14, 1u, 0x524ULL);
    set_lane_u64(14, 3u, 0x52cULL);
    set_lane_u64(16, 0u, 0x530ULL);
    set_lane_u64(16, 1u, 0x534ULL);
    set_lane_u64(16, 3u, 0x53cULL);
    set_lane_u64(18, 0u, 0x540ULL);
    set_lane_u64(18, 1u, 0x544ULL);
    set_lane_u64(18, 3u, 0x54cULL);
    set_lane_u64(24, 0u, 0x560ULL);
    set_lane_u64(24, 1u, 0x568ULL);
    set_lane_u64(24, 3u, 0x578ULL);
    state.vgprs[20][0] = 1u;
    state.vgprs[20][1] = 2u;
    state.vgprs[20][3] = 4u;
    state.vgprs[21][0] = 500u;
    state.vgprs[21][1] = 600u;
    state.vgprs[21][3] = 800u;
    state.vgprs[22][0] = 100u;
    state.vgprs[22][1] = 999u;
    state.vgprs[22][3] = 130u;
    state.vgprs[23][0] = 700u;
    state.vgprs[23][1] = 777u;
    state.vgprs[23][3] = 900u;
    set_lane_u64(26, 0u, 3ULL);
    set_lane_u64(26, 1u, 5ULL);
    set_lane_u64(26, 3u, 9ULL);
    state.vgprs[31][0] = 0xdeadbeefu;
    state.vgprs[31][2] = 0xdeadbeefu;
    state.vgprs[32][2] = 0xdeadbeefu;
    state.vgprs[33][2] = 0xcafebabeu;
    return state;
  };
  auto validate_flat_atomic_state =
      [&](const WaveExecutionState& state,
          LinearExecutionMemory* memory,
          const char* mode) {
        if (memory == nullptr) {
          std::cerr << mode << " missing flat atomic memory\n";
          return false;
        }
        if (!Expect(state.halted, "expected flat atomic test program to halt")) {
          std::cerr << mode << '\n';
          return false;
        }
        if (!Expect(state.vgprs[30][0] == 10u,
                    "expected flat atomic add lane 0 return value") ||
            !Expect(state.vgprs[30][1] == 20u,
                    "expected flat atomic add lane 1 return value") ||
            !Expect(state.vgprs[30][2] == 0u,
                    "expected inactive flat atomic add return to remain untouched") ||
            !Expect(state.vgprs[30][3] == 40u,
                    "expected flat atomic add lane 3 return value") ||
            !Expect(state.vgprs[31][0] == 100u,
                    "expected flat atomic cmpswap lane 0 return value") ||
            !Expect(state.vgprs[31][1] == 110u,
                    "expected flat atomic cmpswap lane 1 return value") ||
            !Expect(state.vgprs[31][2] == 0xdeadbeefu,
                    "expected inactive flat atomic cmpswap destination to remain untouched") ||
            !Expect(state.vgprs[31][3] == 130u,
                    "expected flat atomic cmpswap lane 3 return value")) {
          std::cerr << mode << '\n';
          return false;
        }

        std::uint32_t read_value = 0;
        if (!Expect(memory->ReadU32(0x520, &read_value),
                    "expected flat atomic add lane 0 read") ||
            !Expect(read_value == 11u,
                    "expected flat atomic add lane 0 result") ||
            !Expect(memory->ReadU32(0x524, &read_value),
                    "expected flat atomic add lane 1 read") ||
            !Expect(read_value == 22u,
                    "expected flat atomic add lane 1 result") ||
            !Expect(memory->ReadU32(0x52c, &read_value),
                    "expected flat atomic add lane 3 read") ||
            !Expect(read_value == 44u,
                    "expected flat atomic add lane 3 result") ||
            !Expect(memory->ReadU32(0x530, &read_value),
                    "expected flat atomic swap lane 0 read") ||
            !Expect(read_value == 500u,
                    "expected flat atomic swap lane 0 result") ||
            !Expect(memory->ReadU32(0x534, &read_value),
                    "expected flat atomic swap lane 1 read") ||
            !Expect(read_value == 600u,
                    "expected flat atomic swap lane 1 result") ||
            !Expect(memory->ReadU32(0x53c, &read_value),
                    "expected flat atomic swap lane 3 read") ||
            !Expect(read_value == 800u,
                    "expected flat atomic swap lane 3 result") ||
            !Expect(memory->ReadU32(0x540, &read_value),
                    "expected flat atomic cmpswap lane 0 read") ||
            !Expect(read_value == 700u,
                    "expected flat atomic cmpswap lane 0 result") ||
            !Expect(memory->ReadU32(0x544, &read_value),
                    "expected flat atomic cmpswap lane 1 read") ||
            !Expect(read_value == 110u,
                    "expected flat atomic cmpswap mismatch result") ||
            !Expect(memory->ReadU32(0x54c, &read_value),
                    "expected flat atomic cmpswap lane 3 read") ||
            !Expect(read_value == 900u,
                    "expected flat atomic cmpswap lane 3 result")) {
          std::cerr << mode << '\n';
          return false;
        }

        static constexpr std::array<std::size_t, 3> kObservedLanes = {0u, 1u, 3u};
        static constexpr std::array<std::uint64_t, 3> kExpectedOldX2 = {
            1000ULL, 2000ULL, 3000ULL};
        static constexpr std::array<std::uint64_t, 3> kExpectedNewX2 = {
            1003ULL, 2005ULL, 3009ULL};
        static constexpr std::array<std::uint64_t, 3> kExpectedX2Addresses = {
            0x560ULL, 0x568ULL, 0x578ULL};
        for (std::size_t index = 0; index < kObservedLanes.size(); ++index) {
          const std::size_t lane = kObservedLanes[index];
          const std::uint64_t return_value =
              ComposeU64(state.vgprs[32][lane], state.vgprs[33][lane]);
          if (!Expect(return_value == kExpectedOldX2[index],
                      "expected flat atomic add x2 return value")) {
            std::cerr << mode << " lane=" << lane << '\n';
            return false;
          }
          std::uint32_t low = 0;
          std::uint32_t high = 0;
          if (!Expect(memory->ReadU32(kExpectedX2Addresses[index], &low),
                      "expected flat atomic add x2 low read") ||
              !Expect(memory->ReadU32(kExpectedX2Addresses[index] + 4u, &high),
                      "expected flat atomic add x2 high read") ||
              !Expect(ComposeU64(low, high) == kExpectedNewX2[index],
                      "expected flat atomic add x2 result")) {
            std::cerr << mode << " lane=" << lane << '\n';
            return false;
          }
        }

        if (!Expect(state.vgprs[32][2] == 0xdeadbeefu,
                    "expected inactive flat atomic add x2 low return to remain untouched") ||
            !Expect(state.vgprs[33][2] == 0xcafebabeu,
                    "expected inactive flat atomic add x2 high return to remain untouched")) {
          std::cerr << mode << '\n';
          return false;
        }
        return true;
      };

  LinearExecutionMemory decoded_flat_atomic_memory(0x1000, 0);
  if (!Expect(seed_flat_atomic_memory(&decoded_flat_atomic_memory),
              "expected decoded flat atomic seed writes")) {
    return 1;
  }
  WaveExecutionState decoded_flat_atomic_state = make_flat_atomic_state();
  if (!Expect(interpreter.ExecuteProgram(flat_atomic_program,
                                         &decoded_flat_atomic_state,
                                         &decoded_flat_atomic_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_flat_atomic_state(decoded_flat_atomic_state,
                                  &decoded_flat_atomic_memory, "decoded")) {
    return 1;
  }

  std::vector<CompiledInstruction> compiled_flat_atomic_program;
  if (!Expect(interpreter.CompileProgram(flat_atomic_program,
                                         &compiled_flat_atomic_program,
                                         &error_message),
              error_message.c_str())) {
    return 1;
  }
  LinearExecutionMemory compiled_flat_atomic_memory(0x1000, 0);
  if (!Expect(seed_flat_atomic_memory(&compiled_flat_atomic_memory),
              "expected compiled flat atomic seed writes")) {
    return 1;
  }
  WaveExecutionState compiled_flat_atomic_state = make_flat_atomic_state();
  if (!Expect(interpreter.ExecuteProgram(compiled_flat_atomic_program,
                                         &compiled_flat_atomic_state,
                                         &compiled_flat_atomic_memory,
                                         &error_message),
              error_message.c_str()) ||
      !validate_flat_atomic_state(compiled_flat_atomic_state,
                                  &compiled_flat_atomic_memory, "compiled")) {
    return 1;
  }

  const std::vector<AtomicSemanticCase> atomic_cases = {
      {"GLOBAL_ATOMIC_SUB", 1, 1, OneDword(10u), OneDword(3u), OneDword(7u),
       OneDword(10u)},
      {"GLOBAL_ATOMIC_SMIN", 1, 1, OneDword(7u), OneDword(0xfffffffbu),
       OneDword(0xfffffffbu), OneDword(7u)},
      {"GLOBAL_ATOMIC_UMIN", 1, 1, OneDword(7u), OneDword(5u), OneDword(5u),
       OneDword(7u)},
      {"GLOBAL_ATOMIC_SMAX", 1, 1, OneDword(0xfffffffbu), OneDword(7u),
       OneDword(7u), OneDword(0xfffffffbu)},
      {"GLOBAL_ATOMIC_UMAX", 1, 1, OneDword(7u), OneDword(9u), OneDword(9u),
       OneDword(7u)},
      {"GLOBAL_ATOMIC_AND", 1, 1, OneDword(0x0000f0f0u), OneDword(0x00000ff0u),
       OneDword(0x000000f0u), OneDword(0x0000f0f0u)},
      {"GLOBAL_ATOMIC_OR", 1, 1, OneDword(0x0000f000u), OneDword(0x00000f0fu),
       OneDword(0x0000ff0fu), OneDword(0x0000f000u)},
      {"GLOBAL_ATOMIC_XOR", 1, 1, OneDword(0xaaaa5555u), OneDword(0x00ff00ffu),
       OneDword(0xaa5555aau), OneDword(0xaaaa5555u)},
      {"GLOBAL_ATOMIC_INC", 1, 1, OneDword(3u), OneDword(5u), OneDword(4u),
       OneDword(3u)},
      {"GLOBAL_ATOMIC_DEC", 1, 1, OneDword(7u), OneDword(5u), OneDword(5u),
       OneDword(7u)},
      {"GLOBAL_ATOMIC_ADD_F32", 1, 1, OneDword(0x3fc00000u),
       OneDword(0x40100000u), OneDword(0x40700000u), OneDword(0x3fc00000u)},
      {"GLOBAL_ATOMIC_PK_ADD_F16", 1, 1, OneDword(0x40003c00u),
       OneDword(0x3c004000u), OneDword(0x42004200u), OneDword(0x40003c00u)},
      {"GLOBAL_ATOMIC_PK_ADD_BF16", 1, 1, OneDword(0x40003f80u),
       OneDword(0x3f804000u), OneDword(0x40404040u), OneDword(0x40003f80u)},
      {"GLOBAL_ATOMIC_ADD_F64", 2, 2, TwoDwords(0x3ff8000000000000ULL),
       TwoDwords(0x4002000000000000ULL), TwoDwords(0x400e000000000000ULL),
       TwoDwords(0x3ff8000000000000ULL)},
      {"GLOBAL_ATOMIC_MIN_F64", 2, 2, TwoDwords(0x4014000000000000ULL),
       TwoDwords(0x4000000000000000ULL), TwoDwords(0x4000000000000000ULL),
       TwoDwords(0x4014000000000000ULL)},
      {"GLOBAL_ATOMIC_MAX_F64", 2, 2, TwoDwords(0x4014000000000000ULL),
       TwoDwords(0x401c000000000000ULL), TwoDwords(0x401c000000000000ULL),
       TwoDwords(0x4014000000000000ULL)},
      {"GLOBAL_ATOMIC_SWAP_X2", 2, 2, TwoDwords(0x1111222233334444ULL),
       TwoDwords(0xaaaabbbbccccddddULL), TwoDwords(0xaaaabbbbccccddddULL),
       TwoDwords(0x1111222233334444ULL)},
      {"GLOBAL_ATOMIC_CMPSWAP_X2", 2, 4, TwoDwords(0x1111222233334444ULL),
       FourDwords(0x1111222233334444ULL, 0x5555666677778888ULL),
       TwoDwords(0x5555666677778888ULL), TwoDwords(0x1111222233334444ULL)},
      {"GLOBAL_ATOMIC_ADD_X2", 2, 2, TwoDwords(10ULL), TwoDwords(3ULL),
       TwoDwords(13ULL), TwoDwords(10ULL)},
      {"GLOBAL_ATOMIC_SUB_X2", 2, 2, TwoDwords(10ULL), TwoDwords(3ULL),
       TwoDwords(7ULL), TwoDwords(10ULL)},
      {"GLOBAL_ATOMIC_SMIN_X2", 2, 2, TwoDwords(7ULL),
       TwoDwords(0xfffffffffffffffbULL), TwoDwords(0xfffffffffffffffbULL),
       TwoDwords(7ULL)},
      {"GLOBAL_ATOMIC_UMIN_X2", 2, 2, TwoDwords(7ULL), TwoDwords(5ULL),
       TwoDwords(5ULL), TwoDwords(7ULL)},
      {"GLOBAL_ATOMIC_SMAX_X2", 2, 2, TwoDwords(0xfffffffffffffffbULL),
       TwoDwords(7ULL), TwoDwords(7ULL), TwoDwords(0xfffffffffffffffbULL)},
      {"GLOBAL_ATOMIC_UMAX_X2", 2, 2, TwoDwords(7ULL), TwoDwords(9ULL),
       TwoDwords(9ULL), TwoDwords(7ULL)},
      {"GLOBAL_ATOMIC_AND_X2", 2, 2, TwoDwords(0x00ff00ff00ff00ffULL),
       TwoDwords(0x0f0f0f0f0f0f0f0fULL), TwoDwords(0x000f000f000f000fULL),
       TwoDwords(0x00ff00ff00ff00ffULL)},
      {"GLOBAL_ATOMIC_OR_X2", 2, 2, TwoDwords(0xf000f000f000f000ULL),
       TwoDwords(0x0f0f0f0f0f0f0f0fULL), TwoDwords(0xff0fff0fff0fff0fULL),
       TwoDwords(0xf000f000f000f000ULL)},
      {"GLOBAL_ATOMIC_XOR_X2", 2, 2, TwoDwords(0xaaaa5555aaaa5555ULL),
       TwoDwords(0x00ff00ff00ff00ffULL), TwoDwords(0xaa5555aaaa5555aaULL),
       TwoDwords(0xaaaa5555aaaa5555ULL)},
      {"GLOBAL_ATOMIC_INC_X2", 2, 2, TwoDwords(3ULL), TwoDwords(5ULL),
       TwoDwords(4ULL), TwoDwords(3ULL)},
      {"GLOBAL_ATOMIC_DEC_X2", 2, 2, TwoDwords(7ULL), TwoDwords(5ULL),
       TwoDwords(5ULL), TwoDwords(7ULL)},
  };
  for (const AtomicSemanticCase& test_case : atomic_cases) {
    if (!RunAtomicSemanticCase(interpreter, test_case)) {
      return 1;
    }
  }

  for (const AtomicSemanticCase& test_case : atomic_cases) {
    const std::string buffer_opcode =
        "BUFFER_" + std::string(test_case.opcode.substr(7));
    AtomicSemanticCase buffer_case = test_case;
    buffer_case.opcode = buffer_opcode;
    if (!RunBufferAtomicSemanticCase(interpreter, buffer_case, true)) {
      return 1;
    }
    if (test_case.opcode == "GLOBAL_ATOMIC_SWAP" ||
        test_case.opcode == "GLOBAL_ATOMIC_CMPSWAP" ||
        test_case.opcode == "GLOBAL_ATOMIC_SWAP_X2" ||
        test_case.opcode == "GLOBAL_ATOMIC_CMPSWAP_X2") {
      if (!RunBufferAtomicSemanticCase(interpreter, buffer_case, false)) {
        return 1;
      }
    }
  }

  return 0;
}
