#include <array>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/execution_memory.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx950/binary_decoder.h"
#include "lib/sim/isa/gfx950/interpreter.h"
#include "lib/sim/isa/instruction_catalog.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
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

constexpr std::uint32_t MakeSopk(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t simm16) {
  std::uint32_t word = 0;
  word = SetBits(word, 0xb, 28, 4);
  word = SetBits(word, op, 23, 5);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

constexpr std::uint32_t MakeVop1(std::uint32_t op,
                                 std::uint32_t vdst,
                                 std::uint32_t src0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x3f, 25, 7);
  word = SetBits(word, vdst, 17, 8);
  word = SetBits(word, op, 9, 8);
  word = SetBits(word, src0, 0, 9);
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

constexpr std::uint32_t MakeVopc(std::uint32_t op,
                                 std::uint32_t src0,
                                 std::uint32_t vsrc1) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x3e, 25, 7);
  word = SetBits(word, op, 17, 8);
  word = SetBits(word, vsrc1, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

std::array<std::uint32_t, 2> MakeVop3(std::uint32_t op,
                                      std::uint32_t vdst,
                                      std::uint32_t src0,
                                      std::uint32_t src1,
                                      std::uint32_t src2 = 0) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(op & 0x3ffu) << 16;
  word |= static_cast<std::uint64_t>(0x34u) << 26;
  word |= static_cast<std::uint64_t>(src0 & 0x1ffu) << 32;
  word |= static_cast<std::uint64_t>(src1 & 0x1ffu) << 41;
  word |= static_cast<std::uint64_t>(src2 & 0x1ffu) << 50;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeVop3Sdst(std::uint32_t op,
                                          std::uint32_t vdst,
                                          std::uint32_t sdst,
                                          std::uint32_t src0,
                                          std::uint32_t src1,
                                          std::uint32_t src2 = 0) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 0;
  word |= static_cast<std::uint64_t>(sdst & 0x7fu) << 8;
  word |= static_cast<std::uint64_t>(op & 0x3ffu) << 16;
  word |= static_cast<std::uint64_t>(0x34u) << 26;
  word |= static_cast<std::uint64_t>(src0 & 0x1ffu) << 32;
  word |= static_cast<std::uint64_t>(src1 & 0x1ffu) << 41;
  word |= static_cast<std::uint64_t>(src2 & 0x1ffu) << 50;
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

std::array<std::uint32_t, 2> MakeFlat(std::uint32_t op,
                                      std::uint32_t vdst,
                                      std::uint32_t addr,
                                      std::uint32_t data,
                                      std::uint32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(offset & 0xfffu) << 0;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
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

std::array<std::uint32_t, 2> MakeGlobal(std::uint32_t op,
                                        std::uint32_t vdst,
                                        std::uint32_t addr,
                                        std::uint32_t data,
                                        std::uint32_t saddr,
                                        std::int32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(offset & 0x1fffu) << 0;
  word |= static_cast<std::uint64_t>(2u) << 14;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(saddr & 0x7fu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeGlobalAtomic(std::uint32_t op,
                                              bool return_prior_value,
                                              std::uint32_t vdst,
                                              std::uint32_t addr,
                                              std::uint32_t data,
                                              std::uint32_t saddr,
                                              std::int32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(offset & 0x1fffu) << 0;
  word |= static_cast<std::uint64_t>(2u) << 14;
  word |= static_cast<std::uint64_t>(return_prior_value ? 1u : 0u) << 16;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(saddr & 0x7fu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeFlatAtomic(std::uint32_t op,
                                            bool return_prior_value,
                                            std::uint32_t vdst,
                                            std::uint32_t addr,
                                            std::uint32_t data,
                                            std::uint32_t offset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(55u) << 26;
  word |= static_cast<std::uint64_t>(offset & 0xfffu) << 0;
  word |= static_cast<std::uint64_t>(return_prior_value ? 1u : 0u) << 16;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
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

std::uint64_t ComposeU64(std::uint32_t low, std::uint32_t high) {
  return static_cast<std::uint64_t>(low) |
         (static_cast<std::uint64_t>(high) << 32);
}

struct SaveexecBinaryCase {
  std::string_view opcode;
  std::uint64_t initial_exec = 0;
  std::uint64_t source = 0;
  std::uint64_t expected_exec = 0;
};

struct ScalarUnaryBinaryCase {
  std::string_view opcode;
  std::uint32_t source = 0;
  std::uint32_t initial_dest = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarBinaryBinaryCase {
  std::string_view opcode;
  std::uint32_t lhs = 0;
  std::uint32_t rhs = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairUnaryBinaryCase {
  std::string_view opcode;
  std::uint64_t source = 0;
  std::uint64_t initial_dest = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairFromScalarUnaryBinaryCase {
  std::string_view opcode;
  std::uint32_t source = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarFromPairUnaryBinaryCase {
  std::string_view opcode;
  std::uint64_t source = 0;
  std::uint32_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairBinaryBinaryCase {
  std::string_view opcode;
  std::uint64_t lhs = 0;
  std::uint64_t rhs = 0;
  std::uint64_t expected = 0;
  bool initial_scc = false;
  bool expected_scc = false;
};

struct ScalarPairCompareBinaryCase {
  std::string_view opcode;
  std::uint64_t lhs = 0;
  std::uint64_t rhs = 0;
  bool expected_scc = false;
};

bool RunSaveexecBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const SaveexecBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP1");
  if (!Expect(opcode.has_value(), "expected saveexec opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(*opcode, 40, 20),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }
  if (!Expect(decoded_program.size() == 2, "expected saveexec program size") ||
      !Expect(decoded_program[0].opcode == test_case.opcode,
              "expected saveexec opcode decode")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  WaveExecutionState state;
  state.exec_mask = test_case.initial_exec;
  SplitU64(test_case.source, &state.sgprs[20], &state.sgprs[21]);
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  const std::uint64_t saved_exec = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected saveexec binary test to halt") ||
      !Expect(saved_exec == test_case.initial_exec,
              "expected binary saveexec previous exec capture") ||
      !Expect(state.exec_mask == test_case.expected_exec,
              "expected binary saveexec final exec") ||
      !Expect(state.scc == (test_case.expected_exec != 0),
              "expected binary saveexec scc")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << state.exec_mask
              << " expected=0x" << test_case.expected_exec << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarUnaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarUnaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP1");
  if (!Expect(opcode.has_value(), "expected scalar unary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(*opcode, 40, 20),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.source;
  state.sgprs[40] = test_case.initial_dest;
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  if (!Expect(state.halted, "expected scalar unary binary test to halt") ||
      !Expect(state.sgprs[40] == test_case.expected,
              "expected scalar unary binary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar unary binary SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << state.sgprs[40]
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarBinaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarBinaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP2");
  if (!Expect(opcode.has_value(), "expected scalar binary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop2(*opcode, 40, 20, 24),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.lhs;
  state.sgprs[24] = test_case.rhs;
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  if (!Expect(state.halted, "expected scalar binary decoder test to halt") ||
      !Expect(state.sgprs[40] == test_case.expected,
              "expected scalar binary decoder result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar binary decoder SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << state.sgprs[40]
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairUnaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairUnaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP1");
  if (!Expect(opcode.has_value(), "expected scalar pair unary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(*opcode, 40, 20),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  SplitU64(test_case.initial_dest, &state.sgprs[40], &state.sgprs[41]);
  SplitU64(test_case.source, &state.sgprs[20], &state.sgprs[21]);
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected scalar pair unary binary test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair unary binary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair unary binary SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << result
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairFromScalarUnaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairFromScalarUnaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP1");
  if (!Expect(opcode.has_value(),
              "expected scalar pair-from-scalar unary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(*opcode, 40, 20),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  state.sgprs[20] = test_case.source;
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted,
              "expected scalar pair-from-scalar unary binary test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair-from-scalar unary binary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair-from-scalar unary binary SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << result
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarFromPairUnaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarFromPairUnaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP1");
  if (!Expect(opcode.has_value(), "expected scalar-from-pair unary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(*opcode, 40, 20),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }
  if (!Expect(decoded_program.size() == 2,
              "expected scalar-from-pair unary program size") ||
      !Expect(decoded_program[0].opcode == test_case.opcode,
              "expected scalar-from-pair unary opcode decode")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  SplitU64(test_case.source, &state.sgprs[20], &state.sgprs[21]);
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  if (!Expect(state.halted, "expected scalar-from-pair unary binary test to halt") ||
      !Expect(state.sgprs[40] == test_case.expected,
              "expected scalar-from-pair unary binary result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar-from-pair unary binary SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << state.sgprs[40]
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairBinaryBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairBinaryBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOP2");
  if (!Expect(opcode.has_value(), "expected scalar pair binary opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSop2(*opcode, 40, 20, 24),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  state.scc = test_case.initial_scc;
  SplitU64(test_case.lhs, &state.sgprs[20], &state.sgprs[21]);
  SplitU64(test_case.rhs, &state.sgprs[24], &state.sgprs[25]);
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  const std::uint64_t result = ComposeU64(state.sgprs[40], state.sgprs[41]);
  if (!Expect(state.halted, "expected scalar pair binary decoder test to halt") ||
      !Expect(result == test_case.expected,
              "expected scalar pair binary decoder result") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair binary decoder SCC")) {
    std::cerr << test_case.opcode << " actual=0x" << std::hex << result
              << " expected=0x" << test_case.expected << std::dec << '\n';
    return false;
  }
  return true;
}

bool RunScalarPairCompareBinaryCase(
    const mirage::sim::isa::Gfx950BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx950Interpreter& interpreter,
    const ScalarPairCompareBinaryCase& test_case) {
  using namespace mirage::sim::isa;

  const auto opcode = FindDefaultEncodingOpcode(test_case.opcode, "ENC_SOPC");
  if (!Expect(opcode.has_value(), "expected scalar pair compare opcode lookup")) {
    std::cerr << test_case.opcode << '\n';
    return false;
  }

  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;
  const std::vector<std::uint32_t> encoded_program = {
      MakeSopc(*opcode, 20, 24),
      MakeSopp(1),
  };
  if (!decoder.DecodeProgram(encoded_program, &decoded_program, &error_message)) {
    std::cerr << test_case.opcode << " decode: " << error_message << '\n';
    return false;
  }

  WaveExecutionState state;
  SplitU64(test_case.lhs, &state.sgprs[20], &state.sgprs[21]);
  SplitU64(test_case.rhs, &state.sgprs[24], &state.sgprs[25]);
  if (!interpreter.ExecuteProgram(decoded_program, &state, &error_message)) {
    std::cerr << test_case.opcode << " execute: " << error_message << '\n';
    return false;
  }

  if (!Expect(state.halted,
              "expected scalar pair compare decoder test to halt") ||
      !Expect(state.scc == test_case.expected_scc,
              "expected scalar pair compare decoder SCC")) {
    std::cerr << test_case.opcode << " actual=" << state.scc
              << " expected=" << test_case.expected_scc << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  Gfx950BinaryDecoder decoder;
  std::vector<DecodedInstruction> decoded_program;
  std::string error_message;

  const std::vector<std::uint32_t> encoded_program = {
      MakeSop1(0, 0, 135),          // s_mov_b32 s0, 7
      MakeSop1(0, 1, 133),          // s_mov_b32 s1, 5
      MakeSop2(0, 2, 0, 1),         // s_add_u32 s2, s0, s1
      MakeSop2(16, 3, 2, 143),      // s_xor_b32 s3, s2, 15
      MakeVop1(1, 0, 131),          // v_mov_b32 v0, 3
      MakeVop1(1, 1, 2),            // v_mov_b32 v1, s2
      MakeVop2(52, 2, 256, 1),      // v_add_u32 v2, v0, v1
      MakeVop2(53, 3, 258, 0),      // v_sub_u32 v3, v2, v0
      MakeSopp(1),                  // s_endpgm
  };

  if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program, &error_message),
              error_message.c_str())) {
    return 1;
  }

  if (!Expect(decoded_program.size() == 9, "expected 9 decoded instructions") ||
      !Expect(decoded_program[0].opcode == "S_MOV_B32", "expected SOP1 decode") ||
      !Expect(decoded_program[2].opcode == "S_ADD_U32", "expected SOP2 decode") ||
      !Expect(decoded_program[4].opcode == "V_MOV_B32", "expected VOP1 decode") ||
      !Expect(decoded_program[6].opcode == "V_ADD_U32", "expected VOP2 decode") ||
      !Expect(decoded_program[8].opcode == "S_ENDPGM", "expected SOPP decode")) {
    return 1;
  }

  if (!Expect(decoded_program[0].operands[1].kind == OperandKind::kImm32,
              "expected SOP1 inline constant decode") ||
      !Expect(decoded_program[0].operands[1].imm32 == 7,
              "expected inline constant value 7") ||
      !Expect(decoded_program[6].operands[1].kind == OperandKind::kVgpr,
              "expected VOP2 VGPR source decode") ||
      !Expect(decoded_program[6].operands[1].index == 0,
              "expected VGPR source index 0")) {
    return 1;
  }

  Gfx950Interpreter interpreter;
  WaveExecutionState state;
  state.exec_mask = 0b1011ULL;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &state, &error_message),
              error_message.c_str())) {
    return 1;
  }

  if (!Expect(state.halted, "expected binary program to halt") ||
      !Expect(state.sgprs[2] == 12, "expected decoded S_ADD_U32 result") ||
      !Expect(state.sgprs[3] == (12U ^ 15U), "expected decoded S_XOR_B32 result") ||
      !Expect(state.vgprs[2][0] == 15, "expected decoded V_ADD_U32 lane 0 result") ||
      !Expect(state.vgprs[2][2] == 0, "expected inactive lane to remain untouched") ||
      !Expect(state.vgprs[3][0] == 12, "expected decoded V_SUB_U32 lane 0 result")) {
    return 1;
  }

  const std::array<SaveexecBinaryCase, 10> kSaveexecCases = {{
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
  for (const SaveexecBinaryCase& test_case : kSaveexecCases) {
    if (!RunSaveexecBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarUnaryBinaryCase, 19> kScalarUnaryCases = {{
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
  for (const ScalarUnaryBinaryCase& test_case : kScalarUnaryCases) {
    if (!RunScalarUnaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarBinaryBinaryCase, 30> kScalarBinaryCases = {{
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
  for (const ScalarBinaryBinaryCase& test_case : kScalarBinaryCases) {
    if (!RunScalarBinaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarPairUnaryBinaryCase, 9> kScalarPairUnaryCases = {{
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
  for (const ScalarPairUnaryBinaryCase& test_case : kScalarPairUnaryCases) {
    if (!RunScalarPairUnaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarPairFromScalarUnaryBinaryCase, 1>
      kScalarPairFromScalarUnaryCases = {{
          {"S_BITREPLICATE_B64_B32", 0x00000005U, 0x0000000000000033ULL, true,
           true},
      }};
  for (const ScalarPairFromScalarUnaryBinaryCase& test_case :
       kScalarPairFromScalarUnaryCases) {
    if (!RunScalarPairFromScalarUnaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarFromPairUnaryBinaryCase, 6> kScalarFromPairUnaryCases = {{
      {"S_BCNT0_I32_B64", 0xffffffffffffffffULL, 0U, true, false},
      {"S_BCNT1_I32_B64", 0xf0f0f0f00f0f0f0fULL, 32U, false, true},
      {"S_FF0_I32_B64", 0xfffffffffffffff7ULL, 3U, true, true},
      {"S_FF1_I32_B64", 0x0000000000000010ULL, 4U, false, false},
      {"S_FLBIT_I32_B64", 0x0000001000000000ULL, 27U, true, true},
      {"S_FLBIT_I32_I64", 0xfffffffffffffff0ULL, 60U, false, false},
  }};
  for (const ScalarFromPairUnaryBinaryCase& test_case :
       kScalarFromPairUnaryCases) {
    if (!RunScalarFromPairUnaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarPairBinaryBinaryCase, 17> kScalarPairBinaryCases = {{
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
  for (const ScalarPairBinaryBinaryCase& test_case : kScalarPairBinaryCases) {
    if (!RunScalarPairBinaryBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const std::array<ScalarPairCompareBinaryCase, 4> kScalarPairCompareCases = {{
      {"S_CMP_EQ_U64", 0x123456789abcdef0ULL, 0x123456789abcdef0ULL, true},
      {"S_CMP_EQ_U64", 0x123456789abcdef0ULL, 0x123456789abcdef1ULL, false},
      {"S_CMP_LG_U64", 0x0000000000000001ULL, 0x0000000000000002ULL, true},
      {"S_CMP_LG_U64", 0xabcdef0123456789ULL, 0xabcdef0123456789ULL, false},
  }};
  for (const ScalarPairCompareBinaryCase& test_case : kScalarPairCompareCases) {
    if (!RunScalarPairCompareBinaryCase(decoder, interpreter, test_case)) {
      return 1;
    }
  }

  const auto sopk_mov_opcode =
      FindDefaultEncodingOpcode("S_MOVK_I32", "ENC_SOPK");
  const auto sopk_cmov_opcode =
      FindDefaultEncodingOpcode("S_CMOVK_I32", "ENC_SOPK");
  const auto sopk_add_opcode =
      FindDefaultEncodingOpcode("S_ADDK_I32", "ENC_SOPK");
  const auto sopk_mul_opcode =
      FindDefaultEncodingOpcode("S_MULK_I32", "ENC_SOPK");
  const auto sopk_cmp_lt_i32_opcode =
      FindDefaultEncodingOpcode("S_CMPK_LT_I32", "ENC_SOPK");
  const auto sopk_cmp_gt_u32_opcode =
      FindDefaultEncodingOpcode("S_CMPK_GT_U32", "ENC_SOPK");
  if (!Expect(sopk_mov_opcode.has_value(), "expected S_MOVK_I32 opcode lookup") ||
      !Expect(sopk_cmov_opcode.has_value(),
              "expected S_CMOVK_I32 opcode lookup") ||
      !Expect(sopk_add_opcode.has_value(), "expected S_ADDK_I32 opcode lookup") ||
      !Expect(sopk_mul_opcode.has_value(), "expected S_MULK_I32 opcode lookup") ||
      !Expect(sopk_cmp_lt_i32_opcode.has_value(),
              "expected S_CMPK_LT_I32 opcode lookup") ||
      !Expect(sopk_cmp_gt_u32_opcode.has_value(),
              "expected S_CMPK_GT_U32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> sopk_program = {
      MakeSopk(*sopk_mov_opcode, 0, 0xfffd),         // s_movk_i32 s0, -3
      MakeSopk(*sopk_add_opcode, 0, 5),              // s_addk_i32 s0, 5
      MakeSopk(*sopk_mul_opcode, 0, 0xfffc),         // s_mulk_i32 s0, -4
      MakeSopk(*sopk_cmp_lt_i32_opcode, 0, 0xffff),  // s_cmpk_lt_i32 s0, -1
      MakeSopk(*sopk_cmov_opcode, 1, 7),             // s_cmovk_i32 s1, 7
      MakeSopk(*sopk_cmp_gt_u32_opcode, 0, 1),       // s_cmpk_gt_u32 s0, 1
      MakeSopk(*sopk_cmov_opcode, 2, 0xfff9),        // s_cmovk_i32 s2, -7
      MakeSopp(1),                                   // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(sopk_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 8, "expected decoded sopk program size") ||
      !Expect(decoded_program[0].opcode == "S_MOV_B32",
              "expected S_MOVK_I32 normalization") ||
      !Expect(decoded_program[1].opcode == "S_ADD_U32",
              "expected S_ADDK_I32 normalization") ||
      !Expect(decoded_program[2].opcode == "S_MUL_I32",
              "expected S_MULK_I32 normalization") ||
      !Expect(decoded_program[3].opcode == "S_CMP_LT_I32",
              "expected S_CMPK_LT_I32 normalization") ||
      !Expect(decoded_program[4].opcode == "S_CMOV_B32",
              "expected S_CMOVK_I32 normalization") ||
      !Expect(decoded_program[5].opcode == "S_CMP_GT_U32",
              "expected S_CMPK_GT_U32 normalization") ||
      !Expect(decoded_program[6].opcode == "S_CMOV_B32",
              "expected second S_CMOVK_I32 normalization") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kImm32,
              "expected movk immediate decode") ||
      !Expect(decoded_program[0].operands[1].imm32 ==
                  static_cast<std::uint32_t>(-3),
              "expected movk immediate sign extension") ||
      !Expect(decoded_program[4].operands[1].imm32 == 7u,
              "expected first cmovk immediate value") ||
      !Expect(decoded_program[6].operands[1].imm32 ==
                  static_cast<std::uint32_t>(-7),
              "expected second cmovk immediate sign extension") ||
      !Expect(decoded_program[1].operands[1].kind == OperandKind::kSgpr,
              "expected addk source to reuse destination sgpr") ||
      !Expect(decoded_program[1].operands[1].index == 0,
              "expected addk source sgpr index") ||
      !Expect(decoded_program[1].operands[2].imm32 == 5u,
              "expected addk immediate value")) {
    return 1;
  }

  WaveExecutionState sopk_state;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &sopk_state, &error_message),
              error_message.c_str()) ||
      !Expect(sopk_state.halted, "expected sopk program to halt") ||
      !Expect(sopk_state.sgprs[0] == 0xfffffff8u,
              "expected sopk scalar result") ||
      !Expect(sopk_state.sgprs[1] == 7u,
              "expected first sopk cmov result") ||
      !Expect(sopk_state.sgprs[2] == 0xfffffff9u,
              "expected second sopk cmov result") ||
      !Expect(sopk_state.scc, "expected final sopk compare to set SCC")) {
    return 1;
  }

  const auto v_min_i32_opcode =
      FindDefaultEncodingOpcode("V_MIN_I32", "ENC_VOP2");
  const auto v_max_i32_opcode =
      FindDefaultEncodingOpcode("V_MAX_I32", "ENC_VOP2");
  const auto v_min_u32_opcode =
      FindDefaultEncodingOpcode("V_MIN_U32", "ENC_VOP2");
  const auto v_max_u32_opcode =
      FindDefaultEncodingOpcode("V_MAX_U32", "ENC_VOP2");
  const auto v_lshlrev_b32_opcode =
      FindDefaultEncodingOpcode("V_LSHLREV_B32", "ENC_VOP2");
  const auto v_lshrrev_b32_opcode =
      FindDefaultEncodingOpcode("V_LSHRREV_B32", "ENC_VOP2");
  const auto v_ashrrev_i32_opcode =
      FindDefaultEncodingOpcode("V_ASHRREV_I32", "ENC_VOP2");
  const auto v_and_b32_opcode =
      FindDefaultEncodingOpcode("V_AND_B32", "ENC_VOP2");
  const auto v_or_b32_opcode =
      FindDefaultEncodingOpcode("V_OR_B32", "ENC_VOP2");
  const auto v_xor_b32_opcode =
      FindDefaultEncodingOpcode("V_XOR_B32", "ENC_VOP2");
  if (!Expect(v_min_i32_opcode.has_value(), "expected V_MIN_I32 opcode lookup") ||
      !Expect(v_max_i32_opcode.has_value(), "expected V_MAX_I32 opcode lookup") ||
      !Expect(v_min_u32_opcode.has_value(), "expected V_MIN_U32 opcode lookup") ||
      !Expect(v_max_u32_opcode.has_value(), "expected V_MAX_U32 opcode lookup") ||
      !Expect(v_lshlrev_b32_opcode.has_value(),
              "expected V_LSHLREV_B32 opcode lookup") ||
      !Expect(v_lshrrev_b32_opcode.has_value(),
              "expected V_LSHRREV_B32 opcode lookup") ||
      !Expect(v_ashrrev_i32_opcode.has_value(),
              "expected V_ASHRREV_I32 opcode lookup") ||
      !Expect(v_and_b32_opcode.has_value(), "expected V_AND_B32 opcode lookup") ||
      !Expect(v_or_b32_opcode.has_value(), "expected V_OR_B32 opcode lookup") ||
      !Expect(v_xor_b32_opcode.has_value(), "expected V_XOR_B32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_vop2_program = {
      MakeSop1(0, 0, 255), 0xfffffff0u,                 // s_mov_b32 s0, -16
      MakeSop1(0, 1, 143),                               // s_mov_b32 s1, 15
      MakeSop1(0, 2, 132),                               // s_mov_b32 s2, 4
      MakeSop1(0, 3, 130),                               // s_mov_b32 s3, 2
      MakeVop2(*v_min_i32_opcode, 10, 0, 0),             // v_min_i32 v10, s0, v0
      MakeVop2(*v_max_i32_opcode, 11, 0, 0),             // v_max_i32 v11, s0, v0
      MakeVop2(*v_min_u32_opcode, 12, 1, 1),             // v_min_u32 v12, s1, v1
      MakeVop2(*v_max_u32_opcode, 13, 1, 1),             // v_max_u32 v13, s1, v1
      MakeVop2(*v_lshlrev_b32_opcode, 14, 2, 2),         // v_lshlrev_b32 v14, s2, v2
      MakeVop2(*v_lshrrev_b32_opcode, 15, 3, 3),         // v_lshrrev_b32 v15, s3, v3
      MakeVop2(*v_ashrrev_i32_opcode, 16, 3, 3),         // v_ashrrev_i32 v16, s3, v3
      MakeVop2(*v_and_b32_opcode, 17, 1, 13),            // v_and_b32 v17, s1, v13
      MakeVop2(*v_or_b32_opcode, 18, 1, 12),             // v_or_b32 v18, s1, v12
      MakeVop2(*v_xor_b32_opcode, 19, 1, 13),            // v_xor_b32 v19, s1, v13
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_vop2_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 15,
              "expected decoded vector vop2 program size") ||
      !Expect(decoded_program[4].opcode == "V_MIN_I32",
              "expected V_MIN_I32 decode") ||
      !Expect(decoded_program[5].opcode == "V_MAX_I32",
              "expected V_MAX_I32 decode") ||
      !Expect(decoded_program[6].opcode == "V_MIN_U32",
              "expected V_MIN_U32 decode") ||
      !Expect(decoded_program[7].opcode == "V_MAX_U32",
              "expected V_MAX_U32 decode") ||
      !Expect(decoded_program[8].opcode == "V_LSHLREV_B32",
              "expected V_LSHLREV_B32 decode") ||
      !Expect(decoded_program[9].opcode == "V_LSHRREV_B32",
              "expected V_LSHRREV_B32 decode") ||
      !Expect(decoded_program[10].opcode == "V_ASHRREV_I32",
              "expected V_ASHRREV_I32 decode") ||
      !Expect(decoded_program[11].opcode == "V_AND_B32",
              "expected V_AND_B32 decode") ||
      !Expect(decoded_program[12].opcode == "V_OR_B32",
              "expected V_OR_B32 decode") ||
      !Expect(decoded_program[13].opcode == "V_XOR_B32",
              "expected V_XOR_B32 decode") ||
      !Expect(decoded_program[8].operands[1].kind == OperandKind::kSgpr,
              "expected vector shift SGPR source decode") ||
      !Expect(decoded_program[8].operands[1].index == 2,
              "expected vector shift SGPR source index") ||
      !Expect(decoded_program[11].operands[2].kind == OperandKind::kVgpr,
              "expected vector logic VGPR source decode") ||
      !Expect(decoded_program[11].operands[2].index == 13,
              "expected vector logic VGPR source index")) {
    return 1;
  }

  WaveExecutionState vector_vop2_state;
  vector_vop2_state.exec_mask = 0b1011ULL;
  vector_vop2_state.vgprs[0][0] = 3u;
  vector_vop2_state.vgprs[0][1] = 0xfffffff6u;
  vector_vop2_state.vgprs[0][3] = 7u;
  vector_vop2_state.vgprs[1][0] = 2u;
  vector_vop2_state.vgprs[1][1] = 20u;
  vector_vop2_state.vgprs[1][3] = 8u;
  vector_vop2_state.vgprs[2][0] = 1u;
  vector_vop2_state.vgprs[2][1] = 2u;
  vector_vop2_state.vgprs[2][3] = 4u;
  vector_vop2_state.vgprs[3][0] = 0xfffffff8u;
  vector_vop2_state.vgprs[3][1] = 0xfffffffcu;
  vector_vop2_state.vgprs[3][3] = 16u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_vop2_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_vop2_state.halted,
              "expected vector vop2 program to halt") ||
      !Expect(vector_vop2_state.vgprs[10][0] == 0xfffffff0u,
              "expected decoded v_min_i32 lane 0 result") ||
      !Expect(vector_vop2_state.vgprs[11][1] == 0xfffffff6u,
              "expected decoded v_max_i32 lane 1 result") ||
      !Expect(vector_vop2_state.vgprs[12][1] == 15u,
              "expected decoded v_min_u32 lane 1 result") ||
      !Expect(vector_vop2_state.vgprs[13][1] == 20u,
              "expected decoded v_max_u32 lane 1 result") ||
      !Expect(vector_vop2_state.vgprs[14][3] == 64u,
              "expected decoded v_lshlrev_b32 lane 3 result") ||
      !Expect(vector_vop2_state.vgprs[15][0] == 0x3ffffffeu,
              "expected decoded v_lshrrev_b32 lane 0 result") ||
      !Expect(vector_vop2_state.vgprs[16][1] == 0xffffffffu,
              "expected decoded v_ashrrev_i32 lane 1 result") ||
      !Expect(vector_vop2_state.vgprs[17][1] == 4u,
              "expected decoded v_and_b32 lane 1 result") ||
      !Expect(vector_vop2_state.vgprs[18][3] == 15u,
              "expected decoded v_or_b32 lane 3 result") ||
      !Expect(vector_vop2_state.vgprs[19][1] == 27u,
              "expected decoded v_xor_b32 lane 1 result")) {
    return 1;
  }

  const auto v_add_f32_opcode =
      FindDefaultEncodingOpcode("V_ADD_F32", "ENC_VOP2");
  const auto v_sub_f32_opcode =
      FindDefaultEncodingOpcode("V_SUB_F32", "ENC_VOP2");
  const auto v_mul_f32_opcode =
      FindDefaultEncodingOpcode("V_MUL_F32", "ENC_VOP2");
  const auto v_min_f32_opcode =
      FindDefaultEncodingOpcode("V_MIN_F32", "ENC_VOP2");
  const auto v_max_f32_opcode =
      FindDefaultEncodingOpcode("V_MAX_F32", "ENC_VOP2");
  if (!Expect(v_add_f32_opcode.has_value(), "expected V_ADD_F32 opcode lookup") ||
      !Expect(v_sub_f32_opcode.has_value(), "expected V_SUB_F32 opcode lookup") ||
      !Expect(v_mul_f32_opcode.has_value(), "expected V_MUL_F32 opcode lookup") ||
      !Expect(v_min_f32_opcode.has_value(), "expected V_MIN_F32 opcode lookup") ||
      !Expect(v_max_f32_opcode.has_value(), "expected V_MAX_F32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_float_f32_vop2_program = {
      MakeVop2(*v_add_f32_opcode, 30, 60, 20),
      MakeVop2(*v_sub_f32_opcode, 31, 61, 21),
      MakeVop2(*v_mul_f32_opcode, 32, 62, 22),
      MakeVop2(*v_min_f32_opcode, 33, 63, 23),
      MakeVop2(*v_max_f32_opcode, 34, 64, 24),
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_float_f32_vop2_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 6,
              "expected decoded vector float f32 vop2 program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_F32",
              "expected V_ADD_F32 decode") ||
      !Expect(decoded_program[1].opcode == "V_SUB_F32",
              "expected V_SUB_F32 decode") ||
      !Expect(decoded_program[2].opcode == "V_MUL_F32",
              "expected V_MUL_F32 decode") ||
      !Expect(decoded_program[3].opcode == "V_MIN_F32",
              "expected V_MIN_F32 decode") ||
      !Expect(decoded_program[4].opcode == "V_MAX_F32",
              "expected V_MAX_F32 decode")) {
    return 1;
  }

  WaveExecutionState vector_float_f32_vop2_state;
  vector_float_f32_vop2_state.exec_mask = 0b1011ULL;
  vector_float_f32_vop2_state.sgprs[60] = FloatBits(1.5f);
  vector_float_f32_vop2_state.sgprs[61] = FloatBits(5.0f);
  vector_float_f32_vop2_state.sgprs[62] = FloatBits(-2.0f);
  vector_float_f32_vop2_state.sgprs[63] = FloatBits(1.0f);
  vector_float_f32_vop2_state.sgprs[64] = FloatBits(1.5f);
  vector_float_f32_vop2_state.vgprs[20][0] = FloatBits(2.0f);
  vector_float_f32_vop2_state.vgprs[20][1] = FloatBits(-2.25f);
  vector_float_f32_vop2_state.vgprs[20][3] = FloatBits(0.5f);
  vector_float_f32_vop2_state.vgprs[21][0] = FloatBits(1.25f);
  vector_float_f32_vop2_state.vgprs[21][1] = FloatBits(8.0f);
  vector_float_f32_vop2_state.vgprs[21][3] = FloatBits(-0.5f);
  vector_float_f32_vop2_state.vgprs[22][0] = FloatBits(1.5f);
  vector_float_f32_vop2_state.vgprs[22][1] = FloatBits(-0.5f);
  vector_float_f32_vop2_state.vgprs[22][3] = FloatBits(4.0f);
  vector_float_f32_vop2_state.vgprs[23][0] = FloatBits(0.0f);
  vector_float_f32_vop2_state.vgprs[23][1] = FloatBits(2.0f);
  vector_float_f32_vop2_state.vgprs[23][3] = FloatBits(-5.0f);
  vector_float_f32_vop2_state.vgprs[24][0] = FloatBits(2.0f);
  vector_float_f32_vop2_state.vgprs[24][1] = FloatBits(-3.0f);
  vector_float_f32_vop2_state.vgprs[24][3] = FloatBits(1.5f);
  vector_float_f32_vop2_state.vgprs[30][2] = 0xdeadbeefu;
  vector_float_f32_vop2_state.vgprs[31][2] = 0xdeadbeefu;
  vector_float_f32_vop2_state.vgprs[32][2] = 0xdeadbeefu;
  vector_float_f32_vop2_state.vgprs[33][2] = 0xdeadbeefu;
  vector_float_f32_vop2_state.vgprs[34][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_float_f32_vop2_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_float_f32_vop2_state.halted,
              "expected vector float f32 vop2 program to halt") ||
      !Expect(vector_float_f32_vop2_state.vgprs[30][0] == FloatBits(3.5f),
              "expected decoded V_ADD_F32 lane 0 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[30][1] == FloatBits(-0.75f),
              "expected decoded V_ADD_F32 lane 1 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[30][2] == 0xdeadbeefu,
              "expected inactive decoded V_ADD_F32 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[30][3] == FloatBits(2.0f),
              "expected decoded V_ADD_F32 lane 3 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[31][0] == FloatBits(3.75f),
              "expected decoded V_SUB_F32 lane 0 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[31][1] == FloatBits(-3.0f),
              "expected decoded V_SUB_F32 lane 1 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive decoded V_SUB_F32 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[31][3] == FloatBits(5.5f),
              "expected decoded V_SUB_F32 lane 3 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[32][0] == FloatBits(-3.0f),
              "expected decoded V_MUL_F32 lane 0 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[32][1] == FloatBits(1.0f),
              "expected decoded V_MUL_F32 lane 1 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[32][2] == 0xdeadbeefu,
              "expected inactive decoded V_MUL_F32 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[32][3] == FloatBits(-8.0f),
              "expected decoded V_MUL_F32 lane 3 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[33][0] == FloatBits(0.0f),
              "expected decoded V_MIN_F32 lane 0 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[33][1] == FloatBits(1.0f),
              "expected decoded V_MIN_F32 lane 1 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[33][2] == 0xdeadbeefu,
              "expected inactive decoded V_MIN_F32 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[33][3] == FloatBits(-5.0f),
              "expected decoded V_MIN_F32 lane 3 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[34][0] == FloatBits(2.0f),
              "expected decoded V_MAX_F32 lane 0 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[34][1] == FloatBits(1.5f),
              "expected decoded V_MAX_F32 lane 1 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[34][2] == 0xdeadbeefu,
              "expected inactive decoded V_MAX_F32 result") ||
      !Expect(vector_float_f32_vop2_state.vgprs[34][3] == FloatBits(1.5f),
              "expected decoded V_MAX_F32 lane 3 result")) {
    return 1;
  }

  const auto v_not_b32_opcode =
      FindDefaultEncodingOpcode("V_NOT_B32", "ENC_VOP1");
  const auto v_bfrev_b32_opcode =
      FindDefaultEncodingOpcode("V_BFREV_B32", "ENC_VOP1");
  const auto v_ffbh_u32_opcode =
      FindDefaultEncodingOpcode("V_FFBH_U32", "ENC_VOP1");
  const auto v_ffbl_b32_opcode =
      FindDefaultEncodingOpcode("V_FFBL_B32", "ENC_VOP1");
  const auto v_ffbh_i32_opcode =
      FindDefaultEncodingOpcode("V_FFBH_I32", "ENC_VOP1");
  const auto v_nop_opcode = FindDefaultEncodingOpcode("V_NOP", "ENC_VOP1");
  const auto v_cvt_f16_u16_opcode =
      FindDefaultEncodingOpcode("V_CVT_F16_U16", "ENC_VOP1");
  const auto v_cvt_f16_i16_opcode =
      FindDefaultEncodingOpcode("V_CVT_F16_I16", "ENC_VOP1");
  const auto v_cvt_f32_i32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_I32", "ENC_VOP1");
  const auto v_cvt_f32_u32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_U32", "ENC_VOP1");
  const auto v_cvt_u16_f16_opcode =
      FindDefaultEncodingOpcode("V_CVT_U16_F16", "ENC_VOP1");
  const auto v_cvt_i16_f16_opcode =
      FindDefaultEncodingOpcode("V_CVT_I16_F16", "ENC_VOP1");
  const auto v_sat_pk_u8_i16_opcode =
      FindDefaultEncodingOpcode("V_SAT_PK_U8_I16", "ENC_VOP1");
  const auto v_cvt_f32_ubyte0_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_UBYTE0", "ENC_VOP1");
  const auto v_cvt_f32_ubyte1_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_UBYTE1", "ENC_VOP1");
  const auto v_cvt_f32_ubyte2_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_UBYTE2", "ENC_VOP1");
  const auto v_cvt_f32_ubyte3_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_UBYTE3", "ENC_VOP1");
  const auto v_cvt_u32_f32_opcode =
      FindDefaultEncodingOpcode("V_CVT_U32_F32", "ENC_VOP1");
  const auto v_cvt_i32_f32_opcode =
      FindDefaultEncodingOpcode("V_CVT_I32_F32", "ENC_VOP1");
  const auto v_cvt_i32_f64_opcode =
      FindDefaultEncodingOpcode("V_CVT_I32_F64", "ENC_VOP1");
  const auto v_cvt_u32_f64_opcode =
      FindDefaultEncodingOpcode("V_CVT_U32_F64", "ENC_VOP1");
  const auto v_cvt_f16_f32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F16_F32", "ENC_VOP1");
  const auto v_cvt_f32_f16_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_F16", "ENC_VOP1");
  const auto v_cvt_f32_f64_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_F64", "ENC_VOP1");
  const auto v_cvt_f64_f32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F64_F32", "ENC_VOP1");
  const auto v_cvt_f64_i32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F64_I32", "ENC_VOP1");
  const auto v_cvt_f64_u32_opcode =
      FindDefaultEncodingOpcode("V_CVT_F64_U32", "ENC_VOP1");
  const auto v_exp_legacy_f32_opcode =
      FindDefaultEncodingOpcode("V_EXP_LEGACY_F32", "ENC_VOP1");
  const auto v_log_legacy_f32_opcode =
      FindDefaultEncodingOpcode("V_LOG_LEGACY_F32", "ENC_VOP1");
  if (!Expect(v_not_b32_opcode.has_value(), "expected V_NOT_B32 opcode lookup") ||
      !Expect(v_bfrev_b32_opcode.has_value(),
              "expected V_BFREV_B32 opcode lookup") ||
      !Expect(v_ffbh_u32_opcode.has_value(),
              "expected V_FFBH_U32 opcode lookup") ||
      !Expect(v_ffbl_b32_opcode.has_value(),
              "expected V_FFBL_B32 opcode lookup") ||
      !Expect(v_ffbh_i32_opcode.has_value(),
              "expected V_FFBH_I32 opcode lookup") ||
      !Expect(v_nop_opcode.has_value(), "expected V_NOP opcode lookup") ||
      !Expect(v_cvt_f16_u16_opcode.has_value(),
              "expected V_CVT_F16_U16 opcode lookup") ||
      !Expect(v_cvt_f16_i16_opcode.has_value(),
              "expected V_CVT_F16_I16 opcode lookup") ||
      !Expect(v_cvt_f32_i32_opcode.has_value(),
              "expected V_CVT_F32_I32 opcode lookup") ||
      !Expect(v_cvt_f32_u32_opcode.has_value(),
              "expected V_CVT_F32_U32 opcode lookup") ||
      !Expect(v_cvt_u16_f16_opcode.has_value(),
              "expected V_CVT_U16_F16 opcode lookup") ||
      !Expect(v_cvt_i16_f16_opcode.has_value(),
              "expected V_CVT_I16_F16 opcode lookup") ||
      !Expect(v_sat_pk_u8_i16_opcode.has_value(),
              "expected V_SAT_PK_U8_I16 opcode lookup") ||
      !Expect(v_cvt_f32_ubyte0_opcode.has_value(),
              "expected V_CVT_F32_UBYTE0 opcode lookup") ||
      !Expect(v_cvt_f32_ubyte1_opcode.has_value(),
              "expected V_CVT_F32_UBYTE1 opcode lookup") ||
      !Expect(v_cvt_f32_ubyte2_opcode.has_value(),
              "expected V_CVT_F32_UBYTE2 opcode lookup") ||
      !Expect(v_cvt_f32_ubyte3_opcode.has_value(),
              "expected V_CVT_F32_UBYTE3 opcode lookup") ||
      !Expect(v_cvt_u32_f32_opcode.has_value(),
              "expected V_CVT_U32_F32 opcode lookup") ||
      !Expect(v_cvt_i32_f32_opcode.has_value(),
              "expected V_CVT_I32_F32 opcode lookup") ||
      !Expect(v_cvt_i32_f64_opcode.has_value(),
              "expected V_CVT_I32_F64 opcode lookup") ||
      !Expect(v_cvt_u32_f64_opcode.has_value(),
              "expected V_CVT_U32_F64 opcode lookup") ||
      !Expect(v_cvt_f16_f32_opcode.has_value(),
              "expected V_CVT_F16_F32 opcode lookup") ||
      !Expect(v_cvt_f32_f16_opcode.has_value(),
              "expected V_CVT_F32_F16 opcode lookup") ||
      !Expect(v_cvt_f32_f64_opcode.has_value(),
              "expected V_CVT_F32_F64 opcode lookup") ||
      !Expect(v_cvt_f64_f32_opcode.has_value(),
              "expected V_CVT_F64_F32 opcode lookup") ||
      !Expect(v_cvt_f64_i32_opcode.has_value(),
              "expected V_CVT_F64_I32 opcode lookup") ||
      !Expect(v_cvt_f64_u32_opcode.has_value(),
              "expected V_CVT_F64_U32 opcode lookup") ||
      !Expect(v_exp_legacy_f32_opcode.has_value(),
              "expected V_EXP_LEGACY_F32 opcode lookup") ||
      !Expect(v_log_legacy_f32_opcode.has_value(),
              "expected V_LOG_LEGACY_F32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_vop1_program = {
      MakeSop1(0, 4, 255), 0x0f0f0000u,                 // s_mov_b32 s4, 0x0f0f0000
      MakeSop1(0, 5, 143),                              // s_mov_b32 s5, 15
      MakeSop1(0, 6, 255), 0x00f00000u,                 // s_mov_b32 s6, 0x00f00000
      MakeSop1(0, 7, 255), 0xffff0000u,                 // s_mov_b32 s7, 0xffff0000
      MakeVop1(*v_not_b32_opcode, 25, 4),               // v_not_b32 v25, s4
      MakeVop1(*v_bfrev_b32_opcode, 26, 5),             // v_bfrev_b32 v26, s5
      MakeVop1(*v_ffbh_u32_opcode, 27, 6),              // v_ffbh_u32 v27, s6
      MakeVop1(*v_ffbl_b32_opcode, 28, 6),              // v_ffbl_b32 v28, s6
      MakeVop1(*v_ffbh_i32_opcode, 29, 7),              // v_ffbh_i32 v29, s7
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_vop1_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded vector vop1 program size") ||
      !Expect(decoded_program[4].opcode == "V_NOT_B32",
              "expected V_NOT_B32 decode") ||
      !Expect(decoded_program[5].opcode == "V_BFREV_B32",
              "expected V_BFREV_B32 decode") ||
      !Expect(decoded_program[6].opcode == "V_FFBH_U32",
              "expected V_FFBH_U32 decode") ||
      !Expect(decoded_program[7].opcode == "V_FFBL_B32",
              "expected V_FFBL_B32 decode") ||
      !Expect(decoded_program[8].opcode == "V_FFBH_I32",
              "expected V_FFBH_I32 decode") ||
      !Expect(decoded_program[4].operands[1].kind == OperandKind::kSgpr,
              "expected vector unary SGPR source decode") ||
      !Expect(decoded_program[4].operands[1].index == 4,
              "expected vector unary SGPR source index")) {
    return 1;
  }

  WaveExecutionState vector_vop1_state;
  vector_vop1_state.exec_mask = 0b1011ULL;
  vector_vop1_state.vgprs[25][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_vop1_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_vop1_state.halted,
              "expected vector vop1 program to halt") ||
      !Expect(vector_vop1_state.vgprs[25][0] == 0xf0f0ffffu,
              "expected decoded v_not_b32 lane 0 result") ||
      !Expect(vector_vop1_state.vgprs[25][2] == 0xdeadbeefu,
              "expected decoded inactive v_not_b32 lane result") ||
      !Expect(vector_vop1_state.vgprs[26][1] == 0xf0000000u,
              "expected decoded v_bfrev_b32 lane 1 result") ||
      !Expect(vector_vop1_state.vgprs[27][3] == 8u,
              "expected decoded v_ffbh_u32 lane 3 result") ||
      !Expect(vector_vop1_state.vgprs[28][0] == 20u,
              "expected decoded v_ffbl_b32 lane 0 result") ||
      !Expect(vector_vop1_state.vgprs[29][1] == 16u,
              "expected decoded v_ffbh_i32 lane 1 result")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_conversion_vop1_program = {
      MakeSop1(0, 30, 255), 0xfffffff9u,                 // s_mov_b32 s30, -7
      MakeSop1(0, 31, 137),                              // s_mov_b32 s31, 9
      MakeSop1(0, 32, 255), FloatBits(-1.25f),           // s_mov_b32 s32, -1.25f
      MakeSop1(0, 33, 255), 0xfffffff5u,                 // s_mov_b32 s33, -11
      MakeSop1(0, 34, 140),                              // s_mov_b32 s34, 12
      MakeVop1(*v_cvt_f32_i32_opcode, 60, 30),           // v_cvt_f32_i32 v60, s30
      MakeVop1(*v_cvt_f32_u32_opcode, 61, 31),           // v_cvt_f32_u32 v61, s31
      MakeVop1(*v_cvt_i32_f32_opcode, 62, 326),          // v_cvt_i32_f32 v62, v70
      MakeVop1(*v_cvt_u32_f32_opcode, 63, 327),          // v_cvt_u32_f32 v63, v71
      MakeVop1(*v_cvt_f16_f32_opcode, 64, 328),          // v_cvt_f16_f32 v64, v72
      MakeVop1(*v_cvt_f32_f16_opcode, 65, 329),          // v_cvt_f32_f16 v65, v73
      MakeVop1(*v_cvt_f64_f32_opcode, 66, 32),           // v_cvt_f64_f32 v[66:67], s32
      MakeVop1(*v_cvt_f32_f64_opcode, 68, 330),          // v_cvt_f32_f64 v68, v[74:75]
      MakeVop1(*v_cvt_i32_f64_opcode, 84, 330),          // v_cvt_i32_f64 v84, v[74:75]
      MakeVop1(*v_cvt_u32_f64_opcode, 85, 338),          // v_cvt_u32_f64 v85, v[82:83]
      MakeVop1(*v_cvt_f64_i32_opcode, 86, 33),           // v_cvt_f64_i32 v[86:87], s33
      MakeVop1(*v_cvt_f64_u32_opcode, 88, 34),           // v_cvt_f64_u32 v[88:89], s34
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_conversion_vop1_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 18,
              "expected decoded vector conversion VOP1 program size") ||
      !Expect(decoded_program[5].opcode == "V_CVT_F32_I32",
              "expected V_CVT_F32_I32 decode") ||
      !Expect(decoded_program[6].opcode == "V_CVT_F32_U32",
              "expected V_CVT_F32_U32 decode") ||
      !Expect(decoded_program[7].opcode == "V_CVT_I32_F32",
              "expected V_CVT_I32_F32 decode") ||
      !Expect(decoded_program[8].opcode == "V_CVT_U32_F32",
              "expected V_CVT_U32_F32 decode") ||
      !Expect(decoded_program[9].opcode == "V_CVT_F16_F32",
              "expected V_CVT_F16_F32 decode") ||
      !Expect(decoded_program[10].opcode == "V_CVT_F32_F16",
              "expected V_CVT_F32_F16 decode") ||
      !Expect(decoded_program[11].opcode == "V_CVT_F64_F32",
              "expected V_CVT_F64_F32 decode") ||
      !Expect(decoded_program[12].opcode == "V_CVT_F32_F64",
              "expected V_CVT_F32_F64 decode") ||
      !Expect(decoded_program[13].opcode == "V_CVT_I32_F64",
              "expected V_CVT_I32_F64 decode") ||
      !Expect(decoded_program[14].opcode == "V_CVT_U32_F64",
              "expected V_CVT_U32_F64 decode") ||
      !Expect(decoded_program[15].opcode == "V_CVT_F64_I32",
              "expected V_CVT_F64_I32 decode") ||
      !Expect(decoded_program[16].opcode == "V_CVT_F64_U32",
              "expected V_CVT_F64_U32 decode")) {
    return 1;
  }

  WaveExecutionState vector_conversion_vop1_state;
  vector_conversion_vop1_state.exec_mask = 0b1011ULL;
  vector_conversion_vop1_state.vgprs[70][0] = FloatBits(3.75f);
  vector_conversion_vop1_state.vgprs[70][1] = FloatBits(-2.75f);
  vector_conversion_vop1_state.vgprs[70][3] = FloatBits(-0.5f);
  vector_conversion_vop1_state.vgprs[71][0] = FloatBits(7.75f);
  vector_conversion_vop1_state.vgprs[71][1] = FloatBits(1.0f);
  vector_conversion_vop1_state.vgprs[71][3] = FloatBits(0.5f);
  vector_conversion_vop1_state.vgprs[72][0] = FloatBits(1.5f);
  vector_conversion_vop1_state.vgprs[72][1] = FloatBits(-2.0f);
  vector_conversion_vop1_state.vgprs[72][3] = FloatBits(0.5f);
  vector_conversion_vop1_state.vgprs[73][0] = 0x00003e00u;
  vector_conversion_vop1_state.vgprs[73][1] = 0x0000c000u;
  vector_conversion_vop1_state.vgprs[73][3] = 0x00003800u;
  SplitU64(DoubleBits(2.5), &vector_conversion_vop1_state.vgprs[74][0],
           &vector_conversion_vop1_state.vgprs[75][0]);
  SplitU64(DoubleBits(-0.25), &vector_conversion_vop1_state.vgprs[74][1],
           &vector_conversion_vop1_state.vgprs[75][1]);
  SplitU64(DoubleBits(8.0), &vector_conversion_vop1_state.vgprs[74][3],
           &vector_conversion_vop1_state.vgprs[75][3]);
  SplitU64(DoubleBits(9.5), &vector_conversion_vop1_state.vgprs[82][0],
           &vector_conversion_vop1_state.vgprs[83][0]);
  SplitU64(DoubleBits(1.0), &vector_conversion_vop1_state.vgprs[82][1],
           &vector_conversion_vop1_state.vgprs[83][1]);
  SplitU64(DoubleBits(0.5), &vector_conversion_vop1_state.vgprs[82][3],
           &vector_conversion_vop1_state.vgprs[83][3]);
  vector_conversion_vop1_state.vgprs[60][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[61][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[62][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[63][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[64][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[65][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[66][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[67][2] = 0xcafebabeu;
  vector_conversion_vop1_state.vgprs[68][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[84][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[85][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[86][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[87][2] = 0xcafebabeu;
  vector_conversion_vop1_state.vgprs[88][2] = 0xdeadbeefu;
  vector_conversion_vop1_state.vgprs[89][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_conversion_vop1_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_conversion_vop1_state.halted,
              "expected vector conversion VOP1 program to halt") ||
      !Expect(vector_conversion_vop1_state.vgprs[60][0] == FloatBits(-7.0f),
              "expected VOP1 v_cvt_f32_i32 lane 0 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[60][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f32_i32 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[61][1] == FloatBits(9.0f),
              "expected VOP1 v_cvt_f32_u32 lane 1 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[62][0] == 3u,
              "expected VOP1 v_cvt_i32_f32 lane 0 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[62][1] ==
                  static_cast<std::uint32_t>(-2),
              "expected VOP1 v_cvt_i32_f32 lane 1 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[63][3] == 0u,
              "expected VOP1 v_cvt_u32_f32 lane 3 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[64][0] == 0x00003e00u,
              "expected VOP1 v_cvt_f16_f32 lane 0 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[65][1] == FloatBits(-2.0f),
              "expected VOP1 v_cvt_f32_f16 lane 1 result") ||
      !Expect(ComposeU64(vector_conversion_vop1_state.vgprs[66][0],
                         vector_conversion_vop1_state.vgprs[67][0]) ==
                  DoubleBits(-1.25),
              "expected VOP1 v_cvt_f64_f32 lane 0 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[66][2] == 0xdeadbeefu &&
                  vector_conversion_vop1_state.vgprs[67][2] == 0xcafebabeu,
              "expected inactive VOP1 v_cvt_f64_f32 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[68][3] == FloatBits(8.0f),
              "expected VOP1 v_cvt_f32_f64 lane 3 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[84][0] == 2u,
              "expected VOP1 v_cvt_i32_f64 lane 0 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[85][1] == 1u,
              "expected VOP1 v_cvt_u32_f64 lane 1 result") ||
      !Expect(ComposeU64(vector_conversion_vop1_state.vgprs[86][0],
                         vector_conversion_vop1_state.vgprs[87][0]) ==
                  DoubleBits(-11.0),
              "expected VOP1 v_cvt_f64_i32 lane 0 result") ||
      !Expect(ComposeU64(vector_conversion_vop1_state.vgprs[88][1],
                         vector_conversion_vop1_state.vgprs[89][1]) ==
                  DoubleBits(12.0),
              "expected VOP1 v_cvt_f64_u32 lane 1 result") ||
      !Expect(vector_conversion_vop1_state.vgprs[88][2] == 0xdeadbeefu &&
                  vector_conversion_vop1_state.vgprs[89][2] == 0xcafebabeu,
              "expected inactive VOP1 v_cvt_f64_u32 result")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_byte_conversion_vop1_program = {
      MakeVop1(*v_nop_opcode, 0, 0),                 // v_nop
      MakeVop1(*v_cvt_f16_u16_opcode, 96, 346),      // v_cvt_f16_u16 v96, v90
      MakeVop1(*v_cvt_f32_ubyte0_opcode, 97, 347),   // v_cvt_f32_ubyte0 v97, v91
      MakeVop1(*v_cvt_f32_ubyte1_opcode, 98, 347),   // v_cvt_f32_ubyte1 v98, v91
      MakeVop1(*v_cvt_f32_ubyte2_opcode, 99, 347),   // v_cvt_f32_ubyte2 v99, v91
      MakeVop1(*v_cvt_f32_ubyte3_opcode, 100, 347),  // v_cvt_f32_ubyte3 v100, v91
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_byte_conversion_vop1_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 7,
              "expected decoded vector byte conversion VOP1 program size") ||
      !Expect(decoded_program[0].opcode == "V_NOP", "expected V_NOP decode") ||
      !Expect(decoded_program[0].operand_count == 0,
              "expected V_NOP nullary decode") ||
      !Expect(decoded_program[1].opcode == "V_CVT_F16_U16",
              "expected V_CVT_F16_U16 decode") ||
      !Expect(decoded_program[2].opcode == "V_CVT_F32_UBYTE0",
              "expected V_CVT_F32_UBYTE0 decode") ||
      !Expect(decoded_program[3].opcode == "V_CVT_F32_UBYTE1",
              "expected V_CVT_F32_UBYTE1 decode") ||
      !Expect(decoded_program[4].opcode == "V_CVT_F32_UBYTE2",
              "expected V_CVT_F32_UBYTE2 decode") ||
      !Expect(decoded_program[5].opcode == "V_CVT_F32_UBYTE3",
              "expected V_CVT_F32_UBYTE3 decode")) {
    return 1;
  }

  WaveExecutionState vector_byte_conversion_vop1_state;
  vector_byte_conversion_vop1_state.exec_mask = 0b1011ULL;
  vector_byte_conversion_vop1_state.vgprs[90][0] = 1u;
  vector_byte_conversion_vop1_state.vgprs[90][1] = 2u;
  vector_byte_conversion_vop1_state.vgprs[90][3] = 3u;
  vector_byte_conversion_vop1_state.vgprs[91][0] = 0x44332211u;
  vector_byte_conversion_vop1_state.vgprs[91][1] = 0xaabbccddu;
  vector_byte_conversion_vop1_state.vgprs[91][3] = 0x01020304u;
  vector_byte_conversion_vop1_state.vgprs[96][2] = 0xdeadbeefu;
  vector_byte_conversion_vop1_state.vgprs[97][2] = 0xdeadbeefu;
  vector_byte_conversion_vop1_state.vgprs[98][2] = 0xdeadbeefu;
  vector_byte_conversion_vop1_state.vgprs[99][2] = 0xdeadbeefu;
  vector_byte_conversion_vop1_state.vgprs[100][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_byte_conversion_vop1_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_byte_conversion_vop1_state.halted,
              "expected vector byte conversion VOP1 program to halt") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[96][0] == 0x00003c00u,
              "expected VOP1 v_cvt_f16_u16 lane 0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[96][1] == 0x00004000u,
              "expected VOP1 v_cvt_f16_u16 lane 1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[96][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f16_u16 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[96][3] == 0x00004200u,
              "expected VOP1 v_cvt_f16_u16 lane 3 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[97][0] ==
                  FloatBits(17.0f),
              "expected VOP1 v_cvt_f32_ubyte0 lane 0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[97][1] ==
                  FloatBits(221.0f),
              "expected VOP1 v_cvt_f32_ubyte0 lane 1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[97][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f32_ubyte0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[97][3] ==
                  FloatBits(4.0f),
              "expected VOP1 v_cvt_f32_ubyte0 lane 3 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[98][0] ==
                  FloatBits(34.0f),
              "expected VOP1 v_cvt_f32_ubyte1 lane 0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[98][1] ==
                  FloatBits(204.0f),
              "expected VOP1 v_cvt_f32_ubyte1 lane 1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[98][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f32_ubyte1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[98][3] ==
                  FloatBits(3.0f),
              "expected VOP1 v_cvt_f32_ubyte1 lane 3 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[99][0] ==
                  FloatBits(51.0f),
              "expected VOP1 v_cvt_f32_ubyte2 lane 0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[99][1] ==
                  FloatBits(187.0f),
              "expected VOP1 v_cvt_f32_ubyte2 lane 1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[99][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f32_ubyte2 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[99][3] ==
                  FloatBits(2.0f),
              "expected VOP1 v_cvt_f32_ubyte2 lane 3 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[100][0] ==
                  FloatBits(68.0f),
              "expected VOP1 v_cvt_f32_ubyte3 lane 0 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[100][1] ==
                  FloatBits(170.0f),
              "expected VOP1 v_cvt_f32_ubyte3 lane 1 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[100][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f32_ubyte3 result") ||
      !Expect(vector_byte_conversion_vop1_state.vgprs[100][3] ==
                  FloatBits(1.0f),
              "expected VOP1 v_cvt_f32_ubyte3 lane 3 result")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_half_int_conversion_vop1_program = {
      MakeVop1(*v_cvt_f16_i16_opcode, 101, 348),  // v_cvt_f16_i16 v101, v92
      MakeVop1(*v_cvt_u16_f16_opcode, 102, 349),  // v_cvt_u16_f16 v102, v93
      MakeVop1(*v_cvt_i16_f16_opcode, 103, 350),  // v_cvt_i16_f16 v103, v94
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_half_int_conversion_vop1_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded vector half/int conversion VOP1 program size") ||
      !Expect(decoded_program[0].opcode == "V_CVT_F16_I16",
              "expected V_CVT_F16_I16 decode") ||
      !Expect(decoded_program[1].opcode == "V_CVT_U16_F16",
              "expected V_CVT_U16_F16 decode") ||
      !Expect(decoded_program[2].opcode == "V_CVT_I16_F16",
              "expected V_CVT_I16_F16 decode")) {
    return 1;
  }

  auto vector_half_int_conversion_vop1_state =
      std::make_unique<WaveExecutionState>();
  vector_half_int_conversion_vop1_state->exec_mask = 0b1011ULL;
  vector_half_int_conversion_vop1_state->vgprs[92][0] = 0xffffu;
  vector_half_int_conversion_vop1_state->vgprs[92][1] = 0x0002u;
  vector_half_int_conversion_vop1_state->vgprs[92][3] = 0xfffdu;
  vector_half_int_conversion_vop1_state->vgprs[93][0] = 0x00003e00u;
  vector_half_int_conversion_vop1_state->vgprs[93][1] = 0x00004000u;
  vector_half_int_conversion_vop1_state->vgprs[93][3] = 0x00004300u;
  vector_half_int_conversion_vop1_state->vgprs[94][0] = 0x0000be00u;
  vector_half_int_conversion_vop1_state->vgprs[94][1] = 0x00004000u;
  vector_half_int_conversion_vop1_state->vgprs[94][3] = 0x0000c200u;
  vector_half_int_conversion_vop1_state->vgprs[101][2] = 0xdeadbeefu;
  vector_half_int_conversion_vop1_state->vgprs[102][2] = 0xdeadbeefu;
  vector_half_int_conversion_vop1_state->vgprs[103][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         vector_half_int_conversion_vop1_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_half_int_conversion_vop1_state->halted,
              "expected vector half/int conversion VOP1 program to halt") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[101][0] ==
                  0x0000bc00u,
              "expected VOP1 v_cvt_f16_i16 lane 0 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[101][1] ==
                  0x00004000u,
              "expected VOP1 v_cvt_f16_i16 lane 1 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[101][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_cvt_f16_i16 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[101][3] ==
                  0x0000c200u,
              "expected VOP1 v_cvt_f16_i16 lane 3 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[102][0] == 1u,
              "expected VOP1 v_cvt_u16_f16 lane 0 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[102][1] == 2u,
              "expected VOP1 v_cvt_u16_f16 lane 1 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[102][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_cvt_u16_f16 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[102][3] == 3u,
              "expected VOP1 v_cvt_u16_f16 lane 3 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[103][0] ==
                  0x0000ffffu,
              "expected VOP1 v_cvt_i16_f16 lane 0 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[103][1] ==
                  0x00000002u,
              "expected VOP1 v_cvt_i16_f16 lane 1 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[103][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_cvt_i16_f16 result") ||
      !Expect(vector_half_int_conversion_vop1_state->vgprs[103][3] ==
                  0x0000fffdu,
              "expected VOP1 v_cvt_i16_f16 lane 3 result")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_sat_pk_vop1_program = {
      MakeVop1(*v_sat_pk_u8_i16_opcode, 106, 353),  // v_sat_pk_u8_i16 v106, v97
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_sat_pk_vop1_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_sat_pk_u8_i16 VOP1 program size") ||
      !Expect(decoded_program[0].opcode == "V_SAT_PK_U8_I16",
              "expected V_SAT_PK_U8_I16 decode")) {
    return 1;
  }

  auto vector_sat_pk_vop1_state = std::make_unique<WaveExecutionState>();
  vector_sat_pk_vop1_state->exec_mask = 0b1011ULL;
  vector_sat_pk_vop1_state->vgprs[97][0] = 0x007f0100u;
  vector_sat_pk_vop1_state->vgprs[97][1] = 0xffff0001u;
  vector_sat_pk_vop1_state->vgprs[97][3] = 0x12340080u;
  vector_sat_pk_vop1_state->vgprs[106][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         vector_sat_pk_vop1_state.get(),
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_sat_pk_vop1_state->halted,
              "expected v_sat_pk_u8_i16 VOP1 program to halt") ||
      !Expect(vector_sat_pk_vop1_state->vgprs[106][0] == 0x00007fffu,
              "expected VOP1 v_sat_pk_u8_i16 lane 0 result") ||
      !Expect(vector_sat_pk_vop1_state->vgprs[106][1] == 0x00000001u,
              "expected VOP1 v_sat_pk_u8_i16 lane 1 result") ||
      !Expect(vector_sat_pk_vop1_state->vgprs[106][2] == 0xdeadbeefu,
              "expected inactive VOP1 v_sat_pk_u8_i16 result") ||
      !Expect(vector_sat_pk_vop1_state->vgprs[106][3] == 0x0000ff80u,
              "expected VOP1 v_sat_pk_u8_i16 lane 3 result")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_legacy_float_math_vop1_program = {
      MakeVop1(*v_exp_legacy_f32_opcode, 104, 351),  // v_exp_legacy_f32 v104, v95
      MakeVop1(*v_log_legacy_f32_opcode, 105, 352),  // v_log_legacy_f32 v105, v96
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_legacy_float_math_vop1_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 3,
              "expected decoded legacy float math VOP1 program size") ||
      !Expect(decoded_program[0].opcode == "V_EXP_LEGACY_F32",
              "expected V_EXP_LEGACY_F32 decode") ||
      !Expect(decoded_program[1].opcode == "V_LOG_LEGACY_F32",
              "expected V_LOG_LEGACY_F32 decode")) {
    return 1;
  }

  WaveExecutionState vector_legacy_float_math_vop1_state;
  vector_legacy_float_math_vop1_state.exec_mask = 0b1011ULL;
  vector_legacy_float_math_vop1_state.vgprs[95][0] = FloatBits(1.0f);
  vector_legacy_float_math_vop1_state.vgprs[95][1] = FloatBits(2.0f);
  vector_legacy_float_math_vop1_state.vgprs[95][3] = FloatBits(-1.0f);
  vector_legacy_float_math_vop1_state.vgprs[96][0] = FloatBits(1.0f);
  vector_legacy_float_math_vop1_state.vgprs[96][1] = FloatBits(8.0f);
  vector_legacy_float_math_vop1_state.vgprs[96][3] = FloatBits(0.5f);
  vector_legacy_float_math_vop1_state.vgprs[104][2] = 0xdeadbeefu;
  vector_legacy_float_math_vop1_state.vgprs[105][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_legacy_float_math_vop1_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_legacy_float_math_vop1_state.halted,
              "expected legacy float math VOP1 program to halt") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[104][0] ==
                  FloatBits(2.0f),
              "expected VOP1 v_exp_legacy_f32 lane 0 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[104][1] ==
                  FloatBits(4.0f),
              "expected VOP1 v_exp_legacy_f32 lane 1 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[104][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_exp_legacy_f32 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[104][3] ==
                  FloatBits(0.5f),
              "expected VOP1 v_exp_legacy_f32 lane 3 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[105][0] ==
                  FloatBits(0.0f),
              "expected VOP1 v_log_legacy_f32 lane 0 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[105][1] ==
                  FloatBits(3.0f),
              "expected VOP1 v_log_legacy_f32 lane 1 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[105][2] ==
                  0xdeadbeefu,
              "expected inactive VOP1 v_log_legacy_f32 result") ||
      !Expect(vector_legacy_float_math_vop1_state.vgprs[105][3] ==
                  FloatBits(-1.0f),
              "expected VOP1 v_log_legacy_f32 lane 3 result")) {
    return 1;
  }

  const auto v_mul_lo_u32_opcode =
      FindDefaultEncodingOpcode("V_MUL_LO_U32", "ENC_VOP3");
  const auto v_mul_hi_u32_opcode =
      FindDefaultEncodingOpcode("V_MUL_HI_U32", "ENC_VOP3");
  const auto v_mul_hi_i32_opcode =
      FindDefaultEncodingOpcode("V_MUL_HI_I32", "ENC_VOP3");
  const auto v_add3_u32_opcode =
      FindDefaultEncodingOpcode("V_ADD3_U32", "ENC_VOP3");
  const auto v_lerp_u8_opcode =
      FindDefaultEncodingOpcode("V_LERP_U8", "ENC_VOP3");
  const auto v_perm_b32_opcode =
      FindDefaultEncodingOpcode("V_PERM_B32", "ENC_VOP3");
  const auto v_bfe_u32_opcode =
      FindDefaultEncodingOpcode("V_BFE_U32", "ENC_VOP3");
  const auto v_bfe_i32_opcode =
      FindDefaultEncodingOpcode("V_BFE_I32", "ENC_VOP3");
  const auto v_bfi_b32_opcode =
      FindDefaultEncodingOpcode("V_BFI_B32", "ENC_VOP3");
  const auto v_alignbit_b32_opcode =
      FindDefaultEncodingOpcode("V_ALIGNBIT_B32", "ENC_VOP3");
  const auto v_alignbyte_b32_opcode =
      FindDefaultEncodingOpcode("V_ALIGNBYTE_B32", "ENC_VOP3");
  const auto v_min3_i32_opcode =
      FindDefaultEncodingOpcode("V_MIN3_I32", "ENC_VOP3");
  const auto v_min3_u32_opcode =
      FindDefaultEncodingOpcode("V_MIN3_U32", "ENC_VOP3");
  const auto v_max3_i32_opcode =
      FindDefaultEncodingOpcode("V_MAX3_I32", "ENC_VOP3");
  const auto v_max3_u32_opcode =
      FindDefaultEncodingOpcode("V_MAX3_U32", "ENC_VOP3");
  const auto v_med3_i32_opcode =
      FindDefaultEncodingOpcode("V_MED3_I32", "ENC_VOP3");
  const auto v_med3_u32_opcode =
      FindDefaultEncodingOpcode("V_MED3_U32", "ENC_VOP3");
  const auto v_sad_u8_opcode =
      FindDefaultEncodingOpcode("V_SAD_U8", "ENC_VOP3");
  const auto v_sad_hi_u8_opcode =
      FindDefaultEncodingOpcode("V_SAD_HI_U8", "ENC_VOP3");
  const auto v_sad_u16_opcode =
      FindDefaultEncodingOpcode("V_SAD_U16", "ENC_VOP3");
  const auto v_sad_u32_opcode =
      FindDefaultEncodingOpcode("V_SAD_U32", "ENC_VOP3");
  const auto v_mad_i32_i24_opcode =
      FindDefaultEncodingOpcode("V_MAD_I32_I24", "ENC_VOP3");
  const auto v_mad_u32_u24_opcode =
      FindDefaultEncodingOpcode("V_MAD_U32_U24", "ENC_VOP3");
  const auto v_lshl_add_u32_opcode =
      FindDefaultEncodingOpcode("V_LSHL_ADD_U32", "ENC_VOP3");
  const auto v_bcnt_u32_b32_opcode =
      FindDefaultEncodingOpcode("V_BCNT_U32_B32", "ENC_VOP3");
  const auto v_bfm_b32_opcode =
      FindDefaultEncodingOpcode("V_BFM_B32", "ENC_VOP3");
  if (!Expect(v_mul_lo_u32_opcode.has_value(),
              "expected V_MUL_LO_U32 opcode lookup") ||
      !Expect(v_mul_hi_u32_opcode.has_value(),
              "expected V_MUL_HI_U32 opcode lookup") ||
      !Expect(v_mul_hi_i32_opcode.has_value(),
              "expected V_MUL_HI_I32 opcode lookup") ||
      !Expect(v_add3_u32_opcode.has_value(),
              "expected V_ADD3_U32 opcode lookup") ||
      !Expect(v_lerp_u8_opcode.has_value(),
              "expected V_LERP_U8 opcode lookup") ||
      !Expect(v_perm_b32_opcode.has_value(),
              "expected V_PERM_B32 opcode lookup") ||
      !Expect(v_bfe_u32_opcode.has_value(),
              "expected V_BFE_U32 opcode lookup") ||
      !Expect(v_bfe_i32_opcode.has_value(),
              "expected V_BFE_I32 opcode lookup") ||
      !Expect(v_bfi_b32_opcode.has_value(),
              "expected V_BFI_B32 opcode lookup") ||
      !Expect(v_alignbit_b32_opcode.has_value(),
              "expected V_ALIGNBIT_B32 opcode lookup") ||
      !Expect(v_alignbyte_b32_opcode.has_value(),
              "expected V_ALIGNBYTE_B32 opcode lookup") ||
      !Expect(v_min3_i32_opcode.has_value(),
              "expected V_MIN3_I32 opcode lookup") ||
      !Expect(v_min3_u32_opcode.has_value(),
              "expected V_MIN3_U32 opcode lookup") ||
      !Expect(v_max3_i32_opcode.has_value(),
              "expected V_MAX3_I32 opcode lookup") ||
      !Expect(v_max3_u32_opcode.has_value(),
              "expected V_MAX3_U32 opcode lookup") ||
      !Expect(v_med3_i32_opcode.has_value(),
              "expected V_MED3_I32 opcode lookup") ||
      !Expect(v_med3_u32_opcode.has_value(),
              "expected V_MED3_U32 opcode lookup") ||
      !Expect(v_sad_u8_opcode.has_value(),
              "expected V_SAD_U8 opcode lookup") ||
      !Expect(v_sad_hi_u8_opcode.has_value(),
              "expected V_SAD_HI_U8 opcode lookup") ||
      !Expect(v_sad_u16_opcode.has_value(),
              "expected V_SAD_U16 opcode lookup") ||
      !Expect(v_sad_u32_opcode.has_value(),
              "expected V_SAD_U32 opcode lookup") ||
      !Expect(v_mad_i32_i24_opcode.has_value(),
              "expected V_MAD_I32_I24 opcode lookup") ||
      !Expect(v_mad_u32_u24_opcode.has_value(),
              "expected V_MAD_U32_U24 opcode lookup") ||
      !Expect(v_lshl_add_u32_opcode.has_value(),
              "expected V_LSHL_ADD_U32 opcode lookup") ||
      !Expect(v_bcnt_u32_b32_opcode.has_value(),
              "expected V_BCNT_U32_B32 opcode lookup") ||
      !Expect(v_bfm_b32_opcode.has_value(),
              "expected V_BFM_B32 opcode lookup")) {
    return 1;
  }

  const auto vop3_mul_lo_word = MakeVop3(*v_mul_lo_u32_opcode, 30, 8, 264);
  const auto vop3_mul_hi_u32_word = MakeVop3(*v_mul_hi_u32_opcode, 31, 9, 265);
  const auto vop3_mul_hi_i32_word = MakeVop3(*v_mul_hi_i32_opcode, 32, 10, 266);
  const auto vop3_add3_word = MakeVop3(*v_add3_u32_opcode, 33, 11, 267, 133);
  const auto vop3_lerp_u8_word = MakeVop3(*v_lerp_u8_opcode, 49, 282, 28, 27);
  const auto vop3_perm_b32_word = MakeVop3(*v_perm_b32_opcode, 50, 29, 283, 284);
  const auto vop3_bfe_u32_word = MakeVop3(*v_bfe_u32_opcode, 34, 268, 272, 20);
  const auto vop3_bfe_i32_word = MakeVop3(*v_bfe_i32_opcode, 35, 269, 273, 21);
  const auto vop3_bfi_b32_word = MakeVop3(*v_bfi_b32_opcode, 36, 22, 270, 271);
  const auto vop3_alignbit_b32_word =
      MakeVop3(*v_alignbit_b32_opcode, 37, 274, 23, 132);
  const auto vop3_alignbyte_b32_word =
      MakeVop3(*v_alignbyte_b32_opcode, 38, 274, 23, 129);
  const auto vop3_min3_i32_word = MakeVop3(*v_min3_i32_opcode, 39, 274, 23, 275);
  const auto vop3_max3_i32_word = MakeVop3(*v_max3_i32_opcode, 40, 274, 23, 275);
  const auto vop3_med3_i32_word = MakeVop3(*v_med3_i32_opcode, 41, 274, 23, 275);
  const auto vop3_min3_u32_word = MakeVop3(*v_min3_u32_opcode, 42, 274, 23, 275);
  const auto vop3_max3_u32_word = MakeVop3(*v_max3_u32_opcode, 43, 274, 23, 275);
  const auto vop3_med3_u32_word = MakeVop3(*v_med3_u32_opcode, 44, 274, 23, 275);
  const auto vop3_sad_u8_word = MakeVop3(*v_sad_u8_opcode, 45, 276, 24, 277);
  const auto vop3_sad_hi_u8_word =
      MakeVop3(*v_sad_hi_u8_opcode, 46, 276, 24, 277);
  const auto vop3_sad_u16_word = MakeVop3(*v_sad_u16_opcode, 47, 278, 25, 279);
  const auto vop3_sad_u32_word = MakeVop3(*v_sad_u32_opcode, 48, 280, 26, 281);
  const auto vop3_mad_i32_i24_word =
      MakeVop3(*v_mad_i32_i24_opcode, 51, 309, 30, 310);
  const auto vop3_mad_u32_u24_word =
      MakeVop3(*v_mad_u32_u24_opcode, 52, 311, 31, 312);
  const auto vop3_lshl_add_u32_word =
      MakeVop3(*v_lshl_add_u32_opcode, 57, 317, 32, 318);
  const auto vop3_bcnt_u32_b32_word =
      MakeVop3(*v_bcnt_u32_b32_opcode, 58, 319, 33);
  const auto vop3_bfm_b32_word =
      MakeVop3(*v_bfm_b32_opcode, 59, 320, 34);
  const std::vector<std::uint32_t> vector_vop3_program = {
      vop3_mul_lo_word[0], vop3_mul_lo_word[1],
      vop3_mul_hi_u32_word[0], vop3_mul_hi_u32_word[1],
      vop3_mul_hi_i32_word[0], vop3_mul_hi_i32_word[1],
      vop3_add3_word[0], vop3_add3_word[1],
      vop3_lerp_u8_word[0], vop3_lerp_u8_word[1],
      vop3_perm_b32_word[0], vop3_perm_b32_word[1],
      vop3_bfe_u32_word[0], vop3_bfe_u32_word[1],
      vop3_bfe_i32_word[0], vop3_bfe_i32_word[1],
      vop3_bfi_b32_word[0], vop3_bfi_b32_word[1],
      vop3_alignbit_b32_word[0], vop3_alignbit_b32_word[1],
      vop3_alignbyte_b32_word[0], vop3_alignbyte_b32_word[1],
      vop3_min3_i32_word[0], vop3_min3_i32_word[1],
      vop3_max3_i32_word[0], vop3_max3_i32_word[1],
      vop3_med3_i32_word[0], vop3_med3_i32_word[1],
      vop3_min3_u32_word[0], vop3_min3_u32_word[1],
      vop3_max3_u32_word[0], vop3_max3_u32_word[1],
      vop3_med3_u32_word[0], vop3_med3_u32_word[1],
      vop3_sad_u8_word[0], vop3_sad_u8_word[1],
      vop3_sad_hi_u8_word[0], vop3_sad_hi_u8_word[1],
      vop3_sad_u16_word[0], vop3_sad_u16_word[1],
      vop3_sad_u32_word[0], vop3_sad_u32_word[1],
      vop3_mad_i32_i24_word[0], vop3_mad_i32_i24_word[1],
      vop3_mad_u32_u24_word[0], vop3_mad_u32_u24_word[1],
      vop3_lshl_add_u32_word[0], vop3_lshl_add_u32_word[1],
      vop3_bcnt_u32_b32_word[0], vop3_bcnt_u32_b32_word[1],
      vop3_bfm_b32_word[0], vop3_bfm_b32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_vop3_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 27,
              "expected decoded vector vop3 program size") ||
      !Expect(decoded_program[0].opcode == "V_MUL_LO_U32",
              "expected V_MUL_LO_U32 decode") ||
      !Expect(decoded_program[1].opcode == "V_MUL_HI_U32",
              "expected V_MUL_HI_U32 decode") ||
      !Expect(decoded_program[2].opcode == "V_MUL_HI_I32",
              "expected V_MUL_HI_I32 decode") ||
      !Expect(decoded_program[3].opcode == "V_ADD3_U32",
              "expected V_ADD3_U32 decode") ||
      !Expect(decoded_program[4].opcode == "V_LERP_U8",
              "expected V_LERP_U8 decode") ||
      !Expect(decoded_program[5].opcode == "V_PERM_B32",
              "expected V_PERM_B32 decode") ||
      !Expect(decoded_program[6].opcode == "V_BFE_U32",
              "expected V_BFE_U32 decode") ||
      !Expect(decoded_program[7].opcode == "V_BFE_I32",
              "expected V_BFE_I32 decode") ||
      !Expect(decoded_program[8].opcode == "V_BFI_B32",
              "expected V_BFI_B32 decode") ||
      !Expect(decoded_program[9].opcode == "V_ALIGNBIT_B32",
              "expected V_ALIGNBIT_B32 decode") ||
      !Expect(decoded_program[10].opcode == "V_ALIGNBYTE_B32",
              "expected V_ALIGNBYTE_B32 decode") ||
      !Expect(decoded_program[11].opcode == "V_MIN3_I32",
              "expected V_MIN3_I32 decode") ||
      !Expect(decoded_program[12].opcode == "V_MAX3_I32",
              "expected V_MAX3_I32 decode") ||
      !Expect(decoded_program[13].opcode == "V_MED3_I32",
              "expected V_MED3_I32 decode") ||
      !Expect(decoded_program[14].opcode == "V_MIN3_U32",
              "expected V_MIN3_U32 decode") ||
      !Expect(decoded_program[15].opcode == "V_MAX3_U32",
              "expected V_MAX3_U32 decode") ||
      !Expect(decoded_program[16].opcode == "V_MED3_U32",
              "expected V_MED3_U32 decode") ||
      !Expect(decoded_program[17].opcode == "V_SAD_U8",
              "expected V_SAD_U8 decode") ||
      !Expect(decoded_program[18].opcode == "V_SAD_HI_U8",
              "expected V_SAD_HI_U8 decode") ||
      !Expect(decoded_program[19].opcode == "V_SAD_U16",
              "expected V_SAD_U16 decode") ||
      !Expect(decoded_program[20].opcode == "V_SAD_U32",
              "expected V_SAD_U32 decode") ||
      !Expect(decoded_program[21].opcode == "V_MAD_I32_I24",
              "expected V_MAD_I32_I24 decode") ||
      !Expect(decoded_program[22].opcode == "V_MAD_U32_U24",
              "expected V_MAD_U32_U24 decode") ||
      !Expect(decoded_program[23].opcode == "V_LSHL_ADD_U32",
              "expected V_LSHL_ADD_U32 decode") ||
      !Expect(decoded_program[24].opcode == "V_BCNT_U32_B32",
              "expected V_BCNT_U32_B32 decode") ||
      !Expect(decoded_program[25].opcode == "V_BFM_B32",
              "expected V_BFM_B32 decode") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr,
              "expected vop3 src0 SGPR decode") ||
      !Expect(decoded_program[0].operands[1].index == 8,
              "expected vop3 src0 SGPR index") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
              "expected vop3 src1 VGPR decode") ||
      !Expect(decoded_program[0].operands[2].index == 8,
              "expected vop3 src1 VGPR index") ||
      !Expect(decoded_program[3].operands[3].kind == OperandKind::kImm32,
              "expected vop3 src2 inline immediate decode") ||
      !Expect(decoded_program[3].operands[3].imm32 == 5u,
              "expected vop3 src2 inline immediate value")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_vop3_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_vop3_state.halted,
              "expected vector vop3 program to halt") ||
      !Expect(vector_vop3_state.vgprs[30][0] == 63u,
              "expected decoded v_mul_lo_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[30][1] == 0xfffffff9u,
              "expected decoded v_mul_lo_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[30][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_lo_u32 result") ||
      !Expect(vector_vop3_state.vgprs[30][3] == 0x7f6e5d48u,
              "expected decoded v_mul_lo_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[31][0] == 2u,
              "expected decoded v_mul_hi_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[31][1] == 1u,
              "expected decoded v_mul_hi_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[31][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_hi_u32 result") ||
      !Expect(vector_vop3_state.vgprs[31][3] == 0u,
              "expected decoded v_mul_hi_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[32][0] == 0xffffffffu,
              "expected decoded v_mul_hi_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[32][1] == 0u,
              "expected decoded v_mul_hi_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[32][2] == 0xdeadbeefu,
              "expected inactive lane v_mul_hi_i32 result") ||
      !Expect(vector_vop3_state.vgprs[32][3] == 0xfffffffeu,
              "expected decoded v_mul_hi_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[33][0] == 8u,
              "expected decoded v_add3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[33][1] == 5u,
              "expected decoded v_add3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[33][2] == 0xdeadbeefu,
              "expected inactive lane v_add3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[33][3] == 13u,
              "expected decoded v_add3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[49][0] == 0x03040506u,
              "expected decoded v_lerp_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[49][1] == 0x0b0b0d0eu,
              "expected decoded v_lerp_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[49][2] == 0xdeadbeefu,
              "expected inactive lane v_lerp_u8 result") ||
      !Expect(vector_vop3_state.vgprs[49][3] == 0x82828384u,
              "expected decoded v_lerp_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[50][0] == 0xff007f44u,
              "expected decoded v_perm_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[50][1] == 0xff0000ffu,
              "expected decoded v_perm_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[50][2] == 0xdeadbeefu,
              "expected inactive lane v_perm_b32 result") ||
      !Expect(vector_vop3_state.vgprs[50][3] == 0x8001ffa1u,
              "expected decoded v_perm_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[34][0] == 0x56u,
              "expected decoded v_bfe_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[34][1] == 0x32u,
              "expected decoded v_bfe_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[34][2] == 0xdeadbeefu,
              "expected inactive lane v_bfe_u32 result") ||
      !Expect(vector_vop3_state.vgprs[34][3] == 0x0fu,
              "expected decoded v_bfe_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[35][0] == 0xffffffffu,
              "expected decoded v_bfe_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[35][1] == 0xffffffffu,
              "expected decoded v_bfe_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[35][2] == 0xdeadbeefu,
              "expected inactive lane v_bfe_i32 result") ||
      !Expect(vector_vop3_state.vgprs[35][3] == 0x00000007u,
              "expected decoded v_bfe_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[36][0] == 0x22112211u,
              "expected decoded v_bfi_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[36][1] == 0x55aa55aau,
              "expected decoded v_bfi_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[36][2] == 0xdeadbeefu,
              "expected inactive lane v_bfi_b32 result") ||
      !Expect(vector_vop3_state.vgprs[36][3] == 0x00ff00ffu,
              "expected decoded v_bfi_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[37][0] == 0x45566778u,
              "expected decoded v_alignbit_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[37][1] == 0xf5566778u,
              "expected decoded v_alignbit_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[37][2] == 0xdeadbeefu,
              "expected inactive lane v_alignbit_b32 result") ||
      !Expect(vector_vop3_state.vgprs[37][3] == 0x05566778u,
              "expected decoded v_alignbit_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[38][0] == 0x44556677u,
              "expected decoded v_alignbyte_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[38][1] == 0xef556677u,
              "expected decoded v_alignbyte_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[38][2] == 0xdeadbeefu,
              "expected inactive lane v_alignbyte_b32 result") ||
      !Expect(vector_vop3_state.vgprs[38][3] == 0xf0556677u,
              "expected decoded v_alignbyte_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[39][0] == 0x80000000u,
              "expected decoded v_min3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[39][1] == 0x89abcdefu,
              "expected decoded v_min3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[39][2] == 0xdeadbeefu,
              "expected inactive lane v_min3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[39][3] == 0xf0f0f0f0u,
              "expected decoded v_min3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[40][0] == 0x55667788u,
              "expected decoded v_max3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[40][1] == 0x55667788u,
              "expected decoded v_max3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[40][2] == 0xdeadbeefu,
              "expected inactive lane v_max3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[40][3] == 0x55667788u,
              "expected decoded v_max3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[41][0] == 0x11223344u,
              "expected decoded v_med3_i32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[41][1] == 0xffffffffu,
              "expected decoded v_med3_i32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[41][2] == 0xdeadbeefu,
              "expected inactive lane v_med3_i32 result") ||
      !Expect(vector_vop3_state.vgprs[41][3] == 0x00000010u,
              "expected decoded v_med3_i32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[42][0] == 0x11223344u,
              "expected decoded v_min3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[42][1] == 0x55667788u,
              "expected decoded v_min3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[42][2] == 0xdeadbeefu,
              "expected inactive lane v_min3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[42][3] == 0x00000010u,
              "expected decoded v_min3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[43][0] == 0x80000000u,
              "expected decoded v_max3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[43][1] == 0xffffffffu,
              "expected decoded v_max3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[43][2] == 0xdeadbeefu,
              "expected inactive lane v_max3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[43][3] == 0xf0f0f0f0u,
              "expected decoded v_max3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[44][0] == 0x55667788u,
              "expected decoded v_med3_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[44][1] == 0x89abcdefu,
              "expected decoded v_med3_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[44][2] == 0xdeadbeefu,
              "expected inactive lane v_med3_u32 result") ||
      !Expect(vector_vop3_state.vgprs[44][3] == 0x55667788u,
              "expected decoded v_med3_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[45][0] == 69u,
              "expected decoded v_sad_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[45][1] == 17u,
              "expected decoded v_sad_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[45][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u8 result") ||
      !Expect(vector_vop3_state.vgprs[45][3] == 13u,
              "expected decoded v_sad_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[46][0] == 0x00400005u,
              "expected decoded v_sad_hi_u8 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[46][1] == 0x000a0007u,
              "expected decoded v_sad_hi_u8 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[46][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_hi_u8 result") ||
      !Expect(vector_vop3_state.vgprs[46][3] == 0x00040009u,
              "expected decoded v_sad_hi_u8 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[47][0] == 19u,
              "expected decoded v_sad_u16 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[47][1] == 14u,
              "expected decoded v_sad_u16 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[47][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u16 result") ||
      !Expect(vector_vop3_state.vgprs[47][3] == 59u,
              "expected decoded v_sad_u16 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[48][0] == 28u,
              "expected decoded v_sad_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[48][1] == 30u,
              "expected decoded v_sad_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[48][2] == 0xdeadbeefu,
              "expected inactive lane v_sad_u32 result") ||
      !Expect(vector_vop3_state.vgprs[48][3] == 332u,
              "expected decoded v_sad_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[51][0] == 0xffffffffu,
              "expected decoded v_mad_i32_i24 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[51][1] == 11u,
              "expected decoded v_mad_i32_i24 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[51][2] == 0xdeadbeefu,
              "expected inactive lane v_mad_i32_i24 result") ||
      !Expect(vector_vop3_state.vgprs[51][3] == 0x01000007u,
              "expected decoded v_mad_i32_i24 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[52][0] == 0x00020009u,
              "expected decoded v_mad_u32_u24 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[52][1] == 0xffff000du,
              "expected decoded v_mad_u32_u24 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[52][2] == 0xdeadbeefu,
              "expected inactive lane v_mad_u32_u24 result") ||
      !Expect(vector_vop3_state.vgprs[52][3] == 0x000a0007u,
              "expected decoded v_mad_u32_u24 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[57][0] == 53u,
              "expected decoded v_lshl_add_u32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[57][1] == 0xfffffff7u,
              "expected decoded v_lshl_add_u32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[57][2] == 0xdeadbeefu,
              "expected inactive lane v_lshl_add_u32 result") ||
      !Expect(vector_vop3_state.vgprs[57][3] == 0x23456789u,
              "expected decoded v_lshl_add_u32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[58][0] == 116u,
              "expected decoded v_bcnt_u32_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[58][1] == 102u,
              "expected decoded v_bcnt_u32_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[58][2] == 0xdeadbeefu,
              "expected inactive lane v_bcnt_u32_b32 result") ||
      !Expect(vector_vop3_state.vgprs[58][3] == 132u,
              "expected decoded v_bcnt_u32_b32 lane 3 result") ||
      !Expect(vector_vop3_state.vgprs[59][0] == 0x000000f8u,
              "expected decoded v_bfm_b32 lane 0 result") ||
      !Expect(vector_vop3_state.vgprs[59][1] == 0x000003f8u,
              "expected decoded v_bfm_b32 lane 1 result") ||
      !Expect(vector_vop3_state.vgprs[59][2] == 0xdeadbeefu,
              "expected inactive lane v_bfm_b32 result") ||
      !Expect(vector_vop3_state.vgprs[59][3] == 0x00000008u,
              "expected decoded v_bfm_b32 lane 3 result")) {
    return 1;
  }

  const auto v_mbcnt_lo_u32_b32_opcode =
      FindDefaultEncodingOpcode("V_MBCNT_LO_U32_B32", "ENC_VOP3");
  const auto v_mbcnt_hi_u32_b32_opcode =
      FindDefaultEncodingOpcode("V_MBCNT_HI_U32_B32", "ENC_VOP3");
  if (!Expect(v_mbcnt_lo_u32_b32_opcode.has_value(),
              "expected V_MBCNT_LO_U32_B32 opcode lookup") ||
      !Expect(v_mbcnt_hi_u32_b32_opcode.has_value(),
              "expected V_MBCNT_HI_U32_B32 opcode lookup")) {
    return 1;
  }

  const auto vop3_mbcnt_lo_u32_b32_word =
      MakeVop3(*v_mbcnt_lo_u32_b32_opcode, 60, 326, 35);
  const auto vop3_mbcnt_hi_u32_b32_word =
      MakeVop3(*v_mbcnt_hi_u32_b32_opcode, 61, 326, 36);
  const std::vector<std::uint32_t> vector_mbcnt_program = {
      vop3_mbcnt_lo_u32_b32_word[0], vop3_mbcnt_lo_u32_b32_word[1],
      vop3_mbcnt_hi_u32_b32_word[0], vop3_mbcnt_hi_u32_b32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_mbcnt_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 3, "expected decoded vector MBCNT program size") ||
      !Expect(decoded_program[0].opcode == "V_MBCNT_LO_U32_B32",
              "expected V_MBCNT_LO_U32_B32 decode") ||
      !Expect(decoded_program[1].opcode == "V_MBCNT_HI_U32_B32",
              "expected V_MBCNT_HI_U32_B32 decode")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_mbcnt_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_mbcnt_state.halted,
              "expected vector MBCNT decode program to halt") ||
      !Expect(vector_mbcnt_state.vgprs[60][0] == 10u,
              "expected decoded v_mbcnt_lo_u32_b32 lane 0 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][1] == 11u,
              "expected decoded v_mbcnt_lo_u32_b32 lane 1 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][2] == 0xdeadbeefu,
              "expected inactive decoded v_mbcnt_lo_u32_b32 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][32] == 15u,
              "expected decoded v_mbcnt_lo_u32_b32 lane 32 result") ||
      !Expect(vector_mbcnt_state.vgprs[60][35] == 15u,
              "expected decoded v_mbcnt_lo_u32_b32 lane 35 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][0] == 40u,
              "expected decoded v_mbcnt_hi_u32_b32 lane 0 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][1] == 40u,
              "expected decoded v_mbcnt_hi_u32_b32 lane 1 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][2] == 0xdeadbeefu,
              "expected inactive decoded v_mbcnt_hi_u32_b32 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][32] == 40u,
              "expected decoded v_mbcnt_hi_u32_b32 lane 32 result") ||
      !Expect(vector_mbcnt_state.vgprs[61][35] == 42u,
              "expected decoded v_mbcnt_hi_u32_b32 lane 35 result")) {
    return 1;
  }

  const auto v_mov_b64_opcode =
      FindDefaultEncodingOpcode("V_MOV_B64", "ENC_VOP1");
  if (!Expect(v_mov_b64_opcode.has_value(),
              "expected V_MOV_B64 opcode lookup")) {
    return 1;
  }

  const auto vop1_mov_b64_from_vgpr_word = MakeVop1(*v_mov_b64_opcode, 68, 336);
  const auto vop1_mov_b64_from_sgpr_word = MakeVop1(*v_mov_b64_opcode, 70, 44);
  const std::vector<std::uint32_t> vector_move64_program = {
      vop1_mov_b64_from_vgpr_word,
      vop1_mov_b64_from_sgpr_word,
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_move64_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 3,
              "expected decoded vector move64 program size") ||
      !Expect(decoded_program[0].opcode == "V_MOV_B64",
              "expected first V_MOV_B64 decode") ||
      !Expect(decoded_program[1].opcode == "V_MOV_B64",
              "expected second V_MOV_B64 decode")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_move64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_move64_state.halted,
              "expected vector move64 decode program to halt") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][0],
                         vector_move64_state.vgprs[69][0]) ==
                  0xaaaaaaaa55555555ULL,
              "expected decoded v_mov_b64 from vgpr lane 0 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][1],
                         vector_move64_state.vgprs[69][1]) ==
                  0x0123456789abcdefULL,
              "expected decoded v_mov_b64 from vgpr lane 1 result") ||
      !Expect(vector_move64_state.vgprs[68][2] == 0xdeadbeefu &&
                  vector_move64_state.vgprs[69][2] == 0xcafebabeu,
              "expected inactive decoded v_mov_b64 from vgpr result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[68][35],
                         vector_move64_state.vgprs[69][35]) ==
                  0xfedcba9876543210ULL,
              "expected decoded v_mov_b64 from vgpr lane 35 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][0],
                         vector_move64_state.vgprs[71][0]) ==
                  0x0123456789abcdefULL,
              "expected decoded v_mov_b64 from sgpr lane 0 result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][1],
                         vector_move64_state.vgprs[71][1]) ==
                  0x0123456789abcdefULL,
              "expected decoded v_mov_b64 from sgpr lane 1 result") ||
      !Expect(vector_move64_state.vgprs[70][2] == 0xdeadbeefu &&
                  vector_move64_state.vgprs[71][2] == 0xcafebabeu,
              "expected inactive decoded v_mov_b64 from sgpr result") ||
      !Expect(ComposeU64(vector_move64_state.vgprs[70][35],
                         vector_move64_state.vgprs[71][35]) ==
                  0x0123456789abcdefULL,
              "expected decoded v_mov_b64 from sgpr lane 35 result")) {
    return 1;
  }

  const auto v_lshlrev_b64_opcode =
      FindDefaultEncodingOpcode("V_LSHLREV_B64", "ENC_VOP3");
  const auto v_lshrrev_b64_opcode =
      FindDefaultEncodingOpcode("V_LSHRREV_B64", "ENC_VOP3");
  const auto v_ashrrev_i64_opcode =
      FindDefaultEncodingOpcode("V_ASHRREV_I64", "ENC_VOP3");
  const auto v_lshl_add_u64_opcode =
      FindDefaultEncodingOpcode("V_LSHL_ADD_U64", "ENC_VOP3");
  const auto v_mad_u64_u32_opcode =
      FindDefaultEncodingOpcode("V_MAD_U64_U32", "VOP3_SDST_ENC");
  const auto v_mad_i64_i32_opcode =
      FindDefaultEncodingOpcode("V_MAD_I64_I32", "VOP3_SDST_ENC");
  if (!Expect(v_lshlrev_b64_opcode.has_value(),
              "expected V_LSHLREV_B64 opcode lookup") ||
      !Expect(v_lshrrev_b64_opcode.has_value(),
              "expected V_LSHRREV_B64 opcode lookup") ||
      !Expect(v_ashrrev_i64_opcode.has_value(),
              "expected V_ASHRREV_I64 opcode lookup") ||
      !Expect(v_lshl_add_u64_opcode.has_value(),
              "expected V_LSHL_ADD_U64 opcode lookup") ||
      !Expect(v_mad_u64_u32_opcode.has_value(),
              "expected V_MAD_U64_U32 opcode lookup") ||
      !Expect(v_mad_i64_i32_opcode.has_value(),
              "expected V_MAD_I64_I32 opcode lookup")) {
    return 1;
  }

  const auto vop3_lshlrev_b64_word =
      MakeVop3(*v_lshlrev_b64_opcode, 72, 40, 336);
  const auto vop3_lshrrev_b64_word =
      MakeVop3(*v_lshrrev_b64_opcode, 74, 41, 336);
  const auto vop3_ashrrev_i64_word =
      MakeVop3(*v_ashrrev_i64_opcode, 76, 42, 336);
  const auto vop3_lshl_add_u64_word =
      MakeVop3(*v_lshl_add_u64_opcode, 78, 336, 340, 338);
  const std::vector<std::uint32_t> vector_shift64_program = {
      vop3_lshlrev_b64_word[0], vop3_lshlrev_b64_word[1],
      vop3_lshrrev_b64_word[0], vop3_lshrrev_b64_word[1],
      vop3_ashrrev_i64_word[0], vop3_ashrrev_i64_word[1],
      vop3_lshl_add_u64_word[0], vop3_lshl_add_u64_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_shift64_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded vector shift64 program size") ||
      !Expect(decoded_program[0].opcode == "V_LSHLREV_B64",
              "expected V_LSHLREV_B64 decode") ||
      !Expect(decoded_program[1].opcode == "V_LSHRREV_B64",
              "expected V_LSHRREV_B64 decode") ||
      !Expect(decoded_program[2].opcode == "V_ASHRREV_I64",
              "expected V_ASHRREV_I64 decode") ||
      !Expect(decoded_program[3].opcode == "V_LSHL_ADD_U64",
              "expected V_LSHL_ADD_U64 decode")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_shift64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_shift64_state.halted,
              "expected vector shift64 decode program to halt") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][0],
                         vector_shift64_state.vgprs[73][0]) ==
                  0x123456789abcdef0ULL,
              "expected decoded v_lshlrev_b64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][1],
                         vector_shift64_state.vgprs[73][1]) == 0ULL,
              "expected decoded v_lshlrev_b64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[72][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[73][2] == 0xcafebabeu,
              "expected inactive decoded v_lshlrev_b64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[72][35],
                         vector_shift64_state.vgprs[73][35]) ==
                  0xffffffffffffff00ULL,
              "expected decoded v_lshlrev_b64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][0],
                         vector_shift64_state.vgprs[75][0]) ==
                  0x00123456789abcdeULL,
              "expected decoded v_lshrrev_b64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][1],
                         vector_shift64_state.vgprs[75][1]) ==
                  0x0800000000000000ULL,
              "expected decoded v_lshrrev_b64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[74][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[75][2] == 0xcafebabeu,
              "expected inactive decoded v_lshrrev_b64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[74][35],
                         vector_shift64_state.vgprs[75][35]) ==
                  0x0fffffffffffffffULL,
              "expected decoded v_lshrrev_b64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][0],
                         vector_shift64_state.vgprs[77][0]) ==
                  0x00123456789abcdeULL,
              "expected decoded v_ashrrev_i64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][1],
                         vector_shift64_state.vgprs[77][1]) ==
                  0xf800000000000000ULL,
              "expected decoded v_ashrrev_i64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[76][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[77][2] == 0xcafebabeu,
              "expected inactive decoded v_ashrrev_i64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[76][35],
                         vector_shift64_state.vgprs[77][35]) ==
                  0xffffffffffffffffULL,
              "expected decoded v_ashrrev_i64 lane 35 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][0],
                         vector_shift64_state.vgprs[79][0]) ==
                  0x23456789abcdf001ULL,
              "expected decoded v_lshl_add_u64 lane 0 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][1],
                         vector_shift64_state.vgprs[79][1]) == 3ULL,
              "expected decoded v_lshl_add_u64 lane 1 result") ||
      !Expect(vector_shift64_state.vgprs[78][2] == 0xdeadbeefu &&
                  vector_shift64_state.vgprs[79][2] == 0xcafebabeu,
              "expected inactive decoded v_lshl_add_u64 result") ||
      !Expect(ComposeU64(vector_shift64_state.vgprs[78][35],
                         vector_shift64_state.vgprs[79][35]) ==
                  0xffffffffffffff10ULL,
              "expected decoded v_lshl_add_u64 lane 35 result")) {
    return 1;
  }

  const auto vop3_mad_u64_u32_word =
      MakeVop3Sdst(*v_mad_u64_u32_opcode, 86, 118, 46, 344, 338);
  const auto vop3_mad_i64_i32_word =
      MakeVop3Sdst(*v_mad_i64_i32_opcode, 90, 106, 47, 345, 340);
  const std::vector<std::uint32_t> vector_mad64_program = {
      vop3_mad_u64_u32_word[0], vop3_mad_u64_u32_word[1],
      vop3_mad_i64_i32_word[0], vop3_mad_i64_i32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_mad64_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 3,
              "expected decoded vector mad64 program size") ||
      !Expect(decoded_program[0].opcode == "V_MAD_U64_U32",
              "expected V_MAD_U64_U32 decode") ||
      !Expect(decoded_program[1].opcode == "V_MAD_I64_I32",
              "expected V_MAD_I64_I32 decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_MAD_U64_U32 operand count") ||
      !Expect(decoded_program[1].operand_count == 5,
              "expected V_MAD_I64_I32 operand count")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_mad64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_mad64_state.halted,
              "expected decoded vector mad64 program to halt") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][0],
                         vector_mad64_state.vgprs[87][0]) ==
                  0x00000001ffffffffULL,
              "expected decoded v_mad_u64_u32 lane 0 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][1],
                         vector_mad64_state.vgprs[87][1]) ==
                  0x00000000fffffffeULL,
              "expected decoded v_mad_u64_u32 lane 1 result") ||
      !Expect(vector_mad64_state.vgprs[86][2] == 0xdeadbeefu &&
                  vector_mad64_state.vgprs[87][2] == 0xcafebabeu,
              "expected decoded inactive v_mad_u64_u32 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[86][35],
                         vector_mad64_state.vgprs[87][35]) ==
                  0x0000000000000005ULL,
              "expected decoded v_mad_u64_u32 lane 35 result") ||
      !Expect(vector_mad64_state.sgprs[118] == 0x0000000au &&
                  vector_mad64_state.sgprs[119] == 0u,
              "expected decoded v_mad_u64_u32 sdst mask") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][0],
                         vector_mad64_state.vgprs[91][0]) ==
                  0x8000000000000000ULL,
              "expected decoded v_mad_i64_i32 lane 0 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][1],
                         vector_mad64_state.vgprs[91][1]) ==
                  0x0000000000000005ULL,
              "expected decoded v_mad_i64_i32 lane 1 result") ||
      !Expect(vector_mad64_state.vgprs[90][2] == 0xdeadbeefu &&
                  vector_mad64_state.vgprs[91][2] == 0xcafebabeu,
              "expected decoded inactive v_mad_i64_i32 result") ||
      !Expect(ComposeU64(vector_mad64_state.vgprs[90][35],
                         vector_mad64_state.vgprs[91][35]) ==
                  0x0000000080000009ULL,
              "expected decoded v_mad_i64_i32 lane 35 result") ||
      !Expect(vector_mad64_state.sgprs[106] == 0x00000005u &&
                  vector_mad64_state.sgprs[107] == 0u,
              "expected decoded v_mad_i64_i32 sdst mask") ||
      !Expect(vector_mad64_state.vcc_mask == 0x0000000000000005ULL,
              "expected decoded final vcc mask after v_mad_i64_i32")) {
    return 1;
  }

  const auto v_add_f64_opcode =
      FindDefaultEncodingOpcode("V_ADD_F64", "ENC_VOP3");
  const auto v_mul_f64_opcode =
      FindDefaultEncodingOpcode("V_MUL_F64", "ENC_VOP3");
  const auto v_fma_f32_opcode =
      FindDefaultEncodingOpcode("V_FMA_F32", "ENC_VOP3");
  const auto v_fma_f64_opcode =
      FindDefaultEncodingOpcode("V_FMA_F64", "ENC_VOP3");
  if (!Expect(v_add_f64_opcode.has_value(), "expected V_ADD_F64 opcode lookup") ||
      !Expect(v_mul_f64_opcode.has_value(), "expected V_MUL_F64 opcode lookup") ||
      !Expect(v_fma_f32_opcode.has_value(), "expected V_FMA_F32 opcode lookup") ||
      !Expect(v_fma_f64_opcode.has_value(), "expected V_FMA_F64 opcode lookup")) {
    return 1;
  }

  const auto vop3_add_f64_word = MakeVop3(*v_add_f64_opcode, 50, 80, 296);
  const auto vop3_mul_f64_word = MakeVop3(*v_mul_f64_opcode, 52, 82, 298);
  const auto vop3_fma_f32_word =
      MakeVop3(*v_fma_f32_opcode, 54, 65, 300, 301);
  const auto vop3_fma_f64_word =
      MakeVop3(*v_fma_f64_opcode, 56, 84, 302, 304);
  const std::vector<std::uint32_t> vector_float_vop3_program = {
      vop3_add_f64_word[0], vop3_add_f64_word[1],
      vop3_mul_f64_word[0], vop3_mul_f64_word[1],
      vop3_fma_f32_word[0], vop3_fma_f32_word[1],
      vop3_fma_f64_word[0], vop3_fma_f64_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_float_vop3_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded vector float vop3 program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_F64",
              "expected V_ADD_F64 decode") ||
      !Expect(decoded_program[1].opcode == "V_MUL_F64",
              "expected V_MUL_F64 decode") ||
      !Expect(decoded_program[2].opcode == "V_FMA_F32",
              "expected V_FMA_F32 decode") ||
      !Expect(decoded_program[3].opcode == "V_FMA_F64",
              "expected V_FMA_F64 decode")) {
    return 1;
  }

  WaveExecutionState vector_float_vop3_state;
  vector_float_vop3_state.exec_mask = 0b1011ULL;
  vector_float_vop3_state.sgprs[65] = FloatBits(1.5f);
  SplitU64(DoubleBits(1.25), &vector_float_vop3_state.sgprs[80],
           &vector_float_vop3_state.sgprs[81]);
  SplitU64(DoubleBits(-2.0), &vector_float_vop3_state.sgprs[82],
           &vector_float_vop3_state.sgprs[83]);
  SplitU64(DoubleBits(1.5), &vector_float_vop3_state.sgprs[84],
           &vector_float_vop3_state.sgprs[85]);
  SplitU64(DoubleBits(2.5), &vector_float_vop3_state.vgprs[40][0],
           &vector_float_vop3_state.vgprs[41][0]);
  SplitU64(DoubleBits(-0.25), &vector_float_vop3_state.vgprs[40][1],
           &vector_float_vop3_state.vgprs[41][1]);
  SplitU64(DoubleBits(0.75), &vector_float_vop3_state.vgprs[40][3],
           &vector_float_vop3_state.vgprs[41][3]);
  SplitU64(DoubleBits(1.5), &vector_float_vop3_state.vgprs[42][0],
           &vector_float_vop3_state.vgprs[43][0]);
  SplitU64(DoubleBits(-0.5), &vector_float_vop3_state.vgprs[42][1],
           &vector_float_vop3_state.vgprs[43][1]);
  SplitU64(DoubleBits(4.0), &vector_float_vop3_state.vgprs[42][3],
           &vector_float_vop3_state.vgprs[43][3]);
  vector_float_vop3_state.vgprs[44][0] = FloatBits(2.0f);
  vector_float_vop3_state.vgprs[44][1] = FloatBits(-2.0f);
  vector_float_vop3_state.vgprs[44][3] = FloatBits(4.0f);
  vector_float_vop3_state.vgprs[45][0] = FloatBits(0.5f);
  vector_float_vop3_state.vgprs[45][1] = FloatBits(1.0f);
  vector_float_vop3_state.vgprs[45][3] = FloatBits(-1.0f);
  SplitU64(DoubleBits(2.0), &vector_float_vop3_state.vgprs[46][0],
           &vector_float_vop3_state.vgprs[47][0]);
  SplitU64(DoubleBits(-2.0), &vector_float_vop3_state.vgprs[46][1],
           &vector_float_vop3_state.vgprs[47][1]);
  SplitU64(DoubleBits(4.0), &vector_float_vop3_state.vgprs[46][3],
           &vector_float_vop3_state.vgprs[47][3]);
  SplitU64(DoubleBits(0.5), &vector_float_vop3_state.vgprs[48][0],
           &vector_float_vop3_state.vgprs[49][0]);
  SplitU64(DoubleBits(1.0), &vector_float_vop3_state.vgprs[48][1],
           &vector_float_vop3_state.vgprs[49][1]);
  SplitU64(DoubleBits(-1.0), &vector_float_vop3_state.vgprs[48][3],
           &vector_float_vop3_state.vgprs[49][3]);
  vector_float_vop3_state.vgprs[50][2] = 0xdeadbeefu;
  vector_float_vop3_state.vgprs[51][2] = 0xcafebabeu;
  vector_float_vop3_state.vgprs[52][2] = 0xdeadbeefu;
  vector_float_vop3_state.vgprs[53][2] = 0xcafebabeu;
  vector_float_vop3_state.vgprs[54][2] = 0xdeadbeefu;
  vector_float_vop3_state.vgprs[56][2] = 0xdeadbeefu;
  vector_float_vop3_state.vgprs[57][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_float_vop3_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_float_vop3_state.halted,
              "expected vector float vop3 program to halt") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[50][0],
                         vector_float_vop3_state.vgprs[51][0]) ==
                  DoubleBits(3.75),
              "expected decoded V_ADD_F64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[50][1],
                         vector_float_vop3_state.vgprs[51][1]) ==
                  DoubleBits(1.0),
              "expected decoded V_ADD_F64 lane 1 result") ||
      !Expect(vector_float_vop3_state.vgprs[50][2] == 0xdeadbeefu &&
                  vector_float_vop3_state.vgprs[51][2] == 0xcafebabeu,
              "expected inactive decoded V_ADD_F64 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[50][3],
                         vector_float_vop3_state.vgprs[51][3]) ==
                  DoubleBits(2.0),
              "expected decoded V_ADD_F64 lane 3 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[52][0],
                         vector_float_vop3_state.vgprs[53][0]) ==
                  DoubleBits(-3.0),
              "expected decoded V_MUL_F64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[52][1],
                         vector_float_vop3_state.vgprs[53][1]) ==
                  DoubleBits(1.0),
              "expected decoded V_MUL_F64 lane 1 result") ||
      !Expect(vector_float_vop3_state.vgprs[52][2] == 0xdeadbeefu &&
                  vector_float_vop3_state.vgprs[53][2] == 0xcafebabeu,
              "expected inactive decoded V_MUL_F64 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[52][3],
                         vector_float_vop3_state.vgprs[53][3]) ==
                  DoubleBits(-8.0),
              "expected decoded V_MUL_F64 lane 3 result") ||
      !Expect(vector_float_vop3_state.vgprs[54][0] == FloatBits(3.5f),
              "expected decoded V_FMA_F32 lane 0 result") ||
      !Expect(vector_float_vop3_state.vgprs[54][1] == FloatBits(-2.0f),
              "expected decoded V_FMA_F32 lane 1 result") ||
      !Expect(vector_float_vop3_state.vgprs[54][2] == 0xdeadbeefu,
              "expected inactive decoded V_FMA_F32 result") ||
      !Expect(vector_float_vop3_state.vgprs[54][3] == FloatBits(5.0f),
              "expected decoded V_FMA_F32 lane 3 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[56][0],
                         vector_float_vop3_state.vgprs[57][0]) ==
                  DoubleBits(3.5),
              "expected decoded V_FMA_F64 lane 0 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[56][1],
                         vector_float_vop3_state.vgprs[57][1]) ==
                  DoubleBits(-2.0),
              "expected decoded V_FMA_F64 lane 1 result") ||
      !Expect(vector_float_vop3_state.vgprs[56][2] == 0xdeadbeefu &&
                  vector_float_vop3_state.vgprs[57][2] == 0xcafebabeu,
              "expected inactive decoded V_FMA_F64 result") ||
      !Expect(ComposeU64(vector_float_vop3_state.vgprs[56][3],
                         vector_float_vop3_state.vgprs[57][3]) ==
                  DoubleBits(5.0),
              "expected decoded V_FMA_F64 lane 3 result")) {
    return 1;
  }

  const auto v_add_lshl_u32_opcode =
      FindDefaultEncodingOpcode("V_ADD_LSHL_U32", "ENC_VOP3");
  const auto v_lshl_or_b32_opcode =
      FindDefaultEncodingOpcode("V_LSHL_OR_B32", "ENC_VOP3");
  const auto v_and_or_b32_opcode =
      FindDefaultEncodingOpcode("V_AND_OR_B32", "ENC_VOP3");
  const auto v_or3_b32_opcode =
      FindDefaultEncodingOpcode("V_OR3_B32", "ENC_VOP3");
  const auto v_xad_u32_opcode =
      FindDefaultEncodingOpcode("V_XAD_U32", "ENC_VOP3");
  if (!Expect(v_add_lshl_u32_opcode.has_value(),
              "expected V_ADD_LSHL_U32 opcode lookup") ||
      !Expect(v_lshl_or_b32_opcode.has_value(),
              "expected V_LSHL_OR_B32 opcode lookup") ||
      !Expect(v_and_or_b32_opcode.has_value(),
              "expected V_AND_OR_B32 opcode lookup") ||
      !Expect(v_or3_b32_opcode.has_value(),
              "expected V_OR3_B32 opcode lookup") ||
      !Expect(v_xad_u32_opcode.has_value(),
              "expected V_XAD_U32 opcode lookup")) {
    return 1;
  }

  const auto vop3_add_lshl_u32_word =
      MakeVop3(*v_add_lshl_u32_opcode, 82, 339, 43, 340);
  const auto vop3_lshl_or_b32_word =
      MakeVop3(*v_lshl_or_b32_opcode, 85, 342, 44, 343);
  const auto vop3_and_or_b32_word =
      MakeVop3(*v_and_or_b32_opcode, 88, 345, 45, 346);
  const auto vop3_or3_b32_word =
      MakeVop3(*v_or3_b32_opcode, 91, 348, 46, 349);
  const auto vop3_xad_u32_word =
      MakeVop3(*v_xad_u32_opcode, 94, 351, 47, 352);
  const std::vector<std::uint32_t> vector_ternary_logic_program = {
      vop3_add_lshl_u32_word[0], vop3_add_lshl_u32_word[1],
      vop3_lshl_or_b32_word[0], vop3_lshl_or_b32_word[1],
      vop3_and_or_b32_word[0], vop3_and_or_b32_word[1],
      vop3_or3_b32_word[0], vop3_or3_b32_word[1],
      vop3_xad_u32_word[0], vop3_xad_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_ternary_logic_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 6,
              "expected decoded vector ternary logic program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_LSHL_U32",
              "expected V_ADD_LSHL_U32 decode") ||
      !Expect(decoded_program[1].opcode == "V_LSHL_OR_B32",
              "expected V_LSHL_OR_B32 decode") ||
      !Expect(decoded_program[2].opcode == "V_AND_OR_B32",
              "expected V_AND_OR_B32 decode") ||
      !Expect(decoded_program[3].opcode == "V_OR3_B32",
              "expected V_OR3_B32 decode") ||
      !Expect(decoded_program[4].opcode == "V_XAD_U32",
              "expected V_XAD_U32 decode")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_ternary_logic_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_ternary_logic_state.halted,
              "expected vector ternary logic decode program to halt") ||
      !Expect(vector_ternary_logic_state.vgprs[82][0] == 16u,
              "expected decoded v_add_lshl_u32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][1] == 4u,
              "expected decoded v_add_lshl_u32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][2] == 0xdeadbeefu,
              "expected inactive decoded v_add_lshl_u32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[82][3] == 208u,
              "expected decoded v_add_lshl_u32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][0] == 0x00000123u,
              "expected decoded v_lshl_or_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][1] == 0xffff0010u,
              "expected decoded v_lshl_or_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][2] == 0xdeadbeefu,
              "expected inactive decoded v_lshl_or_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[85][3] == 0x0000000fu,
              "expected decoded v_lshl_or_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][0] == 0x00f000ffu,
              "expected decoded v_and_or_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][1] == 0x0aa00af0u,
              "expected decoded v_and_or_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][2] == 0xdeadbeefu,
              "expected inactive decoded v_and_or_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[88][3] == 0xff000f00u,
              "expected decoded v_and_or_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][0] == 0x00fff00fu,
              "expected decoded v_or3_b32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][1] == 0xf0fff000u,
              "expected decoded v_or3_b32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][2] == 0xdeadbeefu,
              "expected inactive decoded v_or3_b32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[91][3] == 0x0fff0f0fu,
              "expected decoded v_or3_b32 lane 3 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][0] == 0xf0f0f0f1u,
              "expected decoded v_xad_u32 lane 0 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][1] == 0x1d3b5979u,
              "expected decoded v_xad_u32 lane 1 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][2] == 0xdeadbeefu,
              "expected inactive decoded v_xad_u32 result") ||
      !Expect(vector_ternary_logic_state.vgprs[94][3] == 0x0f0f0effu,
              "expected decoded v_xad_u32 lane 3 result")) {
    return 1;
  }

  const auto v_subrev_u32_opcode =
      FindDefaultEncodingOpcode("V_SUBREV_U32", "ENC_VOP2");
  if (!Expect(v_subrev_u32_opcode.has_value(),
              "expected V_SUBREV_U32 opcode lookup")) {
    return 1;
  }

  const auto vop2_subrev_u32_word =
      MakeVop2(*v_subrev_u32_opcode, 98, 48, 97);
  const std::vector<std::uint32_t> vector_binary_extra_program = {
      vop2_subrev_u32_word,
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_binary_extra_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded vector binary extra program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBREV_U32",
              "expected V_SUBREV_U32 decode")) {
    return 1;
  }

  WaveExecutionState vector_binary_extra_state;
  vector_binary_extra_state.exec_mask = 0b1011ULL;
  vector_binary_extra_state.sgprs[48] = 10u;
  vector_binary_extra_state.vgprs[97][0] = 20u;
  vector_binary_extra_state.vgprs[97][1] = 5u;
  vector_binary_extra_state.vgprs[97][3] = 0u;
  vector_binary_extra_state.vgprs[98][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_binary_extra_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_binary_extra_state.halted,
              "expected vector binary extra decode program to halt") ||
      !Expect(vector_binary_extra_state.vgprs[98][0] == 10u,
              "expected decoded v_subrev_u32 lane 0 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][1] == 0xfffffffbu,
              "expected decoded v_subrev_u32 lane 1 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][2] == 0xdeadbeefu,
              "expected inactive decoded v_subrev_u32 result") ||
      !Expect(vector_binary_extra_state.vgprs[98][3] == 0xfffffff6u,
              "expected decoded v_subrev_u32 lane 3 result")) {
    return 1;
  }

  const auto v_not_b32_vop3_opcode =
      FindDefaultEncodingOpcode("V_NOT_B32", "ENC_VOP3");
  const auto v_cvt_f32_i32_vop3_opcode =
      FindDefaultEncodingOpcode("V_CVT_F32_I32", "ENC_VOP3");
  const auto v_cvt_f64_i32_vop3_opcode =
      FindDefaultEncodingOpcode("V_CVT_F64_I32", "ENC_VOP3");
  const auto v_cvt_i32_f64_vop3_opcode =
      FindDefaultEncodingOpcode("V_CVT_I32_F64", "ENC_VOP3");
  const auto v_mov_b64_vop3_opcode =
      FindDefaultEncodingOpcode("V_MOV_B64", "ENC_VOP3");
  const auto v_readfirstlane_b32_vop3_opcode =
      FindDefaultEncodingOpcode("V_READFIRSTLANE_B32", "ENC_VOP3");
  if (!Expect(v_not_b32_vop3_opcode.has_value(),
              "expected V_NOT_B32 VOP3 opcode lookup") ||
      !Expect(v_cvt_f32_i32_vop3_opcode.has_value(),
              "expected V_CVT_F32_I32 VOP3 opcode lookup") ||
      !Expect(v_cvt_f64_i32_vop3_opcode.has_value(),
              "expected V_CVT_F64_I32 VOP3 opcode lookup") ||
      !Expect(v_cvt_i32_f64_vop3_opcode.has_value(),
              "expected V_CVT_I32_F64 VOP3 opcode lookup") ||
      !Expect(v_mov_b64_vop3_opcode.has_value(),
              "expected V_MOV_B64 VOP3 opcode lookup") ||
      !Expect(v_readfirstlane_b32_vop3_opcode.has_value(),
              "expected V_READFIRSTLANE_B32 VOP3 opcode lookup")) {
    return 1;
  }

  const auto vop3_not_b32_word =
      MakeVop3(*v_not_b32_vop3_opcode, 100, 10, 0, 0);
  const auto vop3_cvt_f32_i32_word =
      MakeVop3(*v_cvt_f32_i32_vop3_opcode, 101, 11, 0, 0);
  const auto vop3_cvt_f64_i32_word =
      MakeVop3(*v_cvt_f64_i32_vop3_opcode, 102, 12, 0, 0);
  const auto vop3_cvt_i32_f64_word =
      MakeVop3(*v_cvt_i32_f64_vop3_opcode, 104, 372, 0, 0);
  const auto vop3_mov_b64_word =
      MakeVop3(*v_mov_b64_vop3_opcode, 105, 374, 0, 0);
  const auto vop3_readfirstlane_b32_word =
      MakeVop3(*v_readfirstlane_b32_vop3_opcode, 2, 376, 0, 0);
  const std::vector<std::uint32_t> promoted_vop3_unary_program = {
      vop3_not_b32_word[0], vop3_not_b32_word[1],
      vop3_cvt_f32_i32_word[0], vop3_cvt_f32_i32_word[1],
      vop3_cvt_f64_i32_word[0], vop3_cvt_f64_i32_word[1],
      vop3_cvt_i32_f64_word[0], vop3_cvt_i32_f64_word[1],
      vop3_mov_b64_word[0], vop3_mov_b64_word[1],
      vop3_readfirstlane_b32_word[0], vop3_readfirstlane_b32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(promoted_vop3_unary_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 7,
              "expected decoded promoted VOP3 unary program size") ||
      !Expect(decoded_program[0].opcode == "V_NOT_B32",
              "expected promoted V_NOT_B32 decode") ||
      !Expect(decoded_program[1].opcode == "V_CVT_F32_I32",
              "expected promoted V_CVT_F32_I32 decode") ||
      !Expect(decoded_program[2].opcode == "V_CVT_F64_I32",
              "expected promoted V_CVT_F64_I32 decode") ||
      !Expect(decoded_program[3].opcode == "V_CVT_I32_F64",
              "expected promoted V_CVT_I32_F64 decode") ||
      !Expect(decoded_program[4].opcode == "V_MOV_B64",
              "expected promoted V_MOV_B64 decode") ||
      !Expect(decoded_program[5].opcode == "V_READFIRSTLANE_B32",
              "expected promoted V_READFIRSTLANE_B32 decode") ||
      !Expect(decoded_program[5].operands[0].kind == OperandKind::kSgpr &&
                  decoded_program[5].operands[0].index == 2,
              "expected promoted readfirstlane scalar destination decode")) {
    return 1;
  }

  WaveExecutionState promoted_vop3_unary_state;
  promoted_vop3_unary_state.exec_mask = 0b1011ULL;
  promoted_vop3_unary_state.sgprs[10] = 0x0f0f0000u;
  promoted_vop3_unary_state.sgprs[11] = 0xfffffff9u;
  promoted_vop3_unary_state.sgprs[12] = 0xfffffff5u;
  SplitU64(DoubleBits(2.5), &promoted_vop3_unary_state.vgprs[116][0],
           &promoted_vop3_unary_state.vgprs[117][0]);
  SplitU64(DoubleBits(-0.25), &promoted_vop3_unary_state.vgprs[116][1],
           &promoted_vop3_unary_state.vgprs[117][1]);
  SplitU64(DoubleBits(8.0), &promoted_vop3_unary_state.vgprs[116][3],
           &promoted_vop3_unary_state.vgprs[117][3]);
  SplitU64(0x1122334455667788ULL, &promoted_vop3_unary_state.vgprs[118][0],
           &promoted_vop3_unary_state.vgprs[119][0]);
  SplitU64(0x0123456789abcdefULL, &promoted_vop3_unary_state.vgprs[118][1],
           &promoted_vop3_unary_state.vgprs[119][1]);
  SplitU64(0xfedcba9876543210ULL, &promoted_vop3_unary_state.vgprs[118][3],
           &promoted_vop3_unary_state.vgprs[119][3]);
  promoted_vop3_unary_state.vgprs[120][0] = 111u;
  promoted_vop3_unary_state.vgprs[120][1] = 222u;
  promoted_vop3_unary_state.vgprs[120][3] = 333u;
  promoted_vop3_unary_state.vgprs[100][2] = 0xdeadbeefu;
  promoted_vop3_unary_state.vgprs[101][2] = 0xdeadbeefu;
  promoted_vop3_unary_state.vgprs[102][2] = 0xdeadbeefu;
  promoted_vop3_unary_state.vgprs[103][2] = 0xcafebabeu;
  promoted_vop3_unary_state.vgprs[104][2] = 0xdeadbeefu;
  promoted_vop3_unary_state.vgprs[105][2] = 0xdeadbeefu;
  promoted_vop3_unary_state.vgprs[106][2] = 0xcafebabeu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &promoted_vop3_unary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(promoted_vop3_unary_state.halted,
              "expected promoted VOP3 unary program to halt") ||
      !Expect(promoted_vop3_unary_state.vgprs[100][0] == 0xf0f0ffffu,
              "expected promoted V_NOT_B32 lane 0 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[100][2] == 0xdeadbeefu,
              "expected inactive promoted V_NOT_B32 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[101][1] == FloatBits(-7.0f),
              "expected promoted V_CVT_F32_I32 lane 1 result") ||
      !Expect(ComposeU64(promoted_vop3_unary_state.vgprs[102][0],
                         promoted_vop3_unary_state.vgprs[103][0]) ==
                  DoubleBits(-11.0),
              "expected promoted V_CVT_F64_I32 lane 0 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[102][2] == 0xdeadbeefu &&
                  promoted_vop3_unary_state.vgprs[103][2] == 0xcafebabeu,
              "expected inactive promoted V_CVT_F64_I32 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[104][0] == 2u,
              "expected promoted V_CVT_I32_F64 lane 0 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[104][1] == 0u,
              "expected promoted V_CVT_I32_F64 lane 1 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[104][2] == 0xdeadbeefu,
              "expected inactive promoted V_CVT_I32_F64 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[104][3] == 8u,
              "expected promoted V_CVT_I32_F64 lane 3 result") ||
      !Expect(ComposeU64(promoted_vop3_unary_state.vgprs[105][0],
                         promoted_vop3_unary_state.vgprs[106][0]) ==
                  0x1122334455667788ULL,
              "expected promoted V_MOV_B64 lane 0 result") ||
      !Expect(promoted_vop3_unary_state.vgprs[105][2] == 0xdeadbeefu &&
                  promoted_vop3_unary_state.vgprs[106][2] == 0xcafebabeu,
              "expected inactive promoted V_MOV_B64 result") ||
      !Expect(promoted_vop3_unary_state.sgprs[2] == 111u,
              "expected promoted V_READFIRSTLANE_B32 result")) {
    return 1;
  }

  const auto v_add_f32_vop3_opcode =
      FindDefaultEncodingOpcode("V_ADD_F32", "ENC_VOP3");
  const auto v_min_i32_vop3_opcode =
      FindDefaultEncodingOpcode("V_MIN_I32", "ENC_VOP3");
  const auto v_xor_b32_vop3_opcode =
      FindDefaultEncodingOpcode("V_XOR_B32", "ENC_VOP3");
  const auto v_subrev_u32_vop3_opcode =
      FindDefaultEncodingOpcode("V_SUBREV_U32", "ENC_VOP3");
  if (!Expect(v_add_f32_vop3_opcode.has_value(),
              "expected V_ADD_F32 VOP3 opcode lookup") ||
      !Expect(v_min_i32_vop3_opcode.has_value(),
              "expected V_MIN_I32 VOP3 opcode lookup") ||
      !Expect(v_xor_b32_vop3_opcode.has_value(),
              "expected V_XOR_B32 VOP3 opcode lookup") ||
      !Expect(v_subrev_u32_vop3_opcode.has_value(),
              "expected V_SUBREV_U32 VOP3 opcode lookup")) {
    return 1;
  }

  const auto vop3_add_f32_word =
      MakeVop3(*v_add_f32_vop3_opcode, 107, 20, 377, 0);
  const auto vop3_min_i32_word =
      MakeVop3(*v_min_i32_vop3_opcode, 108, 21, 378, 0);
  const auto vop3_xor_b32_word =
      MakeVop3(*v_xor_b32_vop3_opcode, 109, 22, 379, 0);
  const auto vop3_subrev_u32_word =
      MakeVop3(*v_subrev_u32_vop3_opcode, 110, 23, 380, 0);
  const std::vector<std::uint32_t> promoted_vop3_binary_program = {
      vop3_add_f32_word[0], vop3_add_f32_word[1],
      vop3_min_i32_word[0], vop3_min_i32_word[1],
      vop3_xor_b32_word[0], vop3_xor_b32_word[1],
      vop3_subrev_u32_word[0], vop3_subrev_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(promoted_vop3_binary_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded promoted VOP3 binary program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_F32",
              "expected promoted V_ADD_F32 decode") ||
      !Expect(decoded_program[1].opcode == "V_MIN_I32",
              "expected promoted V_MIN_I32 decode") ||
      !Expect(decoded_program[2].opcode == "V_XOR_B32",
              "expected promoted V_XOR_B32 decode") ||
      !Expect(decoded_program[3].opcode == "V_SUBREV_U32",
              "expected promoted V_SUBREV_U32 decode")) {
    return 1;
  }

  WaveExecutionState promoted_vop3_binary_state;
  promoted_vop3_binary_state.exec_mask = 0b1011ULL;
  promoted_vop3_binary_state.sgprs[20] = FloatBits(1.5f);
  promoted_vop3_binary_state.sgprs[21] = 0xfffffffcu;
  promoted_vop3_binary_state.sgprs[22] = 0x0f0f0f0fu;
  promoted_vop3_binary_state.sgprs[23] = 10u;
  promoted_vop3_binary_state.vgprs[121][0] = FloatBits(2.0f);
  promoted_vop3_binary_state.vgprs[121][1] = FloatBits(-2.25f);
  promoted_vop3_binary_state.vgprs[121][3] = FloatBits(0.5f);
  promoted_vop3_binary_state.vgprs[122][0] = 0xfffffff6u;
  promoted_vop3_binary_state.vgprs[122][1] = 0xfffffffdu;
  promoted_vop3_binary_state.vgprs[122][3] = 12u;
  promoted_vop3_binary_state.vgprs[123][0] = 0xff00ff00u;
  promoted_vop3_binary_state.vgprs[123][1] = 0x12345678u;
  promoted_vop3_binary_state.vgprs[123][3] = 0x0000ffffu;
  promoted_vop3_binary_state.vgprs[124][0] = 20u;
  promoted_vop3_binary_state.vgprs[124][1] = 5u;
  promoted_vop3_binary_state.vgprs[124][3] = 0u;
  promoted_vop3_binary_state.vgprs[107][2] = 0xdeadbeefu;
  promoted_vop3_binary_state.vgprs[108][2] = 0xdeadbeefu;
  promoted_vop3_binary_state.vgprs[109][2] = 0xdeadbeefu;
  promoted_vop3_binary_state.vgprs[110][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &promoted_vop3_binary_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(promoted_vop3_binary_state.halted,
              "expected promoted VOP3 binary program to halt") ||
      !Expect(promoted_vop3_binary_state.vgprs[107][0] == FloatBits(3.5f),
              "expected promoted V_ADD_F32 lane 0 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[107][1] == FloatBits(-0.75f),
              "expected promoted V_ADD_F32 lane 1 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[107][2] == 0xdeadbeefu,
              "expected inactive promoted V_ADD_F32 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[107][3] == FloatBits(2.0f),
              "expected promoted V_ADD_F32 lane 3 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[108][0] == 0xfffffff6u,
              "expected promoted V_MIN_I32 lane 0 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[108][1] == 0xfffffffcu,
              "expected promoted V_MIN_I32 lane 1 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[108][2] == 0xdeadbeefu,
              "expected inactive promoted V_MIN_I32 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[108][3] == 0xfffffffc,
              "expected promoted V_MIN_I32 lane 3 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[109][0] == 0xf00ff00fu,
              "expected promoted V_XOR_B32 lane 0 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[109][1] == 0x1d3b5977u,
              "expected promoted V_XOR_B32 lane 1 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[109][2] == 0xdeadbeefu,
              "expected inactive promoted V_XOR_B32 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[109][3] == 0x0f0ff0f0u,
              "expected promoted V_XOR_B32 lane 3 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[110][0] == 10u,
              "expected promoted V_SUBREV_U32 lane 0 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[110][1] == 0xfffffffbu,
              "expected promoted V_SUBREV_U32 lane 1 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[110][2] == 0xdeadbeefu,
              "expected inactive promoted V_SUBREV_U32 result") ||
      !Expect(promoted_vop3_binary_state.vgprs[110][3] == 0xfffffff6u,
              "expected promoted V_SUBREV_U32 lane 3 result")) {
    return 1;
  }

  const auto v_add_co_u32_opcode =
      FindDefaultEncodingOpcode("V_ADD_CO_U32", "ENC_VOP2");
  const auto v_addc_co_u32_opcode =
      FindDefaultEncodingOpcode("V_ADDC_CO_U32", "ENC_VOP2");
  const auto v_sub_co_u32_opcode =
      FindDefaultEncodingOpcode("V_SUB_CO_U32", "ENC_VOP2");
  const auto v_subb_co_u32_opcode =
      FindDefaultEncodingOpcode("V_SUBB_CO_U32", "ENC_VOP2");
  const auto v_subrev_co_u32_opcode =
      FindDefaultEncodingOpcode("V_SUBREV_CO_U32", "ENC_VOP2");
  const auto v_subbrev_co_u32_opcode =
      FindDefaultEncodingOpcode("V_SUBBREV_CO_U32", "ENC_VOP2");
  if (!Expect(v_add_co_u32_opcode.has_value(),
              "expected V_ADD_CO_U32 opcode lookup") ||
      !Expect(v_addc_co_u32_opcode.has_value(),
              "expected V_ADDC_CO_U32 opcode lookup") ||
      !Expect(v_sub_co_u32_opcode.has_value(),
              "expected V_SUB_CO_U32 opcode lookup") ||
      !Expect(v_subb_co_u32_opcode.has_value(),
              "expected V_SUBB_CO_U32 opcode lookup") ||
      !Expect(v_subrev_co_u32_opcode.has_value(),
              "expected V_SUBREV_CO_U32 opcode lookup") ||
      !Expect(v_subbrev_co_u32_opcode.has_value(),
              "expected V_SUBBREV_CO_U32 opcode lookup")) {
    return 1;
  }

  const auto vop2_add_co_u32_word =
      MakeVop2(*v_add_co_u32_opcode, 103, 51, 104);
  const std::vector<std::uint32_t> vector_add_co_program = {
      vop2_add_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_add_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_add_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_CO_U32",
              "expected V_ADD_CO_U32 decode") ||
      !Expect(decoded_program[0].operand_count == 4,
              "expected V_ADD_CO_U32 operand count") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 106,
              "expected V_ADD_CO_U32 implicit VCC destination decode")) {
    return 1;
  }

  WaveExecutionState vector_add_co_state;
  vector_add_co_state.exec_mask = 0b1011ULL;
  vector_add_co_state.vcc_mask = 0b0100ULL;
  vector_add_co_state.sgprs[51] = 0xffffffffu;
  vector_add_co_state.vgprs[104][0] = 1u;
  vector_add_co_state.vgprs[104][1] = 5u;
  vector_add_co_state.vgprs[104][3] = 0x7fffffffu;
  vector_add_co_state.vgprs[103][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_add_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_add_co_state.halted,
              "expected decoded v_add_co_u32 program to halt") ||
      !Expect(vector_add_co_state.vgprs[103][0] == 0u,
              "expected decoded v_add_co_u32 lane 0 result") ||
      !Expect(vector_add_co_state.vgprs[103][1] == 4u,
              "expected decoded v_add_co_u32 lane 1 result") ||
      !Expect(vector_add_co_state.vgprs[103][2] == 0xdeadbeefu,
              "expected inactive decoded v_add_co_u32 result") ||
      !Expect(vector_add_co_state.vgprs[103][3] == 0x7ffffffeu,
              "expected decoded v_add_co_u32 lane 3 result") ||
      !Expect(vector_add_co_state.sgprs[106] == 0x0000000fu &&
                  vector_add_co_state.sgprs[107] == 0u,
              "expected decoded v_add_co_u32 carry mask") ||
      !Expect(vector_add_co_state.vcc_mask == 0x000000000000000fULL,
              "expected decoded v_add_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop2_sub_co_u32_word =
      MakeVop2(*v_sub_co_u32_opcode, 105, 52, 106);
  const std::vector<std::uint32_t> vector_sub_co_program = {
      vop2_sub_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_sub_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_sub_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_SUB_CO_U32",
              "expected V_SUB_CO_U32 decode")) {
    return 1;
  }

  WaveExecutionState vector_sub_co_state;
  vector_sub_co_state.exec_mask = 0b1011ULL;
  vector_sub_co_state.vcc_mask = 0b0100ULL;
  vector_sub_co_state.sgprs[52] = 0u;
  vector_sub_co_state.vgprs[106][0] = 1u;
  vector_sub_co_state.vgprs[106][1] = 0u;
  vector_sub_co_state.vgprs[106][3] = 0xffffffffu;
  vector_sub_co_state.vgprs[105][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_sub_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_sub_co_state.halted,
              "expected decoded v_sub_co_u32 program to halt") ||
      !Expect(vector_sub_co_state.vgprs[105][0] == 0xffffffffu,
              "expected decoded v_sub_co_u32 lane 0 result") ||
      !Expect(vector_sub_co_state.vgprs[105][1] == 0u,
              "expected decoded v_sub_co_u32 lane 1 result") ||
      !Expect(vector_sub_co_state.vgprs[105][2] == 0xdeadbeefu,
              "expected inactive decoded v_sub_co_u32 result") ||
      !Expect(vector_sub_co_state.vgprs[105][3] == 1u,
              "expected decoded v_sub_co_u32 lane 3 result") ||
      !Expect(vector_sub_co_state.sgprs[106] == 0x0000000du &&
                  vector_sub_co_state.sgprs[107] == 0u,
              "expected decoded v_sub_co_u32 carry mask") ||
      !Expect(vector_sub_co_state.vcc_mask == 0x000000000000000dULL,
              "expected decoded v_sub_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop2_subrev_co_u32_word =
      MakeVop2(*v_subrev_co_u32_opcode, 107, 53, 108);
  const std::vector<std::uint32_t> vector_subrev_co_program = {
      vop2_subrev_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subrev_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subrev_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBREV_CO_U32",
              "expected V_SUBREV_CO_U32 decode")) {
    return 1;
  }

  WaveExecutionState vector_subrev_co_state;
  vector_subrev_co_state.exec_mask = 0b1011ULL;
  vector_subrev_co_state.vcc_mask = 0b0100ULL;
  vector_subrev_co_state.sgprs[53] = 1u;
  vector_subrev_co_state.vgprs[108][0] = 0u;
  vector_subrev_co_state.vgprs[108][1] = 1u;
  vector_subrev_co_state.vgprs[108][3] = 5u;
  vector_subrev_co_state.vgprs[107][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_subrev_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subrev_co_state.halted,
              "expected decoded v_subrev_co_u32 program to halt") ||
      !Expect(vector_subrev_co_state.vgprs[107][0] == 0xffffffffu,
              "expected decoded v_subrev_co_u32 lane 0 result") ||
      !Expect(vector_subrev_co_state.vgprs[107][1] == 0u,
              "expected decoded v_subrev_co_u32 lane 1 result") ||
      !Expect(vector_subrev_co_state.vgprs[107][2] == 0xdeadbeefu,
              "expected inactive decoded v_subrev_co_u32 result") ||
      !Expect(vector_subrev_co_state.vgprs[107][3] == 4u,
              "expected decoded v_subrev_co_u32 lane 3 result") ||
      !Expect(vector_subrev_co_state.sgprs[106] == 0x00000005u &&
                  vector_subrev_co_state.sgprs[107] == 0u,
              "expected decoded v_subrev_co_u32 carry mask") ||
      !Expect(vector_subrev_co_state.vcc_mask == 0x0000000000000005ULL,
              "expected decoded v_subrev_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop2_addc_co_u32_word =
      MakeVop2(*v_addc_co_u32_opcode, 112, 54, 109);
  const std::vector<std::uint32_t> vector_addc_co_program = {
      vop2_addc_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_addc_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_addc_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_ADDC_CO_U32",
              "expected V_ADDC_CO_U32 decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_ADDC_CO_U32 operand count") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 106,
              "expected V_ADDC_CO_U32 implicit VCC destination decode") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_ADDC_CO_U32 implicit VCC input decode")) {
    return 1;
  }

  WaveExecutionState vector_addc_co_state;
  vector_addc_co_state.exec_mask = 0b1011ULL;
  vector_addc_co_state.vcc_mask = 0b0101ULL;
  vector_addc_co_state.sgprs[54] = 0xfffffffeu;
  vector_addc_co_state.sgprs[106] = 0x00000005u;
  vector_addc_co_state.sgprs[107] = 0u;
  vector_addc_co_state.vgprs[109][0] = 1u;
  vector_addc_co_state.vgprs[109][1] = 1u;
  vector_addc_co_state.vgprs[109][3] = 2u;
  vector_addc_co_state.vgprs[112][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_addc_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_addc_co_state.halted,
              "expected decoded v_addc_co_u32 program to halt") ||
      !Expect(vector_addc_co_state.vgprs[112][0] == 0u,
              "expected decoded v_addc_co_u32 lane 0 result") ||
      !Expect(vector_addc_co_state.vgprs[112][1] == 0xffffffffu,
              "expected decoded v_addc_co_u32 lane 1 result") ||
      !Expect(vector_addc_co_state.vgprs[112][2] == 0xdeadbeefu,
              "expected inactive decoded v_addc_co_u32 result") ||
      !Expect(vector_addc_co_state.vgprs[112][3] == 0u,
              "expected decoded v_addc_co_u32 lane 3 result") ||
      !Expect(vector_addc_co_state.sgprs[106] == 0x0000000du &&
                  vector_addc_co_state.sgprs[107] == 0u,
              "expected decoded v_addc_co_u32 carry mask") ||
      !Expect(vector_addc_co_state.vcc_mask == 0x000000000000000dULL,
              "expected decoded v_addc_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop2_subb_co_u32_word =
      MakeVop2(*v_subb_co_u32_opcode, 113, 55, 110);
  const std::vector<std::uint32_t> vector_subb_co_program = {
      vop2_subb_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subb_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subb_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBB_CO_U32",
              "expected V_SUBB_CO_U32 decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_SUBB_CO_U32 operand count") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_SUBB_CO_U32 implicit VCC input decode")) {
    return 1;
  }

  WaveExecutionState vector_subb_co_state;
  vector_subb_co_state.exec_mask = 0b1011ULL;
  vector_subb_co_state.vcc_mask = 0b1110ULL;
  vector_subb_co_state.sgprs[55] = 1u;
  vector_subb_co_state.sgprs[106] = 0x0000000eu;
  vector_subb_co_state.sgprs[107] = 0u;
  vector_subb_co_state.vgprs[110][0] = 1u;
  vector_subb_co_state.vgprs[110][1] = 0u;
  vector_subb_co_state.vgprs[110][3] = 1u;
  vector_subb_co_state.vgprs[113][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_subb_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subb_co_state.halted,
              "expected decoded v_subb_co_u32 program to halt") ||
      !Expect(vector_subb_co_state.vgprs[113][0] == 0u,
              "expected decoded v_subb_co_u32 lane 0 result") ||
      !Expect(vector_subb_co_state.vgprs[113][1] == 0u,
              "expected decoded v_subb_co_u32 lane 1 result") ||
      !Expect(vector_subb_co_state.vgprs[113][2] == 0xdeadbeefu,
              "expected inactive decoded v_subb_co_u32 result") ||
      !Expect(vector_subb_co_state.vgprs[113][3] == 0xffffffffu,
              "expected decoded v_subb_co_u32 lane 3 result") ||
      !Expect(vector_subb_co_state.sgprs[106] == 0x0000000cu &&
                  vector_subb_co_state.sgprs[107] == 0u,
              "expected decoded v_subb_co_u32 carry mask") ||
      !Expect(vector_subb_co_state.vcc_mask == 0x000000000000000cULL,
              "expected decoded v_subb_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop2_subbrev_co_u32_word =
      MakeVop2(*v_subbrev_co_u32_opcode, 114, 56, 111);
  const std::vector<std::uint32_t> vector_subbrev_co_program = {
      vop2_subbrev_co_u32_word, MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subbrev_co_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subbrev_co_u32 program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBBREV_CO_U32",
              "expected V_SUBBREV_CO_U32 decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_SUBBREV_CO_U32 operand count") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_SUBBREV_CO_U32 implicit VCC input decode")) {
    return 1;
  }

  WaveExecutionState vector_subbrev_co_state;
  vector_subbrev_co_state.exec_mask = 0b1011ULL;
  vector_subbrev_co_state.vcc_mask = 0b0110ULL;
  vector_subbrev_co_state.sgprs[56] = 1u;
  vector_subbrev_co_state.sgprs[106] = 0x00000006u;
  vector_subbrev_co_state.sgprs[107] = 0u;
  vector_subbrev_co_state.vgprs[111][0] = 1u;
  vector_subbrev_co_state.vgprs[111][1] = 2u;
  vector_subbrev_co_state.vgprs[111][3] = 0u;
  vector_subbrev_co_state.vgprs[114][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_subbrev_co_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subbrev_co_state.halted,
              "expected decoded v_subbrev_co_u32 program to halt") ||
      !Expect(vector_subbrev_co_state.vgprs[114][0] == 0u,
              "expected decoded v_subbrev_co_u32 lane 0 result") ||
      !Expect(vector_subbrev_co_state.vgprs[114][1] == 0u,
              "expected decoded v_subbrev_co_u32 lane 1 result") ||
      !Expect(vector_subbrev_co_state.vgprs[114][2] == 0xdeadbeefu,
              "expected inactive decoded v_subbrev_co_u32 result") ||
      !Expect(vector_subbrev_co_state.vgprs[114][3] == 0xffffffffu,
              "expected decoded v_subbrev_co_u32 lane 3 result") ||
      !Expect(vector_subbrev_co_state.sgprs[106] == 0x0000000cu &&
                  vector_subbrev_co_state.sgprs[107] == 0u,
              "expected decoded v_subbrev_co_u32 carry mask") ||
      !Expect(vector_subbrev_co_state.vcc_mask == 0x000000000000000cULL,
              "expected decoded v_subbrev_co_u32 vcc mask")) {
    return 1;
  }

  const auto v_add_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_ADD_CO_U32", "VOP3_SDST_ENC");
  const auto v_addc_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_ADDC_CO_U32", "VOP3_SDST_ENC");
  const auto v_sub_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_SUB_CO_U32", "VOP3_SDST_ENC");
  const auto v_subb_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_SUBB_CO_U32", "VOP3_SDST_ENC");
  const auto v_subrev_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_SUBREV_CO_U32", "VOP3_SDST_ENC");
  const auto v_subbrev_co_u32_vop3_sdst_opcode =
      FindDefaultEncodingOpcode("V_SUBBREV_CO_U32", "VOP3_SDST_ENC");
  if (!Expect(v_add_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_ADD_CO_U32 VOP3_SDST opcode lookup") ||
      !Expect(v_addc_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_ADDC_CO_U32 VOP3_SDST opcode lookup") ||
      !Expect(v_sub_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_SUB_CO_U32 VOP3_SDST opcode lookup") ||
      !Expect(v_subb_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_SUBB_CO_U32 VOP3_SDST opcode lookup") ||
      !Expect(v_subrev_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_SUBREV_CO_U32 VOP3_SDST opcode lookup") ||
      !Expect(v_subbrev_co_u32_vop3_sdst_opcode.has_value(),
              "expected V_SUBBREV_CO_U32 VOP3_SDST opcode lookup")) {
    return 1;
  }

  const auto vop3_sdst_add_co_u32_word =
      MakeVop3Sdst(*v_add_co_u32_vop3_sdst_opcode, 115, 106, 57, 383);
  const std::vector<std::uint32_t> vector_add_co_vop3_sdst_program = {
      vop3_sdst_add_co_u32_word[0],
      vop3_sdst_add_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_add_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_add_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_ADD_CO_U32",
              "expected V_ADD_CO_U32 VOP3_SDST decode") ||
      !Expect(decoded_program[0].operand_count == 4,
              "expected V_ADD_CO_U32 VOP3_SDST operand count") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 106,
              "expected V_ADD_CO_U32 VOP3_SDST scalar pair destination")) {
    return 1;
  }

  WaveExecutionState vector_add_co_vop3_sdst_state;
  vector_add_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_add_co_vop3_sdst_state.vcc_mask = 0b0100ULL;
  vector_add_co_vop3_sdst_state.sgprs[57] = 0xffffffffu;
  vector_add_co_vop3_sdst_state.vgprs[127][0] = 1u;
  vector_add_co_vop3_sdst_state.vgprs[127][1] = 5u;
  vector_add_co_vop3_sdst_state.vgprs[127][3] = 0x7fffffffu;
  vector_add_co_vop3_sdst_state.vgprs[115][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_add_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_add_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_add_co_u32 program to halt") ||
      !Expect(vector_add_co_vop3_sdst_state.vgprs[115][0] == 0u,
              "expected decoded VOP3_SDST v_add_co_u32 lane 0 result") ||
      !Expect(vector_add_co_vop3_sdst_state.vgprs[115][1] == 4u,
              "expected decoded VOP3_SDST v_add_co_u32 lane 1 result") ||
      !Expect(vector_add_co_vop3_sdst_state.vgprs[115][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_add_co_u32 result") ||
      !Expect(vector_add_co_vop3_sdst_state.vgprs[115][3] == 0x7ffffffeu,
              "expected decoded VOP3_SDST v_add_co_u32 lane 3 result") ||
      !Expect(vector_add_co_vop3_sdst_state.sgprs[106] == 0x0000000fu &&
                  vector_add_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_add_co_u32 carry mask") ||
      !Expect(vector_add_co_vop3_sdst_state.vcc_mask == 0x000000000000000fULL,
              "expected decoded VOP3_SDST v_add_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop3_sdst_sub_co_u32_word =
      MakeVop3Sdst(*v_sub_co_u32_vop3_sdst_opcode, 116, 106, 58, 377);
  const std::vector<std::uint32_t> vector_sub_co_vop3_sdst_program = {
      vop3_sdst_sub_co_u32_word[0],
      vop3_sdst_sub_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_sub_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_sub_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_SUB_CO_U32",
              "expected V_SUB_CO_U32 VOP3_SDST decode")) {
    return 1;
  }

  WaveExecutionState vector_sub_co_vop3_sdst_state;
  vector_sub_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_sub_co_vop3_sdst_state.vcc_mask = 0b0100ULL;
  vector_sub_co_vop3_sdst_state.sgprs[58] = 0u;
  vector_sub_co_vop3_sdst_state.vgprs[121][0] = 1u;
  vector_sub_co_vop3_sdst_state.vgprs[121][1] = 0u;
  vector_sub_co_vop3_sdst_state.vgprs[121][3] = 0xffffffffu;
  vector_sub_co_vop3_sdst_state.vgprs[116][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_sub_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_sub_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_sub_co_u32 program to halt") ||
      !Expect(vector_sub_co_vop3_sdst_state.vgprs[116][0] == 0xffffffffu,
              "expected decoded VOP3_SDST v_sub_co_u32 lane 0 result") ||
      !Expect(vector_sub_co_vop3_sdst_state.vgprs[116][1] == 0u,
              "expected decoded VOP3_SDST v_sub_co_u32 lane 1 result") ||
      !Expect(vector_sub_co_vop3_sdst_state.vgprs[116][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_sub_co_u32 result") ||
      !Expect(vector_sub_co_vop3_sdst_state.vgprs[116][3] == 1u,
              "expected decoded VOP3_SDST v_sub_co_u32 lane 3 result") ||
      !Expect(vector_sub_co_vop3_sdst_state.sgprs[106] == 0x0000000du &&
                  vector_sub_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_sub_co_u32 carry mask") ||
      !Expect(vector_sub_co_vop3_sdst_state.vcc_mask == 0x000000000000000dULL,
              "expected decoded VOP3_SDST v_sub_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop3_sdst_subrev_co_u32_word =
      MakeVop3Sdst(*v_subrev_co_u32_vop3_sdst_opcode, 117, 106, 59, 378);
  const std::vector<std::uint32_t> vector_subrev_co_vop3_sdst_program = {
      vop3_sdst_subrev_co_u32_word[0],
      vop3_sdst_subrev_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subrev_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subrev_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBREV_CO_U32",
              "expected V_SUBREV_CO_U32 VOP3_SDST decode")) {
    return 1;
  }

  WaveExecutionState vector_subrev_co_vop3_sdst_state;
  vector_subrev_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_subrev_co_vop3_sdst_state.vcc_mask = 0b0100ULL;
  vector_subrev_co_vop3_sdst_state.sgprs[59] = 1u;
  vector_subrev_co_vop3_sdst_state.vgprs[122][0] = 0u;
  vector_subrev_co_vop3_sdst_state.vgprs[122][1] = 1u;
  vector_subrev_co_vop3_sdst_state.vgprs[122][3] = 5u;
  vector_subrev_co_vop3_sdst_state.vgprs[117][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_subrev_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subrev_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_subrev_co_u32 program to halt") ||
      !Expect(vector_subrev_co_vop3_sdst_state.vgprs[117][0] == 0xffffffffu,
              "expected decoded VOP3_SDST v_subrev_co_u32 lane 0 result") ||
      !Expect(vector_subrev_co_vop3_sdst_state.vgprs[117][1] == 0u,
              "expected decoded VOP3_SDST v_subrev_co_u32 lane 1 result") ||
      !Expect(vector_subrev_co_vop3_sdst_state.vgprs[117][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_subrev_co_u32 result") ||
      !Expect(vector_subrev_co_vop3_sdst_state.vgprs[117][3] == 4u,
              "expected decoded VOP3_SDST v_subrev_co_u32 lane 3 result") ||
      !Expect(vector_subrev_co_vop3_sdst_state.sgprs[106] == 0x00000005u &&
                  vector_subrev_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_subrev_co_u32 carry mask") ||
      !Expect(vector_subrev_co_vop3_sdst_state.vcc_mask == 0x0000000000000005ULL,
              "expected decoded VOP3_SDST v_subrev_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop3_sdst_addc_co_u32_word =
      MakeVop3Sdst(*v_addc_co_u32_vop3_sdst_opcode, 118, 106, 60, 379, 106);
  const std::vector<std::uint32_t> vector_addc_co_vop3_sdst_program = {
      vop3_sdst_addc_co_u32_word[0],
      vop3_sdst_addc_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_addc_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_addc_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_ADDC_CO_U32",
              "expected V_ADDC_CO_U32 VOP3_SDST decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_ADDC_CO_U32 VOP3_SDST operand count") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 106,
              "expected V_ADDC_CO_U32 VOP3_SDST scalar pair destination") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_ADDC_CO_U32 VOP3_SDST carry input decode")) {
    return 1;
  }

  WaveExecutionState vector_addc_co_vop3_sdst_state;
  vector_addc_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_addc_co_vop3_sdst_state.vcc_mask = 0b0101ULL;
  vector_addc_co_vop3_sdst_state.sgprs[60] = 0xfffffffeu;
  vector_addc_co_vop3_sdst_state.sgprs[106] = 0x00000005u;
  vector_addc_co_vop3_sdst_state.sgprs[107] = 0u;
  vector_addc_co_vop3_sdst_state.vgprs[123][0] = 1u;
  vector_addc_co_vop3_sdst_state.vgprs[123][1] = 1u;
  vector_addc_co_vop3_sdst_state.vgprs[123][3] = 2u;
  vector_addc_co_vop3_sdst_state.vgprs[118][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_addc_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_addc_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_addc_co_u32 program to halt") ||
      !Expect(vector_addc_co_vop3_sdst_state.vgprs[118][0] == 0u,
              "expected decoded VOP3_SDST v_addc_co_u32 lane 0 result") ||
      !Expect(vector_addc_co_vop3_sdst_state.vgprs[118][1] == 0xffffffffu,
              "expected decoded VOP3_SDST v_addc_co_u32 lane 1 result") ||
      !Expect(vector_addc_co_vop3_sdst_state.vgprs[118][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_addc_co_u32 result") ||
      !Expect(vector_addc_co_vop3_sdst_state.vgprs[118][3] == 0u,
              "expected decoded VOP3_SDST v_addc_co_u32 lane 3 result") ||
      !Expect(vector_addc_co_vop3_sdst_state.sgprs[106] == 0x0000000du &&
                  vector_addc_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_addc_co_u32 carry mask") ||
      !Expect(vector_addc_co_vop3_sdst_state.vcc_mask == 0x000000000000000dULL,
              "expected decoded VOP3_SDST v_addc_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop3_sdst_subb_co_u32_word =
      MakeVop3Sdst(*v_subb_co_u32_vop3_sdst_opcode, 119, 106, 61, 380, 106);
  const std::vector<std::uint32_t> vector_subb_co_vop3_sdst_program = {
      vop3_sdst_subb_co_u32_word[0],
      vop3_sdst_subb_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subb_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subb_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBB_CO_U32",
              "expected V_SUBB_CO_U32 VOP3_SDST decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_SUBB_CO_U32 VOP3_SDST operand count") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_SUBB_CO_U32 VOP3_SDST carry input decode")) {
    return 1;
  }

  WaveExecutionState vector_subb_co_vop3_sdst_state;
  vector_subb_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_subb_co_vop3_sdst_state.vcc_mask = 0b1110ULL;
  vector_subb_co_vop3_sdst_state.sgprs[61] = 1u;
  vector_subb_co_vop3_sdst_state.sgprs[106] = 0x0000000eu;
  vector_subb_co_vop3_sdst_state.sgprs[107] = 0u;
  vector_subb_co_vop3_sdst_state.vgprs[124][0] = 1u;
  vector_subb_co_vop3_sdst_state.vgprs[124][1] = 0u;
  vector_subb_co_vop3_sdst_state.vgprs[124][3] = 1u;
  vector_subb_co_vop3_sdst_state.vgprs[119][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_subb_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subb_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_subb_co_u32 program to halt") ||
      !Expect(vector_subb_co_vop3_sdst_state.vgprs[119][0] == 0u,
              "expected decoded VOP3_SDST v_subb_co_u32 lane 0 result") ||
      !Expect(vector_subb_co_vop3_sdst_state.vgprs[119][1] == 0u,
              "expected decoded VOP3_SDST v_subb_co_u32 lane 1 result") ||
      !Expect(vector_subb_co_vop3_sdst_state.vgprs[119][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_subb_co_u32 result") ||
      !Expect(vector_subb_co_vop3_sdst_state.vgprs[119][3] == 0xffffffffu,
              "expected decoded VOP3_SDST v_subb_co_u32 lane 3 result") ||
      !Expect(vector_subb_co_vop3_sdst_state.sgprs[106] == 0x0000000cu &&
                  vector_subb_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_subb_co_u32 carry mask") ||
      !Expect(vector_subb_co_vop3_sdst_state.vcc_mask == 0x000000000000000cULL,
              "expected decoded VOP3_SDST v_subb_co_u32 vcc mask")) {
    return 1;
  }

  const auto vop3_sdst_subbrev_co_u32_word =
      MakeVop3Sdst(*v_subbrev_co_u32_vop3_sdst_opcode, 120, 106, 62, 381, 106);
  const std::vector<std::uint32_t> vector_subbrev_co_vop3_sdst_program = {
      vop3_sdst_subbrev_co_u32_word[0],
      vop3_sdst_subbrev_co_u32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_subbrev_co_vop3_sdst_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded v_subbrev_co_u32 VOP3_SDST program size") ||
      !Expect(decoded_program[0].opcode == "V_SUBBREV_CO_U32",
              "expected V_SUBBREV_CO_U32 VOP3_SDST decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected V_SUBBREV_CO_U32 VOP3_SDST operand count") ||
      !Expect(decoded_program[0].operands[4].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[4].index == 106,
              "expected V_SUBBREV_CO_U32 VOP3_SDST carry input decode")) {
    return 1;
  }

  WaveExecutionState vector_subbrev_co_vop3_sdst_state;
  vector_subbrev_co_vop3_sdst_state.exec_mask = 0b1011ULL;
  vector_subbrev_co_vop3_sdst_state.vcc_mask = 0b0110ULL;
  vector_subbrev_co_vop3_sdst_state.sgprs[62] = 1u;
  vector_subbrev_co_vop3_sdst_state.sgprs[106] = 0x00000006u;
  vector_subbrev_co_vop3_sdst_state.sgprs[107] = 0u;
  vector_subbrev_co_vop3_sdst_state.vgprs[125][0] = 1u;
  vector_subbrev_co_vop3_sdst_state.vgprs[125][1] = 2u;
  vector_subbrev_co_vop3_sdst_state.vgprs[125][3] = 0u;
  vector_subbrev_co_vop3_sdst_state.vgprs[120][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_subbrev_co_vop3_sdst_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_subbrev_co_vop3_sdst_state.halted,
              "expected decoded VOP3_SDST v_subbrev_co_u32 program to halt") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.vgprs[120][0] == 0u,
              "expected decoded VOP3_SDST v_subbrev_co_u32 lane 0 result") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.vgprs[120][1] == 0u,
              "expected decoded VOP3_SDST v_subbrev_co_u32 lane 1 result") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.vgprs[120][2] == 0xdeadbeefu,
              "expected inactive decoded VOP3_SDST v_subbrev_co_u32 result") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.vgprs[120][3] == 0xffffffffu,
              "expected decoded VOP3_SDST v_subbrev_co_u32 lane 3 result") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.sgprs[106] == 0x0000000cu &&
                  vector_subbrev_co_vop3_sdst_state.sgprs[107] == 0u,
              "expected decoded VOP3_SDST v_subbrev_co_u32 carry mask") ||
      !Expect(vector_subbrev_co_vop3_sdst_state.vcc_mask == 0x000000000000000cULL,
              "expected decoded VOP3_SDST v_subbrev_co_u32 vcc mask")) {
    return 1;
  }

  const auto v_cndmask_b32_vop3_opcode =
      FindDefaultEncodingOpcode("V_CNDMASK_B32", "ENC_VOP3");
  if (!Expect(v_cndmask_b32_vop3_opcode.has_value(),
              "expected V_CNDMASK_B32 VOP3 opcode lookup")) {
    return 1;
  }

  const auto vop3_cndmask_b32_word =
      MakeVop3(*v_cndmask_b32_vop3_opcode, 121, 106, 382, 63);
  const std::vector<std::uint32_t> promoted_vop3_cndmask_program = {
      vop3_cndmask_b32_word[0],
      vop3_cndmask_b32_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(promoted_vop3_cndmask_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 2,
              "expected decoded promoted V_CNDMASK_B32 program size") ||
      !Expect(decoded_program[0].opcode == "V_CNDMASK_B32",
              "expected promoted V_CNDMASK_B32 decode") ||
      !Expect(decoded_program[0].operand_count == 3,
              "expected promoted V_CNDMASK_B32 operand count") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 63,
              "expected promoted V_CNDMASK_B32 false-value decode") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr &&
                  decoded_program[0].operands[2].index == 126,
              "expected promoted V_CNDMASK_B32 true-value decode")) {
    return 1;
  }

  WaveExecutionState promoted_vop3_cndmask_state;
  promoted_vop3_cndmask_state.exec_mask = 0b1011ULL;
  promoted_vop3_cndmask_state.vcc_mask = 0b1001ULL;
  promoted_vop3_cndmask_state.sgprs[63] = 0x13572468u;
  promoted_vop3_cndmask_state.sgprs[106] = 0x00000009u;
  promoted_vop3_cndmask_state.sgprs[107] = 0u;
  promoted_vop3_cndmask_state.vgprs[126][0] = 0xaaaabbbbu;
  promoted_vop3_cndmask_state.vgprs[126][1] = 0xccccddddu;
  promoted_vop3_cndmask_state.vgprs[126][3] = 0xeeeeffffu;
  promoted_vop3_cndmask_state.vgprs[121][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &promoted_vop3_cndmask_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(promoted_vop3_cndmask_state.halted,
              "expected promoted V_CNDMASK_B32 program to halt") ||
      !Expect(promoted_vop3_cndmask_state.vgprs[121][0] == 0xaaaabbbbu,
              "expected promoted V_CNDMASK_B32 lane 0 result") ||
      !Expect(promoted_vop3_cndmask_state.vgprs[121][1] == 0x13572468u,
              "expected promoted V_CNDMASK_B32 lane 1 result") ||
      !Expect(promoted_vop3_cndmask_state.vgprs[121][2] == 0xdeadbeefu,
              "expected inactive promoted V_CNDMASK_B32 result") ||
      !Expect(promoted_vop3_cndmask_state.vgprs[121][3] == 0xeeeeffffu,
              "expected promoted V_CNDMASK_B32 lane 3 result")) {
    return 1;
  }

  const auto v_cmp_eq_u32_opcode =
      FindDefaultEncodingOpcode("V_CMP_EQ_U32", "ENC_VOP3");
  const auto v_cmp_lt_i32_opcode =
      FindDefaultEncodingOpcode("V_CMP_LT_I32", "ENC_VOP3");
  const auto v_cmp_ge_u32_opcode =
      FindDefaultEncodingOpcode("V_CMP_GE_U32", "ENC_VOP3");
  const auto v_cndmask_b32_opcode =
      FindDefaultEncodingOpcode("V_CNDMASK_B32", "ENC_VOP2");
  if (!Expect(v_cmp_eq_u32_opcode.has_value(),
              "expected V_CMP_EQ_U32 opcode lookup") ||
      !Expect(v_cmp_lt_i32_opcode.has_value(),
              "expected V_CMP_LT_I32 opcode lookup") ||
      !Expect(v_cmp_ge_u32_opcode.has_value(),
              "expected V_CMP_GE_U32 opcode lookup") ||
      !Expect(v_cndmask_b32_opcode.has_value(),
              "expected V_CNDMASK_B32 opcode lookup")) {
    return 1;
  }

  const auto vop3_cmp_eq_u32_word = MakeVop3(*v_cmp_eq_u32_opcode, 106, 12, 268);
  const auto vop3_cmp_lt_i32_word = MakeVop3(*v_cmp_lt_i32_opcode, 108, 13, 269);
  const auto vop3_cmp_ge_u32_word = MakeVop3(*v_cmp_ge_u32_opcode, 110, 14, 270);
  const auto vop2_cndmask_word = MakeVop2(*v_cndmask_b32_opcode, 35, 15, 15);
  const std::vector<std::uint32_t> vector_compare_program = {
      vop3_cmp_eq_u32_word[0], vop3_cmp_eq_u32_word[1],
      vop3_cmp_lt_i32_word[0], vop3_cmp_lt_i32_word[1],
      vop3_cmp_ge_u32_word[0], vop3_cmp_ge_u32_word[1],
      vop2_cndmask_word,
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(vector_compare_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded vector compare program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_EQ_U32",
              "expected V_CMP_EQ_U32 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_LT_I32",
              "expected V_CMP_LT_I32 decode") ||
      !Expect(decoded_program[2].opcode == "V_CMP_GE_U32",
              "expected V_CMP_GE_U32 decode") ||
      !Expect(decoded_program[3].opcode == "V_CNDMASK_B32",
              "expected V_CNDMASK_B32 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr,
              "expected vector compare scalar destination decode") ||
      !Expect(decoded_program[0].operands[0].index == 106,
              "expected vector compare destination pair index") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
              "expected vector compare vsrc1 decode") ||
      !Expect(decoded_program[0].operands[2].index == 12,
              "expected vector compare vsrc1 index") ||
      !Expect(decoded_program[3].operands[1].kind == OperandKind::kSgpr,
              "expected cndmask src0 SGPR decode") ||
      !Expect(decoded_program[3].operands[1].index == 15,
              "expected cndmask src0 SGPR index") ||
      !Expect(decoded_program[3].operands[2].kind == OperandKind::kVgpr,
              "expected cndmask vsrc1 decode") ||
      !Expect(decoded_program[3].operands[2].index == 15,
              "expected cndmask vsrc1 index")) {
    return 1;
  }

  WaveExecutionState vector_compare_state;
  vector_compare_state.exec_mask = 0b1011ULL;
  vector_compare_state.vcc_mask = 0b0100ULL;
  vector_compare_state.sgprs[12] = 7u;
  vector_compare_state.sgprs[13] = static_cast<std::uint32_t>(-2);
  vector_compare_state.sgprs[14] = 10u;
  vector_compare_state.sgprs[15] = 127u;
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_compare_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_state.halted,
              "expected vector compare decode program to halt") ||
      !Expect(vector_compare_state.sgprs[106] == 13u,
              "expected decoded v_cmp_eq_u32 low mask result") ||
      !Expect(vector_compare_state.sgprs[108] == 5u,
              "expected decoded v_cmp_lt_i32 low mask result") ||
      !Expect(vector_compare_state.sgprs[110] == 13u,
              "expected decoded v_cmp_ge_u32 low mask result") ||
      !Expect(vector_compare_state.vgprs[35][0] == 100u,
              "expected decoded v_cndmask_b32 lane 0 result") ||
      !Expect(vector_compare_state.vgprs[35][1] == 127u,
              "expected decoded v_cndmask_b32 lane 1 result") ||
      !Expect(vector_compare_state.vgprs[35][2] == 0xdeadbeefu,
              "expected decoded inactive v_cndmask_b32 result") ||
      !Expect(vector_compare_state.vgprs[35][3] == 300u,
              "expected decoded v_cndmask_b32 lane 3 result") ||
      !Expect(vector_compare_state.vcc_mask == 13u,
              "expected decoded final VCC mask result")) {
    return 1;
  }

  const auto v_cmp_eq_u32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_EQ_U32", "ENC_VOPC");
  const auto v_cmp_ne_u32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_NE_U32", "ENC_VOPC");
  if (!Expect(v_cmp_eq_u32_vopc_opcode.has_value(),
              "expected V_CMP_EQ_U32 VOPC opcode lookup") ||
      !Expect(v_cmp_ne_u32_vopc_opcode.has_value(),
              "expected V_CMP_NE_U32 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_compare_vopc_program = {
      MakeVopc(*v_cmp_eq_u32_vopc_opcode, 12, 12),      // v_cmp_eq_u32 vcc, s12, v12
      MakeVopc(*v_cmp_ne_u32_vopc_opcode, 255, 13), 9u,  // v_cmp_ne_u32 vcc, 9, v13
      MakeVop2(*v_cndmask_b32_opcode, 36, 15, 15),       // v_cndmask_b32 v36, s15, v15
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_compare_vopc_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded VOPC vector compare program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_EQ_U32",
              "expected VOPC V_CMP_EQ_U32 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_NE_U32",
              "expected VOPC V_CMP_NE_U32 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[0].index == 106,
              "expected VOPC compare implicit VCC destination decode") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 12,
              "expected VOPC compare SGPR src0 decode") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr &&
                  decoded_program[0].operands[2].index == 12,
              "expected VOPC compare VGPR src1 decode") ||
      !Expect(decoded_program[1].operands[1].kind == OperandKind::kImm32 &&
                  decoded_program[1].operands[1].imm32 == 9u,
              "expected VOPC literal src0 decode") ||
      !Expect(decoded_program[1].operands[2].kind == OperandKind::kVgpr &&
                  decoded_program[1].operands[2].index == 13,
              "expected VOPC literal compare VGPR src1 decode")) {
    return 1;
  }

  WaveExecutionState vector_compare_vopc_state;
  vector_compare_vopc_state.exec_mask = 0b1011ULL;
  vector_compare_vopc_state.sgprs[12] = 7u;
  vector_compare_vopc_state.sgprs[15] = 127u;
  vector_compare_vopc_state.vgprs[12][0] = 7u;
  vector_compare_vopc_state.vgprs[12][1] = 5u;
  vector_compare_vopc_state.vgprs[12][3] = 7u;
  vector_compare_vopc_state.vgprs[13][0] = 9u;
  vector_compare_vopc_state.vgprs[13][1] = 8u;
  vector_compare_vopc_state.vgprs[13][3] = 8u;
  vector_compare_vopc_state.vgprs[15][0] = 100u;
  vector_compare_vopc_state.vgprs[15][1] = 200u;
  vector_compare_vopc_state.vgprs[15][3] = 300u;
  vector_compare_vopc_state.vgprs[36][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_compare_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_vopc_state.halted,
              "expected VOPC vector compare program to halt") ||
      !Expect(vector_compare_vopc_state.sgprs[106] == 10u &&
                  vector_compare_vopc_state.sgprs[107] == 0u,
              "expected VOPC compare low VCC mask result") ||
      !Expect(vector_compare_vopc_state.vgprs[36][0] == 127u,
              "expected VOPC v_cndmask lane 0 result") ||
      !Expect(vector_compare_vopc_state.vgprs[36][1] == 200u,
              "expected VOPC v_cndmask lane 1 result") ||
      !Expect(vector_compare_vopc_state.vgprs[36][2] == 0xdeadbeefu,
              "expected inactive VOPC v_cndmask result") ||
      !Expect(vector_compare_vopc_state.vgprs[36][3] == 300u,
              "expected VOPC v_cndmask lane 3 result") ||
      !Expect(vector_compare_vopc_state.vcc_mask == 10u,
              "expected VOPC final VCC mask result")) {
    return 1;
  }

  const auto v_cmp_lt_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_LT_F32", "ENC_VOPC");
  const auto v_cmp_u_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_U_F32", "ENC_VOPC");
  const auto v_cmp_ngt_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_NGT_F32", "ENC_VOPC");
  if (!Expect(v_cmp_lt_f32_vopc_opcode.has_value(),
              "expected V_CMP_LT_F32 VOPC opcode lookup") ||
      !Expect(v_cmp_u_f32_vopc_opcode.has_value(),
              "expected V_CMP_U_F32 VOPC opcode lookup") ||
      !Expect(v_cmp_ngt_f32_vopc_opcode.has_value(),
              "expected V_CMP_NGT_F32 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_compare_f32_vopc_program = {
      MakeVopc(*v_cmp_lt_f32_vopc_opcode, 16, 16),      // v_cmp_lt_f32 vcc, s16, v16
      MakeVopc(*v_cmp_u_f32_vopc_opcode, 17, 17),       // v_cmp_u_f32 vcc, s17, v17
      MakeVopc(*v_cmp_ngt_f32_vopc_opcode, 18, 18),     // v_cmp_ngt_f32 vcc, s18, v18
      MakeVop2(*v_cndmask_b32_opcode, 37, 20, 20),      // v_cndmask_b32 v37, s20, v20
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_compare_f32_vopc_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded VOPC f32 compare program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_LT_F32",
              "expected VOPC V_CMP_LT_F32 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_U_F32",
              "expected VOPC V_CMP_U_F32 decode") ||
      !Expect(decoded_program[2].opcode == "V_CMP_NGT_F32",
              "expected VOPC V_CMP_NGT_F32 decode")) {
    return 1;
  }

  WaveExecutionState vector_compare_f32_vopc_state;
  vector_compare_f32_vopc_state.exec_mask = 0b1011ULL;
  vector_compare_f32_vopc_state.sgprs[16] = FloatBits(1.5f);
  vector_compare_f32_vopc_state.sgprs[17] = 0x7fc00000u;
  vector_compare_f32_vopc_state.sgprs[18] = FloatBits(2.0f);
  vector_compare_f32_vopc_state.sgprs[20] = 55u;
  vector_compare_f32_vopc_state.vgprs[16][0] = FloatBits(2.0f);
  vector_compare_f32_vopc_state.vgprs[16][1] = FloatBits(1.0f);
  vector_compare_f32_vopc_state.vgprs[16][3] = FloatBits(1.5f);
  vector_compare_f32_vopc_state.vgprs[17][0] = FloatBits(0.0f);
  vector_compare_f32_vopc_state.vgprs[17][1] = FloatBits(4.0f);
  vector_compare_f32_vopc_state.vgprs[17][3] = FloatBits(-1.0f);
  vector_compare_f32_vopc_state.vgprs[18][0] = FloatBits(1.0f);
  vector_compare_f32_vopc_state.vgprs[18][1] = FloatBits(2.0f);
  vector_compare_f32_vopc_state.vgprs[18][3] = 0x7fc00000u;
  vector_compare_f32_vopc_state.vgprs[20][0] = 100u;
  vector_compare_f32_vopc_state.vgprs[20][1] = 200u;
  vector_compare_f32_vopc_state.vgprs[20][3] = 300u;
  vector_compare_f32_vopc_state.vgprs[37][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_compare_f32_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_f32_vopc_state.halted,
              "expected VOPC f32 compare program to halt") ||
      !Expect(vector_compare_f32_vopc_state.sgprs[106] == 10u &&
                  vector_compare_f32_vopc_state.sgprs[107] == 0u,
              "expected VOPC f32 final low VCC mask result") ||
      !Expect(vector_compare_f32_vopc_state.vgprs[37][0] == 55u,
              "expected VOPC f32 v_cndmask lane 0 result") ||
      !Expect(vector_compare_f32_vopc_state.vgprs[37][1] == 200u,
              "expected VOPC f32 v_cndmask lane 1 result") ||
      !Expect(vector_compare_f32_vopc_state.vgprs[37][2] == 0xdeadbeefu,
              "expected inactive VOPC f32 v_cndmask result") ||
      !Expect(vector_compare_f32_vopc_state.vgprs[37][3] == 300u,
              "expected VOPC f32 v_cndmask lane 3 result") ||
      !Expect(vector_compare_f32_vopc_state.vcc_mask == 10u,
              "expected VOPC f32 final VCC mask result")) {
    return 1;
  }

  const auto v_cmp_lt_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_LT_F64", "ENC_VOPC");
  const auto v_cmp_u_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_U_F64", "ENC_VOPC");
  const auto v_cmp_ngt_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_NGT_F64", "ENC_VOPC");
  if (!Expect(v_cmp_lt_f64_vopc_opcode.has_value(),
              "expected V_CMP_LT_F64 VOPC opcode lookup") ||
      !Expect(v_cmp_u_f64_vopc_opcode.has_value(),
              "expected V_CMP_U_F64 VOPC opcode lookup") ||
      !Expect(v_cmp_ngt_f64_vopc_opcode.has_value(),
              "expected V_CMP_NGT_F64 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_compare_f64_vopc_program = {
      MakeVopc(*v_cmp_lt_f64_vopc_opcode, 70, 30),  // v_cmp_lt_f64 vcc, s[70:71], v[30:31]
      MakeVopc(*v_cmp_u_f64_vopc_opcode, 72, 32),   // v_cmp_u_f64 vcc, s[72:73], v[32:33]
      MakeVopc(*v_cmp_ngt_f64_vopc_opcode, 74, 34), // v_cmp_ngt_f64 vcc, s[74:75], v[34:35]
      MakeVop2(*v_cndmask_b32_opcode, 38, 20, 20),  // v_cndmask_b32 v38, s20, v20
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_compare_f64_vopc_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded VOPC f64 compare program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_LT_F64",
              "expected VOPC V_CMP_LT_F64 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_U_F64",
              "expected VOPC V_CMP_U_F64 decode") ||
      !Expect(decoded_program[2].opcode == "V_CMP_NGT_F64",
              "expected VOPC V_CMP_NGT_F64 decode") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr &&
                  decoded_program[0].operands[1].index == 70,
              "expected VOPC f64 sgpr pair decode") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr &&
                  decoded_program[0].operands[2].index == 30,
              "expected VOPC f64 vgpr pair decode")) {
    return 1;
  }

  constexpr std::uint64_t kQuietNan64 = 0x7ff8000000000000ULL;
  WaveExecutionState vector_compare_f64_vopc_state;
  vector_compare_f64_vopc_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &vector_compare_f64_vopc_state.sgprs[70],
           &vector_compare_f64_vopc_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_compare_f64_vopc_state.sgprs[72],
           &vector_compare_f64_vopc_state.sgprs[73]);
  SplitU64(DoubleBits(2.0), &vector_compare_f64_vopc_state.sgprs[74],
           &vector_compare_f64_vopc_state.sgprs[75]);
  vector_compare_f64_vopc_state.sgprs[20] = 55u;
  SplitU64(DoubleBits(2.0), &vector_compare_f64_vopc_state.vgprs[30][0],
           &vector_compare_f64_vopc_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &vector_compare_f64_vopc_state.vgprs[30][1],
           &vector_compare_f64_vopc_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &vector_compare_f64_vopc_state.vgprs[30][3],
           &vector_compare_f64_vopc_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &vector_compare_f64_vopc_state.vgprs[32][0],
           &vector_compare_f64_vopc_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &vector_compare_f64_vopc_state.vgprs[32][1],
           &vector_compare_f64_vopc_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &vector_compare_f64_vopc_state.vgprs[32][3],
           &vector_compare_f64_vopc_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &vector_compare_f64_vopc_state.vgprs[34][0],
           &vector_compare_f64_vopc_state.vgprs[35][0]);
  SplitU64(DoubleBits(2.0), &vector_compare_f64_vopc_state.vgprs[34][1],
           &vector_compare_f64_vopc_state.vgprs[35][1]);
  SplitU64(kQuietNan64, &vector_compare_f64_vopc_state.vgprs[34][3],
           &vector_compare_f64_vopc_state.vgprs[35][3]);
  vector_compare_f64_vopc_state.vgprs[20][0] = 100u;
  vector_compare_f64_vopc_state.vgprs[20][1] = 200u;
  vector_compare_f64_vopc_state.vgprs[20][3] = 300u;
  vector_compare_f64_vopc_state.vgprs[38][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_compare_f64_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare_f64_vopc_state.halted,
              "expected VOPC f64 compare program to halt") ||
      !Expect(vector_compare_f64_vopc_state.sgprs[106] == 10u &&
                  vector_compare_f64_vopc_state.sgprs[107] == 0u,
              "expected VOPC f64 final low VCC mask result") ||
      !Expect(vector_compare_f64_vopc_state.vgprs[38][0] == 55u,
              "expected VOPC f64 v_cndmask lane 0 result") ||
      !Expect(vector_compare_f64_vopc_state.vgprs[38][1] == 200u,
              "expected VOPC f64 v_cndmask lane 1 result") ||
      !Expect(vector_compare_f64_vopc_state.vgprs[38][2] == 0xdeadbeefu,
              "expected inactive VOPC f64 v_cndmask result") ||
      !Expect(vector_compare_f64_vopc_state.vgprs[38][3] == 300u,
              "expected VOPC f64 v_cndmask lane 3 result") ||
      !Expect(vector_compare_f64_vopc_state.vcc_mask == 10u,
              "expected VOPC f64 final VCC mask result")) {
    return 1;
  }

  const auto v_cmp_class_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMP_CLASS_F32", "ENC_VOPC");
  if (!Expect(v_cmp_class_f32_vopc_opcode.has_value(),
              "expected V_CMP_CLASS_F32 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_class_f32_vopc_program = {
      MakeVopc(*v_cmp_class_f32_vopc_opcode, 16, 40),  // v_cmp_class_f32 vcc, s16, v40
      MakeVopc(*v_cmp_class_f32_vopc_opcode, 17, 41),  // v_cmp_class_f32 vcc, s17, v41
      MakeVop2(*v_cndmask_b32_opcode, 42, 20, 20),     // v_cndmask_b32 v42, s20, v20
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_class_f32_vopc_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded VOPC f32 class program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_CLASS_F32",
              "expected VOPC V_CMP_CLASS_F32 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_CLASS_F32",
              "expected second VOPC V_CMP_CLASS_F32 decode")) {
    return 1;
  }

  WaveExecutionState vector_class_f32_vopc_state;
  vector_class_f32_vopc_state.exec_mask = 0b1011ULL;
  vector_class_f32_vopc_state.sgprs[16] = FloatBits(-0.0f);
  vector_class_f32_vopc_state.sgprs[17] = 0x7fc00000u;
  vector_class_f32_vopc_state.sgprs[20] = 55u;
  vector_class_f32_vopc_state.vgprs[40][0] = 0x20u;
  vector_class_f32_vopc_state.vgprs[40][1] = 0x40u;
  vector_class_f32_vopc_state.vgprs[40][3] = 0x60u;
  vector_class_f32_vopc_state.vgprs[41][0] = 0x1u;
  vector_class_f32_vopc_state.vgprs[41][1] = 0x2u;
  vector_class_f32_vopc_state.vgprs[41][3] = 0x3u;
  vector_class_f32_vopc_state.vgprs[20][0] = 100u;
  vector_class_f32_vopc_state.vgprs[20][1] = 200u;
  vector_class_f32_vopc_state.vgprs[20][3] = 300u;
  vector_class_f32_vopc_state.vgprs[42][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_class_f32_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_class_f32_vopc_state.halted,
              "expected VOPC f32 class program to halt") ||
      !Expect(vector_class_f32_vopc_state.sgprs[106] == 10u &&
                  vector_class_f32_vopc_state.sgprs[107] == 0u,
              "expected VOPC f32 class final low VCC mask result") ||
      !Expect(vector_class_f32_vopc_state.vgprs[42][0] == 55u,
              "expected VOPC f32 class lane 0 cndmask result") ||
      !Expect(vector_class_f32_vopc_state.vgprs[42][1] == 200u,
              "expected VOPC f32 class lane 1 cndmask result") ||
      !Expect(vector_class_f32_vopc_state.vgprs[42][2] == 0xdeadbeefu,
              "expected VOPC f32 class inactive lane result") ||
      !Expect(vector_class_f32_vopc_state.vgprs[42][3] == 300u,
              "expected VOPC f32 class lane 3 cndmask result") ||
      !Expect(vector_class_f32_vopc_state.vcc_mask == 10u,
              "expected VOPC f32 class final VCC mask result")) {
    return 1;
  }

  const auto v_cmp_lt_i64_opcode =
      FindDefaultEncodingOpcode("V_CMP_LT_I64", "ENC_VOP3");
  const auto v_cmp_ge_u64_opcode =
      FindDefaultEncodingOpcode("V_CMP_GE_U64", "ENC_VOP3");
  const auto v_cmp_f_i64_opcode =
      FindDefaultEncodingOpcode("V_CMP_F_I64", "ENC_VOP3");
  const auto v_cmp_t_u64_opcode =
      FindDefaultEncodingOpcode("V_CMP_T_U64", "ENC_VOP3");
  if (!Expect(v_cmp_lt_i64_opcode.has_value(),
              "expected V_CMP_LT_I64 opcode lookup") ||
      !Expect(v_cmp_ge_u64_opcode.has_value(),
              "expected V_CMP_GE_U64 opcode lookup") ||
      !Expect(v_cmp_f_i64_opcode.has_value(),
              "expected V_CMP_F_I64 opcode lookup") ||
      !Expect(v_cmp_t_u64_opcode.has_value(),
              "expected V_CMP_T_U64 opcode lookup")) {
    return 1;
  }

  const auto vop3_cmp_lt_i64_word = MakeVop3(*v_cmp_lt_i64_opcode, 112, 60, 276);
  const auto vop3_cmp_ge_u64_word = MakeVop3(*v_cmp_ge_u64_opcode, 106, 62, 278);
  const auto vop3_cmp_f_i64_word = MakeVop3(*v_cmp_f_i64_opcode, 114, 64, 280);
  const auto vop3_cmp_t_u64_word = MakeVop3(*v_cmp_t_u64_opcode, 116, 66, 282);
  const std::vector<std::uint32_t> vector_compare64_program = {
      vop3_cmp_lt_i64_word[0], vop3_cmp_lt_i64_word[1],
      vop3_cmp_ge_u64_word[0], vop3_cmp_ge_u64_word[1],
      vop3_cmp_f_i64_word[0], vop3_cmp_f_i64_word[1],
      vop3_cmp_t_u64_word[0], vop3_cmp_t_u64_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_compare64_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded vector compare64 program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_LT_I64",
              "expected V_CMP_LT_I64 decode") ||
      !Expect(decoded_program[1].opcode == "V_CMP_GE_U64",
              "expected V_CMP_GE_U64 decode") ||
      !Expect(decoded_program[2].opcode == "V_CMP_F_I64",
              "expected V_CMP_F_I64 decode") ||
      !Expect(decoded_program[3].opcode == "V_CMP_T_U64",
              "expected V_CMP_T_U64 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr,
              "expected vector compare64 scalar destination decode") ||
      !Expect(decoded_program[0].operands[0].index == 112,
              "expected vector compare64 destination pair index") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr,
              "expected vector compare64 sgpr pair decode") ||
      !Expect(decoded_program[0].operands[1].index == 60,
              "expected vector compare64 sgpr pair index") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
              "expected vector compare64 vgpr pair decode") ||
      !Expect(decoded_program[0].operands[2].index == 20,
              "expected vector compare64 vgpr pair index")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_compare64_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_compare64_state.halted,
              "expected vector compare64 decode program to halt") ||
      !Expect(vector_compare64_state.sgprs[112] == 5u,
              "expected decoded v_cmp_lt_i64 low mask result") ||
      !Expect(vector_compare64_state.sgprs[106] == 14u,
              "expected decoded v_cmp_ge_u64 low mask result") ||
      !Expect(vector_compare64_state.sgprs[114] == 4u,
              "expected decoded v_cmp_f_i64 low mask result") ||
      !Expect(vector_compare64_state.sgprs[116] == 15u,
              "expected decoded v_cmp_t_u64 low mask result") ||
      !Expect(vector_compare64_state.vcc_mask == 15u,
              "expected decoded final VCC mask result for compare64")) {
    return 1;
  }

  const auto v_cmpx_eq_u32_opcode =
      FindDefaultEncodingOpcode("V_CMPX_EQ_U32", "ENC_VOP3");
  const auto v_cmpx_lt_i64_opcode =
      FindDefaultEncodingOpcode("V_CMPX_LT_I64", "ENC_VOP3");
  const auto v_cmpx_gt_u64_opcode =
      FindDefaultEncodingOpcode("V_CMPX_GT_U64", "ENC_VOP3");
  const auto s_cbranch_execz_cmpx_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_EXECZ", "ENC_SOPP");
  const auto s_cbranch_execnz_cmpx_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_EXECNZ", "ENC_SOPP");
  if (!Expect(v_cmpx_eq_u32_opcode.has_value(),
              "expected V_CMPX_EQ_U32 opcode lookup") ||
      !Expect(v_cmpx_lt_i64_opcode.has_value(),
              "expected V_CMPX_LT_I64 opcode lookup") ||
      !Expect(v_cmpx_gt_u64_opcode.has_value(),
              "expected V_CMPX_GT_U64 opcode lookup") ||
      !Expect(s_cbranch_execz_cmpx_opcode.has_value(),
              "expected S_CBRANCH_EXECZ opcode lookup for cmpx test") ||
      !Expect(s_cbranch_execnz_cmpx_opcode.has_value(),
              "expected S_CBRANCH_EXECNZ opcode lookup for cmpx test")) {
    return 1;
  }

  const auto vop3_cmpx_eq_u32_word =
      MakeVop3(*v_cmpx_eq_u32_opcode, 106, 12, 268);
  const auto vop3_cmpx_lt_i64_word =
      MakeVop3(*v_cmpx_lt_i64_opcode, 108, 60, 276);
  const auto vop3_cmpx_gt_u64_word =
      MakeVop3(*v_cmpx_gt_u64_opcode, 110, 62, 278);
  const std::vector<std::uint32_t> vector_cmpx_program = {
      vop3_cmpx_eq_u32_word[0], vop3_cmpx_eq_u32_word[1],
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),
      MakeSop1(0, 0, 255), 111u,
      vop3_cmpx_lt_i64_word[0], vop3_cmpx_lt_i64_word[1],
      MakeSopp(*s_cbranch_execnz_cmpx_opcode, 1),
      MakeSop1(0, 1, 255), 222u,
      vop3_cmpx_gt_u64_word[0], vop3_cmpx_gt_u64_word[1],
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),
      MakeSop1(0, 2, 255), 333u,
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_cmpx_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded vector cmpx program size") ||
      !Expect(decoded_program[0].opcode == "V_CMPX_EQ_U32",
              "expected V_CMPX_EQ_U32 decode") ||
      !Expect(decoded_program[1].opcode == "S_CBRANCH_EXECZ",
              "expected first EXECZ branch decode") ||
      !Expect(decoded_program[3].opcode == "V_CMPX_LT_I64",
              "expected V_CMPX_LT_I64 decode") ||
      !Expect(decoded_program[4].opcode == "S_CBRANCH_EXECNZ",
              "expected EXECNZ branch decode") ||
      !Expect(decoded_program[6].opcode == "V_CMPX_GT_U64",
              "expected V_CMPX_GT_U64 decode") ||
      !Expect(decoded_program[7].opcode == "S_CBRANCH_EXECZ",
              "expected second EXECZ branch decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr,
              "expected cmpx scalar destination decode") ||
      !Expect(decoded_program[0].operands[0].index == 106,
              "expected cmpx destination pair index") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr,
              "expected cmpx src0 scalar decode") ||
      !Expect(decoded_program[0].operands[1].index == 12,
              "expected cmpx src0 scalar index") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
              "expected cmpx src1 vector decode") ||
      !Expect(decoded_program[0].operands[2].index == 12,
              "expected cmpx src1 vector index")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_cmpx_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_state.halted,
              "expected decoded vector cmpx program to halt") ||
      !Expect(vector_cmpx_state.sgprs[106] == 9u,
              "expected decoded v_cmpx_eq_u32 low mask result") ||
      !Expect(vector_cmpx_state.sgprs[108] == 1u,
              "expected decoded v_cmpx_lt_i64 low mask result") ||
      !Expect(vector_cmpx_state.sgprs[110] == 0u,
              "expected decoded v_cmpx_gt_u64 low mask result") ||
      !Expect(vector_cmpx_state.sgprs[0] == 111u,
              "expected decoded EXECZ fallthrough after first cmpx") ||
      !Expect(vector_cmpx_state.sgprs[1] == 0u,
              "expected decoded EXECNZ branch to skip second move") ||
      !Expect(vector_cmpx_state.sgprs[2] == 0u,
              "expected decoded EXECZ branch to skip third move") ||
      !Expect(vector_cmpx_state.exec_mask == 0u,
              "expected decoded final EXEC mask after cmpx chain") ||
      !Expect(vector_cmpx_state.vcc_mask == 0u,
              "expected decoded final VCC mask after cmpx chain")) {
    return 1;
  }

  const auto v_cmpx_eq_u32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_EQ_U32", "ENC_VOPC");
  const auto v_cmpx_ne_u32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_NE_U32", "ENC_VOPC");
  if (!Expect(v_cmpx_eq_u32_vopc_opcode.has_value(),
              "expected V_CMPX_EQ_U32 VOPC opcode lookup") ||
      !Expect(v_cmpx_ne_u32_vopc_opcode.has_value(),
              "expected V_CMPX_NE_U32 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_cmpx_vopc_program = {
      MakeVopc(*v_cmpx_eq_u32_vopc_opcode, 12, 12),         // v_cmpx_eq_u32 vcc, s12, v12
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),            // s_cbranch_execz +1
      MakeSop1(0, 3, 255), 111u,                            // s_mov_b32 s3, 111
      MakeVopc(*v_cmpx_ne_u32_vopc_opcode, 255, 13), 8u,    // v_cmpx_ne_u32 vcc, 8, v13
      MakeSopp(*s_cbranch_execnz_cmpx_opcode, 1),           // s_cbranch_execnz +1
      MakeSop1(0, 4, 255), 222u,                            // s_mov_b32 s4, 222
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_cmpx_vopc_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 7,
              "expected decoded VOPC cmpx program size") ||
      !Expect(decoded_program[0].opcode == "V_CMPX_EQ_U32",
              "expected VOPC V_CMPX_EQ_U32 decode") ||
      !Expect(decoded_program[3].opcode == "V_CMPX_NE_U32",
              "expected VOPC V_CMPX_NE_U32 decode") ||
      !Expect(decoded_program[3].operands[1].kind == OperandKind::kImm32 &&
                  decoded_program[3].operands[1].imm32 == 8u,
              "expected VOPC cmpx literal src0 decode")) {
    return 1;
  }

  WaveExecutionState vector_cmpx_vopc_state;
  vector_cmpx_vopc_state.exec_mask = 0b1011ULL;
  vector_cmpx_vopc_state.sgprs[12] = 7u;
  vector_cmpx_vopc_state.vgprs[12][0] = 7u;
  vector_cmpx_vopc_state.vgprs[12][1] = 5u;
  vector_cmpx_vopc_state.vgprs[12][3] = 7u;
  vector_cmpx_vopc_state.vgprs[13][0] = 8u;
  vector_cmpx_vopc_state.vgprs[13][1] = 8u;
  vector_cmpx_vopc_state.vgprs[13][3] = 9u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_cmpx_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_vopc_state.halted,
              "expected VOPC cmpx program to halt") ||
      !Expect(vector_cmpx_vopc_state.sgprs[3] == 111u,
              "expected first move to execute after nonzero cmpx exec") ||
      !Expect(vector_cmpx_vopc_state.sgprs[4] == 0u,
              "expected execnz branch to skip second move after VOPC cmpx") ||
      !Expect(vector_cmpx_vopc_state.sgprs[106] == 8u &&
                  vector_cmpx_vopc_state.sgprs[107] == 0u,
              "expected VOPC cmpx low VCC mask result") ||
      !Expect(vector_cmpx_vopc_state.vcc_mask == 8u,
              "expected VOPC cmpx final VCC mask result") ||
      !Expect(vector_cmpx_vopc_state.exec_mask == 8u,
              "expected VOPC cmpx final EXEC mask result")) {
    return 1;
  }

  const auto v_cmpx_lt_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_LT_F32", "ENC_VOPC");
  const auto v_cmpx_u_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_U_F32", "ENC_VOPC");
  const auto v_cmpx_o_f32_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_O_F32", "ENC_VOPC");
  if (!Expect(v_cmpx_lt_f32_vopc_opcode.has_value(),
              "expected V_CMPX_LT_F32 VOPC opcode lookup") ||
      !Expect(v_cmpx_u_f32_vopc_opcode.has_value(),
              "expected V_CMPX_U_F32 VOPC opcode lookup") ||
      !Expect(v_cmpx_o_f32_vopc_opcode.has_value(),
              "expected V_CMPX_O_F32 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_cmpx_f32_vopc_program = {
      MakeVopc(*v_cmpx_lt_f32_vopc_opcode, 16, 16),         // v_cmpx_lt_f32 vcc, s16, v16
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),            // s_cbranch_execz +1
      MakeSop1(0, 5, 255), 111u,                            // s_mov_b32 s5, 111
      MakeVopc(*v_cmpx_u_f32_vopc_opcode, 255, 17), 0x7fc00000u, // v_cmpx_u_f32 vcc, nan, v17
      MakeSopp(*s_cbranch_execnz_cmpx_opcode, 1),           // s_cbranch_execnz +1
      MakeSop1(0, 6, 255), 222u,                            // s_mov_b32 s6, 222
      MakeVopc(*v_cmpx_o_f32_vopc_opcode, 19, 19),          // v_cmpx_o_f32 vcc, s19, v19
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),            // s_cbranch_execz +1
      MakeSop1(0, 7, 255), 333u,                            // s_mov_b32 s7, 333
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_cmpx_f32_vopc_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded VOPC f32 cmpx program size") ||
      !Expect(decoded_program[0].opcode == "V_CMPX_LT_F32",
              "expected VOPC V_CMPX_LT_F32 decode") ||
      !Expect(decoded_program[3].opcode == "V_CMPX_U_F32",
              "expected VOPC V_CMPX_U_F32 decode") ||
      !Expect(decoded_program[6].opcode == "V_CMPX_O_F32",
              "expected VOPC V_CMPX_O_F32 decode") ||
      !Expect(decoded_program[3].operands[1].kind == OperandKind::kImm32 &&
                  decoded_program[3].operands[1].imm32 == 0x7fc00000u,
              "expected VOPC f32 cmpx literal src0 decode")) {
    return 1;
  }

  WaveExecutionState vector_cmpx_f32_vopc_state;
  vector_cmpx_f32_vopc_state.exec_mask = 0b1011ULL;
  vector_cmpx_f32_vopc_state.sgprs[16] = FloatBits(1.5f);
  vector_cmpx_f32_vopc_state.sgprs[19] = 0x7fc00000u;
  vector_cmpx_f32_vopc_state.vgprs[16][0] = FloatBits(2.0f);
  vector_cmpx_f32_vopc_state.vgprs[16][1] = FloatBits(1.0f);
  vector_cmpx_f32_vopc_state.vgprs[16][3] = FloatBits(1.5f);
  vector_cmpx_f32_vopc_state.vgprs[17][0] = FloatBits(0.0f);
  vector_cmpx_f32_vopc_state.vgprs[17][1] = FloatBits(4.0f);
  vector_cmpx_f32_vopc_state.vgprs[17][3] = FloatBits(-1.0f);
  vector_cmpx_f32_vopc_state.vgprs[19][0] = FloatBits(1.0f);
  vector_cmpx_f32_vopc_state.vgprs[19][1] = FloatBits(0.0f);
  vector_cmpx_f32_vopc_state.vgprs[19][3] = FloatBits(2.0f);
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_cmpx_f32_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_f32_vopc_state.halted,
              "expected VOPC f32 cmpx program to halt") ||
      !Expect(vector_cmpx_f32_vopc_state.sgprs[5] == 111u,
              "expected first move to execute after f32 cmpx exec") ||
      !Expect(vector_cmpx_f32_vopc_state.sgprs[6] == 0u,
              "expected execnz branch to skip second move after f32 cmpx") ||
      !Expect(vector_cmpx_f32_vopc_state.sgprs[7] == 0u,
              "expected execz branch to skip third move after f32 cmpx") ||
      !Expect(vector_cmpx_f32_vopc_state.sgprs[106] == 0u &&
                  vector_cmpx_f32_vopc_state.sgprs[107] == 0u,
              "expected VOPC f32 cmpx final low VCC mask result") ||
      !Expect(vector_cmpx_f32_vopc_state.vcc_mask == 0u,
              "expected VOPC f32 cmpx final VCC mask result") ||
      !Expect(vector_cmpx_f32_vopc_state.exec_mask == 0u,
              "expected VOPC f32 cmpx final EXEC mask result")) {
    return 1;
  }

  const auto v_cmpx_lt_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_LT_F64", "ENC_VOPC");
  const auto v_cmpx_u_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_U_F64", "ENC_VOPC");
  const auto v_cmpx_o_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_O_F64", "ENC_VOPC");
  if (!Expect(v_cmpx_lt_f64_vopc_opcode.has_value(),
              "expected V_CMPX_LT_F64 VOPC opcode lookup") ||
      !Expect(v_cmpx_u_f64_vopc_opcode.has_value(),
              "expected V_CMPX_U_F64 VOPC opcode lookup") ||
      !Expect(v_cmpx_o_f64_vopc_opcode.has_value(),
              "expected V_CMPX_O_F64 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_cmpx_f64_vopc_program = {
      MakeVopc(*v_cmpx_lt_f64_vopc_opcode, 70, 30),  // v_cmpx_lt_f64 vcc, s[70:71], v[30:31]
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),     // s_cbranch_execz +1
      MakeSop1(0, 8, 255), 111u,                     // s_mov_b32 s8, 111
      MakeVopc(*v_cmpx_u_f64_vopc_opcode, 72, 32),   // v_cmpx_u_f64 vcc, s[72:73], v[32:33]
      MakeSopp(*s_cbranch_execnz_cmpx_opcode, 1),    // s_cbranch_execnz +1
      MakeSop1(0, 9, 255), 222u,                     // s_mov_b32 s9, 222
      MakeVopc(*v_cmpx_o_f64_vopc_opcode, 74, 34),   // v_cmpx_o_f64 vcc, s[74:75], v[34:35]
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),     // s_cbranch_execz +1
      MakeSop1(0, 10, 255), 333u,                    // s_mov_b32 s10, 333
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_cmpx_f64_vopc_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded VOPC f64 cmpx program size") ||
      !Expect(decoded_program[0].opcode == "V_CMPX_LT_F64",
              "expected VOPC V_CMPX_LT_F64 decode") ||
      !Expect(decoded_program[3].opcode == "V_CMPX_U_F64",
              "expected VOPC V_CMPX_U_F64 decode") ||
      !Expect(decoded_program[6].opcode == "V_CMPX_O_F64",
              "expected VOPC V_CMPX_O_F64 decode")) {
    return 1;
  }

  WaveExecutionState vector_cmpx_f64_vopc_state;
  vector_cmpx_f64_vopc_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(1.5), &vector_cmpx_f64_vopc_state.sgprs[70],
           &vector_cmpx_f64_vopc_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_cmpx_f64_vopc_state.sgprs[72],
           &vector_cmpx_f64_vopc_state.sgprs[73]);
  SplitU64(kQuietNan64, &vector_cmpx_f64_vopc_state.sgprs[74],
           &vector_cmpx_f64_vopc_state.sgprs[75]);
  SplitU64(DoubleBits(2.0), &vector_cmpx_f64_vopc_state.vgprs[30][0],
           &vector_cmpx_f64_vopc_state.vgprs[31][0]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_f64_vopc_state.vgprs[30][1],
           &vector_cmpx_f64_vopc_state.vgprs[31][1]);
  SplitU64(DoubleBits(1.5), &vector_cmpx_f64_vopc_state.vgprs[30][3],
           &vector_cmpx_f64_vopc_state.vgprs[31][3]);
  SplitU64(DoubleBits(0.0), &vector_cmpx_f64_vopc_state.vgprs[32][0],
           &vector_cmpx_f64_vopc_state.vgprs[33][0]);
  SplitU64(DoubleBits(4.0), &vector_cmpx_f64_vopc_state.vgprs[32][1],
           &vector_cmpx_f64_vopc_state.vgprs[33][1]);
  SplitU64(DoubleBits(-1.0), &vector_cmpx_f64_vopc_state.vgprs[32][3],
           &vector_cmpx_f64_vopc_state.vgprs[33][3]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_f64_vopc_state.vgprs[34][0],
           &vector_cmpx_f64_vopc_state.vgprs[35][0]);
  SplitU64(DoubleBits(0.0), &vector_cmpx_f64_vopc_state.vgprs[34][1],
           &vector_cmpx_f64_vopc_state.vgprs[35][1]);
  SplitU64(DoubleBits(2.0), &vector_cmpx_f64_vopc_state.vgprs[34][3],
           &vector_cmpx_f64_vopc_state.vgprs[35][3]);
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_cmpx_f64_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_f64_vopc_state.halted,
              "expected VOPC f64 cmpx program to halt") ||
      !Expect(vector_cmpx_f64_vopc_state.sgprs[8] == 111u,
              "expected first move to execute after f64 cmpx exec") ||
      !Expect(vector_cmpx_f64_vopc_state.sgprs[9] == 0u,
              "expected execnz branch to skip second move after f64 cmpx") ||
      !Expect(vector_cmpx_f64_vopc_state.sgprs[10] == 0u,
              "expected execz branch to skip third move after f64 cmpx") ||
      !Expect(vector_cmpx_f64_vopc_state.sgprs[106] == 0u &&
                  vector_cmpx_f64_vopc_state.sgprs[107] == 0u,
              "expected VOPC f64 cmpx final low VCC mask result") ||
      !Expect(vector_cmpx_f64_vopc_state.vcc_mask == 0u,
              "expected VOPC f64 cmpx final VCC mask result") ||
      !Expect(vector_cmpx_f64_vopc_state.exec_mask == 0u,
              "expected VOPC f64 cmpx final EXEC mask result")) {
    return 1;
  }

  const auto v_cmpx_class_f64_vopc_opcode =
      FindDefaultEncodingOpcode("V_CMPX_CLASS_F64", "ENC_VOPC");
  if (!Expect(v_cmpx_class_f64_vopc_opcode.has_value(),
              "expected V_CMPX_CLASS_F64 VOPC opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> vector_cmpx_class_f64_vopc_program = {
      MakeVopc(*v_cmpx_class_f64_vopc_opcode, 70, 40),  // v_cmpx_class_f64 vcc, s[70:71], v40
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),         // s_cbranch_execz +1
      MakeSop1(0, 11, 255), 111u,                        // s_mov_b32 s11, 111
      MakeVopc(*v_cmpx_class_f64_vopc_opcode, 72, 41),  // v_cmpx_class_f64 vcc, s[72:73], v41
      MakeSopp(*s_cbranch_execnz_cmpx_opcode, 1),       // s_cbranch_execnz +1
      MakeSop1(0, 12, 255), 222u,                       // s_mov_b32 s12, 222
      MakeVopc(*v_cmpx_class_f64_vopc_opcode, 74, 42),  // v_cmpx_class_f64 vcc, s[74:75], v42
      MakeSopp(*s_cbranch_execz_cmpx_opcode, 1),        // s_cbranch_execz +1
      MakeSop1(0, 13, 255), 333u,                       // s_mov_b32 s13, 333
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vector_cmpx_class_f64_vopc_program,
                                    &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded VOPC f64 class cmpx program size") ||
      !Expect(decoded_program[0].opcode == "V_CMPX_CLASS_F64",
              "expected VOPC V_CMPX_CLASS_F64 decode") ||
      !Expect(decoded_program[3].opcode == "V_CMPX_CLASS_F64",
              "expected second VOPC V_CMPX_CLASS_F64 decode") ||
      !Expect(decoded_program[6].opcode == "V_CMPX_CLASS_F64",
              "expected third VOPC V_CMPX_CLASS_F64 decode")) {
    return 1;
  }

  WaveExecutionState vector_cmpx_class_f64_vopc_state;
  vector_cmpx_class_f64_vopc_state.exec_mask = 0b1011ULL;
  SplitU64(DoubleBits(-0.0), &vector_cmpx_class_f64_vopc_state.sgprs[70],
           &vector_cmpx_class_f64_vopc_state.sgprs[71]);
  SplitU64(kQuietNan64, &vector_cmpx_class_f64_vopc_state.sgprs[72],
           &vector_cmpx_class_f64_vopc_state.sgprs[73]);
  SplitU64(DoubleBits(1.0), &vector_cmpx_class_f64_vopc_state.sgprs[74],
           &vector_cmpx_class_f64_vopc_state.sgprs[75]);
  vector_cmpx_class_f64_vopc_state.vgprs[40][0] = 0x20u;
  vector_cmpx_class_f64_vopc_state.vgprs[40][1] = 0x40u;
  vector_cmpx_class_f64_vopc_state.vgprs[40][3] = 0x60u;
  vector_cmpx_class_f64_vopc_state.vgprs[41][0] = 0x2u;
  vector_cmpx_class_f64_vopc_state.vgprs[41][3] = 0x1u;
  vector_cmpx_class_f64_vopc_state.vgprs[42][0] = 0x20u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program,
                                         &vector_cmpx_class_f64_vopc_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_cmpx_class_f64_vopc_state.halted,
              "expected VOPC f64 class cmpx program to halt") ||
      !Expect(vector_cmpx_class_f64_vopc_state.sgprs[11] == 111u,
              "expected first move to execute after f64 class cmpx exec") ||
      !Expect(vector_cmpx_class_f64_vopc_state.sgprs[12] == 0u,
              "expected execnz branch to skip second move after f64 class cmpx") ||
      !Expect(vector_cmpx_class_f64_vopc_state.sgprs[13] == 0u,
              "expected execz branch to skip third move after f64 class cmpx") ||
      !Expect(vector_cmpx_class_f64_vopc_state.sgprs[106] == 0u &&
                  vector_cmpx_class_f64_vopc_state.sgprs[107] == 0u,
              "expected VOPC f64 class cmpx final low VCC mask result") ||
      !Expect(vector_cmpx_class_f64_vopc_state.vcc_mask == 0u,
              "expected VOPC f64 class cmpx final VCC mask result") ||
      !Expect(vector_cmpx_class_f64_vopc_state.exec_mask == 0u,
              "expected VOPC f64 class cmpx final EXEC mask result")) {
    return 1;
  }

  DecodedInstruction literal_instruction;
  std::size_t words_consumed = 0;
  const std::vector<std::uint32_t> literal_program = {
      MakeSop1(0, 4, 255),
      0x12345678,
  };
  if (!Expect(decoder.DecodeInstruction(literal_program, &literal_instruction,
                                        &words_consumed, &error_message),
              error_message.c_str()) ||
      !Expect(words_consumed == 2, "expected literal decode to consume two words") ||
      !Expect(literal_instruction.operands[1].kind == OperandKind::kImm32,
              "expected literal source operand") ||
      !Expect(literal_instruction.operands[1].imm32 == 0x12345678u,
              "expected literal payload to match")) {
    return 1;
  }

  const std::vector<std::uint32_t> branch_program = {
      MakeSop1(0, 0, 131),          // s_mov_b32 s0, 3
      MakeSop1(0, 1, 128),          // s_mov_b32 s1, 0
      MakeSop2(0, 1, 1, 129),       // s_add_u32 s1, s1, 1
      MakeSop2(1, 0, 0, 129),       // s_sub_u32 s0, s0, 1
      MakeSopc(7, 0, 128),          // s_cmp_lg_u32 s0, 0
      MakeSopp(5, 0xfffc),          // s_cbranch_scc1 -4
      MakeSopp(1),                  // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(branch_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 7, "expected 7 decoded branch instructions") ||
      !Expect(decoded_program[4].opcode == "S_CMP_LG_U32", "expected SOPC decode") ||
      !Expect(decoded_program[5].opcode == "S_CBRANCH_SCC1",
              "expected conditional branch decode") ||
      !Expect(decoded_program[5].operands[0].imm32 ==
                  static_cast<std::uint32_t>(-4),
              "expected sign-extended branch delta")) {
    return 1;
  }

  WaveExecutionState branch_state;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &branch_state, &error_message),
              error_message.c_str()) ||
      !Expect(branch_state.halted, "expected branch decode program to halt") ||
      !Expect(branch_state.sgprs[0] == 0, "expected loop counter to reach zero") ||
      !Expect(branch_state.sgprs[1] == 3, "expected decoded loop to iterate three times") ||
      !Expect(!branch_state.scc, "expected final SCC to be false")) {
    return 1;
  }

  const auto v_cmp_eq_u32_branch_opcode =
      FindDefaultEncodingOpcode("V_CMP_EQ_U32", "ENC_VOP3");
  if (!Expect(v_cmp_eq_u32_branch_opcode.has_value(),
              "expected branch V_CMP_EQ_U32 opcode lookup")) {
    return 1;
  }
  const auto vcc_cmp_word = MakeVop3(*v_cmp_eq_u32_branch_opcode, 106, 12, 268);
  const std::vector<std::uint32_t> vcc_branch_program = {
      vcc_cmp_word[0], vcc_cmp_word[1],
      MakeSopp(7, 2),               // s_cbranch_vccnz +2
      MakeSop1(0, 2, 255), 111u,    // s_mov_b32 s2, 111
      MakeSopp(6, 1),               // s_cbranch_vccz +1
      MakeSop1(0, 3, 255), 222u,    // s_mov_b32 s3, 222
      MakeSopp(1),                  // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(vcc_branch_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 6, "expected decoded VCC branch program size") ||
      !Expect(decoded_program[0].opcode == "V_CMP_EQ_U32",
              "expected VCC compare decode") ||
      !Expect(decoded_program[1].opcode == "S_CBRANCH_VCCNZ",
              "expected VCCNZ branch decode") ||
      !Expect(decoded_program[3].opcode == "S_CBRANCH_VCCZ",
              "expected VCCZ branch decode") ||
      !Expect(decoded_program[1].operands[0].imm32 == 2u,
              "expected VCCNZ branch delta") ||
      !Expect(decoded_program[3].operands[0].imm32 == 1u,
              "expected VCCZ branch delta")) {
    return 1;
  }

  WaveExecutionState vcc_branch_state;
  vcc_branch_state.exec_mask = 0b1011ULL;
  vcc_branch_state.sgprs[12] = 7u;
  vcc_branch_state.vgprs[12][0] = 7u;
  vcc_branch_state.vgprs[12][1] = 5u;
  vcc_branch_state.vgprs[12][3] = 7u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vcc_branch_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vcc_branch_state.halted,
              "expected decoded VCC branch program to halt") ||
      !Expect(vcc_branch_state.sgprs[2] == 0u,
              "expected decoded VCCNZ branch to skip first move") ||
      !Expect(vcc_branch_state.sgprs[3] == 222u,
              "expected decoded VCCZ branch to fall through") ||
      !Expect(vcc_branch_state.vcc_mask == 9u,
              "expected decoded VCC mask after branch program")) {
    return 1;
  }

  const auto s_and_saveexec_b64_opcode =
      FindDefaultEncodingOpcode("S_AND_SAVEEXEC_B64", "ENC_SOP1");
  const auto s_or_saveexec_b64_opcode =
      FindDefaultEncodingOpcode("S_OR_SAVEEXEC_B64", "ENC_SOP1");
  const auto s_cbranch_execz_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_EXECZ", "ENC_SOPP");
  const auto s_cbranch_execnz_opcode =
      FindDefaultEncodingOpcode("S_CBRANCH_EXECNZ", "ENC_SOPP");
  const auto s_bitcmp0_b32_opcode =
      FindDefaultEncodingOpcode("S_BITCMP0_B32", "ENC_SOPC");
  const auto s_bitcmp1_b32_opcode =
      FindDefaultEncodingOpcode("S_BITCMP1_B32", "ENC_SOPC");
  const auto s_bitcmp1_b64_opcode =
      FindDefaultEncodingOpcode("S_BITCMP1_B64", "ENC_SOPC");
  const auto v_readfirstlane_b32_opcode =
      FindDefaultEncodingOpcode("V_READFIRSTLANE_B32", "ENC_VOP1");
  const auto v_readlane_b32_opcode =
      FindDefaultEncodingOpcode("V_READLANE_B32", "ENC_VOP3");
  const auto v_writelane_b32_opcode =
      FindDefaultEncodingOpcode("V_WRITELANE_B32", "ENC_VOP3");
  if (!Expect(s_and_saveexec_b64_opcode.has_value(),
              "expected S_AND_SAVEEXEC_B64 opcode lookup") ||
      !Expect(s_or_saveexec_b64_opcode.has_value(),
              "expected S_OR_SAVEEXEC_B64 opcode lookup") ||
      !Expect(s_cbranch_execz_opcode.has_value(),
              "expected S_CBRANCH_EXECZ opcode lookup") ||
      !Expect(s_cbranch_execnz_opcode.has_value(),
              "expected S_CBRANCH_EXECNZ opcode lookup") ||
      !Expect(s_bitcmp0_b32_opcode.has_value(),
              "expected S_BITCMP0_B32 opcode lookup") ||
      !Expect(s_bitcmp1_b32_opcode.has_value(),
              "expected S_BITCMP1_B32 opcode lookup") ||
      !Expect(s_bitcmp1_b64_opcode.has_value(),
              "expected S_BITCMP1_B64 opcode lookup") ||
      !Expect(v_readfirstlane_b32_opcode.has_value(),
              "expected V_READFIRSTLANE_B32 opcode lookup") ||
      !Expect(v_readlane_b32_opcode.has_value(),
              "expected V_READLANE_B32 opcode lookup") ||
      !Expect(v_writelane_b32_opcode.has_value(),
              "expected V_WRITELANE_B32 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> exec_mask_program = {
      MakeSop1(*s_and_saveexec_b64_opcode, 30, 20),   // s_and_saveexec_b64 s[30:31], s[20:21]
      MakeVop1(1, 20, 26),                            // v_mov_b32 v20, s26
      MakeSop1(*s_and_saveexec_b64_opcode, 32, 22),   // s_and_saveexec_b64 s[32:33], s[22:23]
      MakeVop1(1, 21, 27),                            // v_mov_b32 v21, s27
      MakeSopp(*s_cbranch_execnz_opcode, 1),          // s_cbranch_execnz +1
      MakeSop1(0, 0, 255), 111u,                      // s_mov_b32 s0, 111
      MakeSopp(*s_cbranch_execz_opcode, 1),           // s_cbranch_execz +1
      MakeSop1(0, 1, 255), 222u,                      // s_mov_b32 s1, 222
      MakeSop1(*s_or_saveexec_b64_opcode, 34, 24),    // s_or_saveexec_b64 s[34:35], s[24:25]
      MakeVop1(1, 22, 28),                            // v_mov_b32 v22, s28
      MakeVop1(*v_readfirstlane_b32_opcode, 2, 278),  // v_readfirstlane_b32 s2, v22
      MakeSopp(1),                                    // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(exec_mask_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 12, "expected decoded exec-mask program size") ||
      !Expect(decoded_program[0].opcode == "S_AND_SAVEEXEC_B64",
              "expected S_AND_SAVEEXEC_B64 decode") ||
      !Expect(decoded_program[4].opcode == "S_CBRANCH_EXECNZ",
              "expected S_CBRANCH_EXECNZ decode") ||
      !Expect(decoded_program[6].opcode == "S_CBRANCH_EXECZ",
              "expected S_CBRANCH_EXECZ decode") ||
      !Expect(decoded_program[10].opcode == "V_READFIRSTLANE_B32",
              "expected V_READFIRSTLANE_B32 decode") ||
      !Expect(decoded_program[10].operands[0].kind == OperandKind::kSgpr,
              "expected readfirstlane scalar destination decode") ||
      !Expect(decoded_program[10].operands[0].index == 2,
              "expected readfirstlane scalar destination index") ||
      !Expect(decoded_program[10].operands[1].kind == OperandKind::kVgpr,
              "expected readfirstlane vector source decode") ||
      !Expect(decoded_program[10].operands[1].index == 22,
              "expected readfirstlane vector source index")) {
    return 1;
  }

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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &exec_mask_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(exec_mask_state.halted,
              "expected decoded exec-mask program to halt") ||
      !Expect(exec_mask_state.sgprs[30] == 0b1011u,
              "expected decoded first saveexec low result") ||
      !Expect(exec_mask_state.sgprs[32] == 0b1001u,
              "expected decoded second saveexec low result") ||
      !Expect(exec_mask_state.sgprs[34] == 0u,
              "expected decoded third saveexec low result") ||
      !Expect(exec_mask_state.sgprs[0] == 111u,
              "expected decoded EXECNZ fallthrough move result") ||
      !Expect(exec_mask_state.sgprs[1] == 0u,
              "expected decoded EXECZ branch to skip move") ||
      !Expect(exec_mask_state.sgprs[2] == 77u,
              "expected decoded readfirstlane result") ||
      !Expect(exec_mask_state.exec_mask == 0b0010u,
              "expected decoded final exec mask") ||
      !Expect(exec_mask_state.vgprs[20][0] == 55u,
              "expected decoded exec-masked vector move lane 0 result") ||
      !Expect(exec_mask_state.vgprs[20][1] == 0u,
              "expected decoded exec-masked vector move lane 1 result") ||
      !Expect(exec_mask_state.vgprs[20][3] == 55u,
              "expected decoded exec-masked vector move lane 3 result") ||
      !Expect(exec_mask_state.vgprs[21][2] == 0xdeadbeefu,
              "expected decoded zero-exec inactive lane preservation") ||
      !Expect(exec_mask_state.vgprs[22][1] == 77u,
              "expected decoded restored-exec vector move lane 1 result")) {
    return 1;
  }

  const auto s_mov_b64_exec_opcode =
      FindDefaultEncodingOpcode("S_MOV_B64", "ENC_SOP1");
  const auto s_and_b64_exec_opcode =
      FindDefaultEncodingOpcode("S_AND_B64", "ENC_SOP2");
  const auto s_or_b64_exec_opcode =
      FindDefaultEncodingOpcode("S_OR_B64", "ENC_SOP2");
  if (!Expect(s_mov_b64_exec_opcode.has_value(),
              "expected S_MOV_B64 opcode lookup for exec-pair test") ||
      !Expect(s_and_b64_exec_opcode.has_value(),
              "expected S_AND_B64 opcode lookup for exec-pair test") ||
      !Expect(s_or_b64_exec_opcode.has_value(),
              "expected S_OR_B64 opcode lookup for exec-pair test")) {
    return 1;
  }

  const std::vector<std::uint32_t> exec_pair_program = {
      MakeSop1(*s_mov_b64_exec_opcode, 126, 20),            // s_mov_b64 exec, s[20:21]
      MakeSopp(*s_cbranch_execz_opcode, 1),                 // s_cbranch_execz +1
      MakeSop1(0, 10, 255), 111u,                           // s_mov_b32 s10, 111
      MakeSop2(*s_and_b64_exec_opcode, 126, 126, 22),       // s_and_b64 exec, exec, s[22:23]
      MakeSopp(*s_cbranch_execz_opcode, 1),                 // s_cbranch_execz +1
      MakeSop1(0, 11, 255), 222u,                           // s_mov_b32 s11, 222
      MakeSop2(*s_or_b64_exec_opcode, 126, 126, 24),        // s_or_b64 exec, exec, s[24:25]
      MakeSopp(*s_cbranch_execnz_opcode, 1),                // s_cbranch_execnz +1
      MakeSop1(0, 12, 255), 333u,                           // s_mov_b32 s12, 333
      MakeSopp(1),                                          // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(exec_pair_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 10,
              "expected decoded exec-pair program size") ||
      !Expect(decoded_program[0].opcode == "S_MOV_B64",
              "expected exec-pair move decode") ||
      !Expect(decoded_program[3].opcode == "S_AND_B64",
              "expected exec-pair and decode") ||
      !Expect(decoded_program[6].opcode == "S_OR_B64",
              "expected exec-pair or decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr,
              "expected exec-pair scalar destination decode") ||
      !Expect(decoded_program[0].operands[0].index == 126,
              "expected exec-pair destination index")) {
    return 1;
  }

  WaveExecutionState exec_pair_state;
  exec_pair_state.exec_mask = 0b1011ULL;
  exec_pair_state.sgprs[20] = 0b1001u;
  exec_pair_state.sgprs[21] = 0u;
  exec_pair_state.sgprs[22] = 0b0010u;
  exec_pair_state.sgprs[23] = 0u;
  exec_pair_state.sgprs[24] = 0b0100u;
  exec_pair_state.sgprs[25] = 0u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &exec_pair_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(exec_pair_state.halted,
              "expected decoded exec-pair binary program to halt") ||
      !Expect(exec_pair_state.sgprs[10] == 111u,
              "expected decoded exec-pair first move result") ||
      !Expect(exec_pair_state.sgprs[11] == 0u,
              "expected decoded execz branch to skip second move") ||
      !Expect(exec_pair_state.sgprs[12] == 0u,
              "expected decoded execnz branch to skip third move") ||
      !Expect(exec_pair_state.exec_mask == 0b0100u,
              "expected decoded final exec mask from scalar pair writes")) {
    return 1;
  }

  const auto s_andn1_wrexec_b64_opcode =
      FindDefaultEncodingOpcode("S_ANDN1_WREXEC_B64", "ENC_SOP1");
  const auto s_andn2_wrexec_b64_opcode =
      FindDefaultEncodingOpcode("S_ANDN2_WREXEC_B64", "ENC_SOP1");
  if (!Expect(s_andn1_wrexec_b64_opcode.has_value(),
              "expected S_ANDN1_WREXEC_B64 opcode lookup") ||
      !Expect(s_andn2_wrexec_b64_opcode.has_value(),
              "expected S_ANDN2_WREXEC_B64 opcode lookup")) {
    return 1;
  }

  const std::vector<std::uint32_t> wrexec_program = {
      MakeSop1(*s_andn1_wrexec_b64_opcode, 30, 20),        // s_andn1_wrexec_b64 s[30:31], s[20:21]
      MakeSopp(*s_cbranch_execnz_opcode, 1),               // s_cbranch_execnz +1
      MakeSop1(0, 16, 255), 111u,                          // s_mov_b32 s16, 111
      MakeSop1(*s_andn2_wrexec_b64_opcode, 32, 22),        // s_andn2_wrexec_b64 s[32:33], s[22:23]
      MakeSopp(*s_cbranch_execz_opcode, 1),                // s_cbranch_execz +1
      MakeSop1(0, 17, 255), 222u,                          // s_mov_b32 s17, 222
      MakeSopp(1),                                         // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(wrexec_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 7,
              "expected decoded wrexec program size") ||
      !Expect(decoded_program[0].opcode == "S_ANDN1_WREXEC_B64",
              "expected andn1_wrexec decode") ||
      !Expect(decoded_program[3].opcode == "S_ANDN2_WREXEC_B64",
              "expected andn2_wrexec decode")) {
    return 1;
  }

  WaveExecutionState wrexec_state;
  wrexec_state.exec_mask = 0b1011ULL;
  wrexec_state.sgprs[20] = 0b1001u;
  wrexec_state.sgprs[21] = 0u;
  wrexec_state.sgprs[22] = 0b0010u;
  wrexec_state.sgprs[23] = 0u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &wrexec_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(wrexec_state.halted,
              "expected decoded wrexec program to halt") ||
      !Expect(wrexec_state.sgprs[30] == 0b0010u,
              "expected decoded andn1_wrexec low result") ||
      !Expect(wrexec_state.sgprs[32] == 0u,
              "expected decoded andn2_wrexec low result") ||
      !Expect(wrexec_state.sgprs[16] == 0u,
              "expected decoded execnz branch to skip first move") ||
      !Expect(wrexec_state.sgprs[17] == 0u,
              "expected decoded execz branch to skip second move") ||
      !Expect(wrexec_state.exec_mask == 0u,
              "expected decoded final exec mask after wrexec ops") ||
      !Expect(!wrexec_state.scc,
              "expected decoded final scc after zero wrexec result")) {
    return 1;
  }

  const std::vector<std::uint32_t> special_source_program = {
      MakeSop1(0, 13, 251),  // s_mov_b32 s13, src_vccz
      MakeSop1(0, 14, 252),  // s_mov_b32 s14, src_execz
      MakeSop1(0, 15, 253),  // s_mov_b32 s15, src_scc
      MakeSopp(1),           // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(special_source_program, &decoded_program,
                                    &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded special-source program size") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kSgpr,
              "expected src_vccz decode kind") ||
      !Expect(decoded_program[0].operands[1].index == 251,
              "expected src_vccz decode index") ||
      !Expect(decoded_program[1].operands[1].index == 252,
              "expected src_execz decode index") ||
      !Expect(decoded_program[2].operands[1].index == 253,
              "expected src_scc decode index")) {
    return 1;
  }

  WaveExecutionState special_source_state;
  special_source_state.exec_mask = 0u;
  special_source_state.vcc_mask = 0u;
  special_source_state.scc = true;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &special_source_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(special_source_state.sgprs[13] == 1u,
              "expected decoded src_vccz result") ||
      !Expect(special_source_state.sgprs[14] == 1u,
              "expected decoded src_execz result") ||
      !Expect(special_source_state.sgprs[15] == 1u,
              "expected decoded src_scc result")) {
    return 1;
  }

  const std::vector<std::uint32_t> bitcmp_program = {
      MakeSopc(*s_bitcmp0_b32_opcode, 0, 129),   // s_bitcmp0_b32 s0, 1
      MakeSop1(0, 10, 255), 111u,                // s_mov_b32 s10, 111
      MakeSopc(*s_bitcmp1_b32_opcode, 0, 129),   // s_bitcmp1_b32 s0, 1
      MakeSopp(4, 1),                            // s_cbranch_scc0 +1
      MakeSop1(0, 11, 255), 222u,                // s_mov_b32 s11, 222
      MakeSopc(*s_bitcmp1_b64_opcode, 2, 191),   // s_bitcmp1_b64 s[2:3], 63
      MakeSopp(4, 1),                            // s_cbranch_scc0 +1
      MakeSop1(0, 12, 255), 333u,                // s_mov_b32 s12, 333
      MakeSopp(1),                               // s_endpgm
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(bitcmp_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 9, "expected decoded bitcmp program size") ||
      !Expect(decoded_program[0].opcode == "S_BITCMP0_B32",
              "expected S_BITCMP0_B32 decode") ||
      !Expect(decoded_program[2].opcode == "S_BITCMP1_B32",
              "expected S_BITCMP1_B32 decode") ||
      !Expect(decoded_program[5].opcode == "S_BITCMP1_B64",
              "expected S_BITCMP1_B64 decode") ||
      !Expect(decoded_program[5].operands[0].kind == OperandKind::kSgpr,
              "expected bitcmp1_b64 scalar-pair base decode") ||
      !Expect(decoded_program[5].operands[0].index == 2,
              "expected bitcmp1_b64 scalar-pair base index") ||
      !Expect(decoded_program[5].operands[1].kind == OperandKind::kImm32,
              "expected bitcmp1_b64 lane immediate decode") ||
      !Expect(decoded_program[5].operands[1].imm32 == 63u,
              "expected bitcmp1_b64 lane immediate value")) {
    return 1;
  }

  WaveExecutionState bitcmp_state;
  bitcmp_state.sgprs[0] = 0x8u;
  bitcmp_state.sgprs[2] = 0u;
  bitcmp_state.sgprs[3] = 0x80000000u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &bitcmp_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(bitcmp_state.halted,
              "expected decoded binary bitcmp program to halt") ||
      !Expect(bitcmp_state.sgprs[10] == 111u,
              "expected decoded binary bitcmp0 true-path move result") ||
      !Expect(bitcmp_state.sgprs[11] == 0u,
              "expected decoded binary bitcmp1 false-path branch to skip move") ||
      !Expect(bitcmp_state.sgprs[12] == 333u,
              "expected decoded binary bitcmp1_b64 true-path move result") ||
      !Expect(bitcmp_state.scc,
              "expected decoded binary final bitcmp SCC to be true")) {
    return 1;
  }

  const auto readlane_words = MakeVop3(*v_readlane_b32_opcode, 4, 266, 6);
  const auto writelane_words = MakeVop3(*v_writelane_b32_opcode, 12, 8, 9);
  const std::vector<std::uint32_t> lane_ops_program = {
      readlane_words[0], readlane_words[1],
      writelane_words[0], writelane_words[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(lane_ops_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 3, "expected decoded lane-ops program size") ||
      !Expect(decoded_program[0].opcode == "V_READLANE_B32",
              "expected V_READLANE_B32 decode") ||
      !Expect(decoded_program[1].opcode == "V_WRITELANE_B32",
              "expected V_WRITELANE_B32 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kSgpr,
              "expected readlane scalar destination decode") ||
      !Expect(decoded_program[0].operands[0].index == 4,
              "expected readlane scalar destination index") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
              "expected readlane vector source decode") ||
      !Expect(decoded_program[0].operands[1].index == 10,
              "expected readlane vector source index") ||
      !Expect(decoded_program[0].operands[2].kind == OperandKind::kSgpr,
              "expected readlane lane-select decode") ||
      !Expect(decoded_program[0].operands[2].index == 6,
              "expected readlane lane-select index") ||
      !Expect(decoded_program[1].operands[0].kind == OperandKind::kVgpr,
              "expected writelane vector destination decode") ||
      !Expect(decoded_program[1].operands[0].index == 12,
              "expected writelane vector destination index") ||
      !Expect(decoded_program[1].operands[1].kind == OperandKind::kSgpr,
              "expected writelane scalar source decode") ||
      !Expect(decoded_program[1].operands[1].index == 8,
              "expected writelane scalar source index") ||
      !Expect(decoded_program[1].operands[2].kind == OperandKind::kSgpr,
              "expected writelane lane-select decode") ||
      !Expect(decoded_program[1].operands[2].index == 9,
              "expected writelane lane-select index")) {
    return 1;
  }

  WaveExecutionState lane_ops_state;
  lane_ops_state.exec_mask = 0x1ULL;
  lane_ops_state.sgprs[6] = 5u;
  lane_ops_state.sgprs[8] = 0xfeedfaceu;
  lane_ops_state.sgprs[9] = 7u;
  lane_ops_state.vgprs[10][5] = 0x12345678u;
  lane_ops_state.vgprs[12][6] = 0xaaaaaaaau;
  lane_ops_state.vgprs[12][7] = 0xbbbbbbbbu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &lane_ops_state,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(lane_ops_state.halted,
              "expected decoded binary lane-ops program to halt") ||
      !Expect(lane_ops_state.sgprs[4] == 0x12345678u,
              "expected decoded binary v_readlane_b32 result") ||
      !Expect(lane_ops_state.vgprs[12][6] == 0xaaaaaaaau,
              "expected decoded binary v_writelane_b32 to preserve neighboring lane") ||
      !Expect(lane_ops_state.vgprs[12][7] == 0xfeedfaceu,
              "expected decoded binary v_writelane_b32 result")) {
    return 1;
  }

  const auto load_word = MakeSmem(0, 4, 0, true, 0);
  const auto loadx2_word = MakeSmem(1, 6, 0, true, 4);
  const auto store_word = MakeSmem(16, 4, 0, false, 2, true);
  const std::vector<std::uint32_t> smem_program = {
      load_word[0], load_word[1],
      loadx2_word[0], loadx2_word[1],
      store_word[0], store_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(smem_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4, "expected decoded smem program size") ||
      !Expect(decoded_program[0].opcode == "S_LOAD_DWORD", "expected smem load decode") ||
      !Expect(decoded_program[1].opcode == "S_LOAD_DWORDX2",
              "expected smem loadx2 decode") ||
      !Expect(decoded_program[2].opcode == "S_STORE_DWORD",
              "expected smem store decode") ||
      !Expect(decoded_program[2].operands[2].kind == OperandKind::kSgpr,
              "expected soffset register decode") ||
      !Expect(decoded_program[2].operands[2].index == 2,
              "expected soffset sgpr index")) {
    return 1;
  }

  LinearExecutionMemory memory(0x400, 0);
  if (!Expect(memory.WriteU32(0x100, 0x11223344u), "expected memory seed write") ||
      !Expect(memory.WriteU32(0x104, 0x55667788u), "expected memory seed write") ||
      !Expect(memory.WriteU32(0x108, 0x99aabbccu), "expected memory seed write")) {
    return 1;
  }
  WaveExecutionState smem_state;
  smem_state.sgprs[0] = 0x100;
  smem_state.sgprs[1] = 0;
  smem_state.sgprs[2] = 0x10;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &smem_state, &memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(smem_state.sgprs[4] == 0x11223344u,
              "expected decoded smem load result") ||
      !Expect(smem_state.sgprs[6] == 0x55667788u,
              "expected decoded smem loadx2 low result") ||
      !Expect(smem_state.sgprs[7] == 0x99aabbccu,
              "expected decoded smem loadx2 high result")) {
    return 1;
  }

  std::uint32_t stored_value = 0;
  stored_value = 0;
  if (!Expect(memory.ReadU32(0x110, &stored_value), "expected decoded store read") ||
      !Expect(stored_value == 0x11223344u, "expected decoded smem store result")) {
    return 1;
  }

  const auto ds_nop_opcode = FindDefaultEncodingOpcode("DS_NOP", "ENC_DS");
  const auto ds_write_opcode =
      FindDefaultEncodingOpcode("DS_WRITE_B32", "ENC_DS");
  const auto ds_add_opcode =
      FindDefaultEncodingOpcode("DS_ADD_U32", "ENC_DS");
  const auto ds_read_opcode =
      FindDefaultEncodingOpcode("DS_READ_B32", "ENC_DS");
  if (!Expect(ds_nop_opcode.has_value(), "expected ds nop opcode lookup") ||
      !Expect(ds_write_opcode.has_value(), "expected ds write opcode lookup") ||
      !Expect(ds_add_opcode.has_value(), "expected ds add opcode lookup") ||
      !Expect(ds_read_opcode.has_value(), "expected ds read opcode lookup")) {
    return 1;
  }

  const auto ds_nop_word = MakeDs(*ds_nop_opcode, 0, 0, 0, 0, 0);
  const auto ds_write_word = MakeDs(*ds_write_opcode, 0, 0, 1, 0, 0);
  const auto ds_add_word = MakeDs(*ds_add_opcode, 0, 0, 2, 0, 0);
  const auto ds_read_word = MakeDs(*ds_read_opcode, 3, 0, 0, 0, 0);
  const std::vector<std::uint32_t> ds_program = {
      ds_nop_word[0],   ds_nop_word[1],
      ds_write_word[0], ds_write_word[1],
      ds_add_word[0], ds_add_word[1],
      ds_read_word[0], ds_read_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(ds_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 5, "expected decoded ds program size") ||
      !Expect(decoded_program[0].opcode == "DS_NOP",
              "expected ds nop decode") ||
      !Expect(decoded_program[1].opcode == "DS_WRITE_B32",
              "expected ds write decode") ||
      !Expect(decoded_program[2].opcode == "DS_ADD_U32",
              "expected ds add decode") ||
      !Expect(decoded_program[3].opcode == "DS_READ_B32",
              "expected ds read decode") ||
      !Expect(decoded_program[3].operands[0].kind == OperandKind::kVgpr,
              "expected ds read destination decode") ||
      !Expect(decoded_program[3].operands[0].index == 3,
              "expected ds read destination index")) {
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &ds_state, &error_message),
              error_message.c_str()) ||
      !Expect(ds_state.vgprs[3][0] == 11u,
              "expected decoded ds lane 0 result") ||
      !Expect(ds_state.vgprs[3][1] == 22u,
              "expected decoded ds lane 1 result") ||
      !Expect(ds_state.vgprs[3][2] == 0xdeadbeefu,
              "expected decoded ds inactive lane result") ||
      !Expect(ds_state.vgprs[3][3] == 44u,
              "expected decoded ds lane 3 result")) {
    return 1;
  }

  const std::array<std::string_view, 18> kExtendedDsOpcodes = {
      "DS_SUB_U32", "DS_RSUB_U32", "DS_INC_U32", "DS_DEC_U32",
      "DS_MIN_I32", "DS_MAX_I32",  "DS_MIN_U32", "DS_MAX_U32",
      "DS_AND_B32", "DS_OR_B32",   "DS_XOR_B32", "DS_ADD_F32",
      "DS_MIN_F32", "DS_MAX_F32",  "DS_WRITE_B8", "DS_WRITE_B16",
      "DS_PK_ADD_F16", "DS_PK_ADD_BF16",
  };
  for (std::string_view opcode_name : kExtendedDsOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected extended ds opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 0, 1, 0, 0);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
            error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected extended ds decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected extended ds opcode decode")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  {
    const auto ds_swizzle_opcode =
        FindDefaultEncodingOpcode("DS_SWIZZLE_B32", "ENC_DS");
    if (!Expect(ds_swizzle_opcode.has_value(),
                "expected ds swizzle opcode lookup")) {
      return 1;
    }

    const auto encoded = MakeDs(*ds_swizzle_opcode, 9, 1, 0, 0, 0xc1, 0x80);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds swizzle decode program size") ||
        !Expect(decoded_program[0].opcode == "DS_SWIZZLE_B32",
                "expected ds swizzle opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds swizzle operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds swizzle destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds swizzle destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds swizzle source kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds swizzle source index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds swizzle offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 0x80c1u,
                "expected ds swizzle combined offset value")) {
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPermuteOpcodes = {
      "DS_PERMUTE_B32",
      "DS_BPERMUTE_B32",
  };
  for (std::string_view opcode_name : kDsPermuteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds permute opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 2, 0, 0x34, 0x12);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds permute decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds permute opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected ds permute operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds permute destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds permute destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds permute address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds permute address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds permute data kind") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected ds permute data index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected ds permute offset kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 0x1234u,
                "expected ds permute combined offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairWriteOpcodes = {
      "DS_WRITE2_B32",
      "DS_WRITE2ST64_B32",
  };
  for (std::string_view opcode_name : kDsPairWriteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds pair-write opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 3, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds pair-write decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds pair-write opcode decode") ||
        !Expect(decoded_program[0].operand_count == 5,
                "expected ds pair-write operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds pair-write address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds pair-write address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds pair-write data0 kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds pair-write data0 index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds pair-write data1 kind") ||
        !Expect(decoded_program[0].operands[2].index == 3,
                "expected ds pair-write data1 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected ds pair-write offset0 kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 4u,
                "expected ds pair-write offset0 value") ||
        !Expect(decoded_program[0].operands[4].kind == OperandKind::kImm32,
                "expected ds pair-write offset1 kind") ||
        !Expect(decoded_program[0].operands[4].imm32 == 7u,
                "expected ds pair-write offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairReadOpcodes = {
      "DS_READ2_B32",
      "DS_READ2ST64_B32",
  };
  for (std::string_view opcode_name : kDsPairReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds pair-read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 0, 0, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds pair-read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds pair-read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected ds pair-read operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds pair-read destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds pair-read destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds pair-read address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds pair-read address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds pair-read offset0 kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 4u,
                "expected ds pair-read offset0 value") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected ds pair-read offset1 kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 7u,
                "expected ds pair-read offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsPairReturnOpcodes = {
      "DS_WRXCHG2_RTN_B32",
      "DS_WRXCHG2ST64_RTN_B32",
  };
  for (std::string_view opcode_name : kDsPairReturnOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds pair-return opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 2, 3, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds pair-return decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds pair-return opcode decode") ||
        !Expect(decoded_program[0].operand_count == 6,
                "expected ds pair-return operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds pair-return destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds pair-return destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds pair-return address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds pair-return address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds pair-return data0 kind") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected ds pair-return data0 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kVgpr,
                "expected ds pair-return data1 kind") ||
        !Expect(decoded_program[0].operands[3].index == 3,
                "expected ds pair-return data1 index") ||
        !Expect(decoded_program[0].operands[4].kind == OperandKind::kImm32,
                "expected ds pair-return offset0 kind") ||
        !Expect(decoded_program[0].operands[4].imm32 == 4u,
                "expected ds pair-return offset0 value") ||
        !Expect(decoded_program[0].operands[5].kind == OperandKind::kImm32,
                "expected ds pair-return offset1 kind") ||
        !Expect(decoded_program[0].operands[5].imm32 == 7u,
                "expected ds pair-return offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsWideWriteOpcodes = {
      "DS_WRITE_B64",
  };
  for (std::string_view opcode_name : kDsWideWriteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds wide-write opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide-write decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide-write opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds wide-write operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wide-write address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds wide-write address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds wide-write data kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds wide-write data index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds wide-write offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds wide-write offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 1> kDsWideReadOpcodes = {
      "DS_READ_B64",
  };
  for (std::string_view opcode_name : kDsWideReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds wide-read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 0, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide-read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide-read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds wide-read operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wide-read destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds wide-read destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds wide-read address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds wide-read address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds wide-read offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds wide-read offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairWriteOpcodes = {
      "DS_WRITE2_B64",
      "DS_WRITE2ST64_B64",
  };
  for (std::string_view opcode_name : kDsWidePairWriteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds wide pair-write opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 4, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide pair-write decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide pair-write opcode decode") ||
        !Expect(decoded_program[0].operand_count == 5,
                "expected ds wide pair-write operand count") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds wide pair-write address index") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds wide pair-write data0 index") ||
        !Expect(decoded_program[0].operands[2].index == 4,
                "expected ds wide pair-write data1 index") ||
        !Expect(decoded_program[0].operands[3].imm32 == 4u,
                "expected ds wide pair-write offset0 value") ||
        !Expect(decoded_program[0].operands[4].imm32 == 7u,
                "expected ds wide pair-write offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairReadOpcodes = {
      "DS_READ2_B64",
      "DS_READ2ST64_B64",
  };
  for (std::string_view opcode_name : kDsWidePairReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds wide pair-read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 0, 0, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide pair-read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide pair-read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected ds wide pair-read operand count") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds wide pair-read destination index") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds wide pair-read address index") ||
        !Expect(decoded_program[0].operands[2].imm32 == 4u,
                "expected ds wide pair-read offset0 value") ||
        !Expect(decoded_program[0].operands[3].imm32 == 7u,
                "expected ds wide pair-read offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsMultiDwordWriteOpcodes = {
      "DS_WRITE_B96",
      "DS_WRITE_B128",
  };
  for (std::string_view opcode_name : kDsMultiDwordWriteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds multi-dword write opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds multi-dword write decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds multi-dword write opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds multi-dword write operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds multi-dword write address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds multi-dword write address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds multi-dword write data kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds multi-dword write data index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds multi-dword write offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds multi-dword write offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsMultiDwordReadOpcodes = {
      "DS_READ_B96",
      "DS_READ_B128",
  };
  for (std::string_view opcode_name : kDsMultiDwordReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds multi-dword read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 0, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds multi-dword read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds multi-dword read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds multi-dword read operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds multi-dword read destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds multi-dword read destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds multi-dword read address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds multi-dword read address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds multi-dword read offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds multi-dword read offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsWidePairReturnOpcodes = {
      "DS_WRXCHG2_RTN_B64",
      "DS_WRXCHG2ST64_RTN_B64",
  };
  for (std::string_view opcode_name : kDsWidePairReturnOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds wide pair-return opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 8, 1, 2, 4, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide pair-return decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide pair-return opcode decode") ||
        !Expect(decoded_program[0].operand_count == 6,
                "expected ds wide pair-return operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wide pair-return destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 8,
                "expected ds wide pair-return destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds wide pair-return address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds wide pair-return address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds wide pair-return data0 kind") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected ds wide pair-return data0 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kVgpr,
                "expected ds wide pair-return data1 kind") ||
        !Expect(decoded_program[0].operands[3].index == 4,
                "expected ds wide pair-return data1 index") ||
        !Expect(decoded_program[0].operands[4].kind == OperandKind::kImm32,
                "expected ds wide pair-return offset0 kind") ||
        !Expect(decoded_program[0].operands[4].imm32 == 4u,
                "expected ds wide pair-return offset0 value") ||
        !Expect(decoded_program[0].operands[5].kind == OperandKind::kImm32,
                "expected ds wide pair-return offset1 kind") ||
        !Expect(decoded_program[0].operands[5].imm32 == 7u,
                "expected ds wide pair-return offset1 value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 15> kDsWideUpdateOpcodes = {
      "DS_ADD_U64", "DS_SUB_U64", "DS_RSUB_U64", "DS_INC_U64", "DS_DEC_U64",
      "DS_MIN_I64", "DS_MAX_I64", "DS_MIN_U64", "DS_MAX_U64", "DS_AND_B64",
      "DS_OR_B64",  "DS_XOR_B64", "DS_ADD_F64",  "DS_MIN_F64", "DS_MAX_F64",
  };
  for (std::string_view opcode_name : kDsWideUpdateOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds wide update opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide update decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide update opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds wide update operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wide update address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds wide update address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds wide update data kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds wide update data index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds wide update offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds wide update offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 4> kDsNarrowReadOpcodes = {
      "DS_READ_I8",
      "DS_READ_U8",
      "DS_READ_I16",
      "DS_READ_U16",
  };
  for (std::string_view opcode_name : kDsNarrowReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds narrow-read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 0, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds narrow-read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds narrow-read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds narrow-read operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds narrow-read destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds narrow-read destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds narrow-read address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds narrow-read address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds narrow-read offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds narrow-read offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 2> kDsD16WriteOpcodes = {
      "DS_WRITE_B8_D16_HI",
      "DS_WRITE_B16_D16_HI",
  };
  for (std::string_view opcode_name : kDsD16WriteOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds d16 write opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds d16 write decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds d16 write opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds d16 write operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds d16 write address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds d16 write address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds d16 write data kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds d16 write data index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds d16 write offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds d16 write offset value")) {
      std::cerr << opcode_name << '\n';
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
  for (std::string_view opcode_name : kDsD16ReadOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds d16 read opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 0, 0, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds d16 read decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds d16 read opcode decode") ||
        !Expect(decoded_program[0].operand_count == 3,
                "expected ds d16 read operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds d16 read destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds d16 read destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds d16 read address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds d16 read address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kImm32,
                "expected ds d16 read offset kind") ||
        !Expect(decoded_program[0].operands[2].imm32 == 6u,
                "expected ds d16 read offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const auto ds_write_addtid_opcode =
      FindDefaultEncodingOpcode("DS_WRITE_ADDTID_B32", "ENC_DS");
  if (!Expect(ds_write_addtid_opcode.has_value(),
              "expected ds write_addtid opcode lookup")) {
    return 1;
  }
  {
    const auto encoded = MakeDs(*ds_write_addtid_opcode, 0, 13, 2, 0, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds write_addtid decode program size") ||
        !Expect(decoded_program[0].opcode == "DS_WRITE_ADDTID_B32",
                "expected ds write_addtid opcode decode") ||
        !Expect(decoded_program[0].operand_count == 2,
                "expected ds write_addtid operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds write_addtid data kind") ||
        !Expect(decoded_program[0].operands[0].index == 2,
                "expected ds write_addtid data index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kImm32,
                "expected ds write_addtid offset kind") ||
        !Expect(decoded_program[0].operands[1].imm32 == 7u,
                "expected ds write_addtid offset value")) {
      return 1;
    }
  }

  const auto ds_read_addtid_opcode =
      FindDefaultEncodingOpcode("DS_READ_ADDTID_B32", "ENC_DS");
  if (!Expect(ds_read_addtid_opcode.has_value(),
              "expected ds read_addtid opcode lookup")) {
    return 1;
  }
  {
    const auto encoded = MakeDs(*ds_read_addtid_opcode, 9, 13, 0, 0, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds read_addtid decode program size") ||
        !Expect(decoded_program[0].opcode == "DS_READ_ADDTID_B32",
                "expected ds read_addtid opcode decode") ||
        !Expect(decoded_program[0].operand_count == 2,
                "expected ds read_addtid operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds read_addtid destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds read_addtid destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kImm32,
                "expected ds read_addtid offset kind") ||
        !Expect(decoded_program[0].operands[1].imm32 == 7u,
                "expected ds read_addtid offset value")) {
      return 1;
    }
  }

  struct DsWaveCounterDecodeCase {
    std::string_view opcode_name;
    std::uint32_t destination_index;
    std::uint32_t offset0;
    std::uint32_t offset1;
    std::uint32_t expected_offset;
  };
  const std::array<DsWaveCounterDecodeCase, 2> kDsWaveCounterDecodeCases = {{
      {"DS_CONSUME", 9u, 0x20u, 0x01u, 0x0120u},
      {"DS_APPEND", 11u, 0x50u, 0x04u, 0x0450u},
  }};
  for (const DsWaveCounterDecodeCase& test_case : kDsWaveCounterDecodeCases) {
    const auto opcode = FindDefaultEncodingOpcode(test_case.opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds wave-counter opcode lookup")) {
      std::cerr << test_case.opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, test_case.destination_index, 0, 0, 0,
                                test_case.offset0, test_case.offset1);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wave-counter decode program size") ||
        !Expect(decoded_program[0].opcode == test_case.opcode_name,
                "expected ds wave-counter opcode decode") ||
        !Expect(decoded_program[0].operand_count == 2,
                "expected ds wave-counter operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wave-counter destination kind") ||
        !Expect(decoded_program[0].operands[0].index ==
                    test_case.destination_index,
                "expected ds wave-counter destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kImm32,
                "expected ds wave-counter offset kind") ||
        !Expect(decoded_program[0].operands[1].imm32 ==
                    test_case.expected_offset,
                "expected ds wave-counter offset value")) {
      std::cerr << test_case.opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kDsDualDataOpcodes = {
      "DS_MSKOR_B32",
      "DS_CMPST_B32",
      "DS_CMPST_F32",
  };
  for (std::string_view opcode_name : kDsDualDataOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected ds dual-data opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 3, 6);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds dual-data decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds dual-data opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected ds dual-data operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds dual-data address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds dual-data address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds dual-data data0 kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds dual-data data0 index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds dual-data data1 kind") ||
        !Expect(decoded_program[0].operands[2].index == 3,
                "expected ds dual-data data1 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected ds dual-data offset kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 6u,
                "expected ds dual-data offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kDsWideDualDataOpcodes = {
      "DS_MSKOR_B64",
      "DS_CMPST_B64",
      "DS_CMPST_F64",
  };
  for (std::string_view opcode_name : kDsWideDualDataOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds wide dual-data opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 0, 1, 2, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds wide dual-data decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds wide dual-data opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected ds wide dual-data operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds wide dual-data address kind") ||
        !Expect(decoded_program[0].operands[0].index == 1,
                "expected ds wide dual-data address index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds wide dual-data data0 kind") ||
        !Expect(decoded_program[0].operands[1].index == 2,
                "expected ds wide dual-data data0 index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds wide dual-data data1 kind") ||
        !Expect(decoded_program[0].operands[2].index == 4,
                "expected ds wide dual-data data1 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected ds wide dual-data offset kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 7u,
                "expected ds wide dual-data offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 4> kDsDualDataReturnOpcodes = {
      "DS_MSKOR_RTN_B32",
      "DS_CMPST_RTN_B32",
      "DS_CMPST_RTN_F32",
      "DS_WRAP_RTN_B32",
  };
  for (std::string_view opcode_name : kDsDualDataReturnOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected ds dual-data return opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 2, 3, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected ds dual-data return decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected ds dual-data return opcode decode") ||
        !Expect(decoded_program[0].operand_count == 5,
                "expected ds dual-data return operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected ds dual-data return destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected ds dual-data return destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected ds dual-data return address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected ds dual-data return address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected ds dual-data return data0 kind") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected ds dual-data return data0 index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kVgpr,
                "expected ds dual-data return data1 kind") ||
        !Expect(decoded_program[0].operands[3].index == 3,
                "expected ds dual-data return data1 index") ||
        !Expect(decoded_program[0].operands[4].kind == OperandKind::kImm32,
                "expected ds dual-data return offset kind") ||
        !Expect(decoded_program[0].operands[4].imm32 == 7u,
                "expected ds dual-data return offset value")) {
      std::cerr << opcode_name << '\n';
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
  for (std::string_view opcode_name : kReturningDsOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(), "expected returning ds opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 2, 0, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected returning ds decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected returning ds opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected returning ds operand count") ||
        !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
                "expected returning ds destination kind") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected returning ds destination index") ||
        !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
                "expected returning ds address kind") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected returning ds address index") ||
        !Expect(decoded_program[0].operands[2].kind == OperandKind::kVgpr,
                "expected returning ds data kind") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected returning ds data index") ||
        !Expect(decoded_program[0].operands[3].kind == OperandKind::kImm32,
                "expected returning ds offset kind") ||
        !Expect(decoded_program[0].operands[3].imm32 == 7u,
                "expected returning ds offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const std::array<std::string_view, 3> kWideReturningDsDualDataOpcodes = {
      "DS_MSKOR_RTN_B64",
      "DS_CMPST_RTN_B64",
      "DS_CMPST_RTN_F64",
  };
  for (std::string_view opcode_name : kWideReturningDsDualDataOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected wide returning ds dual-data opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 2, 4, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected wide returning ds dual-data decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected wide returning ds dual-data opcode decode") ||
        !Expect(decoded_program[0].operand_count == 5,
                "expected wide returning ds dual-data operand count") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected wide returning ds dual-data destination index") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected wide returning ds dual-data address index") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected wide returning ds dual-data data0 index") ||
        !Expect(decoded_program[0].operands[3].index == 4,
                "expected wide returning ds dual-data data1 index") ||
        !Expect(decoded_program[0].operands[4].imm32 == 7u,
                "expected wide returning ds dual-data offset value")) {
      std::cerr << opcode_name << '\n';
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
  for (std::string_view opcode_name : kWideReturningDsOpcodes) {
    const auto opcode = FindDefaultEncodingOpcode(opcode_name, "ENC_DS");
    if (!Expect(opcode.has_value(),
                "expected wide returning ds opcode lookup")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }

    const auto encoded = MakeDs(*opcode, 9, 1, 2, 0, 7);
    const std::vector<std::uint32_t> encoded_program = {
        encoded[0], encoded[1], MakeSopp(1),
    };
    decoded_program.clear();
    if (!Expect(decoder.DecodeProgram(encoded_program, &decoded_program,
                                      &error_message),
                error_message.c_str()) ||
        !Expect(decoded_program.size() == 2,
                "expected wide returning ds decode program size") ||
        !Expect(decoded_program[0].opcode == opcode_name,
                "expected wide returning ds opcode decode") ||
        !Expect(decoded_program[0].operand_count == 4,
                "expected wide returning ds operand count") ||
        !Expect(decoded_program[0].operands[0].index == 9,
                "expected wide returning ds destination index") ||
        !Expect(decoded_program[0].operands[1].index == 1,
                "expected wide returning ds address index") ||
        !Expect(decoded_program[0].operands[2].index == 2,
                "expected wide returning ds data index") ||
        !Expect(decoded_program[0].operands[3].imm32 == 7u,
                "expected wide returning ds offset value")) {
      std::cerr << opcode_name << '\n';
      return 1;
    }
  }

  const auto flat_load_word = MakeFlat(20, 10, 0, 0, 4);
  const auto flat_store_word = MakeFlat(28, 0, 0, 2, 0);
  const auto global_load_word = MakeGlobal(20, 11, 4, 0, 0, -4);
  const auto global_store_word = MakeGlobal(28, 0, 4, 3, 0, 4);
  const std::vector<std::uint32_t> flat_global_program = {
      flat_load_word[0],   flat_load_word[1],
      flat_store_word[0],  flat_store_word[1],
      global_load_word[0], global_load_word[1],
      global_store_word[0], global_store_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(flat_global_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 5,
              "expected decoded flat/global program size") ||
      !Expect(decoded_program[0].opcode == "FLAT_LOAD_DWORD",
              "expected flat load decode") ||
      !Expect(decoded_program[1].opcode == "FLAT_STORE_DWORD",
              "expected flat store decode") ||
      !Expect(decoded_program[2].opcode == "GLOBAL_LOAD_DWORD",
              "expected global load decode") ||
      !Expect(decoded_program[3].opcode == "GLOBAL_STORE_DWORD",
              "expected global store decode") ||
      !Expect(decoded_program[0].operands[1].kind == OperandKind::kVgpr,
              "expected flat address vgpr decode") ||
      !Expect(decoded_program[0].operands[2].imm32 == 4u,
              "expected flat offset decode") ||
      !Expect(decoded_program[2].operands[2].kind == OperandKind::kSgpr,
              "expected global scalar base decode") ||
      !Expect(decoded_program[2].operands[3].imm32 ==
                  static_cast<std::uint32_t>(-4),
              "expected sign-extended global offset decode")) {
    return 1;
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

  WaveExecutionState vector_state;
  vector_state.exec_mask = 0b1011ULL;
  vector_state.sgprs[0] = 0x200;
  vector_state.sgprs[1] = 0x0;
  vector_state.vgprs[0][0] = 0x180;
  vector_state.vgprs[0][1] = 0x184;
  vector_state.vgprs[0][3] = 0x18c;
  vector_state.vgprs[2][0] = 0xdead0001u;
  vector_state.vgprs[2][1] = 0xdead0002u;
  vector_state.vgprs[2][3] = 0xdead0004u;
  vector_state.vgprs[3][0] = 0xbeef0011u;
  vector_state.vgprs[3][1] = 0xbeef0022u;
  vector_state.vgprs[3][3] = 0xbeef0044u;
  vector_state.vgprs[4][0] = 0x24;
  vector_state.vgprs[4][1] = 0x28;
  vector_state.vgprs[4][3] = 0x30;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &vector_state, &vector_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(vector_state.vgprs[10][0] == 0x11111111u,
              "expected decoded flat load lane 0 result") ||
      !Expect(vector_state.vgprs[10][1] == 0x22222222u,
              "expected decoded flat load lane 1 result") ||
      !Expect(vector_state.vgprs[10][2] == 0x0u,
              "expected decoded flat load inactive lane") ||
      !Expect(vector_state.vgprs[10][3] == 0x33333333u,
              "expected decoded flat load lane 3 result") ||
      !Expect(vector_state.vgprs[11][0] == 0xaaaabbbbu,
              "expected decoded global load lane 0 result") ||
      !Expect(vector_state.vgprs[11][1] == 0xccccddddu,
              "expected decoded global load lane 1 result") ||
      !Expect(vector_state.vgprs[11][3] == 0x12345678u,
              "expected decoded global load lane 3 result")) {
    return 1;
  }

  std::uint32_t vector_store_value = 0;
  if (!Expect(vector_memory.ReadU32(0x180, &vector_store_value),
              "expected decoded flat store lane 0 read") ||
      !Expect(vector_store_value == 0xdead0001u,
              "expected decoded flat store lane 0 result") ||
      !Expect(vector_memory.ReadU32(0x228, &vector_store_value),
              "expected decoded global store lane 0 read") ||
      !Expect(vector_store_value == 0xbeef0011u,
              "expected decoded global store lane 0 result") ||
      !Expect(vector_memory.ReadU32(0x22c, &vector_store_value),
              "expected decoded global store lane 1 read") ||
      !Expect(vector_store_value == 0xbeef0022u,
              "expected decoded global store lane 1 result") ||
      !Expect(vector_memory.ReadU32(0x234, &vector_store_value),
              "expected decoded global store lane 3 read") ||
      !Expect(vector_store_value == 0xbeef0044u,
              "expected decoded global store lane 3 result")) {
    return 1;
  }

  const auto flat_load_ubyte_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_UBYTE", "ENC_FLAT");
  const auto flat_load_sbyte_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_SBYTE", "ENC_FLAT");
  const auto flat_load_ushort_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_USHORT", "ENC_FLAT");
  const auto flat_load_sshort_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_SSHORT", "ENC_FLAT");
  const auto flat_store_byte_opcode =
      FindDefaultEncodingOpcode("FLAT_STORE_BYTE", "ENC_FLAT");
  const auto flat_store_short_opcode =
      FindDefaultEncodingOpcode("FLAT_STORE_SHORT", "ENC_FLAT");
  const auto global_load_ubyte_opcode =
      FindDefaultEncodingOpcode("GLOBAL_LOAD_UBYTE", "ENC_FLAT_GLBL");
  const auto global_load_sbyte_opcode =
      FindDefaultEncodingOpcode("GLOBAL_LOAD_SBYTE", "ENC_FLAT_GLBL");
  const auto global_load_ushort_opcode =
      FindDefaultEncodingOpcode("GLOBAL_LOAD_USHORT", "ENC_FLAT_GLBL");
  const auto global_load_sshort_opcode =
      FindDefaultEncodingOpcode("GLOBAL_LOAD_SSHORT", "ENC_FLAT_GLBL");
  const auto global_store_byte_opcode =
      FindDefaultEncodingOpcode("GLOBAL_STORE_BYTE", "ENC_FLAT_GLBL");
  const auto global_store_short_opcode =
      FindDefaultEncodingOpcode("GLOBAL_STORE_SHORT", "ENC_FLAT_GLBL");
  if (!Expect(flat_load_ubyte_opcode.has_value(),
              "expected flat load ubyte opcode lookup") ||
      !Expect(flat_load_sbyte_opcode.has_value(),
              "expected flat load sbyte opcode lookup") ||
      !Expect(flat_load_ushort_opcode.has_value(),
              "expected flat load ushort opcode lookup") ||
      !Expect(flat_load_sshort_opcode.has_value(),
              "expected flat load sshort opcode lookup") ||
      !Expect(flat_store_byte_opcode.has_value(),
              "expected flat store byte opcode lookup") ||
      !Expect(flat_store_short_opcode.has_value(),
              "expected flat store short opcode lookup") ||
      !Expect(global_load_ubyte_opcode.has_value(),
              "expected global load ubyte opcode lookup") ||
      !Expect(global_load_sbyte_opcode.has_value(),
              "expected global load sbyte opcode lookup") ||
      !Expect(global_load_ushort_opcode.has_value(),
              "expected global load ushort opcode lookup") ||
      !Expect(global_load_sshort_opcode.has_value(),
              "expected global load sshort opcode lookup") ||
      !Expect(global_store_byte_opcode.has_value(),
              "expected global store byte opcode lookup") ||
      !Expect(global_store_short_opcode.has_value(),
              "expected global store short opcode lookup")) {
    return 1;
  }

  const auto flat_load_ubyte_word = MakeFlat(*flat_load_ubyte_opcode, 30, 0, 0, 0);
  const auto flat_load_sbyte_word = MakeFlat(*flat_load_sbyte_opcode, 31, 2, 0, 0);
  const auto flat_load_ushort_word = MakeFlat(*flat_load_ushort_opcode, 32, 4, 0, 0);
  const auto flat_load_sshort_word = MakeFlat(*flat_load_sshort_opcode, 33, 6, 0, 0);
  const auto global_load_ubyte_word =
      MakeGlobal(*global_load_ubyte_opcode, 34, 8, 0, 0, 0);
  const auto global_load_sbyte_word =
      MakeGlobal(*global_load_sbyte_opcode, 35, 10, 0, 0, 0);
  const auto global_load_ushort_word =
      MakeGlobal(*global_load_ushort_opcode, 36, 12, 0, 0, 0);
  const auto global_load_sshort_word =
      MakeGlobal(*global_load_sshort_opcode, 37, 14, 0, 0, 0);
  const auto flat_store_byte_word =
      MakeFlat(*flat_store_byte_opcode, 0, 16, 40, 0);
  const auto flat_store_short_word =
      MakeFlat(*flat_store_short_opcode, 0, 18, 41, 0);
  const auto global_store_byte_word =
      MakeGlobal(*global_store_byte_opcode, 0, 20, 42, 0, 0);
  const auto global_store_short_word =
      MakeGlobal(*global_store_short_opcode, 0, 22, 43, 0, 0);
  const std::vector<std::uint32_t> subword_program = {
      flat_load_ubyte_word[0],  flat_load_ubyte_word[1],
      flat_load_sbyte_word[0],  flat_load_sbyte_word[1],
      flat_load_ushort_word[0], flat_load_ushort_word[1],
      flat_load_sshort_word[0], flat_load_sshort_word[1],
      global_load_ubyte_word[0], global_load_ubyte_word[1],
      global_load_sbyte_word[0], global_load_sbyte_word[1],
      global_load_ushort_word[0], global_load_ushort_word[1],
      global_load_sshort_word[0], global_load_sshort_word[1],
      flat_store_byte_word[0],  flat_store_byte_word[1],
      flat_store_short_word[0], flat_store_short_word[1],
      global_store_byte_word[0], global_store_byte_word[1],
      global_store_short_word[0], global_store_short_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(subword_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 13,
              "expected decoded subword program size") ||
      !Expect(decoded_program[0].opcode == "FLAT_LOAD_UBYTE",
              "expected flat load ubyte decode") ||
      !Expect(decoded_program[8].opcode == "FLAT_STORE_BYTE",
              "expected flat store byte decode") ||
      !Expect(decoded_program[10].opcode == "GLOBAL_STORE_BYTE",
              "expected global store byte decode")) {
    return 1;
  }

  LinearExecutionMemory subword_memory(0x2000, 0);
  if (!Expect(WriteU8(&subword_memory, 0x600, 0x7au),
              "expected decoded flat ubyte seed write") ||
      !Expect(WriteU8(&subword_memory, 0x610, 0x80u),
              "expected decoded flat sbyte seed write") ||
      !Expect(WriteU16(&subword_memory, 0x620, 0x1234u),
              "expected decoded flat ushort seed write") ||
      !Expect(WriteU16(&subword_memory, 0x630, 0x8001u),
              "expected decoded flat sshort seed write") ||
      !Expect(WriteU8(&subword_memory, 0xa20, 0xa5u),
              "expected decoded global ubyte seed write") ||
      !Expect(WriteU8(&subword_memory, 0xa30, 0xf0u),
              "expected decoded global sbyte seed write") ||
      !Expect(WriteU16(&subword_memory, 0xa40, 0x5678u),
              "expected decoded global ushort seed write") ||
      !Expect(WriteU16(&subword_memory, 0xa50, 0x8002u),
              "expected decoded global sshort seed write")) {
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &subword_state,
                                         &subword_memory, &error_message),
              error_message.c_str()) ||
      !Expect(subword_state.vgprs[30][0] == 0x7au,
              "expected decoded flat ubyte load result") ||
      !Expect(subword_state.vgprs[31][0] == 0xffffff80u,
              "expected decoded flat sbyte load result") ||
      !Expect(subword_state.vgprs[32][0] == 0x1234u,
              "expected decoded flat ushort load result") ||
      !Expect(subword_state.vgprs[33][0] == 0xffff8001u,
              "expected decoded flat sshort load result") ||
      !Expect(subword_state.vgprs[34][0] == 0xa5u,
              "expected decoded global ubyte load result") ||
      !Expect(subword_state.vgprs[35][0] == 0xfffffff0u,
              "expected decoded global sbyte load result") ||
      !Expect(subword_state.vgprs[36][0] == 0x5678u,
              "expected decoded global ushort load result") ||
      !Expect(subword_state.vgprs[37][0] == 0xffff8002u,
              "expected decoded global sshort load result")) {
    return 1;
  }

  std::uint8_t stored_byte = 0;
  std::uint16_t stored_short = 0;
  if (!Expect(ReadU8(subword_memory, 0x640, &stored_byte),
              "expected decoded flat byte store read") ||
      !Expect(stored_byte == 0xabu,
              "expected decoded flat byte store result") ||
      !Expect(ReadU16(subword_memory, 0x650, &stored_short),
              "expected decoded flat short store read") ||
      !Expect(stored_short == 0xcdefu,
              "expected decoded flat short store result") ||
      !Expect(ReadU8(subword_memory, 0xa60, &stored_byte),
              "expected decoded global byte store read") ||
      !Expect(stored_byte == 0x88u,
              "expected decoded global byte store result") ||
      !Expect(ReadU16(subword_memory, 0xa70, &stored_short),
              "expected decoded global short store read") ||
      !Expect(stored_short == 0xc3d4u,
              "expected decoded global short store result")) {
    return 1;
  }

  const auto global_load_x2_word = MakeGlobal(21, 30, 6, 0, 0, 0);
  const auto global_store_x2_word = MakeGlobal(29, 0, 8, 20, 0, 0);
  const std::vector<std::uint32_t> global_x2_program = {
      global_load_x2_word[0],  global_load_x2_word[1],
      global_store_x2_word[0], global_store_x2_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(global_x2_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 3,
              "expected decoded global x2 program size") ||
      !Expect(decoded_program[0].opcode == "GLOBAL_LOAD_DWORDX2",
              "expected global load x2 decode") ||
      !Expect(decoded_program[1].opcode == "GLOBAL_STORE_DWORDX2",
              "expected global store x2 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
              "expected global load x2 vdst decode") ||
      !Expect(decoded_program[0].operands[0].index == 30,
              "expected global load x2 vdst start index") ||
      !Expect(decoded_program[1].operands[1].kind == OperandKind::kVgpr,
              "expected global store x2 data decode") ||
      !Expect(decoded_program[1].operands[1].index == 20,
              "expected global store x2 data start index")) {
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
  global_x2_state.vgprs[8][0] = 0x80;
  global_x2_state.vgprs[8][1] = 0x88;
  global_x2_state.vgprs[8][3] = 0x98;
  global_x2_state.vgprs[20][0] = 0xaaaabbbb;
  global_x2_state.vgprs[20][1] = 0xccccdddd;
  global_x2_state.vgprs[20][3] = 0xeeeeffff;
  global_x2_state.vgprs[21][0] = 0x11112222;
  global_x2_state.vgprs[21][1] = 0x33334444;
  global_x2_state.vgprs[21][3] = 0x55556666;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &global_x2_state,
                                         &global_x2_memory, &error_message),
              error_message.c_str()) ||
      !Expect(global_x2_state.vgprs[30][0] == 0x01020304u,
              "expected decoded global x2 lane 0 low load result") ||
      !Expect(global_x2_state.vgprs[31][0] == 0x05060708u,
              "expected decoded global x2 lane 0 high load result") ||
      !Expect(global_x2_state.vgprs[30][1] == 0x11121314u,
              "expected decoded global x2 lane 1 low load result") ||
      !Expect(global_x2_state.vgprs[31][1] == 0x15161718u,
              "expected decoded global x2 lane 1 high load result") ||
      !Expect(global_x2_state.vgprs[30][2] == 0x0u,
              "expected decoded global x2 inactive lane low result") ||
      !Expect(global_x2_state.vgprs[31][2] == 0x0u,
              "expected decoded global x2 inactive lane high result") ||
      !Expect(global_x2_state.vgprs[30][3] == 0x21222324u,
              "expected decoded global x2 lane 3 low load result") ||
      !Expect(global_x2_state.vgprs[31][3] == 0x25262728u,
              "expected decoded global x2 lane 3 high load result")) {
    return 1;
  }

  std::uint32_t global_x2_store_value = 0;
  if (!Expect(global_x2_memory.ReadU32(0x380, &global_x2_store_value),
              "expected decoded global x2 lane 0 low store read") ||
      !Expect(global_x2_store_value == 0xaaaabbbbu,
              "expected decoded global x2 lane 0 low store result") ||
      !Expect(global_x2_memory.ReadU32(0x384, &global_x2_store_value),
              "expected decoded global x2 lane 0 high store read") ||
      !Expect(global_x2_store_value == 0x11112222u,
              "expected decoded global x2 lane 0 high store result") ||
      !Expect(global_x2_memory.ReadU32(0x388, &global_x2_store_value),
              "expected decoded global x2 lane 1 low store read") ||
      !Expect(global_x2_store_value == 0xccccddddu,
              "expected decoded global x2 lane 1 low store result") ||
      !Expect(global_x2_memory.ReadU32(0x38c, &global_x2_store_value),
              "expected decoded global x2 lane 1 high store read") ||
      !Expect(global_x2_store_value == 0x33334444u,
              "expected decoded global x2 lane 1 high store result") ||
      !Expect(global_x2_memory.ReadU32(0x398, &global_x2_store_value),
              "expected decoded global x2 lane 3 low store read") ||
      !Expect(global_x2_store_value == 0xeeeeffffu,
              "expected decoded global x2 lane 3 low store result") ||
      !Expect(global_x2_memory.ReadU32(0x39c, &global_x2_store_value),
              "expected decoded global x2 lane 3 high store read") ||
      !Expect(global_x2_store_value == 0x55556666u,
              "expected decoded global x2 lane 3 high store result")) {
    return 1;
  }

  const auto global_load_x4_word = MakeGlobal(23, 50, 10, 0, 0, 0);
  const auto global_store_x4_word = MakeGlobal(31, 0, 12, 40, 0, 0);
  const std::vector<std::uint32_t> global_x4_program = {
      global_load_x4_word[0],  global_load_x4_word[1],
      global_store_x4_word[0], global_store_x4_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(global_x4_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 3,
              "expected decoded global x4 program size") ||
      !Expect(decoded_program[0].opcode == "GLOBAL_LOAD_DWORDX4",
              "expected global load x4 decode") ||
      !Expect(decoded_program[1].opcode == "GLOBAL_STORE_DWORDX4",
              "expected global store x4 decode") ||
      !Expect(decoded_program[0].operands[0].kind == OperandKind::kVgpr,
              "expected global load x4 vdst decode") ||
      !Expect(decoded_program[0].operands[0].index == 50,
              "expected global load x4 vdst start index") ||
      !Expect(decoded_program[1].operands[1].kind == OperandKind::kVgpr,
              "expected global store x4 data decode") ||
      !Expect(decoded_program[1].operands[1].index == 40,
              "expected global store x4 data start index")) {
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
  global_x4_state.vgprs[12][0] = 0x80;
  global_x4_state.vgprs[12][1] = 0x90;
  global_x4_state.vgprs[12][3] = 0xb0;
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &global_x4_state,
                                         &global_x4_memory, &error_message),
              error_message.c_str()) ||
      !Expect(global_x4_state.vgprs[50][0] == 0x10111213u,
              "expected decoded global x4 lane 0 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][0] == 0x14151617u,
              "expected decoded global x4 lane 0 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][0] == 0x18191a1bu,
              "expected decoded global x4 lane 0 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][0] == 0x1c1d1e1fu,
              "expected decoded global x4 lane 0 dword 3 load result") ||
      !Expect(global_x4_state.vgprs[50][1] == 0x20212223u,
              "expected decoded global x4 lane 1 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][1] == 0x24252627u,
              "expected decoded global x4 lane 1 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][1] == 0x28292a2bu,
              "expected decoded global x4 lane 1 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][1] == 0x2c2d2e2fu,
              "expected decoded global x4 lane 1 dword 3 load result") ||
      !Expect(global_x4_state.vgprs[50][2] == 0x0u,
              "expected decoded global x4 inactive lane dword 0 result") ||
      !Expect(global_x4_state.vgprs[53][2] == 0x0u,
              "expected decoded global x4 inactive lane dword 3 result") ||
      !Expect(global_x4_state.vgprs[50][3] == 0x30313233u,
              "expected decoded global x4 lane 3 dword 0 load result") ||
      !Expect(global_x4_state.vgprs[51][3] == 0x34353637u,
              "expected decoded global x4 lane 3 dword 1 load result") ||
      !Expect(global_x4_state.vgprs[52][3] == 0x38393a3bu,
              "expected decoded global x4 lane 3 dword 2 load result") ||
      !Expect(global_x4_state.vgprs[53][3] == 0x3c3d3e3fu,
              "expected decoded global x4 lane 3 dword 3 load result")) {
    return 1;
  }

  std::uint32_t global_x4_store_value = 0;
  if (!Expect(global_x4_memory.ReadU32(0x480, &global_x4_store_value),
              "expected decoded global x4 lane 0 dword 0 store read") ||
      !Expect(global_x4_store_value == 0xa0a1a2a3u,
              "expected decoded global x4 lane 0 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x484, &global_x4_store_value),
              "expected decoded global x4 lane 0 dword 1 store read") ||
      !Expect(global_x4_store_value == 0xa4a5a6a7u,
              "expected decoded global x4 lane 0 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x488, &global_x4_store_value),
              "expected decoded global x4 lane 0 dword 2 store read") ||
      !Expect(global_x4_store_value == 0xa8a9aaabu,
              "expected decoded global x4 lane 0 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x48c, &global_x4_store_value),
              "expected decoded global x4 lane 0 dword 3 store read") ||
      !Expect(global_x4_store_value == 0xacadaeafu,
              "expected decoded global x4 lane 0 dword 3 store result") ||
      !Expect(global_x4_memory.ReadU32(0x490, &global_x4_store_value),
              "expected decoded global x4 lane 1 dword 0 store read") ||
      !Expect(global_x4_store_value == 0xb0b1b2b3u,
              "expected decoded global x4 lane 1 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x494, &global_x4_store_value),
              "expected decoded global x4 lane 1 dword 1 store read") ||
      !Expect(global_x4_store_value == 0xb4b5b6b7u,
              "expected decoded global x4 lane 1 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x498, &global_x4_store_value),
              "expected decoded global x4 lane 1 dword 2 store read") ||
      !Expect(global_x4_store_value == 0xb8b9babbu,
              "expected decoded global x4 lane 1 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x49c, &global_x4_store_value),
              "expected decoded global x4 lane 1 dword 3 store read") ||
      !Expect(global_x4_store_value == 0xbcbdbebfu,
              "expected decoded global x4 lane 1 dword 3 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b0, &global_x4_store_value),
              "expected decoded global x4 lane 3 dword 0 store read") ||
      !Expect(global_x4_store_value == 0xc0c1c2c3u,
              "expected decoded global x4 lane 3 dword 0 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b4, &global_x4_store_value),
              "expected decoded global x4 lane 3 dword 1 store read") ||
      !Expect(global_x4_store_value == 0xc4c5c6c7u,
              "expected decoded global x4 lane 3 dword 1 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4b8, &global_x4_store_value),
              "expected decoded global x4 lane 3 dword 2 store read") ||
      !Expect(global_x4_store_value == 0xc8c9cacbu,
              "expected decoded global x4 lane 3 dword 2 store result") ||
      !Expect(global_x4_memory.ReadU32(0x4bc, &global_x4_store_value),
              "expected decoded global x4 lane 3 dword 3 store read") ||
      !Expect(global_x4_store_value == 0xcccdcecfu,
              "expected decoded global x4 lane 3 dword 3 store result")) {
    return 1;
  }

  const auto flat_load_x2_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_DWORDX2", "ENC_FLAT");
  const auto flat_store_x2_opcode =
      FindDefaultEncodingOpcode("FLAT_STORE_DWORDX2", "ENC_FLAT");
  const auto flat_load_x3_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_DWORDX3", "ENC_FLAT");
  const auto flat_store_x3_opcode =
      FindDefaultEncodingOpcode("FLAT_STORE_DWORDX3", "ENC_FLAT");
  const auto flat_load_x4_opcode =
      FindDefaultEncodingOpcode("FLAT_LOAD_DWORDX4", "ENC_FLAT");
  const auto flat_store_x4_opcode =
      FindDefaultEncodingOpcode("FLAT_STORE_DWORDX4", "ENC_FLAT");
  const auto global_load_x3_opcode =
      FindDefaultEncodingOpcode("GLOBAL_LOAD_DWORDX3", "ENC_FLAT_GLBL");
  const auto global_store_x3_opcode =
      FindDefaultEncodingOpcode("GLOBAL_STORE_DWORDX3", "ENC_FLAT_GLBL");
  if (!Expect(flat_load_x2_opcode.has_value(),
              "expected flat load x2 opcode lookup") ||
      !Expect(flat_store_x2_opcode.has_value(),
              "expected flat store x2 opcode lookup") ||
      !Expect(flat_load_x3_opcode.has_value(),
              "expected flat load x3 opcode lookup") ||
      !Expect(flat_store_x3_opcode.has_value(),
              "expected flat store x3 opcode lookup") ||
      !Expect(flat_load_x4_opcode.has_value(),
              "expected flat load x4 opcode lookup") ||
      !Expect(flat_store_x4_opcode.has_value(),
              "expected flat store x4 opcode lookup") ||
      !Expect(global_load_x3_opcode.has_value(),
              "expected global load x3 opcode lookup") ||
      !Expect(global_store_x3_opcode.has_value(),
              "expected global store x3 opcode lookup")) {
    return 1;
  }

  const auto flat_load_x2_word = MakeFlat(*flat_load_x2_opcode, 50, 0, 0, 0);
  const auto flat_store_x2_word = MakeFlat(*flat_store_x2_opcode, 0, 2, 70, 0);
  const auto flat_load_x3_word = MakeFlat(*flat_load_x3_opcode, 52, 4, 0, 0);
  const auto flat_store_x3_word = MakeFlat(*flat_store_x3_opcode, 0, 6, 72, 0);
  const auto flat_load_x4_word = MakeFlat(*flat_load_x4_opcode, 55, 8, 0, 0);
  const auto flat_store_x4_word = MakeFlat(*flat_store_x4_opcode, 0, 10, 73, 0);
  const auto global_load_x3_word = MakeGlobal(*global_load_x3_opcode, 59, 12, 0, 0, 0);
  const auto global_store_x3_word =
      MakeGlobal(*global_store_x3_opcode, 0, 14, 79, 0, 0);
  const std::vector<std::uint32_t> mixed_width_program = {
      flat_load_x2_word[0],  flat_load_x2_word[1],
      flat_store_x2_word[0], flat_store_x2_word[1],
      flat_load_x3_word[0],  flat_load_x3_word[1],
      flat_store_x3_word[0], flat_store_x3_word[1],
      flat_load_x4_word[0],  flat_load_x4_word[1],
      flat_store_x4_word[0], flat_store_x4_word[1],
      global_load_x3_word[0], global_load_x3_word[1],
      global_store_x3_word[0], global_store_x3_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(mixed_width_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 9,
              "expected decoded mixed-width program size") ||
      !Expect(decoded_program[0].opcode == "FLAT_LOAD_DWORDX2",
              "expected flat load x2 decode") ||
      !Expect(decoded_program[2].opcode == "FLAT_LOAD_DWORDX3",
              "expected flat load x3 decode") ||
      !Expect(decoded_program[4].opcode == "FLAT_LOAD_DWORDX4",
              "expected flat load x4 decode") ||
      !Expect(decoded_program[6].opcode == "GLOBAL_LOAD_DWORDX3",
              "expected global load x3 decode")) {
    return 1;
  }

  LinearExecutionMemory mixed_width_memory(0x3000, 0);
  if (!Expect(mixed_width_memory.WriteU32(0x900, 0x10101010u),
              "expected decoded flat x2 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x904, 0x20202020u),
              "expected decoded flat x2 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x920, 0x30303030u),
              "expected decoded flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x924, 0x40404040u),
              "expected decoded flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x928, 0x50505050u),
              "expected decoded flat x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x940, 0x60606060u),
              "expected decoded flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x944, 0x70707070u),
              "expected decoded flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x948, 0x80808080u),
              "expected decoded flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0x94c, 0x90909090u),
              "expected decoded flat x4 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc20, 0xa0a0a0a0u),
              "expected decoded global x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc24, 0xb0b0b0b0u),
              "expected decoded global x3 load seed write") ||
      !Expect(mixed_width_memory.WriteU32(0xc28, 0xc0c0c0c0u),
              "expected decoded global x3 load seed write")) {
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &mixed_width_state,
                                         &mixed_width_memory, &error_message),
              error_message.c_str()) ||
      !Expect(mixed_width_state.vgprs[50][0] == 0x10101010u,
              "expected decoded flat x2 low load result") ||
      !Expect(mixed_width_state.vgprs[51][0] == 0x20202020u,
              "expected decoded flat x2 high load result") ||
      !Expect(mixed_width_state.vgprs[52][0] == 0x30303030u,
              "expected decoded flat x3 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[53][0] == 0x40404040u,
              "expected decoded flat x3 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[54][0] == 0x50505050u,
              "expected decoded flat x3 dword 2 load result") ||
      !Expect(mixed_width_state.vgprs[55][0] == 0x60606060u,
              "expected decoded flat x4 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[56][0] == 0x70707070u,
              "expected decoded flat x4 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[57][0] == 0x80808080u,
              "expected decoded flat x4 dword 2 load result") ||
      !Expect(mixed_width_state.vgprs[58][0] == 0x90909090u,
              "expected decoded flat x4 dword 3 load result") ||
      !Expect(mixed_width_state.vgprs[59][0] == 0xa0a0a0a0u,
              "expected decoded global x3 dword 0 load result") ||
      !Expect(mixed_width_state.vgprs[60][0] == 0xb0b0b0b0u,
              "expected decoded global x3 dword 1 load result") ||
      !Expect(mixed_width_state.vgprs[61][0] == 0xc0c0c0c0u,
              "expected decoded global x3 dword 2 load result")) {
    return 1;
  }

  std::uint32_t mixed_width_value = 0;
  if (!Expect(mixed_width_memory.ReadU32(0x980, &mixed_width_value),
              "expected decoded flat x2 store dword 0 read") ||
      !Expect(mixed_width_value == 0xd1d2d3d4u,
              "expected decoded flat x2 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x984, &mixed_width_value),
              "expected decoded flat x2 store dword 1 read") ||
      !Expect(mixed_width_value == 0xe1e2e3e4u,
              "expected decoded flat x2 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a0, &mixed_width_value),
              "expected decoded flat x3 store dword 0 read") ||
      !Expect(mixed_width_value == 0xf1f2f3f4u,
              "expected decoded flat x3 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a4, &mixed_width_value),
              "expected decoded flat x3 store dword 1 read") ||
      !Expect(mixed_width_value == 0x11121314u,
              "expected decoded flat x3 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9a8, &mixed_width_value),
              "expected decoded flat x3 store dword 2 read") ||
      !Expect(mixed_width_value == 0x21222324u,
              "expected decoded flat x3 store dword 2 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c0, &mixed_width_value),
              "expected decoded flat x4 store dword 0 read") ||
      !Expect(mixed_width_value == 0x11121314u,
              "expected decoded flat x4 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c4, &mixed_width_value),
              "expected decoded flat x4 store dword 1 read") ||
      !Expect(mixed_width_value == 0x21222324u,
              "expected decoded flat x4 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9c8, &mixed_width_value),
              "expected decoded flat x4 store dword 2 read") ||
      !Expect(mixed_width_value == 0x31323334u,
              "expected decoded flat x4 store dword 2 result") ||
      !Expect(mixed_width_memory.ReadU32(0x9cc, &mixed_width_value),
              "expected decoded flat x4 store dword 3 read") ||
      !Expect(mixed_width_value == 0x41424344u,
              "expected decoded flat x4 store dword 3 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc40, &mixed_width_value),
              "expected decoded global x3 store dword 0 read") ||
      !Expect(mixed_width_value == 0x51525354u,
              "expected decoded global x3 store dword 0 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc44, &mixed_width_value),
              "expected decoded global x3 store dword 1 read") ||
      !Expect(mixed_width_value == 0x61626364u,
              "expected decoded global x3 store dword 1 result") ||
      !Expect(mixed_width_memory.ReadU32(0xc48, &mixed_width_value),
              "expected decoded global x3 store dword 2 read") ||
      !Expect(mixed_width_value == 0x71727374u,
              "expected decoded global x3 store dword 2 result")) {
    return 1;
  }

  const auto atomic_add_word = MakeGlobalAtomic(66, true, 30, 14, 20, 0, 0);
  const auto atomic_swap_word = MakeGlobalAtomic(64, false, 9, 16, 21, 0, 0);
  const auto atomic_cmpswap_word = MakeGlobalAtomic(65, true, 31, 18, 22, 0, 0);
  const std::vector<std::uint32_t> atomic_program = {
      atomic_add_word[0], atomic_add_word[1],
      atomic_swap_word[0], atomic_swap_word[1],
      atomic_cmpswap_word[0], atomic_cmpswap_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(decoder.DecodeProgram(atomic_program, &decoded_program, &error_message),
              error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded atomic program size") ||
      !Expect(decoded_program[0].opcode == "GLOBAL_ATOMIC_ADD",
              "expected global atomic add decode") ||
      !Expect(decoded_program[0].operand_count == 5,
              "expected atomic add return form operand count") ||
      !Expect(decoded_program[1].opcode == "GLOBAL_ATOMIC_SWAP",
              "expected global atomic swap decode") ||
      !Expect(decoded_program[1].operand_count == 4,
              "expected atomic swap no-return operand count") ||
      !Expect(decoded_program[2].opcode == "GLOBAL_ATOMIC_CMPSWAP",
              "expected global atomic cmpswap decode") ||
      !Expect(decoded_program[2].operand_count == 5,
              "expected atomic cmpswap return form operand count") ||
      !Expect(decoded_program[2].operands[2].kind == OperandKind::kVgpr,
              "expected atomic cmpswap data operand decode") ||
      !Expect(decoded_program[2].operands[2].index == 22,
              "expected atomic cmpswap data start index")) {
    return 1;
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
  atomic_state.vgprs[16][0] = 0x30;
  atomic_state.vgprs[16][1] = 0x34;
  atomic_state.vgprs[16][3] = 0x3c;
  atomic_state.vgprs[18][0] = 0x40;
  atomic_state.vgprs[18][1] = 0x44;
  atomic_state.vgprs[18][3] = 0x4c;
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
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &atomic_state, &atomic_memory,
                                         &error_message),
              error_message.c_str()) ||
      !Expect(atomic_state.vgprs[30][0] == 10u,
              "expected decoded atomic add lane 0 return value") ||
      !Expect(atomic_state.vgprs[30][1] == 20u,
              "expected decoded atomic add lane 1 return value") ||
      !Expect(atomic_state.vgprs[30][2] == 0u,
              "expected decoded atomic add inactive lane return") ||
      !Expect(atomic_state.vgprs[30][3] == 40u,
              "expected decoded atomic add lane 3 return value") ||
      !Expect(atomic_state.vgprs[31][0] == 100u,
              "expected decoded atomic cmpswap lane 0 return value") ||
      !Expect(atomic_state.vgprs[31][1] == 110u,
              "expected decoded atomic cmpswap lane 1 return value") ||
      !Expect(atomic_state.vgprs[31][2] == 0xdeadbeefu,
              "expected decoded atomic cmpswap inactive lane destination") ||
      !Expect(atomic_state.vgprs[31][3] == 130u,
              "expected decoded atomic cmpswap lane 3 return value")) {
    return 1;
  }

  std::uint32_t atomic_value = 0;
  if (!Expect(atomic_memory.ReadU32(0x520, &atomic_value),
              "expected decoded atomic add lane 0 read") ||
      !Expect(atomic_value == 11u,
              "expected decoded atomic add lane 0 result") ||
      !Expect(atomic_memory.ReadU32(0x524, &atomic_value),
              "expected decoded atomic add lane 1 read") ||
      !Expect(atomic_value == 22u,
              "expected decoded atomic add lane 1 result") ||
      !Expect(atomic_memory.ReadU32(0x52c, &atomic_value),
              "expected decoded atomic add lane 3 read") ||
      !Expect(atomic_value == 44u,
              "expected decoded atomic add lane 3 result") ||
      !Expect(atomic_memory.ReadU32(0x530, &atomic_value),
              "expected decoded atomic swap lane 0 read") ||
      !Expect(atomic_value == 500u,
              "expected decoded atomic swap lane 0 result") ||
      !Expect(atomic_memory.ReadU32(0x534, &atomic_value),
              "expected decoded atomic swap lane 1 read") ||
      !Expect(atomic_value == 600u,
              "expected decoded atomic swap lane 1 result") ||
      !Expect(atomic_memory.ReadU32(0x53c, &atomic_value),
              "expected decoded atomic swap lane 3 read") ||
      !Expect(atomic_value == 800u,
              "expected decoded atomic swap lane 3 result") ||
      !Expect(atomic_memory.ReadU32(0x540, &atomic_value),
              "expected decoded atomic cmpswap lane 0 read") ||
      !Expect(atomic_value == 700u,
              "expected decoded atomic cmpswap lane 0 result") ||
      !Expect(atomic_memory.ReadU32(0x544, &atomic_value),
              "expected decoded atomic cmpswap lane 1 read") ||
      !Expect(atomic_value == 110u,
              "expected decoded atomic cmpswap mismatch result") ||
      !Expect(atomic_memory.ReadU32(0x54c, &atomic_value),
              "expected decoded atomic cmpswap lane 3 read") ||
      !Expect(atomic_value == 900u,
              "expected decoded atomic cmpswap lane 3 result")) {
    return 1;
  }

  const auto flat_atomic_add_word = MakeFlatAtomic(66, true, 30, 14, 20, 0);
  const auto flat_atomic_swap_word = MakeFlatAtomic(64, false, 9, 16, 21, 0);
  const auto flat_atomic_cmpswap_word =
      MakeFlatAtomic(65, true, 31, 18, 22, 0);
  const std::vector<std::uint32_t> flat_atomic_program = {
      flat_atomic_add_word[0],      flat_atomic_add_word[1],
      flat_atomic_swap_word[0],     flat_atomic_swap_word[1],
      flat_atomic_cmpswap_word[0],  flat_atomic_cmpswap_word[1],
      MakeSopp(1),
  };
  decoded_program.clear();
  if (!Expect(
          decoder.DecodeProgram(flat_atomic_program, &decoded_program, &error_message),
          error_message.c_str()) ||
      !Expect(decoded_program.size() == 4,
              "expected decoded flat atomic program size") ||
      !Expect(decoded_program[0].opcode == "FLAT_ATOMIC_ADD",
              "expected flat atomic add decode") ||
      !Expect(decoded_program[0].operand_count == 4,
              "expected flat atomic add return form operand count") ||
      !Expect(decoded_program[1].opcode == "FLAT_ATOMIC_SWAP",
              "expected flat atomic swap decode") ||
      !Expect(decoded_program[1].operand_count == 3,
              "expected flat atomic swap no-return operand count") ||
      !Expect(decoded_program[2].opcode == "FLAT_ATOMIC_CMPSWAP",
              "expected flat atomic cmpswap decode") ||
      !Expect(decoded_program[2].operand_count == 4,
              "expected flat atomic cmpswap return form operand count") ||
      !Expect(decoded_program[2].operands[2].kind == OperandKind::kVgpr,
              "expected flat atomic cmpswap data operand decode") ||
      !Expect(decoded_program[2].operands[2].index == 22,
              "expected flat atomic cmpswap data start index")) {
    return 1;
  }

  LinearExecutionMemory flat_atomic_memory(0x1000, 0);
  if (!Expect(flat_atomic_memory.WriteU32(0x520, 10u),
              "expected flat atomic add seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x524, 20u),
              "expected flat atomic add seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x52c, 40u),
              "expected flat atomic add seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x530, 50u),
              "expected flat atomic swap seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x534, 60u),
              "expected flat atomic swap seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x53c, 80u),
              "expected flat atomic swap seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x540, 100u),
              "expected flat atomic cmpswap seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x544, 110u),
              "expected flat atomic cmpswap seed write") ||
      !Expect(flat_atomic_memory.WriteU32(0x54c, 130u),
              "expected flat atomic cmpswap seed write")) {
    return 1;
  }

  WaveExecutionState flat_atomic_state;
  flat_atomic_state.exec_mask = 0b1011ULL;
  flat_atomic_state.vgprs[14][0] = 0x520;
  flat_atomic_state.vgprs[14][1] = 0x524;
  flat_atomic_state.vgprs[14][3] = 0x52c;
  flat_atomic_state.vgprs[15][0] = 0x0;
  flat_atomic_state.vgprs[15][1] = 0x0;
  flat_atomic_state.vgprs[15][3] = 0x0;
  flat_atomic_state.vgprs[16][0] = 0x530;
  flat_atomic_state.vgprs[16][1] = 0x534;
  flat_atomic_state.vgprs[16][3] = 0x53c;
  flat_atomic_state.vgprs[17][0] = 0x0;
  flat_atomic_state.vgprs[17][1] = 0x0;
  flat_atomic_state.vgprs[17][3] = 0x0;
  flat_atomic_state.vgprs[18][0] = 0x540;
  flat_atomic_state.vgprs[18][1] = 0x544;
  flat_atomic_state.vgprs[18][3] = 0x54c;
  flat_atomic_state.vgprs[19][0] = 0x0;
  flat_atomic_state.vgprs[19][1] = 0x0;
  flat_atomic_state.vgprs[19][3] = 0x0;
  flat_atomic_state.vgprs[20][0] = 1u;
  flat_atomic_state.vgprs[20][1] = 2u;
  flat_atomic_state.vgprs[20][3] = 4u;
  flat_atomic_state.vgprs[21][0] = 500u;
  flat_atomic_state.vgprs[21][1] = 600u;
  flat_atomic_state.vgprs[21][3] = 800u;
  flat_atomic_state.vgprs[22][0] = 100u;
  flat_atomic_state.vgprs[22][1] = 999u;
  flat_atomic_state.vgprs[22][3] = 130u;
  flat_atomic_state.vgprs[23][0] = 700u;
  flat_atomic_state.vgprs[23][1] = 777u;
  flat_atomic_state.vgprs[23][3] = 900u;
  flat_atomic_state.vgprs[31][0] = 0xdeadbeefu;
  flat_atomic_state.vgprs[31][2] = 0xdeadbeefu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &flat_atomic_state,
                                         &flat_atomic_memory, &error_message),
              error_message.c_str()) ||
      !Expect(flat_atomic_state.vgprs[30][0] == 10u,
              "expected decoded flat atomic add lane 0 return value") ||
      !Expect(flat_atomic_state.vgprs[30][1] == 20u,
              "expected decoded flat atomic add lane 1 return value") ||
      !Expect(flat_atomic_state.vgprs[30][2] == 0u,
              "expected decoded flat atomic add inactive lane return") ||
      !Expect(flat_atomic_state.vgprs[30][3] == 40u,
              "expected decoded flat atomic add lane 3 return value") ||
      !Expect(flat_atomic_state.vgprs[31][0] == 100u,
              "expected decoded flat atomic cmpswap lane 0 return value") ||
      !Expect(flat_atomic_state.vgprs[31][1] == 110u,
              "expected decoded flat atomic cmpswap lane 1 return value") ||
      !Expect(flat_atomic_state.vgprs[31][2] == 0xdeadbeefu,
              "expected decoded flat atomic cmpswap inactive lane destination") ||
      !Expect(flat_atomic_state.vgprs[31][3] == 130u,
              "expected decoded flat atomic cmpswap lane 3 return value")) {
    return 1;
  }

  if (!Expect(flat_atomic_memory.ReadU32(0x520, &atomic_value),
              "expected decoded flat atomic add lane 0 read") ||
      !Expect(atomic_value == 11u,
              "expected decoded flat atomic add lane 0 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x524, &atomic_value),
              "expected decoded flat atomic add lane 1 read") ||
      !Expect(atomic_value == 22u,
              "expected decoded flat atomic add lane 1 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x52c, &atomic_value),
              "expected decoded flat atomic add lane 3 read") ||
      !Expect(atomic_value == 44u,
              "expected decoded flat atomic add lane 3 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x530, &atomic_value),
              "expected decoded flat atomic swap lane 0 read") ||
      !Expect(atomic_value == 500u,
              "expected decoded flat atomic swap lane 0 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x534, &atomic_value),
              "expected decoded flat atomic swap lane 1 read") ||
      !Expect(atomic_value == 600u,
              "expected decoded flat atomic swap lane 1 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x53c, &atomic_value),
              "expected decoded flat atomic swap lane 3 read") ||
      !Expect(atomic_value == 800u,
              "expected decoded flat atomic swap lane 3 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x540, &atomic_value),
              "expected decoded flat atomic cmpswap lane 0 read") ||
      !Expect(atomic_value == 700u,
              "expected decoded flat atomic cmpswap lane 0 result") ||
      !Expect(flat_atomic_memory.ReadU32(0x544, &atomic_value),
              "expected decoded flat atomic cmpswap lane 1 read") ||
      !Expect(atomic_value == 110u,
              "expected decoded flat atomic cmpswap mismatch result") ||
      !Expect(flat_atomic_memory.ReadU32(0x54c, &atomic_value),
              "expected decoded flat atomic cmpswap lane 3 read") ||
      !Expect(atomic_value == 900u,
              "expected decoded flat atomic cmpswap lane 3 result")) {
    return 1;
  }

  const std::array<std::string_view, 20> kAdditionalFlatMemoryOpcodes = {
      "FLAT_LOAD_UBYTE",        "FLAT_LOAD_UBYTE_D16",
      "FLAT_LOAD_UBYTE_D16_HI", "FLAT_LOAD_SBYTE",
      "FLAT_LOAD_SBYTE_D16",    "FLAT_LOAD_SBYTE_D16_HI",
      "FLAT_LOAD_USHORT",       "FLAT_LOAD_SSHORT",
      "FLAT_LOAD_SHORT_D16",    "FLAT_LOAD_SHORT_D16_HI",
      "FLAT_LOAD_DWORDX2",      "FLAT_LOAD_DWORDX3",
      "FLAT_LOAD_DWORDX4",      "FLAT_STORE_BYTE",
      "FLAT_STORE_BYTE_D16_HI", "FLAT_STORE_SHORT",
      "FLAT_STORE_SHORT_D16_HI","FLAT_STORE_DWORDX2",
      "FLAT_STORE_DWORDX3",     "FLAT_STORE_DWORDX4",
  };
  for (std::string_view opcode_name : kAdditionalFlatMemoryOpcodes) {
    const std::optional<std::uint32_t> opcode_value =
        FindDefaultEncodingOpcode(opcode_name, "ENC_FLAT");
    if (!Expect(opcode_value.has_value(),
                ("expected catalog opcode for " + std::string(opcode_name)).c_str())) {
      return 1;
    }

    const bool is_load = opcode_name.starts_with("FLAT_LOAD_");
    const auto word = MakeFlat(*opcode_value, 60, 10, 20, 4);
    const std::array<std::uint32_t, 2> encoded_word = {word[0], word[1]};
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!Expect(decoder.DecodeInstruction(encoded_word, &instruction, &words_consumed,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }
    if (!Expect(words_consumed == 2,
                ("expected two-word decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.opcode == opcode_name,
                ("expected opcode decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operand_count == 3,
                ("expected operand count for " + std::string(opcode_name)).c_str())) {
      return 1;
    }
    if (is_load) {
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 60,
                  ("expected load destination for " + std::string(opcode_name)).c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 10,
                  ("expected load address for " + std::string(opcode_name)).c_str())) {
        return 1;
      }
    } else {
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 10,
                  ("expected store address for " + std::string(opcode_name)).c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 20,
                  ("expected store data for " + std::string(opcode_name)).c_str())) {
        return 1;
      }
    }
  }

  const std::array<std::string_view, 16> kAdditionalGlobalMemoryOpcodes = {
      "GLOBAL_LOAD_UBYTE",        "GLOBAL_LOAD_UBYTE_D16",
      "GLOBAL_LOAD_UBYTE_D16_HI", "GLOBAL_LOAD_SBYTE",
      "GLOBAL_LOAD_SBYTE_D16",    "GLOBAL_LOAD_SBYTE_D16_HI",
      "GLOBAL_LOAD_USHORT",       "GLOBAL_LOAD_SSHORT",
      "GLOBAL_LOAD_SHORT_D16",    "GLOBAL_LOAD_SHORT_D16_HI",
      "GLOBAL_LOAD_DWORDX3",      "GLOBAL_STORE_BYTE",
      "GLOBAL_STORE_BYTE_D16_HI", "GLOBAL_STORE_SHORT",
      "GLOBAL_STORE_SHORT_D16_HI","GLOBAL_STORE_DWORDX3",
  };
  for (std::string_view opcode_name : kAdditionalGlobalMemoryOpcodes) {
    const std::optional<std::uint32_t> opcode_value =
        FindDefaultEncodingOpcode(opcode_name, "ENC_FLAT_GLBL");
    if (!Expect(opcode_value.has_value(),
                ("expected catalog opcode for " + std::string(opcode_name)).c_str())) {
      return 1;
    }

    const bool is_load = opcode_name.starts_with("GLOBAL_LOAD_");
    const auto word = MakeGlobal(*opcode_value, 60, 10, 20, 4, -8);
    const std::array<std::uint32_t, 2> encoded_word = {word[0], word[1]};
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!Expect(decoder.DecodeInstruction(encoded_word, &instruction, &words_consumed,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }
    if (!Expect(words_consumed == 2,
                ("expected two-word decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.opcode == opcode_name,
                ("expected opcode decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operand_count == 4,
                ("expected operand count for " + std::string(opcode_name)).c_str())) {
      return 1;
    }
    if (is_load) {
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 60,
                  ("expected load destination for " + std::string(opcode_name)).c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 10,
                  ("expected load address for " + std::string(opcode_name)).c_str())) {
        return 1;
      }
    } else {
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 10,
                  ("expected store address for " + std::string(opcode_name)).c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 20,
                  ("expected store data for " + std::string(opcode_name)).c_str())) {
        return 1;
      }
    }
    if (!Expect(instruction.operands[2].kind == OperandKind::kSgpr &&
                    instruction.operands[2].index == 4,
                ("expected scalar base for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operands[3].kind == OperandKind::kImm32 &&
                    instruction.operands[3].imm32 ==
                        static_cast<std::uint32_t>(-8),
                ("expected offset for " + std::string(opcode_name)).c_str())) {
      return 1;
    }
  }

  const std::array<std::string_view, 7> kGlobalLoadLdsOpcodes = {
      "GLOBAL_LOAD_LDS_UBYTE",  "GLOBAL_LOAD_LDS_SBYTE",
      "GLOBAL_LOAD_LDS_USHORT", "GLOBAL_LOAD_LDS_SSHORT",
      "GLOBAL_LOAD_LDS_DWORD",  "GLOBAL_LOAD_LDS_DWORDX3",
      "GLOBAL_LOAD_LDS_DWORDX4",
  };
  for (std::string_view opcode_name : kGlobalLoadLdsOpcodes) {
    const std::optional<std::uint32_t> opcode_value =
        FindDefaultEncodingOpcode(opcode_name, "ENC_FLAT_GLBL");
    if (!Expect(opcode_value.has_value(),
                ("expected catalog opcode for " + std::string(opcode_name)).c_str())) {
      return 1;
    }

    const auto word = MakeGlobal(*opcode_value, 60, 10, 20, 4, -8);
    const std::array<std::uint32_t, 2> encoded_word = {word[0], word[1]};
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!Expect(decoder.DecodeInstruction(encoded_word, &instruction, &words_consumed,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }
    if (!Expect(words_consumed == 2,
                ("expected two-word decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.opcode == opcode_name,
                ("expected opcode decode for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operand_count == 3,
                ("expected operand count for " + std::string(opcode_name)).c_str())) {
      return 1;
    }
    if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                    instruction.operands[0].index == 10,
                ("expected address operand for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operands[1].kind == OperandKind::kSgpr &&
                    instruction.operands[1].index == 4,
                ("expected scalar base for " + std::string(opcode_name)).c_str()) ||
        !Expect(instruction.operands[2].kind == OperandKind::kImm32 &&
                    instruction.operands[2].imm32 ==
                        static_cast<std::uint32_t>(-8),
                ("expected offset for " + std::string(opcode_name)).c_str())) {
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
  for (std::size_t opcode_index = 0; opcode_index < kGlobalAtomicOpcodes.size();
       ++opcode_index) {
    const std::string_view opcode_name = kGlobalAtomicOpcodes[opcode_index];
    const std::optional<std::uint32_t> opcode_value =
        FindDefaultEncodingOpcode(opcode_name, "ENC_FLAT_GLBL");
    if (!Expect(opcode_value.has_value(),
                ("expected catalog opcode for " + std::string(opcode_name)).c_str())) {
      return 1;
    }

    const bool return_prior_value = (opcode_index % 2) == 0;
    const auto atomic_word =
        MakeGlobalAtomic(*opcode_value, return_prior_value, 60, 10, 20, 4, -8);
    const std::array<std::uint32_t, 2> encoded_atomic_word = {atomic_word[0],
                                                               atomic_word[1]};
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!Expect(decoder.DecodeInstruction(encoded_atomic_word, &instruction,
                                          &words_consumed,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }

    const std::string consumed_message =
        "expected two-word decode for " + std::string(opcode_name);
    if (!Expect(words_consumed == 2, consumed_message.c_str())) {
      return 1;
    }
    const std::string opcode_message =
        "expected opcode decode for " + std::string(opcode_name);
    if (!Expect(instruction.opcode == opcode_name, opcode_message.c_str())) {
      return 1;
    }
    const std::string operand_count_message =
        "expected operand count for " + std::string(opcode_name);
    if (!Expect(instruction.operand_count == (return_prior_value ? 5 : 4),
                operand_count_message.c_str())) {
      return 1;
    }

    if (return_prior_value) {
      const std::string dst_message =
          "expected return VGPR decode for " + std::string(opcode_name);
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 60,
                  dst_message.c_str()) ||
          !Expect(instruction.operands[2].kind == OperandKind::kVgpr &&
                      instruction.operands[2].index == 20,
                  ("expected data VGPR decode for " + std::string(opcode_name))
                      .c_str()) ||
          !Expect(instruction.operands[3].kind == OperandKind::kSgpr &&
                      instruction.operands[3].index == 4,
                  ("expected scalar base decode for " + std::string(opcode_name))
                      .c_str()) ||
          !Expect(instruction.operands[4].kind == OperandKind::kImm32 &&
                      instruction.operands[4].imm32 ==
                          static_cast<std::uint32_t>(-8),
                  ("expected offset decode for " + std::string(opcode_name))
                      .c_str())) {
        return 1;
      }
    } else {
      const std::string addr_message =
          "expected address VGPR decode for " + std::string(opcode_name);
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 10,
                  addr_message.c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 20,
                  ("expected data VGPR decode for " + std::string(opcode_name))
                      .c_str()) ||
          !Expect(instruction.operands[2].kind == OperandKind::kSgpr &&
                      instruction.operands[2].index == 4,
                  ("expected scalar base decode for " + std::string(opcode_name))
                      .c_str()) ||
          !Expect(instruction.operands[3].kind == OperandKind::kImm32 &&
                      instruction.operands[3].imm32 ==
                          static_cast<std::uint32_t>(-8),
                  ("expected offset decode for " + std::string(opcode_name))
                      .c_str())) {
        return 1;
      }
    }
  }

  for (std::size_t opcode_index = 0; opcode_index < kGlobalAtomicOpcodes.size();
       ++opcode_index) {
    const std::string_view global_opcode_name = kGlobalAtomicOpcodes[opcode_index];
    const std::string opcode_name =
        "FLAT_" + std::string(global_opcode_name.substr(7));
    const std::optional<std::uint32_t> opcode_value =
        FindDefaultEncodingOpcode(opcode_name, "ENC_FLAT");
    if (!Expect(opcode_value.has_value(),
                ("expected catalog opcode for " + opcode_name).c_str())) {
      return 1;
    }

    const bool return_prior_value = (opcode_index % 2) == 0;
    const auto atomic_word =
        MakeFlatAtomic(*opcode_value, return_prior_value, 60, 10, 20, 4);
    const std::array<std::uint32_t, 2> encoded_atomic_word = {atomic_word[0],
                                                               atomic_word[1]};
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!Expect(decoder.DecodeInstruction(encoded_atomic_word, &instruction,
                                          &words_consumed,
                                          &error_message),
                error_message.c_str())) {
      return 1;
    }

    const std::string consumed_message =
        "expected two-word decode for " + opcode_name;
    if (!Expect(words_consumed == 2, consumed_message.c_str())) {
      return 1;
    }
    const std::string opcode_message =
        "expected opcode decode for " + opcode_name;
    if (!Expect(instruction.opcode == opcode_name, opcode_message.c_str())) {
      return 1;
    }
    const std::string operand_count_message =
        "expected operand count for " + opcode_name;
    if (!Expect(instruction.operand_count == (return_prior_value ? 4 : 3),
                operand_count_message.c_str())) {
      return 1;
    }

    if (return_prior_value) {
      const std::string dst_message =
          "expected return VGPR decode for " + opcode_name;
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 60,
                  dst_message.c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 10,
                  ("expected address VGPR decode for " + opcode_name).c_str()) ||
          !Expect(instruction.operands[2].kind == OperandKind::kVgpr &&
                      instruction.operands[2].index == 20,
                  ("expected data VGPR decode for " + opcode_name).c_str()) ||
          !Expect(instruction.operands[3].kind == OperandKind::kImm32 &&
                      instruction.operands[3].imm32 == 4u,
                  ("expected offset decode for " + opcode_name).c_str())) {
        return 1;
      }
    } else {
      const std::string addr_message =
          "expected address VGPR decode for " + opcode_name;
      if (!Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                      instruction.operands[0].index == 10,
                  addr_message.c_str()) ||
          !Expect(instruction.operands[1].kind == OperandKind::kVgpr &&
                      instruction.operands[1].index == 20,
                  ("expected data VGPR decode for " + opcode_name).c_str()) ||
          !Expect(instruction.operands[2].kind == OperandKind::kImm32 &&
                      instruction.operands[2].imm32 == 4u,
                  ("expected offset decode for " + opcode_name).c_str())) {
        return 1;
      }
    }
  }

  return 0;
}
