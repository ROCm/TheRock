#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/interpreter.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
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

bool ExpectUnaryInstruction(const mirage::sim::isa::DecodedInstruction& instruction,
                            std::string_view expected_opcode,
                            mirage::sim::isa::OperandKind dst_kind,
                            std::uint16_t dst_index,
                            mirage::sim::isa::OperandKind src_kind,
                            std::uint32_t src_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 2u) {
    return false;
  }
  if (instruction.operands[0].kind != dst_kind ||
      instruction.operands[0].index != dst_index ||
      instruction.operands[1].kind != src_kind) {
    return false;
  }
  if (src_kind == OperandKind::kImm32) {
    return instruction.operands[1].imm32 == src_value_or_index;
  }
  return instruction.operands[1].index ==
         static_cast<std::uint16_t>(src_value_or_index);
}

bool ExpectBinaryInstruction(const mirage::sim::isa::DecodedInstruction& instruction,
                             std::string_view expected_opcode,
                             mirage::sim::isa::OperandKind dst_kind,
                             std::uint16_t dst_index,
                             mirage::sim::isa::OperandKind src0_kind,
                             std::uint32_t src0_value_or_index,
                             mirage::sim::isa::OperandKind src1_kind,
                             std::uint32_t src1_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 3u) {
    return false;
  }
  if (instruction.operands[0].kind != dst_kind ||
      instruction.operands[0].index != dst_index ||
      instruction.operands[1].kind != src0_kind ||
      instruction.operands[2].kind != src1_kind) {
    return false;
  }
  const bool src0_matches =
      src0_kind == OperandKind::kImm32
          ? instruction.operands[1].imm32 == src0_value_or_index
          : instruction.operands[1].index ==
                static_cast<std::uint16_t>(src0_value_or_index);
  const bool src1_matches =
      src1_kind == OperandKind::kImm32
          ? instruction.operands[2].imm32 == src1_value_or_index
          : instruction.operands[2].index ==
                static_cast<std::uint16_t>(src1_value_or_index);
  return src0_matches && src1_matches;
}

bool ExpectArithmeticSeedProgramState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[1] == 10u && state.sgprs[2] == 3u &&
         state.sgprs[3] == 13u && state.sgprs[4] == 10u && state.scc &&
         state.vgprs[1][0] == 10u && state.vgprs[1][1] == 10u &&
         state.vgprs[1][2] == 0xfeedfaceu && state.vgprs[1][3] == 10u &&
         state.vgprs[2][0] == 20u && state.vgprs[2][1] == 20u &&
         state.vgprs[2][2] == 0xabcdef01u && state.vgprs[2][3] == 20u &&
         state.halted && !state.waiting_on_barrier && state.pc == 7u;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  Gfx1201BinaryDecoder decoder;
  Gfx1201Interpreter interpreter;
  std::string error_message;
  DecodedInstruction instruction;
  std::size_t words_consumed = 0;

  const std::array<std::uint32_t, 2> scalar_literal_words{
      MakeSop1(0u, 7u, 255u), 0x12345678u};
  if (!Expect(decoder.DecodeInstruction(scalar_literal_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_MOV_B32 literal decode success") ||
      !Expect(words_consumed == 2u, "expected literal decode to consume 2 dwords") ||
      !Expect(ExpectUnaryInstruction(instruction, "S_MOV_B32",
                                     OperandKind::kSgpr, 7u,
                                     OperandKind::kImm32, 0x12345678u),
              "expected decoded S_MOV_B32 literal operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> movk_words{MakeSopk(0u, 4u, 0xfffeu)};
  if (!Expect(decoder.DecodeInstruction(movk_words, &instruction, &words_consumed,
                                        &error_message),
              "expected S_MOVK_I32 decode success") ||
      !Expect(words_consumed == 1u, "expected S_MOVK_I32 to consume 1 dword") ||
      !Expect(ExpectUnaryInstruction(instruction, "S_MOVK_I32",
                                     OperandKind::kSgpr, 4u,
                                     OperandKind::kImm32, 0xfffffffeu),
              "expected decoded S_MOVK_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_add_words{MakeSop2(0u, 6u, 1u, 2u)};
  if (!Expect(decoder.DecodeInstruction(scalar_add_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_ADD_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "S_ADD_U32",
                                      OperandKind::kSgpr, 6u,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kSgpr, 2u),
              "expected decoded S_ADD_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_add_i32_words{
      MakeSop2(2u, 7u, 1u, 2u)};
  if (!Expect(decoder.DecodeInstruction(scalar_add_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_ADD_I32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "S_ADD_I32",
                                      OperandKind::kSgpr, 7u,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kSgpr, 2u),
              "expected decoded S_ADD_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_sub_words{MakeSop2(1u, 8u, 1u, 2u)};
  if (!Expect(decoder.DecodeInstruction(scalar_sub_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_SUB_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "S_SUB_U32",
                                      OperandKind::kSgpr, 8u,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kSgpr, 2u),
              "expected decoded S_SUB_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_move_words{MakeVop1(1u, 3u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_move_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOV_B32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_MOV_B32",
                                     OperandKind::kVgpr, 3u,
                                     OperandKind::kSgpr, 2u),
              "expected decoded V_MOV_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_add_words{
      MakeVop2(37u, 4u, 257u, 3u)};
  if (!Expect(decoder.DecodeInstruction(vector_add_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_ADD_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_ADD_U32",
                                      OperandKind::kVgpr, 4u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 3u),
              "expected decoded V_ADD_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 8> seed_program_words{
      MakeSopk(0u, 1u, 10u),
      MakeSopk(0u, 2u, 3u),
      MakeSop2(0u, 3u, 1u, 2u),
      MakeSop2(1u, 4u, 3u, 2u),
      MakeVop1(1u, 1u, 1u),
      MakeVop2(37u, 2u, 257u, 1u),
      MakeSopp(0u, 5u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> decoded_program;
  if (!Expect(decoder.DecodeProgram(seed_program_words, &decoded_program,
                                    &error_message),
              "expected arithmetic/control seed program decode success") ||
      !Expect(decoded_program.size() == 8u,
              "expected eight decoded instructions") ||
      !Expect(decoded_program[2].opcode == "S_ADD_U32",
              "expected decoded S_ADD_U32") ||
      !Expect(decoded_program[3].opcode == "S_SUB_U32",
              "expected decoded S_SUB_U32") ||
      !Expect(decoded_program[5].opcode == "V_ADD_U32",
              "expected decoded V_ADD_U32")) {
    return 1;
  }

  WaveExecutionState decoded_state;
  decoded_state.exec_mask = 0xbu;
  decoded_state.vgprs[1][2] = 0xfeedfaceu;
  decoded_state.vgprs[2][2] = 0xabcdef01u;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &decoded_state,
                                         &error_message),
              "expected decoded arithmetic/control execution success") ||
      !Expect(ExpectArithmeticSeedProgramState(decoded_state),
              "expected decoded arithmetic/control state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(decoded_program, &compiled_program,
                                         &error_message),
              "expected compiled arithmetic/control seed program success") ||
      !Expect(compiled_program.size() == decoded_program.size(),
              "expected compiled instruction count") ||
      !Expect(compiled_program[2].opcode == Gfx1201CompiledOpcode::kSAddU32,
              "expected compiled S_ADD_U32 opcode") ||
      !Expect(compiled_program[3].opcode == Gfx1201CompiledOpcode::kSSubU32,
              "expected compiled S_SUB_U32 opcode") ||
      !Expect(compiled_program[5].opcode == Gfx1201CompiledOpcode::kVAddU32,
              "expected compiled V_ADD_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_state;
  compiled_state.exec_mask = 0xbu;
  compiled_state.vgprs[1][2] = 0xfeedfaceu;
  compiled_state.vgprs[2][2] = 0xabcdef01u;
  if (!Expect(interpreter.ExecuteProgram(compiled_program, &compiled_state,
                                         &error_message),
              "expected compiled arithmetic/control execution success") ||
      !Expect(ExpectArithmeticSeedProgramState(compiled_state),
              "expected compiled arithmetic/control state")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> add_i32_words{MakeSop2(2u, 12u, 10u, 11u)};
  if (!Expect(decoder.DecodeInstruction(add_i32_words, &instruction, &words_consumed,
                                        &error_message),
              "expected S_ADD_I32 overflow decode success")) {
    return 1;
  }

  WaveExecutionState add_i32_state;
  add_i32_state.sgprs[10] = 0xffffffffu;
  add_i32_state.sgprs[11] = 1u;
  const std::array<DecodedInstruction, 1> add_i32_program{instruction};
  if (!Expect(interpreter.ExecuteProgram(add_i32_program, &add_i32_state,
                                         &error_message),
              "expected S_ADD_I32 execution success") ||
      !Expect(add_i32_state.sgprs[12] == 0u,
              "expected S_ADD_I32 wrapped result") ||
      !Expect(add_i32_state.scc, "expected S_ADD_I32 carry-out in SCC") ||
      !Expect(add_i32_state.pc == 1u, "expected S_ADD_I32 program advance") ||
      !Expect(!add_i32_state.halted,
              "expected single-instruction add program to remain running")) {
    return 1;
  }

  return 0;
}
