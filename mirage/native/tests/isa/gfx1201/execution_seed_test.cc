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

bool ExpectCompareInstruction(const mirage::sim::isa::DecodedInstruction& instruction,
                              std::string_view expected_opcode,
                              mirage::sim::isa::OperandKind src0_kind,
                              std::uint32_t src0_value_or_index,
                              mirage::sim::isa::OperandKind src1_kind,
                              std::uint32_t src1_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 2u ||
      instruction.operands[0].kind != src0_kind ||
      instruction.operands[1].kind != src1_kind) {
    return false;
  }
  const bool src0_matches =
      src0_kind == OperandKind::kImm32
          ? instruction.operands[0].imm32 == src0_value_or_index
          : instruction.operands[0].index ==
                static_cast<std::uint16_t>(src0_value_or_index);
  const bool src1_matches =
      src1_kind == OperandKind::kImm32
          ? instruction.operands[1].imm32 == src1_value_or_index
          : instruction.operands[1].index ==
                static_cast<std::uint16_t>(src1_value_or_index);
  return src0_matches && src1_matches;
}

bool ExpectBranchInstruction(const mirage::sim::isa::DecodedInstruction& instruction,
                             std::string_view expected_opcode,
                             std::uint32_t expected_delta) {
  using namespace mirage::sim::isa;
  return instruction.opcode == expected_opcode &&
         instruction.operand_count == 1u &&
         instruction.operands[0].kind == OperandKind::kImm32 &&
         instruction.operands[0].imm32 == expected_delta;
}

bool ExpectArithmeticSeedProgramState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[1] == 10u && state.sgprs[2] == 3u &&
         state.sgprs[3] == 13u && state.sgprs[4] == 10u && state.scc &&
         state.vgprs[1][0] == 10u && state.vgprs[1][1] == 10u &&
         state.vgprs[1][2] == 0xfeedfaceu && state.vgprs[1][3] == 10u &&
         state.vgprs[2][0] == 20u && state.vgprs[2][1] == 20u &&
         state.vgprs[2][2] == 0xabcdef01u && state.vgprs[2][3] == 20u &&
         state.vgprs[3][0] == 10u && state.vgprs[3][1] == 10u &&
         state.vgprs[3][2] == 0x13579bdfu && state.vgprs[3][3] == 10u &&
         state.halted && !state.waiting_on_barrier && state.pc == 8u;
}

bool ExpectCompareBranchEqState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[10] == 5u && state.sgprs[11] == 5u &&
         state.sgprs[12] == 222u && state.scc && state.halted &&
         !state.waiting_on_barrier && state.pc == 7u;
}

bool ExpectBranchAndCmpLgState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[13] == 88u && state.sgprs[14] == 88u &&
         state.sgprs[15] == 2u && !state.scc && state.halted &&
         !state.waiting_on_barrier && state.pc == 8u;
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

  const std::array<std::uint32_t, 1> scalar_cmp_eq_words{MakeSopc(6u, 1u, 2u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_eq_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_EQ_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_EQ_U32",
                                       OperandKind::kSgpr, 1u,
                                       OperandKind::kSgpr, 2u),
              "expected decoded S_CMP_EQ_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_lg_words{MakeSopc(7u, 3u, 4u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_lg_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LG_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LG_U32",
                                       OperandKind::kSgpr, 3u,
                                       OperandKind::kSgpr, 4u),
              "expected decoded S_CMP_LG_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> branch_words{MakeSopp(32u, 0xfffeu)};
  if (!Expect(decoder.DecodeInstruction(branch_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_BRANCH decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_BRANCH", 0xfffffffeu),
              "expected decoded S_BRANCH immediate")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> cbranch0_words{MakeSopp(33u, 1u)};
  if (!Expect(decoder.DecodeInstruction(cbranch0_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_SCC0 decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_SCC0", 1u),
              "expected decoded S_CBRANCH_SCC0 immediate")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> cbranch1_words{MakeSopp(34u, 3u)};
  if (!Expect(decoder.DecodeInstruction(cbranch1_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_SCC1 decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_SCC1", 3u),
              "expected decoded S_CBRANCH_SCC1 immediate")) {
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

  const std::array<std::uint32_t, 1> vector_sub_words{
      MakeVop2(38u, 5u, 258u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_sub_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SUB_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_SUB_U32",
                                      OperandKind::kVgpr, 5u,
                                      OperandKind::kVgpr, 2u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_SUB_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 9> arithmetic_program_words{
      MakeSopk(0u, 1u, 10u),
      MakeSopk(0u, 2u, 3u),
      MakeSop2(0u, 3u, 1u, 2u),
      MakeSop2(1u, 4u, 3u, 2u),
      MakeVop1(1u, 1u, 1u),
      MakeVop2(37u, 2u, 257u, 1u),
      MakeVop2(38u, 3u, 258u, 1u),
      MakeSopp(0u, 5u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> arithmetic_program;
  if (!Expect(decoder.DecodeProgram(arithmetic_program_words, &arithmetic_program,
                                    &error_message),
              "expected arithmetic/vector seed program decode success") ||
      !Expect(arithmetic_program.size() == 9u,
              "expected nine decoded arithmetic/vector instructions") ||
      !Expect(arithmetic_program[2].opcode == "S_ADD_U32",
              "expected decoded S_ADD_U32") ||
      !Expect(arithmetic_program[3].opcode == "S_SUB_U32",
              "expected decoded S_SUB_U32") ||
      !Expect(arithmetic_program[5].opcode == "V_ADD_U32",
              "expected decoded V_ADD_U32") ||
      !Expect(arithmetic_program[6].opcode == "V_SUB_U32",
              "expected decoded V_SUB_U32")) {
    return 1;
  }

  WaveExecutionState decoded_arithmetic_state;
  decoded_arithmetic_state.exec_mask = 0xbu;
  decoded_arithmetic_state.vgprs[1][2] = 0xfeedfaceu;
  decoded_arithmetic_state.vgprs[2][2] = 0xabcdef01u;
  decoded_arithmetic_state.vgprs[3][2] = 0x13579bdfu;
  if (!Expect(interpreter.ExecuteProgram(arithmetic_program, &decoded_arithmetic_state,
                                         &error_message),
              "expected decoded arithmetic/vector execution success") ||
      !Expect(ExpectArithmeticSeedProgramState(decoded_arithmetic_state),
              "expected decoded arithmetic/vector state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_arithmetic_program;
  if (!Expect(interpreter.CompileProgram(arithmetic_program,
                                         &compiled_arithmetic_program,
                                         &error_message),
              "expected compiled arithmetic/vector seed program success") ||
      !Expect(compiled_arithmetic_program.size() == arithmetic_program.size(),
              "expected compiled arithmetic/vector instruction count") ||
      !Expect(compiled_arithmetic_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSAddU32,
              "expected compiled S_ADD_U32 opcode") ||
      !Expect(compiled_arithmetic_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSSubU32,
              "expected compiled S_SUB_U32 opcode") ||
      !Expect(compiled_arithmetic_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVAddU32,
              "expected compiled V_ADD_U32 opcode") ||
      !Expect(compiled_arithmetic_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVSubU32,
              "expected compiled V_SUB_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_arithmetic_state;
  compiled_arithmetic_state.exec_mask = 0xbu;
  compiled_arithmetic_state.vgprs[1][2] = 0xfeedfaceu;
  compiled_arithmetic_state.vgprs[2][2] = 0xabcdef01u;
  compiled_arithmetic_state.vgprs[3][2] = 0x13579bdfu;
  if (!Expect(interpreter.ExecuteProgram(compiled_arithmetic_program,
                                         &compiled_arithmetic_state,
                                         &error_message),
              "expected compiled arithmetic/vector execution success") ||
      !Expect(ExpectArithmeticSeedProgramState(compiled_arithmetic_state),
              "expected compiled arithmetic/vector state")) {
    return 1;
  }

  const std::array<std::uint32_t, 8> compare_branch_eq_words{
      MakeSopk(0u, 10u, 5u),
      MakeSopk(0u, 11u, 5u),
      MakeSopc(6u, 10u, 11u),
      MakeSopp(34u, 2u),
      MakeSopk(0u, 12u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 12u, 222u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> compare_branch_eq_program;
  if (!Expect(decoder.DecodeProgram(compare_branch_eq_words,
                                    &compare_branch_eq_program, &error_message),
              "expected compare/branch EQ program decode success") ||
      !Expect(compare_branch_eq_program.size() == 8u,
              "expected eight decoded compare/branch EQ instructions") ||
      !Expect(compare_branch_eq_program[2].opcode == "S_CMP_EQ_U32",
              "expected decoded S_CMP_EQ_U32") ||
      !Expect(compare_branch_eq_program[3].opcode == "S_CBRANCH_SCC1",
              "expected decoded S_CBRANCH_SCC1") ||
      !Expect(compare_branch_eq_program[5].opcode == "S_BRANCH",
              "expected decoded S_BRANCH")) {
    return 1;
  }

  WaveExecutionState decoded_compare_branch_eq_state;
  if (!Expect(interpreter.ExecuteProgram(compare_branch_eq_program,
                                         &decoded_compare_branch_eq_state,
                                         &error_message),
              "expected decoded compare/branch EQ execution success") ||
      !Expect(ExpectCompareBranchEqState(decoded_compare_branch_eq_state),
              "expected decoded compare/branch EQ state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_compare_branch_eq_program;
  if (!Expect(interpreter.CompileProgram(compare_branch_eq_program,
                                         &compiled_compare_branch_eq_program,
                                         &error_message),
              "expected compiled compare/branch EQ program success") ||
      !Expect(compiled_compare_branch_eq_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCmpEqU32,
              "expected compiled S_CMP_EQ_U32 opcode") ||
      !Expect(compiled_compare_branch_eq_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchScc1,
              "expected compiled S_CBRANCH_SCC1 opcode") ||
      !Expect(compiled_compare_branch_eq_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSBranch,
              "expected compiled S_BRANCH opcode")) {
    return 1;
  }

  WaveExecutionState compiled_compare_branch_eq_state;
  if (!Expect(interpreter.ExecuteProgram(compiled_compare_branch_eq_program,
                                         &compiled_compare_branch_eq_state,
                                         &error_message),
              "expected compiled compare/branch EQ execution success") ||
      !Expect(ExpectCompareBranchEqState(compiled_compare_branch_eq_state),
              "expected compiled compare/branch EQ state")) {
    return 1;
  }

  const std::array<std::uint32_t, 9> branch_cmp_lg_words{
      MakeSopp(32u, 1u),
      MakeSopk(0u, 13u, 77u),
      MakeSopk(0u, 13u, 88u),
      MakeSopk(0u, 14u, 88u),
      MakeSopc(7u, 13u, 14u),
      MakeSopp(33u, 1u),
      MakeSopk(0u, 15u, 1u),
      MakeSopk(0u, 15u, 2u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> branch_cmp_lg_program;
  if (!Expect(decoder.DecodeProgram(branch_cmp_lg_words, &branch_cmp_lg_program,
                                    &error_message),
              "expected branch/compare LG program decode success") ||
      !Expect(branch_cmp_lg_program.size() == 9u,
              "expected nine decoded branch/compare LG instructions") ||
      !Expect(branch_cmp_lg_program[0].opcode == "S_BRANCH",
              "expected decoded leading S_BRANCH") ||
      !Expect(branch_cmp_lg_program[4].opcode == "S_CMP_LG_U32",
              "expected decoded S_CMP_LG_U32") ||
      !Expect(branch_cmp_lg_program[5].opcode == "S_CBRANCH_SCC0",
              "expected decoded S_CBRANCH_SCC0")) {
    return 1;
  }

  WaveExecutionState decoded_branch_cmp_lg_state;
  if (!Expect(interpreter.ExecuteProgram(branch_cmp_lg_program,
                                         &decoded_branch_cmp_lg_state,
                                         &error_message),
              "expected decoded branch/compare LG execution success") ||
      !Expect(ExpectBranchAndCmpLgState(decoded_branch_cmp_lg_state),
              "expected decoded branch/compare LG state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_branch_cmp_lg_program;
  if (!Expect(interpreter.CompileProgram(branch_cmp_lg_program,
                                         &compiled_branch_cmp_lg_program,
                                         &error_message),
              "expected compiled branch/compare LG program success") ||
      !Expect(compiled_branch_cmp_lg_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSBranch,
              "expected compiled leading S_BRANCH opcode") ||
      !Expect(compiled_branch_cmp_lg_program[4].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLgU32,
              "expected compiled S_CMP_LG_U32 opcode") ||
      !Expect(compiled_branch_cmp_lg_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchScc0,
              "expected compiled S_CBRANCH_SCC0 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_branch_cmp_lg_state;
  if (!Expect(interpreter.ExecuteProgram(compiled_branch_cmp_lg_program,
                                         &compiled_branch_cmp_lg_state,
                                         &error_message),
              "expected compiled branch/compare LG execution success") ||
      !Expect(ExpectBranchAndCmpLgState(compiled_branch_cmp_lg_state),
              "expected compiled branch/compare LG state")) {
    return 1;
  }

  const std::array<DecodedInstruction, 1> add_i32_program{
      DecodedInstruction::Binary("S_ADD_I32", InstructionOperand::Sgpr(12),
                                 InstructionOperand::Sgpr(10),
                                 InstructionOperand::Sgpr(11)),
  };
  WaveExecutionState add_i32_state;
  add_i32_state.sgprs[10] = 0xffffffffu;
  add_i32_state.sgprs[11] = 1u;
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
