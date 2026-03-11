#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

std::uint32_t FloatBits(float value) {
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

constexpr std::uint32_t ReverseBits32(std::uint32_t value) {
  value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
  value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
  value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
  value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
  return (value << 16) | (value >> 16);
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

bool ExpectExtendedCompareState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[20] == 0xffffffffu && state.sgprs[21] == 1u &&
         state.sgprs[24] == 11u && state.sgprs[25] == 33u &&
         state.sgprs[26] == 4u && state.sgprs[27] == 9u &&
         state.sgprs[28] == 55u && state.sgprs[29] == 77u && state.scc &&
         state.halted && !state.waiting_on_barrier && state.pc == 19u;
}

bool ExpectExecBranchState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[30] == 2u && state.sgprs[31] == 4u &&
         state.exec_mask == 5u && state.halted && !state.waiting_on_barrier &&
         state.pc == 10u;
}

bool ExpectConversionSeedState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[1][0] == 0xfffffff9u && state.vgprs[1][1] == 0xfffffff9u &&
         state.vgprs[1][2] == 0x11111111u && state.vgprs[1][3] == 0xfffffff9u &&
         state.vgprs[2][0] == 9u && state.vgprs[2][1] == 9u &&
         state.vgprs[2][2] == 0x22222222u && state.vgprs[2][3] == 9u &&
         state.vgprs[3][0] == FloatBits(3.75f) &&
         state.vgprs[3][1] == FloatBits(3.75f) &&
         state.vgprs[3][2] == 0x33333333u &&
         state.vgprs[3][3] == FloatBits(3.75f) &&
         state.vgprs[4][0] == FloatBits(-3.75f) &&
         state.vgprs[4][1] == FloatBits(-3.75f) &&
         state.vgprs[4][2] == 0x44444444u &&
         state.vgprs[4][3] == FloatBits(-3.75f) &&
         state.vgprs[5][0] == FloatBits(-7.0f) &&
         state.vgprs[5][1] == FloatBits(-7.0f) &&
         state.vgprs[5][2] == 0x55555555u &&
         state.vgprs[5][3] == FloatBits(-7.0f) &&
         state.vgprs[6][0] == FloatBits(9.0f) &&
         state.vgprs[6][1] == FloatBits(9.0f) &&
         state.vgprs[6][2] == 0x66666666u &&
         state.vgprs[6][3] == FloatBits(9.0f) &&
         state.vgprs[7][0] == 3u && state.vgprs[7][1] == 3u &&
         state.vgprs[7][2] == 0x77777777u && state.vgprs[7][3] == 3u &&
         state.vgprs[8][0] == 0xfffffffdu &&
         state.vgprs[8][1] == 0xfffffffdu &&
         state.vgprs[8][2] == 0x88888888u &&
         state.vgprs[8][3] == 0xfffffffdu && state.halted &&
         !state.waiting_on_barrier && state.pc == 8u;
}

bool ExpectRemainingCompareState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[40] == 0xffffffffu && state.sgprs[41] == 0xffffffffu &&
         state.sgprs[42] == 1u && state.sgprs[43] == 4u &&
         state.sgprs[44] == 9u && state.sgprs[80] == 10u &&
         state.sgprs[81] == 11u && state.sgprs[82] == 12u &&
         state.sgprs[83] == 13u && state.sgprs[84] == 14u &&
         state.sgprs[85] == 15u && state.scc && state.halted &&
         !state.waiting_on_barrier && state.pc == 29u;
}

bool ExpectVcczBranchState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[60] == 2u && state.vcc_mask == 0u && state.halted &&
         !state.waiting_on_barrier && state.pc == 4u;
}

bool ExpectVccnzBranchState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[61] == 4u && state.vcc_mask == 1u && state.halted &&
         !state.waiting_on_barrier && state.pc == 4u;
}

bool ExpectVectorUnaryBatchState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[10][0] == 0x01020380u &&
         state.vgprs[10][1] == 0x01020380u &&
         state.vgprs[10][2] == 0x10101010u &&
         state.vgprs[10][3] == 0x01020380u &&
         state.vgprs[11][0] == 0xfefdfc7fu &&
         state.vgprs[11][1] == 0xfefdfc7fu &&
         state.vgprs[11][2] == 0x11111111u &&
         state.vgprs[11][3] == 0xfefdfc7fu &&
         state.vgprs[12][0] == ReverseBits32(0x01020380u) &&
         state.vgprs[12][1] == ReverseBits32(0x01020380u) &&
         state.vgprs[12][2] == 0x12121212u &&
         state.vgprs[12][3] == ReverseBits32(0x01020380u) &&
         state.vgprs[13][0] == FloatBits(128.0f) &&
         state.vgprs[13][1] == FloatBits(128.0f) &&
         state.vgprs[13][2] == 0x13131313u &&
         state.vgprs[13][3] == FloatBits(128.0f) &&
         state.vgprs[14][0] == FloatBits(3.0f) &&
         state.vgprs[14][1] == FloatBits(3.0f) &&
         state.vgprs[14][2] == 0x14141414u &&
         state.vgprs[14][3] == FloatBits(3.0f) &&
         state.vgprs[15][0] == FloatBits(2.0f) &&
         state.vgprs[15][1] == FloatBits(2.0f) &&
         state.vgprs[15][2] == 0x15151515u &&
         state.vgprs[15][3] == FloatBits(2.0f) &&
         state.vgprs[16][0] == FloatBits(1.0f) &&
         state.vgprs[16][1] == FloatBits(1.0f) &&
         state.vgprs[16][2] == 0x16161616u &&
         state.vgprs[16][3] == FloatBits(1.0f) && state.halted &&
         !state.waiting_on_barrier && state.pc == 7u;
}

bool ExpectVectorBinaryBatchState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[30][0] == 4u && state.vgprs[30][1] == 4u &&
         state.vgprs[30][2] == 0x30303030u && state.vgprs[30][3] == 4u &&
         state.vgprs[31][0] == 0xfffffff0u &&
         state.vgprs[31][1] == 0xfffffff0u &&
         state.vgprs[31][2] == 0x31313131u &&
         state.vgprs[31][3] == 0xfffffff0u &&
         state.vgprs[32][0] == 9u && state.vgprs[32][1] == 9u &&
         state.vgprs[32][2] == 0x32323232u && state.vgprs[32][3] == 9u &&
         state.vgprs[33][0] == 9u && state.vgprs[33][1] == 9u &&
         state.vgprs[33][2] == 0x33333333u && state.vgprs[33][3] == 9u &&
         state.vgprs[34][0] == 0xfffffff0u &&
         state.vgprs[34][1] == 0xfffffff0u &&
         state.vgprs[34][2] == 0x34343434u &&
         state.vgprs[34][3] == 0xfffffff0u &&
         state.vgprs[35][0] == 0x003fc03eu &&
         state.vgprs[35][1] == 0x003fc03eu &&
         state.vgprs[35][2] == 0x35353535u &&
         state.vgprs[35][3] == 0x003fc03eu &&
         state.vgprs[36][0] == 0xfffffffcu &&
         state.vgprs[36][1] == 0xfffffffcu &&
         state.vgprs[36][2] == 0x36363636u &&
         state.vgprs[36][3] == 0xfffffffcu &&
         state.vgprs[37][0] == 72u && state.vgprs[37][1] == 72u &&
         state.vgprs[37][2] == 0x37373737u && state.vgprs[37][3] == 72u &&
         state.vgprs[38][0] == 8u && state.vgprs[38][1] == 8u &&
         state.vgprs[38][2] == 0x38383838u && state.vgprs[38][3] == 8u &&
         state.vgprs[39][0] == 0x00ff00f9u &&
         state.vgprs[39][1] == 0x00ff00f9u &&
         state.vgprs[39][2] == 0x39393939u &&
         state.vgprs[39][3] == 0x00ff00f9u &&
         state.vgprs[40][0] == 0x00ff00f1u &&
         state.vgprs[40][1] == 0x00ff00f1u &&
         state.vgprs[40][2] == 0x40404040u &&
         state.vgprs[40][3] == 0x00ff00f1u && state.halted &&
         !state.waiting_on_barrier && state.pc == 15u;
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

  const std::array<std::uint32_t, 1> scalar_cmp_ge_i32_words{
      MakeSopc(3u, 5u, 6u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_ge_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_GE_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_GE_I32",
                                       OperandKind::kSgpr, 5u,
                                       OperandKind::kSgpr, 6u),
              "expected decoded S_CMP_GE_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_eq_i32_words{
      MakeSopc(0u, 5u, 6u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_eq_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_EQ_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_EQ_I32",
                                       OperandKind::kSgpr, 5u,
                                       OperandKind::kSgpr, 6u),
              "expected decoded S_CMP_EQ_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_lg_i32_words{
      MakeSopc(1u, 5u, 6u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_lg_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LG_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LG_I32",
                                       OperandKind::kSgpr, 5u,
                                       OperandKind::kSgpr, 6u),
              "expected decoded S_CMP_LG_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_gt_i32_words{
      MakeSopc(2u, 5u, 6u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_gt_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_GT_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_GT_I32",
                                       OperandKind::kSgpr, 5u,
                                       OperandKind::kSgpr, 6u),
              "expected decoded S_CMP_GT_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_lt_i32_words{
      MakeSopc(4u, 7u, 8u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_lt_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LT_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LT_I32",
                                       OperandKind::kSgpr, 7u,
                                       OperandKind::kSgpr, 8u),
              "expected decoded S_CMP_LT_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_le_i32_words{
      MakeSopc(5u, 7u, 8u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_le_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LE_I32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LE_I32",
                                       OperandKind::kSgpr, 7u,
                                       OperandKind::kSgpr, 8u),
              "expected decoded S_CMP_LE_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_ge_u32_words{
      MakeSopc(9u, 9u, 10u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_ge_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_GE_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_GE_U32",
                                       OperandKind::kSgpr, 9u,
                                       OperandKind::kSgpr, 10u),
              "expected decoded S_CMP_GE_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_gt_u32_words{
      MakeSopc(8u, 9u, 10u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_gt_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_GT_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_GT_U32",
                                       OperandKind::kSgpr, 9u,
                                       OperandKind::kSgpr, 10u),
              "expected decoded S_CMP_GT_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_lt_u32_words{
      MakeSopc(10u, 11u, 12u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_lt_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LT_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LT_U32",
                                       OperandKind::kSgpr, 11u,
                                       OperandKind::kSgpr, 12u),
              "expected decoded S_CMP_LT_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scalar_cmp_le_u32_words{
      MakeSopc(11u, 11u, 12u)};
  if (!Expect(decoder.DecodeInstruction(scalar_cmp_le_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CMP_LE_U32 decode success") ||
      !Expect(ExpectCompareInstruction(instruction, "S_CMP_LE_U32",
                                       OperandKind::kSgpr, 11u,
                                       OperandKind::kSgpr, 12u),
              "expected decoded S_CMP_LE_U32 operands")) {
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

  const std::array<std::uint32_t, 1> cbranch_vccz_words{MakeSopp(35u, 2u)};
  if (!Expect(decoder.DecodeInstruction(cbranch_vccz_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_VCCZ decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_VCCZ", 2u),
              "expected decoded S_CBRANCH_VCCZ immediate")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> cbranch_vccnz_words{MakeSopp(36u, 1u)};
  if (!Expect(decoder.DecodeInstruction(cbranch_vccnz_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_VCCNZ decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_VCCNZ", 1u),
              "expected decoded S_CBRANCH_VCCNZ immediate")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> cbranch_execz_words{MakeSopp(37u, 2u)};
  if (!Expect(decoder.DecodeInstruction(cbranch_execz_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_EXECZ decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_EXECZ", 2u),
              "expected decoded S_CBRANCH_EXECZ immediate")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> cbranch_execnz_words{
      MakeSopp(38u, 0xfffeu)};
  if (!Expect(decoder.DecodeInstruction(cbranch_execnz_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_CBRANCH_EXECNZ decode success") ||
      !Expect(ExpectBranchInstruction(instruction, "S_CBRANCH_EXECNZ",
                                      0xfffffffeu),
              "expected decoded S_CBRANCH_EXECNZ immediate")) {
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

  const std::array<std::uint32_t, 1> vector_not_words{MakeVop1(55u, 4u, 257u)};
  if (!Expect(decoder.DecodeInstruction(vector_not_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_NOT_B32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_NOT_B32",
                                     OperandKind::kVgpr, 4u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_NOT_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_bfrev_words{MakeVop1(56u, 5u, 258u)};
  if (!Expect(decoder.DecodeInstruction(vector_bfrev_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_BFREV_B32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_BFREV_B32",
                                     OperandKind::kVgpr, 5u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_BFREV_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_ubyte0_words{
      MakeVop1(17u, 6u, 259u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_ubyte0_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_UBYTE0 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_UBYTE0",
                                     OperandKind::kVgpr, 6u,
                                     OperandKind::kVgpr, 3u),
              "expected decoded V_CVT_F32_UBYTE0 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_ubyte3_words{
      MakeVop1(20u, 9u, 260u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_ubyte3_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_UBYTE3 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_UBYTE3",
                                     OperandKind::kVgpr, 9u,
                                     OperandKind::kVgpr, 4u),
              "expected decoded V_CVT_F32_UBYTE3 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_i32_words{MakeVop1(5u, 6u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_I32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_I32",
                                     OperandKind::kVgpr, 6u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_F32_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_u32_words{MakeVop1(6u, 7u, 258u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_U32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_U32",
                                     OperandKind::kVgpr, 7u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_CVT_F32_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_u32_f32_words{MakeVop1(7u, 8u, 259u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_u32_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_U32_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_U32_F32",
                                     OperandKind::kVgpr, 8u,
                                     OperandKind::kVgpr, 3u),
              "expected decoded V_CVT_U32_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_i32_f32_words{MakeVop1(8u, 9u, 260u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_i32_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_I32_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_I32_F32",
                                     OperandKind::kVgpr, 9u,
                                     OperandKind::kVgpr, 4u),
              "expected decoded V_CVT_I32_F32 operands")) {
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

  const std::array<std::uint32_t, 1> vector_subrev_words{
      MakeVop2(39u, 6u, 257u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_subrev_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SUBREV_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_SUBREV_U32",
                                      OperandKind::kVgpr, 6u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_SUBREV_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_min_i32_words{
      MakeVop2(17u, 7u, 257u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_min_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MIN_I32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MIN_I32",
                                      OperandKind::kVgpr, 7u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_MIN_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_lshrrev_words{
      MakeVop2(25u, 8u, 130u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_lshrrev_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_LSHRREV_B32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_LSHRREV_B32",
                                      OperandKind::kVgpr, 8u,
                                      OperandKind::kImm32, 2u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_LSHRREV_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_xor_words{
      MakeVop2(29u, 9u, 257u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_xor_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_XOR_B32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_XOR_B32",
                                      OperandKind::kVgpr, 9u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_XOR_B32 operands")) {
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

  const std::array<std::uint32_t, 20> extended_compare_words{
      MakeSopk(0u, 20u, 0xffffu),
      MakeSopk(0u, 21u, 1u),
      MakeSopc(4u, 20u, 21u),
      MakeSopp(33u, 2u),
      MakeSopk(0u, 24u, 11u),
      MakeSopc(3u, 20u, 21u),
      MakeSopp(33u, 1u),
      MakeSopk(0u, 25u, 22u),
      MakeSopk(0u, 25u, 33u),
      MakeSopk(0u, 26u, 4u),
      MakeSopk(0u, 27u, 9u),
      MakeSopc(10u, 26u, 27u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 28u, 44u),
      MakeSopk(0u, 28u, 55u),
      MakeSopc(9u, 27u, 26u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 29u, 66u),
      MakeSopk(0u, 29u, 77u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> extended_compare_program;
  if (!Expect(decoder.DecodeProgram(extended_compare_words,
                                    &extended_compare_program, &error_message),
              "expected extended compare program decode success") ||
      !Expect(extended_compare_program.size() == 20u,
              "expected twenty decoded extended compare instructions") ||
      !Expect(extended_compare_program[2].opcode == "S_CMP_LT_I32",
              "expected decoded S_CMP_LT_I32") ||
      !Expect(extended_compare_program[5].opcode == "S_CMP_GE_I32",
              "expected decoded S_CMP_GE_I32") ||
      !Expect(extended_compare_program[11].opcode == "S_CMP_LT_U32",
              "expected decoded S_CMP_LT_U32") ||
      !Expect(extended_compare_program[15].opcode == "S_CMP_GE_U32",
              "expected decoded S_CMP_GE_U32")) {
    return 1;
  }

  WaveExecutionState decoded_extended_compare_state;
  if (!Expect(interpreter.ExecuteProgram(extended_compare_program,
                                         &decoded_extended_compare_state,
                                         &error_message),
              "expected decoded extended compare execution success") ||
      !Expect(ExpectExtendedCompareState(decoded_extended_compare_state),
              "expected decoded extended compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_extended_compare_program;
  if (!Expect(interpreter.CompileProgram(extended_compare_program,
                                         &compiled_extended_compare_program,
                                         &error_message),
              "expected compiled extended compare program success") ||
      !Expect(compiled_extended_compare_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLtI32,
              "expected compiled S_CMP_LT_I32 opcode") ||
      !Expect(compiled_extended_compare_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSCmpGeI32,
              "expected compiled S_CMP_GE_I32 opcode") ||
      !Expect(compiled_extended_compare_program[11].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLtU32,
              "expected compiled S_CMP_LT_U32 opcode") ||
      !Expect(compiled_extended_compare_program[15].opcode ==
                  Gfx1201CompiledOpcode::kSCmpGeU32,
              "expected compiled S_CMP_GE_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_extended_compare_state;
  if (!Expect(interpreter.ExecuteProgram(compiled_extended_compare_program,
                                         &compiled_extended_compare_state,
                                         &error_message),
              "expected compiled extended compare execution success") ||
      !Expect(ExpectExtendedCompareState(compiled_extended_compare_state),
              "expected compiled extended compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 11> exec_branch_words{
      MakeSopk(0u, 126u, 0u),
      MakeSopk(0u, 127u, 0u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 30u, 1u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 30u, 2u),
      MakeSopk(0u, 126u, 5u),
      MakeSopp(38u, 1u),
      MakeSopk(0u, 31u, 3u),
      MakeSopk(0u, 31u, 4u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> exec_branch_program;
  if (!Expect(decoder.DecodeProgram(exec_branch_words, &exec_branch_program,
                                    &error_message),
              "expected exec branch program decode success") ||
      !Expect(exec_branch_program.size() == 11u,
              "expected eleven decoded exec branch instructions") ||
      !Expect(exec_branch_program[2].opcode == "S_CBRANCH_EXECZ",
              "expected decoded S_CBRANCH_EXECZ") ||
      !Expect(exec_branch_program[7].opcode == "S_CBRANCH_EXECNZ",
              "expected decoded S_CBRANCH_EXECNZ")) {
    return 1;
  }

  WaveExecutionState decoded_exec_branch_state;
  if (!Expect(interpreter.ExecuteProgram(exec_branch_program,
                                         &decoded_exec_branch_state,
                                         &error_message),
              "expected decoded exec branch execution success") ||
      !Expect(ExpectExecBranchState(decoded_exec_branch_state),
              "expected decoded exec branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_exec_branch_program;
  if (!Expect(interpreter.CompileProgram(exec_branch_program,
                                         &compiled_exec_branch_program,
                                         &error_message),
              "expected compiled exec branch program success") ||
      !Expect(compiled_exec_branch_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchExecz,
              "expected compiled S_CBRANCH_EXECZ opcode") ||
      !Expect(compiled_exec_branch_program[7].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchExecnz,
              "expected compiled S_CBRANCH_EXECNZ opcode")) {
    return 1;
  }

  WaveExecutionState compiled_exec_branch_state;
  if (!Expect(interpreter.ExecuteProgram(compiled_exec_branch_program,
                                         &compiled_exec_branch_state,
                                         &error_message),
              "expected compiled exec branch execution success") ||
      !Expect(ExpectExecBranchState(compiled_exec_branch_state),
              "expected compiled exec branch state")) {
    return 1;
  }

  const std::array<std::uint32_t, 13> conversion_words{
      MakeVop1(1u, 1u, 255u),
      0xfffffff9u,
      MakeVop1(5u, 5u, 257u),
      MakeVop1(1u, 2u, 255u),
      9u,
      MakeVop1(6u, 6u, 258u),
      MakeVop1(1u, 3u, 255u),
      0x40700000u,
      MakeVop1(7u, 7u, 259u),
      MakeVop1(1u, 4u, 255u),
      0xc0700000u,
      MakeVop1(8u, 8u, 260u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> conversion_program;
  if (!Expect(decoder.DecodeProgram(conversion_words, &conversion_program,
                                    &error_message),
              "expected conversion program decode success") ||
      !Expect(conversion_program.size() == 9u,
              "expected nine decoded conversion instructions") ||
      !Expect(conversion_program[1].opcode == "V_CVT_F32_I32",
              "expected decoded V_CVT_F32_I32") ||
      !Expect(conversion_program[3].opcode == "V_CVT_F32_U32",
              "expected decoded V_CVT_F32_U32") ||
      !Expect(conversion_program[5].opcode == "V_CVT_U32_F32",
              "expected decoded V_CVT_U32_F32") ||
      !Expect(conversion_program[7].opcode == "V_CVT_I32_F32",
              "expected decoded V_CVT_I32_F32")) {
    return 1;
  }

  WaveExecutionState decoded_conversion_state;
  decoded_conversion_state.exec_mask = 0xbu;
  decoded_conversion_state.vgprs[1][2] = 0x11111111u;
  decoded_conversion_state.vgprs[2][2] = 0x22222222u;
  decoded_conversion_state.vgprs[3][2] = 0x33333333u;
  decoded_conversion_state.vgprs[4][2] = 0x44444444u;
  decoded_conversion_state.vgprs[5][2] = 0x55555555u;
  decoded_conversion_state.vgprs[6][2] = 0x66666666u;
  decoded_conversion_state.vgprs[7][2] = 0x77777777u;
  decoded_conversion_state.vgprs[8][2] = 0x88888888u;
  if (!Expect(interpreter.ExecuteProgram(conversion_program,
                                         &decoded_conversion_state,
                                         &error_message),
              "expected decoded conversion execution success") ||
      !Expect(ExpectConversionSeedState(decoded_conversion_state),
              "expected decoded conversion state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_conversion_program;
  if (!Expect(interpreter.CompileProgram(conversion_program,
                                         &compiled_conversion_program,
                                         &error_message),
              "expected compiled conversion program success") ||
      !Expect(compiled_conversion_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32I32,
              "expected compiled V_CVT_F32_I32 opcode") ||
      !Expect(compiled_conversion_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32U32,
              "expected compiled V_CVT_F32_U32 opcode") ||
      !Expect(compiled_conversion_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVCvtU32F32,
              "expected compiled V_CVT_U32_F32 opcode") ||
      !Expect(compiled_conversion_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCvtI32F32,
              "expected compiled V_CVT_I32_F32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_conversion_state;
  compiled_conversion_state.exec_mask = 0xbu;
  compiled_conversion_state.vgprs[1][2] = 0x11111111u;
  compiled_conversion_state.vgprs[2][2] = 0x22222222u;
  compiled_conversion_state.vgprs[3][2] = 0x33333333u;
  compiled_conversion_state.vgprs[4][2] = 0x44444444u;
  compiled_conversion_state.vgprs[5][2] = 0x55555555u;
  compiled_conversion_state.vgprs[6][2] = 0x66666666u;
  compiled_conversion_state.vgprs[7][2] = 0x77777777u;
  compiled_conversion_state.vgprs[8][2] = 0x88888888u;
  if (!Expect(interpreter.ExecuteProgram(compiled_conversion_program,
                                         &compiled_conversion_state,
                                         &error_message),
              "expected compiled conversion execution success") ||
      !Expect(ExpectConversionSeedState(compiled_conversion_state),
              "expected compiled conversion state")) {
    return 1;
  }

  const std::array<std::uint32_t, 30> remaining_compare_words{
      MakeSopk(0u, 40u, 0xffffu),
      MakeSopk(0u, 41u, 0xffffu),
      MakeSopk(0u, 42u, 1u),
      MakeSopk(0u, 43u, 4u),
      MakeSopk(0u, 44u, 9u),
      MakeSopc(0u, 40u, 41u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 80u, 100u),
      MakeSopk(0u, 80u, 10u),
      MakeSopc(1u, 40u, 42u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 81u, 101u),
      MakeSopk(0u, 81u, 11u),
      MakeSopc(2u, 42u, 40u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 82u, 102u),
      MakeSopk(0u, 82u, 12u),
      MakeSopc(5u, 40u, 41u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 83u, 103u),
      MakeSopk(0u, 83u, 13u),
      MakeSopc(8u, 44u, 43u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 84u, 104u),
      MakeSopk(0u, 84u, 14u),
      MakeSopc(11u, 43u, 44u),
      MakeSopp(34u, 1u),
      MakeSopk(0u, 85u, 105u),
      MakeSopk(0u, 85u, 15u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> remaining_compare_program;
  if (!Expect(decoder.DecodeProgram(remaining_compare_words,
                                    &remaining_compare_program, &error_message),
              "expected remaining compare program decode success") ||
      !Expect(remaining_compare_program.size() == 30u,
              "expected thirty decoded remaining compare instructions") ||
      !Expect(remaining_compare_program[5].opcode == "S_CMP_EQ_I32",
              "expected decoded S_CMP_EQ_I32") ||
      !Expect(remaining_compare_program[9].opcode == "S_CMP_LG_I32",
              "expected decoded S_CMP_LG_I32") ||
      !Expect(remaining_compare_program[13].opcode == "S_CMP_GT_I32",
              "expected decoded S_CMP_GT_I32") ||
      !Expect(remaining_compare_program[17].opcode == "S_CMP_LE_I32",
              "expected decoded S_CMP_LE_I32") ||
      !Expect(remaining_compare_program[21].opcode == "S_CMP_GT_U32",
              "expected decoded S_CMP_GT_U32") ||
      !Expect(remaining_compare_program[25].opcode == "S_CMP_LE_U32",
              "expected decoded S_CMP_LE_U32")) {
    return 1;
  }

  WaveExecutionState decoded_remaining_compare_state;
  if (!Expect(interpreter.ExecuteProgram(remaining_compare_program,
                                         &decoded_remaining_compare_state,
                                         &error_message),
              "expected decoded remaining compare execution success") ||
      !Expect(ExpectRemainingCompareState(decoded_remaining_compare_state),
              "expected decoded remaining compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_remaining_compare_program;
  if (!Expect(interpreter.CompileProgram(remaining_compare_program,
                                         &compiled_remaining_compare_program,
                                         &error_message),
              "expected compiled remaining compare program success") ||
      !Expect(compiled_remaining_compare_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSCmpEqI32,
              "expected compiled S_CMP_EQ_I32 opcode") ||
      !Expect(compiled_remaining_compare_program[9].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLgI32,
              "expected compiled S_CMP_LG_I32 opcode") ||
      !Expect(compiled_remaining_compare_program[13].opcode ==
                  Gfx1201CompiledOpcode::kSCmpGtI32,
              "expected compiled S_CMP_GT_I32 opcode") ||
      !Expect(compiled_remaining_compare_program[17].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLeI32,
              "expected compiled S_CMP_LE_I32 opcode") ||
      !Expect(compiled_remaining_compare_program[21].opcode ==
                  Gfx1201CompiledOpcode::kSCmpGtU32,
              "expected compiled S_CMP_GT_U32 opcode") ||
      !Expect(compiled_remaining_compare_program[25].opcode ==
                  Gfx1201CompiledOpcode::kSCmpLeU32,
              "expected compiled S_CMP_LE_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_remaining_compare_state;
  if (!Expect(interpreter.ExecuteProgram(compiled_remaining_compare_program,
                                         &compiled_remaining_compare_state,
                                         &error_message),
              "expected compiled remaining compare execution success") ||
      !Expect(ExpectRemainingCompareState(compiled_remaining_compare_state),
              "expected compiled remaining compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> vccz_branch_words{
      MakeSopp(35u, 2u),
      MakeSopk(0u, 60u, 1u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 60u, 2u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> vccz_branch_program;
  if (!Expect(decoder.DecodeProgram(vccz_branch_words, &vccz_branch_program,
                                    &error_message),
              "expected VCCZ branch program decode success") ||
      !Expect(vccz_branch_program.size() == 5u,
              "expected five decoded VCCZ branch instructions") ||
      !Expect(vccz_branch_program[0].opcode == "S_CBRANCH_VCCZ",
              "expected decoded S_CBRANCH_VCCZ")) {
    return 1;
  }

  WaveExecutionState decoded_vccz_branch_state;
  decoded_vccz_branch_state.vcc_mask = 0u;
  if (!Expect(interpreter.ExecuteProgram(vccz_branch_program,
                                         &decoded_vccz_branch_state,
                                         &error_message),
              "expected decoded VCCZ branch execution success") ||
      !Expect(ExpectVcczBranchState(decoded_vccz_branch_state),
              "expected decoded VCCZ branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_vccz_branch_program;
  if (!Expect(interpreter.CompileProgram(vccz_branch_program,
                                         &compiled_vccz_branch_program,
                                         &error_message),
              "expected compiled VCCZ branch program success") ||
      !Expect(compiled_vccz_branch_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchVccz,
              "expected compiled S_CBRANCH_VCCZ opcode")) {
    return 1;
  }

  WaveExecutionState compiled_vccz_branch_state;
  compiled_vccz_branch_state.vcc_mask = 0u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vccz_branch_program,
                                         &compiled_vccz_branch_state,
                                         &error_message),
              "expected compiled VCCZ branch execution success") ||
      !Expect(ExpectVcczBranchState(compiled_vccz_branch_state),
              "expected compiled VCCZ branch state")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> vccnz_branch_words{
      MakeSopp(36u, 2u),
      MakeSopk(0u, 61u, 3u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 61u, 4u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> vccnz_branch_program;
  if (!Expect(decoder.DecodeProgram(vccnz_branch_words, &vccnz_branch_program,
                                    &error_message),
              "expected VCCNZ branch program decode success") ||
      !Expect(vccnz_branch_program.size() == 5u,
              "expected five decoded VCCNZ branch instructions") ||
      !Expect(vccnz_branch_program[0].opcode == "S_CBRANCH_VCCNZ",
              "expected decoded S_CBRANCH_VCCNZ")) {
    return 1;
  }

  WaveExecutionState decoded_vccnz_branch_state;
  decoded_vccnz_branch_state.vcc_mask = 1u;
  if (!Expect(interpreter.ExecuteProgram(vccnz_branch_program,
                                         &decoded_vccnz_branch_state,
                                         &error_message),
              "expected decoded VCCNZ branch execution success") ||
      !Expect(ExpectVccnzBranchState(decoded_vccnz_branch_state),
              "expected decoded VCCNZ branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_vccnz_branch_program;
  if (!Expect(interpreter.CompileProgram(vccnz_branch_program,
                                         &compiled_vccnz_branch_program,
                                         &error_message),
              "expected compiled VCCNZ branch program success") ||
      !Expect(compiled_vccnz_branch_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchVccnz,
              "expected compiled S_CBRANCH_VCCNZ opcode")) {
    return 1;
  }

  WaveExecutionState compiled_vccnz_branch_state;
  compiled_vccnz_branch_state.vcc_mask = 1u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vccnz_branch_program,
                                         &compiled_vccnz_branch_state,
                                         &error_message),
              "expected compiled VCCNZ branch execution success") ||
      !Expect(ExpectVccnzBranchState(compiled_vccnz_branch_state),
              "expected compiled VCCNZ branch state")) {
    return 1;
  }

  const std::array<std::uint32_t, 9> vector_unary_words{
      MakeVop1(1u, 10u, 255u),
      0x01020380u,
      MakeVop1(55u, 11u, 266u),
      MakeVop1(56u, 12u, 266u),
      MakeVop1(17u, 13u, 266u),
      MakeVop1(18u, 14u, 266u),
      MakeVop1(19u, 15u, 266u),
      MakeVop1(20u, 16u, 266u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> vector_unary_program;
  if (!Expect(decoder.DecodeProgram(vector_unary_words, &vector_unary_program,
                                    &error_message),
              "expected vector unary batch decode success") ||
      !Expect(vector_unary_program.size() == 8u,
              "expected eight decoded vector unary instructions") ||
      !Expect(vector_unary_program[1].opcode == "V_NOT_B32",
              "expected decoded V_NOT_B32") ||
      !Expect(vector_unary_program[2].opcode == "V_BFREV_B32",
              "expected decoded V_BFREV_B32") ||
      !Expect(vector_unary_program[3].opcode == "V_CVT_F32_UBYTE0",
              "expected decoded V_CVT_F32_UBYTE0") ||
      !Expect(vector_unary_program[6].opcode == "V_CVT_F32_UBYTE3",
              "expected decoded V_CVT_F32_UBYTE3")) {
    return 1;
  }

  WaveExecutionState decoded_vector_unary_state;
  decoded_vector_unary_state.exec_mask = 0xbu;
  decoded_vector_unary_state.vgprs[10][2] = 0x10101010u;
  decoded_vector_unary_state.vgprs[11][2] = 0x11111111u;
  decoded_vector_unary_state.vgprs[12][2] = 0x12121212u;
  decoded_vector_unary_state.vgprs[13][2] = 0x13131313u;
  decoded_vector_unary_state.vgprs[14][2] = 0x14141414u;
  decoded_vector_unary_state.vgprs[15][2] = 0x15151515u;
  decoded_vector_unary_state.vgprs[16][2] = 0x16161616u;
  if (!Expect(interpreter.ExecuteProgram(vector_unary_program,
                                         &decoded_vector_unary_state,
                                         &error_message),
              "expected decoded vector unary execution success") ||
      !Expect(ExpectVectorUnaryBatchState(decoded_vector_unary_state),
              "expected decoded vector unary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_vector_unary_program;
  if (!Expect(interpreter.CompileProgram(vector_unary_program,
                                         &compiled_vector_unary_program,
                                         &error_message),
              "expected compiled vector unary program success") ||
      !Expect(compiled_vector_unary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVNotB32,
              "expected compiled V_NOT_B32 opcode") ||
      !Expect(compiled_vector_unary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVBfrevB32,
              "expected compiled V_BFREV_B32 opcode") ||
      !Expect(compiled_vector_unary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32Ubyte0,
              "expected compiled V_CVT_F32_UBYTE0 opcode") ||
      !Expect(compiled_vector_unary_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32Ubyte3,
              "expected compiled V_CVT_F32_UBYTE3 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_vector_unary_state;
  compiled_vector_unary_state.exec_mask = 0xbu;
  compiled_vector_unary_state.vgprs[10][2] = 0x10101010u;
  compiled_vector_unary_state.vgprs[11][2] = 0x11111111u;
  compiled_vector_unary_state.vgprs[12][2] = 0x12121212u;
  compiled_vector_unary_state.vgprs[13][2] = 0x13131313u;
  compiled_vector_unary_state.vgprs[14][2] = 0x14141414u;
  compiled_vector_unary_state.vgprs[15][2] = 0x15151515u;
  compiled_vector_unary_state.vgprs[16][2] = 0x16161616u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_unary_program,
                                         &compiled_vector_unary_state,
                                         &error_message),
              "expected compiled vector unary execution success") ||
      !Expect(ExpectVectorUnaryBatchState(compiled_vector_unary_state),
              "expected compiled vector unary state")) {
    return 1;
  }

  const std::array<std::uint32_t, 17> vector_binary_words{
      MakeVop1(1u, 20u, 133u),
      MakeVop1(1u, 21u, 137u),
      MakeVop1(1u, 22u, 208u),
      MakeVop1(1u, 23u, 255u),
      0x00ff00f8u,
      MakeVop2(39u, 30u, 276u, 21u),
      MakeVop2(17u, 31u, 278u, 21u),
      MakeVop2(18u, 32u, 278u, 21u),
      MakeVop2(19u, 33u, 278u, 21u),
      MakeVop2(20u, 34u, 278u, 21u),
      MakeVop2(25u, 35u, 130u, 23u),
      MakeVop2(26u, 36u, 130u, 22u),
      MakeVop2(24u, 37u, 131u, 21u),
      MakeVop2(27u, 38u, 279u, 21u),
      MakeVop2(28u, 39u, 279u, 21u),
      MakeVop2(29u, 40u, 279u, 21u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> vector_binary_program;
  if (!Expect(decoder.DecodeProgram(vector_binary_words, &vector_binary_program,
                                    &error_message),
              "expected vector binary batch decode success") ||
      !Expect(vector_binary_program.size() == 16u,
              "expected sixteen decoded vector binary instructions") ||
      !Expect(vector_binary_program[4].opcode == "V_SUBREV_U32",
              "expected decoded V_SUBREV_U32") ||
      !Expect(vector_binary_program[5].opcode == "V_MIN_I32",
              "expected decoded V_MIN_I32") ||
      !Expect(vector_binary_program[9].opcode == "V_LSHRREV_B32",
              "expected decoded V_LSHRREV_B32") ||
      !Expect(vector_binary_program[14].opcode == "V_XOR_B32",
              "expected decoded V_XOR_B32")) {
    return 1;
  }

  WaveExecutionState decoded_vector_binary_state;
  decoded_vector_binary_state.exec_mask = 0xbu;
  decoded_vector_binary_state.vgprs[30][2] = 0x30303030u;
  decoded_vector_binary_state.vgprs[31][2] = 0x31313131u;
  decoded_vector_binary_state.vgprs[32][2] = 0x32323232u;
  decoded_vector_binary_state.vgprs[33][2] = 0x33333333u;
  decoded_vector_binary_state.vgprs[34][2] = 0x34343434u;
  decoded_vector_binary_state.vgprs[35][2] = 0x35353535u;
  decoded_vector_binary_state.vgprs[36][2] = 0x36363636u;
  decoded_vector_binary_state.vgprs[37][2] = 0x37373737u;
  decoded_vector_binary_state.vgprs[38][2] = 0x38383838u;
  decoded_vector_binary_state.vgprs[39][2] = 0x39393939u;
  decoded_vector_binary_state.vgprs[40][2] = 0x40404040u;
  if (!Expect(interpreter.ExecuteProgram(vector_binary_program,
                                         &decoded_vector_binary_state,
                                         &error_message),
              "expected decoded vector binary execution success") ||
      !Expect(ExpectVectorBinaryBatchState(decoded_vector_binary_state),
              "expected decoded vector binary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_vector_binary_program;
  if (!Expect(interpreter.CompileProgram(vector_binary_program,
                                         &compiled_vector_binary_program,
                                         &error_message),
              "expected compiled vector binary program success") ||
      !Expect(compiled_vector_binary_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVSubrevU32,
              "expected compiled V_SUBREV_U32 opcode") ||
      !Expect(compiled_vector_binary_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVMinI32,
              "expected compiled V_MIN_I32 opcode") ||
      !Expect(compiled_vector_binary_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVMaxI32,
              "expected compiled V_MAX_I32 opcode") ||
      !Expect(compiled_vector_binary_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVMinU32,
              "expected compiled V_MIN_U32 opcode") ||
      !Expect(compiled_vector_binary_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVMaxU32,
              "expected compiled V_MAX_U32 opcode") ||
      !Expect(compiled_vector_binary_program[9].opcode ==
                  Gfx1201CompiledOpcode::kVLshrrevB32,
              "expected compiled V_LSHRREV_B32 opcode") ||
      !Expect(compiled_vector_binary_program[10].opcode ==
                  Gfx1201CompiledOpcode::kVAshrrevI32,
              "expected compiled V_ASHRREV_I32 opcode") ||
      !Expect(compiled_vector_binary_program[11].opcode ==
                  Gfx1201CompiledOpcode::kVLshlrevB32,
              "expected compiled V_LSHLREV_B32 opcode") ||
      !Expect(compiled_vector_binary_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVAndB32,
              "expected compiled V_AND_B32 opcode") ||
      !Expect(compiled_vector_binary_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVOrB32,
              "expected compiled V_OR_B32 opcode") ||
      !Expect(compiled_vector_binary_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVXorB32,
              "expected compiled V_XOR_B32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_vector_binary_state;
  compiled_vector_binary_state.exec_mask = 0xbu;
  compiled_vector_binary_state.vgprs[30][2] = 0x30303030u;
  compiled_vector_binary_state.vgprs[31][2] = 0x31313131u;
  compiled_vector_binary_state.vgprs[32][2] = 0x32323232u;
  compiled_vector_binary_state.vgprs[33][2] = 0x33333333u;
  compiled_vector_binary_state.vgprs[34][2] = 0x34343434u;
  compiled_vector_binary_state.vgprs[35][2] = 0x35353535u;
  compiled_vector_binary_state.vgprs[36][2] = 0x36363636u;
  compiled_vector_binary_state.vgprs[37][2] = 0x37373737u;
  compiled_vector_binary_state.vgprs[38][2] = 0x38383838u;
  compiled_vector_binary_state.vgprs[39][2] = 0x39393939u;
  compiled_vector_binary_state.vgprs[40][2] = 0x40404040u;
  if (!Expect(interpreter.ExecuteProgram(compiled_vector_binary_program,
                                         &compiled_vector_binary_state,
                                         &error_message),
              "expected compiled vector binary execution success") ||
      !Expect(ExpectVectorBinaryBatchState(compiled_vector_binary_state),
              "expected compiled vector binary state")) {
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
