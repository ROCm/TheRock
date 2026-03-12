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

constexpr std::uint16_t kImplicitVccPairSgprIndex = 248;
constexpr std::uint32_t kQuietNaNF32Bits = 0x7fc00000u;
constexpr std::uint64_t kQuietNaNF64Bits = 0x7ff8000000000000ULL;

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

std::uint64_t DoubleBits(double value) {
  std::uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void SplitU64(std::uint64_t value, std::uint32_t* low, std::uint32_t* high) {
  *low = static_cast<std::uint32_t>(value);
  *high = static_cast<std::uint32_t>(value >> 32);
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

bool ExpectOperandDescriptor(
    const mirage::sim::isa::InstructionOperand& operand,
    mirage::sim::isa::OperandRole expected_role,
    mirage::sim::isa::OperandSlotKind expected_slot_kind,
    mirage::sim::isa::OperandValueClass expected_value_class,
    mirage::sim::isa::OperandAccess expected_access,
    mirage::sim::isa::FragmentKind expected_fragment_kind,
    std::uint8_t expected_element_bit_width,
    std::uint8_t expected_component_count,
    bool expected_is_implicit) {
  using namespace mirage::sim::isa;
  return operand.descriptor.role == expected_role &&
         operand.descriptor.slot_kind == expected_slot_kind &&
         operand.descriptor.value_class == expected_value_class &&
         operand.descriptor.access == expected_access &&
         operand.descriptor.fragment_shape.kind == expected_fragment_kind &&
         operand.descriptor.fragment_shape.element_bit_width ==
             expected_element_bit_width &&
         operand.descriptor.component_count == expected_component_count &&
         operand.descriptor.is_implicit == expected_is_implicit;
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

bool ExpectWideConversionSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t neg_two_point_five_low = 0;
  std::uint32_t neg_two_point_five_high = 0;
  SplitU64(DoubleBits(-2.5), &neg_two_point_five_low, &neg_two_point_five_high);
  std::uint32_t neg_three_low = 0;
  std::uint32_t neg_three_high = 0;
  SplitU64(DoubleBits(-3.0), &neg_three_low, &neg_three_high);
  std::uint32_t seven_low = 0;
  std::uint32_t seven_high = 0;
  SplitU64(DoubleBits(7.0), &seven_low, &seven_high);

  return state.vgprs[20][0] == neg_two_point_five_low &&
         state.vgprs[20][1] == neg_two_point_five_low &&
         state.vgprs[20][2] == 0x20202020u &&
         state.vgprs[20][3] == neg_two_point_five_low &&
         state.vgprs[21][0] == neg_two_point_five_high &&
         state.vgprs[21][1] == neg_two_point_five_high &&
         state.vgprs[21][2] == 0x21212121u &&
         state.vgprs[21][3] == neg_two_point_five_high &&
         state.vgprs[22][0] == neg_three_low &&
         state.vgprs[22][1] == neg_three_low &&
         state.vgprs[22][2] == 0x22222222u &&
         state.vgprs[22][3] == neg_three_low &&
         state.vgprs[23][0] == neg_three_high &&
         state.vgprs[23][1] == neg_three_high &&
         state.vgprs[23][2] == 0x23232323u &&
         state.vgprs[23][3] == neg_three_high &&
         state.vgprs[24][0] == seven_low && state.vgprs[24][1] == seven_low &&
         state.vgprs[24][2] == 0x24242424u &&
         state.vgprs[24][3] == seven_low &&
         state.vgprs[25][0] == seven_high &&
         state.vgprs[25][1] == seven_high &&
         state.vgprs[25][2] == 0x25252525u &&
         state.vgprs[25][3] == seven_high &&
         state.vgprs[4][0] == FloatBits(-2.5f) &&
         state.vgprs[4][1] == FloatBits(-2.5f) &&
         state.vgprs[4][2] == 0x44444444u &&
         state.vgprs[4][3] == FloatBits(-2.5f) &&
         state.vgprs[5][0] == 0xfffffffeu &&
         state.vgprs[5][1] == 0xfffffffeu &&
         state.vgprs[5][2] == 0x55555555u &&
         state.vgprs[5][3] == 0xfffffffeu &&
         state.vgprs[6][0] == 7u && state.vgprs[6][1] == 7u &&
         state.vgprs[6][2] == 0x66666666u && state.vgprs[6][3] == 7u &&
         state.halted && !state.waiting_on_barrier && state.pc == 9u;
}

bool ExpectRoundingSeedState(const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t source_f64_low = 0;
  std::uint32_t source_f64_high = 0;
  SplitU64(DoubleBits(-2.75), &source_f64_low, &source_f64_high);
  std::uint32_t neg_two_f64_low = 0;
  std::uint32_t neg_two_f64_high = 0;
  SplitU64(DoubleBits(-2.0), &neg_two_f64_low, &neg_two_f64_high);
  std::uint32_t neg_three_f64_low = 0;
  std::uint32_t neg_three_f64_high = 0;
  SplitU64(DoubleBits(-3.0), &neg_three_f64_low, &neg_three_f64_high);

  return state.vgprs[4][0] == FloatBits(-2.0f) &&
         state.vgprs[4][1] == FloatBits(-2.0f) &&
         state.vgprs[4][2] == 0x44444444u &&
         state.vgprs[4][3] == FloatBits(-2.0f) &&
         state.vgprs[5][0] == FloatBits(-2.0f) &&
         state.vgprs[5][1] == FloatBits(-2.0f) &&
         state.vgprs[5][2] == 0x55555555u &&
         state.vgprs[5][3] == FloatBits(-2.0f) &&
         state.vgprs[6][0] == FloatBits(-3.0f) &&
         state.vgprs[6][1] == FloatBits(-3.0f) &&
         state.vgprs[6][2] == 0x66666666u &&
         state.vgprs[6][3] == FloatBits(-3.0f) &&
         state.vgprs[7][0] == FloatBits(-3.0f) &&
         state.vgprs[7][1] == FloatBits(-3.0f) &&
         state.vgprs[7][2] == 0x77777777u &&
         state.vgprs[7][3] == FloatBits(-3.0f) &&
         state.vgprs[20][0] == source_f64_low &&
         state.vgprs[20][1] == source_f64_low &&
         state.vgprs[20][2] == 0x20202020u &&
         state.vgprs[20][3] == source_f64_low &&
         state.vgprs[21][0] == source_f64_high &&
         state.vgprs[21][1] == source_f64_high &&
         state.vgprs[21][2] == 0x21212121u &&
         state.vgprs[21][3] == source_f64_high &&
         state.vgprs[30][0] == neg_two_f64_low &&
         state.vgprs[30][1] == neg_two_f64_low &&
         state.vgprs[30][2] == 0x30303030u &&
         state.vgprs[30][3] == neg_two_f64_low &&
         state.vgprs[31][0] == neg_two_f64_high &&
         state.vgprs[31][1] == neg_two_f64_high &&
         state.vgprs[31][2] == 0x31313131u &&
         state.vgprs[31][3] == neg_two_f64_high &&
         state.vgprs[32][0] == neg_two_f64_low &&
         state.vgprs[32][1] == neg_two_f64_low &&
         state.vgprs[32][2] == 0x32323232u &&
         state.vgprs[32][3] == neg_two_f64_low &&
         state.vgprs[33][0] == neg_two_f64_high &&
         state.vgprs[33][1] == neg_two_f64_high &&
         state.vgprs[33][2] == 0x33333333u &&
         state.vgprs[33][3] == neg_two_f64_high &&
         state.vgprs[34][0] == neg_three_f64_low &&
         state.vgprs[34][1] == neg_three_f64_low &&
         state.vgprs[34][2] == 0x34343434u &&
         state.vgprs[34][3] == neg_three_f64_low &&
         state.vgprs[35][0] == neg_three_f64_high &&
         state.vgprs[35][1] == neg_three_f64_high &&
         state.vgprs[35][2] == 0x35353535u &&
         state.vgprs[35][3] == neg_three_f64_high &&
         state.vgprs[36][0] == neg_three_f64_low &&
         state.vgprs[36][1] == neg_three_f64_low &&
         state.vgprs[36][2] == 0x36363636u &&
         state.vgprs[36][3] == neg_three_f64_low &&
         state.vgprs[37][0] == neg_three_f64_high &&
         state.vgprs[37][1] == neg_three_f64_high &&
         state.vgprs[37][2] == 0x37373737u &&
         state.vgprs[37][3] == neg_three_f64_high && state.halted &&
         !state.waiting_on_barrier && state.pc == 10u;
}

bool ExpectFractFrexpSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t source_f64_low = 0;
  std::uint32_t source_f64_high = 0;
  SplitU64(DoubleBits(6.5), &source_f64_low, &source_f64_high);
  std::uint32_t mantissa_f64_low = 0;
  std::uint32_t mantissa_f64_high = 0;
  SplitU64(DoubleBits(0.8125), &mantissa_f64_low, &mantissa_f64_high);
  std::uint32_t fract_f64_low = 0;
  std::uint32_t fract_f64_high = 0;
  SplitU64(DoubleBits(0.5), &fract_f64_low, &fract_f64_high);

  return state.vgprs[4][0] == FloatBits(0.25f) &&
         state.vgprs[4][1] == FloatBits(0.25f) &&
         state.vgprs[4][2] == 0x44444444u &&
         state.vgprs[4][3] == FloatBits(0.25f) &&
         state.vgprs[5][0] == 3u && state.vgprs[5][1] == 3u &&
         state.vgprs[5][2] == 0x55555555u && state.vgprs[5][3] == 3u &&
         state.vgprs[6][0] == FloatBits(-0.71875f) &&
         state.vgprs[6][1] == FloatBits(-0.71875f) &&
         state.vgprs[6][2] == 0x66666666u &&
         state.vgprs[6][3] == FloatBits(-0.71875f) &&
         state.vgprs[8][0] == 3u && state.vgprs[8][1] == 3u &&
         state.vgprs[8][2] == 0x88888888u && state.vgprs[8][3] == 3u &&
         state.vgprs[20][0] == source_f64_low &&
         state.vgprs[20][1] == source_f64_low &&
         state.vgprs[20][2] == 0x20202020u &&
         state.vgprs[20][3] == source_f64_low &&
         state.vgprs[21][0] == source_f64_high &&
         state.vgprs[21][1] == source_f64_high &&
         state.vgprs[21][2] == 0x21212121u &&
         state.vgprs[21][3] == source_f64_high &&
         state.vgprs[30][0] == mantissa_f64_low &&
         state.vgprs[30][1] == mantissa_f64_low &&
         state.vgprs[30][2] == 0x30303030u &&
         state.vgprs[30][3] == mantissa_f64_low &&
         state.vgprs[31][0] == mantissa_f64_high &&
         state.vgprs[31][1] == mantissa_f64_high &&
         state.vgprs[31][2] == 0x31313131u &&
         state.vgprs[31][3] == mantissa_f64_high &&
         state.vgprs[32][0] == fract_f64_low &&
         state.vgprs[32][1] == fract_f64_low &&
         state.vgprs[32][2] == 0x32323232u &&
         state.vgprs[32][3] == fract_f64_low &&
         state.vgprs[33][0] == fract_f64_high &&
         state.vgprs[33][1] == fract_f64_high &&
         state.vgprs[33][2] == 0x33333333u &&
         state.vgprs[33][3] == fract_f64_high && state.halted &&
         !state.waiting_on_barrier && state.pc == 9u;
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

bool ExpectUnsignedVectorCompareState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[70] == 7u && state.sgprs[71] == 5u &&
         state.sgprs[90] == 222u && state.sgprs[91] == 444u &&
         state.vcc_mask == 10u && state.halted &&
         !state.waiting_on_barrier && state.pc == 17u;
}

bool ExpectSignedVectorCompareState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[72] == 0xfffffffdu && state.sgprs[73] == 2u &&
         state.sgprs[92] == 666u && state.sgprs[93] == 888u &&
         state.vcc_mask == 9u && state.halted &&
         !state.waiting_on_barrier && state.pc == 17u;
}

bool ExpectMaskedVectorComparePreservesInactiveVccState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vcc_mask == 5u && state.exec_mask == 1u && state.halted &&
         !state.waiting_on_barrier && state.pc == 1u;
}

bool ExpectUnsignedVectorCmpxState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[74] == 7u && state.sgprs[75] == 5u &&
         state.sgprs[94] == 111u && state.sgprs[95] == 444u &&
         state.sgprs[96] == 666u && state.exec_mask == 0u &&
         state.vcc_mask == 0u && state.halted &&
         !state.waiting_on_barrier && state.pc == 20u;
}

bool ExpectSignedVectorCmpxState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[76] == 0xfffffffdu && state.sgprs[77] == 2u &&
         state.sgprs[97] == 777u && state.sgprs[98] == 1001u &&
         state.sgprs[99] == 321u && state.exec_mask == 0u &&
         state.vcc_mask == 0u && state.halted &&
         !state.waiting_on_barrier && state.pc == 20u;
}

bool ExpectFloatVectorCompareState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[100] == 222u && state.sgprs[101] == 333u &&
         state.sgprs[110] == FloatBits(1.5f) &&
         state.sgprs[111] == FloatBits(2.5f) &&
         state.sgprs[112] == kQuietNaNF32Bits && state.vcc_mask == 9u &&
         state.halted && !state.waiting_on_barrier && state.pc == 19u;
}

bool ExpectFloatVectorCmpxState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[102] == 111u && state.sgprs[103] == 444u &&
         state.sgprs[104] == 666u &&
         state.sgprs[113] == FloatBits(1.5f) &&
         state.sgprs[114] == FloatBits(2.5f) &&
         state.sgprs[115] == kQuietNaNF32Bits && state.exec_mask == 0u &&
         state.vcc_mask == 0u && state.halted &&
         !state.waiting_on_barrier && state.pc == 23u;
}

bool ExpectVccDrivenCndmaskState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[116] == FloatBits(2.0f) && state.sgprs[120] == 222u &&
         state.vgprs[42][0] == 10u && state.vgprs[42][1] == 10u &&
         state.vgprs[42][2] == 10u && state.vgprs[42][3] == 20u &&
         state.vcc_mask == 8u && state.exec_mask == 0xfu && state.halted &&
         !state.waiting_on_barrier && state.pc == 7u;
}

bool ExpectFloatVectorCmpxExecBranchState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[117] == FloatBits(2.0f) && state.sgprs[121] == 222u &&
         state.vcc_mask == 8u && state.exec_mask == 8u && state.halted &&
         !state.waiting_on_barrier && state.pc == 6u;
}

bool ExpectF64ClassCndmaskState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[60][0] == 20u && state.vgprs[60][1] == 10u &&
         state.vgprs[60][2] == 20u && state.vgprs[60][3] == 10u &&
         state.vcc_mask == 5u && state.exec_mask == 0xfu && state.halted &&
         !state.waiting_on_barrier && state.pc == 4u;
}

bool ExpectF64CmpxClassBranchState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[119] == 222u && state.vcc_mask == 6u &&
         state.exec_mask == 6u && state.halted &&
         !state.waiting_on_barrier && state.pc == 7u;
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
              "expected decoded S_ADD_U32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_ADD_U32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_ADD_U32 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_ADD_U32 source1 descriptor")) {
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
              "expected decoded V_MOV_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_MOV_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_MOV_B32 source descriptor")) {
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

  const std::array<std::uint32_t, 1> v_cvt_f64_i32_words{
      MakeVop1(4u, 10u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f64_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F64_I32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F64_I32",
                                     OperandKind::kVgpr, 10u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_F64_I32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_CVT_F64_I32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CVT_F64_I32 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_f64_words{
      MakeVop1(15u, 11u, 120u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_F64 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_F64",
                                     OperandKind::kVgpr, 11u,
                                     OperandKind::kSgpr, 120u),
              "expected decoded V_CVT_F32_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CVT_F32_F64 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CVT_F32_F64 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_trunc_f32_words{
      MakeVop1(33u, 12u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_trunc_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_TRUNC_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_TRUNC_F32",
                                     OperandKind::kVgpr, 12u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_TRUNC_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_ceil_f64_words{
      MakeVop1(24u, 13u, 120u)};
  if (!Expect(decoder.DecodeInstruction(v_ceil_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CEIL_F64 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CEIL_F64",
                                     OperandKind::kVgpr, 13u,
                                     OperandKind::kSgpr, 120u),
              "expected decoded V_CEIL_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_CEIL_F64 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CEIL_F64 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_fract_f32_words{
      MakeVop1(32u, 14u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_fract_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FRACT_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_FRACT_F32",
                                     OperandKind::kVgpr, 14u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_FRACT_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_frexp_exp_i32_f64_words{
      MakeVop1(60u, 15u, 120u)};
  if (!Expect(decoder.DecodeInstruction(v_frexp_exp_i32_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FREXP_EXP_I32_F64 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_FREXP_EXP_I32_F64",
                                     OperandKind::kVgpr, 15u,
                                     OperandKind::kSgpr, 120u),
              "expected decoded V_FREXP_EXP_I32_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_FREXP_EXP_I32_F64 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_FREXP_EXP_I32_F64 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_fract_f64_words{
      MakeVop1(62u, 16u, 276u)};
  if (!Expect(decoder.DecodeInstruction(v_fract_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FRACT_F64 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_FRACT_F64",
                                     OperandKind::kVgpr, 16u,
                                     OperandKind::kVgpr, 20u),
              "expected decoded V_FRACT_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_FRACT_F64 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_FRACT_F64 source descriptor")) {
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

  const std::array<std::uint32_t, 1> v_cmp_eq_u32_words{
      MakeVopc(74u, 1u, 4u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_U32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_CMP_EQ_U32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMP_EQ_U32 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMP_EQ_U32 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmp_ne_i32_literal_words{
      MakeVopc(69u, 255u, 5u), 0xfffffffdu};
  if (!Expect(decoder.DecodeInstruction(v_cmp_ne_i32_literal_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_NE_I32 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMP_NE_I32 decode to consume 2 dwords") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_NE_I32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, 0xfffffffdu,
                                      OperandKind::kVgpr, 5u),
              "expected decoded V_CMP_NE_I32 literal operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMP_NE_I32 literal source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmpx_eq_u32_words{
      MakeVopc(202u, 1u, 4u)};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_eq_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMPX_EQ_U32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_EQ_U32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_CMPX_EQ_U32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMPX_EQ_U32")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMPX_EQ_U32 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMPX_EQ_U32 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmpx_ne_i32_literal_words{
      MakeVopc(197u, 255u, 5u), 0xfffffffdu};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_ne_i32_literal_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMPX_NE_I32 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMPX_NE_I32 decode to consume 2 dwords") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_NE_I32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, 0xfffffffdu,
                                      OperandKind::kVgpr, 5u),
              "expected decoded V_CMPX_NE_I32 literal operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMPX_NE_I32 literal source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmp_eq_f32_words{
      MakeVopc(18u, 110u, 70u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_F32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 110u,
                                      OperandKind::kVgpr, 70u),
              "expected decoded V_CMP_EQ_F32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMP_EQ_F32")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMP_EQ_F32 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMP_EQ_F32 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmpx_class_f32_literal_words{
      MakeVopc(254u, 255u, 71u), kQuietNaNF32Bits};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_class_f32_literal_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CMPX_CLASS_F32 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMPX_CLASS_F32 decode to consume 2 dwords")
      ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_CLASS_F32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, kQuietNaNF32Bits,
                                      OperandKind::kVgpr, 71u),
              "expected decoded V_CMPX_CLASS_F32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMPX_CLASS_F32 literal source descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMPX_CLASS_F32 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmp_o_f32_words{MakeVopc(23u, 116u, 88u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_o_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_O_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_O_F32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 116u,
                                      OperandKind::kVgpr, 88u),
              "expected decoded V_CMP_O_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmpx_u_f32_literal_words{
      MakeVopc(152u, 255u, 89u), FloatBits(2.0f)};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_u_f32_literal_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMPX_U_F32 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMPX_U_F32 decode to consume 2 dwords") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_U_F32",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, FloatBits(2.0f),
                                      OperandKind::kVgpr, 89u),
              "expected decoded V_CMPX_U_F32 literal operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cndmask_b32_words{
      MakeVop2(1u, 10u, 257u, 4u)};
  if (!Expect(decoder.DecodeInstruction(v_cndmask_b32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CNDMASK_B32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CNDMASK_B32",
                                      OperandKind::kVgpr, 10u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_CNDMASK_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CNDMASK_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CNDMASK_B32 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CNDMASK_B32 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmp_eq_f64_words{
      MakeVopc(34u, 120u, 94u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_F64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_F64",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 120u,
                                      OperandKind::kVgpr, 94u),
              "expected decoded V_CMP_EQ_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMP_EQ_F64")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CMP_EQ_F64 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_CMP_EQ_F64 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmpx_class_f64_words{
      MakeVopc(255u, 122u, 95u)};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_class_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMPX_CLASS_F64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_CLASS_F64",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 122u,
                                      OperandKind::kVgpr, 95u),
              "expected decoded V_CMPX_CLASS_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CMPX_CLASS_F64 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMPX_CLASS_F64 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmp_eq_i64_words{
      MakeVopc(82u, 118u, 100u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_i64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_I64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_I64",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 118u,
                                      OperandKind::kVgpr, 100u),
              "expected decoded V_CMP_EQ_I64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMP_EQ_I64")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CMP_EQ_I64 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_CMP_EQ_I64 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmpx_eq_u64_words{
      MakeVopc(218u, 120u, 102u)};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_eq_u64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMPX_EQ_U64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_EQ_U64",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 120u,
                                      OperandKind::kVgpr, 102u),
              "expected decoded V_CMPX_EQ_U64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_CMPX_EQ_U64 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_CMPX_EQ_U64 source1 descriptor")) {
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

  const std::array<std::uint32_t, 13> wide_conversion_words{
      MakeVop1(1u, 1u, 255u),
      FloatBits(-2.5f),
      MakeVop1(16u, 20u, 257u),
      MakeVop1(1u, 2u, 255u),
      0xfffffffdu,
      MakeVop1(4u, 22u, 258u),
      MakeVop1(1u, 3u, 255u),
      7u,
      MakeVop1(22u, 24u, 259u),
      MakeVop1(15u, 4u, 276u),
      MakeVop1(3u, 5u, 276u),
      MakeVop1(21u, 6u, 280u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> wide_conversion_program;
  if (!Expect(decoder.DecodeProgram(wide_conversion_words,
                                    &wide_conversion_program, &error_message),
              "expected wide conversion program decode success") ||
      !Expect(wide_conversion_program.size() == 10u,
              "expected ten decoded wide conversion instructions") ||
      !Expect(wide_conversion_program[1].opcode == "V_CVT_F64_F32",
              "expected decoded V_CVT_F64_F32") ||
      !Expect(wide_conversion_program[3].opcode == "V_CVT_F64_I32",
              "expected decoded V_CVT_F64_I32") ||
      !Expect(wide_conversion_program[5].opcode == "V_CVT_F64_U32",
              "expected decoded V_CVT_F64_U32") ||
      !Expect(wide_conversion_program[6].opcode == "V_CVT_F32_F64",
              "expected decoded V_CVT_F32_F64") ||
      !Expect(wide_conversion_program[7].opcode == "V_CVT_I32_F64",
              "expected decoded V_CVT_I32_F64") ||
      !Expect(wide_conversion_program[8].opcode == "V_CVT_U32_F64",
              "expected decoded V_CVT_U32_F64")) {
    return 1;
  }

  auto initialize_wide_conversion_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[2][2] = 0x12121212u;
    state->vgprs[3][2] = 0x13131313u;
    state->vgprs[20][2] = 0x20202020u;
    state->vgprs[21][2] = 0x21212121u;
    state->vgprs[22][2] = 0x22222222u;
    state->vgprs[23][2] = 0x23232323u;
    state->vgprs[24][2] = 0x24242424u;
    state->vgprs[25][2] = 0x25252525u;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[6][2] = 0x66666666u;
  };

  WaveExecutionState decoded_wide_conversion_state;
  initialize_wide_conversion_state(&decoded_wide_conversion_state);
  if (!Expect(interpreter.ExecuteProgram(wide_conversion_program,
                                         &decoded_wide_conversion_state,
                                         &error_message),
              "expected decoded wide conversion execution success") ||
      !Expect(ExpectWideConversionSeedState(decoded_wide_conversion_state),
              "expected decoded wide conversion state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_wide_conversion_program;
  if (!Expect(interpreter.CompileProgram(wide_conversion_program,
                                         &compiled_wide_conversion_program,
                                         &error_message),
              "expected compiled wide conversion program success") ||
      !Expect(compiled_wide_conversion_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF64F32,
              "expected compiled V_CVT_F64_F32 opcode") ||
      !Expect(compiled_wide_conversion_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF64I32,
              "expected compiled V_CVT_F64_I32 opcode") ||
      !Expect(compiled_wide_conversion_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF64U32,
              "expected compiled V_CVT_F64_U32 opcode") ||
      !Expect(compiled_wide_conversion_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32F64,
              "expected compiled V_CVT_F32_F64 opcode") ||
      !Expect(compiled_wide_conversion_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCvtI32F64,
              "expected compiled V_CVT_I32_F64 opcode") ||
      !Expect(compiled_wide_conversion_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVCvtU32F64,
              "expected compiled V_CVT_U32_F64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_wide_conversion_state;
  initialize_wide_conversion_state(&compiled_wide_conversion_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_wide_conversion_program,
                                         &compiled_wide_conversion_state,
                                         &error_message),
              "expected compiled wide conversion execution success") ||
      !Expect(ExpectWideConversionSeedState(compiled_wide_conversion_state),
              "expected compiled wide conversion state")) {
    return 1;
  }

  const std::array<std::uint32_t, 12> rounding_words{
      MakeVop1(1u, 1u, 255u),
      FloatBits(-2.75f),
      MakeVop1(33u, 4u, 257u),
      MakeVop1(34u, 5u, 257u),
      MakeVop1(35u, 6u, 257u),
      MakeVop1(36u, 7u, 257u),
      MakeVop1(16u, 20u, 257u),
      MakeVop1(23u, 30u, 276u),
      MakeVop1(24u, 32u, 276u),
      MakeVop1(25u, 34u, 276u),
      MakeVop1(26u, 36u, 276u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> rounding_program;
  if (!Expect(decoder.DecodeProgram(rounding_words, &rounding_program,
                                    &error_message),
              "expected rounding program decode success") ||
      !Expect(rounding_program.size() == 11u,
              "expected eleven decoded rounding instructions") ||
      !Expect(rounding_program[1].opcode == "V_TRUNC_F32",
              "expected decoded V_TRUNC_F32") ||
      !Expect(rounding_program[2].opcode == "V_CEIL_F32",
              "expected decoded V_CEIL_F32") ||
      !Expect(rounding_program[3].opcode == "V_RNDNE_F32",
              "expected decoded V_RNDNE_F32") ||
      !Expect(rounding_program[4].opcode == "V_FLOOR_F32",
              "expected decoded V_FLOOR_F32") ||
      !Expect(rounding_program[6].opcode == "V_TRUNC_F64",
              "expected decoded V_TRUNC_F64") ||
      !Expect(rounding_program[7].opcode == "V_CEIL_F64",
              "expected decoded V_CEIL_F64") ||
      !Expect(rounding_program[8].opcode == "V_RNDNE_F64",
              "expected decoded V_RNDNE_F64") ||
      !Expect(rounding_program[9].opcode == "V_FLOOR_F64",
              "expected decoded V_FLOOR_F64")) {
    return 1;
  }

  auto initialize_rounding_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[7][2] = 0x77777777u;
    state->vgprs[20][2] = 0x20202020u;
    state->vgprs[21][2] = 0x21212121u;
    state->vgprs[30][2] = 0x30303030u;
    state->vgprs[31][2] = 0x31313131u;
    state->vgprs[32][2] = 0x32323232u;
    state->vgprs[33][2] = 0x33333333u;
    state->vgprs[34][2] = 0x34343434u;
    state->vgprs[35][2] = 0x35353535u;
    state->vgprs[36][2] = 0x36363636u;
    state->vgprs[37][2] = 0x37373737u;
  };

  WaveExecutionState decoded_rounding_state;
  initialize_rounding_state(&decoded_rounding_state);
  if (!Expect(interpreter.ExecuteProgram(rounding_program, &decoded_rounding_state,
                                         &error_message),
              "expected decoded rounding execution success") ||
      !Expect(ExpectRoundingSeedState(decoded_rounding_state),
              "expected decoded rounding state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_rounding_program;
  if (!Expect(interpreter.CompileProgram(rounding_program,
                                         &compiled_rounding_program,
                                         &error_message),
              "expected compiled rounding program success") ||
      !Expect(compiled_rounding_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVTruncF32,
              "expected compiled V_TRUNC_F32 opcode") ||
      !Expect(compiled_rounding_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCeilF32,
              "expected compiled V_CEIL_F32 opcode") ||
      !Expect(compiled_rounding_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVRndneF32,
              "expected compiled V_RNDNE_F32 opcode") ||
      !Expect(compiled_rounding_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVFloorF32,
              "expected compiled V_FLOOR_F32 opcode") ||
      !Expect(compiled_rounding_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVTruncF64,
              "expected compiled V_TRUNC_F64 opcode") ||
      !Expect(compiled_rounding_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCeilF64,
              "expected compiled V_CEIL_F64 opcode") ||
      !Expect(compiled_rounding_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVRndneF64,
              "expected compiled V_RNDNE_F64 opcode") ||
      !Expect(compiled_rounding_program[9].opcode ==
                  Gfx1201CompiledOpcode::kVFloorF64,
              "expected compiled V_FLOOR_F64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_rounding_state;
  initialize_rounding_state(&compiled_rounding_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_rounding_program,
                                         &compiled_rounding_state,
                                         &error_message),
              "expected compiled rounding execution success") ||
      !Expect(ExpectRoundingSeedState(compiled_rounding_state),
              "expected compiled rounding state")) {
    return 1;
  }

  const std::array<std::uint32_t, 12> fract_frexp_words{
      MakeVop1(1u, 1u, 255u),
      FloatBits(-5.75f),
      MakeVop1(32u, 4u, 257u),
      MakeVop1(63u, 5u, 257u),
      MakeVop1(64u, 6u, 257u),
      MakeVop1(1u, 2u, 255u),
      FloatBits(6.5f),
      MakeVop1(16u, 20u, 258u),
      MakeVop1(60u, 8u, 276u),
      MakeVop1(61u, 30u, 276u),
      MakeVop1(62u, 32u, 276u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> fract_frexp_program;
  if (!Expect(decoder.DecodeProgram(fract_frexp_words, &fract_frexp_program,
                                    &error_message),
              "expected fract/frexp program decode success") ||
      !Expect(fract_frexp_program.size() == 10u,
              "expected ten decoded fract/frexp instructions") ||
      !Expect(fract_frexp_program[1].opcode == "V_FRACT_F32",
              "expected decoded V_FRACT_F32") ||
      !Expect(fract_frexp_program[2].opcode == "V_FREXP_EXP_I32_F32",
              "expected decoded V_FREXP_EXP_I32_F32") ||
      !Expect(fract_frexp_program[3].opcode == "V_FREXP_MANT_F32",
              "expected decoded V_FREXP_MANT_F32") ||
      !Expect(fract_frexp_program[6].opcode == "V_FREXP_EXP_I32_F64",
              "expected decoded V_FREXP_EXP_I32_F64") ||
      !Expect(fract_frexp_program[7].opcode == "V_FREXP_MANT_F64",
              "expected decoded V_FREXP_MANT_F64") ||
      !Expect(fract_frexp_program[8].opcode == "V_FRACT_F64",
              "expected decoded V_FRACT_F64")) {
    return 1;
  }

  auto initialize_fract_frexp_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[8][2] = 0x88888888u;
    state->vgprs[20][2] = 0x20202020u;
    state->vgprs[21][2] = 0x21212121u;
    state->vgprs[30][2] = 0x30303030u;
    state->vgprs[31][2] = 0x31313131u;
    state->vgprs[32][2] = 0x32323232u;
    state->vgprs[33][2] = 0x33333333u;
  };

  WaveExecutionState decoded_fract_frexp_state;
  initialize_fract_frexp_state(&decoded_fract_frexp_state);
  if (!Expect(interpreter.ExecuteProgram(fract_frexp_program,
                                         &decoded_fract_frexp_state,
                                         &error_message),
              "expected decoded fract/frexp execution success") ||
      !Expect(ExpectFractFrexpSeedState(decoded_fract_frexp_state),
              "expected decoded fract/frexp state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_fract_frexp_program;
  if (!Expect(interpreter.CompileProgram(fract_frexp_program,
                                         &compiled_fract_frexp_program,
                                         &error_message),
              "expected compiled fract/frexp program success") ||
      !Expect(compiled_fract_frexp_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVFractF32,
              "expected compiled V_FRACT_F32 opcode") ||
      !Expect(compiled_fract_frexp_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpExpI32F32,
              "expected compiled V_FREXP_EXP_I32_F32 opcode") ||
      !Expect(compiled_fract_frexp_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpMantF32,
              "expected compiled V_FREXP_MANT_F32 opcode") ||
      !Expect(compiled_fract_frexp_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpExpI32F64,
              "expected compiled V_FREXP_EXP_I32_F64 opcode") ||
      !Expect(compiled_fract_frexp_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpMantF64,
              "expected compiled V_FREXP_MANT_F64 opcode") ||
      !Expect(compiled_fract_frexp_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVFractF64,
              "expected compiled V_FRACT_F64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_fract_frexp_state;
  initialize_fract_frexp_state(&compiled_fract_frexp_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_fract_frexp_program,
                                         &compiled_fract_frexp_state,
                                         &error_message),
              "expected compiled fract/frexp execution success") ||
      !Expect(ExpectFractFrexpSeedState(compiled_fract_frexp_state),
              "expected compiled fract/frexp state")) {
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

  const std::array<std::uint32_t, 18> unsigned_vector_compare_words{
      MakeSopk(0u, 70u, 7u),
      MakeSopk(0u, 71u, 5u),
      MakeVopc(74u, 70u, 50u),
      MakeSopp(36u, 2u),
      MakeSopk(0u, 90u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 90u, 222u),
      MakeVopc(76u, 71u, 54u),
      MakeSopp(35u, 2u),
      MakeSopk(0u, 91u, 333u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 91u, 444u),
      MakeVopc(77u, 137u, 51u),
      MakeVopc(73u, 71u, 52u),
      MakeVopc(75u, 71u, 52u),
      MakeVopc(76u, 71u, 52u),
      MakeVopc(78u, 71u, 52u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> unsigned_vector_compare_program;
  if (!Expect(decoder.DecodeProgram(unsigned_vector_compare_words,
                                    &unsigned_vector_compare_program,
                                    &error_message),
              "expected unsigned vector compare program decode success") ||
      !Expect(unsigned_vector_compare_program.size() == 18u,
              "expected eighteen decoded unsigned vector compare instructions") ||
      !Expect(unsigned_vector_compare_program[2].opcode == "V_CMP_EQ_U32",
              "expected decoded V_CMP_EQ_U32") ||
      !Expect(unsigned_vector_compare_program[7].opcode == "V_CMP_GT_U32",
              "expected decoded zero-mask V_CMP_GT_U32") ||
      !Expect(unsigned_vector_compare_program[12].opcode == "V_CMP_NE_U32",
              "expected decoded V_CMP_NE_U32") ||
      !Expect(unsigned_vector_compare_program[13].opcode == "V_CMP_LT_U32",
              "expected decoded V_CMP_LT_U32") ||
      !Expect(unsigned_vector_compare_program[14].opcode == "V_CMP_LE_U32",
              "expected decoded V_CMP_LE_U32") ||
      !Expect(unsigned_vector_compare_program[16].opcode == "V_CMP_GE_U32",
              "expected decoded V_CMP_GE_U32")) {
    return 1;
  }

  WaveExecutionState decoded_unsigned_vector_compare_state;
  decoded_unsigned_vector_compare_state.exec_mask = 0xbu;
  decoded_unsigned_vector_compare_state.vgprs[50][0] = 7u;
  decoded_unsigned_vector_compare_state.vgprs[50][1] = 4u;
  decoded_unsigned_vector_compare_state.vgprs[50][3] = 7u;
  decoded_unsigned_vector_compare_state.vgprs[51][0] = 9u;
  decoded_unsigned_vector_compare_state.vgprs[51][1] = 8u;
  decoded_unsigned_vector_compare_state.vgprs[51][3] = 8u;
  decoded_unsigned_vector_compare_state.vgprs[52][0] = 6u;
  decoded_unsigned_vector_compare_state.vgprs[52][1] = 5u;
  decoded_unsigned_vector_compare_state.vgprs[52][3] = 4u;
  decoded_unsigned_vector_compare_state.vgprs[54][0] = 6u;
  decoded_unsigned_vector_compare_state.vgprs[54][1] = 7u;
  decoded_unsigned_vector_compare_state.vgprs[54][3] = 8u;
  if (!Expect(interpreter.ExecuteProgram(unsigned_vector_compare_program,
                                         &decoded_unsigned_vector_compare_state,
                                         &error_message),
              "expected decoded unsigned vector compare execution success") ||
      !Expect(ExpectUnsignedVectorCompareState(
                  decoded_unsigned_vector_compare_state),
              "expected decoded unsigned vector compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction>
      compiled_unsigned_vector_compare_program;
  if (!Expect(interpreter.CompileProgram(unsigned_vector_compare_program,
                                         &compiled_unsigned_vector_compare_program,
                                         &error_message),
              "expected compiled unsigned vector compare program success") ||
      !Expect(compiled_unsigned_vector_compare_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpEqU32,
              "expected compiled V_CMP_EQ_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGtU32,
              "expected compiled zero-mask V_CMP_GT_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVCmpNeU32,
              "expected compiled V_CMP_NE_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLtU32,
              "expected compiled V_CMP_LT_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLeU32,
              "expected compiled V_CMP_LE_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[15].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGtU32,
              "expected compiled V_CMP_GT_U32 opcode") ||
      !Expect(compiled_unsigned_vector_compare_program[16].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGeU32,
              "expected compiled V_CMP_GE_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_unsigned_vector_compare_state;
  compiled_unsigned_vector_compare_state.exec_mask = 0xbu;
  compiled_unsigned_vector_compare_state.vgprs[50][0] = 7u;
  compiled_unsigned_vector_compare_state.vgprs[50][1] = 4u;
  compiled_unsigned_vector_compare_state.vgprs[50][3] = 7u;
  compiled_unsigned_vector_compare_state.vgprs[51][0] = 9u;
  compiled_unsigned_vector_compare_state.vgprs[51][1] = 8u;
  compiled_unsigned_vector_compare_state.vgprs[51][3] = 8u;
  compiled_unsigned_vector_compare_state.vgprs[52][0] = 6u;
  compiled_unsigned_vector_compare_state.vgprs[52][1] = 5u;
  compiled_unsigned_vector_compare_state.vgprs[52][3] = 4u;
  compiled_unsigned_vector_compare_state.vgprs[54][0] = 6u;
  compiled_unsigned_vector_compare_state.vgprs[54][1] = 7u;
  compiled_unsigned_vector_compare_state.vgprs[54][3] = 8u;
  if (!Expect(interpreter.ExecuteProgram(compiled_unsigned_vector_compare_program,
                                         &compiled_unsigned_vector_compare_state,
                                         &error_message),
              "expected compiled unsigned vector compare execution success") ||
      !Expect(ExpectUnsignedVectorCompareState(
                  compiled_unsigned_vector_compare_state),
              "expected compiled unsigned vector compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 18> signed_vector_compare_words{
      MakeSopk(0u, 72u, 0xfffdu),
      MakeSopk(0u, 73u, 2u),
      MakeVopc(66u, 72u, 53u),
      MakeSopp(36u, 2u),
      MakeSopk(0u, 92u, 555u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 92u, 666u),
      MakeVopc(65u, 73u, 55u),
      MakeSopp(35u, 2u),
      MakeSopk(0u, 93u, 777u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 93u, 888u),
      MakeVopc(69u, 72u, 53u),
      MakeVopc(65u, 72u, 53u),
      MakeVopc(67u, 72u, 53u),
      MakeVopc(68u, 73u, 53u),
      MakeVopc(70u, 73u, 53u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> signed_vector_compare_program;
  if (!Expect(decoder.DecodeProgram(signed_vector_compare_words,
                                    &signed_vector_compare_program,
                                    &error_message),
              "expected signed vector compare program decode success") ||
      !Expect(signed_vector_compare_program.size() == 18u,
              "expected eighteen decoded signed vector compare instructions") ||
      !Expect(signed_vector_compare_program[2].opcode == "V_CMP_EQ_I32",
              "expected decoded V_CMP_EQ_I32") ||
      !Expect(signed_vector_compare_program[7].opcode == "V_CMP_LT_I32",
              "expected decoded zero-mask V_CMP_LT_I32") ||
      !Expect(signed_vector_compare_program[12].opcode == "V_CMP_NE_I32",
              "expected decoded V_CMP_NE_I32") ||
      !Expect(signed_vector_compare_program[14].opcode == "V_CMP_LE_I32",
              "expected decoded V_CMP_LE_I32") ||
      !Expect(signed_vector_compare_program[16].opcode == "V_CMP_GE_I32",
              "expected decoded V_CMP_GE_I32")) {
    return 1;
  }

  WaveExecutionState decoded_signed_vector_compare_state;
  decoded_signed_vector_compare_state.exec_mask = 0xbu;
  decoded_signed_vector_compare_state.vgprs[53][0] = 0xfffffffdu;
  decoded_signed_vector_compare_state.vgprs[53][1] = 4u;
  decoded_signed_vector_compare_state.vgprs[53][3] = 0xfffffffbu;
  decoded_signed_vector_compare_state.vgprs[55][0] = 0xffffffffu;
  decoded_signed_vector_compare_state.vgprs[55][1] = 0u;
  decoded_signed_vector_compare_state.vgprs[55][3] = 1u;
  if (!Expect(interpreter.ExecuteProgram(signed_vector_compare_program,
                                         &decoded_signed_vector_compare_state,
                                         &error_message),
              "expected decoded signed vector compare execution success") ||
      !Expect(ExpectSignedVectorCompareState(decoded_signed_vector_compare_state),
              "expected decoded signed vector compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_signed_vector_compare_program;
  if (!Expect(interpreter.CompileProgram(signed_vector_compare_program,
                                         &compiled_signed_vector_compare_program,
                                         &error_message),
              "expected compiled signed vector compare program success") ||
      !Expect(compiled_signed_vector_compare_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpEqI32,
              "expected compiled V_CMP_EQ_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLtI32,
              "expected compiled zero-mask V_CMP_LT_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVCmpNeI32,
              "expected compiled V_CMP_NE_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLtI32,
              "expected compiled V_CMP_LT_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLeI32,
              "expected compiled V_CMP_LE_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[15].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGtI32,
              "expected compiled V_CMP_GT_I32 opcode") ||
      !Expect(compiled_signed_vector_compare_program[16].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGeI32,
              "expected compiled V_CMP_GE_I32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_signed_vector_compare_state;
  compiled_signed_vector_compare_state.exec_mask = 0xbu;
  compiled_signed_vector_compare_state.vgprs[53][0] = 0xfffffffdu;
  compiled_signed_vector_compare_state.vgprs[53][1] = 4u;
  compiled_signed_vector_compare_state.vgprs[53][3] = 0xfffffffbu;
  compiled_signed_vector_compare_state.vgprs[55][0] = 0xffffffffu;
  compiled_signed_vector_compare_state.vgprs[55][1] = 0u;
  compiled_signed_vector_compare_state.vgprs[55][3] = 1u;
  if (!Expect(interpreter.ExecuteProgram(compiled_signed_vector_compare_program,
                                         &compiled_signed_vector_compare_state,
                                         &error_message),
              "expected compiled signed vector compare execution success") ||
      !Expect(ExpectSignedVectorCompareState(
                  compiled_signed_vector_compare_state),
              "expected compiled signed vector compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> masked_vector_compare_words{
      MakeVopc(74u, 1u, 4u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> masked_vector_compare_program;
  if (!Expect(decoder.DecodeProgram(masked_vector_compare_words,
                                    &masked_vector_compare_program,
                                    &error_message),
              "expected masked vector compare program decode success") ||
      !Expect(masked_vector_compare_program.size() == 2u,
              "expected two decoded masked vector compare instructions")) {
    return 1;
  }

  WaveExecutionState decoded_masked_vector_compare_state;
  decoded_masked_vector_compare_state.exec_mask = 1u;
  decoded_masked_vector_compare_state.vcc_mask = 4u;
  decoded_masked_vector_compare_state.sgprs[1] = 7u;
  decoded_masked_vector_compare_state.vgprs[4][0] = 7u;
  if (!Expect(interpreter.ExecuteProgram(masked_vector_compare_program,
                                         &decoded_masked_vector_compare_state,
                                         &error_message),
              "expected decoded masked vector compare execution success") ||
      !Expect(ExpectMaskedVectorComparePreservesInactiveVccState(
                  decoded_masked_vector_compare_state),
              "expected decoded masked vector compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_masked_vector_compare_program;
  if (!Expect(interpreter.CompileProgram(masked_vector_compare_program,
                                         &compiled_masked_vector_compare_program,
                                         &error_message),
              "expected compiled masked vector compare program success") ||
      !Expect(compiled_masked_vector_compare_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVCmpEqU32,
              "expected compiled masked V_CMP_EQ_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_masked_vector_compare_state;
  compiled_masked_vector_compare_state.exec_mask = 1u;
  compiled_masked_vector_compare_state.vcc_mask = 4u;
  compiled_masked_vector_compare_state.sgprs[1] = 7u;
  compiled_masked_vector_compare_state.vgprs[4][0] = 7u;
  if (!Expect(interpreter.ExecuteProgram(compiled_masked_vector_compare_program,
                                         &compiled_masked_vector_compare_state,
                                         &error_message),
              "expected compiled masked vector compare execution success") ||
      !Expect(ExpectMaskedVectorComparePreservesInactiveVccState(
                  compiled_masked_vector_compare_state),
              "expected compiled masked vector compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 21> unsigned_vector_cmpx_words{
      MakeSopk(0u, 74u, 7u),
      MakeSopk(0u, 75u, 5u),
      MakeVopc(202u, 74u, 60u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 94u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 94u, 222u),
      MakeVopc(205u, 137u, 61u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 95u, 333u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 95u, 444u),
      MakeVopc(201u, 75u, 62u),
      MakeVopc(203u, 75u, 62u),
      MakeVopc(204u, 75u, 63u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 96u, 555u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 96u, 666u),
      MakeVopc(206u, 75u, 63u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> unsigned_vector_cmpx_program;
  if (!Expect(decoder.DecodeProgram(unsigned_vector_cmpx_words,
                                    &unsigned_vector_cmpx_program,
                                    &error_message),
              "expected unsigned vector CMPX program decode success") ||
      !Expect(unsigned_vector_cmpx_program.size() == 21u,
              "expected twenty-one decoded unsigned vector CMPX instructions")
      ||
      !Expect(unsigned_vector_cmpx_program[2].opcode == "V_CMPX_EQ_U32",
              "expected decoded V_CMPX_EQ_U32") ||
      !Expect(unsigned_vector_cmpx_program[7].opcode == "V_CMPX_NE_U32",
              "expected decoded V_CMPX_NE_U32") ||
      !Expect(unsigned_vector_cmpx_program[12].opcode == "V_CMPX_LT_U32",
              "expected decoded V_CMPX_LT_U32") ||
      !Expect(unsigned_vector_cmpx_program[13].opcode == "V_CMPX_LE_U32",
              "expected decoded V_CMPX_LE_U32") ||
      !Expect(unsigned_vector_cmpx_program[14].opcode == "V_CMPX_GT_U32",
              "expected decoded V_CMPX_GT_U32") ||
      !Expect(unsigned_vector_cmpx_program[19].opcode == "V_CMPX_GE_U32",
              "expected decoded V_CMPX_GE_U32")) {
    return 1;
  }

  WaveExecutionState decoded_unsigned_vector_cmpx_state;
  decoded_unsigned_vector_cmpx_state.exec_mask = 0xbu;
  decoded_unsigned_vector_cmpx_state.vgprs[60][0] = 7u;
  decoded_unsigned_vector_cmpx_state.vgprs[60][1] = 4u;
  decoded_unsigned_vector_cmpx_state.vgprs[60][3] = 7u;
  decoded_unsigned_vector_cmpx_state.vgprs[61][0] = 8u;
  decoded_unsigned_vector_cmpx_state.vgprs[61][1] = 9u;
  decoded_unsigned_vector_cmpx_state.vgprs[61][3] = 9u;
  decoded_unsigned_vector_cmpx_state.vgprs[62][0] = 6u;
  decoded_unsigned_vector_cmpx_state.vgprs[63][0] = 6u;
  if (!Expect(interpreter.ExecuteProgram(unsigned_vector_cmpx_program,
                                         &decoded_unsigned_vector_cmpx_state,
                                         &error_message),
              "expected decoded unsigned vector CMPX execution success") ||
      !Expect(ExpectUnsignedVectorCmpxState(
                  decoded_unsigned_vector_cmpx_state),
              "expected decoded unsigned vector CMPX state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_unsigned_vector_cmpx_program;
  if (!Expect(interpreter.CompileProgram(unsigned_vector_cmpx_program,
                                         &compiled_unsigned_vector_cmpx_program,
                                         &error_message),
              "expected compiled unsigned vector CMPX program success") ||
      !Expect(compiled_unsigned_vector_cmpx_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxEqU32,
              "expected compiled V_CMPX_EQ_U32 opcode") ||
      !Expect(compiled_unsigned_vector_cmpx_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxNeU32,
              "expected compiled V_CMPX_NE_U32 opcode") ||
      !Expect(compiled_unsigned_vector_cmpx_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLtU32,
              "expected compiled V_CMPX_LT_U32 opcode") ||
      !Expect(compiled_unsigned_vector_cmpx_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLeU32,
              "expected compiled V_CMPX_LE_U32 opcode") ||
      !Expect(compiled_unsigned_vector_cmpx_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGtU32,
              "expected compiled V_CMPX_GT_U32 opcode") ||
      !Expect(compiled_unsigned_vector_cmpx_program[19].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGeU32,
              "expected compiled V_CMPX_GE_U32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_unsigned_vector_cmpx_state;
  compiled_unsigned_vector_cmpx_state.exec_mask = 0xbu;
  compiled_unsigned_vector_cmpx_state.vgprs[60][0] = 7u;
  compiled_unsigned_vector_cmpx_state.vgprs[60][1] = 4u;
  compiled_unsigned_vector_cmpx_state.vgprs[60][3] = 7u;
  compiled_unsigned_vector_cmpx_state.vgprs[61][0] = 8u;
  compiled_unsigned_vector_cmpx_state.vgprs[61][1] = 9u;
  compiled_unsigned_vector_cmpx_state.vgprs[61][3] = 9u;
  compiled_unsigned_vector_cmpx_state.vgprs[62][0] = 6u;
  compiled_unsigned_vector_cmpx_state.vgprs[63][0] = 6u;
  if (!Expect(interpreter.ExecuteProgram(compiled_unsigned_vector_cmpx_program,
                                         &compiled_unsigned_vector_cmpx_state,
                                         &error_message),
              "expected compiled unsigned vector CMPX execution success") ||
      !Expect(ExpectUnsignedVectorCmpxState(
                  compiled_unsigned_vector_cmpx_state),
              "expected compiled unsigned vector CMPX state")) {
    return 1;
  }

  const std::array<std::uint32_t, 21> signed_vector_cmpx_words{
      MakeSopk(0u, 76u, 0xfffdu),
      MakeSopk(0u, 77u, 2u),
      MakeVopc(194u, 76u, 64u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 97u, 777u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 97u, 888u),
      MakeVopc(197u, 76u, 65u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 98u, 999u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 98u, 1001u),
      MakeVopc(193u, 76u, 66u),
      MakeVopc(195u, 76u, 66u),
      MakeVopc(196u, 77u, 67u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 99u, 123u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 99u, 321u),
      MakeVopc(198u, 77u, 67u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> signed_vector_cmpx_program;
  if (!Expect(decoder.DecodeProgram(signed_vector_cmpx_words,
                                    &signed_vector_cmpx_program,
                                    &error_message),
              "expected signed vector CMPX program decode success") ||
      !Expect(signed_vector_cmpx_program.size() == 21u,
              "expected twenty-one decoded signed vector CMPX instructions") ||
      !Expect(signed_vector_cmpx_program[2].opcode == "V_CMPX_EQ_I32",
              "expected decoded V_CMPX_EQ_I32") ||
      !Expect(signed_vector_cmpx_program[7].opcode == "V_CMPX_NE_I32",
              "expected decoded V_CMPX_NE_I32") ||
      !Expect(signed_vector_cmpx_program[12].opcode == "V_CMPX_LT_I32",
              "expected decoded V_CMPX_LT_I32") ||
      !Expect(signed_vector_cmpx_program[13].opcode == "V_CMPX_LE_I32",
              "expected decoded V_CMPX_LE_I32") ||
      !Expect(signed_vector_cmpx_program[14].opcode == "V_CMPX_GT_I32",
              "expected decoded V_CMPX_GT_I32") ||
      !Expect(signed_vector_cmpx_program[19].opcode == "V_CMPX_GE_I32",
              "expected decoded V_CMPX_GE_I32")) {
    return 1;
  }

  WaveExecutionState decoded_signed_vector_cmpx_state;
  decoded_signed_vector_cmpx_state.exec_mask = 0xbu;
  decoded_signed_vector_cmpx_state.vgprs[64][0] = 0xfffffffdu;
  decoded_signed_vector_cmpx_state.vgprs[64][1] = 4u;
  decoded_signed_vector_cmpx_state.vgprs[64][3] = 0xfffffffdu;
  decoded_signed_vector_cmpx_state.vgprs[65][0] = 0xfffffffcu;
  decoded_signed_vector_cmpx_state.vgprs[65][1] = 0xfffffffdu;
  decoded_signed_vector_cmpx_state.vgprs[65][3] = 0xfffffffdu;
  decoded_signed_vector_cmpx_state.vgprs[66][0] = 0xffffffffu;
  decoded_signed_vector_cmpx_state.vgprs[67][0] = 3u;
  if (!Expect(interpreter.ExecuteProgram(signed_vector_cmpx_program,
                                         &decoded_signed_vector_cmpx_state,
                                         &error_message),
              "expected decoded signed vector CMPX execution success") ||
      !Expect(ExpectSignedVectorCmpxState(
                  decoded_signed_vector_cmpx_state),
              "expected decoded signed vector CMPX state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_signed_vector_cmpx_program;
  if (!Expect(interpreter.CompileProgram(signed_vector_cmpx_program,
                                         &compiled_signed_vector_cmpx_program,
                                         &error_message),
              "expected compiled signed vector CMPX program success") ||
      !Expect(compiled_signed_vector_cmpx_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxEqI32,
              "expected compiled V_CMPX_EQ_I32 opcode") ||
      !Expect(compiled_signed_vector_cmpx_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxNeI32,
              "expected compiled V_CMPX_NE_I32 opcode") ||
      !Expect(compiled_signed_vector_cmpx_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLtI32,
              "expected compiled V_CMPX_LT_I32 opcode") ||
      !Expect(compiled_signed_vector_cmpx_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLeI32,
              "expected compiled V_CMPX_LE_I32 opcode") ||
      !Expect(compiled_signed_vector_cmpx_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGtI32,
              "expected compiled V_CMPX_GT_I32 opcode") ||
      !Expect(compiled_signed_vector_cmpx_program[19].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGeI32,
              "expected compiled V_CMPX_GE_I32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_signed_vector_cmpx_state;
  compiled_signed_vector_cmpx_state.exec_mask = 0xbu;
  compiled_signed_vector_cmpx_state.vgprs[64][0] = 0xfffffffdu;
  compiled_signed_vector_cmpx_state.vgprs[64][1] = 4u;
  compiled_signed_vector_cmpx_state.vgprs[64][3] = 0xfffffffdu;
  compiled_signed_vector_cmpx_state.vgprs[65][0] = 0xfffffffcu;
  compiled_signed_vector_cmpx_state.vgprs[65][1] = 0xfffffffdu;
  compiled_signed_vector_cmpx_state.vgprs[65][3] = 0xfffffffdu;
  compiled_signed_vector_cmpx_state.vgprs[66][0] = 0xffffffffu;
  compiled_signed_vector_cmpx_state.vgprs[67][0] = 3u;
  if (!Expect(interpreter.ExecuteProgram(compiled_signed_vector_cmpx_program,
                                         &compiled_signed_vector_cmpx_state,
                                         &error_message),
              "expected compiled signed vector CMPX execution success") ||
      !Expect(ExpectSignedVectorCmpxState(
                  compiled_signed_vector_cmpx_state),
              "expected compiled signed vector CMPX state")) {
    return 1;
  }

  const std::array<std::uint32_t, 23> float_vector_compare_words{
      MakeSop1(0u, 110u, 255u),
      FloatBits(1.5f),
      MakeSop1(0u, 111u, 255u),
      FloatBits(2.5f),
      MakeSop1(0u, 112u, 255u),
      kQuietNaNF32Bits,
      MakeVopc(18u, 110u, 70u),
      MakeSopp(36u, 2u),
      MakeSopk(0u, 100u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 100u, 222u),
      MakeVopc(17u, 110u, 71u),
      MakeSopp(35u, 2u),
      MakeSopk(0u, 101u, 333u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 101u, 444u),
      MakeVopc(19u, 110u, 71u),
      MakeVopc(20u, 111u, 74u),
      MakeVopc(22u, 111u, 74u),
      MakeVopc(21u, 110u, 75u),
      MakeVopc(29u, 110u, 75u),
      MakeVopc(126u, 112u, 76u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> float_vector_compare_program;
  if (!Expect(decoder.DecodeProgram(float_vector_compare_words,
                                    &float_vector_compare_program,
                                    &error_message),
              "expected float vector compare program decode success") ||
      !Expect(float_vector_compare_program.size() == 20u,
              "expected twenty decoded float vector compare instructions") ||
      !Expect(float_vector_compare_program[3].opcode == "V_CMP_EQ_F32",
              "expected decoded V_CMP_EQ_F32") ||
      !Expect(float_vector_compare_program[8].opcode == "V_CMP_LT_F32",
              "expected decoded V_CMP_LT_F32") ||
      !Expect(float_vector_compare_program[13].opcode == "V_CMP_LE_F32",
              "expected decoded V_CMP_LE_F32") ||
      !Expect(float_vector_compare_program[14].opcode == "V_CMP_GT_F32",
              "expected decoded V_CMP_GT_F32") ||
      !Expect(float_vector_compare_program[15].opcode == "V_CMP_GE_F32",
              "expected decoded V_CMP_GE_F32") ||
      !Expect(float_vector_compare_program[16].opcode == "V_CMP_LG_F32",
              "expected decoded V_CMP_LG_F32") ||
      !Expect(float_vector_compare_program[17].opcode == "V_CMP_NEQ_F32",
              "expected decoded V_CMP_NEQ_F32") ||
      !Expect(float_vector_compare_program[18].opcode == "V_CMP_CLASS_F32",
              "expected decoded V_CMP_CLASS_F32")) {
    return 1;
  }

  WaveExecutionState decoded_float_vector_compare_state;
  decoded_float_vector_compare_state.exec_mask = 0xbu;
  decoded_float_vector_compare_state.vgprs[70][0] = FloatBits(1.5f);
  decoded_float_vector_compare_state.vgprs[70][1] = FloatBits(2.0f);
  decoded_float_vector_compare_state.vgprs[70][3] = FloatBits(1.5f);
  decoded_float_vector_compare_state.vgprs[71][0] = FloatBits(2.0f);
  decoded_float_vector_compare_state.vgprs[71][1] = FloatBits(1.5f);
  decoded_float_vector_compare_state.vgprs[71][3] = kQuietNaNF32Bits;
  decoded_float_vector_compare_state.vgprs[74][0] = FloatBits(1.0f);
  decoded_float_vector_compare_state.vgprs[74][1] = FloatBits(2.5f);
  decoded_float_vector_compare_state.vgprs[74][3] = FloatBits(1.0f);
  decoded_float_vector_compare_state.vgprs[75][0] = FloatBits(2.0f);
  decoded_float_vector_compare_state.vgprs[75][1] = FloatBits(1.5f);
  decoded_float_vector_compare_state.vgprs[75][3] = kQuietNaNF32Bits;
  decoded_float_vector_compare_state.vgprs[76][0] = 0x002u;
  decoded_float_vector_compare_state.vgprs[76][1] = 0x001u;
  decoded_float_vector_compare_state.vgprs[76][3] = 0x003u;
  if (!Expect(interpreter.ExecuteProgram(float_vector_compare_program,
                                         &decoded_float_vector_compare_state,
                                         &error_message),
              "expected decoded float vector compare execution success") ||
      !Expect(ExpectFloatVectorCompareState(decoded_float_vector_compare_state),
              "expected decoded float vector compare state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_float_vector_compare_program;
  if (!Expect(interpreter.CompileProgram(float_vector_compare_program,
                                         &compiled_float_vector_compare_program,
                                         &error_message),
              "expected compiled float vector compare program success") ||
      !Expect(compiled_float_vector_compare_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCmpEqF32,
              "expected compiled V_CMP_EQ_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLtF32,
              "expected compiled V_CMP_LT_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLeF32,
              "expected compiled V_CMP_LE_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGtF32,
              "expected compiled V_CMP_GT_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[15].opcode ==
                  Gfx1201CompiledOpcode::kVCmpGeF32,
              "expected compiled V_CMP_GE_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[16].opcode ==
                  Gfx1201CompiledOpcode::kVCmpLgF32,
              "expected compiled V_CMP_LG_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[17].opcode ==
                  Gfx1201CompiledOpcode::kVCmpNeqF32,
              "expected compiled V_CMP_NEQ_F32 opcode") ||
      !Expect(compiled_float_vector_compare_program[18].opcode ==
                  Gfx1201CompiledOpcode::kVCmpClassF32,
              "expected compiled V_CMP_CLASS_F32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_float_vector_compare_state;
  compiled_float_vector_compare_state.exec_mask = 0xbu;
  compiled_float_vector_compare_state.vgprs[70][0] = FloatBits(1.5f);
  compiled_float_vector_compare_state.vgprs[70][1] = FloatBits(2.0f);
  compiled_float_vector_compare_state.vgprs[70][3] = FloatBits(1.5f);
  compiled_float_vector_compare_state.vgprs[71][0] = FloatBits(2.0f);
  compiled_float_vector_compare_state.vgprs[71][1] = FloatBits(1.5f);
  compiled_float_vector_compare_state.vgprs[71][3] = kQuietNaNF32Bits;
  compiled_float_vector_compare_state.vgprs[74][0] = FloatBits(1.0f);
  compiled_float_vector_compare_state.vgprs[74][1] = FloatBits(2.5f);
  compiled_float_vector_compare_state.vgprs[74][3] = FloatBits(1.0f);
  compiled_float_vector_compare_state.vgprs[75][0] = FloatBits(2.0f);
  compiled_float_vector_compare_state.vgprs[75][1] = FloatBits(1.5f);
  compiled_float_vector_compare_state.vgprs[75][3] = kQuietNaNF32Bits;
  compiled_float_vector_compare_state.vgprs[76][0] = 0x002u;
  compiled_float_vector_compare_state.vgprs[76][1] = 0x001u;
  compiled_float_vector_compare_state.vgprs[76][3] = 0x003u;
  if (!Expect(interpreter.ExecuteProgram(compiled_float_vector_compare_program,
                                         &compiled_float_vector_compare_state,
                                         &error_message),
              "expected compiled float vector compare execution success") ||
      !Expect(ExpectFloatVectorCompareState(
                  compiled_float_vector_compare_state),
              "expected compiled float vector compare state")) {
    return 1;
  }

  const std::array<std::uint32_t, 27> float_vector_cmpx_words{
      MakeSop1(0u, 113u, 255u),
      FloatBits(1.5f),
      MakeSop1(0u, 114u, 255u),
      FloatBits(2.5f),
      MakeSop1(0u, 115u, 255u),
      kQuietNaNF32Bits,
      MakeVopc(146u, 113u, 80u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 102u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 102u, 222u),
      MakeVopc(145u, 113u, 81u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 103u, 333u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 103u, 444u),
      MakeVopc(147u, 113u, 81u),
      MakeVopc(148u, 114u, 84u),
      MakeVopc(150u, 114u, 84u),
      MakeVopc(149u, 113u, 85u),
      MakeVopc(157u, 113u, 86u),
      MakeVopc(254u, 115u, 87u),
      MakeSopp(37u, 2u),
      MakeSopk(0u, 104u, 555u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 104u, 666u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> float_vector_cmpx_program;
  if (!Expect(decoder.DecodeProgram(float_vector_cmpx_words,
                                    &float_vector_cmpx_program,
                                    &error_message),
              "expected float vector CMPX program decode success") ||
      !Expect(float_vector_cmpx_program.size() == 24u,
              "expected twenty-four decoded float vector CMPX instructions") ||
      !Expect(float_vector_cmpx_program[3].opcode == "V_CMPX_EQ_F32",
              "expected decoded V_CMPX_EQ_F32") ||
      !Expect(float_vector_cmpx_program[8].opcode == "V_CMPX_LT_F32",
              "expected decoded V_CMPX_LT_F32") ||
      !Expect(float_vector_cmpx_program[13].opcode == "V_CMPX_LE_F32",
              "expected decoded V_CMPX_LE_F32") ||
      !Expect(float_vector_cmpx_program[14].opcode == "V_CMPX_GT_F32",
              "expected decoded V_CMPX_GT_F32") ||
      !Expect(float_vector_cmpx_program[15].opcode == "V_CMPX_GE_F32",
              "expected decoded V_CMPX_GE_F32") ||
      !Expect(float_vector_cmpx_program[16].opcode == "V_CMPX_LG_F32",
              "expected decoded V_CMPX_LG_F32") ||
      !Expect(float_vector_cmpx_program[17].opcode == "V_CMPX_NEQ_F32",
              "expected decoded V_CMPX_NEQ_F32") ||
      !Expect(float_vector_cmpx_program[18].opcode == "V_CMPX_CLASS_F32",
              "expected decoded V_CMPX_CLASS_F32")) {
    return 1;
  }

  WaveExecutionState decoded_float_vector_cmpx_state;
  decoded_float_vector_cmpx_state.exec_mask = 0xbu;
  decoded_float_vector_cmpx_state.vgprs[80][0] = FloatBits(1.5f);
  decoded_float_vector_cmpx_state.vgprs[80][1] = FloatBits(2.0f);
  decoded_float_vector_cmpx_state.vgprs[80][3] = FloatBits(1.5f);
  decoded_float_vector_cmpx_state.vgprs[81][0] = FloatBits(2.0f);
  decoded_float_vector_cmpx_state.vgprs[81][1] = FloatBits(1.5f);
  decoded_float_vector_cmpx_state.vgprs[81][3] = kQuietNaNF32Bits;
  decoded_float_vector_cmpx_state.vgprs[84][0] = FloatBits(1.0f);
  decoded_float_vector_cmpx_state.vgprs[85][0] = FloatBits(2.0f);
  decoded_float_vector_cmpx_state.vgprs[86][0] = kQuietNaNF32Bits;
  decoded_float_vector_cmpx_state.vgprs[87][0] = 0x001u;
  if (!Expect(interpreter.ExecuteProgram(float_vector_cmpx_program,
                                         &decoded_float_vector_cmpx_state,
                                         &error_message),
              "expected decoded float vector CMPX execution success") ||
      !Expect(ExpectFloatVectorCmpxState(decoded_float_vector_cmpx_state),
              "expected decoded float vector CMPX state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_float_vector_cmpx_program;
  if (!Expect(interpreter.CompileProgram(float_vector_cmpx_program,
                                         &compiled_float_vector_cmpx_program,
                                         &error_message),
              "expected compiled float vector CMPX program success") ||
      !Expect(compiled_float_vector_cmpx_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxEqF32,
              "expected compiled V_CMPX_EQ_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLtF32,
              "expected compiled V_CMPX_LT_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLeF32,
              "expected compiled V_CMPX_LE_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGtF32,
              "expected compiled V_CMPX_GT_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[15].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxGeF32,
              "expected compiled V_CMPX_GE_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[16].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxLgF32,
              "expected compiled V_CMPX_LG_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[17].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxNeqF32,
              "expected compiled V_CMPX_NEQ_F32 opcode") ||
      !Expect(compiled_float_vector_cmpx_program[18].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxClassF32,
              "expected compiled V_CMPX_CLASS_F32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_float_vector_cmpx_state;
  compiled_float_vector_cmpx_state.exec_mask = 0xbu;
  compiled_float_vector_cmpx_state.vgprs[80][0] = FloatBits(1.5f);
  compiled_float_vector_cmpx_state.vgprs[80][1] = FloatBits(2.0f);
  compiled_float_vector_cmpx_state.vgprs[80][3] = FloatBits(1.5f);
  compiled_float_vector_cmpx_state.vgprs[81][0] = FloatBits(2.0f);
  compiled_float_vector_cmpx_state.vgprs[81][1] = FloatBits(1.5f);
  compiled_float_vector_cmpx_state.vgprs[81][3] = kQuietNaNF32Bits;
  compiled_float_vector_cmpx_state.vgprs[84][0] = FloatBits(1.0f);
  compiled_float_vector_cmpx_state.vgprs[85][0] = FloatBits(2.0f);
  compiled_float_vector_cmpx_state.vgprs[86][0] = kQuietNaNF32Bits;
  compiled_float_vector_cmpx_state.vgprs[87][0] = 0x001u;
  if (!Expect(interpreter.ExecuteProgram(compiled_float_vector_cmpx_program,
                                         &compiled_float_vector_cmpx_state,
                                         &error_message),
              "expected compiled float vector CMPX execution success") ||
      !Expect(ExpectFloatVectorCmpxState(compiled_float_vector_cmpx_state),
              "expected compiled float vector CMPX state")) {
    return 1;
  }

  struct RemainingFloatCompareCase {
    const char* opcode;
    std::uint32_t encoded_opcode;
    std::uint64_t expected_mask;
    Gfx1201CompiledOpcode compiled_opcode;
    bool writes_exec;
  };

  constexpr std::array<RemainingFloatCompareCase, 14>
      kRemainingFloatCompareCases{{
          {"V_CMP_O_F32", 23u, 7u, Gfx1201CompiledOpcode::kVCmpOF32, false},
          {"V_CMP_U_F32", 24u, 8u, Gfx1201CompiledOpcode::kVCmpUF32, false},
          {"V_CMP_NGE_F32", 25u, 12u, Gfx1201CompiledOpcode::kVCmpNgeF32,
           false},
          {"V_CMP_NLG_F32", 26u, 10u, Gfx1201CompiledOpcode::kVCmpNlgF32,
           false},
          {"V_CMP_NGT_F32", 27u, 14u, Gfx1201CompiledOpcode::kVCmpNgtF32,
           false},
          {"V_CMP_NLE_F32", 28u, 9u, Gfx1201CompiledOpcode::kVCmpNleF32,
           false},
          {"V_CMP_NLT_F32", 30u, 11u, Gfx1201CompiledOpcode::kVCmpNltF32,
           false},
          {"V_CMPX_O_F32", 151u, 7u, Gfx1201CompiledOpcode::kVCmpxOF32, true},
          {"V_CMPX_U_F32", 152u, 8u, Gfx1201CompiledOpcode::kVCmpxUF32, true},
          {"V_CMPX_NGE_F32", 153u, 12u, Gfx1201CompiledOpcode::kVCmpxNgeF32,
           true},
          {"V_CMPX_NLG_F32", 154u, 10u, Gfx1201CompiledOpcode::kVCmpxNlgF32,
           true},
          {"V_CMPX_NGT_F32", 155u, 14u, Gfx1201CompiledOpcode::kVCmpxNgtF32,
           true},
          {"V_CMPX_NLE_F32", 156u, 9u, Gfx1201CompiledOpcode::kVCmpxNleF32,
           true},
          {"V_CMPX_NLT_F32", 158u, 11u, Gfx1201CompiledOpcode::kVCmpxNltF32,
           true},
      }};

  for (const RemainingFloatCompareCase& test_case : kRemainingFloatCompareCases) {
    const std::array<std::uint32_t, 4> remaining_float_compare_words{
        MakeSop1(0u, 118u, 255u),
        FloatBits(2.0f),
        MakeVopc(test_case.encoded_opcode, 118u, 92u),
        MakeSopp(48u),
    };
    std::vector<DecodedInstruction> remaining_float_compare_program;
    if (!Expect(decoder.DecodeProgram(remaining_float_compare_words,
                                      &remaining_float_compare_program,
                                      &error_message),
                "expected remaining float compare program decode success") ||
        !Expect(remaining_float_compare_program.size() == 3u,
                "expected three decoded remaining float compare instructions")
        ||
        !Expect(remaining_float_compare_program[1].opcode == test_case.opcode,
                "expected decoded remaining float compare opcode")) {
      return 1;
    }

    WaveExecutionState decoded_remaining_float_compare_state;
    decoded_remaining_float_compare_state.exec_mask = 0xfu;
    decoded_remaining_float_compare_state.vcc_mask = 0x80u;
    decoded_remaining_float_compare_state.vgprs[92][0] = FloatBits(1.0f);
    decoded_remaining_float_compare_state.vgprs[92][1] = FloatBits(2.0f);
    decoded_remaining_float_compare_state.vgprs[92][2] = FloatBits(3.0f);
    decoded_remaining_float_compare_state.vgprs[92][3] = kQuietNaNF32Bits;
    if (!Expect(interpreter.ExecuteProgram(remaining_float_compare_program,
                                           &decoded_remaining_float_compare_state,
                                           &error_message),
                "expected decoded remaining float compare execution success") ||
        !Expect(decoded_remaining_float_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected remaining float compare VCC mask") ||
        !Expect(decoded_remaining_float_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected remaining float compare EXEC mask") ||
        !Expect(decoded_remaining_float_compare_state.halted,
                "expected remaining float compare program to halt") ||
        !Expect(decoded_remaining_float_compare_state.pc == 2u,
                "expected remaining float compare program advance")) {
      return 1;
    }

    std::vector<Gfx1201CompiledInstruction>
        compiled_remaining_float_compare_program;
    if (!Expect(interpreter.CompileProgram(remaining_float_compare_program,
                                           &compiled_remaining_float_compare_program,
                                           &error_message),
                "expected compiled remaining float compare program success") ||
        !Expect(compiled_remaining_float_compare_program[1].opcode ==
                    test_case.compiled_opcode,
                "expected compiled remaining float compare opcode")) {
      return 1;
    }

    WaveExecutionState compiled_remaining_float_compare_state;
    compiled_remaining_float_compare_state.exec_mask = 0xfu;
    compiled_remaining_float_compare_state.vcc_mask = 0x80u;
    compiled_remaining_float_compare_state.vgprs[92][0] = FloatBits(1.0f);
    compiled_remaining_float_compare_state.vgprs[92][1] = FloatBits(2.0f);
    compiled_remaining_float_compare_state.vgprs[92][2] = FloatBits(3.0f);
    compiled_remaining_float_compare_state.vgprs[92][3] = kQuietNaNF32Bits;
    if (!Expect(interpreter.ExecuteProgram(
                    compiled_remaining_float_compare_program,
                    &compiled_remaining_float_compare_state, &error_message),
                "expected compiled remaining float compare execution success")
        ||
        !Expect(compiled_remaining_float_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected compiled remaining float compare VCC mask") ||
        !Expect(compiled_remaining_float_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected compiled remaining float compare EXEC mask") ||
        !Expect(compiled_remaining_float_compare_state.halted,
                "expected compiled remaining float compare program to halt") ||
        !Expect(compiled_remaining_float_compare_state.pc == 2u,
                "expected compiled remaining float compare program advance")) {
      return 1;
    }
  }

  const std::array<std::uint32_t, 9> vcc_cndmask_words{
      MakeSop1(0u, 116u, 255u),
      FloatBits(2.0f),
      MakeVopc(24u, 116u, 90u),
      MakeSopp(36u, 2u),
      MakeSopk(0u, 120u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 120u, 222u),
      MakeVop2(1u, 42u, 296u, 41u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> vcc_cndmask_program;
  if (!Expect(decoder.DecodeProgram(vcc_cndmask_words, &vcc_cndmask_program,
                                    &error_message),
              "expected VCC-driven cndmask program decode success") ||
      !Expect(vcc_cndmask_program.size() == 8u,
              "expected eight decoded VCC-driven cndmask instructions") ||
      !Expect(vcc_cndmask_program[1].opcode == "V_CMP_U_F32",
              "expected decoded V_CMP_U_F32") ||
      !Expect(vcc_cndmask_program[2].opcode == "S_CBRANCH_VCCNZ",
              "expected decoded S_CBRANCH_VCCNZ") ||
      !Expect(vcc_cndmask_program[6].opcode == "V_CNDMASK_B32",
              "expected decoded V_CNDMASK_B32")) {
    return 1;
  }

  WaveExecutionState decoded_vcc_cndmask_state;
  decoded_vcc_cndmask_state.exec_mask = 0xfu;
  decoded_vcc_cndmask_state.vgprs[40][0] = 10u;
  decoded_vcc_cndmask_state.vgprs[40][1] = 10u;
  decoded_vcc_cndmask_state.vgprs[40][2] = 10u;
  decoded_vcc_cndmask_state.vgprs[40][3] = 10u;
  decoded_vcc_cndmask_state.vgprs[41][0] = 20u;
  decoded_vcc_cndmask_state.vgprs[41][1] = 20u;
  decoded_vcc_cndmask_state.vgprs[41][2] = 20u;
  decoded_vcc_cndmask_state.vgprs[41][3] = 20u;
  decoded_vcc_cndmask_state.vgprs[42][0] = 0xaaaaaaaau;
  decoded_vcc_cndmask_state.vgprs[42][1] = 0xbbbbbbbbu;
  decoded_vcc_cndmask_state.vgprs[42][2] = 0xccccccccu;
  decoded_vcc_cndmask_state.vgprs[42][3] = 0xddddddddu;
  decoded_vcc_cndmask_state.vgprs[90][0] = FloatBits(1.0f);
  decoded_vcc_cndmask_state.vgprs[90][1] = FloatBits(2.0f);
  decoded_vcc_cndmask_state.vgprs[90][2] = FloatBits(3.0f);
  decoded_vcc_cndmask_state.vgprs[90][3] = kQuietNaNF32Bits;
  if (!Expect(interpreter.ExecuteProgram(vcc_cndmask_program,
                                         &decoded_vcc_cndmask_state,
                                         &error_message),
              "expected decoded VCC-driven cndmask execution success") ||
      !Expect(ExpectVccDrivenCndmaskState(decoded_vcc_cndmask_state),
              "expected decoded VCC-driven cndmask state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_vcc_cndmask_program;
  if (!Expect(interpreter.CompileProgram(vcc_cndmask_program,
                                         &compiled_vcc_cndmask_program,
                                         &error_message),
              "expected compiled VCC-driven cndmask program success") ||
      !Expect(compiled_vcc_cndmask_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCmpUF32,
              "expected compiled V_CMP_U_F32 opcode") ||
      !Expect(compiled_vcc_cndmask_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchVccnz,
              "expected compiled S_CBRANCH_VCCNZ opcode") ||
      !Expect(compiled_vcc_cndmask_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCndmaskB32,
              "expected compiled V_CNDMASK_B32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_vcc_cndmask_state;
  compiled_vcc_cndmask_state.exec_mask = 0xfu;
  compiled_vcc_cndmask_state.vgprs[40][0] = 10u;
  compiled_vcc_cndmask_state.vgprs[40][1] = 10u;
  compiled_vcc_cndmask_state.vgprs[40][2] = 10u;
  compiled_vcc_cndmask_state.vgprs[40][3] = 10u;
  compiled_vcc_cndmask_state.vgprs[41][0] = 20u;
  compiled_vcc_cndmask_state.vgprs[41][1] = 20u;
  compiled_vcc_cndmask_state.vgprs[41][2] = 20u;
  compiled_vcc_cndmask_state.vgprs[41][3] = 20u;
  compiled_vcc_cndmask_state.vgprs[42][0] = 0xaaaaaaaau;
  compiled_vcc_cndmask_state.vgprs[42][1] = 0xbbbbbbbbu;
  compiled_vcc_cndmask_state.vgprs[42][2] = 0xccccccccu;
  compiled_vcc_cndmask_state.vgprs[42][3] = 0xddddddddu;
  compiled_vcc_cndmask_state.vgprs[90][0] = FloatBits(1.0f);
  compiled_vcc_cndmask_state.vgprs[90][1] = FloatBits(2.0f);
  compiled_vcc_cndmask_state.vgprs[90][2] = FloatBits(3.0f);
  compiled_vcc_cndmask_state.vgprs[90][3] = kQuietNaNF32Bits;
  if (!Expect(interpreter.ExecuteProgram(compiled_vcc_cndmask_program,
                                         &compiled_vcc_cndmask_state,
                                         &error_message),
              "expected compiled VCC-driven cndmask execution success") ||
      !Expect(ExpectVccDrivenCndmaskState(compiled_vcc_cndmask_state),
              "expected compiled VCC-driven cndmask state")) {
    return 1;
  }

  const std::array<std::uint32_t, 8> float_cmpx_exec_branch_words{
      MakeSop1(0u, 117u, 255u),
      FloatBits(2.0f),
      MakeVopc(152u, 117u, 91u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 121u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 121u, 222u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> float_cmpx_exec_branch_program;
  if (!Expect(decoder.DecodeProgram(float_cmpx_exec_branch_words,
                                    &float_cmpx_exec_branch_program,
                                    &error_message),
              "expected float CMPX exec-branch program decode success") ||
      !Expect(float_cmpx_exec_branch_program.size() == 7u,
              "expected seven decoded float CMPX exec-branch instructions") ||
      !Expect(float_cmpx_exec_branch_program[1].opcode == "V_CMPX_U_F32",
              "expected decoded V_CMPX_U_F32") ||
      !Expect(float_cmpx_exec_branch_program[2].opcode == "S_CBRANCH_EXECNZ",
              "expected decoded S_CBRANCH_EXECNZ")) {
    return 1;
  }

  WaveExecutionState decoded_float_cmpx_exec_branch_state;
  decoded_float_cmpx_exec_branch_state.exec_mask = 0xfu;
  decoded_float_cmpx_exec_branch_state.vgprs[91][0] = FloatBits(1.0f);
  decoded_float_cmpx_exec_branch_state.vgprs[91][1] = FloatBits(2.0f);
  decoded_float_cmpx_exec_branch_state.vgprs[91][2] = FloatBits(3.0f);
  decoded_float_cmpx_exec_branch_state.vgprs[91][3] = kQuietNaNF32Bits;
  if (!Expect(interpreter.ExecuteProgram(float_cmpx_exec_branch_program,
                                         &decoded_float_cmpx_exec_branch_state,
                                         &error_message),
              "expected decoded float CMPX exec-branch execution success") ||
      !Expect(ExpectFloatVectorCmpxExecBranchState(
                  decoded_float_cmpx_exec_branch_state),
              "expected decoded float CMPX exec-branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction>
      compiled_float_cmpx_exec_branch_program;
  if (!Expect(interpreter.CompileProgram(float_cmpx_exec_branch_program,
                                         &compiled_float_cmpx_exec_branch_program,
                                         &error_message),
              "expected compiled float CMPX exec-branch program success") ||
      !Expect(compiled_float_cmpx_exec_branch_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxUF32,
              "expected compiled V_CMPX_U_F32 opcode") ||
      !Expect(compiled_float_cmpx_exec_branch_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchExecnz,
              "expected compiled S_CBRANCH_EXECNZ opcode")) {
    return 1;
  }

  WaveExecutionState compiled_float_cmpx_exec_branch_state;
  compiled_float_cmpx_exec_branch_state.exec_mask = 0xfu;
  compiled_float_cmpx_exec_branch_state.vgprs[91][0] = FloatBits(1.0f);
  compiled_float_cmpx_exec_branch_state.vgprs[91][1] = FloatBits(2.0f);
  compiled_float_cmpx_exec_branch_state.vgprs[91][2] = FloatBits(3.0f);
  compiled_float_cmpx_exec_branch_state.vgprs[91][3] = kQuietNaNF32Bits;
  if (!Expect(interpreter.ExecuteProgram(compiled_float_cmpx_exec_branch_program,
                                         &compiled_float_cmpx_exec_branch_state,
                                         &error_message),
              "expected compiled float CMPX exec-branch execution success") ||
      !Expect(ExpectFloatVectorCmpxExecBranchState(
                  compiled_float_cmpx_exec_branch_state),
              "expected compiled float CMPX exec-branch state")) {
    return 1;
  }

  struct RemainingF64CompareCase {
    const char* opcode;
    std::uint32_t encoded_opcode;
    std::uint64_t expected_mask;
    Gfx1201CompiledOpcode compiled_opcode;
    bool writes_exec;
  };

  constexpr std::array<RemainingF64CompareCase, 28> kRemainingF64CompareCases{{
      {"V_CMP_EQ_F64", 34u, 2u, Gfx1201CompiledOpcode::kVCmpEqF64, false},
      {"V_CMP_GE_F64", 38u, 3u, Gfx1201CompiledOpcode::kVCmpGeF64, false},
      {"V_CMP_GT_F64", 36u, 1u, Gfx1201CompiledOpcode::kVCmpGtF64, false},
      {"V_CMP_LE_F64", 35u, 6u, Gfx1201CompiledOpcode::kVCmpLeF64, false},
      {"V_CMP_LG_F64", 37u, 5u, Gfx1201CompiledOpcode::kVCmpLgF64, false},
      {"V_CMP_LT_F64", 33u, 4u, Gfx1201CompiledOpcode::kVCmpLtF64, false},
      {"V_CMP_NEQ_F64", 45u, 13u, Gfx1201CompiledOpcode::kVCmpNeqF64, false},
      {"V_CMP_O_F64", 39u, 7u, Gfx1201CompiledOpcode::kVCmpOF64, false},
      {"V_CMP_U_F64", 40u, 8u, Gfx1201CompiledOpcode::kVCmpUF64, false},
      {"V_CMP_NGE_F64", 41u, 12u, Gfx1201CompiledOpcode::kVCmpNgeF64, false},
      {"V_CMP_NLG_F64", 42u, 10u, Gfx1201CompiledOpcode::kVCmpNlgF64, false},
      {"V_CMP_NGT_F64", 43u, 14u, Gfx1201CompiledOpcode::kVCmpNgtF64, false},
      {"V_CMP_NLE_F64", 44u, 9u, Gfx1201CompiledOpcode::kVCmpNleF64, false},
      {"V_CMP_NLT_F64", 46u, 11u, Gfx1201CompiledOpcode::kVCmpNltF64, false},
      {"V_CMPX_EQ_F64", 162u, 2u, Gfx1201CompiledOpcode::kVCmpxEqF64, true},
      {"V_CMPX_GE_F64", 166u, 3u, Gfx1201CompiledOpcode::kVCmpxGeF64, true},
      {"V_CMPX_GT_F64", 164u, 1u, Gfx1201CompiledOpcode::kVCmpxGtF64, true},
      {"V_CMPX_LE_F64", 163u, 6u, Gfx1201CompiledOpcode::kVCmpxLeF64, true},
      {"V_CMPX_LG_F64", 165u, 5u, Gfx1201CompiledOpcode::kVCmpxLgF64, true},
      {"V_CMPX_LT_F64", 161u, 4u, Gfx1201CompiledOpcode::kVCmpxLtF64, true},
      {"V_CMPX_NEQ_F64", 173u, 13u, Gfx1201CompiledOpcode::kVCmpxNeqF64,
       true},
      {"V_CMPX_O_F64", 167u, 7u, Gfx1201CompiledOpcode::kVCmpxOF64, true},
      {"V_CMPX_U_F64", 168u, 8u, Gfx1201CompiledOpcode::kVCmpxUF64, true},
      {"V_CMPX_NGE_F64", 169u, 12u, Gfx1201CompiledOpcode::kVCmpxNgeF64,
       true},
      {"V_CMPX_NLG_F64", 170u, 10u, Gfx1201CompiledOpcode::kVCmpxNlgF64,
       true},
      {"V_CMPX_NGT_F64", 171u, 14u, Gfx1201CompiledOpcode::kVCmpxNgtF64,
       true},
      {"V_CMPX_NLE_F64", 172u, 9u, Gfx1201CompiledOpcode::kVCmpxNleF64, true},
      {"V_CMPX_NLT_F64", 174u, 11u, Gfx1201CompiledOpcode::kVCmpxNltF64,
       true},
  }};

  const std::uint64_t lhs_f64_bits = DoubleBits(2.0);
  const std::uint64_t rhs_f64_lane0 = DoubleBits(1.0);
  const std::uint64_t rhs_f64_lane1 = DoubleBits(2.0);
  const std::uint64_t rhs_f64_lane2 = DoubleBits(3.0);

  for (const RemainingF64CompareCase& test_case : kRemainingF64CompareCases) {
    std::uint32_t lhs_low = 0;
    std::uint32_t lhs_high = 0;
    SplitU64(lhs_f64_bits, &lhs_low, &lhs_high);
    const std::array<std::uint32_t, 6> remaining_f64_compare_words{
        MakeSop1(0u, 120u, 255u),
        lhs_low,
        MakeSop1(0u, 121u, 255u),
        lhs_high,
        MakeVopc(test_case.encoded_opcode, 120u, 96u),
        MakeSopp(48u),
    };
    std::vector<DecodedInstruction> remaining_f64_compare_program;
    if (!Expect(decoder.DecodeProgram(remaining_f64_compare_words,
                                      &remaining_f64_compare_program,
                                      &error_message),
                "expected remaining F64 compare program decode success") ||
        !Expect(remaining_f64_compare_program.size() == 4u,
                "expected four decoded remaining F64 compare instructions") ||
        !Expect(remaining_f64_compare_program[2].opcode == test_case.opcode,
                "expected decoded remaining F64 compare opcode")) {
      return 1;
    }

    auto initialize_f64_compare_state = [&](WaveExecutionState* state) {
      state->exec_mask = 0xfu;
      state->vcc_mask = 0x80u;
      SplitU64(rhs_f64_lane0, &state->vgprs[96][0], &state->vgprs[97][0]);
      SplitU64(rhs_f64_lane1, &state->vgprs[96][1], &state->vgprs[97][1]);
      SplitU64(rhs_f64_lane2, &state->vgprs[96][2], &state->vgprs[97][2]);
      SplitU64(kQuietNaNF64Bits, &state->vgprs[96][3], &state->vgprs[97][3]);
    };

    WaveExecutionState decoded_remaining_f64_compare_state;
    initialize_f64_compare_state(&decoded_remaining_f64_compare_state);
    if (!Expect(interpreter.ExecuteProgram(remaining_f64_compare_program,
                                           &decoded_remaining_f64_compare_state,
                                           &error_message),
                "expected decoded remaining F64 compare execution success") ||
        !Expect(decoded_remaining_f64_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected remaining F64 compare VCC mask") ||
        !Expect(decoded_remaining_f64_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected remaining F64 compare EXEC mask") ||
        !Expect(decoded_remaining_f64_compare_state.halted,
                "expected remaining F64 compare program to halt") ||
        !Expect(decoded_remaining_f64_compare_state.pc == 3u,
                "expected remaining F64 compare program advance")) {
      return 1;
    }

    std::vector<Gfx1201CompiledInstruction>
        compiled_remaining_f64_compare_program;
    if (!Expect(interpreter.CompileProgram(remaining_f64_compare_program,
                                           &compiled_remaining_f64_compare_program,
                                           &error_message),
                "expected compiled remaining F64 compare program success") ||
        !Expect(compiled_remaining_f64_compare_program[2].opcode ==
                    test_case.compiled_opcode,
                "expected compiled remaining F64 compare opcode")) {
      return 1;
    }

    WaveExecutionState compiled_remaining_f64_compare_state;
    initialize_f64_compare_state(&compiled_remaining_f64_compare_state);
    if (!Expect(interpreter.ExecuteProgram(
                    compiled_remaining_f64_compare_program,
                    &compiled_remaining_f64_compare_state, &error_message),
                "expected compiled remaining F64 compare execution success")
        ||
        !Expect(compiled_remaining_f64_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected compiled remaining F64 compare VCC mask") ||
        !Expect(compiled_remaining_f64_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected compiled remaining F64 compare EXEC mask") ||
        !Expect(compiled_remaining_f64_compare_state.halted,
                "expected compiled remaining F64 compare program to halt") ||
        !Expect(compiled_remaining_f64_compare_state.pc == 3u,
                "expected compiled remaining F64 compare program advance")) {
      return 1;
    }
  }

  struct RemainingI64U64CompareCase {
    const char* opcode;
    std::uint32_t encoded_opcode;
    std::uint64_t expected_mask;
    Gfx1201CompiledOpcode compiled_opcode;
    bool writes_exec;
    bool is_signed;
  };

  constexpr std::array<RemainingI64U64CompareCase, 24>
      kRemainingI64U64CompareCases{{
          {"V_CMP_EQ_I64", 82u, 2u, Gfx1201CompiledOpcode::kVCmpEqI64, false,
           true},
          {"V_CMP_NE_I64", 85u, 13u, Gfx1201CompiledOpcode::kVCmpNeI64, false,
           true},
          {"V_CMP_LT_I64", 81u, 12u, Gfx1201CompiledOpcode::kVCmpLtI64, false,
           true},
          {"V_CMP_LE_I64", 83u, 14u, Gfx1201CompiledOpcode::kVCmpLeI64, false,
           true},
          {"V_CMP_GT_I64", 84u, 1u, Gfx1201CompiledOpcode::kVCmpGtI64, false,
           true},
          {"V_CMP_GE_I64", 86u, 3u, Gfx1201CompiledOpcode::kVCmpGeI64, false,
           true},
          {"V_CMP_EQ_U64", 90u, 2u, Gfx1201CompiledOpcode::kVCmpEqU64, false,
           false},
          {"V_CMP_NE_U64", 93u, 13u, Gfx1201CompiledOpcode::kVCmpNeU64, false,
           false},
          {"V_CMP_LT_U64", 89u, 12u, Gfx1201CompiledOpcode::kVCmpLtU64, false,
           false},
          {"V_CMP_LE_U64", 91u, 14u, Gfx1201CompiledOpcode::kVCmpLeU64, false,
           false},
          {"V_CMP_GT_U64", 92u, 1u, Gfx1201CompiledOpcode::kVCmpGtU64, false,
           false},
          {"V_CMP_GE_U64", 94u, 3u, Gfx1201CompiledOpcode::kVCmpGeU64, false,
           false},
          {"V_CMPX_EQ_I64", 210u, 2u, Gfx1201CompiledOpcode::kVCmpxEqI64, true,
           true},
          {"V_CMPX_NE_I64", 213u, 13u, Gfx1201CompiledOpcode::kVCmpxNeI64,
           true, true},
          {"V_CMPX_LT_I64", 209u, 12u, Gfx1201CompiledOpcode::kVCmpxLtI64,
           true, true},
          {"V_CMPX_LE_I64", 211u, 14u, Gfx1201CompiledOpcode::kVCmpxLeI64,
           true, true},
          {"V_CMPX_GT_I64", 212u, 1u, Gfx1201CompiledOpcode::kVCmpxGtI64, true,
           true},
          {"V_CMPX_GE_I64", 214u, 3u, Gfx1201CompiledOpcode::kVCmpxGeI64, true,
           true},
          {"V_CMPX_EQ_U64", 218u, 2u, Gfx1201CompiledOpcode::kVCmpxEqU64, true,
           false},
          {"V_CMPX_NE_U64", 221u, 13u, Gfx1201CompiledOpcode::kVCmpxNeU64,
           true, false},
          {"V_CMPX_LT_U64", 217u, 12u, Gfx1201CompiledOpcode::kVCmpxLtU64,
           true, false},
          {"V_CMPX_LE_U64", 219u, 14u, Gfx1201CompiledOpcode::kVCmpxLeU64,
           true, false},
          {"V_CMPX_GT_U64", 220u, 1u, Gfx1201CompiledOpcode::kVCmpxGtU64, true,
           false},
          {"V_CMPX_GE_U64", 222u, 3u, Gfx1201CompiledOpcode::kVCmpxGeU64, true,
           false},
      }};

  const std::uint64_t lhs_i64_bits =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(-2));
  const std::uint64_t rhs_i64_lane0 =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(-3));
  const std::uint64_t rhs_i64_lane1 =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(-2));
  const std::uint64_t rhs_i64_lane2 =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(1));
  const std::uint64_t rhs_i64_lane3 =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(-1));
  constexpr std::uint64_t lhs_u64_bits = 2u;
  constexpr std::uint64_t rhs_u64_lane0 = 1u;
  constexpr std::uint64_t rhs_u64_lane1 = 2u;
  constexpr std::uint64_t rhs_u64_lane2 = 3u;
  constexpr std::uint64_t rhs_u64_lane3 = 0xffffffffffffffffULL;

  for (const RemainingI64U64CompareCase& test_case :
       kRemainingI64U64CompareCases) {
    const std::uint64_t lhs_bits =
        test_case.is_signed ? lhs_i64_bits : lhs_u64_bits;
    const std::uint64_t rhs_lane0 =
        test_case.is_signed ? rhs_i64_lane0 : rhs_u64_lane0;
    const std::uint64_t rhs_lane1 =
        test_case.is_signed ? rhs_i64_lane1 : rhs_u64_lane1;
    const std::uint64_t rhs_lane2 =
        test_case.is_signed ? rhs_i64_lane2 : rhs_u64_lane2;
    const std::uint64_t rhs_lane3 =
        test_case.is_signed ? rhs_i64_lane3 : rhs_u64_lane3;

    std::uint32_t lhs_low = 0;
    std::uint32_t lhs_high = 0;
    SplitU64(lhs_bits, &lhs_low, &lhs_high);
    const std::array<std::uint32_t, 6> remaining_i64_u64_compare_words{
        MakeSop1(0u, 118u, 255u),
        lhs_low,
        MakeSop1(0u, 119u, 255u),
        lhs_high,
        MakeVopc(test_case.encoded_opcode, 118u, 100u),
        MakeSopp(48u),
    };
    std::vector<DecodedInstruction> remaining_i64_u64_compare_program;
    if (!Expect(decoder.DecodeProgram(remaining_i64_u64_compare_words,
                                      &remaining_i64_u64_compare_program,
                                      &error_message),
                "expected remaining I64/U64 compare program decode success") ||
        !Expect(remaining_i64_u64_compare_program.size() == 4u,
                "expected four decoded remaining I64/U64 compare instructions")
        ||
        !Expect(remaining_i64_u64_compare_program[2].opcode == test_case.opcode,
                "expected decoded remaining I64/U64 compare opcode")) {
      return 1;
    }

    auto initialize_i64_u64_compare_state = [&](WaveExecutionState* state) {
      state->exec_mask = 0xfu;
      state->vcc_mask = 0x80u;
      SplitU64(rhs_lane0, &state->vgprs[100][0], &state->vgprs[101][0]);
      SplitU64(rhs_lane1, &state->vgprs[100][1], &state->vgprs[101][1]);
      SplitU64(rhs_lane2, &state->vgprs[100][2], &state->vgprs[101][2]);
      SplitU64(rhs_lane3, &state->vgprs[100][3], &state->vgprs[101][3]);
    };

    WaveExecutionState decoded_remaining_i64_u64_compare_state;
    initialize_i64_u64_compare_state(&decoded_remaining_i64_u64_compare_state);
    if (!Expect(interpreter.ExecuteProgram(remaining_i64_u64_compare_program,
                                           &decoded_remaining_i64_u64_compare_state,
                                           &error_message),
                "expected decoded remaining I64/U64 compare execution success")
        ||
        !Expect(decoded_remaining_i64_u64_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected remaining I64/U64 compare VCC mask") ||
        !Expect(decoded_remaining_i64_u64_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected remaining I64/U64 compare EXEC mask") ||
        !Expect(decoded_remaining_i64_u64_compare_state.halted,
                "expected remaining I64/U64 compare program to halt") ||
        !Expect(decoded_remaining_i64_u64_compare_state.pc == 3u,
                "expected remaining I64/U64 compare program advance")) {
      return 1;
    }

    std::vector<Gfx1201CompiledInstruction>
        compiled_remaining_i64_u64_compare_program;
    if (!Expect(interpreter.CompileProgram(
                    remaining_i64_u64_compare_program,
                    &compiled_remaining_i64_u64_compare_program,
                    &error_message),
                "expected compiled remaining I64/U64 compare program success")
        ||
        !Expect(compiled_remaining_i64_u64_compare_program[2].opcode ==
                    test_case.compiled_opcode,
                "expected compiled remaining I64/U64 compare opcode")) {
      return 1;
    }

    WaveExecutionState compiled_remaining_i64_u64_compare_state;
    initialize_i64_u64_compare_state(&compiled_remaining_i64_u64_compare_state);
    if (!Expect(interpreter.ExecuteProgram(
                    compiled_remaining_i64_u64_compare_program,
                    &compiled_remaining_i64_u64_compare_state, &error_message),
                "expected compiled remaining I64/U64 compare execution success")
        ||
        !Expect(compiled_remaining_i64_u64_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected compiled remaining I64/U64 compare VCC mask") ||
        !Expect(compiled_remaining_i64_u64_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected compiled remaining I64/U64 compare EXEC mask") ||
        !Expect(compiled_remaining_i64_u64_compare_state.halted,
                "expected compiled remaining I64/U64 compare program to halt")
        ||
        !Expect(compiled_remaining_i64_u64_compare_state.pc == 3u,
                "expected compiled remaining I64/U64 compare program advance")) {
      return 1;
    }
  }

  std::uint32_t neg_zero_low = 0;
  std::uint32_t neg_zero_high = 0;
  SplitU64(DoubleBits(-0.0), &neg_zero_low, &neg_zero_high);
  const std::array<std::uint32_t, 7> f64_class_cndmask_words{
      MakeSop1(0u, 124u, 255u),
      neg_zero_low,
      MakeSop1(0u, 125u, 255u),
      neg_zero_high,
      MakeVopc(127u, 124u, 98u),
      MakeVop2(1u, 60u, 314u, 59u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f64_class_cndmask_program;
  if (!Expect(decoder.DecodeProgram(f64_class_cndmask_words,
                                    &f64_class_cndmask_program,
                                    &error_message),
              "expected F64 class cndmask program decode success") ||
      !Expect(f64_class_cndmask_program.size() == 5u,
              "expected five decoded F64 class cndmask instructions") ||
      !Expect(f64_class_cndmask_program[2].opcode == "V_CMP_CLASS_F64",
              "expected decoded V_CMP_CLASS_F64") ||
      !Expect(f64_class_cndmask_program[3].opcode == "V_CNDMASK_B32",
              "expected decoded V_CNDMASK_B32 after F64 class")) {
    return 1;
  }

  auto initialize_f64_class_cndmask_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xfu;
    state->vgprs[58][0] = 10u;
    state->vgprs[58][1] = 10u;
    state->vgprs[58][2] = 10u;
    state->vgprs[58][3] = 10u;
    state->vgprs[59][0] = 20u;
    state->vgprs[59][1] = 20u;
    state->vgprs[59][2] = 20u;
    state->vgprs[59][3] = 20u;
    state->vgprs[98][0] = 0x20u;
    state->vgprs[98][1] = 0x40u;
    state->vgprs[98][2] = 0x60u;
    state->vgprs[98][3] = 0x001u;
  };

  WaveExecutionState decoded_f64_class_cndmask_state;
  initialize_f64_class_cndmask_state(&decoded_f64_class_cndmask_state);
  if (!Expect(interpreter.ExecuteProgram(f64_class_cndmask_program,
                                         &decoded_f64_class_cndmask_state,
                                         &error_message),
              "expected decoded F64 class cndmask execution success") ||
      !Expect(ExpectF64ClassCndmaskState(decoded_f64_class_cndmask_state),
              "expected decoded F64 class cndmask state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f64_class_cndmask_program;
  if (!Expect(interpreter.CompileProgram(f64_class_cndmask_program,
                                         &compiled_f64_class_cndmask_program,
                                         &error_message),
              "expected compiled F64 class cndmask program success") ||
      !Expect(compiled_f64_class_cndmask_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpClassF64,
              "expected compiled V_CMP_CLASS_F64 opcode") ||
      !Expect(compiled_f64_class_cndmask_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCndmaskB32,
              "expected compiled V_CNDMASK_B32 after F64 class")) {
    return 1;
  }

  WaveExecutionState compiled_f64_class_cndmask_state;
  initialize_f64_class_cndmask_state(&compiled_f64_class_cndmask_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f64_class_cndmask_program,
                                         &compiled_f64_class_cndmask_state,
                                         &error_message),
              "expected compiled F64 class cndmask execution success") ||
      !Expect(ExpectF64ClassCndmaskState(compiled_f64_class_cndmask_state),
              "expected compiled F64 class cndmask state")) {
    return 1;
  }

  std::uint32_t qnan64_low = 0;
  std::uint32_t qnan64_high = 0;
  SplitU64(kQuietNaNF64Bits, &qnan64_low, &qnan64_high);
  const std::array<std::uint32_t, 10> f64_cmpx_class_branch_words{
      MakeSop1(0u, 114u, 255u),
      qnan64_low,
      MakeSop1(0u, 115u, 255u),
      qnan64_high,
      MakeVopc(255u, 114u, 99u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 119u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 119u, 222u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f64_cmpx_class_branch_program;
  if (!Expect(decoder.DecodeProgram(f64_cmpx_class_branch_words,
                                    &f64_cmpx_class_branch_program,
                                    &error_message),
              "expected F64 CMPX class branch program decode success") ||
      !Expect(f64_cmpx_class_branch_program.size() == 8u,
              "expected eight decoded F64 CMPX class branch instructions") ||
      !Expect(f64_cmpx_class_branch_program[2].opcode == "V_CMPX_CLASS_F64",
              "expected decoded V_CMPX_CLASS_F64") ||
      !Expect(f64_cmpx_class_branch_program[3].opcode == "S_CBRANCH_EXECNZ",
              "expected decoded S_CBRANCH_EXECNZ after F64 CMPX class")) {
    return 1;
  }

  auto initialize_f64_cmpx_class_branch_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xfu;
    state->vgprs[99][0] = 0x001u;
    state->vgprs[99][1] = 0x002u;
    state->vgprs[99][2] = 0x003u;
    state->vgprs[99][3] = 0x200u;
  };

  WaveExecutionState decoded_f64_cmpx_class_branch_state;
  initialize_f64_cmpx_class_branch_state(&decoded_f64_cmpx_class_branch_state);
  if (!Expect(interpreter.ExecuteProgram(f64_cmpx_class_branch_program,
                                         &decoded_f64_cmpx_class_branch_state,
                                         &error_message),
              "expected decoded F64 CMPX class branch execution success") ||
      !Expect(ExpectF64CmpxClassBranchState(
                  decoded_f64_cmpx_class_branch_state),
              "expected decoded F64 CMPX class branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f64_cmpx_class_branch_program;
  if (!Expect(interpreter.CompileProgram(f64_cmpx_class_branch_program,
                                         &compiled_f64_cmpx_class_branch_program,
                                         &error_message),
              "expected compiled F64 CMPX class branch program success") ||
      !Expect(compiled_f64_cmpx_class_branch_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxClassF64,
              "expected compiled V_CMPX_CLASS_F64 opcode") ||
      !Expect(compiled_f64_cmpx_class_branch_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchExecnz,
              "expected compiled S_CBRANCH_EXECNZ after F64 CMPX class")) {
    return 1;
  }

  WaveExecutionState compiled_f64_cmpx_class_branch_state;
  initialize_f64_cmpx_class_branch_state(&compiled_f64_cmpx_class_branch_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f64_cmpx_class_branch_program,
                                         &compiled_f64_cmpx_class_branch_state,
                                         &error_message),
              "expected compiled F64 CMPX class branch execution success") ||
      !Expect(ExpectF64CmpxClassBranchState(
                  compiled_f64_cmpx_class_branch_state),
              "expected compiled F64 CMPX class branch state")) {
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
